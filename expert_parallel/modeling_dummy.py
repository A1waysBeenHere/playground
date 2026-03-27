import torch.nn.functional as F
import os
import random
import argparse
import numpy as np
import torch, torch_npu
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor
from typing import Any, List, Optional, Tuple, Literal


# --- VeOmni EP Helper Functions ---

def all_to_all(group, input, output_split_sizes=None, input_split_sizes=None):
    """Encapsulates All-to-All communication."""
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input

    input = input.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(input)
    else:
        output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
    dist.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def permute(tokens: torch.Tensor, routing_map: torch.Tensor):
    """Permutes tokens according to the routing map."""
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()

    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def unpermute(
    tokens: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
):
    """Unpermutes tokens and applies weights."""
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    unpermuted_tokens.scatter_add_(0, permutation_mapping.unsqueeze(1).expand(-1, hidden_states_shape[-1]), tokens)
    return unpermuted_tokens


def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    """Sorts input tensor chunks back by original indices."""
    input_chunks = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input_chunks[i] for i in sorted_idxs], dim=0)
    return output


def preprocess(expert_mask: torch.Tensor, num_experts: int, ep_group: dist.ProcessGroup):
    """Calculates split sizes for All-to-All communication."""
    ep_size = dist.get_world_size(group=ep_group)
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    
    # [num_experts]
    num_local_tokens_per_expert = expert_mask.to(torch.int).sum(dim=1)
    
    # How many tokens this rank sends to each other rank
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    
    # Synchronize tokens per expert across all EP ranks
    num_global_tokens_per_expert = torch.zeros(
        ep_size, num_experts, dtype=num_local_tokens_per_expert.dtype, device=num_local_tokens_per_expert.device
    )
    dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)
    
    # How many tokens this rank receives from each other rank
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    
    # Cumulative sum tokens for local experts
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
    
    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def token_pre_all2all(hidden_states, expert_mask, num_experts, input_splits, output_splits, num_global_tokens_per_local_expert, ep_group):
    """Performs permutation and All-to-All dispatch."""
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_shape = hidden_states.shape
    
    permuted_hidden, input_mapping = permute(hidden_states, expert_mask)
    global_permuted_hidden = all_to_all(ep_group, permuted_hidden, output_splits, input_splits)
    
    # Re-sort tokens by local expert index
    ep_size = dist.get_world_size(group=ep_group)
    num_local_experts = num_experts // ep_size
    # Interleave indices to group by LOCAL expert
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    global_permuted_hidden = sort_chunks_by_idxs(
        global_permuted_hidden,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )
    
    return global_permuted_hidden, input_mapping, org_shape


def tokens_post_all2all(expert_outputs, routing_weights, selected_experts, num_experts, input_splits, output_splits, num_global_tokens_per_local_expert, expert_mask, input_mapping, org_shape, ep_group):
    """Performs All-to-All receive and unpermutation."""
    ep_size = dist.get_world_size(group=ep_group)
    num_local_experts = num_experts // ep_size
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )
    
    unpermuted_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)
    
    # Build routing weights matrix for unpermute
    weights_idx = torch.zeros((org_shape[0], num_experts), dtype=routing_weights.dtype, device=routing_weights.device)
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    
    result = unpermute(unpermuted_outputs, weights_idx, org_shape, input_mapping, expert_mask)
    return result




class LLMDummyDataset(torch.utils.data.Dataset):
    """训练 LLM 用的 dummy 数据集"""
    def __init__(self, vocab_size: int, seq_len: int = 32, size: int = 100):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # LLM 训练通常使用 input_ids 右移一位作为 labels
        # 此处简化直接生成 dummy labels
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, labels


def print_sharding_info(model: nn.Module):
    """打印模型参数的切分信息"""
    if dist.get_rank() != 0:
        return
        
    print("\n" + "="*50)
    print("模型参数切分状态 (Sharding Status):")
    print("="*50)
    
    for name, param in model.named_parameters():
        param_type = type(param).__name__
        
        # 尝试获取分片信息
        sharding_desc = "Unknown / Not Sharded"
        # 使用 isinstance 判断基类，减少静态分析报错
        if hasattr(param, "placements"):
             # param: Any (DTensor)
             sharding_desc = f"DTensor | Placements: {getattr(param, 'placements', 'N/A')}"
        elif "FlatParameter" in param_type:
             sharding_desc = "FSDP1 FlatParameter"
             
        print(f"Layer: {name:<40} | Type: {param_type:<15}")
        print(f"  -> {sharding_desc}")
        # 通过 getattr 规避属性不存在的报错
        print(f"  -> Total Shape: {list(getattr(param, 'shape', []))}")
        if hasattr(param, "to_local"):
            local_tensor = getattr(param, "to_local")()
            print(f"  -> Local Shape: {list(local_tensor.shape)}")
        print("-" * 50)
    print("="*50 + "\n")


# 移除旧的 Expert 类，改为在 MoELayer 中直接使用 concat 后的参数


class TopKGating(nn.Module):
    """Top-K路由门控（核心MoE路由逻辑）"""
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 门控网络：输出每个专家的权重
        self.gate = nn.Linear(input_dim, num_experts)
        # 防止某类样本只路由到少数专家的损失系数
        self.aux_loss_coeff = 0.01

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算专家权重、选择Top-K专家、计算负载均衡损失
        Args:
            x: 输入张量，shape [batch_size, seq_len, input_dim]
        Returns:
            gate_scores: Top-K专家的归一化权重，shape [batch_size, seq_len, top_k]
            expert_indices: Top-K专家的索引，shape [batch_size, seq_len, top_k]
            aux_loss: 负载均衡辅助损失（用于训练）
        """
        # 1. 计算所有专家的原始权重 [batch_size, seq_len, num_experts]
        logits = self.gate(x)
        # 2. 选择Top-K专家的索引和权重
        top_k_logits, expert_indices = torch.topk(logits, k=self.top_k, dim=-1)
        # 3. 对Top-K权重做softmax归一化
        gate_scores = F.softmax(top_k_logits, dim=-1)
        
        # 4. 计算负载均衡损失（防止专家负载不均）
        # 计算每个专家被选中的概率 [batch_size*seq_len, num_experts]
        gate_probs = F.softmax(logits, dim=-1).reshape(-1, self.num_experts)
        # 计算每个专家的负载（被选中的次数）
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).sum(dim=-2)  # [bs, seq, num_experts]
        expert_load = expert_mask.sum(dim=[0, 1]) / expert_mask.numel()  # [num_experts]
        # 理想负载：1/num_experts，计算KL散度作为损失
        ideal_load = torch.ones_like(expert_load) / self.num_experts
        aux_loss = F.kl_div(gate_probs.log(), ideal_load.unsqueeze(0), reduction='mean')
        
        return gate_scores, expert_indices, self.aux_loss_coeff * aux_loss


class MoELayer(nn.Module):
    """MoE层（整合门控和专家网络，完全遵循 VeOmni EP 实现逻辑）"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_hidden_dim: int,
        ep_mesh: Optional[Any] = None, # (dp, ep) mesh
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_mesh = ep_mesh
        if ep_mesh is not None:
            self.ep_group = ep_mesh.get_group()
            self.ep_size = dist.get_world_size(group=self.ep_group)
        else:
            self.ep_group = None
            self.ep_size = 1
        self.num_local_experts = num_experts // self.ep_size

        # 1. 初始化门控
        self.gating = TopKGating(input_dim, num_experts, top_k)
        
        # 2. 初始化专家网络
        # 在 EP 模式下，我们要确保每张卡只负责自己的 local experts
        # 为了演示，我们先全量初始化，然后在 forward 中根据 EP 分离
        self.w1 = nn.Parameter(torch.randn(num_experts, input_dim, expert_hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, expert_hidden_dim, output_dim))
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        # 1. 计算门控权重和专家索引
        gate_scores, expert_indices, aux_loss = self.gating(x) 
        
        if self.ep_group is None or self.ep_size == 1:
            # 非 EP 模式：原有逻辑
            x_flat = x.reshape(-1, self.input_dim)
            gate_scores_flat = gate_scores.reshape(-1, self.top_k) 
            expert_indices_flat = expert_indices.reshape(-1, self.top_k) 
            outputs = []
            for k in range(self.top_k):
                indices = expert_indices_flat[:, k] 
                w1_selected = self.w1[indices]
                w2_selected = self.w2[indices]
                hidden = torch.bmm(x_flat.unsqueeze(1), w1_selected)
                hidden = self.activation(hidden)
                hidden = self.dropout(hidden)
                out = torch.bmm(hidden, w2_selected)
                outputs.append(out.squeeze(1))
            final_output_flat = torch.zeros_like(x_flat)[:, :self.output_dim]
            for k in range(self.top_k):
                final_output_flat += gate_scores_flat[:, k:k+1] * outputs[k]
            output = final_output_flat.reshape(batch_size, seq_len, self.output_dim)
            return output, aux_loss

        # --- EP 模式：遵循 VeOmni All-to-All 逻辑 ---
        
        # a. 准备 expert_mask [E, total_tokens]
        total_tokens = batch_size * seq_len
        # expert_indices shape: [bs, seq, top_k] -> [total_tokens, top_k]
        expert_indices_flat = expert_indices.reshape(total_tokens, self.top_k)
        gate_scores_flat = gate_scores.reshape(total_tokens, self.top_k)
        
        expert_mask = F.one_hot(expert_indices_flat, num_classes=self.num_experts).sum(dim=1).T # [E, total_tokens]
        
        # b. Preprocess (计算 All-to-All split sizes)
        input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = preprocess(
            expert_mask, self.num_experts, self.ep_group
        )
        
        # c. Token Pre-All2All (Permute + All-to-All)
        permuted_tokens, input_mapping, org_shape = token_pre_all2all(
            x, expert_mask, self.num_experts, input_splits, output_splits, 
            num_global_tokens_per_local_expert, self.ep_group
        )
        
        # d. Local Expert Execution
        # 获取本地专家的参数
        rank = dist.get_rank(self.ep_group)
        start_idx = rank * self.num_local_experts
        end_idx = (rank + 1) * self.num_local_experts
        local_w1 = self.w1[start_idx:end_idx] # [num_local_experts, I, H]
        local_w2 = self.w2[start_idx:end_idx] # [num_local_experts, H, O]
        
        # 按照本地专家分配的 token 数进行 split
        local_tokens_per_expert = num_global_sum_tokens_per_local_expert.tolist()
        expert_inputs = torch.split(permuted_tokens, local_tokens_per_expert, dim=0)
        
        expert_outputs = []
        for i, token_chunk in enumerate(expert_inputs):
            if token_chunk.shape[0] == 0:
                expert_outputs.append(torch.empty(0, self.output_dim, device=x.device, dtype=x.dtype))
                continue
            # 单个专家计算 (模拟 Group Gemm)
            # local_w1[i] shape: [input_dim, expert_hidden_dim]
            h = torch.matmul(token_chunk, local_w1[i])
            h = self.activation(h)
            h = self.dropout(h)
            out = torch.matmul(h, local_w2[i])
            expert_outputs.append(out)
        
        combined_expert_outputs = torch.cat(expert_outputs, dim=0)
        
        # e. Token Post-All2All (All-to-All + Unpermute)
        final_hidden_states = tokens_post_all2all(
            combined_expert_outputs, gate_scores_flat, expert_indices_flat, 
            self.num_experts, input_splits, output_splits, 
            num_global_tokens_per_local_expert, expert_mask, 
            input_mapping, org_shape, self.ep_group
        )
        
        output = final_hidden_states.reshape(batch_size, seq_len, self.output_dim)
        return output, aux_loss


class MoEModel(nn.Module):
    """完整的MoE模型（包含嵌入层+MoE层+输出层）"""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_experts: int = 8,
        expert_hidden_dim: int = 1024,
        top_k: int = 2,
        dropout: float = 0.1,
        mesh: Optional[Any] = None  # (dp, ep) mesh
    ):
        super().__init__()
        # 1. 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 2. MoE核心层
        self.moe_layer = MoELayer(
            input_dim=embed_dim,
            output_dim=embed_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            top_k=top_k,
            dropout=dropout,
            ep_mesh=mesh["ep"] if mesh is not None else None
        )
        # 3. 输出层（LLM 任务：投影回词表大小）
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = self.dropout(x)
        
        moe_output, aux_loss = self.moe_layer(x)  # [batch_size, seq_len, embed_dim]
        
        logits = self.lm_head(moe_output)  # [batch_size, seq_len, vocab_size]
        
        return logits, aux_loss




seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  

# --- 初始化与并行配置 ---
vocab_size = 10000
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

# FSDP2 + EP: 定义 2D Device Mesh
ep_size = 2 if world_size >= 2 else 1
dp_size = world_size // ep_size
mesh = init_device_mesh("npu", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))

model = MoEModel(
    vocab_size=vocab_size,
    embed_dim=512,
    num_experts=8,
    expert_hidden_dim=1024,
    top_k=2,
    mesh=mesh
).to(f"npu:{local_rank}")

# 根据 VeOmni 逻辑，我们需要对专家参数进行 EP 切分 (Shard 0)
# 在 FSDP2 中，我们可以直接通过 fully_shard 对 moe_layer 内部的 w1, w2 进行特定维度的切分
# 但为了更贴合 VeOmni 的模式，我们这里演示对手动管理的 w1, w2 进行 EP 维度的 fully_shard

# 修改：不再直接对 moe_layer 整体做 ep 维度的 fully_shard，因为那会将 gating 也 shard 掉
# 我们只 shard w1 和 w2
fully_shard(model.moe_layer, mesh=mesh["ep"], reshard_after_forward=False)

# 对非 MoE 部分做普通的 DP (FSDP)
for name, module in reversed(list(model.named_modules())):
    if "moe_layer" in name or isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)):
        continue
    fully_shard(module, mesh=mesh["dp"])



dataset = LLMDummyDataset(vocab_size=vocab_size, seq_len=32, size=100)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=4, drop_last=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()

if rank == 0:
    print(f"启动训练... Vocab: {vocab_size}, World Size: {world_size}")

for epoch in range(5):
    sampler.set_epoch(epoch)
    total_loss = 0.0
    for step, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(f"npu:{local_rank}")
        labels = labels.to(f"npu:{local_rank}")
        
        optimizer.zero_grad()
        logits, aux_loss = model(input_ids)
        
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        total_step_loss = loss + aux_loss
        
        total_step_loss.backward()
        optimizer.step()
        total_loss += total_step_loss.item()
        
    avg_loss = total_loss / len(dataloader)
    if rank == 0:
        print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")

if rank == 0:
    print("训练完成")