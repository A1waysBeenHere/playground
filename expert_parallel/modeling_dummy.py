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
from typing import Any, List, Optional, Tuple


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
    """MoE层（整合门控和专家网络，支持 Expert Parallel）"""
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

        # 1. 初始化门控
        self.gating = TopKGating(input_dim, num_experts, top_k)
        
        # 2. 初始化专家网络（将各专家 concat 到同一个 Tensor 中，方便 EP 切分和批量计算）
        # 形状: [num_experts, input_dim, expert_hidden_dim]
        self.w1 = nn.Parameter(torch.randn(num_experts, input_dim, expert_hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, expert_hidden_dim, output_dim))
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：利用 concat 后的参数进行批量计算，避免 Python for 循环
        """
        batch_size, seq_len, _ = x.shape
        # 1. 计算门控权重和专家索引
        gate_scores, expert_indices, aux_loss = self.gating(x) 
        
        # 2. 展平维度
        x_flat = x.reshape(-1, self.input_dim) # [tokens, input_dim]
        gate_scores_flat = gate_scores.reshape(-1, self.top_k) 
        expert_indices_flat = expert_indices.reshape(-1, self.top_k) 
        
        # 3. 批量计算
        outputs = []
        for k in range(self.top_k):
            indices = expert_indices_flat[:, k] 
            w1_selected = self.w1[indices]  # [tokens, input_dim, hidden_dim]
            w2_selected = self.w2[indices]  # [tokens, hidden_dim, output_dim]
            
            # 第一层
            hidden = torch.bmm(x_flat.unsqueeze(1), w1_selected)
            hidden = self.activation(hidden)
            hidden = self.dropout(hidden)
            
            # 第二层
            out = torch.bmm(hidden, w2_selected)
            outputs.append(out.squeeze(1)) # [tokens, output_dim]
            
        # 4. 加权融合
        final_output_flat = torch.zeros_like(x_flat)[:, :self.output_dim]
        for k in range(self.top_k):
            final_output_flat += gate_scores_flat[:, k:k+1] * outputs[k]
        
        output = final_output_flat.reshape(batch_size, seq_len, self.output_dim)
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

print_sharding_info(model.moe_layer)
fully_shard(model.moe_layer, mesh=mesh["ep"], reshard_after_forward=False)
# distribute_tensor(model.moe_layer.w1, mesh["ep"], [Shard(0)])
# distribute_tensor(model.moe_layer.w2, mesh["ep"], [Shard(0)])
print_sharding_info(model.moe_layer)

for name, module in reversed(list(model.named_modules())):
    if isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict)) or "moe_layer" in name:
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