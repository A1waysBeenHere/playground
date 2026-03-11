import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Expert(nn.Module):
    """单个专家网络（简单的两层全连接）"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，shape [batch_size, seq_len, input_dim]
        Returns:
            输出张量，shape [batch_size, seq_len, output_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
    """MoE层（整合门控和专家网络）"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_hidden_dim: int,
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
        # 2. 初始化专家网络（用ModuleList管理多个专家）
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：路由输入到Top-K专家，融合专家输出
        Args:
            x: 输入张量，shape [batch_size, seq_len, input_dim]
        Returns:
            output: 融合后的输出，shape [batch_size, seq_len, output_dim]
            aux_loss: 负载均衡辅助损失
        """
        batch_size, seq_len, _ = x.shape
        # 1. 计算门控权重和专家索引
        gate_scores, expert_indices, aux_loss = self.gating(x)  # [bs, seq, k], [bs, seq, k]
        
        # 2. 展平维度，方便批量处理 [batch_size*seq_len, input_dim]
        x_flat = x.reshape(-1, self.input_dim)
        gate_scores_flat = gate_scores.reshape(-1, self.top_k)  # [bs*seq, k]
        expert_indices_flat = expert_indices.reshape(-1, self.top_k)  # [bs*seq, k]
        
        # 3. 初始化输出张量
        output_flat = torch.zeros_like(x_flat)[:, :self.output_dim]
        
        # 4. 遍历每个专家，处理分配给它的样本
        for expert_id, expert in enumerate(self.experts):
            # 找到所有选择了当前专家的样本索引
            mask = (expert_indices_flat == expert_id)  # [bs*seq, k]
            if not mask.any():
                continue  # 无样本分配给该专家，跳过
            
            # 提取这些样本的输入和对应的门控权重
            selected_indices = mask.nonzero()[:, 0]  # 样本索引
            selected_x = x_flat[selected_indices]
            selected_scores = gate_scores_flat[mask]
            
            # 专家处理样本
            expert_output = expert(selected_x)
            # 加权融合到输出中
            output_flat[selected_indices] += selected_scores.unsqueeze(1) * expert_output
        
        # 5. 恢复原始维度
        output = output_flat.reshape(batch_size, seq_len, self.output_dim)
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
        num_classes: int = 10  # 分类任务的类别数
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
            dropout=dropout
        )
        # 3. 输出层（分类任务）
        self.fc_out = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            input_ids: 输入token索引，shape [batch_size, seq_len]
        Returns:
            logits: 分类输出，shape [batch_size, num_classes]
            total_aux_loss: 负载均衡辅助损失（训练时需要加到主损失中）
        """
        # 1. 嵌入层
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = self.dropout(x)
        
        # 2. MoE层
        moe_output, aux_loss = self.moe_layer(x)  # [batch_size, seq_len, embed_dim]
        
        # 3. 池化（取序列最后一个token或均值，这里用均值）
        pooled = moe_output.mean(dim=1)  # [batch_size, embed_dim]
        
        # 4. 输出层
        logits = self.fc_out(pooled)  # [batch_size, num_classes]
        
        return logits, aux_loss

