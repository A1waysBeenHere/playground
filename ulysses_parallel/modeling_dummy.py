import os
import random
import numpy as np
import torch, torch_npu
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed._composable.fsdp import fully_shard
from eager_attn import eager_attn_forward


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)

        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)


    def forward(self, hidden_states):
        """
        hidden_states: [B, S, hidden_size]
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1 , self.head_dim)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(hidden_shape).transpose(1, 2)
        k = k.view(hidden_shape).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)

        # self_attn compute
        attn_output = eager_attn_forward(q, k, v)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        out = self.out_proj(attn_output)

        return out

class DummyModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads):
        super().__init__()
        # 1. 嵌入层：通过词表大小和隐藏维度初始化
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # 2. 注意力层
        self.attn = Attention(hidden_size, num_heads)
        # 3. 输出投影层（LM Head）
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        """
        x: 输入 Token IDs, 形状 [B, S]
        """
        # input_ids -> hidden_states [B, S, H]
        h = self.embedding(x)
        # 经过注意力计算 [B, S, H]
        h = self.attn(h)
        # 映射回词表空间 [B, S, V]
        logits = self.lm_head(h)
        return logits