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
        x: [B, S, hidden_size]
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
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads)

    def forward(self, x):
        return self.attn(x)