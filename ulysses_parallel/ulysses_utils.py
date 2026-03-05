import os
import random
import argparse
import numpy as np
import torch, torch_npu
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed._composable.fsdp import fully_shard


from eager_attn import eager_attn_forward


# def slice_tensor(hidden_states):
#     rank = int(os.environ["RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])
#     local_rank = int(os.environ["LOCAL_RANK"])
#     B, S, H = hidden_states.shape
#     chunk_size = H // world_size
#     start = rank * chunk_size
#     end = (rank + 1) * chunk_size if rank != world_size - 1 else H
#     hidden_states = hidden_states[..., start:end]

class AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, world_size):
        ctx.world_size = world_size

        input_list = [
            t.contiguous()
            for t in torch.tensor_split(input, world_size, dim=-1)
        ]

        output_list = [
            torch.empty_like(input_list[0])
            for _ in range(world_size)
        ]

        dist.all_to_all(output_list, input_list)

        return torch.cat(output_list, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = ctx.world_size

        input_list = [
            t.contiguous()
            for t in torch.tensor_split(grad_output, world_size, dim=-1)
        ]

        output_list = [
            torch.empty_like(input_list[0])
            for _ in range(world_size)
        ]

        dist.all_to_all(output_list, input_list)

        return torch.cat(output_list, dim=-1), None

def ulysses_parellel_forward(self, hidden_states):
    # hidden_states = slice_tensor(hidden_states)

    world_size = int(os.environ["WORLD_SIZE"])
    # input_list = [t.contiguous() for t in torch.tensor_split(hidden_states, world_size, -1)]
    
    # output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    # dist.all_to_all(output_list, input_list)
    # full_hidden_states = torch.cat(output_list, dim=-1)
    full_hidden_states = AllToAll.apply(hidden_states, world_size)
    # print(f"====>hidden_states.shape: {hidden_states.shape}\nfull_hidden_states.shape: {full_hidden_states.shape}")
    input_shape = full_hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1 , self.head_dim)
    q = self.q_proj(full_hidden_states)
    k = self.k_proj(full_hidden_states)
    v = self.v_proj(full_hidden_states)

    q = q.view(hidden_shape).transpose(1, 2)
    k = k.view(hidden_shape).transpose(1, 2)
    v = v.view(hidden_shape).transpose(1, 2)

    # self_attn compute
    attn_output = eager_attn_forward(q, k, v)

    # input_list = [t.contiguous() for t in torch.tensor_split(attn_output, world_size, -1)]
    # output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    # dist.all_to_all(output_list, input_list)
    # attn_output = torch.cat(output_list, dim=-1)
    res = AllToAll.apply(attn_output, world_size)
    res = res.reshape(*input_shape, -1).contiguous()

    out = self.out_proj(res)

    return out