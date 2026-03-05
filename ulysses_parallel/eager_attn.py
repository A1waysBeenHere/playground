import os
import random
import argparse
import numpy as np
import torch, torch_npu
import torch.nn as nn
import torch.distributed as dist

def eager_attn_forward(q, k, v):
    attn_weights = torch.matmul(q, k.transpose(2, 3))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = nn.functional.dropout(attn_weights)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output