# Here, we only care about the ulysses parallel case,
# so we assume world_size = ulysses_parallel_size

# torchrun --nproc_per_node=2 sp_test.py --sp_enabled True

import os
import random
import argparse
import numpy as np
import torch, torch_npu
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed._composable.fsdp import fully_shard

from modeling_dummy import DummyModel, Attention
from dataset_dummy import DummyDataset
from ulysses_utils import ulysses_parellel_forward

def train(args, local_rank, world_size, device_type="npu"):
    device = torch.device(f"{device_type}:{local_rank}")
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    torch.npu.set_device(device)
    if args.sp_enabled:
        Attention.forward = ulysses_parellel_forward
        print(f"====>[WARNING]: Attention.forward being replaced with ulysses_parellel_forward.", flush=True)
    model = DummyModel(hidden_size=hidden_size, num_heads=num_heads).to(device)
    
    
    dist.barrier()

    for name, module in reversed(list(model.named_modules())):
        fully_shard(module)

    dataset = DummyDataset(hidden_size=hidden_size)
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if args.sp_enabled:
                inputs = torch.split(x, world_size, dim=1) 
                output = model(inputs[local_rank])
            else:
                output = model(x)
            
            # actually the loss of naive and SP one are different, 
            # sould not use CrossEntropyLoss in SP case
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()
        
        local_loss = loss.detach()
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)

        if local_rank == 0:
            print(f"[Rank {local_rank}] Epoch {epoch + 1}, loss: {loss.item():.4f}")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed SPAttention training")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of attention")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--sp_enabled", type=bool, default=False, help="Ulysses parallel trigger")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    train(args, local_rank, world_size, device_type="npu")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# 
# [Rank 0] Epoch 1, loss: 1.4148
# [Rank 0] Epoch 2, loss: 1.3984
# [Rank 0] Epoch 3, loss: 1.3920
# [Rank 0] Epoch 4, loss: 1.3906
# [Rank 0] Epoch 5, loss: 1.3873
# [Rank 0] Epoch 6, loss: 1.3618
# [Rank 0] Epoch 7, loss: 1.3281
# [Rank 0] Epoch 8, loss: 1.2578
# [Rank 0] Epoch 9, loss: 1.3602
# [Rank 0] Epoch 10, loss: 1.3301