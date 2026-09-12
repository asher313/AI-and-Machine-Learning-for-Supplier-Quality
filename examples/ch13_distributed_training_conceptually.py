# Chapter 13 — 13.4 Distributed Training, Conceptually
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

sampler = DistributedSampler(dataset)
loader = DataLoader(
    dataset, batch_size=32, sampler=sampler
)
# launch with:
#   torchrun --nproc_per_node=4 train.py
