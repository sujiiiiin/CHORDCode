import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


def init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def to_segment_pairs(values):
    return [(values[2 * i], values[2 * i + 1]) for i in range(len(values) // 2)]


@dataclass
class DistributedInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self):
        return self.rank == 0


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        local_rank = args.device
        torch.cuda.set_device(local_rank)

    if args.seed >= 0:
        seed = args.seed + rank
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return DistributedInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device(f"cuda:{local_rank}"),
        distributed=distributed,
    )


def cleanup_distributed(ddp):
    if ddp.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ddp):
    if ddp.distributed:
        dist.barrier()


def broadcast_object(obj, ddp, src=0):
    if not ddp.distributed:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def mean_metric(value, ddp):
    if torch.is_tensor(value):
        value = value.detach().float().item()
    tensor = torch.tensor([float(value)], device=ddp.device)
    if ddp.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(ddp.world_size)
    return tensor.item()
