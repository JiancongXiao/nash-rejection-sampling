"""Minimal torchrun/NCCL all-reduce validation."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.ones(1, device="cuda")
    dist.all_reduce(value)
    if value.item() != dist.get_world_size():
        raise RuntimeError(f"bad all-reduce result: {value.item()}")
    print(os.environ["RANK"], value.item(), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
