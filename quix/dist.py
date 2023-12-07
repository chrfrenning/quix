import copy
import datetime
import errno
import hashlib
import os
import time
from collections import defaultdict, deque, OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from .cfg import RunConfig, TMod, TDat, TOpt, TLog, TAug, TSch


def setup_for_distributed(is_master:bool) -> None:
    """Disables printing when not in master process

    From https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized() -> bool:
    """Checks availablility and init for distributed training .

    From https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    
    Returns
    -------
    bool
        Flag for checking distributed training.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    '''Returns world size.

    From https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    Returns
    -------
    int
        World size of distributed process.
    '''
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    '''Returns rank of process.

    From https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    Returns
    -------
    int
        Rank of distributed process.
    '''
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    '''Returns True if process is main process.

    From https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    Returns
    -------
    bool
        Whether current process is main process.
    '''
    return get_rank() == 0


def save_on_master(*args, **kwargs) -> None:
    '''Helper function to save given master process.

    From https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    '''
    if is_main_process():
        torch.save(*args, **kwargs)

def _init_distributed_mode(args):
    '''Function to initialize distributed mode.

    Adapted from https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    '''
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# def init_distributed_mode(cfg:RunConfig[TMod,TDat,TOpt,TLog,TAug,TSch]) -> None:
#     '''Function to initialize distributed mode.
#     '''
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#         gpu = int(os.environ["LOCAL_RANK"])
#     elif "SLURM_PROCID" in os.environ:
#         rank = int(os.environ["SLURM_PROCID"])
#         gpu = rank % torch.cuda.device_count()
#     elif hasattr(args, "rank"):
#         pass
#     else:
#         print("Not using distributed mode")
#         distributed = False
#         return

#     distributed = True

#     torch.cuda.set_device(gpu)
#     args.dist_backend = "nccl"
#     print(f"| distributed init (rank {rank}): {cfg.ddp_url}", flush=True)
#     torch.distributed.init_process_group(
#         backend=cfg.ddp_backend, init_method=cfg.ddp_url, world_size=world_size, rank=rank
#     )
#     torch.distributed.barrier()
#     setup_for_distributed(rank == 0)