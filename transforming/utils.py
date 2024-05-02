import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo.eval_frame import OptimizedModule
import time
import torch
import numpy as np
import os
import socket
from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    from .network import Transformer
        

def get_time():
    """Returns the current time, accounting for CUDA synchronization. Useful for profiling."""
    torch.cuda.synchronize()
    return time.time()


def get_device_type():
    """Returns the default device type that should be used. Can modify this to switch the entire codebase to CPU for debugging
    purposes."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def rprint(*args, **kwargs):
    """Does a print that includes the rank of the process."""
    print("rank", get_rank(), *args, **kwargs)

def oprint(*args, **kwargs):
    """Print, but only if you are rank 0"""
    if get_rank() == 0:
        print(*args, **kwargs)


def get_random_unused_port():
    """Returns a random unused port. Useful for setting up distributed training. Technically doesn't guarantee that the port is
    unused, but it's unlikely that it will be used."""
    s=socket.socket()
    s.bind(("", 0))
    port_name = str(s.getsockname()[1])
    s.close()
    return port_name


def get_rank():  # consider making this a decorator (with a parameter that specifies a default return value)
    """Returns the rank of the current process. If not in a distributed setting, returns 0."""
    try:
        return dist.get_rank()
    except RuntimeError:
        return 0
    

def get_local_rank():
    """Returns the local rank of the current process. If not in a distributed setting, returns 0."""
    return int(os.environ.get("LOCAL_RANK", 0))
    

def get_world_size():
    """Returns the world size. If not in a distributed setting, returns 1."""
    try:
        return dist.get_world_size()
    except RuntimeError:
        return 1
    

def barrier():
    """Synchronizes all processes. If not in a distributed setting, does nothing."""
    try:
        dist.barrier()
    except RuntimeError:
        pass
    
 
 # slightly awkward way to consistently access underyling Transformer object
def get_raw(net: Union[OptimizedModule, DDP, "Transformer"]) -> "Transformer":
    """Returns the underlying Transformer object, regardless of whether it is wrapped in a DDP or an OptimizedModule."""
    if isinstance(net, OptimizedModule):
        return get_raw(net._modules["_orig_mod"]) # type: ignore
    if isinstance(net, DDP):
        return get_raw(net.module)  # do recursion so its independent of the order of DDP and .compile()
    return net   # while loop is more annoying due to circular dependency and needing Transformer defined