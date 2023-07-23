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
    torch.cuda.synchronize()
    return time.time()


def get_device_type():
    # return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def rprint(*args, **kwargs):
    print("rank", get_rank(), *args, **kwargs)

def oprint(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def get_random_unused_port():
    s=socket.socket()
    s.bind(("", 0))
    port_name = str(s.getsockname()[1])
    s.close()
    return port_name


def get_rank():  # consider making this a decorator (with a parameter that specifies a default return value)
    try:
        return dist.get_rank()
    except RuntimeError:
        return 0
    

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))
    

def get_world_size():
    try:
        return dist.get_world_size()
    except RuntimeError:
        return 1
    

def barrier():
    try:
        dist.barrier()
    except RuntimeError:
        pass
    
 
 # slightly awkward way to consistently access underyling Transformer object
def get_raw(net: Union[OptimizedModule, DDP, "Transformer"]) -> "Transformer":
    if isinstance(net, OptimizedModule):
        return get_raw(net._modules["_orig_mod"]) # type: ignore
    if isinstance(net, DDP):
        return get_raw(net.module)  # do recursion so its independent of the order of DDP and .compile()
    return net   # while loop is more annoying due to circular dependency and needing Transformer defined