import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo.eval_frame import OptimizedModule
import time
import torch
import numpy as np
import os
import socket
        

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
    

def get_raw(net): # slightly awkward way to consistently access underyling Transformer object
    if isinstance(net, OptimizedModule):
        net = net._modules["_orig_mod"]
    if isinstance(net, DDP):
        net = net.module
    return net