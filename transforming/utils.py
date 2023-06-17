import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import torch
import numpy as np
import os
import socket

def get_time():
    torch.cuda.synchronize()
    return time.time()


def get_device_type():
    return "cuda" if torch.cuda.is_available() else "cpu"


def rprint(*args, **kwargs):
    print("rank", get_rank(), *args, **kwargs)

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
    return net.module if isinstance(net, DDP) else net


def sample_random_start_word(dset, window_size):   # for generating the samples to show qualitative progress over time
    while True:  # look over all(-ish) sentence starting words in the dataset, pick one at random
        idx = np.random.randint(0, len(dset))
        random_subset = dset[idx][0]  # [0] selects x over y
        try:  # this can fail if eos token not found, or if eos only found at the very end of the sequence
            return random_subset[np.where(random_subset == dset.encoder.eos_token)[0][0] + 1].item()  # go 1 token past an eos token
        except:  # so just wrap it in a try except until it works
            pass
    

# sample generate a bunch of sentences using random start words
def sample_random_sentences(net, dsets, exp_config, num_sample=5, temperature=0.2):
    start_tokens = [sample_random_start_word(dsets["eval"], 20*exp_config.block_size) for _ in range(num_sample)]
    # print("found start tokens", start_tokens, dsets["eval"].encoder.decode(start_tokens))
    return [net.generate(dsets["eval"].encoder, tok, temperature=temperature) for tok in start_tokens]


def traverse_modules(func, mod):
    has_sub_mods = False
    for sub_mod in mod.children():
        traverse_modules(func, sub_mod)
        has_sub_mods = True
    if not has_sub_mods: # if no submodules, it is an actual operation
        func(mod)