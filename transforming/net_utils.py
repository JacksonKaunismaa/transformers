import torch
import numpy as np
from einops import rearrange
import os.path as osp
import glob
import json
import dataclasses
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo.eval_frame import OptimizedModule
from typing import TYPE_CHECKING, Optional, Union

from . import utils
from .utils import rprint, oprint

if TYPE_CHECKING:
    from .config_objects import ExperimentCfg
    from .network import Transformer


class Resumer():
    def __init__(self, resume_path: str, job_id: int, model_ckpt_dir="/checkpoint/jackk"):
        self.resume_path = resume_path
        self.model_ckpt_dir = model_ckpt_dir

        self.wandb_run_id = ""
        self.model_save_path = ""
        self.job_id = job_id

        # if osp.exists(resume_path):
        #     existing_paths = glob.glob(osp.join(resume_path, "*"))
        #     nums = [int(path.split(".")[-1]) for path in existing_paths] + [0]  # add [0] so there is always at least 1 element
        #     next_num = max(nums) + 1
        #     self.resume_path = f"{resume_path}.{next_num}"
        #     rprint("Warning: Creating new resume file since you specified an already existing resume path, but resume=False")
        #     rprint("New path is: ", self.resume_path)

        self.name = osp.basename(self.resume_path)
        rprint("Model name is", self.name)

    def update_model_save_path(self):
        self.model_save_path = osp.join(self.model_ckpt_dir, str(self.job_id), self.name)
        rprint("Updated save path to", self.model_save_path)

    def save(self, net: "Transformer", optim=None, **kwargs):  # kwargs should contain optimizer,scheduler, and scaler        
        # save state_dicts
        oprint(f"starting saving, {net.best_loss=}")
        save_dict = {"model": net.state_dict(),
                     "cfg": net.cfg,
                     "dset_cfg": net.dset_cfg,
                     "best_loss": net.best_loss}

        if net.cfg.ddp and net.cfg.zero and optim is not None:  # sharded optimizer => need to collate before saving
            rprint("is consolidating")
            # rprint("Has state", optim.state.keys())
            rprint("pre consolidate, has num_params", len(optim.param_groups[0]["params"]))
            optim.consolidate_state_dict()
            rprint("post consolidate, rank, has num_params", len(optim.param_groups[0]["params"]))

            if utils.get_rank() == 0:
                # rprint("is saving", optim.state_dict().keys())
                rprint("saving num_params", len(optim.state_dict()["param_groups"][0]["params"]))
                try:
                    rprint("state_dict device is", optim.state_dict()["state"][0]["exp_avg"].device)
                except KeyError as e:
                    rprint("error happened on getting device of state dict", e)
                    rprint("state_dict keys are", optim.state_dict()["state"].keys())
                save_dict["optim"] = optim.state_dict()  # it transfers shards to CPU first, so GPU mem is fine
            rprint("done consolidating")
        elif optim is not None:
            save_dict["optim"] = optim.state_dict()

        for k,obj in kwargs.items():  # add scheduler, scaler state dicts
            save_dict[k] = obj.state_dict()

        if utils.get_rank() == 0: # only do the actual IO if rank == 0
            rprint("saving being done")
            torch.save(save_dict, self.model_save_path)  # type: ignore 

        if utils.get_rank() == 0:   # update resume file, only actually create this file once the model_state dict is saved
            with open(self.resume_path, "w") as f:
                resume_info = dict(wandb_run_id=self.wandb_run_id,
                                   cfg=dataclasses.asdict(net.cfg),
                                   model_save_path=self.model_save_path)
                json.dump(resume_info, f, indent=True)
        oprint("done saving")


    def load(self, config: Optional["ExperimentCfg"]=None, net: Optional["Transformer"]=None, map_location=None, 
             update_model_save_path=False, **kwargs) -> bool:  # returns True if loading succeeds
        if not osp.exists(self.resume_path):  # if resume_path doesn't even exist, assume this is a fresh run
            rprint("No resume file found, assuming fresh model run")
            self.update_model_save_path()  # there will be nothing to load so we have to set up save path properly
            return True
        if config is not None:
            with open(self.resume_path, "r") as f:
                resume_info = json.load(f)
            self.wandb_run_id = resume_info["wandb_run_id"]
            self.model_save_path = resume_info["model_save_path"]
            config.replace_in_place(**resume_info["cfg"])

        if net is None:
            rprint("net not included in args, skipping loading state_dicts")
            return True
        
        if not osp.exists(self.model_save_path):
            rprint("No existing model found", self.model_save_path)
            return False
        
        rprint("Found path of", self.model_save_path, "loading to", map_location)
        load_dict = torch.load(self.model_save_path, map_location=map_location)  # load onto cpu so sharded models don't OOM
        rprint("saved model keys", load_dict["model"].keys())
        rprint("keys", load_dict.keys())
        rprint("is loading to", map_location)
        # rprint("has num_params", len(kwargs["optim"].param_groups[0]["params"]))

        if net.dset_cfg.vocab_size != load_dict["dset_cfg"].vocab_size:  # since we expanded vocab size halfway through,
            new_size = net.dset_cfg.vocab_size  # add some dummy entries to the saved weights (we don't use those tokens
            old_size = load_dict["dset_cfg"].vocab_size  # anyway, its just for efficiency reasons)
            load_dict['model']['embed.weight'] = add_rows(load_dict['model']['embed.weight'], new_size-old_size, map_location)
            load_dict['model']['unembed.weight'] = add_rows(load_dict['model']['unembed.weight'], new_size-old_size, map_location)
                                                       
            for k in load_dict['optim']['state']:
                if old_size in load_dict['optim']['state'][k]['exp_avg'].shape:  # unsafe to assume this, but not much else to go on
                    load_dict['optim']['state'][k]['exp_avg'] = add_rows(load_dict['optim']['state'][k]['exp_avg'], 
                                                                         new_size-old_size, map_location)
                    load_dict['optim']['state'][k]['exp_avg_sq'] = add_rows(load_dict['optim']['state'][k]['exp_avg_sq'], 
                                                                            new_size-old_size, map_location)
            load_dict['dset_cfg'].vocab_size = new_size

        net.dset_cfg = load_dict["dset_cfg"]  # correctly initialized already
        net.best_loss = load_dict["best_loss"]

        # there should probably just be a thing that just regenerates all buffers based on the new cfg, rather than loading them in
        if "shift_indices" in load_dict["model"]:  # adjust for if the batch size changed
            load_dict["model"]["shift_indices"] = get_rel_bias_indices(net.cfg.block_size,
                                                                       net.cfg.rel_bias_max_posn,
                                                                       net.cfg.rel_bias_num_buckets)
        # actually start loading state dicts
        for k,obj in kwargs.items():  # load optimizer, scheduler, scaler state dicts
            if k == "optim":
                # rprint([(i, t.shape) if hasattr(t, "shape") else (i, t, type(t)) for i, t in enumerate(obj.param_groups[0]["params"])])
                rprint(len(obj.param_groups[0]["params"]), type(obj.param_groups[0]["params"][0]), len(load_dict[k]["param_groups"][0]["params"]))
                # rprint([(i, t["exp_avg"].shape) for i, t in enumerate(load_dict[k]["state"].values())])
            obj.load_state_dict(load_dict[k])
        net.load_state_dict(load_dict["model"])
        
        if update_model_save_path:  # 1. load model based on value in resume file (potentially old /checkpoint/* dir)
            self.update_model_save_path()  # 2. subsequent saves of the model should be to /checkpoint/curr_job_id/self.name
        return True
    



def add_rows(tensor, num_rows, device):
    return torch.cat([tensor, torch.ones(num_rows, tensor.shape[1], device=device)])


def sinusoid_matrix(seq_len, vec_size):  # for position embeddings
    posn = torch.zeros(seq_len, vec_size)
    pow_arr = torch.pow(10_000, -torch.arange(0,vec_size,2)/vec_size)
    seq_arr = torch.arange(seq_len)
    combined = torch.outer(seq_arr, pow_arr)

    posn[:, ::2] = torch.sin(combined)
    posn[:, 1::2] = torch.cos(combined)
    return posn



# RELATIVE
def get_index_shifter(seq_len, batch_size, n_head):  # for relative position embeddings
    return torch.remainder((torch.arange(seq_len)*-1)[:,None] + torch.arange(seq_len) - 1, seq_len) + \
        seq_len*torch.arange(seq_len)[:,None] + (torch.arange(n_head)*(seq_len**2))[:,None,None] + \
        (torch.arange(batch_size)*(n_head*seq_len**2))[:,None,None,None]


def get_variable_index_shifter(seq_len, device):  # could use a cache here to save a bit of speed, but might cause OOM on train
    col_idx = torch.remainder((torch.arange(seq_len, device=device)*-1)[:,None] + torch.arange(seq_len, device=device) - 1, seq_len)
    row_idx = torch.arange(seq_len, device=device)[:,None]
    return row_idx, col_idx



# REL_BIAS
def get_rel_bias_indices(block_size, max_distance, num_buckets):
    # adapted from https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/t5/modeling_t5.py#L388
    context_position = torch.arange(block_size, dtype=torch.long)[:, None]
    memory_position = torch.arange(block_size, dtype=torch.long )[None, :]
    relative_position = memory_position - context_position

    # elementwise minimum, basically zeroes out upper right triangle
    relative_position = -torch.min(relative_position, torch.zeros_like(relative_position)) 
    # now relative_position is in the range [0, inf)

    # half of the buckets are for single increment
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    # seq_len - max_exact is the num of positions we have for the log-bins
    # but we only want to go up to position max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)   # ie. log(rel_posn) - log(max_exact)
        / np.log(max_distance / max_exact)  # ie. log(max_distance) - log(max_exact) => at posn max_distance the log -> 1
        * (num_buckets - max_exact)   # so that now at max_distance the log is num_buckets - max_exact
    ).long()

    # ie. basically set stuff past max_position to num_buckets-1
    # set anything that went past num_buckets to num_buckets-1
     # we are definietly "large" out here, so it makes sense
    relative_position_if_large = torch.min(      
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1) 
    )
                                                                               
    return torch.where(is_small, relative_position, relative_position_if_large)



# ROTARY
# from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
def get_rotary_freqs(block_size, theta, rotary_dim):
    freqs = 1. / (theta ** (torch.arange(0, rotary_dim, 2)[:(rotary_dim // 2)].float() / rotary_dim))
    seq = torch.arange(block_size)
    m_freq = torch.outer(seq, freqs)
    return m_freq

# from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py#L33
def rotate_half(x):
    x1, x2 = rearrange(x, '... (d r) -> r ... d', r = 2)
    return rearrange([-x2, x1], 'r ... d -> ... (d r)')  # einops symmetry!


def rotate_q_or_k(q_or_k, freqs, rot_dim):
    t, t_right = q_or_k[..., :rot_dim], q_or_k[..., rot_dim:]  # will be neglibile anyway (unless they are ~0)
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())  # elementwise product of cos(m*theta_i), rotate each 2-subspace
    return torch.cat([t, t_right], dim=-1)   # concatenate rotated part (t) and unrotated part (t_right)

# from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py#L109
def rotate_q_and_k(q, k, freqs):  # inpt is (batch, n_heads, seq_len, head_size)
    rot_dim = freqs.shape[-1]  # only use the first rot_dim units of embedding dimension, since effect on units past this
    freqs = freqs[:q.shape[2]]  # select up to seq_len (freqs is (seq_len, rotary_dim))
    return rotate_q_or_k(q, freqs, rot_dim), rotate_q_or_k(k, freqs, rot_dim)
    


# GENERATING RANDOM SENTENCES
# for text task, randomly samples from words in the dataset that start sentences
def sample_random_start_word(dset):   # for generating the samples to show qualitative progress over time
    while True:  # look over all(-ish) sentence starting words in the dataset, pick one at random
        idx = np.random.randint(0, len(dset))
        random_subset = dset[idx][0]  # [0] selects x over y
        try:  # this can fail if eos token not found, or if eos only found at the very end of the sequence
            return random_subset[np.where(random_subset == dset.encoder.eos_token)[0][0] + 1].item()  # go 1 token past an eos token
        except:  # so just wrap it in a try except until it works
            pass
    
# generic sampling function, takes in a function that produces start words for each of the samples
def generate_samples(net, dsets, start_token_func, num_samples=None, temperature=None):
    if num_samples is None:
        num_samples = net.cfg.num_sample
    start_tokens = [start_token_func(dsets["eval"]) for _ in range(num_samples)]
    stacked_tokens = torch.tensor(start_tokens).unsqueeze(-1).to(net.device)  # (num_samples, 1)
    # print("found start tokens", start_tokens, dsets["eval"].encoder.decode(start_tokens))
    return net.generate(dsets["eval"].encoder, stacked_tokens, temperature=temperature) 


# wrapper for generate_samples that returns randomly generated sentences in a format usable for wandb logging (task == 'text')
def wandb_sample_random_sentences(net, dsets, step):
    rand_sentences = [[step, sent] for sent in generate_samples(net, dsets, sample_random_start_word)]
    return {"rand_sentences": wandb.Table(columns=["step", "sentence"], data=rand_sentences)}


# wrapper for generate_samples that returns randomly generated videos in a format usable for wandb logging (task == 'commavq')
def wandb_sample_random_videos(net, dsets, step):
    start_token_func = lambda x: dsets["eval"].encoder.bos_token
    rand_videos = [video for video in generate_samples(net, dsets, start_token_func)]
    return {f"video{i}": video for i, video in enumerate(rand_videos)}


# ACTIVATION CHECKPOINTING (and potentially saving activations)
def traverse_modules(func, mod):
    has_sub_mods = False
    for sub_mod in mod.children():
        traverse_modules(func, sub_mod)
        has_sub_mods = True
    if not has_sub_mods: # if no submodules, it is an actual operation
        func(mod)