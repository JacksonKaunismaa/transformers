import torch
import numpy as np
from einops import rearrange
import os.path as osp
import glob
import json
import dataclasses

from . import utils
from .utils import rprint

class Resumer():
    def __init__(self, resume_path, net, resume=True, model_ckpt_dir="/checkpoint"):
        self.resume_path = resume_path
        self.model_ckpt_dir = model_ckpt_dir

        self.wandb_run_id = None
        self.model_save_path = None
        self.job_id = net.cfg.job_id
        self.net = net

        if osp.exists(resume_path):
            if resume:
                self.load()  # this call only loads in resume parameters, sets net.cfg, but doesn't load state_dicts
            else:
                existing_paths = glob.glob(osp.join(resume_path, "*"))
                nums = [int(path.split(".")[-1]) for path in existing_paths] + [0]  # add [0] so there is always at least 1 element
                next_num = max(nums) + 1
                self.resume_path = osp.join(resume_path, f".{next_num}")
                print("Warning: Creating new resume file since you specified an already existing resume path, but resume=False")
                print("New path is: ", self.resume_path)

        self.name = osp.basename(self.resume_path)
        if not hasattr(self, "model_save_path"):  # only way in which its set is if resume=True and resume_path exists
            self.update_model_save_path()    # otherwise, set it to /checkpoint/curr_job_id/name

        print("Model name is", self.name, "will be saving to", self.model_save_path)

    def update_model_save_path(self):
        self.model_save_path = osp.join(self.model_ckpt_dir, str(self.job_id), self.name)
        print("Updated save path to", self.model_save_path)

    def save(self, optim=None, **kwargs):  # kwargs should contain optimizer,scheduler, and scaler
        if utils.get_rank() == 0:   # update resume file
            with open(self.resume_path, "w") as f:
                resume_info = dict(wandb_run_id=self.wandb_run_id,
                                   cfg=dataclasses.asdict(self.net.cfg),
                                   model_save_path=self.model_save_path)
                json.dump(resume_info, f)

        # save state_dicts
        save_dict = {"model": self.net.state_dict(),
                     "cfg": self.net.cfg,
                     "dset_cfg": self.net.dset_cfg,
                     "best_loss": self.net.best_loss}

        if self.net.cfg.ddp and self.net.cfg.zero and optim is not None:  # sharded optimizer => need to collate before saving
            rprint("is consolidating")
            # rprint("Has state", optim.state.keys())
            rprint("pre consolidate, has num_params", len(optim.param_groups[0]["params"]))
            optim.consolidate_state_dict()
            rprint("post consolidate, rank, has num_params", len(optim.param_groups[0]["params"]))

            if utils.get_rank() == 0:
                # rprint("is saving", optim.state_dict().keys())
                rprint("saving num_params", len(optim.state_dict()["param_groups"][0]["params"]))
                rprint("state_dict device is", optim.state_dict()["state"][0]["exp_avg"].device)
                save_dict["optim"] = optim.state_dict()  # it transfers shards to CPU first, so GPU mem is fine
            rprint("done consolidating")
        elif optim is not None:
            save_dict["optim"] = optim.state_dict()

        for k,obj in kwargs.items():  # add scheduler, scaler state dicts
            save_dict[k] = obj.state_dict()

        if utils.get_rank() == 0: # only do the actual IO if rank == 0
            rprint("saving being done")
            torch.save(save_dict, self.model_save_path)  # type: ignore 

    def load(self, map_location=None, update_model_save_path=False, **kwargs) -> bool:  # returns True if loading succeeds
        with open(self.resume_path, "r") as f:
            resume_info = json.load(f)
        self.wandb_run_id = resume_info["wandb_run_id"]
        self.model_save_path = resume_info["model_save_path"]
        dataclasses.replace(self.net.cfg, **resume_info["cfg"])

        if map_location is None:  # if map_location is not specified, don't actually load anything
            print("map_location not specified, skipping loading state_dicts")
            return True
        
        if not osp.exists(self.model_save_path):
            print("No existing model found", self.model_save_path)
            return False
        
        print("Found path of", self.model_save_path)
        load_dict = torch.load(self.model_save_path, map_location=map_location)  # load onto cpu so sharded models don't OOM
        rprint(load_dict["model"].keys())
        rprint("keys", load_dict.keys())
        rprint("is loading to", map_location)
        # rprint("has num_params", len(kwargs["optim"].param_groups[0]["params"]))
        for k,obj in kwargs.items():  # load optimizer, scheduler, scaler state dicts
            if k == "optim":
                rprint(len(obj.param_groups[0]["params"]), type(obj.param_groups[0]["params"][0]), len(load_dict[k]["param_groups"][0]["params"]))
            obj.load_state_dict(load_dict[k])

        # make sure all layer sizes, blocks are correctly initialized before loading model state dict
        self.net.cfg = load_dict["cfg"]  # this shouldn't be necessary since loading optim beforehand requires that it be
        self.net.dset_cfg = load_dict["dset_cfg"]  # correctly initialized already
        # self.initialize_architecture()   # bad for setting up checkpointing reasons

        # there should probably just be a thing that just regenerates all buffers based on the new cfg, rather than loading them in
        if "shift_indices" in load_dict["model"]:  # adjust for batch size
            load_dict["model"]["shift_indices"] = load_dict["model"]["shift_indices"][:self.net.cfg.batch_size]
        self.net.load_state_dict(load_dict["model"])
        self.net.best_loss = load_dict["best_loss"]

        if update_model_save_path:  # 1. load model based on value in resume file (potentially old /checkpoint/* dir)
            self.update_model_save_path()  # 2. subsequent saves of the model should be to /checkpoint/curr_job_id/self.name
        return True
    




def sinusoid_matrix(seq_len, vec_size):  # for position embeddings
    posn = torch.zeros(seq_len, vec_size)
    pow_arr = torch.pow(10_000, -torch.arange(0,vec_size,2)/vec_size)
    seq_arr = torch.arange(seq_len)
    combined = torch.outer(seq_arr, pow_arr)

    posn[:, ::2] = torch.sin(combined)
    posn[:, 1::2] = torch.cos(combined)
    return posn


def get_index_shifter(seq_len, batch_size, n_head):  # for relative position embeddings
    return torch.remainder((torch.arange(seq_len)*-1)[:,None] + torch.arange(seq_len) - 1, seq_len) + \
        seq_len*torch.arange(seq_len)[:,None] + (torch.arange(n_head)*(seq_len**2))[:,None,None] + \
        (torch.arange(batch_size)*(n_head*seq_len**2))[:,None,None,None]


def get_variable_index_shifter(seq_len, device):  # could use a cache here to save a bit of speed, but might cause OOM on train
    col_idx = torch.remainder((torch.arange(seq_len, device=device)*-1)[:,None] + torch.arange(seq_len, device=device) - 1, seq_len)
    row_idx = torch.arange(seq_len, device=device)[:,None]
    return row_idx, col_idx


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


# from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py#L33
def rotate_half(x):
    x1, x2 = rearrange(x, '... (d r) -> r ... d', r = 2)
    return rearrange([-x2, x1], 'r ... d -> ... (d r)')  # einops symmetry!


# from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py#L109
def rotate_keys_or_queries(q_or_k, cos_freqs, sin_freqs):
    rot_dim = cos_freqs.shape[-1]  # only use the first rot_dim units of embedding dimension, since effect on units past this
    # t, t_right = q_or_k[..., :rot_dim], q_or_k[..., rot_dim:]  # will be neglibile anyway (unless they are ~0)
    q_or_k[..., :rot_dim] = (q_or_k[..., rot_dim] * cos_freqs) + (rotate_half(q_or_k[..., rot_dim]) * sin_freqs)  # elementwise product of cos(m*theta_i), rotate each 2-subspace
    #return torch.cat([t, t_right], dim=-1)   # concatenate rotated part (t) and unrotated part (t_right)
    return q_or_k



def sample_random_start_word(dset):   # for generating the samples to show qualitative progress over time
    while True:  # look over all(-ish) sentence starting words in the dataset, pick one at random
        idx = np.random.randint(0, len(dset))
        random_subset = dset[idx][0]  # [0] selects x over y
        try:  # this can fail if eos token not found, or if eos only found at the very end of the sequence
            return random_subset[np.where(random_subset == dset.encoder.eos_token)[0][0] + 1].item()  # go 1 token past an eos token
        except:  # so just wrap it in a try except until it works
            pass
    

# sample generate a bunch of sentences using random start words
def sample_random_sentences(net, dsets, num_sample=5, temperature=0.2):
    start_tokens = [sample_random_start_word(dsets["eval"]) for _ in range(num_sample)]
    stacked_tokens = torch.tensor(start_tokens).unsqueeze(-1).to(net.device)  # (num_samples, 1)
    # print("found start tokens", start_tokens, dsets["eval"].encoder.decode(start_tokens))
    return net.generate(dsets["eval"].encoder, stacked_tokens, temperature=temperature) 


def traverse_modules(func, mod):
    has_sub_mods = False
    for sub_mod in mod.children():
        traverse_modules(func, sub_mod)
        has_sub_mods = True
    if not has_sub_mods: # if no submodules, it is an actual operation
        func(mod)