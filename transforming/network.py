import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR
import numpy as np
from os import path as osp
from contextlib import nullcontext

from . import config_objects


class MLPBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_1 = nn.Linear(cfg.vec_size, cfg.vec_size*4)
        self.act_func1 = nn.GELU()
        self.w_2 = nn.Linear(cfg.vec_size*4, cfg.vec_size)
        self.dropout = nn.Dropout(p=cfg.dropout_mlp)

    def forward(self, x):
        # x -> (batch, seq_len, vec_size)
        return self.w_2(self.act_func1(self.w_1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vec_size % cfg.n_heads == 0

        self.vec_size = cfg.vec_size
        self.n_heads = cfg.n_heads
        self.head_size = cfg.vec_size // cfg.n_heads
        self.flash = cfg.flash
        self.attn_dropout_p = cfg.dropout_attn

        self.qkv = nn.Linear(self.head_size, 3*self.head_size, bias=False)
        self.out = nn.Linear(self.vec_size, self.vec_size, bias=False)

        self.attn_dropout = nn.Dropout(p=self.attn_dropout_p)
        self.out_dropout = nn.Dropout(p=cfg.dropout_out)

        # lower left triangle (rows = correspond to given location, left/right within a row (cols) = where we are attending to)
        # register_buffer = untrainable parameter, but gets moved onto/off GPU as requested when doing model.to()
        self.register_buffer("causal_mask",
                             torch.tril(~torch.ones(cfg.block_size, cfg.block_size, 
                                                   dtype=torch.bool)).reshape(1,1, cfg.block_size, cfg.block_size))


    def forward(self, x):
        # x -> (batch, seq_len, vec_size)
        batch, seq_len, _ = x.shape
        x_head = x.view(batch, seq_len, self.n_heads, self.head_size)

        q, k, v = torch.split(self.qkv(x_head), self.head_size, dim=-1)  # heads are (batch, seq_len, n_heads, head_size)

        if self.flash:  # scaled dot product attention
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                                  dropout_p=self.attn_dropout_p if self.training else 0)
        else:
            # q is (batch, n_heads, seq_len, head_size) and k is (batch, n_heads, head_size, seq_len)
            attn_dots = q.transpose(1,2) @ k.transpose(1,2).transpose(2,3)  # attn_dots is (batch, n_heads, seq_len, seq_len)

            # mask out the future
            causal_attn = attn_dots.masked_fill(self.causal_mask[..., :seq_len, :seq_len], -float("inf"))

            attn_scores = F.softmax(causal_attn / np.sqrt(self.head_size), dim=-1) # softmax is (batch, n_heads, seq_len, seq_len)
            attn_scores = self.attn_dropout(attn_scores)

            attn = attn_scores @ v.transpose(1,2)  # v is (batch, n_heads, seq_len, head_size), attn is (batch, n_heads, seq_len, head_size)

        out = self.out(attn.transpose(1,2).reshape(batch, seq_len, self.vec_size))  # (batch, seq_len, vec_size)
        return self.out_dropout(out)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        # design pattern: never pass in parameters directly into initializing a nn.Module, always use the cfg object
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vec_size)
        self.mha = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vec_size)
        self.mlp = MLPBlock(cfg)
        self.layer_norm_type = cfg.layer_norm_type

    def forward(self, x):  
        # however, it seems like "werid" might have the same properties as Pre-LN, since norms of the residual stream will 
        # be increasing with depth. The actual minGPT repo, upon which this is inspired, does mha(ln(x)), which is the actual 
        # Pre-LN, but I'm curious how this performs, so I'll leave it for now. It feels like it kinda defeats the point of LN,
        # which is supposed to keep the distributions input into a given layer typical, but who knows.
        if self.layer_norm_type == "weird":
            x = x + self.ln1(self.mha(x))
            x = x + self.ln2(self.mlp(x))
        elif self.layer_norm_type == "post":
            x = self.ln1(x + self.mha(x))
            x = self.ln2(x + self.mlp(x))
        else: #  self.layer_norm_type == "pre":
            x = x + self.mha(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, path, model_cfg: config_objects.ExperimentCfg, dset_cfg: config_objects.DatasetCfg):
        super().__init__()

        # for saving stuff purposes
        self.path = path
        self.cfg = model_cfg
        self.dset_cfg = dset_cfg
        self.best_loss = float('inf')
        self.initialize_architecture()  # uses saved cfg to make the layers
    
    def initialize_architecture(self):
        self.embed = nn.Embedding(self.dset_cfg.vocab_size, self.cfg.vec_size)
        if self.cfg.posn_embed_type == "embeds":  # learnable position embeddings rather than sin waves
            self.posn_embed = nn.Embedding(self.cfg.block_size, self.cfg.vec_size)
        else:
            raise NotImplementedError

        self.blocks = nn.ModuleList([TransformerBlock(self.cfg)
                                      for _ in range(self.cfg.n_layer)])

        self.unembed = nn.Linear(self.cfg.vec_size, self.dset_cfg.vocab_size)
        
        # set up autocasting
        self.fwd_ctx = torch.autocast(device_type=self.dset_cfg.device, dtype=getattr(torch, self.cfg.dtype))

        print(f"Num parameters: {self.num_params()} M")
        print(f"Approximate expected train vram usage: {self.expected_vram_size():.2f} GB")

    def num_params(self): # returns num parameters in millions
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1_000_000

    def expected_vram_size(self):  # returns size in approximate GiB during training
        # https://arxiv.org/pdf/2205.05198.pdf
        dtype_size = int(self.cfg.dtype[-2:]) // 8  # won't work with 128-bit precision or 8-bit precision
        mem_per_layer = self.cfg.block_size * self.cfg.batch_size * self.cfg.vec_size * \
            (34 + 5*self.cfg.n_heads*self.cfg.block_size / self.cfg.vec_size)*dtype_size/2 # since paper assumes float16
        mem_model = self.num_params() * dtype_size * 4  # *4 since Adam stores 2, gradients is +11
        return (mem_model + mem_per_layer*self.cfg.n_layer) / (1_024**3)

    def forward(self, x, targets=None):
        with self.fwd_ctx:  # autocast, technically this is slightly different than doing `with fwd_ctx: model(x);`
            _, seq_len = x.shape  # not strictly "correct" since we sort of cut sequences up so position embeddings are wrong here
            posn_embeds = self.posn_embed(torch.arange(seq_len, device=self.dset_cfg.device)).unsqueeze(0)
            x = self.embed(x) + posn_embeds
            for block in self.blocks:
                x = block(x)

            out = self.unembed(x)
            if targets is None:  # ie. inference mode
                return out   # F.softmax(out, dim=-1)
            else:  # flatten across batches and positions so that we can compare indices (in `targets`) to distributions (in `out`)
                #print(out, targets)
                #print(F.cross_entropy(out.view(-1, self.dset_cfg.vocab_size), targets.view(-1)))
                return F.cross_entropy(out.view(-1, self.dset_cfg.vocab_size), targets.view(-1))

    
    def save_model_state_dict(self, **kwargs):  # kwargs should contain optimizer,scheduler, and scaler
        save_dict = {"model": self.state_dict(),
                     "cfg": self.cfg,
                     "dset_cfg": self.dset_cfg,
                     "best_loss": self.best_loss}
        
        for k,obj in kwargs.items():  # add optimizer, scheduler, scaler state dicts
            save_dict[k] = obj.state_dict()

        torch.save(save_dict, self.path)

    def load_model_state_dict(self, map_location, **kwargs):  # kwargs should contain optimizer,scheduler, and scaler
        if not osp.exists(self.path):
            print("No existing model found", self.path)
            return
        print("Found path of", self.path)
        load_dict = torch.load(self.path, map_location=map_location)

        for k,obj in kwargs.items():  # load optimizer, scheduler, scaler state dicts
            obj.load_state_dict(load_dict[k])

        # make sure all layer sizes, blocks are correctly initialized before loading model state dict
        self.cfg = load_dict["cfg"]  # this shouldn't be necessary since loading optim beforehand requires that it be
        self.dset_cfg = load_dict["dset_cfg"]  # correctly initialized already
        # self.initialize_architecture()
        
        self.load_state_dict(load_dict["model"])
        self.best_loss = load_dict["best_loss"]
        

    def get_optim(self): # optim, sched, scaler, autocast context
        # set initial LR to 1 so that the LinearLR/warmup stage works better
        # ZeroRedundancyOptimizer
        optim = torch.optim.Adam(self.parameters(), weight_decay=self.cfg.weight_decay, lr=self.cfg.lr_max)
        warmup_sched = LinearLR(optim, start_factor=self.cfg.lr_min/self.cfg.lr_max, 
                                end_factor=1., total_iters=self.cfg.t_warmup)
        # t_decay - t_warmup since we include the first t_warmup iters in the "decay" period
        decay_sched = CosineAnnealingLR(optim, T_max=self.cfg.t_decay-self.cfg.t_warmup, eta_min=self.cfg.lr_min)
        const_sched = MultiplicativeLR(optim, lr_lambda=lambda step: 1)

        combined_sched = SequentialLR(optim, [warmup_sched, decay_sched, const_sched], 
                                      milestones=[self.cfg.t_warmup, self.cfg.t_decay])
        
        scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.dtype == "float16"))  # only need scaling on float16
        return scaler, combined_sched, optim


# class OptimizerState():  # if there is one more thing, ill consider actually implementing something like this
#     def __init__(self, scheduler, scaler, optimizer):
#         self.optim = optimizer
#         self.sched = scheduler
#         self.scaler = scaler

#     def state_dict(self):
#         save_dict = {}
#         for key,obj in self.items():
#           save_dict[key] = obj.state_dict()

#     def load_state_dict(self, state_dict):
#           for key,obj in self.items():
#               obj.load_state_dict(state_dict[key])


