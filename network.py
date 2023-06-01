import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os import path as osp
import config_objects


def get_lr(curr_lr, step, cfg: config_objects.ModelCfg):  # Cosine annealing with warmup
    if step < cfg.t_warmup:
        return step / cfg.t_warmup * cfg.lr_max
    


class MLPBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_1 = nn.Linear(cfg.vec_size, cfg.vec_size*4)
        self.act_func1 = nn.GELU()
        self.w_2 = nn.Linear(cfg.vec_size*4, cfg.vec_size)
        self.dropout = nn.Dropout(p=cfg.mlp_dropout)

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

        self.qkv = nn.Linear(self.head_size, 3*self.head_size, bias=False)
        self.out = nn.Linear(self.vec_size, self.vec_size, bias=False)

        self.attn_dropout = nn.Dropout(p=cfg.attn_dropout)
        self.out_dropout = nn.Dropout(p=cfg.out_dropout)

        # lower left triangle (rows = correspond to given location, left/right within a row (cols) = where we are attending to)
        # register_buffer = untrainable parameter, but gets moved onto/off GPU as requested when doing model.to()
        self.register_buffer("causal_mask",
                             torch.tril(torch.ones(cfg.block_size, cfg.block_size)).reshape(1,1, cfg.block_size, cfg.block_size))


    def forward(self, x):
        # x -> (batch, seq_len, vec_size)
        batch, seq_len, _ = x.shape
        x_head = x.view(batch, seq_len, self.n_heads, self.head_size)

        q, k, v = torch.split(self.qkv(x_head), self.head_size, dim=-1)  # heads are (batch, seq_len, n_heads, head_size)

        # q is (batch, n_heads, seq_len, head_size) and k is (batch, n_heads, head_size, seq_len)
        attn_dots = q.transpose(1,2) @ k.transpose(1,2).transpose(2,3)  # attn_dots is (batch, n_heads, seq_len, seq_len)

        # mask out the future
        causal_attn = attn_dots.masked_fill(self.causal_mask[..., :seq_len, :seq_len], -float("inf"))

        attn_scores = F.softmax(causal_attn / np.sqrt(self.head_size), dim=-1) # softmax is (batch, n_heads, seq_len, seq_len)
        attn_scores = self.attn_dropout(attn_scores)

        attn = attn_scores @ v.transpose(1,2)  # v is (batch, n_heads, seq_len, head_size), attn is (batch, n_heads, seq_len, head_size)

        out = self.out(attn.transpose(1,2).reshape(batch, seq_len, self.vec_size))  # (batch, seq_len, vec_size)
        return self.out_dropout(out)

class TransformerBlock():
    def __init__(self, cfg):
        # design pattern: never pass in parameters directly into initializing a nn.Module, always use the cfg object
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vec_size)
        self.mha = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vec_size)
        self.mlp = MLPBlock(cfg)

    def forward(self, x):  
        # this is not Pre-LN nor Post-LN (Pre-LN would be x + self.mha(self.ln1(x)))
        # however, it seems like it might have the same properties, since norms of the residual stream will be increasing
        # with depth. The actual minGPT repo, upon which this is inspired, does mha(ln(x)), which is the actual Pre-LN,
        # but I'm curious how this performs, so I'll leave it for now. It feels like it kinda defeats the point of LN,
        # which is supposed to keep the distributions input into a given layer typical, but who knows.
        x = x + self.ln1(self.mha(x))
        x = x + self.ln2(self.mlp(x))
        return x

class Transformer(nn.Module):
    def __init__(self, name, model_cfg, dset_cfg):
        super().__init__()

        # for saving stuff purposes
        self.name = name
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

    def forward(self, x, logits=False):
        _, seq_len = x.shape
        posn_embeds = self.posn_embed(torch.arange(seq_len)).unsqueeze(0)
        x = self.embed(x) + posn_embeds
        for block in self.blocks:
            x = block(x)
        y = self.unembed(x)
        if logits:
            return y
        return F.softmax(y, dim=-1)
    
    def save_model_state_dict(self, path=None, optim=None):
        if path is None:
            path = self.path
        save_dict = {"model": self.state_dict(),
                     "cfg": self.cfg,
                     "dset_cfg": self.dset_cfg,
                     "best_loss": self.best_loss}
        if optim is not None:
            save_dict["optim"] = optim.state_dict()
        torch.save(save_dict, path)

    def load_model_state_dict(self, path=None, optim=None):
        if path is None:
            path = self.path
        if not osp.exists(path):
            print("No existing model found", path)
            return
        print("Found path of", path)
        load_dict = torch.load(path)
        if optim is not None:
            try:
                optim.load_state_dict(load_dict["optim"])
            except KeyError:
                print("Optimizer state not found!")

        # make sure all layer sizes, blocks are correctly initialized before loading model state dict
        self.cfg = load_dict["cfg"]
        self.dset_cfg = load_dict["dset_cfg"]
        self.initialize_architecture()
        
        self.load_state_dict(load_dict["model"])
        self.best_loss = load_dict["best_loss"]
        



