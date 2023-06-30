import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR
from torch.distributed.optim import ZeroRedundancyOptimizer # type: ignore
import numpy as np
from os import path as osp
from typing import List, Union
import torch.utils.checkpoint as ckpt
import warnings
from contextlib import nullcontext


from . import config_objects
from . import utils
from .utils import rprint


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


def get_variable_index_shifter(seq_len, device):
    col_idx = torch.remainder((torch.arange(seq_len, device=device)*-1)[:,None] + torch.arange(seq_len, device=device) - 1, seq_len)
    row_idx = torch.arange(seq_len, device=device)[:,None]
    return row_idx, col_idx



class MLPBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_1 = nn.Linear(cfg.vec_size, cfg.vec_size*4, bias=cfg.linear_bias)
        self.act_func = nn.GELU()
        self.w_2 = nn.Linear(cfg.vec_size*4, cfg.vec_size, bias=cfg.linear_bias)
        self.dropout = nn.Dropout(p=cfg.dropout_mlp)

    def forward(self, x):
        # x -> (batch, seq_len, vec_size)
        # print("s0", x.norm())
        # s1 = self.w_1(x)
        # print("s1", s1.norm())
        # s2 = self.act_func(s1)
        # print("s2", s2.norm())
        # s3 = self.w_2(s2)
        # print("s3", s3.norm())
        # s4 = self.dropout(s3)
        # print("s4", s4.norm())
        # return s4
        return self.dropout(self.w_2(self.act_func(self.w_1(x))))


class CustomNormalizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width = cfg.vec_size

        self.scale = nn.Parameter(torch.ones(self.width))
        self.bias = nn.Parameter(torch.zeros(self.width)) if cfg.normalizer_bias else None
        self.norm_type = cfg.normalizer_type
        self.p = cfg.rmsnorm_p  # only has effect if norm_type == "RMSNorm", modifies norm calculation to be pRMSNorm instead
        self.eps = cfg.normalizer_eps
        self.frac_size = torch.tensor(int(self.width * self.p))

    def forward(self, x):
        if self.norm_type == "LayerNorm":
            return F.layer_norm(x, self.width, self.scale, self.bias, eps=self.eps)
        elif self.norm_type == "RMSNorm":  # mostly from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
            norm_selection = x[..., :self.frac_size]  # they do a split, not sure if that's better or worse
            rms_norm = norm_selection.norm(2, dim=-1, keepdim=True) / torch.sqrt(self.frac_size)
            # print("norm full", x.norm(2, dim=-1, keepdim=True) / np.sqrt(self.width), "norm calc", rms_norm)


            x_normed = x / (rms_norm + self.eps)
            # print("after normed", x_normed.norm(), self.scale)
            if self.bias:
                return x_normed * self.scale + self.bias
            return x_normed * self.scale
        else:
            raise NotImplementedError(f"Unrecognized normalizer type '{self.norm_type}'")


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, main_transformer: "Transformer"):
        super().__init__()
        assert cfg.vec_size % cfg.n_heads == 0

        self.vec_size = cfg.vec_size
        self.n_heads = cfg.n_heads
        self.head_size = cfg.vec_size // cfg.n_heads
        self.flash = cfg.flash
        self.attn_dropout_p = cfg.dropout_attn
        self.posn_embed_type = cfg.posn_embed_type

        # need to do this weird call to bypass nn.Module setattr checks
        object.__setattr__(self, "main_transformer", main_transformer)  # save this so we can access the needed buffers


        if cfg.flash and cfg.posn_embed_type != "base":
            raise ValueError(f"Flash attention is currently incompatible with posn_embed type '{cfg.posn_embed_type}'."
                             f" Flash attention only supports posn_embed_type == 'base'")


        if cfg.posn_embed_type == "relative":  # https://arxiv.org/pdf/1901.02860.pdf
            self.posn_vect_u = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_size))
            self.posn_vect_v = nn.Parameter(torch.zeros(1, self.n_heads, 1, self.head_size))
            self.posn_key_mat = nn.Linear(self.vec_size, self.vec_size, bias=False)  # W_k,R in their notation

            if cfg.posn_embed_learnable:  # disable this since we are doing the global_buffers thing
                raise ValueError("Probably should not combine learnable position embeddings with relative embeddings,"
                                 " since the sinusoid one uses a learnable matrix anyway")
                #self.relative_posn_embed = nn.Linear(cfg.block_size, cfg.vec_size)


        self.qkv = nn.Linear(self.vec_size, 3*self.vec_size, bias=False)
        self.out = nn.Linear(self.vec_size, self.vec_size, bias=False)

        self.attn_dropout = nn.Dropout(p=self.attn_dropout_p)
        self.out_dropout = nn.Dropout(p=cfg.dropout_out)


    def forward(self, x):
        # x -> (batch, seq_len, vec_size)
        # print(x.shape)
        batch, seq_len, _ = x.shape
        # print(x.shape)
        # x_head = x.view(batch, seq_len, self.n_heads, self.head_size)

        qkv = torch.split(self.qkv(x), self.vec_size, dim=-1)  # q,k,v are (batch, seq_len, vec_size)
        # transpose so that multiplications are done to each position, not each head
        # q,k,v are now (batch, n_head, seq_len, head_size)
        q,k,v = [mat.view(batch, seq_len, self.n_heads, self.head_size).transpose(1,2) for mat in qkv]

        if self.flash:  # scaled dot product attention
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                                  dropout_p=self.attn_dropout_p if self.training else 0)
        else:
            if self.posn_embed_type == "relative":
                # can combine terms a) and c) in XL transformer paper since they both are multiplied by k
                # posn_vect_u (1, n_head, head_size, 1) @ k (batch, n_head, seq_len, head_size) -> (batch, n_head, seq_len, 1)
                attn_dots = (q + self.posn_vect_u) @ k.transpose(2,3)

                # (seq_len, vec_size) @ (vec_size, vec_size) -> (seq_len, vec_size)
                # W_{k,R} @ R_{i-j} in the paper
                rel_posn = self.posn_key_mat(self.main_transformer.relative_posn_embed[-seq_len:]) # type: ignore
                # reshape into (1, n_head, head_size, seq_len)
                # unsqueeze(0) <=> add batch dim <=> apply same "relative" position embeds to all samples in batch
                rel_posn = rel_posn.T.view(self.n_heads, self.head_size, seq_len).unsqueeze(0)
                # (batch, n_head, seq_len, head_size) @ (1, n_head, head_size, seq_len) -> (batch, n_head, seq_len, seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):  # type: ignore
                    query_posn = (q + self.posn_vect_v).float() @ rel_posn.float()
                    # print("query_posn", query_posn.isinf().sum())

                    # if full shape, take advantage of .take speed up
                    if query_posn.shape == self.main_transformer.shift_indices.shape:  # type:ignore
                        #shifted_posn = torch.take(query_posn, self.main_transformer.shift_indices)  # type: ignore
                        shifted_posn = torch.take(query_posn, self.main_transformer.shift_indices)  # type: ignore
                    else:  # about 50% slower
                        row_idx, col_idx = get_variable_index_shifter(seq_len, device=query_posn.device)
                        shifted_posn = query_posn[..., row_idx, col_idx]
                    # object.__setattr__(self, "shifted_posn", shifted_posn)

                    attn_dots = attn_dots.float() + shifted_posn
                    # mask out the future
                    # object.__setattr__(self, "attn_dots", attn_dots)

                    attn_dots = attn_dots.masked_fill(self.main_transformer.causal_mask[..., :seq_len, :seq_len], -float("inf")) #type: ignore
                    # print("attn_dots", attn_dots.isnan().sum())
                    attn_scores = F.softmax(attn_dots / np.sqrt(self.head_size), dim=-1)
                    # print("attn_scores", attn_dots.isnan().sum())


                # print("post_attn_dots", attn_dots.isnan().sum(), attn_dots.norm(dim=-2).max())
            else:  # ie. vanilla transformer
                # q (batch, n_heads, seq_len, head_size) @ k.T (batch, n_heads, head_size, seq_len)
                attn_dots = q @ k.transpose(2,3)  # attn_dots is (batch, n_heads, seq_len, seq_len)

                # mask out the future

                attn_dots = attn_dots.masked_fill(self.main_transformer.causal_mask[..., :seq_len, :seq_len], -float("inf")) #type: ignore

                attn_scores = F.softmax(attn_dots / np.sqrt(self.head_size), dim=-1)
            # print("attn_scores", attn_scores.isnan().sum())

            attn_scores = self.attn_dropout(attn_scores)  # attn_scores is (batch, n_heads, seq_len, seq_len)

            # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, head_size) -> (batch, n_head, seq_len, vec_size)

            attn = attn_scores @ v 


        # transpose first so that the reshape operates on the last 2 dimensions only, stacking heads
        out = self.out(attn.transpose(1,2).reshape(batch, seq_len, self.vec_size))  # (batch, seq_len, vec_size)
        # print("end mha", out.norm())
        return self.out_dropout(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, main_transformer: "Transformer"):
        # design pattern: never pass in parameters directly into initializing a nn.Module, always use the cfg object
        super().__init__()
        self.ln1 = CustomNormalizer(cfg)
        self.mha = MultiHeadAttention(cfg, main_transformer)
        self.ln2 = CustomNormalizer(cfg)
        self.mlp = MLPBlock(cfg)
        self.layer_norm_posn = cfg.layer_norm_posn

    def forward(self, x):
        # however, it seems like "werid" might have the same properties as Pre-LN, since norms of the residual stream will
        # be increasing with depth. The actual minGPT repo, upon which this is inspired, does mha(ln(x)), which is the actual
        # Pre-LN, but I'm curious how this performs, so I'll leave it for now. It feels like it kinda defeats the point of LN,
        # which is supposed to keep the distributions input into a given layer typical, but who knows.
        if self.layer_norm_posn == "weird":
            x = x + self.ln1(self.mha(x))
            x = x + self.ln2(self.mlp(x))
        elif self.layer_norm_posn == "post":
            x = self.ln1(x + self.mha(x))
            x = self.ln2(x + self.mlp(x))
        elif self.layer_norm_posn == "pre":
            # print("pre", x.norm(), x.isnan().sum())
            x = x + self.mha(self.ln1(x))
            # print("post mha", x.norm(), x.isnan().sum())
            x = x + self.mlp(self.ln2(x))
            # print("post mlp", x.norm(), x.isnan().sum())

        else:
            raise NotImplementedError(f"Layer norm posn '{self.layer_norm_posn}' not supported.")
        return x

class Transformer(nn.Module):
    def __init__(self, path, model_cfg: config_objects.ExperimentCfg, dset_cfg: config_objects.DatasetCfg):
        super().__init__()

        # for saving stuff purposes
        self.path = path
        self.cfg = model_cfg
        self.dset_cfg = dset_cfg
        self.best_loss = float('inf')
        self.checkpointing_setup = False  # flag to make sure we only ever set up checkpointing once
        self.initialize_architecture()  # uses saved cfg to make the layers

    def initialize_architecture(self):
        assert self.cfg.posn_embed_type in ["base", "relative"]


        if self.cfg.posn_embed_type == "relative":  # register these only once to save memory
            # .flip to match decreasing order of pos embeddings, per XL transformer paper (see appendix B, defn of Q matrix)
            self.register_buffer("relative_posn_embed", sinusoid_matrix(self.cfg.block_size, self.cfg.vec_size).flip(0).contiguous())
            
            # for full sized inputs, pre-computing everything and doing a .take is fastest, so we have shift_indices
            # for different sized inputs, .take wouldn't work since it requires linear indices, so we use the method
            # of shift_col_indices and shift_row_indices, which is still faster than the original XL transformer
            # implementation, but allows indexing for variable length inputs
            self.register_buffer("shift_indices", get_index_shifter(self.cfg.block_size, self.cfg.batch_size, self.cfg.n_heads))

        # lower left triangle (rows = correspond to given location, left/right within a row (cols) = where we are attending to)
        # register_buffer = untrainable parameter, but gets moved onto/off GPU as requested when doing model.to()
        self.register_buffer("causal_mask",
                            ~torch.tril(torch.ones(self.cfg.block_size, self.cfg.block_size, dtype=torch.bool))
                             .reshape(1,1, self.cfg.block_size, self.cfg.block_size))
        
        self.embed = nn.Embedding(self.dset_cfg.vocab_size, self.cfg.vec_size)

        if self.cfg.posn_embed_type == "base":
            if self.cfg.posn_embed_learnable:  # learnable position embeddings rather than sin waves
                self.posn_embed = nn.Embedding(self.cfg.block_size, self.cfg.vec_size)
            else:
                self.register_buffer("posn_embed", sinusoid_matrix(self.cfg.block_size, self.cfg.vec_size))

        self.blocks = nn.ModuleList([TransformerBlock(self.cfg, self) for _ in range(self.cfg.n_layer)])

        if self.cfg.learnable_unembed:
            self.unembed = nn.Linear(self.cfg.vec_size, self.dset_cfg.vocab_size, bias=False)

        # set up autocasting
        rprint(utils.get_device_type(), "deiece")
        if utils.get_device_type() == "cpu":
            self.fwd_ctx = nullcontext()
        else:
            self.fwd_ctx = torch.autocast(device_type=utils.get_device_type(), dtype=getattr(torch, self.cfg.dtype)) # type: ignore

        if self.cfg.checkpointing:
            self.add_activation_checkpointing()

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
            # rprint("xdev", x.device, "proper_dev", self.device, "posn_embed", self.posn_embed.weight.device)
            x = self.embed(x)
            if self.cfg.posn_embed_type == "base":
                if self.cfg.posn_embed_learnable:
                    x = x + self.posn_embed(torch.arange(seq_len, device=self.device)).unsqueeze(0)
                else:
                    x = x + self.posn_embed[:seq_len].view(1,seq_len,self.cfg.vec_size) # type: ignore

            for i, block in enumerate(self.blocks):
                # if i == 1:  #
                #     return x
                # print(i, torch.topk(x.flatten().abs(), 5).values, x.norm()) 
                x = block(x)  # (batch, seq_len, vec_size)
            
            if self.cfg.learnable_unembed:
                out = self.unembed(x)  # (batch, seq_len, vec_size) @ (vec_size, vocab_size) 
            else:
                out = x @ self.embed.weight.T

            if targets is None:  # ie. inference mode, return logits so we can do temperature stuff
                return out   # F.softmax(out, dim=-1)
            else:  # flatten across batches and positions so that we can compare indices (in `targets`) to distributions (in `out`)
                #print(out, targets)
                #print(F.cross_entropy(out.view(-1, self.dset_cfg.vocab_size), targets.view(-1)))
                loss = F.cross_entropy(out.view(-1, self.dset_cfg.vocab_size), targets.view(-1), label_smoothing=self.cfg.label_smoothing)
                return loss, out

    @torch.no_grad()  # batched sampling
    def generate(self, encoder, prompt: Union[str, List[int], torch.Tensor], temperature=0.2) -> List[str]:
        self.eval()
        if isinstance(prompt, str):
            prompt = encoder.encode(prompt)
        
        if isinstance(prompt, list):
            tokens = torch.tensor(prompt).unsqueeze(0).to(self.device)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt
        else:
            raise NotImplementedError(f"prompt needs to be a str, list[int], or Tensor, instead was {type(prompt)}")

        if tokens.dim() == 1:  # account for prompt size 1
            tokens = tokens.unsqueeze(0)
        # => 2*block_size is max generated size
        finished_sentences = []
        while tokens.shape[0] != 0:
            # unsqueeze to add batch dim, 0 to rm it, -1 to see last posn in sequence
            # print("token shape", tokens.shape)
            logits = self(tokens[:, -self.cfg.block_size:])
            # print("logit shape", logits.shape)
            if temperature > 0:
                #print(logits[:, -1, :].shape)
                # input("about to div temp")
                # print("logits", logits[0, -1].isinf().sum())
                logits[:, -1, :] /= temperature
                # input("about to softm")
                # print("logits", logits[0, -1])
                probs = F.softmax(logits, dim=-1)  # 0 to select batch, -1 to select last position in sequence
                # print("probs", probs[0,-1])

                # input("about to multinomail")
                #print("probs shape", probs.shape, probs[:, -1, :].shape)
                if probs.isnan().any():
                    if logits[:, -1, :].isnan().any():
                        print("DETECTED NANs, ABORTING...")
                        for sentence in tokens:
                            finished_sentences.append(sentence)
                        break
                    else:
                        print("DETECTED NANs, DOING ARGMAX BACKUP...")
                        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
                else:
                    next_token = torch.multinomial(probs[:, -1, :], 1)
            else:  # argmax, temp 0
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)  # concat onto seq_len dimension
            # print("tokens shape", tokens.shape)
            retain = list(range(tokens.shape[0]))
            for i, sentence in enumerate(tokens):
                if sentence[-1] == encoder.eos_token or sentence.shape[0] >= 2*self.cfg.block_size:
                    finished_sentences.append(sentence)
                    retain.remove(i)
                    print("done with", i, "shape was", sentence.shape, "retain is", retain)

            tokens = tokens[retain]
            #print("new shape", tokens.shape, tokens[0, -1], next_token, self.embed.weight.shape)
        return [encoder.decode(sent.squeeze().detach().cpu().numpy()) for sent in finished_sentences]

    def save_model_state_dict(self, optim=None, path=None, **kwargs):  # kwargs should contain optimizer,scheduler, and scaler
        if path is None:
            path = self.path

        save_dict = {"model": self.state_dict(),
                     "cfg": self.cfg,
                     "dset_cfg": self.dset_cfg,
                     "best_loss": self.best_loss}

        if self.cfg.ddp and self.cfg.zero and optim is not None:  # sharded optimizer => need to collate before saving
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

        if utils.get_rank() == 0:
            rprint("saving being done")
            torch.save(save_dict, path)  # only do teh actual io if rank 0

    def load_model_state_dict(self, map_location, path=None, **kwargs) -> bool:
        # kwargs should contain optimizer,scheduler, and scaler
        if path is None:
            path = self.path

        if not osp.exists(path):
            print("No existing model found", path)
            return False
        print("Found path of", path)
        load_dict = torch.load(path, map_location=map_location)  # load everything onto cpu so that for sharded models, it doesn't fail
        rprint(load_dict["model"].keys())
        rprint("keys", load_dict.keys())
        rprint("is loading to", map_location)
        # rprint("has num_params", len(kwargs["optim"].param_groups[0]["params"]))
        for k,obj in kwargs.items():  # load optimizer, scheduler, scaler state dicts
            if k == "optim":
                rprint(len(obj.param_groups[0]["params"]), type(obj.param_groups[0]["params"][0]), len(load_dict[k]["param_groups"][0]["params"]))
            obj.load_state_dict(load_dict[k])

        # make sure all layer sizes, blocks are correctly initialized before loading model state dict
        self.cfg = load_dict["cfg"]  # this shouldn't be necessary since loading optim beforehand requires that it be
        self.dset_cfg = load_dict["dset_cfg"]  # correctly initialized already
        # self.initialize_architecture()   # bad for setting up checkpointing reasons

        self.load_state_dict(load_dict["model"])
        self.best_loss = load_dict["best_loss"]
        return True


    def get_optim(self): # optim, sched, scaler, autocast context
        # set initial LR to 1 so that the LinearLR/warmup stage works better
        if self.cfg.zero and self.cfg.ddp:
            optim = ZeroRedundancyOptimizer(self.parameters(),
                                            optimizer_class=torch.optim.Adam,
                                            lr=self.cfg.lr_max, weight_decay=self.cfg.weight_decay)
        else:
            optim = torch.optim.Adam(self.parameters(), weight_decay=self.cfg.weight_decay, lr=self.cfg.lr_max)
        warmup_sched = LinearLR(optim, start_factor=self.cfg.lr_min/self.cfg.lr_max,
                                end_factor=1., total_iters=self.cfg.t_warmup)
        # t_decay - t_warmup since we include the first t_warmup iters in the "decay" period
        decay_sched = CosineAnnealingLR(optim, T_max=self.cfg.t_decay-self.cfg.t_warmup, eta_min=self.cfg.lr_min)
        const_sched = MultiplicativeLR(optim, lr_lambda=lambda step: 1)

        combined_sched = SequentialLR(optim, [warmup_sched, decay_sched, const_sched],
                                      milestones=[self.cfg.t_warmup, self.cfg.t_decay])

        # only need scaling on float16
        scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.dtype == "float16" and utils.get_device_type() == "cuda"))  # type: ignore
        return scaler, combined_sched, optim

    def add_activation_checkpointing(self):
        if self.checkpointing_setup:
            return
        if self.cfg.compile:
            raise ValueError("Activation checkpointing is currently incompatible with torch.compile.")
        self.checkpointing_setup = True

        def add_checkpointing(module):
            if isinstance(module, (nn.GELU, CustomNormalizer)):
                orig_forward = module.forward
                def checkpoint_fwd(*inputs):
                    # use_reentrant=False since then we can do do autograd.grad() (more features in general are supported)
                    return ckpt.checkpoint(orig_forward, *inputs, preserve_rng_state=False, use_reentrant=False)
                module.forward = checkpoint_fwd  # type: ignore

        utils.traverse_modules(add_checkpointing, self)
        # to get model vram, you need to do looped waiting
        # for speed-wise on small, estimates were smoothed with gamma=0.999, sampled at the last step where all are available
        # for small sizes (batch=1, seq_len=10, heads=1, layers=10, vec_size=1280)
            # memory-wise
                # model only (12.5), baseline (57.04), traverse (57.04), GeLU+Norm (57.04), MHA+MLP+GeLU+Norm (56.41), blocks (56.4)
                # traverse <=> manually doing it on the respective operations (57.04 vs. 57.04)
                # checkpointing on GeLU and Normalizer saves almost nothing compared to baseline
                # checkpointing on whole blocks <=> checkpointing on all of MHA, MLP, GeLU, Normalizer
            # speed-wise
                # baseline (7.22e-3), GeLU+Norm (7.31e-3), traverse (7.34e-3), MHA+MLP+GeLU+Norm (8.04e-3), blocks (8.90e-3)
                # its very close, but traverse <=> manually doing it (7.22e-3 vs 7.31e-3)
                # checkpointing on blocks is slowest (8.90e-3 tok_time)
                # checkpointing whole blocks is slower than checkpointing all of MHA, MLP, GeLU, Normalizer (8.90e-3 vs. 8.04e-3)
        # for large sizes (batch=4, seq_len=1024, heads=12, layers=12, vec_size=1536)
            # memory-wise
                # model only (14.6), baseline (90.27), GeLu+Normalizer (86.54), traverse (86.46), blocks (84.38)
                # traverse is very slightly better than specifying on GeLU and normalizer (86.54 vs. 86.46)
                # traverse saves about ~3.5% memory usage, blocks saves about double that (~6%)
            # speed-wise
                # baseline (1.96e-4), GeLU+Normalizer (1.99e-4), traverse (2.01e-4), blocks (2.60e-4)
                # traverse <=> GeLU+Normalizer (2.008e-4 vs 1.993e-4)
                # full blocks is significantly slower ~25% (2.60e-4 vs 1.96e-4)
                # traverse very close to no checkpointing (2.008e-4 vs 1.960e-4)

    @property  # so that it works through a .to
    def device(self):
        return self.embed.weight.device

