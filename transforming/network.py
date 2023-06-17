import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR
from torch.distributed.optim import ZeroRedundancyOptimizer # type: ignore
import numpy as np
from os import path as osp
from typing import List, Union
import torch.utils.checkpoint as ckpt


from . import config_objects
from . import utils
from .utils import rprint


class MLPBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_1 = nn.Linear(cfg.vec_size, cfg.vec_size*4, bias=cfg.linear_bias)
        self.act_func = nn.GELU()
        self.w_2 = nn.Linear(cfg.vec_size*4, cfg.vec_size, bias=cfg.linear_bias)
        self.dropout = nn.Dropout(p=cfg.dropout_mlp)

    def forward(self, x):
        # x -> (batch, seq_len, vec_size)
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

    def forward(self, x):
        if self.norm_type == "LayerNorm":
            return F.layer_norm(x, self.width, self.scale, self.bias, eps=self.eps)
        elif self.norm_type == "RMSNorm":  # mostly from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
            frac_size = int(self.width * self.p)

            norm_selection = x[..., :frac_size]  # they do a split, not sure if that's better or worse
            rms_norm = norm_selection.norm(2, dim=-1, keepdim=True) / np.sqrt(frac_size)

            x_normed = x / (rms_norm + self.eps)
            if self.bias:
                return x_normed * self.scale + self.bias
            return x_normed * self.scale
        else:
            raise NotImplementedError("Unrecognized normalizer type", self.norm_type)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vec_size % cfg.n_heads == 0

        self.vec_size = cfg.vec_size
        self.n_heads = cfg.n_heads
        self.head_size = cfg.vec_size // cfg.n_heads
        self.flash = cfg.flash
        self.attn_dropout_p = cfg.dropout_attn

        self.qkv = nn.Linear(self.vec_size, 3*self.vec_size, bias=False)
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
        # x_head = x.view(batch, seq_len, self.n_heads, self.head_size)

        qkv = torch.split(self.qkv(x), self.vec_size, dim=-1)  # q,k,v are (batch, seq_len, n_heads, head_size)
        # transpose so that multiplications are done to each position, not each head
        q,k,v = [mat.view(batch, seq_len, self.n_heads, self.head_size).transpose(1,2) for mat in qkv]

        if self.flash:  # scaled dot product attention
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                                  dropout_p=self.attn_dropout_p if self.training else 0)
        else:
            # q is (batch, n_heads, seq_len, head_size) and k is (batch, n_heads, head_size, seq_len)
            attn_dots = q @ k.transpose(2,3)  # attn_dots is (batch, n_heads, seq_len, seq_len)

            # mask out the future
            if self.training:
                attn_dots = attn_dots.masked_fill(self.causal_mask[..., :seq_len, :seq_len], -float("inf")) # type: ignore

            attn_scores = F.softmax(attn_dots / np.sqrt(self.head_size), dim=-1)
            attn_scores = self.attn_dropout(attn_scores)  # attn_scores is (batch, n_heads, seq_len, seq_len)

            attn = attn_scores @ v  # v is (batch, n_heads, seq_len, head_size), attn is (batch, n_heads, seq_len, head_size)

        # transpose first so that the reshape operates on the last 2 dimensions only, stacking heads
        out = self.out(attn.transpose(1,2).reshape(batch, seq_len, self.vec_size))  # (batch, seq_len, vec_size)
        return self.out_dropout(out)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        # design pattern: never pass in parameters directly into initializing a nn.Module, always use the cfg object
        super().__init__()
        self.ln1 = CustomNormalizer(cfg)
        self.mha = MultiHeadAttention(cfg)
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
            x = x + self.mha(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
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
        self.embed = nn.Embedding(self.dset_cfg.vocab_size, self.cfg.vec_size)
        if self.cfg.posn_embed_type == "embeds":  # learnable position embeddings rather than sin waves
            self.posn_embed = nn.Embedding(self.cfg.block_size, self.cfg.vec_size)
        else:
            raise NotImplementedError

        self.blocks = nn.ModuleList([TransformerBlock(self.cfg)
                                      for _ in range(self.cfg.n_layer)])

        self.unembed = nn.Linear(self.cfg.vec_size, self.dset_cfg.vocab_size)
        
        # set up autocasting
        rprint(utils.get_device_type(), "deiece")
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
            posn_embeds = self.posn_embed(torch.arange(seq_len, device=self.device)).unsqueeze(0)
            x = self.embed(x) + posn_embeds
            for block in self.blocks:
                x = block(x)

            out = self.unembed(x)
            if targets is None:  # ie. inference mode, return logits so we can do temperature stuff
                return out   # F.softmax(out, dim=-1)
            else:  # flatten across batches and positions so that we can compare indices (in `targets`) to distributions (in `out`)
                #print(out, targets)
                #print(F.cross_entropy(out.view(-1, self.dset_cfg.vocab_size), targets.view(-1)))
                return F.cross_entropy(out.view(-1, self.dset_cfg.vocab_size), targets.view(-1)), out

    @torch.no_grad()    # probably should make a batched version of this ?
    def generate(self, encoder, prompt: Union[str, List[int]], temperature=0.2) -> str:  
        self.eval()
        if isinstance(prompt, str):
            prompt = encoder.encode(prompt)

        tokens = torch.tensor(prompt).unsqueeze(0).to(self.device)
        if tokens.dim() == 1:  # account for prompt size 1
            tokens = tokens.unsqueeze(0)
        # => 2*block_size is max generated size
        while tokens[0, -1] != encoder.eos_token and tokens.shape[1] < 2*self.cfg.block_size:
            # unsqueeze to add batch dim, 0 to rm it, -1 to see last posn in sequence
            # print("token shape", tokens.shape)
            logits = self(tokens[:, -self.cfg.block_size:])
            #print("logit shape", logits.shape)
            if temperature > 0:
                #print(logits[:, -1, :].shape)
                # input("about to div temp")
                logits[:, -1, :] /= temperature
                # input("about to softm")
                probs = F.softmax(logits, dim=-1)  # 0 to select batch, -1 to select last position in sequence
                # input("about to multinomail")
                #print("probs shape", probs.shape, probs[:, -1, :].shape)
                next_token = torch.multinomial(probs[:, -1, :], 1)
            else:  # argmax, temp 0
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)  # concat onto seq_len dimension
            #print("new shape", tokens.shape, tokens[0, -1], next_token, self.embed.weight.shape)
        return encoder.decode(tokens.squeeze().detach().cpu().numpy())
    
    def save_model_state_dict(self, optim=None, path=None, **kwargs):  # kwargs should contain optimizer,scheduler, and scaler
        if path is None:
            path = self.path

        save_dict = {"model": self.state_dict(),
                     "cfg": self.cfg,
                     "dset_cfg": self.dset_cfg,
                     "best_loss": self.best_loss}

        if self.cfg.ddp and self.cfg.zero and optim is not None:  # sharded optimizer => need to collate before saving
            rprint("is consolidating")
            rprint("Has state", optim.state.keys())
            rprint("pre consolidate, has num_params", len(optim.param_groups[0]["params"]))
            optim.consolidate_state_dict()
            rprint("post consolidate, rank, has num_params", len(optim.param_groups[0]["params"]))

            if utils.get_rank() == 0:
                rprint("is saving", optim.state_dict().keys())
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
        scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.dtype == "float16"))  # type: ignore 
        return scaler, combined_sched, optim

    def add_activation_checkpointing(self):
        if self.checkpointing_setup:
            return
        self.checkpointing_setup = True

        def add_checkpointing(module):
            if isinstance(module, (nn.GELU, CustomNormalizer)):
                orig_forward = module.forward
                def checkpoint_fwd(*inputs):
                    return ckpt.checkpoint(orig_forward, *inputs, preserve_rng_state=False)
                module.forward = checkpoint_fwd  # type: ignore
        
        # def add_checkpointing(mod):
            # def save_inpt_hook(layer, inpt):  # save the input so that the ckpt.checkpoint knows what inpt is
            #     self._inputs[curr_name] = inpt
            # def free_inpt_hook(layer, inpt, outpt):  # free the memory after so we don't waste space in the _inputs dict
            #     self._inputs[curr_name] = None # mark memory as free
                # mod.register_forward_pre_hook(save_inpt_hook)  # register hooks so input is available to the ckpt.checkpoint
                # mod.register_forward_hook(free_inpt_hook)
                # replace forward with checkpointed version
                    
                # mod.forward = checkpoint_fwd(mod)#ckpt.checkpoint(mod.forward, self._inputs[curr_name]) # type: ignore

        utils.traverse_modules(add_checkpointing, self)
    
    @property  # so that it works through a .to
    def device(self):
        return self.embed.weight.device


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


