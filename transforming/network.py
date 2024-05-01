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
from . import net_utils


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
    def __init__(self, cfg: config_objects.ExperimentCfg, main_transformer: "Transformer"):
        super().__init__()
        assert cfg.vec_size % cfg.n_heads == 0

        self.head_size = cfg.vec_size // cfg.n_heads
        self.cfg = cfg

        # need to do this weird call to bypass nn.Module setattr checks
        # save this so we can access the needed buffers shared between layers
        object.__setattr__(self, "main_transformer", main_transformer) 


        if cfg.flash and cfg.posn_embed_type not in ["base_sinusoid", "base_learnable", "none", "rotary"]:
            raise ValueError(f"Flash attention is currently incompatible with posn_embed type '{cfg.posn_embed_type}'."
                             f" Flash attention only supports posn_embed_type in ['base_sinusoid', 'base_learnable', 'none', 'rotary']")


        if cfg.posn_embed_type == "relative":  # https://arxiv.org/pdf/1901.02860.pdf
            self.posn_vect_u = nn.Parameter(torch.zeros(1, cfg.n_heads, 1, self.head_size))
            self.posn_vect_v = nn.Parameter(torch.zeros(1, cfg.n_heads, 1, self.head_size))
            self.posn_key_mat = nn.Linear(cfg.vec_size, cfg.vec_size, bias=False)  # W_k,R in their notation

        if cfg.mqa_attn:
            self.qkv = nn.Linear(cfg.vec_size, cfg.vec_size + 2*self.head_size, bias=False)
        else:
            self.qkv = nn.Linear(cfg.vec_size, 3*cfg.vec_size, bias=False)
        self.out = nn.Linear(cfg.vec_size, cfg.vec_size, bias=False)

        self.attn_dropout = nn.Dropout(p=cfg.dropout_attn)
        self.out_dropout = nn.Dropout(p=cfg.dropout_out)


    def forward(self, x):
        # x -> (batch, seq_len, vec_size)
        batch, seq_len, _ = x.shape
        
        qkv = self.qkv(x)
        if self.cfg.mqa_attn:
            qkv = (qkv[..., :self.cfg.vec_size],   # q (batch, seq_len, vec_size)
                   qkv[..., self.cfg.vec_size:self.cfg.vec_size+self.head_size],  # k (batch, seq_len, head_size)
                   qkv[..., self.cfg.vec_size+self.head_size:])  # v (batch, seq_len, head_size)
        else:
            qkv = torch.split(qkv, self.cfg.vec_size, dim=-1)  # q,k,v are (batch, seq_len, vec_size)
        
        # transpose so that multiplications are done to each position, not each head
        # q,k,v are now (batch, n_heads, seq_len, head_size), (n_heads=1 for k,v if mqa_attn=True)
        # we use the -1 so that for mqa_attn, the n_heads can be inferred as 1 
        q,k,v = [mat.view(batch, seq_len, -1, self.head_size).transpose(1,2) for mat in qkv]

        if self.cfg.posn_embed_type == "rotary":
            q, k = net_utils.rotate_q_and_k(q, k, self.main_transformer.rotary_freqs) # type: ignore

        if self.cfg.flash:  # scaled dot product attention
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                                  dropout_p=self.cfg.dropout_attn if self.training else 0)
        else:
            if self.cfg.posn_embed_type == "relative":
                # can combine terms a) and c) in XL transformer paper since they both are multiplied by k
                # posn_vect_u (1, n_head, head_size, 1) @ k (batch, n_head, seq_len, head_size) -> (batch, n_head, seq_len, 1)
                attn_dots = (q + self.posn_vect_u) @ k.transpose(2,3)

                # (seq_len, vec_size) @ (vec_size, vec_size) -> (seq_len, vec_size)
                # W_{k,R} @ R_{i-j} in the paper
                rel_posn = self.posn_key_mat(self.main_transformer.relative_posn_embed[-seq_len:]) # type: ignore
                # reshape into (1, n_head, head_size, seq_len)
                # unsqueeze(0) <=> add batch dim <=> apply same "relative" position embeds to all samples in batch
                rel_posn = rel_posn.T.view(self.cfg.n_heads, self.head_size, seq_len).unsqueeze(0)
                # force running in float32 here to avoid numerical instability issues
                with torch.autocast(device_type=utils.get_device_type(), enabled=not self.cfg.relative_float32_attn): # type: ignore
                    if self.cfg.relative_float32_attn:
                        rel_posn = rel_posn.float()

                    # terms b) and d) in the paper
                    # (batch, n_head, seq_len, head_size) @ (1, n_head, head_size, seq_len) -> (batch, n_head, seq_len, seq_len)
                    query_posn = (q + self.posn_vect_v) @ rel_posn
                    # if full shape, take advantage of .take speed up
                    if query_posn.shape == self.main_transformer.shift_indices.shape:  # type:ignore
                        shifted_posn = torch.take(query_posn, self.main_transformer.shift_indices)  # type: ignore
                    else:  # about 50% slower
                        row_idx, col_idx = net_utils.get_variable_index_shifter(seq_len, device=query_posn.device)
                        shifted_posn = query_posn[..., row_idx, col_idx]

                    attn_dots += shifted_posn

            elif self.cfg.posn_embed_type == "rel_bias":
                rel_bias = self.main_transformer.rel_bias(self.main_transformer.rel_bias_indices[:seq_len, :seq_len]) # type:ignore
                rel_bias = rel_bias.permute([2, 0, 1]).unsqueeze(0)
                attn_dots = q @ k.transpose(2, 3) + rel_bias
            else:  # ie. vanilla transformer, assume position emebeddings were applied at the base of the model
                # q (batch, n_heads, seq_len, head_size) @ k.T (batch, n_heads, head_size, seq_len)
                attn_dots = q @ k.transpose(2,3)  # attn_dots is (batch, n_heads, seq_len, seq_len)

            # mask out the future
            attn_dots = attn_dots.masked_fill(self.main_transformer.causal_mask[..., :seq_len, :seq_len], -float("inf")) #type: ignore
            attn_scores = F.softmax(attn_dots / np.sqrt(self.head_size), dim=-1)
            attn_scores = self.attn_dropout(attn_scores)  # attn_scores is (batch, n_heads, seq_len, seq_len)

            # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, head_size) -> (batch, n_head, seq_len, vec_size)
            attn = attn_scores @ v

        # transpose first so that the reshape operates on the last 2 dimensions only, stacking heads
        out = self.out(attn.transpose(1,2).reshape(batch, seq_len, self.cfg.vec_size))  # (batch, seq_len, vec_size)
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
            x = x + self.mha(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            raise NotImplementedError(f"Layer norm posn '{self.layer_norm_posn}' not supported.")
        return x

class Transformer(nn.Module):
    def __init__(self, model_cfg: config_objects.ExperimentCfg, dset_cfg: config_objects.DatasetCfg, no_init=False):
        super().__init__()

        # for saving stuff purposes
        self.cfg = model_cfg
        self.dset_cfg = dset_cfg
        self.best_loss = float('inf')
        self.checkpointing_setup = False  # flag to make sure we only ever set up checkpointing once
        if not no_init:
            self.initialize_architecture()  # uses saved cfg to make the layers

    def initialize_architecture(self):
        assert self.cfg.posn_embed_type in ["base_sinusoid", "base_learnable", "relative", "none", "rel_bias", "rotary"]


        if self.cfg.posn_embed_type == "relative":  # register these only once to save memory
            # .flip to match decreasing order of pos embeddings, per XL transformer paper (see appendix B, defn of Q matrix)
            self.register_buffer("relative_posn_embed", 
                                 net_utils.sinusoid_matrix(self.cfg.block_size, self.cfg.vec_size).flip(0).contiguous())
            
            # for full sized inputs, pre-computing everything and doing a .take is fastest, so we have shift_indices
            # for different sized inputs, .take wouldn't work since it requires linear indices, so we use the method
            # of shift_col_indices and shift_row_indices, which is still faster than the original XL transformer
            # implementation, but allows indexing for variable length inputs
            self.register_buffer("shift_indices", 
                                 net_utils.get_index_shifter(self.cfg.block_size, self.cfg.batch_size, self.cfg.n_heads))
        elif self.cfg.posn_embed_type == "rel_bias":
            self.rel_bias = nn.Embedding(self.cfg.block_size, self.cfg.n_heads)
            self.register_buffer("rel_bias_indices", net_utils.get_rel_bias_indices(self.cfg.block_size,
                                                                                    self.cfg.rel_bias_max_posn,
                                                                                    self.cfg.rel_bias_num_buckets))
        elif self.cfg.posn_embed_type == "base_learnable":
            self.posn_embed = nn.Embedding(self.cfg.block_size, self.cfg.vec_size)
        elif self.cfg.posn_embed_type == "base_sinusoid":
            self.register_buffer("posn_embed", net_utils.sinusoid_matrix(self.cfg.block_size, self.cfg.vec_size))
        elif self.cfg.posn_embed_type == "rotary":
            freqs = net_utils.get_rotary_freqs(self.cfg.block_size, 10_000, self.cfg.rotary_dim)
            self.rotary_freqs = nn.Parameter(freqs, requires_grad=self.cfg.rotary_learnable_freqs)

        # lower left triangle (rows = correspond to given location, left/right within a row (cols) = where we are attending to)
        # register_buffer = untrainable parameter, but gets moved onto/off GPU as requested when doing model.to()
        self.register_buffer("causal_mask",
                            ~torch.tril(torch.ones(self.cfg.block_size, self.cfg.block_size, dtype=torch.bool))
                             .reshape(1,1, self.cfg.block_size, self.cfg.block_size))
        
        self.embed = nn.Embedding(self.dset_cfg.vocab_size, self.cfg.vec_size)

        self.blocks = nn.ModuleList([TransformerBlock(self.cfg, self) for _ in range(self.cfg.n_layer)])

        self.unembed = nn.Linear(self.cfg.vec_size, self.dset_cfg.vocab_size, bias=False)
        # gives the embed matrix a smaller initialization (nanoGPT approach), hopefully makes training more stable
        if not self.cfg.learnable_unembed:  # nn.Embedding gives it a standard normal, this gives it a uniform distrib
            self.embed.weight = self.unembed.weight  # with width ~4.5e-3, for a variance of 6.6e-6 (if vocab_size=50_304)

        # set up autocasting
        rprint(utils.get_device_type(), "deiece")
        if utils.get_device_type() == "cpu":
            self.fwd_ctx = nullcontext()
        else:
            self.fwd_ctx = torch.autocast(device_type=utils.get_device_type(), dtype=getattr(torch, self.cfg.dtype)) # type: ignore

        if self.cfg.checkpointing:
            self.add_activation_checkpointing()

        print(f"Num parameters: {self.num_params()/1_000_000} M")
        print(f"Approximate expected train vram usage: {self.expected_vram_size():.2f} GB")

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property  # so that it works through a .to
    def device(self):
        return self.embed.weight.device

    def expected_vram_size(self):  # returns size in approximate GiB during training
        # https://arxiv.org/pdf/2205.05198.pdf
        dtype_size = int(self.cfg.dtype[-2:]) // 8  # won't work with 128-bit precision or 8-bit precision
        mem_per_layer = self.cfg.block_size * self.cfg.batch_size * self.cfg.vec_size * \
            (34 + 5*self.cfg.n_heads*self.cfg.block_size / self.cfg.vec_size)*dtype_size/2 # since paper assumes float16
        mem_model = self.num_params() * (32//8) * 4  # *4 since Adam stores 2, gradients is 1, and model itself is 1
        return (mem_model + mem_per_layer*self.cfg.n_layer) / (1_024**3)

    def forward(self, x, targets=None):
        with self.fwd_ctx:  # autocast, technically this is slightly different than doing `with fwd_ctx: model(x);`
            _, seq_len = x.shape  # not strictly "correct" since we sort of cut sequences up so absolute position embeddings are wrong here
            x = self.embed(x)
            if self.cfg.posn_embed_type == "base_learnable":
                x = x + self.posn_embed(torch.arange(seq_len, device=self.device)).unsqueeze(0) # type: ignore
            elif self.cfg.posn_embed_type == "base_sinusoid":
                x = x + self.posn_embed[:seq_len].unsqueeze(0) # type: ignore

            for i, block in enumerate(self.blocks):
                x = block(x)  # (batch, seq_len, vec_size)
            
            out = self.unembed(x)  # (batch, seq_len, vec_size) @ (vec_size, vocab_size) 

            if targets is None:  # ie. inference mode, return logits so we can do temperature stuff
                return out   # F.softmax(out, dim=-1)
            else:  # flatten across batches and positions so that we can compare indices (in `targets`) to distributions (in `out`)
                loss = F.cross_entropy(out.reshape(-1, self.dset_cfg.vocab_size), targets.reshape(-1), label_smoothing=self.cfg.label_smoothing)
                return out, loss

    @torch.no_grad()  # batched sampling
    def generate(self, encoder, prompt: Union[str, List[int], torch.Tensor], temperature=None) -> List[str]:
        """Generates a sequence of text given a prompt. If prompt is a string, it will be encoded using the provided encoder.
        If prompt is a list of integers, it will be used as is. If prompt is a tensor, it will be used as is. If the tensor
        has 2 dimensions, it generates sequences for each row in the tensor. In all cases, generate tokens until 2*block_size
        or an EOS token is generated.
        
        Args:
            encoder: the encoder to use for encoding the prompt
            prompt: the prompt to start generating from
            temperature: the temperature to use for sampling, if None, will use the default temperature
        Returns:
            A list of strings, each string is a generated sequence of text."""
        self.eval()
        if temperature is None:
            temperature = self.cfg.default_temperature
            
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
            logits = self(tokens[:, -self.cfg.block_size:])  # probably should add something for models that have unlimited context
            if temperature > 0:
                logits[:, -1, :] /= temperature
                probs = F.softmax(logits, dim=-1)  # 0 to select batch, -1 to select last position in sequence

                if probs.isnan().any():
                    if logits[:, -1, :].isnan().any():
                        print("DETECTED NANs, ABORTING...")
                        for sentence in tokens:
                            finished_sentences.append(sentence)
                        break
                    else:
                        print("DETECTED NANs, DOING ARGMAX BACKUP...")
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs[:, -1, :], 1)
            else:  # argmax, temp 0
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            tokens = torch.cat((tokens, next_token), dim=1)  # concat onto seq_len dimension
            retain = list(range(tokens.shape[0]))
            for i, sentence in enumerate(tokens):
                if sentence[-1] == encoder.eos_token or sentence.shape[0] >= self.cfg.max_generation_len:
                    finished_sentences.append(sentence)
                    retain.remove(i)
                    print("done with", i, "shape was", sentence.shape, "retain is", retain)

            tokens = tokens[retain]
        return [encoder.decode(sent.detach().cpu().numpy()) for sent in finished_sentences]


    def get_optim(self): # optim, sched, scaler
        """Returns the optimizer, scheduler, and gradient scaler for the model. 
        Default scheduler is linear -> cosine -> constant. Optimizer can optionally be ZeroRedundancyOptimizer if using DDP."""
        if self.cfg.weight_decay > 0:
            grad_params = [p for p in self.parameters() if p.requires_grad]
            weight_decay_params = [p for p in grad_params if p.dim() >= 2]  # matrices should be weight decayed
            non_decay_params = [p for p in grad_params if p.dim() < 2]  # biases and Normalizer scalings should not be
            param_groups = [{"params": weight_decay_params, "weight_decay": self.cfg.weight_decay},
                            {"params": non_decay_params, "weight_decay": 0}]
        else:
            param_groups = self.parameters()  # to account for the old optimizer state dicts
        
        maybe_fused = dict(fused=True) if utils.get_device_type() == "cuda" else dict()

        if self.cfg.zero and self.cfg.ddp:
            optim = ZeroRedundancyOptimizer(param_groups,
                                            optimizer_class=getattr(torch.optim, self.cfg.optimizer_type),
                                            lr=self.cfg.lr_max,
                                            **maybe_fused)
        else:
            optim = getattr(torch.optim, self.cfg.optimizer_type)(param_groups, lr=self.cfg.lr_max, **maybe_fused)
        
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
        """Adds activation checkpointing to the model. This will checkpoint the model before each GELU and Normalizer, since
        those layers are cheap to compute and can save a bit of memory. This is only done once, so calling this function multiple
        times will have no effect."""
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

        # recursively iterate through modules and apply the add_checkpointing function, which will checkpoint the specified layers
        net_utils.traverse_modules(add_checkpointing, self)  # type: ignore
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

