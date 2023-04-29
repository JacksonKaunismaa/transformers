import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, vec_size):
        super().__init__()
        self.w_1 = nn.Linear(vec_size, vec_size*4)
        self.act_func1 = nn.GELU()
        self.w_2 = nn.Linear(vec_size*4, vec_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x -> (batch, seq_len, vec_size)
        return self.w_2(self.act_func1(self.w_1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, vec_size, n_heads, block_size):
        super().__init__()
        assert vec_size % n_heads == 0

        self.vec_size = vec_size
        self.n_heads = n_heads
        self.head_size = vec_size//n_heads

        self.qkv = nn.Linear(self.head_size, 3*self.head_size, bias=False)
        self.out = nn.Linear(self.vec_size, self.vec_size, bias=False)

        self.attn_dropout = nn.Dropout(p=0.1)
        self.out_dropout = nn.Dropout(p=0.1)

        # lower left triangle (rows = correspond to given location, left/right within a row (cols) = where we are attending to)
        # register_buffer = untrainable parameter, but gets moved onto/off GPU as requested when doing model.to()
        self.register_buffer("causal_mask",
                             torch.tril(torch.ones(block_size, block_size)).reshape(1,1, block_size, block_size))


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
    def __init__(self, vec_size, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(vec_size)
        self.mha = MultiHeadAttention(vec_size, n_heads, block_size)
        self.ln2 = nn.LayerNorm(vec_size)
        self.mlp = MLPBlock(vec_size)
        self.embed_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x + self.ln1(self.mha(x))
        x = x + self.ln2(self.mlp(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_layer, vec_size, n_heads, block_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, vec_size)
        self.posn_embed = nn.Embedding(block_size, vec_size)  # learnable position embeddings rather than sin waves

        self.blocks = nn.ModuleList([TransformerBlock(vec_size, n_heads, block_size) for _ in range(n_layer)])

        self.unembed = nn.Linear(vec_size, vocab_size)

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
