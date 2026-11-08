"""A minimal decoder-only (GPT-style) transformer in raw PyTorch.

Only torch.nn building blocks (Linear, Embedding, Dropout) are used.
Attention, the causal mask, normalization, positional encodings, the MLP and
sampling are written out by hand so every step is visible.

Each design choice is a config option, so experiments can swap one part and
keep everything else the same:

    pos_encoding   learned | sinusoidal | rope | none
    norm           layernorm | rmsnorm
    norm_position  pre | post
    activation     gelu | relu | swiglu
    tie_weights    True | False

Shapes used in comments:
    B = batch size, T = sequence length, C = embedding size (d_model),
    H = number of heads, D = C // H (size of one head), V = vocab size
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256      # longest context the model can see
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True          # bias in Linear and LayerNorm layers
    pos_encoding: str = "learned"
    norm: str = "layernorm"
    norm_position: str = "pre"
    activation: str = "gelu"
    tie_weights: bool = True
    use_fused_attention: bool = False  # True -> F.scaled_dot_product_attention

    def __post_init__(self):
        assert self.pos_encoding in ("learned", "sinusoidal", "rope", "none")
        assert self.norm in ("layernorm", "rmsnorm")
        assert self.norm_position in ("pre", "post")
        assert self.activation in ("gelu", "relu", "swiglu")
        assert self.n_embd % self.n_head == 0


# ---------------------------------------------------------------------- #
# Normalization
# ---------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    """Subtract the mean, divide by the standard deviation, then scale and shift."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight
        return x + self.bias if self.bias is not None else x


class RMSNorm(nn.Module):
    """Like LayerNorm but only rescales (no mean subtraction, no bias). Used in Llama."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg: GPTConfig) -> nn.Module:
    return LayerNorm(cfg.n_embd, bias=cfg.bias) if cfg.norm == "layernorm" else RMSNorm(cfg.n_embd)


# ---------------------------------------------------------------------- #
# Positional encodings
# ---------------------------------------------------------------------- #
def sinusoidal_table(T: int, C: int) -> torch.Tensor:
    """Fixed sine/cosine waves of different frequencies (original Transformer paper)."""
    pos = torch.arange(T).unsqueeze(1)
    freq = torch.exp(torch.arange(0, C, 2) * (-math.log(10000.0) / C))
    table = torch.zeros(T, C)
    table[:, 0::2] = torch.sin(pos * freq)
    table[:, 1::2] = torch.cos(pos * freq)
    return table


def rope_tables(T: int, D: int, base: float = 10000.0):
    """Rotary embeddings: angles for rotating each pair of dimensions by position."""
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))  # (D/2,)
    angles = torch.outer(torch.arange(T).float(), inv_freq)          # (T, D/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x0, x1), (x2, x3), ... of x (B, H, T, D) by position-dependent angles.

    After rotation, the dot product q.k depends only on how far apart the two
    positions are, which is what attention needs.
    """
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------------- #
# Transformer layers
# ---------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position only sees earlier ones."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_fused = cfg.use_fused_attention
        self.dropout = cfg.dropout
        self.rope = cfg.pos_encoding == "rope"
        # One matrix computes queries, keys and values for all heads at once.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular mask: position t may attend to positions 0..t.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)
        if self.rope:
            cos, sin = rope_tables(cfg.block_size, self.head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                       # 3 x (B, T, C)
        # Split C into H heads of size D: (B, T, C) -> (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.rope:
            q = apply_rope(q, self.cos[:T], self.sin[:T])
            k = apply_rope(k, self.cos[:T], self.sin[:T])

        if self.use_fused:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Attention scores: how much each position "looks at" each other one.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))  # hide the future
            att = F.softmax(att, dim=-1)                                # rows sum to 1
            att = self.attn_drop(att)
            y = att @ v                                                 # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)               # join the heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network.

    gelu / relu: expand 4x, activation, project back.
    swiglu: silu(x W1) * (x W3), then project back (Llama-style). The hidden
            size is 2/3 of 4x so the parameter count matches the other two.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.activation = cfg.activation
        hidden = 4 * cfg.n_embd
        if cfg.activation == "swiglu":
            hidden = int(2 * hidden / 3)
            self.gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            h = F.silu(self.gate(x)) * self.fc(x)
        elif self.activation == "relu":
            h = F.relu(self.fc(x))
        else:
            h = F.gelu(self.fc(x))
        return self.drop(self.proj(h))


class Block(nn.Module):
    """One transformer layer.

    pre-norm  (GPT-2 and later):   x = x + attn(norm(x));  x = x + mlp(norm(x))
    post-norm (original 2017 paper): x = norm(x + attn(x)); x = norm(x + mlp(x))
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.pre = cfg.norm_position == "pre"
        self.ln1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.mlp(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token id -> vector
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # position -> vector
        elif cfg.pos_encoding == "sinusoidal":
            # Scaled to match the token embeddings (std 0.02). Unscaled, the
            # waves (size ~1) drown out the tokens and training stalls: we
            # measured 2.88 bits/char vs 1.98 (docs/EXPERIMENTS.md). The 2017
            # paper solves the same problem by multiplying embeddings by sqrt(C).
            table = sinusoidal_table(cfg.block_size, cfg.n_embd) * 0.02
            self.register_buffer("pos_table", table, persistent=False)
        # "rope" is applied inside attention; "none" relies on the causal mask alone.
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # Post-norm blocks already end with a norm.
        self.ln_f = make_norm(cfg) if cfg.norm_position == "pre" else nn.Identity()
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # The input embedding and the output layer share weights: fewer
            # parameters and usually slightly better results.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the layers that write into the residual stream,
        # so the stream does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """idx: (B, T) token ids. Returns logits (B, T, V) and the loss if targets are given."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} is longer than block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)                                       # (B, T, C)
        if self.cfg.pos_encoding == "learned":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif self.cfg.pos_encoding == "sinusoidal":
            x = x + self.pos_table[:T]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                         # (B, T, V)

        loss = None
        if targets is not None:
            # Next-token prediction: compare each position with the token after it.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        """AdamW, with weight decay on weight matrices and embeddings only (not biases or norms)."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_token: Optional[int] = None,
    ) -> torch.Tensor:
        """Append `max_new_tokens` sampled tokens to idx (B, T)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]                # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                               # only the last position
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)       # greedy
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    # Nucleus sampling: keep the smallest set of tokens whose
                    # probabilities add up to top_p.
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    remove = torch.cumsum(probs, dim=-1) - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and (next_id == stop_token).all():
                break
        return idx
