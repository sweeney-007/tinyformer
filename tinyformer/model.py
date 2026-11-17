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
    n_experts      0 (dense MLP) | E (mixture-of-experts MLP, see MoE)
    mup            False (standard) | True (maximal update parametrization)

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
    # Mixture-of-experts MLP (n_experts = 0 keeps the ordinary dense MLP).
    n_experts: int = 0
    moe_top_k: int = 2            # experts each token is sent to
    moe_capacity_factor: float = 1.25  # slots per expert vs an even split; <= 0: never drop
    moe_aux_coef: float = 0.01    # weight of the load-balancing loss
    moe_z_coef: float = 1e-3      # weight of the router z-loss
    # Maximal update parametrization (muP). At n_embd == mup_base_width the
    # model is identical to the standard one; wider models rescale init,
    # learning rates and a few multipliers so the best settings stay put.
    mup: bool = False
    mup_base_width: int = 128
    mup_base_head_dim: Optional[int] = None  # default: mup_base_width // n_head

    def __post_init__(self):
        assert self.pos_encoding in ("learned", "sinusoidal", "rope", "none")
        assert self.norm in ("layernorm", "rmsnorm")
        assert self.norm_position in ("pre", "post")
        assert self.activation in ("gelu", "relu", "swiglu")
        assert self.n_embd % self.n_head == 0
        assert self.n_experts == 0 or 1 <= self.moe_top_k <= self.n_experts
        if self.mup_base_head_dim is None:
            self.mup_base_head_dim = self.mup_base_width // self.n_head

    @property
    def width_mult(self) -> float:
        """How many times wider than the base model (1.0 when muP is off)."""
        return self.n_embd / self.mup_base_width if self.mup else 1.0


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
        # Standard attention divides scores by sqrt(D). muP divides by D (so
        # scores do not grow as heads get wider), written as sqrt(D0) / D so
        # it equals the standard 1/sqrt(D) at the base head size D0.
        if cfg.mup:
            self.scale = math.sqrt(cfg.mup_base_head_dim) / self.head_dim
        else:
            self.scale = 1.0 / math.sqrt(self.head_dim)
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
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
        else:
            # Attention scores: how much each position "looks at" each other one.
            att = (q @ k.transpose(-2, -1)) * self.scale                # (B, H, T, T)
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


def load_balance_loss(probs: torch.Tensor, top_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch Transformer balancing loss: E * sum_i f_i * P_i.

    f_i: fraction of routing choices that went to expert i.
    P_i: average router probability for expert i.
    Equals 1.0 when routing is perfectly even and n_experts when every token
    goes to one expert with full confidence. f has no gradient (it counts);
    the gradient flows through P and pushes probability off busy experts.
    """
    f = F.one_hot(top_idx.reshape(-1), n_experts).float().mean(0)
    P = probs.mean(dim=0)
    return n_experts * (f * P).sum()


class MoE(nn.Module):
    """Mixture-of-experts MLP: a router sends each token to its top-k experts.

    Each expert is a small MLP. With hidden size (dense hidden) / k, a token
    uses about the same number of weights as in the dense MLP, while the
    layer stores n_experts / k times more of them.

    Steps for N tokens of size C:
      1. router logits (N, E) -> softmax -> keep the top-k experts per token,
         renormalize their weights to sum to 1 (Mixtral-style);
      2. place each (token, expert) choice in that expert's buffer of
         `capacity` slots; first choices get priority over second choices,
         and choices that do not fit are dropped (training only);
      3. every expert runs on its buffer in one batched matmul (E, cap, C);
      4. outputs are weighted and added back to their tokens.

    Two extra losses keep the router healthy: the load-balancing loss
    (spreads tokens across experts) and the z-loss (keeps router logits
    small). Both are stored in self.aux_loss for GPT.forward to add.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        E, C, k = cfg.n_experts, cfg.n_embd, cfg.moe_top_k
        dense_hidden = 4 * C if cfg.activation != "swiglu" else int(2 * 4 * C / 3)
        self.hidden = max(1, dense_hidden // k)
        self.activation = cfg.activation
        self.router = nn.Linear(C, E, bias=False)
        # Expert weights stacked so all experts run in one batched matmul.
        self.w_fc = nn.Parameter(torch.empty(E, C, self.hidden))
        if cfg.activation == "swiglu":
            self.w_gate = nn.Parameter(torch.empty(E, C, self.hidden))
        self.w_proj = nn.Parameter(torch.empty(E, self.hidden, C))
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)  # GPT re-scales w_proj for depth
        self.drop = nn.Dropout(cfg.dropout)
        # muP: the router's output size (E) does not grow with width, so it is
        # an output-like layer and its logits get the same 1/width_mult as lm_head.
        self.router_mult = 1.0 / cfg.width_mult
        self.aux_loss = torch.zeros(())
        self.stats = {}

    def capacity(self, n_tokens: int) -> int:
        cf = self.cfg.moe_capacity_factor
        if not self.training or cf <= 0:
            return n_tokens  # an expert can never get more than every token once
        return max(1, math.ceil(cf * n_tokens * self.cfg.moe_top_k / self.cfg.n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        E, k = self.cfg.n_experts, self.cfg.moe_top_k
        xf = x.reshape(-1, C)                                        # (N, C)
        N = xf.size(0)

        # Scale the input, not the logits (same result: the router is linear). A float multiply
        # right after a matmul trips a pattern-matching bug in torch 2.5's Inductor.
        logits = self.router(xf * self.router_mult)                  # (N, E)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_w, top_i = probs.topk(k, dim=-1)                         # (N, k)
        top_w = top_w / top_w.sum(-1, keepdim=True)

        # Choice-major order: all first choices, then all second choices, so
        # first choices claim expert slots first.
        e = top_i.t().reshape(-1)                                    # (k*N,) expert of each choice
        t = torch.arange(N, device=x.device).repeat(k)               # token of each choice
        w = top_w.t().reshape(-1)
        # Slot of each choice inside its expert = how many earlier choices
        # picked the same expert.
        onehot = F.one_hot(e, E)
        slot = (onehot.cumsum(0) - 1).gather(1, e[:, None]).squeeze(1)
        cap = self.capacity(N)
        keep = slot < cap
        # Dropped choices go to an extra "overflow" slot (index cap) and get
        # weight 0. This keeps every tensor shape fixed from step to step,
        # which torch.compile needs, instead of filtering with keep.
        slot = torch.where(keep, slot, torch.full_like(slot, cap))
        w = w * keep

        buf = xf.new_zeros(E, cap + 1, C)
        buf[e, slot] = xf[t]
        if self.activation == "swiglu":
            h = F.silu(buf @ self.w_gate) * (buf @ self.w_fc)
        elif self.activation == "relu":
            h = F.relu(buf @ self.w_fc)
        else:
            h = F.gelu(buf @ self.w_fc)
        out = h @ self.w_proj                                        # (E, cap + 1, C)
        y = xf.new_zeros(N, C)
        y.index_add_(0, t, (out[e, slot] * w[:, None].to(out.dtype)).to(y.dtype))

        z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        self.aux_loss = (self.cfg.moe_aux_coef * load_balance_loss(probs, top_i, E)
                         + self.cfg.moe_z_coef * z)
        if self.training:  # report what training saw (evaluation never drops)
            self.stats = {
                "load": onehot.detach().float().mean(0),
                "dropped": 1.0 - keep.float().mean(),  # tensor: no GPU sync
            }
        return self.drop(y.view(B, T, C))


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
        self.mlp = MoE(cfg) if cfg.n_experts > 0 else MLP(cfg)

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
            # measured 2.88 bits/char vs 1.98. The 2017
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
        for name, p in self.named_parameters():
            if name.endswith(("mlp.w_fc", "mlp.w_gate")):
                nn.init.normal_(p, mean=0.0, std=0.02)
            # GPT-2 trick: shrink the layers that write into the residual
            # stream, so the stream does not grow with depth.
            if name.endswith(("proj.weight", "w_proj")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # muP: hidden matrices (both sides grow with width) start with std
        # shrunk by sqrt(width_mult), i.e. std ~ 1/sqrt(fan_in).
        self.out_mult = 1.0 / cfg.width_mult  # muP readout multiplier on the logits
        if cfg.mup:
            with torch.no_grad():
                for name, p in self.named_parameters():
                    if self.param_kind(name, p) == "hidden":
                        p.mul_(1.0 / math.sqrt(cfg.width_mult))

    def param_kind(self, name: str, p: torch.Tensor) -> str:
        """muP role of a parameter.

        embedding: maps a fixed-size input (token / position) to width
        readout:   maps width to a fixed-size output (lm_head, MoE router)
        hidden:    width to width (attention, MLP and expert matrices)
        vector:    biases and norm gains
        """
        if p.dim() < 2:
            return "vector"
        if name.startswith(("tok_emb", "pos_emb")):
            return "embedding"
        if name.startswith("lm_head") or name.endswith("router.weight"):
            return "readout"
        return "hidden"

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

    def param_counts(self) -> dict:
        """total; non_embedding (excludes token/position tables and lm_head);
        active (non-embedding weights one token actually uses: k of E experts)."""
        cfg = self.cfg
        total = self.num_params()
        non_emb = active = 0.0
        for name, p in self.named_parameters():
            if name.startswith(("tok_emb", "pos_emb", "lm_head")):
                continue
            non_emb += p.numel()
            expert = ".mlp.w_" in name
            active += p.numel() * (cfg.moe_top_k / cfg.n_experts if expert else 1.0)
        return {"total": total, "non_embedding": int(non_emb), "active": int(active)}

    def flops_per_token(self) -> float:
        """Training FLOPs per token (forward + backward = 3x forward), Kaplan-style:
        6 x active weights, + the output layer (C x V), + attention scores
        (2 x layers x context x C per forward)."""
        cfg = self.cfg
        n = self.param_counts()["active"] + cfg.n_embd * cfg.vocab_size
        return 6 * n + 6 * cfg.n_layer * cfg.block_size * cfg.n_embd

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
        # lm_head is linear, so scaling its input equals scaling the logits (see MoE.forward).
        logits = self.lm_head(self.ln_f(x) * self.out_mult)         # (B, T, V)

        loss = None
        if targets is not None:
            # Next-token prediction: compare each position with the token after it.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            if self.cfg.n_experts and self.training:
                # MoE router losses join the training loss only; evaluation
                # reports pure cross-entropy so bits/char stay comparable.
                loss = loss + sum(b.mlp.aux_loss for b in self.blocks)
        return logits, loss

    def moe_stats(self) -> dict:
        """Expert load (fraction of choices per expert) and dropped fraction, per layer."""
        if not self.cfg.n_experts:
            return {}
        return {
            "load": [b.mlp.stats["load"].tolist() for b in self.blocks],
            "dropped": [float(b.mlp.stats["dropped"]) for b in self.blocks],
        }

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        """AdamW, with weight decay on weight matrices and embeddings only (not biases or norms).

        Each group carries an "lr_mult". The training loop sets
        lr = schedule(step) * lr_mult. Without muP every multiplier is 1.
        With muP, hidden matrices get 1 / width_mult: Adam moves every weight
        by about lr per step whatever its gradient size, and a wider layer
        sums more of those moves, so the step must shrink with width.
        """
        groups = {}
        for name, p in self.named_parameters():  # tied weights are listed once
            if not p.requires_grad:
                continue
            kind = self.param_kind(name, p)
            mult = 1.0 / self.cfg.width_mult if kind == "hidden" else 1.0
            wd = weight_decay if p.dim() >= 2 else 0.0
            g = groups.setdefault((mult, wd), {"params": [], "weight_decay": wd, "lr_mult": mult})
            g["params"].append(p)
        groups = list(groups.values())
        for g in groups:
            g["lr"] = lr * g["lr_mult"]
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
