"""Tests for the mixture-of-experts MLP and the muP parametrization."""

import math

import pytest
import torch

from tinyformer.model import GPT, MLP, MoE, GPTConfig, load_balance_loss

CFG = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


# ---------------------------------------------------------------------- #
# Mixture of experts
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_one_expert_equals_the_dense_mlp(activation):
    """With 1 expert, top-1 routing and no dropping, MoE is just the dense MLP."""
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG, bias=False, activation=activation, n_experts=1, moe_top_k=1,
                    moe_capacity_factor=0)
    moe, dense = MoE(cfg), MLP(cfg)
    with torch.no_grad():
        moe.w_fc.copy_(dense.fc.weight.t()[None])
        moe.w_proj.copy_(dense.proj.weight.t()[None])
        if activation == "swiglu":
            moe.w_gate.copy_(dense.gate.weight.t()[None])
    x = torch.randn(3, 16, 32)
    assert torch.allclose(moe(x), dense(x), atol=1e-5)


def test_load_balance_loss_is_one_when_even_and_E_when_collapsed():
    E, N = 4, 8
    even_probs = torch.full((N, E), 1 / E)
    even_idx = torch.arange(N).remainder(E)[:, None]           # each expert gets 2 tokens
    assert load_balance_loss(even_probs, even_idx, E).item() == pytest.approx(1.0)
    collapsed_probs = torch.zeros(N, E)
    collapsed_probs[:, 0] = 1.0
    collapsed_idx = torch.zeros(N, 1, dtype=torch.long)
    assert load_balance_loss(collapsed_probs, collapsed_idx, E).item() == pytest.approx(E)


def test_capacity_drops_overflow_tokens_in_training_only():
    torch.manual_seed(0)
    cfg = GPTConfig(**CFG, n_experts=4, moe_top_k=1, moe_capacity_factor=1.0)
    moe = MoE(cfg)
    with torch.no_grad():                   # force every token to expert 0
        moe.router.weight.zero_()
    x = torch.randn(1, 16, 32)
    # Zero router weights tie all experts; topk then picks expert 0 for every token.
    moe.train()
    y = moe(x)
    cap = math.ceil(1.0 * 16 * 1 / 4)       # 4 slots
    assert moe.stats["dropped"].item() == pytest.approx(1 - cap / 16)
    assert torch.count_nonzero(y[0, cap:].abs().sum(-1)) == 0   # dropped tokens output 0
    assert torch.count_nonzero(y[0, :cap].abs().sum(-1)) == cap
    moe.eval()
    y_eval = moe(x)
    assert torch.count_nonzero(y_eval[0].abs().sum(-1)) == 16  # evaluation drops nothing


def test_moe_model_is_causal_and_router_learns():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**CFG, n_experts=4, moe_top_k=2))
    x = torch.randint(0, 64, (2, 16))
    model.train()
    _, loss = model(x, x)
    loss.backward()
    router = model.blocks[0].mlp.router.weight
    assert router.grad is not None and router.grad.abs().sum() > 0
    model.eval()
    x2 = x.clone()
    x2[0, 10] = (x[0, 10] + 1) % 64
    a, _ = model(x)
    b, _ = model(x2)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)


def test_moe_training_loss_includes_router_losses_and_eval_does_not():
    torch.manual_seed(0)
    # No dropping, so train and eval mode run the same experts.
    model = GPT(GPTConfig(**CFG, n_experts=4, moe_capacity_factor=0))
    x = torch.randint(0, 64, (2, 16))
    model.train()
    _, train_loss = model(x, x)
    aux = sum(b.mlp.aux_loss for b in model.blocks)
    model.eval()
    _, eval_loss = model(x, x)
    assert aux.item() > 0
    assert train_loss.item() == pytest.approx(eval_loss.item() + aux.item(), rel=1e-5)


def test_moe_active_parameters_match_dense():
    """Each token uses k experts of hidden size H/k: same weights per token as dense."""
    cfg = GPTConfig(**CFG, bias=False, n_experts=8, moe_top_k=2)
    moe, dense = MoE(cfg), MLP(cfg)
    active = cfg.moe_top_k * (moe.w_fc[0].numel() + moe.w_proj[0].numel())
    assert active == dense.fc.weight.numel() + dense.proj.weight.numel()


# ---------------------------------------------------------------------- #
# muP
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("n_experts", [0, 4])
def test_mup_at_base_width_equals_the_standard_model(n_experts):
    x = torch.randint(0, 64, (2, 16))
    outs = []
    for mup in (False, True):
        torch.manual_seed(0)
        model = GPT(GPTConfig(**CFG, n_experts=n_experts, mup=mup, mup_base_width=32)).eval()
        outs.append(model(x)[0])
        opt = model.configure_optimizer(1e-3, 0.1)
        assert all(g["lr_mult"] == 1.0 for g in opt.param_groups)
    assert torch.allclose(outs[0], outs[1])


def test_mup_scales_hidden_init_and_learning_rate_with_width():
    base, wide = 32, 128                    # width_mult = 4
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**CFG, "n_embd": wide}, n_experts=4, mup=True, mup_base_width=base))
    m = wide / base
    qkv = model.blocks[0].attn.qkv.weight
    assert qkv.std().item() == pytest.approx(0.02 / math.sqrt(m), rel=0.05)
    assert model.tok_emb.weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert model.out_mult == pytest.approx(1 / m)
    opt = model.configure_optimizer(1e-3, 0.1)
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    assert lr_of[id(qkv)] == pytest.approx(1e-3 / m)
    assert lr_of[id(model.blocks[0].mlp.w_fc)] == pytest.approx(1e-3 / m)
    assert lr_of[id(model.blocks[0].mlp.router.weight)] == pytest.approx(1e-3)
    assert lr_of[id(model.tok_emb.weight)] == pytest.approx(1e-3)
    n_listed = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_listed == model.num_params()   # every parameter exactly once


def test_mup_attention_scale_is_one_over_head_dim():
    cfg = GPTConfig(**{**CFG, "n_embd": 128}, mup=True, mup_base_width=32)
    attn = GPT(cfg).blocks[0].attn
    D, D0 = 64, 16
    assert attn.scale == pytest.approx(math.sqrt(D0) / D)


def test_mup_manual_attention_matches_fused_kernel():
    torch.manual_seed(0)
    cfg = dict(**{**CFG, "n_embd": 64}, mup=True, mup_base_width=32)
    manual = GPT(GPTConfig(**cfg)).eval()
    fused = GPT(GPTConfig(**cfg, use_fused_attention=True)).eval()
    fused.load_state_dict(manual.state_dict())
    x = torch.randint(0, 64, (2, 16))
    assert torch.allclose(manual(x)[0], fused(x)[0], atol=1e-5)
