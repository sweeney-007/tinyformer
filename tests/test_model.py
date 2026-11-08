import torch

from tinyformer.model import GPT, GPTConfig

CFG = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_output_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**CFG))
    x = torch.randint(0, 64, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 64)
    # An untrained model should be close to guessing uniformly: loss ~ ln(64).
    assert abs(loss.item() - torch.log(torch.tensor(64.0)).item()) < 0.5


def test_causal_mask_hides_the_future():
    """Changing token t must not change the predictions for positions before t."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(**CFG)).eval()
    x = torch.randint(0, 64, (1, 16))
    x2 = x.clone()
    x2[0, 10] = (x[0, 10] + 1) % 64
    a, _ = model(x)
    b, _ = model(x2)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_manual_attention_matches_fused_kernel():
    torch.manual_seed(0)
    manual = GPT(GPTConfig(**CFG)).eval()
    fused = GPT(GPTConfig(**CFG, use_fused_attention=True)).eval()
    fused.load_state_dict(manual.state_dict())
    x = torch.randint(0, 64, (2, 16))
    assert torch.allclose(manual(x)[0], fused(x)[0], atol=1e-5)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**CFG))
    opt = model.configure_optimizer(lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, 64, (4, 17))
    inp, tgt = x[:, :-1], x[:, 1:]
    for _ in range(200):
        _, loss = model(inp, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_weights_are_tied():
    model = GPT(GPTConfig(**CFG))
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_generate_lengths_and_sampling_options():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**CFG))
    start = torch.zeros((2, 1), dtype=torch.long)
    for kwargs in [dict(temperature=0), dict(top_k=5), dict(top_p=0.9), dict(top_k=5, top_p=0.5)]:
        out = model.generate(start, max_new_tokens=30, **kwargs)  # longer than block_size
        assert out.shape == (2, 31)
        assert out.min() >= 0 and out.max() < 64


import pytest

VARIANTS = [
    dict(pos_encoding="rope"), dict(pos_encoding="sinusoidal"), dict(pos_encoding="none"),
    dict(norm="rmsnorm"), dict(norm_position="post"), dict(activation="relu"),
    dict(activation="swiglu"), dict(tie_weights=False),
]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(map(str, v.values())))
def test_every_variant_is_causal_and_trains(variant):
    torch.manual_seed(0)
    model = GPT(GPTConfig(**CFG, **variant))
    x = torch.randint(0, 64, (1, 16))
    x2 = x.clone()
    x2[0, 10] = (x[0, 10] + 1) % 64
    model.eval()
    assert torch.allclose(model(x)[0][0, :10], model(x2)[0][0, :10], atol=1e-5)
    model.train()
    opt = model.configure_optimizer(lr=3e-3, weight_decay=0.0)
    batch = torch.randint(0, 64, (4, 17))
    first = None
    for _ in range(50):
        _, loss = model(batch[:, :-1], batch[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7


def test_rope_depends_only_on_relative_position():
    from tinyformer.model import apply_rope, rope_tables

    cos, sin = rope_tables(32, 8)
    q, k = torch.randn(8), torch.randn(8)

    def score(i, j):
        qi = apply_rope(q.view(1, 1, 1, 8), cos[i:i + 1], sin[i:i + 1])
        kj = apply_rope(k.view(1, 1, 1, 8), cos[j:j + 1], sin[j:j + 1])
        return (qi * kj).sum()

    assert torch.allclose(score(5, 2), score(20, 17), atol=1e-5)


def test_handwritten_norms_match_pytorch():
    from tinyformer.model import LayerNorm

    x = torch.randn(4, 32)
    assert torch.allclose(LayerNorm(32)(x), torch.nn.LayerNorm(32)(x), atol=1e-5)
