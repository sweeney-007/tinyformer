"""Tests for the memory-mapped data path, FLOPs counting and the scaling-analysis helpers."""

import pytest
import torch

np = pytest.importorskip("numpy")  # the memmap path needs numpy; the core model does not

import scaling
from tinyformer.model import GPT, GPTConfig
from tinyformer.utils import get_batch, load_split


def test_memmap_batches_match_tensor_batches(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    tokens.tofile(tmp_path / "train.bin")
    mm = load_split(str(tmp_path), "train")
    assert isinstance(mm, np.memmap) and len(mm) == 1000
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    x1, y1 = get_batch(mm, 16, 4, "cpu", generator=g1)
    x2, y2 = get_batch(torch.from_numpy(tokens.astype(np.int64)), 16, 4, "cpu", generator=g2)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.equal(y1[:, :-1], x1[:, 1:])  # targets are inputs shifted by one


def test_flops_formula_matches_the_model():
    """scaling.py's closed-form estimate agrees with counting the real model."""
    w = 256
    cfg = GPTConfig(vocab_size=scaling.VOCAB, block_size=scaling.BLOCK, n_layer=scaling.N_LAYER,
                    n_head=w // 64, n_embd=w, bias=False, activation="swiglu", pos_encoding="rope",
                    norm="rmsnorm")
    model = GPT(cfg)
    # The formula ignores norm gains and rounds the SwiGLU hidden size.
    assert model.param_counts()["non_embedding"] == pytest.approx(scaling.non_embedding_params(w), rel=0.01)
    assert model.flops_per_token() == pytest.approx(scaling.flops_per_token(w), rel=0.01)
    moe = GPT(GPTConfig(**{**cfg.__dict__, "n_experts": 8, "moe_top_k": 2}))
    assert moe.param_counts()["active"] == pytest.approx(model.param_counts()["active"], rel=0.01)
    assert moe.param_counts()["total"] > 3 * model.param_counts()["non_embedding"]


def test_best_lr_interpolates_between_grid_points():
    pts = [(-10, 2.0), (-9, 1.8), (-8, 1.9), (-7, 2.5)]
    assert -9 < scaling.best_lr(pts) < -8.5
    assert scaling.best_lr([(-10, 1.0), (-9, 2.0)]) == -10  # edge of the grid


def test_power_law_fit_recovers_exponent():
    C = [1e14, 1e15, 1e16, 1e17]
    a, b = scaling.fit_power_law(C, [3.0 * c ** -0.07 for c in C])
    assert a == pytest.approx(3.0) and b == pytest.approx(0.07)
