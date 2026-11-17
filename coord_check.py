"""Coordinate check: does activation size stay the same as the model gets wider?

The standard test from the muP paper (Yang et al., "Tensor Programs V").
For several widths, train a few Adam steps at one fixed learning rate and
record the typical size of each activation entry ("coordinate"), measured as
mean |x|. Under the standard parametrization some activations grow with
width, which is why the best learning rate shifts as models get wider.
Under muP the curves should be flat.

Runs on a laptop CPU in a few minutes:
    python coord_check.py --data data/pycode_bpe
Writes docs/results/coord_check.png and coord_check.json.
"""

import argparse
import json
import os

import torch

from tinyformer.model import GPT, GPTConfig
from tinyformer.utils import get_batch


def measure(cfg: GPTConfig, data, args, seed: int):
    """Mean |activation| at each step for: embeddings, every block's output, logits."""
    torch.manual_seed(seed)
    model = GPT(cfg)
    opt = model.configure_optimizer(args.lr, weight_decay=0.0)
    acts = {}
    hooks = [model.tok_emb.register_forward_hook(lambda m, i, o: acts.__setitem__("embedding", o))]
    for n, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(
            lambda m, i, o, n=n: acts.__setitem__(f"block {n + 1}", o)))
    g = torch.Generator().manual_seed(seed)
    history = []
    for step in range(args.steps + 1):
        torch.manual_seed(seed * 1000 + step)  # same batches for every width
        x, y = get_batch(data, cfg.block_size, args.batch_size, "cpu")
        logits, loss = model(x, y)
        acts["logits"] = logits
        history.append({k: v.detach().abs().mean().item() for k, v in acts.items()})
        if step < args.steps:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for h in hooks:
        h.remove()
    del g
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/pycode_bpe")
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--base-width", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-2, help="deliberately high, to make growth visible")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="docs/results")
    args = ap.parse_args()

    data = torch.load(os.path.join(args.data, "train.pt"))
    vocab = int(data.max()) + 1
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    results = {}
    for arch, n_experts in (("dense", 0), ("moe", 8)):
        for param in ("standard", "mup"):
            for width in args.widths:
                cfg = GPTConfig(
                    vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=width, dropout=0.0, bias=False,
                    pos_encoding="rope", norm="rmsnorm", activation="swiglu",
                    n_experts=n_experts, moe_capacity_factor=0,
                    mup=(param == "mup"), mup_base_width=args.base_width,
                )
                runs = [measure(cfg, data, args, s) for s in args.seeds]
                mean = [{k: sum(r[t][k] for r in runs) / len(runs) for k in runs[0][t]}
                        for t in range(args.steps + 1)]
                results[f"{arch}/{param}/{width}"] = mean
                last = mean[-1]
                print(f"{arch:5s} {param:8s} width {width:5d} | after {args.steps} steps: "
                      + "  ".join(f"{k} {v:.3g}" for k, v in last.items()))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "coord_check.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=1)
    plot(results, args)


def plot(results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["embedding", f"block {args.n_layer}", "logits"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    cmap = plt.get_cmap("viridis")
    for r, param in enumerate(("standard", "mup")):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            for t in range(args.steps + 1):
                color = cmap(t / max(1, args.steps))
                for arch, style in (("dense", "-"), ("moe", "--")):
                    ys = [results[f"{arch}/{param}/{w}"][t][key] for w in args.widths]
                    ax.plot(args.widths, ys, style, marker="o", ms=3, color=color,
                            label=f"step {t}" if arch == "dense" and r == 0 and c == 0 else None)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{'Standard' if param == 'standard' else 'µP'}: {key}", fontsize=10)
            if r == 1:
                ax.set_xlabel("width (n_embd)")
            if c == 0:
                ax.set_ylabel("mean |activation|")
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8, title="solid dense, dashed MoE", title_fontsize=8)
    fig.suptitle(f"Coordinate check: {args.n_layer} layers, Adam lr {args.lr}, "
                 f"mean of {len(args.seeds)} seeds (µP base width {args.base_width})", fontsize=11)
    fig.tight_layout()
    path = os.path.join(args.out, "coord_check.png")
    fig.savefig(path, dpi=130)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
