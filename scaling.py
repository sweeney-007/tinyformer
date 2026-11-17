"""Phase 2 (learning-rate transfer) and Phase 3 (scaling ladder): run lists and analysis.

    python scaling.py plan --phase lr          # list the runs and their estimated cost
    python scaling.py analyze --runs runs      # tables and charts from fetched results

Model family: 6 layers, heads of 64 dimensions (n_head = width / 64), context
256, batch 64 (16,384 tokens per step), RoPE + RMSNorm + SwiGLU, no biases,
no dropout. The only variables are width, learning rate, the
parametrization (standard or muP, base width 128) and the MLP (dense or
8-expert top-2 MoE with the same active weights per token).
"""

import argparse
import glob
import json
import math
import os

N_LAYER, BLOCK, BATCH, VOCAB = 6, 256, 64, 4096
TOKENS_PER_STEP = BATCH * BLOCK
BASE = (f"python train.py --data /data/code --n-layer {N_LAYER} --block-size {BLOCK} --batch-size {BATCH} "
        "--dropout 0 --pos-encoding rope --norm rmsnorm --activation swiglu --no-bias --weight-decay 0.1 "
        "--compile --fused-attention --peak-tflops 121 --no-ckpt --eval-iters 20 --final-eval-iters 100")
MOE = "--n-experts 8 --moe-top-k 2 --moe-capacity 1.25"
MUP = "--mup --mup-base-width 128"

LR_WIDTHS = [128, 256, 512, 1024]
LR_EXPONENTS = list(range(12, 5, -1))     # lr = 2^-12 ... 2^-6
LR_TOKENS = 16_000_000
LADDER_WIDTHS = [128, 192, 256, 384, 512]
LADDER_SEEDS = [1, 2, 3]
TOKENS_PER_PARAM = 20


def non_embedding_params(width: int) -> int:
    """Attention 4C^2 + SwiGLU MLP 3 x C x (8C/3) = 12 C^2 per layer (norm gains ignored)."""
    return 12 * N_LAYER * width * width


def flops_per_token(width: int) -> float:
    return 6 * (non_embedding_params(width) + width * VOCAB) + 6 * N_LAYER * BLOCK * width


def model_args(width: int, arch: str, param: str) -> str:
    a = f"--n-embd {width} --n-head {width // 64}"
    if arch == "moe":
        a += " " + MOE
    if param == "mup":
        a += " " + MUP
    return a


def jobs(phase: str, runs_root: str, lr_dense: float = 0.0, lr_moe: float = 0.0):
    """List of (command, out_dir, estimated_flops)."""
    out = []
    if phase == "lr":
        steps = math.ceil(LR_TOKENS / TOKENS_PER_STEP)
        for arch in ("dense", "moe"):
            for param in ("standard", "mup"):
                for w in LR_WIDTHS:
                    for k in LR_EXPONENTS:
                        d = f"{runs_root}/lr/{arch}_{param}_w{w}_lr{k}"
                        cmd = (f"{BASE} {model_args(w, arch, param)} --lr {2.0 ** -k} --max-iters {steps} "
                               f"--warmup-iters 100 --eval-interval 250 --seed 1 --out {d}")
                        out.append((cmd, d, flops_per_token(w) * LR_TOKENS * (1.6 if arch == "moe" else 1)))
    elif phase == "ladder":
        assert lr_dense > 0 and lr_moe > 0, "give the learning rates chosen in Phase 2"
        for arch, lr in (("dense", lr_dense), ("moe", lr_moe)):
            for w in LADDER_WIDTHS:
                tokens = TOKENS_PER_PARAM * non_embedding_params(w)
                steps = math.ceil(tokens / TOKENS_PER_STEP)
                for s in LADDER_SEEDS:
                    d = f"{runs_root}/ladder/{arch}_w{w}_s{s}"
                    cmd = (f"{BASE} {model_args(w, arch, 'mup')} --lr {lr} --max-iters {steps} "
                           f"--warmup-iters {max(100, steps // 50)} --eval-interval {max(250, steps // 10)} "
                           f"--seed {s} --out {d}")
                    out.append((cmd, d, flops_per_token(w) * tokens * (1.6 if arch == "moe" else 1)))
    else:
        raise ValueError(phase)
    return out


def balance(job_list, groups: int):
    """Split jobs into `groups` lists of similar total cost (largest first, to the lightest group)."""
    bins = [[0.0, []] for _ in range(groups)]
    for cmd, d, cost in sorted(job_list, key=lambda j: -j[2]):
        b = min(bins, key=lambda x: x[0])
        b[0] += cost
        b[1].append((cmd, d))
    return [b[1] for b in bins if b[1]]


def smoke_jobs(runs_root: str):
    single = []
    for w in LR_WIDTHS:
        for arch in ("dense", "moe"):
            d = f"{runs_root}/smoke/{arch}_w{w}"
            single.append((f"{BASE} {model_args(w, arch, 'mup')} --lr {2.0 ** -9} --max-iters 300 "
                           f"--warmup-iters 50 --eval-interval 150 --out {d}", d))
    d = f"{runs_root}/smoke/ddp2_dense_w512"
    ddp_cmd = (f"{BASE} {model_args(512, 'dense', 'mup')} --lr {2.0 ** -9} --max-iters 300 "
               f"--warmup-iters 50 --eval-interval 150 --out {d}").replace(
        "python train.py", "torchrun --standalone --nproc_per_node 2 train.py")
    return {"single": single, "ddp": [(ddp_cmd, d)]}


# ---------------------------------------------------------------------- #
# Analysis
# ---------------------------------------------------------------------- #
def load(runs_dir: str, phase: str):
    rows = []
    for path in glob.glob(os.path.join(runs_dir, phase, "*", "result.json")):
        with open(path) as f:
            r = json.load(f)
        r["name"] = os.path.basename(os.path.dirname(path))
        rows.append(r)
    return rows


def best_lr(points):
    """Best log2(lr) from (log2_lr, loss) points: a parabola through the lowest
    point and its two neighbours, so the optimum can fall between grid values."""
    pts = sorted((x, y) for x, y in points if math.isfinite(y))
    if not pts:
        return float("nan")
    i = min(range(len(pts)), key=lambda j: pts[j][1])
    if i == 0 or i == len(pts) - 1:
        return pts[i][0]  # at the edge of the grid: report the edge
    (x0, y0), (x1, y1), (x2, y2) = pts[i - 1 : i + 2]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    return -b / (2 * a) if a > 0 else x1


def fit_power_law(C, L):
    """Least squares fit of log L = log a - b log C. Returns (a, b)."""
    xs = [math.log(c) for c in C]
    ys = [math.log(v) for v in L]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = math.exp(my + b * mx)
    return a, b


def mean_std(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s


def analyze_lr(runs_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(runs_dir, "lr")
    curves = {}
    for r in rows:
        arch, param, w, k = r["name"].split("_")
        loss = r["final_val_bpc"] if not r.get("diverged") else float("inf")
        if not math.isfinite(loss) or loss > 10:
            loss = float("inf")
        curves.setdefault((arch, param), {}).setdefault(int(w[1:]), []).append((-int(k[2:]), loss))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    cmap = plt.get_cmap("viridis")
    lines = ["| MLP | Parametrization | " + " | ".join(f"width {w}" for w in LR_WIDTHS)
             + " | Drift 128→1024 |", "|---|---|" + "---|" * (len(LR_WIDTHS) + 1)]
    summary = {}
    for r_, arch in enumerate(("dense", "moe")):
        for c_, param in enumerate(("standard", "mup")):
            ax = axes[r_, c_]
            best = {}
            for i, w in enumerate(LR_WIDTHS):
                pts = sorted(curves.get((arch, param), {}).get(w, []))
                if not pts:
                    continue
                color = cmap(i / (len(LR_WIDTHS) - 1))
                ax.plot([p[0] for p in pts], [p[1] if math.isfinite(p[1]) else float("nan") for p in pts],
                        marker="o", ms=4, color=color, label=f"width {w}")
                best[w] = best_lr(pts)
                fin = [p for p in pts if math.isfinite(p[1])]
                if fin:
                    m = min(fin, key=lambda p: p[1])
                    ax.plot([m[0]], [m[1]], marker="*", ms=12, color=color, mec="black", mew=0.5)
            ax.set_title(f"{'Dense' if arch == 'dense' else 'MoE (8 experts, top-2)'}, "
                         f"{'standard' if param == 'standard' else 'µP'}", fontsize=10)
            ax.set_xlabel("log2(learning rate)")
            if c_ == 0:
                ax.set_ylabel("validation bits per character")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            drift = best.get(LR_WIDTHS[-1], float("nan")) - best.get(LR_WIDTHS[0], float("nan"))
            summary[f"{arch}/{param}"] = {"best_log2_lr": best, "drift": drift}
            lines.append(f"| {arch} | {'µP' if param == 'mup' else 'standard'} | "
                         + " | ".join(f"2^{best[w]:.2f}" if w in best else "—" for w in LR_WIDTHS)
                         + f" | {drift:+.2f} |")
    ymins = [p[1] for d in curves.values() for v in d.values() for p in v if math.isfinite(p[1])]
    if ymins:
        lo = min(ymins)
        axes[0, 0].set_ylim(lo - 0.02, lo + 0.6)
    fig.suptitle(f"Learning-rate transfer: {LR_TOKENS / 1e6:.0f}M tokens per run, stars = best per width", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lr_transfer.png"), dpi=130)

    table = ["| MLP | Parametrization | Width | " + " | ".join(f"2^-{k}" for k in LR_EXPONENTS) + " |",
             "|---|---|---|" + "---|" * len(LR_EXPONENTS)]
    for (arch, param), by_w in sorted(curves.items()):
        for w in LR_WIDTHS:
            d = dict(by_w.get(w, []))
            table.append(f"| {arch} | {param} | {w} | " + " | ".join(
                (f"{d[-k]:.3f}" if math.isfinite(d.get(-k, float('nan'))) else ("diverged" if -k in d else "—"))
                for k in LR_EXPONENTS) + " |")
    md = ["# Phase 2: learning-rate transfer", "",
          "Best learning rate per width (log2, parabola through the best grid point and its neighbours). "
          "Drift = how far the best learning rate moves from width 128 to 1024, in powers of 2.", ""] + lines + [
          "", "Validation bits per character, every run:", ""] + table
    with open(os.path.join(out_dir, "lr_transfer.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "lr_transfer.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("\n".join(lines))
    return summary


def analyze_ladder(runs_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(runs_dir, "ladder")
    by = {}
    for r in rows:
        arch, w, s = r["name"].split("_")
        by.setdefault(arch, {}).setdefault(int(w[1:]), []).append(r)
    stats = {}
    for arch, d in by.items():
        for w, rs in sorted(d.items()):
            m, s = mean_std([r["final_val_bpc"] for r in rs])
            stats[(arch, w)] = {
                "bpc": m, "std": s, "n": len(rs), "flops": rs[0]["train_flops"], "tokens": rs[0]["tokens"],
                "params_active": rs[0]["params_active"], "params": rs[0]["params"],
                "tok_s": sum(r["tokens_per_s"] for r in rs) / len(rs),
                "mfu": sum((r["mfu"] or 0) for r in rs) / len(rs),
                "hours": sum(r["seconds"] for r in rs) / len(rs) / 3600,
                "max_load": max(max(max(layer) for layer in r["moe"]["load"]) for r in rs) if arch == "moe" else None,
                "dropped": sum(sum(r["moe"]["dropped"]) / len(r["moe"]["dropped"]) for r in rs) / len(rs)
                if arch == "moe" else None,
            }
    fits = {}
    for arch in by:
        ws = sorted(w for a, w in stats if a == arch)
        fits[arch] = fit_power_law([stats[(arch, w)]["flops"] for w in ws], [stats[(arch, w)]["bpc"] for w in ws])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"dense": "#3b6fb6", "moe": "#d1762b"}
    for arch in ("dense", "moe"):
        ws = sorted(w for a, w in stats if a == arch)
        if not ws:
            continue
        C = [stats[(arch, w)]["flops"] for w in ws]
        L = [stats[(arch, w)]["bpc"] for w in ws]
        E = [stats[(arch, w)]["std"] for w in ws]
        a, b = fits[arch]
        ax.errorbar(C, L, yerr=E, fmt="o", color=colors[arch], capsize=3,
                    label=f"{'Dense' if arch == 'dense' else 'MoE 8×top-2'}: bpc ≈ {a:.2f}·C^-{b:.3f}")
        cs = [min(C) * (max(C) / min(C)) ** (i / 50) for i in range(51)]
        ax.plot(cs, [a * c ** -b for c in cs], "-", color=colors[arch], alpha=0.7)
        for w, c, l in zip(ws, C, L):
            if arch == "dense":
                ax.annotate(f"w{w}", (c, l), textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training compute (FLOPs)")
    ax.set_ylabel("validation bits per character")
    ax.set_title("Loss vs compute, µP, ~20 tokens per parameter, 3 seeds", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    mult_rows = []
    if "dense" in fits and "moe" in fits:
        a_d, b_d = fits["dense"]
        ws = sorted(w for a, w in stats if a == "moe" and ("dense", w) in stats)
        mults = []
        for w in ws:
            sm = stats[("moe", w)]
            c_needed = (sm["bpc"] / a_d) ** (-1 / b_d)
            mult = c_needed / sm["flops"]
            wall = mult * stats[("moe", w)]["tok_s"] / stats[("dense", w)]["tok_s"]
            mults.append((w, mult, wall))
            mult_rows.append((w, mult, wall))
        ax2.plot([m[0] for m in mults], [m[1] for m in mults], "o-", color=colors["moe"], label="per FLOP")
        ax2.plot([m[0] for m in mults], [m[2] for m in mults], "s--", color="#666", label="per GPU-second (measured speed)")
        ax2.axhline(1.0, color="black", lw=0.8)
        ax2.set_xscale("log", base=2)
        ax2.set_xlabel("width (n_embd)")
        ax2.set_ylabel("dense compute needed / MoE compute")
        ax2.set_title("MoE compute multiplier (>1: MoE is more efficient)", fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scaling_ladder.png"), dpi=130)

    lines = ["| MLP | Width | Active params | Total params | Tokens | FLOPs | Val bpc (mean ± sd) | tok/s | MFU | "
             "Busiest expert | Dropped |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (arch, w), st in sorted(stats.items()):
        lines.append(
            f"| {arch} | {w} | {st['params_active'] / 1e6:.1f}M | {st['params'] / 1e6:.1f}M | {st['tokens'] / 1e6:.0f}M | "
            f"{st['flops']:.2e} | {st['bpc']:.4f} ± {st['std']:.4f} | {st['tok_s']:,.0f} | {st['mfu']:.1%} | "
            + (f"{st['max_load']:.3f} (even 0.125) | {st['dropped']:.2%} |" if arch == "moe" else "— | — |"))
    md = ["# Phase 3: scaling ladder", "", *lines, "",
          "Power-law fits (bits/char = a · C^-b, C = training FLOPs):", ""]
    for arch, (a, b) in fits.items():
        md.append(f"- {arch}: a = {a:.3f}, b = {b:.4f}")
    if mult_rows:
        md += ["", "| Width | MoE compute multiplier (per FLOP) | Per GPU-second |", "|---|---|---|"]
        md += [f"| {w} | {m:.2f}× | {wl:.2f}× |" for w, m, wl in mult_rows]
    with open(os.path.join(out_dir, "scaling_ladder.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return stats, fits, mult_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--phase", default="lr")
    p.add_argument("--lr-dense", type=float, default=1e-3)
    p.add_argument("--lr-moe", type=float, default=1e-3)
    p.add_argument("--tflops", type=float, default=30.0, help="assumed achieved TFLOPS per GPU")
    p = sub.add_parser("analyze")
    p.add_argument("--runs", default="runs")
    p.add_argument("--phase", choices=["lr", "ladder"], default="lr")
    p.add_argument("--out", default="docs/results")
    args = ap.parse_args()
    if args.cmd == "plan":
        js = jobs(args.phase, "/runs", args.lr_dense, args.lr_moe)
        total = sum(j[2] for j in js)
        print(f"{len(js)} runs, {total:.2e} FLOPs (MoE weighted 1.6x), "
              f"~{total / (args.tflops * 1e12) / 3600:.1f} GPU-hours at {args.tflops} TFLOPS")
    else:
        os.makedirs(args.out, exist_ok=True)
        (analyze_lr if args.phase == "lr" else analyze_ladder)(args.runs, args.out)


if __name__ == "__main__":
    main()
