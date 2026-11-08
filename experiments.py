"""Run controlled experiments: change one thing, keep everything else equal.

Suites:
    tokenizer     same model, different prepared datasets (e.g. BPE vs char)
    arch          same data, one architecture choice changed per run

Examples:
    python experiments.py --suite tokenizer --data data/code_bpe data/code_char --iters 1000
    python experiments.py --suite arch --data data/code_bpe --iters 1000 --seeds 1 2 3
    python experiments.py --suite arch --data data/code_bpe --only baseline rope swiglu

Anything after `--` is passed straight to train.py, e.g. `-- --preset small --device cuda`.

Each run is saved in runs/<suite>/<variant>_s<seed>/. Finished runs are
skipped, so an interrupted suite can be restarted. Results are written to
runs/<suite>/results.md (a Markdown table) and curves.png (if matplotlib is installed).
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys

# One change per variant, compared against "baseline" (GPT-2 style defaults).
ARCH_VARIANTS = {
    "baseline":   [],
    "rope":       ["--pos-encoding", "rope"],
    "sinusoidal": ["--pos-encoding", "sinusoidal"],
    "no-pos":     ["--pos-encoding", "none"],
    "rmsnorm":    ["--norm", "rmsnorm"],
    "post-norm":  ["--norm-position", "post"],
    "relu":       ["--activation", "relu"],
    "swiglu":     ["--activation", "swiglu"],
    "untied":     ["--no-tie-weights"],
}


def mean_std(xs):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s


def read_curve(run_dir):
    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    return [int(r["iter"]) for r in rows], [float(r["val_bpc"]) for r in rows]


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["tokenizer", "arch"], required=True)
    ap.add_argument("--data", nargs="+", required=True, help="prepared data folder(s)")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--only", nargs="+", help="arch suite: run only these variants")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)

    if args.suite == "tokenizer":
        variants = {os.path.basename(os.path.normpath(d)): (d, []) for d in args.data}
    else:
        names = args.only or list(ARCH_VARIANTS)
        variants = {n: (args.data[0], ARCH_VARIANTS[n]) for n in names}

    suite_dir = os.path.join(args.runs_dir, args.suite)
    results = {}
    for name, (data, flags) in variants.items():
        for seed in args.seeds:
            run_dir = os.path.join(suite_dir, f"{name}_s{seed}")
            result_path = os.path.join(run_dir, "result.json")
            if not os.path.exists(result_path):
                cmd = [sys.executable, "-u", "train.py", "--data", data, "--out", run_dir,
                       "--max-iters", str(args.iters), "--seed", str(seed)] + flags + passthrough
                print(f"\n=== {name} (seed {seed}) ===\n{' '.join(cmd)}", flush=True)
                subprocess.run(cmd, check=True)
            else:
                print(f"=== {name} (seed {seed}): already done, skipping")
            with open(result_path) as f:
                results.setdefault(name, []).append((run_dir, json.load(f)))

    # ---- Results table ------------------------------------------------- #
    base_name = "baseline" if "baseline" in results else next(iter(results))
    base_bpc = mean_std([r["best_val_bpc"] for _, r in results[base_name]])[0]
    lines = [
        f"# {args.suite} experiment",
        "",
        f"{args.iters} training steps, seeds {args.seeds}. Lower bits/char is better.",
        f"Δ is compared with `{base_name}`.",
        "",
        "| variant | val bits/char | Δ | params | tokens/char | time (s) |",
        "|---|---|---|---|---|---|",
    ]
    rows = sorted(results.items(), key=lambda kv: mean_std([r["best_val_bpc"] for _, r in kv[1]])[0])
    for name, runs in rows:
        m, s = mean_std([r["best_val_bpc"] for _, r in runs])
        t = sum(r["seconds"] for _, r in runs) / len(runs)
        params = runs[0][1]["params"]
        meta_path = os.path.join(variants[name][0], "meta.json")
        tpc = ""
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            tpc = f"{meta['val_tokens'] / meta['val_chars']:.3f}"
        pm = f" ± {s:.3f}" if len(runs) > 1 else ""
        lines.append(f"| {name} | {m:.3f}{pm} | {m - base_bpc:+.3f} | {params / 1e6:.2f}M | {tpc} | {t:.0f} |")
    table = "\n".join(lines) + "\n"
    with open(os.path.join(suite_dir, "results.md"), "w") as f:
        f.write(table)
    print("\n" + table)

    # ---- Learning curves (optional) ------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to also get curves.png")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, runs in rows:
        its, bpcs = read_curve(runs[0][0])
        ax.plot(its[1:], bpcs[1:], label=name)  # skip step 0 (untrained)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation bits per character")
    ax.set_title(f"{args.suite} experiment (seed {args.seeds[0]})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(suite_dir, "curves.png"), dpi=120)
    print(f"Saved {os.path.join(suite_dir, 'curves.png')}")


if __name__ == "__main__":
    main()
