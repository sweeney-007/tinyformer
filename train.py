"""Train the transformer on data made by prepare.py.

Examples:
    # Quick CPU run on an iMac (a few minutes):
    python train.py --data data/shakespeare --preset tiny

    # Bigger model on a free Colab/Kaggle GPU:
    python train.py --data data/shakespeare --preset small --max-iters 5000

Saves the best checkpoint (lowest validation loss) to <out>/ckpt.pt and a
loss log to <out>/log.csv.
"""

import argparse
import csv
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from tinyformer.model import GPT, GPTConfig
from tinyformer.tokenizer import load_tokenizer
from tinyformer.utils import get_batch, pick_device

PRESETS = {
    # Fits an iMac CPU/GPU: ~1M parameters with a 2048-token vocabulary.
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=32, lr=1e-3),
    # Good on a free Colab/Kaggle T4 GPU: ~11M parameters.
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, lr=6e-4),
}


@torch.no_grad()
def estimate_loss(model, data, args, device, ctx):
    model.eval()
    out = {}
    for split, d in data.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(d, args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(it, args):
    """Linear warmup, then cosine decay down to 10% of the peak."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    min_lr = args.lr * 0.1
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with train.pt, val.pt, tokenizer.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--n-layer", type=int)
    ap.add_argument("--n-head", type=int)
    ap.add_argument("--n-embd", type=int)
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--dropout", type=float, default=0.1)
    # Architecture choices, for experiments (defaults = GPT-2 style)
    ap.add_argument("--pos-encoding", choices=["learned", "sinusoidal", "rope", "none"], default="learned")
    ap.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    ap.add_argument("--norm-position", choices=["pre", "post"], default="pre")
    ap.add_argument("--activation", choices=["gelu", "relu", "swiglu"], default="gelu")
    ap.add_argument("--no-tie-weights", action="store_true", help="separate input and output embeddings")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--warmup-iters", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    ap.add_argument("--fused-attention", action="store_true", help="use PyTorch's fast attention kernel")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # Preset values fill in anything not given on the command line.
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    tok = load_tokenizer(os.path.join(args.data, "tokenizer.json"))
    data = {
        "train": torch.load(os.path.join(args.data, "train.pt")),
        "val": torch.load(os.path.join(args.data, "val.pt")),
    }
    print(f"device={device}  vocab={tok.vocab_size}  train tokens={len(data['train']):,}")

    # Bits per character: loss converted from "nats per token" to "bits per
    # character of the original text". Unlike plain loss, it can be compared
    # across different tokenizers.
    meta_path = os.path.join(args.data, "meta.json")
    tokens_per_char = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        tokens_per_char = meta["val_tokens"] / meta["val_chars"]

    def bpc(loss):
        return loss * tokens_per_char / math.log(2) if tokens_per_char else float("nan")

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
        use_fused_attention=args.fused_attention, pos_encoding=args.pos_encoding,
        norm=args.norm, norm_position=args.norm_position, activation=args.activation,
        tie_weights=not args.no_tie_weights,
    )
    model = GPT(cfg).to(device)
    optimizer = model.configure_optimizer(args.lr, args.weight_decay)
    start_iter, best_val = 0, float("inf")

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, best_val = ckpt["iter"] + 1, ckpt["best_val"]
        print(f"Resumed from iteration {ckpt['iter']}")
    print(f"Model: {model.num_params() / 1e6:.2f}M parameters")

    # Mixed precision on NVIDIA GPUs: faster and uses less memory.
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    else:
        ctx, scaler = nullcontext(), None

    shutil.copy(os.path.join(args.data, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))
    log_path = os.path.join(args.out, "log.csv")
    log_file = open(log_path, "a", newline="")
    log = csv.writer(log_file)
    if os.path.getsize(log_path) == 0:
        log.writerow(["iter", "train_loss", "val_loss", "val_bpc", "lr", "seconds"])

    t0 = time.time()
    model.train()
    for it in range(start_iter, args.max_iters + 1):
        lr = lr_at(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, data, args, device, ctx)
            elapsed = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| val bpc {bpc(losses['val']):.3f} | lr {lr:.2e} | {elapsed:.0f}s")
            log.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                          f"{bpc(losses['val']):.4f}", f"{lr:.2e}", f"{elapsed:.1f}"])
            log_file.flush()
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": asdict(cfg), "iter": it, "best_val": best_val,
                }, ckpt_path)
            if it == args.max_iters:
                break

        x, y = get_batch(data["train"], args.block_size, args.batch_size, device)
        with ctx:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

    log_file.close()
    result = {
        "best_val_loss": best_val, "best_val_bpc": bpc(best_val),
        "params": model.num_params(), "seconds": time.time() - t0, "device": device,
        "iters": args.max_iters, "seed": args.seed, "config": asdict(cfg),
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"Done. Best val loss {best_val:.4f}, {bpc(best_val):.3f} bits/char. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
