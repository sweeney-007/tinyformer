"""Train the transformer on data made by prepare.py (or prepare_code.py).

Examples:
    # Quick CPU run on an iMac (a few minutes):
    python train.py --data data/shakespeare --preset tiny

    # Bigger model on a free Colab/Kaggle GPU:
    python train.py --data data/shakespeare --preset small --max-iters 5000

    # GPU run with a token budget, bf16 and torch.compile:
    python train.py --data /data/code --n-layer 6 --n-embd 512 --n-head 8 --tokens 300e6 --compile

    # Several GPUs on one machine (DistributedDataParallel):
    torchrun --standalone --nproc_per_node 2 train.py --data /data/code ... --compile

Saves the best checkpoint (lowest validation loss) to <out>/ckpt.pt, a loss
log to <out>/log.csv and a summary to <out>/result.json.
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tinyformer.model import GPT, GPTConfig
from tinyformer.tokenizer import load_tokenizer
from tinyformer.utils import get_batch, load_split, pick_device

PRESETS = {
    # Fits an iMac CPU/GPU: ~1M parameters with a 2048-token vocabulary.
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=32, lr=1e-3),
    # Good on a free Colab/Kaggle T4 GPU: ~11M parameters.
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, lr=6e-4),
}


@torch.no_grad()
def estimate_loss(model, data, args, device, ctx, iters):
    """Mean loss over `iters` batches per split. The windows come from a fixed
    seed, so every evaluation (and every run) scores exactly the same text."""
    model.eval()
    out = {}
    for split, d in data.items():
        g = torch.Generator().manual_seed(1234)
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(d, args.block_size, args.batch_size, device, generator=g)
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with train/val tokens (.pt or uint16 .bin) and tokenizer.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--n-layer", type=int)
    ap.add_argument("--n-head", type=int)
    ap.add_argument("--n-embd", type=int)
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--batch-size", type=int, help="sequences per micro-batch, per GPU")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--dropout", type=float, default=0.1)
    # Architecture choices, for experiments (defaults = GPT-2 style)
    ap.add_argument("--pos-encoding", choices=["learned", "sinusoidal", "rope", "none"], default="learned")
    ap.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    ap.add_argument("--norm-position", choices=["pre", "post"], default="pre")
    ap.add_argument("--activation", choices=["gelu", "relu", "swiglu"], default="gelu")
    ap.add_argument("--no-tie-weights", action="store_true", help="separate input and output embeddings")
    ap.add_argument("--no-bias", action="store_true", help="no biases in Linear and LayerNorm layers")
    ap.add_argument("--n-experts", type=int, default=0, help="mixture-of-experts MLP with this many experts (0 = dense)")
    ap.add_argument("--moe-top-k", type=int, default=2)
    ap.add_argument("--moe-capacity", type=float, default=1.25, help="expert capacity factor (<= 0: never drop tokens)")
    ap.add_argument("--moe-aux-coef", type=float, default=0.01, help="weight of the load-balancing loss")
    ap.add_argument("--mup", action="store_true", help="maximal update parametrization (muP)")
    ap.add_argument("--mup-base-width", type=int, default=128, help="width at which muP equals the standard model")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--tokens", type=float, help="training token budget; sets --max-iters")
    ap.add_argument("--warmup-iters", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--final-eval-iters", type=int, help="batches for the last evaluation (default: --eval-iters)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    ap.add_argument("--fused-attention", action="store_true", help="use PyTorch's fast attention kernel")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (PyTorch 2, CUDA)")
    ap.add_argument("--peak-tflops", type=float, default=0.0,
                    help="GPU peak bf16 TFLOPS for the MFU column (NVIDIA L4: 121)")
    ap.add_argument("--no-ckpt", action="store_true", help="do not save checkpoints (for sweeps)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)
    # Preset values fill in anything not given on the command line.
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def main(argv=None):
    args = parse_args(argv)

    # DistributedDataParallel: torchrun starts one process per GPU and sets RANK.
    ddp = "RANK" in os.environ
    if ddp:
        rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        use_cuda = torch.cuda.is_available() and args.device in ("auto", "cuda")
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        device = f"cuda:{local_rank}" if use_cuda else "cpu"
        if use_cuda:
            torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = pick_device(args.device)
    master = rank == 0

    def say(*a, **kw):
        if master:
            print(*a, **kw, flush=True)

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum * world
    if args.tokens:
        args.max_iters = math.ceil(args.tokens / tokens_per_step)
    if args.final_eval_iters is None:
        args.final_eval_iters = args.eval_iters

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(os.path.join(args.data, "tokenizer.json"))
    data = {"train": load_split(args.data, "train"), "val": load_split(args.data, "val")}
    say(f"device={device} x{world}  vocab={tok.vocab_size}  train tokens={len(data['train']):,}")

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
        tie_weights=not args.no_tie_weights, bias=not args.no_bias,
        n_experts=args.n_experts, moe_top_k=args.moe_top_k,
        moe_capacity_factor=args.moe_capacity, moe_aux_coef=args.moe_aux_coef,
        mup=args.mup, mup_base_width=args.mup_base_width,
    )
    raw_model = GPT(cfg).to(device)
    optimizer = raw_model.configure_optimizer(args.lr, args.weight_decay)
    start_iter, best_val = 0, float("inf")
    if ddp:
        # Same weights on every GPU (DDP copies rank 0's), different batches.
        torch.manual_seed(args.seed + 1000 * rank)

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, best_val = ckpt["iter"] + 1, ckpt["best_val"]
        say(f"Resumed from iteration {ckpt['iter']}")
    counts = raw_model.param_counts()
    flops_tok = raw_model.flops_per_token()
    say(f"Model: {counts['total'] / 1e6:.2f}M parameters ({counts['active'] / 1e6:.2f}M active non-embedding), "
        f"{flops_tok / 1e6:.1f} MFLOPs/token, {args.max_iters} steps x {tokens_per_step:,} tokens")

    model = torch.compile(raw_model) if args.compile else raw_model
    if ddp:
        model = DDP(model, device_ids=[int(device.split(":")[1])] if device.startswith("cuda") else None)

    # Mixed precision on NVIDIA GPUs: faster and uses less memory.
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        ctx, scaler = nullcontext(), None

    log_file = log = None
    if master:
        shutil.copy(os.path.join(args.data, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))
        log_path = os.path.join(args.out, "log.csv")
        log_file = open(log_path, "a", newline="")
        log = csv.writer(log_file)
        if os.path.getsize(log_path) == 0:
            log.writerow(["iter", "tokens", "train_loss", "val_loss", "val_bpc", "lr", "seconds", "tok_per_s"])

    t0 = time.time()
    train_time, train_tokens, diverged = 0.0, 0, False
    model.train()
    for it in range(start_iter, args.max_iters + 1):
        lr = lr_at(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)  # muP scales some groups

        last = it == args.max_iters
        if it % args.eval_interval == 0 or last:
            losses = estimate_loss(raw_model if ddp else model, data, args, device, ctx,
                                   args.final_eval_iters if last else args.eval_iters)
            elapsed = time.time() - t0
            tps = train_tokens / train_time if train_time else 0.0
            mfu = f" | MFU {flops_tok * tps / (args.peak_tflops * 1e12 * world):.1%}" if args.peak_tflops and tps else ""
            say(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| val bpc {bpc(losses['val']):.3f} | lr {lr:.2e} | {elapsed:.0f}s | {tps:,.0f} tok/s{mfu}")
            if cfg.n_experts and it > start_iter:
                st = raw_model.moe_stats()
                busiest = max(max(layer) for layer in st["load"])
                say(f"           moe: busiest expert {busiest:.2f} of choices "
                    f"(even = {1 / cfg.n_experts:.2f}), dropped {max(st['dropped']):.3f}")
            if master:
                log.writerow([it, it * tokens_per_step, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                              f"{bpc(losses['val']):.4f}", f"{lr:.2e}", f"{elapsed:.1f}", f"{tps:.0f}"])
                log_file.flush()
            if not math.isfinite(losses["train"]):
                diverged = True
                say("Loss is not finite: stopping (diverged).")
                break
            if losses["val"] < best_val:
                best_val = losses["val"]
                if master and not args.no_ckpt:
                    torch.save({
                        "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": asdict(cfg), "iter": it, "best_val": best_val,
                    }, ckpt_path)
            if last:
                break

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts = time.time()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = get_batch(data["train"], args.block_size, args.batch_size, device)
            if ddp:  # only all-reduce gradients after the last micro-batch
                model.require_backward_grad_sync = micro == args.grad_accum - 1
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if it > start_iter + 10:  # skip warm-up steps (compilation) when timing
            train_time += time.time() - ts
            train_tokens += tokens_per_step

    if master:
        log_file.close()
        tps = train_tokens / train_time if train_time else 0.0
        result = {
            "best_val_loss": best_val, "best_val_bpc": bpc(best_val),
            "final_val_loss": losses["val"], "final_val_bpc": bpc(losses["val"]),
            "final_train_loss": losses["train"], "diverged": diverged,
            "params": counts["total"], "params_non_embedding": counts["non_embedding"],
            "params_active": counts["active"], "flops_per_token": flops_tok,
            "tokens": args.max_iters * tokens_per_step, "train_flops": flops_tok * args.max_iters * tokens_per_step,
            "tokens_per_s": tps, "mfu": flops_tok * tps / (args.peak_tflops * 1e12 * world) if args.peak_tflops else None,
            "seconds": time.time() - t0, "device": device, "world_size": world,
            "iters": args.max_iters, "seed": args.seed, "lr": args.lr, "config": asdict(cfg),
            "moe": raw_model.moe_stats() if cfg.n_experts else None,
        }
        with open(os.path.join(args.out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)
        say(f"Done. Final val loss {losses['val']:.4f}, {bpc(losses['val']):.3f} bits/char.")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
