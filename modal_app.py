"""Run the scaling experiments on cloud GPUs with Modal (https://modal.com).

    modal run modal_app.py::prepare --shards 60          # build the corpus (CPU only)
    modal run modal_app.py::smoke                        # throughput / DDP check on L4
    modal run --detach modal_app.py::sweep --phase lr    # Phase 2: learning-rate transfer
    modal run --detach modal_app.py::sweep --phase ladder --lr-dense 4e-3 --lr-moe 4e-3
    modal volume get tinyformer-runs lr runs/            # fetch results

Two volumes: `tinyformer-data` holds the token files, `tinyformer-runs` the
logs and result.json of every run. A run whose result.json already exists is
skipped, so a sweep can be restarted after an interruption.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess

import modal

GPU = "L4"
data_vol = modal.Volume.from_name("tinyformer-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("tinyformer-runs", create_if_missing=True)
DATA, RUNS = "/data", "/runs"
CODE = "/root/tf"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy>=2", "pyarrow", "huggingface_hub", "matplotlib")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir(".", CODE, ignore=["**/.venv/**", "**/__pycache__/**", "**/data/**", "**/runs/**",
                                      "**/private/**", "**/.git/**", "**/out/**", "**/.pytest_cache/**",
                                      "**/notebooks/**", "**/docs/**"])
)
app = modal.App("tinyformer", image=image)


def _setup():
    import sys

    os.chdir(CODE)
    if CODE not in sys.path:
        sys.path.insert(0, CODE)


# ---------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------- #
@app.function(volumes={DATA: data_vol}, cpu=2, memory=8192, timeout=1800, max_containers=20)
def extract_shard(shard: int) -> dict:
    _setup()
    import prepare_code

    out = prepare_code.extract(shard, f"{DATA}/raw")
    data_vol.commit()
    return out


@app.function(volumes={DATA: data_vol}, cpu=4, memory=16384, timeout=3600)
def train_tokenizer(vocab_size: int = 4096) -> None:
    _setup()
    import prepare_code

    data_vol.reload()
    prepare_code.train_tok(f"{DATA}/raw", f"{DATA}/code", vocab_size)
    data_vol.commit()


@app.function(volumes={DATA: data_vol}, cpu=8, memory=16384, timeout=3600, max_containers=20)
def encode_file(name: str) -> dict:
    _setup()
    import prepare_code

    data_vol.reload()
    os.makedirs(f"{DATA}/parts", exist_ok=True)
    out = prepare_code.encode(f"{DATA}/raw/{name}.txt", f"{DATA}/code/tokenizer.json",
                              f"{DATA}/parts/{name}.bin", procs=8)
    data_vol.commit()
    return {"name": name, "split": name.split("_")[0], **out}


@app.function(volumes={DATA: data_vol}, cpu=2, memory=8192, timeout=3600)
def concat_parts(stats: list, note: dict) -> dict:
    _setup()
    import prepare_code

    data_vol.reload()
    meta = prepare_code.concat(f"{DATA}/parts", f"{DATA}/code", stats, note)
    data_vol.commit()
    return meta


@app.local_entrypoint()
def prepare(shards: int = 60, vocab_size: int = 4096):
    """Extract `shards` parquet shards in parallel, train the tokenizer, encode, concatenate."""
    ext = list(extract_shard.map(range(shards)))
    lic, chars = {}, {"train": 0, "val": 0}
    for e in ext:
        for k, v in e["licenses"].items():
            lic[k] = lic.get(k, 0) + v
        chars["train"] += e["train_chars"]
        chars["val"] += e["val_chars"]
    print(f"extracted {shards} shards: {chars['train'] / 1e9:.2f} GB train, {chars['val'] / 1e6:.1f} MB val; "
          f"licenses {lic}")
    train_tokenizer.remote(vocab_size)
    names = [f"{s}_{e['shard']:05d}" for e in ext for s in ("train", "val")]
    stats = list(encode_file.map(names))
    meta = concat_parts.remote(stats, {"shards": shards, "licenses": lic,
                                       "files": sum(e["train_files"] + e["val_files"] for e in ext)})
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def _run_one(cmd: str, out_dir: str) -> dict:
    result_path = os.path.join(out_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    print(f"$ {cmd}", flush=True)
    with open(os.path.join(out_dir, "stdout.txt"), "w") as log:
        rc = subprocess.call(cmd, shell=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
    runs_vol.commit()
    if rc or not os.path.exists(result_path):
        with open(os.path.join(out_dir, "stdout.txt")) as f:
            tail = f.read()[-3000:]
        print(f"FAILED ({rc}): {cmd}\n{tail}", flush=True)
        return {"failed": True, "cmd": cmd}
    with open(result_path) as f:
        r = json.load(f)
    print(f"done {out_dir}: bpc {r['final_val_bpc']:.4f}, {r['tokens_per_s']:,.0f} tok/s, "
          f"MFU {r['mfu'] or 0:.1%}, {r['seconds']:.0f}s", flush=True)
    return r


@app.function(gpu=GPU, volumes={DATA: data_vol, RUNS: runs_vol}, timeout=24 * 3600, max_containers=10)
def run_group(jobs: list) -> list:
    """Run a list of (command, out_dir) jobs one after another on one GPU."""
    return [_run_one(cmd, out) for cmd, out in jobs]


@app.function(gpu=f"{GPU}:2", volumes={DATA: data_vol, RUNS: runs_vol}, timeout=3600)
def run_ddp(jobs: list) -> list:
    return [_run_one(cmd, out) for cmd, out in jobs]


@app.local_entrypoint()
def smoke():
    """Throughput of each phase-2 width on one L4, then the same 512-wide run on 2 L4s with DDP."""
    import scaling

    jobs = scaling.smoke_jobs(RUNS)
    for r in run_group.remote(jobs["single"]):
        print({k: r.get(k) for k in ("params", "tokens_per_s", "mfu", "final_val_bpc", "failed")})
    for r in run_ddp.remote(jobs["ddp"]):
        print("DDP x2:", {k: r.get(k) for k in ("params", "tokens_per_s", "mfu", "final_val_bpc", "failed")})


@app.local_entrypoint()
def sweep(phase: str = "lr", lr_dense: float = 0.0, lr_moe: float = 0.0, groups: int = 10):
    import scaling

    jobs = scaling.jobs(phase, RUNS, lr_dense=lr_dense, lr_moe=lr_moe)
    batches = scaling.balance(jobs, groups)
    print(f"{len(jobs)} runs in {len(batches)} GPU groups")
    n_fail = 0
    for res in run_group.map(batches, return_exceptions=True):
        if isinstance(res, Exception):
            print("group error:", res)
            n_fail += 1
            continue
        n_fail += sum(1 for r in res if r.get("failed"))
    print(f"finished; {n_fail} failures")
