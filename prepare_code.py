"""Build a large Python corpus for the scaling experiments.

Source: codeparrot/github-code-clean on Hugging Face (deduplicated GitHub
files with their repository's license). We keep Python files from
repositories under permissive licenses only (MIT, Apache-2.0, BSD, ISC,
CC0, Unlicense), and put about 1% of repositories, chosen by a hash of the
repository name, in the validation split, so no repository appears in both.

Steps (each can run on its own machine; modal_app.py runs them in parallel):
    extract   one parquet shard -> raw/train_<k>.txt, raw/val_<k>.txt
    tokenizer train a BPE tokenizer on a sample of the training text
    encode    raw text -> uint16 token files, in parallel processes
    concat    join the parts into train.bin / val.bin and write meta.json

The token files are read with np.memmap during training, so the corpus
does not need to fit in memory.
"""

import glob
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np

REPO = "codeparrot/github-code-clean"
N_SHARDS = 880
PERMISSIVE = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "cc0-1.0", "unlicense"]
SEP = "\n\n\n"            # between files
MAX_FILE_CHARS = 200_000  # skip huge files (mostly data dumps)


def is_val(repo_name: str, val_pct: int) -> bool:
    return zlib.crc32(repo_name.encode()) % 100 < val_pct


def extract(shard: int, raw_dir: str, val_pct: int = 1) -> dict:
    """Download one shard, keep permissive Python files, split by repository."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, f"data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet", repo_type="dataset")
    table = pq.read_table(path, columns=["code", "repo_name", "language", "license"],
                          filters=[("language", "=", "Python"), ("license", "in", PERMISSIVE)])
    os.remove(path)
    rows = {"train": [], "val": []}
    licenses = {}
    for code, repo, lic in zip(table["code"].to_pylist(), table["repo_name"].to_pylist(),
                               table["license"].to_pylist()):
        if not code or len(code) > MAX_FILE_CHARS:
            continue
        rows["val" if is_val(repo, val_pct) else "train"].append(code)
        licenses[lic] = licenses.get(lic, 0) + 1
    os.makedirs(raw_dir, exist_ok=True)
    out = {"shard": shard, "licenses": licenses}
    for split, files in rows.items():
        text = SEP.join(files)
        with open(os.path.join(raw_dir, f"{split}_{shard:05d}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        out[f"{split}_files"], out[f"{split}_chars"] = len(files), len(text)
    return out


def train_tok(raw_dir: str, out_dir: str, vocab_size: int = 4096, sample_chars: int = 40_000_000) -> None:
    """Train BPE on the first `sample_chars` characters of the training shards."""
    from tinyformer.tokenizer import train_tokenizer

    parts, n = [], 0
    for path in sorted(glob.glob(os.path.join(raw_dir, "train_*.txt"))):
        with open(path, encoding="utf-8") as f:
            parts.append(f.read(sample_chars - n))
        n += len(parts[-1])
        if n >= sample_chars:
            break
    tok = train_tokenizer(SEP.join(parts), vocab_size)
    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))


_TOK = None


def _encode_piece(args):
    global _TOK
    tok_path, text = args
    if _TOK is None:
        from tinyformer.tokenizer import load_tokenizer

        _TOK = load_tokenizer(tok_path)
    return np.array(_TOK.encode(text), dtype=np.uint16)


def encode(txt_path: str, tok_path: str, out_path: str, procs: int = 4) -> dict:
    """Encode one text file to a uint16 .bin file, splitting the work by source file."""
    with open(txt_path, encoding="utf-8") as f:
        text = f.read()
    files = text.split(SEP)
    # Re-attach the separator so the token stream matches encoding the whole text.
    pieces, buf, size = [], [], 0
    for i, f_ in enumerate(files):
        buf.append(f_ + (SEP if i < len(files) - 1 else ""))
        size += len(f_)
        if size > 1_000_000:
            pieces.append("".join(buf))
            buf, size = [], 0
    if buf:
        pieces.append("".join(buf))
    with Pool(procs) as pool:
        arrays = pool.map(_encode_piece, [(tok_path, p) for p in pieces])
    ids = np.concatenate(arrays) if arrays else np.zeros(0, np.uint16)
    ids.tofile(out_path)
    return {"chars": len(text), "tokens": int(ids.size)}


def concat(parts_dir: str, out_dir: str, stats: list, source_note: dict) -> dict:
    """Join encoded parts into train.bin / val.bin and write meta.json."""
    meta = {"tokenizer": "bpe", "source": REPO, **source_note}
    for split in ("train", "val"):
        paths = sorted(glob.glob(os.path.join(parts_dir, f"{split}_*.bin")))
        total = sum(os.path.getsize(p) // 2 for p in paths)
        out = np.memmap(os.path.join(out_dir, f"{split}.bin"), dtype=np.uint16, mode="w+", shape=(total,))
        pos = 0
        for p in paths:
            a = np.fromfile(p, dtype=np.uint16)
            out[pos : pos + a.size] = a
            pos += a.size
        out.flush()
        meta[f"{split}_tokens"] = int(total)
        meta[f"{split}_chars"] = int(sum(s["chars"] for s in stats if s["split"] == split))
    from tinyformer.tokenizer import load_tokenizer

    meta["vocab_size"] = load_tokenizer(os.path.join(out_dir, "tokenizer.json")).vocab_size
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta
