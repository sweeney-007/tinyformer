"""Train the tokenizer on a text file and turn the text into token tensors.

Examples:
    python prepare.py --dataset shakespeare --vocab-size 2048
    python prepare.py --input my_corpus.txt --out data/mycorpus --vocab-size 4096

Writes into --out:
    tokenizer.json   the learned tokenizer (BPE merges or character list)
    train.pt         tokens of the first 90% of the text (torch.int32 tensor)
    val.pt           tokens of the last 10%, for measuring overfitting
    meta.json        character and token counts (used for bits per character)

Compare tokenizers:
    python prepare.py --input code.txt --out data/code_bpe  --tokenizer bpe
    python prepare.py --input code.txt --out data/code_char --tokenizer char
"""

import argparse
import json
import os
import time
import urllib.request

import torch

from tinyformer.tokenizer import CharTokenizer, train_tokenizer

DATASETS = {
    # ~1.1 MB of Shakespeare plays, the classic small benchmark.
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def python_stdlib_corpus(max_chars: int) -> str:
    """Source code of the Python standard library on this machine. No download needed."""
    root = os.path.dirname(os.__file__)
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip tests, vendored packages and installed third-party code.
        dirnames[:] = sorted(d for d in dirnames if d not in ("test", "tests", "site-packages", "idlelib")
                             and not d.startswith("lib2to3"))
        files += [os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".py")]
    parts, total = [], 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        parts.append(src + "\n<|endoftext|>\n")
        total += len(parts[-1])
        if total >= max_chars:
            break
    return "".join(parts)[:max_chars]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=sorted(DATASETS) + ["python-stdlib"],
                     help="a known dataset (python-stdlib is built from local files)")
    src.add_argument("--input", help="path to a UTF-8 text file")
    ap.add_argument("--out", help="output folder (default: data/<dataset or file name>)")
    ap.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
    ap.add_argument("--vocab-size", type=int, default=2048, help="BPE only")
    ap.add_argument("--tokenizer-sample", type=int, default=5_000_000,
                    help="train the tokenizer on at most this many characters (speed)")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--max-chars", type=int, default=3_000_000, help="python-stdlib: corpus size")
    args = ap.parse_args()

    name = args.dataset or os.path.splitext(os.path.basename(args.input))[0]
    out = args.out or os.path.join("data", name)
    os.makedirs(out, exist_ok=True)

    if args.dataset == "python-stdlib":
        path = os.path.join(out, "input.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(python_stdlib_corpus(args.max_chars))
    elif args.dataset:
        path = os.path.join(out, "input.txt")
        if not os.path.exists(path):
            print(f"Downloading {args.dataset} ...")
            urllib.request.urlretrieve(DATASETS[args.dataset], path)
    else:
        path = args.input

    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters from {path}")

    # Split the text (not the tokens), so every tokenizer gets the exact same
    # validation text and their scores can be compared fairly.
    n_val_chars = int(len(text) * args.val_fraction)
    train_text, val_text = text[:-n_val_chars], text[-n_val_chars:]

    t0 = time.time()
    if args.tokenizer == "bpe":
        tok = train_tokenizer(train_text[: args.tokenizer_sample], args.vocab_size,
                              special_tokens=["<|endoftext|>"], verbose=True)
    else:
        tok = CharTokenizer()
        tok.train(train_text, special_tokens=["<|endoftext|>"])
    tok.save(os.path.join(out, "tokenizer.json"))
    print(f"{args.tokenizer} tokenizer: {tok.vocab_size} tokens, trained in {time.time() - t0:.1f}s")

    t0 = time.time()
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.int32)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.int32)
    print(f"Encoded into {len(train_ids) + len(val_ids):,} tokens "
          f"({len(text) / (len(train_ids) + len(val_ids)):.2f} chars/token) in {time.time() - t0:.1f}s")

    torch.save(train_ids, os.path.join(out, "train.pt"))
    torch.save(val_ids, os.path.join(out, "val.pt"))
    meta = {
        "tokenizer": args.tokenizer, "vocab_size": tok.vocab_size, "source": path,
        "train_chars": len(train_text), "train_tokens": len(train_ids),
        "val_chars": len(val_text), "val_tokens": len(val_ids),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"Saved {len(train_ids):,} train / {len(val_ids):,} val tokens to {out}/")


if __name__ == "__main__":
    main()
