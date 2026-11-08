"""Generate text from a trained checkpoint.

Examples:
    python sample.py --out out --prompt "ROMEO:" --max-new-tokens 200
    python sample.py --out out --interactive
"""

import argparse
import os

import torch

from tinyformer.model import GPT, GPTConfig
from tinyformer.tokenizer import load_tokenizer
from tinyformer.utils import pick_device


def load(out_dir, device):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok = load_tokenizer(os.path.join(out_dir, "tokenizer.json"))
    return model, tok


def complete(model, tok, prompt, args, device):
    ids = tok.encode(prompt) or [tok.special_tokens.get("<|endoftext|>", 0)]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, args.max_new_tokens, temperature=args.temperature,
                       top_k=args.top_k, top_p=args.top_p,
                       stop_token=tok.special_tokens.get("<|endoftext|>"))
    return tok.decode(y[0].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out", help="folder with ckpt.pt and tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = always pick the most likely token")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--interactive", action="store_true", help="type prompts in a loop")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, tok = load(args.out, device)

    if args.interactive:
        print("Type a prompt and press Enter (Ctrl+C to quit).")
        while True:
            try:
                prompt = input("\n> ")
            except (KeyboardInterrupt, EOFError):
                break
            print(complete(model, tok, prompt, args, device))
        return

    for i in range(args.num_samples):
        print(complete(model, tok, args.prompt, args, device))
        if i < args.num_samples - 1:
            print("-" * 40)


if __name__ == "__main__":
    main()
