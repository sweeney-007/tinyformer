# Design notes

Why each part of tinyformer is built the way it is. The rule for the whole
project: **use only Python and raw PyTorch tensors**, so every mechanism is
visible in the code. No Hugging Face, no `tiktoken`, no `sentencepiece`, and
no `nn.Transformer` or `nn.MultiheadAttention`.

## Tokenizer

### Byte-level BPE ([`tinyformer/tokenizer.py`](../tinyformer/tokenizer.py))

| Choice | Why |
|---|---|
| Start from the 256 **byte** values, not characters | Any text in any language (and emoji) can be encoded. There is no "unknown token". |
| **Regex pre-splitting** into words, numbers, punctuation and spaces | Merges never cross word borders, so the vocabulary isn't wasted on pieces like `"e."` or `"the dog"`. Written for the standard `re` module (`[^\W\d_]` = any Unicode letter), so there are no dependencies. |
| Numbers split into chunks of at most 3 digits | Keeps the vocabulary from filling up with specific numbers. |
| **Incremental pair counts + a heap** while training | The textbook algorithm recounts every pair after every merge: O(merges × corpus). We update only the chunks that contain the merged pair. 2,000 merges on 3 MB take about 2 seconds. A test checks this gives exactly the same merges as the textbook version. |
| Identical chunks counted once | `" the"` may appear 10,000 times but is processed once, weighted by its count. |
| Special tokens (`<|endoftext|>`) placed after the learned vocabulary | They are never split, and they mark document borders. |
| Plain JSON save format | Easy to read, compare and inspect in a pull request. |

### Character tokenizer (the baseline)

One token per Unicode character, plus `<|unk|>` for unseen characters. It's
the simplest possible tokenizer, so any gain from BPE has to beat it.

### Comparing tokenizers fairly: bits per character

Loss per token can't be compared across tokenizers: a BPE token holds about 3
characters, so predicting one is harder but covers more text. We convert to
**bits per character (BPC)**:

```
bpc = val_loss_per_token × (val_tokens / val_chars) / ln(2)
```

`prepare.py` splits the **text** (not the tokens) into train and validation,
so every tokenizer is scored on exactly the same validation characters.

## Model ([`tinyformer/model.py`](../tinyformer/model.py))

A decoder-only transformer (the GPT family). The default settings follow GPT-2:

| Part | Default | Alternatives (for experiments) |
|---|---|---|
| Positional information | learned embedding | sinusoidal (2017 paper), **RoPE** (Llama), none |
| Normalization | LayerNorm (written by hand) | RMSNorm (Llama) |
| Norm placement | pre-norm | post-norm (2017 paper) |
| MLP | 4× expansion + GELU | ReLU, **SwiGLU** (Llama, parameter-matched) |
| Output layer | tied to the token embedding | separate weights |
| Attention | written out by hand | `F.scaled_dot_product_attention` (flag) |

Other details:
- **Causal mask** built with `torch.tril`. A test changes a future token and
  checks that earlier predictions stay the same.
- **Initialization:** normal(0, 0.02). Layers that write into the residual
  stream are scaled down by `1/sqrt(2 × n_layer)`, as in GPT-2.
- **Weight decay** only on matrices and embeddings, not biases or norm
  weights.
- **Sampling:** temperature, top-k and top-p (nucleus), plus an optional stop
  token.

## Training ([`train.py`](../train.py))

| Choice | Why |
|---|---|
| AdamW, betas (0.9, 0.95) | The standard for transformers. |
| Linear warmup, then cosine decay to 10% | Stable start, better final loss. |
| Gradient clipping at 1.0 | Protects against loss spikes. |
| Random windows from one long token stream | Simple, and no padding needed. |
| bf16/fp16 autocast on CUDA only | Speeds up on NVIDIA GPUs. Apple MPS and CPU run in fp32. |
| Best checkpoint by validation loss | Avoids keeping an overfitted model. |
| CSV log + `result.json` | Easy to plot and compare in experiments. |

## Hardware

Measured on an Intel iMac (i5-9600K, Radeon Pro 580X, PyTorch 2.2.2), with
the `tiny` preset (≈1M parameters) and 200 steps:

| Device | Time |
|---|---|
| CPU (6 cores) | 135 s |
| Radeon GPU via Metal (MPS) | 53 s |

Both reached the same loss. `--device auto` picks CUDA, then MPS, then CPU.
PyTorch 2.2.2 is the last version for Intel Macs, so the code supports
Python 3.8+.
