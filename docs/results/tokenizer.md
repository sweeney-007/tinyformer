# tokenizer experiment

600 training steps, seeds [1]. Lower bits/char is better.
Δ is compared with `pycode_bpe`.

| variant | val bits/char | Δ | params | tokens/char | time (s) |
|---|---|---|---|---|---|
| pycode_bpe | 1.988 | +0.000 | 1.07M | 0.317 | 147 |
| pycode_char | 2.877 | +0.888 | 0.82M | 0.999 | 156 |
