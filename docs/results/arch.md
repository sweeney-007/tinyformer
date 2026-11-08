# arch experiment

600 training steps, seeds [1]. Lower bits/char is better.
Δ is compared with `baseline`.

| variant | val bits/char | Δ | params | tokens/char | time (s) |
|---|---|---|---|---|---|
| swiglu | 1.910 | -0.078 | 1.07M | 0.317 | 380 |
| rope | 1.913 | -0.075 | 1.06M | 0.317 | 165 |
| untied | 1.930 | -0.058 | 1.33M | 0.317 | 160 |
| relu | 1.935 | -0.053 | 1.07M | 0.317 | 142 |
| sinusoidal | 1.982 | -0.007 | 1.06M | 0.317 | 139 |
| baseline | 1.988 | +0.000 | 1.07M | 0.317 | 141 |
| rmsnorm | 1.996 | +0.007 | 1.07M | 0.317 | 139 |
| no-pos | 2.010 | +0.021 | 1.06M | 0.317 | 141 |
| post-norm | 2.033 | +0.044 | 1.07M | 0.317 | 139 |
