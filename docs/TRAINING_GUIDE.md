# Where and how to train

## Hardware options

| Where | Cost | Good for | Notes |
|---|---|---|---|
| **Your Mac** (CPU or Apple/AMD GPU via MPS) | free | development, tests, `tiny` preset (≈1M params), experiments of a few thousand steps | On an Intel iMac the Radeon GPU was 2.5× faster than the CPU. |
| **Google Colab** (free T4 GPU) | free | `small` preset (≈11M params), 5–10k steps | Sessions stop after a few hours of use. Save checkpoints to Google Drive. |
| **Kaggle Notebooks** (free T4 ×2 or P100) | free, about 30 GPU-hours per week | longer runs and multi-seed experiment suites | Upload the prepared `data/` folder as a Kaggle dataset. |
| **Lightning AI Studios** | free monthly credits | a VS Code-like setup with a GPU | |
| Paid (RunPod, Lambda, Vast.ai) | ≈ $0.2–2/hour | anything bigger | Only needed beyond about 50M parameters. |

A good workflow: **write and test code on the Mac, run the big experiment
suites on Colab/Kaggle.** The code is identical everywhere, and
`--device auto` picks the right hardware.

Run on Colab with [`notebooks/train_colab.ipynb`](../notebooks/train_colab.ipynb).

## Datasets

| Dataset | Size | Problem | Why it's good here |
|---|---|---|---|
| **Python standard library** (`--dataset python-stdlib`) | 3 MB (adjustable) | code completion | Already on your machine, no download. Code shows tokenizer effects clearly. |
| **TinyShakespeare** (`--dataset shakespeare`) | 1.1 MB | text generation | The classic benchmark. Easy to compare with other people's results. |
| **TinyStories** (Hugging Face) | about 2 GB | simple English stories | Built so small models can write coherent stories. Use a slice. |
| **Project Gutenberg** books | any | style imitation | Public-domain books in many languages. Good for a multilingual tokenizer test. |
| **Your own notes, chat logs or code** | any | a personal autocomplete | A fun demo, but keep private data out of the public repo. |

Any UTF-8 text file works: `python prepare.py --input file.txt`.

## Problems a small model is actually good at

A model with 1–50M parameters won't be a chatbot. It *can* do focused jobs
well, and these make better portfolio stories:

1. **Code completion for one language or codebase**: completes the next line
   in your style. Measure exact-match accuracy of the next line.
2. **Structured text generation**: SQL queries, regular expressions, music
   in ABC notation, chess games in PGN, SMILES chemical formulas. Small
   vocabulary, strict grammar, and you can automatically check if the output
   is valid.
3. **Learning algorithms from examples**: addition, sorting, string reversal.
   Good for studying *how* transformers learn (e.g. "grokking"), with
   perfect ground truth.
4. **Text classification / spam / sentiment**: add a classification head on
   the final hidden state and fine-tune.
5. **A language-model tokenizer study for a low-resource language**: compare
   how many tokens BPE needs per word in English vs your language.

## Measuring quality

- **Validation bits per character**: the main number, comparable across
  tokenizers and models.
- **Perplexity** = `exp(loss per token)`. Common, but only comparable at the
  same tokenizer.
- **Task metrics**: for code, the share of generated snippets that parse
  (`ast.parse`). For structured formats, the share of valid outputs.
- **Samples**: always include a few in the README. People remember them.
