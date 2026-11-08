"""Byte-level Byte Pair Encoding (BPE) tokenizer, written from scratch.

Only the Python standard library is used. The algorithm is the one behind
GPT-2 style tokenizers:

1. Text is turned into UTF-8 bytes, so the base vocabulary is the 256 byte
   values. Any string, in any language, can be encoded, and nothing is ever
   "unknown".
2. Before training, text is split into chunks (words, numbers, punctuation,
   whitespace) with a regex. Merges never cross chunk borders, so " the" and
   "the." do not turn into one strange token.
3. Training repeatedly finds the most frequent pair of adjacent tokens and
   replaces it with a new token, until the vocabulary reaches the target size.
4. Encoding replays the learned merges in the order they were learned.
"""

import heapq
import json
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# Splits text into chunks before BPE. Close to GPT-2's pattern, but written for
# the standard `re` module: [^\W\d_] means "any Unicode letter".
SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"   # English contractions: 's 't 'll 've 're ...
    r"| ?[^\W\d_]+"           # words, with an optional leading space
    r"| ?\d{1,3}"             # numbers, at most 3 digits per chunk
    r"| ?(?:[^\s\w]|_)+"      # punctuation and symbols
    r"|\s+(?!\S)"             # trailing whitespace
    r"|\s+"                   # any other whitespace
)

Pair = Tuple[int, int]


def _merge(ids: List[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, pattern: str = SPLIT_PATTERN):
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self.merges: Dict[Pair, int] = {}          # (a, b) -> new token id
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {}   # "<|endoftext|>" -> id
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merges from `text`.

        Instead of recounting every pair after every merge (slow), we keep
        running pair counts and only update the chunks that contain the pair
        being merged. A heap gives the most frequent pair quickly; stale heap
        entries are skipped when popped.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (the bytes)"
        num_merges = vocab_size - 256

        # Count identical chunks once: "the" may appear 10,000 times.
        chunk_freqs = Counter(self._regex.findall(text))
        words: List[List[int]] = [list(c.encode("utf-8")) for c in chunk_freqs]
        freqs: List[int] = list(chunk_freqs.values())

        pair_counts: Counter = Counter()
        where: Dict[Pair, set] = defaultdict(set)  # pair -> chunk indices
        for idx, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(idx)

        # Max-heap via negative counts. Ties broken by the smaller pair, so
        # training is deterministic.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for m in range(num_merges):
            best = None
            while heap:
                neg_c, p = heapq.heappop(heap)
                if pair_counts.get(p, 0) == -neg_c and -neg_c > 0:
                    best = p
                    break
            if best is None:
                if verbose:
                    print(f"No more pairs to merge after {m} merges.")
                break

            new_id = 256 + m
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            touched = set()
            for idx in list(where[best]):
                w = words[idx]
                f = freqs[idx]
                # Remove this chunk's old pairs from the counts...
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                w = _merge(w, best, new_id)
                words[idx] = w
                # ...and add its new pairs back.
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where[p].add(idx)
                    touched.add(p)
            del where[best]
            pair_counts.pop(best, None)

            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                elif p in pair_counts:
                    del pair_counts[p]

            if verbose and (m + 1) % max(1, num_merges // 10) == 0:
                tok = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {m + 1}/{num_merges}: {best} -> {new_id} {tok!r}")

        self._cache = {}
        # Special tokens keep their ids above the learned vocabulary.
        specials = list(self.special_tokens)
        self.special_tokens = {}
        self.add_special_tokens(specials)

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        for t in tokens:
            if t not in self.special_tokens:
                self.special_tokens[t] = self.base_vocab_size + len(self.special_tokens)

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    @property
    def base_vocab_size(self) -> int:
        return 256 + len(self.merges)

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.special_tokens)

    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Pick the pair that was learned first (lowest new id).
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text: str) -> List[int]:
        out: List[int] = []
        for chunk in self._regex.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        """Text -> token ids. Special tokens in the text are kept whole."""
        if not allow_special or not self.special_tokens:
            return self._encode_ordinary(text)
        special_re = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        out: List[int] = []
        for part in re.split(special_re, text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids -> text. Invalid UTF-8 (e.g. a cut-off emoji) shows as �."""
        inverse_special = {v: k.encode("utf-8") for k, v in self.special_tokens.items()}
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in inverse_special:
                parts.append(inverse_special[i])
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Readable form of one token, handy for inspecting the vocabulary."""
        for k, v in self.special_tokens.items():
            if v == i:
                return k
        return self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Save / load (plain JSON, easy to read)
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        data = {
            "type": "byte-level-bpe",
            "pattern": self.pattern,
            # Order matters: merges are replayed in this order.
            "merges": [[a, b] for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(pattern=data["pattern"])
        for i, (a, b) in enumerate(data["merges"]):
            new_id = 256 + i
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        return tok


class CharTokenizer:
    """One token per Unicode character: the simplest possible tokenizer.

    Used as the baseline in the tokenizer experiment. Its vocabulary is every
    character seen in training, plus <|unk|> for characters it has never seen.
    """

    UNK = "<|unk|>"

    def __init__(self):
        self.chars: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.special_tokens: Dict[str, int] = {}

    def train(self, text: str, special_tokens: Optional[List[str]] = None) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.special_tokens = {}
        for t in [self.UNK] + list(special_tokens or []):
            self.special_tokens[t] = len(self.chars) + len(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + len(self.special_tokens)

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        unk = self.special_tokens[self.UNK]
        specials = [t for t in self.special_tokens if t != self.UNK] if allow_special else []
        parts = re.split("(" + "|".join(re.escape(t) for t in specials) + ")", text) if specials else [text]
        out: List[int] = []
        for part in parts:
            if part in self.special_tokens and allow_special:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.stoi.get(c, unk) for c in part)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        inverse = {v: k for k, v in self.special_tokens.items()}
        inverse[self.special_tokens[self.UNK]] = "�"
        return "".join(self.chars[i] if i < len(self.chars) else inverse[i] for i in ids)

    def token_str(self, i: int) -> str:
        return self.decode([i])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "chars": self.chars, "special_tokens": self.special_tokens},
                      f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.chars = data["chars"]
        tok.stoi = {c: i for i, c in enumerate(tok.chars)}
        tok.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        return tok


def train_tokenizer(
    text: str, vocab_size: int, special_tokens: Optional[List[str]] = None, verbose: bool = False
) -> BPETokenizer:
    tok = BPETokenizer()
    tok.special_tokens = {t: 0 for t in (special_tokens or [])}
    tok.train(text, vocab_size - len(special_tokens or []), verbose=verbose)
    return tok


def load_tokenizer(path: str):
    """Load either tokenizer type from its JSON file."""
    with open(path, encoding="utf-8") as f:
        kind = json.load(f).get("type")
    return CharTokenizer.load(path) if kind == "char" else BPETokenizer.load(path)
