from tinyformer.tokenizer import BPETokenizer, train_tokenizer

CORPUS = (
    "the cat sat on the mat. the dog sat on the log. "
    "The theory of the thing is that there is no other thing. "
) * 50 + "Café, naïve, 東京, emoji 🚀🚀, numbers 12345 and tabs\tand\nnewlines.\n"


def test_roundtrip_is_lossless():
    tok = train_tokenizer(CORPUS, vocab_size=300)
    samples = [CORPUS, "", " ", "unseen wörds ✓ 🚀", "\n\n  \t", "x" * 1000]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s


def test_merges_shorten_text():
    tok = train_tokenizer(CORPUS, vocab_size=320)
    assert tok.vocab_size == 320
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode("utf-8")) / 2


def test_common_words_become_single_tokens():
    tok = train_tokenizer(CORPUS, vocab_size=320)
    assert len(tok.encode(" the")) == 1


def test_special_tokens_are_atomic():
    tok = train_tokenizer(CORPUS, vocab_size=300, special_tokens=["<|endoftext|>"])
    eot = tok.special_tokens["<|endoftext|>"]
    assert eot == tok.vocab_size - 1
    ids = tok.encode("the cat<|endoftext|>the dog")
    assert ids.count(eot) == 1
    assert tok.decode(ids) == "the cat<|endoftext|>the dog"
    # With specials turned off, the same text is just ordinary characters.
    assert eot not in tok.encode("<|endoftext|>", allow_special=False)


def test_training_is_deterministic():
    a = train_tokenizer(CORPUS, vocab_size=300)
    b = train_tokenizer(CORPUS, vocab_size=300)
    assert a.merges == b.merges


def test_matches_naive_bpe():
    """The fast incremental trainer must learn the same merges as the textbook version."""
    import re
    from collections import Counter
    from tinyformer.tokenizer import SPLIT_PATTERN, _merge

    chunks = [list(c.encode("utf-8")) for c in re.findall(SPLIT_PATTERN, CORPUS)]
    naive = []
    for m in range(40):
        counts = Counter(p for c in chunks for p in zip(c, c[1:]))
        best = max(counts, key=lambda p: (counts[p], tuple(-x for x in p)))
        naive.append(best)
        chunks = [_merge(c, best, 256 + m) for c in chunks]

    tok = train_tokenizer(CORPUS, vocab_size=256 + 40)
    fast = sorted(tok.merges, key=tok.merges.get)
    assert fast == naive


def test_save_and_load(tmp_path):
    tok = train_tokenizer(CORPUS, vocab_size=300, special_tokens=["<|endoftext|>"])
    path = tmp_path / "tok.json"
    tok.save(str(path))
    loaded = BPETokenizer.load(str(path))
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    s = "the dog sat 🚀<|endoftext|>"
    assert loaded.encode(s) == tok.encode(s)


def test_char_tokenizer(tmp_path):
    from tinyformer.tokenizer import CharTokenizer, load_tokenizer

    tok = CharTokenizer()
    tok.train("hello world", special_tokens=["<|endoftext|>"])
    assert tok.vocab_size == len(set("hello world")) + 2  # + <|unk|> and <|endoftext|>
    assert tok.decode(tok.encode("hello<|endoftext|>world")) == "hello<|endoftext|>world"
    assert tok.decode(tok.encode("hex")) == "he�"  # unseen 'x' -> unknown
    path = tmp_path / "char.json"
    tok.save(str(path))
    loaded = load_tokenizer(str(path))
    assert isinstance(loaded, CharTokenizer)
    assert loaded.encode("hello world") == tok.encode("hello world")
