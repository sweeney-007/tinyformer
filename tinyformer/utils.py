"""Small helpers shared by the scripts: device choice and batch sampling."""

import torch


def pick_device(name: str = "auto") -> str:
    """'auto' picks CUDA (NVIDIA), then MPS (Apple Metal), then CPU."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(data_dir: str, split: str):
    """Tokens of one split: a uint16 memory-mapped <split>.bin if present (corpora
    larger than RAM, read from disk on demand), else the <split>.pt tensor."""
    import os

    path = os.path.join(data_dir, f"{split}.bin")
    if os.path.exists(path):
        import numpy as np

        return np.memmap(path, dtype=np.uint16, mode="r")
    return torch.load(os.path.join(data_dir, f"{split}.pt"))


def get_batch(data, block_size: int, batch_size: int, device: str, generator=None):
    """Random windows of `block_size` tokens. Targets are the inputs shifted by one.

    `data` is a 1-D tensor or a numpy memmap. Pass a seeded `generator` to get
    the same windows every time (used for a fixed validation set).
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    if isinstance(data, torch.Tensor):
        x = torch.stack([data[i : i + block_size] for i in ix]).long()
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).long()
    else:
        import numpy as np

        win = np.stack([data[i : i + block_size + 1].astype(np.int64) for i in ix.tolist()])
        x, y = torch.from_numpy(win[:, :-1]), torch.from_numpy(win[:, 1:])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
