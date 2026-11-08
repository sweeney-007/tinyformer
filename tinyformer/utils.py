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


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Random windows of `block_size` tokens. Targets are the inputs shifted by one."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).long()
    return x.to(device), y.to(device)
