"""Centralized hardware accelerator detection (Apple Silicon MPS, CUDA, or CPU)."""

from __future__ import annotations


def get_torch_device() -> str:
    """Detect the fastest available PyTorch compute device ('mps', 'cuda', or 'cpu')."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, AttributeError, RuntimeError):
        pass
    return "cpu"
