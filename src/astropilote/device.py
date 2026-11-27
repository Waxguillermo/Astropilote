"""Centralized ONNX Runtime hardware accelerator detection."""

from __future__ import annotations


def get_compute_device() -> str:
    """Detect the fastest available ONNX Runtime execution provider ('coreml', 'cuda', or 'cpu')."""
    try:
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in providers:
            return "cuda"
        if "CoreMLExecutionProvider" in providers:
            return "coreml"
    except (ImportError, AttributeError, RuntimeError):
        pass
    return "cpu"


get_torch_device = get_compute_device
