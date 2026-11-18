"""Which compute devices PyTorch can use, and which one a run should use."""
from __future__ import annotations


def available_devices() -> list[str]:
    """Device names for the Settings menu: 'auto', 'cpu', then any GPUs."""
    devices = ["auto", "cpu"]
    try:
        import torch
    except ImportError:
        return devices
    devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():  # Apple silicon
        devices.append("mps")
    return devices


def gpu_names() -> list[str]:
    try:
        import torch
    except ImportError:
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def pick_device(choice: str = "auto") -> str:
    """'auto' means: NVIDIA GPU if there is one, then Apple GPU, then CPU."""
    if choice and choice != "auto":
        return choice
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
