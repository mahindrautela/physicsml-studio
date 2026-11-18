"""Helper shared by the PyTorch-based adapters."""
from __future__ import annotations

import numpy as np


def run_in_batches(model, x: np.ndarray, batch_size: int, device: str, pad_to_batch: bool = False) -> np.ndarray:
    """Feed x (B, k, F, C, D, H, W) through a PyTorch model in chunks, without gradients.

    The model may return a tensor or a tuple whose last item is the prediction
    (MORPH returns (encoding, latent, prediction)). With ``pad_to_batch`` the
    last chunk is zero-padded to ``batch_size``, for models exported with a
    fixed batch size.
    """
    import torch

    outputs = []
    with torch.inference_mode():
        for i in range(0, x.shape[0], batch_size):
            chunk = np.ascontiguousarray(x[i:i + batch_size], dtype=np.float32)
            n = chunk.shape[0]
            if pad_to_batch and n < batch_size:
                chunk = np.concatenate([chunk, np.zeros((batch_size - n, *chunk.shape[1:]), np.float32)])
            out = model(torch.from_numpy(chunk).to(device))
            if isinstance(out, (tuple, list)):
                out = out[-1]
            outputs.append(out[:n].float().cpu().numpy())
    return np.concatenate(outputs, axis=0)
