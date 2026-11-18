"""MORPH (lanl/MORPH): PDE foundation model, sizes Ti / S / M / L.

Follows the inference path of MORPH's own scripts
(scripts/finetune_MORPH_general.py with --n_epochs 0):
7-D data -> RevIN per trajectory -> MORPH (ViT3DRegression) -> undo RevIN.

Card ``config`` keys (all optional):
  size           Ti | S | M | L                        (default Ti)
  max_ar         max autoregressive order of the checkpoint (pretrained Ti/S/M: 1, L: 16)
  ar_order       input frames used per prediction      (default 1)
  heads_xa       cross-attention heads                 (default 32)
  lora_r_attn    LoRA rank in attention, for fine-tuned checkpoints (default 0)
  lora_r_mlp     LoRA rank in the MLP, for fine-tuned checkpoints   (default 0)
  normalization  trajectory | context | none           (default trajectory, as in MORPH)
  batch_size     frames per forward pass               (default 8)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import register
from ._revin import RevIN
from ._torch import run_in_batches
from .base import Adapter, IncompatibleData

# Architecture of each size, from MORPH's scripts: conv filters, dim, heads, depth, MLP dim
SIZES = {
    "Ti": (8, 256, 4, 4, 1024),
    "S": (8, 512, 8, 4, 2048),
    "M": (8, 768, 12, 8, 3072),
    "L": (8, 1024, 16, 16, 4096),
}
# Limits of the pretrained foundation model
PATCH = 8
MAX_PATCHES = 4096
MAX_FIELDS = 3
MAX_COMPONENTS = 3
INSTALL = 'pip install "git+https://github.com/lanl/MORPH.git@package-morph"'


@register("morph")
class MorphAdapter(Adapter):
    def load(self, weights: Optional[Path]) -> None:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("PyTorch isn't installed") from e
        if weights is None:
            raise RuntimeError("This model card has no weights source")
        self.torch = torch
        self.size = str(self.cfg.get("size", "Ti"))
        if self.size not in SIZES:
            raise ValueError(f"Unknown MORPH size '{self.size}' (use Ti, S, M or L)")
        self.max_ar = int(self.cfg.get("max_ar", 16 if self.size == "L" else 1))
        if self.ar_order > self.max_ar:
            raise ValueError(f"ar_order {self.ar_order} is larger than the checkpoint's max_ar {self.max_ar}")

        self.log(f"Loading weights {Path(weights).name}")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}  # saved from DataParallel

        self.model = self._model_matching(state).to(self.device).eval()
        self.revin = RevIN(str(self.cfg.get("normalization", "trajectory")))
        self.batch_size = int(self.cfg.get("batch_size", 8))

    def _build(self, position_variant: str):
        try:
            from morph_pde import MORPH
        except ImportError as e:
            raise RuntimeError(f"MORPH's code isn't installed. Run:  {INSTALL}") from e
        filters, dim, heads, depth, mlp_dim = SIZES[self.size]
        return MORPH(patch_size=PATCH, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                     heads_xa=int(self.cfg.get("heads_xa", 32)), conv_filter=filters,
                     max_components=MAX_COMPONENTS, max_fields=MAX_FIELDS, max_patches=MAX_PATCHES,
                     max_ar=self.max_ar, dropout=0.1, emb_dropout=0.1,
                     lora_r_attn=int(self.cfg.get("lora_r_attn", 0)), lora_r_mlp=int(self.cfg.get("lora_r_mlp", 0)),
                     lora_alpha=None, lora_p=0.0, model_size=position_variant)

    def _model_matching(self, state: dict):
        """Build the model and load the weights.

        MORPH picks its positional encoding from `model_size`, and checkpoints of
        MORPH-L exist with either variant, so for L both are tried.
        """
        variants = [self.size, "Ti"] if self.size == "L" else [self.size]
        first_error = None
        for variant in variants:
            model = self._build(variant)
            try:
                model.load_state_dict(state, strict=True)
                return model
            except RuntimeError as e:
                first_error = first_error or e
        raise RuntimeError(f"Weights don't match MORPH-{self.size}: {str(first_error)[:400]}")

    def check(self, shape) -> None:
        _, T, F, C, D, H, W = shape
        if F > MAX_FIELDS or C > MAX_COMPONENTS:
            raise IncompatibleData(f"MORPH supports up to {MAX_FIELDS} fields and {MAX_COMPONENTS} components "
                                   f"(dataset has {F} and {C})")
        for name, n in (("depth", D), ("height", H), ("width", W)):
            if n > 1 and n % PATCH:
                raise IncompatibleData(f"Grid {name} {n} must be divisible by {PATCH}")
        patches = int(np.prod([n // PATCH if n > 1 else 1 for n in (D, H, W)]))
        if patches > MAX_PATCHES:
            raise IncompatibleData(f"Grid {D}x{H}x{W} gives {patches} patches; MORPH allows {MAX_PATCHES} "
                                   f"(at most 128x128x128)")
        if T <= self.ar_order:
            raise IncompatibleData(f"Need more than {self.ar_order} time steps")

    def begin(self, trajectories: np.ndarray) -> None:
        self.revin.fit(trajectories)

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        x, (mu, sd) = self.revin.normalize(frames, traj)
        out = run_in_batches(self.model, x, self.batch_size, self.device)
        return RevIN.denormalize(out, mu, sd)

    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.model.parameters())
        return {"parameters_m": round(n_params / 1e6, 2), "normalization": self.revin.mode}

    def unload(self) -> None:
        self.model = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
