"""MORPH (lanl/MORPH): PDE foundation model, sizes Ti / S / M / L.

Mirrors the inference path of MORPH's own scripts
(scripts/finetune_MORPH_general.py with --n_epochs 0):
UPTF-7 data -> RevIN per trajectory -> ViT3DRegression -> denormalize.

Card ``config`` keys:
  size           Ti | S | M | L                       (default Ti)
  max_ar         max autoregressive order of the checkpoint (FM Ti/S/M: 1, FM L: 16)
  ar_order       input frames used for prediction     (default 1)
  heads_xa       cross-attention heads                (default 32)
  lora_r_attn    LoRA rank in attention, for fine-tuned checkpoints (default 0)
  lora_r_mlp     LoRA rank in the MLP, for fine-tuned checkpoints   (default 0)
  normalization  trajectory | context | none          (default trajectory, as in MORPH)
  batch_size     frames per forward pass              (default 8)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import register
from ._revin import RevIN
from .base import Adapter, IncompatibleData

SIZES = {  # conv filters, dim, heads, depth, mlp_dim  (from MORPH's scripts)
    "Ti": [8, 256, 4, 4, 1024],
    "S": [8, 512, 8, 4, 2048],
    "M": [8, 768, 12, 8, 3072],
    "L": [8, 1024, 16, 16, 4096],
}
PATCH = 8
MAX_PATCHES = 4096
MAX_FIELDS = 3
MAX_COMPONENTS = 3


@register("morph")
class MorphAdapter(Adapter):
    def load(self, weights: Optional[Path]) -> None:
        try:
            import torch
            from morph_pde import MORPH
        except ImportError as e:
            raise RuntimeError(
                "MORPH's code isn't installed. Run:  pip install "
                "\"git+https://github.com/lanl/MORPH.git@package-morph\"") from e
        self.torch = torch
        size = str(self.cfg.get("size", "Ti"))
        if size not in SIZES:
            raise ValueError(f"Unknown MORPH size '{size}' (use Ti, S, M or L)")
        filters, dim, heads, depth, mlp = SIZES[size]
        self.max_ar = int(self.cfg.get("max_ar", 16 if size == "L" else 1))
        if self.ar_order > self.max_ar:
            raise ValueError(f"ar_order {self.ar_order} is larger than the checkpoint's max_ar {self.max_ar}")
        r_attn, r_mlp = int(self.cfg.get("lora_r_attn", 0)), int(self.cfg.get("lora_r_mlp", 0))

        def build(size_flag: str):
            return MORPH(patch_size=PATCH, dim=dim, depth=depth, heads=heads,
                         heads_xa=int(self.cfg.get("heads_xa", 32)), mlp_dim=mlp,
                         max_components=MAX_COMPONENTS, conv_filter=filters, max_ar=self.max_ar,
                         max_patches=MAX_PATCHES, max_fields=MAX_FIELDS, dropout=0.1, emb_dropout=0.1,
                         lora_r_attn=r_attn, lora_r_mlp=r_mlp, lora_alpha=None,
                         lora_p=0.0, model_size=size_flag)

        model = build(size)
        if weights is None:
            raise RuntimeError("This model card has no weights source")
        self.log(f"Loading weights {Path(weights).name}")
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as first:
            # MORPH-L's positional encoding depends on how the checkpoint was built; try the other variant.
            alt = build("Ti" if size == "L" else size)
            try:
                alt.load_state_dict(state, strict=True)
                model = alt
            except RuntimeError:
                raise RuntimeError(f"Weights don't match MORPH-{size}: {str(first)[:400]}") from first
        self.model = model.to(self.device).eval()
        self.n_params = sum(p.numel() for p in model.parameters())
        self.revin = RevIN(str(self.cfg.get("normalization", "trajectory")))
        self.batch_size = int(self.cfg.get("batch_size", 8))

    def check(self, shape) -> None:
        _, T, F, C, D, H, W = shape
        if F > MAX_FIELDS or C > MAX_COMPONENTS:
            raise IncompatibleData(f"MORPH supports up to {MAX_FIELDS} fields and {MAX_COMPONENTS} components "
                                   f"(dataset has {F} and {C})")
        for name, n in (("depth", D), ("height", H), ("width", W)):
            if n > 1 and n % PATCH:
                raise IncompatibleData(f"Grid {name} {n} must be divisible by {PATCH}")
        patches = 1
        for n in (D, H, W):
            patches *= (n // PATCH) if n > 1 else 1
        if patches > MAX_PATCHES:
            raise IncompatibleData(f"Grid {D}x{H}x{W} gives {patches} patches; MORPH allows {MAX_PATCHES} "
                                   f"(at most 128x128x128)")
        if T <= self.ar_order:
            raise IncompatibleData(f"Need more than {self.ar_order} time steps")

    def begin(self, trajectories: np.ndarray) -> None:
        self.revin.fit(trajectories)

    def predict(self, frames: np.ndarray, traj: np.ndarray) -> np.ndarray:
        torch = self.torch
        x, (mu, sd) = self.revin.normalize(frames, traj)
        outs = []
        with torch.inference_mode():
            for i in range(0, x.shape[0], self.batch_size):
                inp = torch.from_numpy(np.ascontiguousarray(x[i:i + self.batch_size], dtype=np.float32)).to(self.device)
                _, _, out = self.model(inp)
                outs.append(out.float().cpu().numpy())
        out = np.concatenate(outs, axis=0)
        return RevIN.denormalize(out, mu, sd)

    def info(self) -> dict:
        return {"parameters_m": round(self.n_params / 1e6, 2), "normalization": self.revin.mode}

    def unload(self) -> None:
        self.model = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
