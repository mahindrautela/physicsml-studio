"""Read trajectory datasets (.npy, .nc, .h5) into the standard 7-D layout.

Every dataset is served to models as a float32 array shaped
(N, T, F, C, D, H, W): trajectories, time steps, fields, components, and a
D x H x W grid. Only the trajectories that are needed are read from disk, so
multi-GB files are fine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .catalog import DatasetCard

CANON = "NTFCDHW"


def detect_format(path: Path, fmt: str = "auto") -> str:
    if fmt != "auto":
        return fmt
    ext = path.suffix.lower()
    if ext == ".npy":
        return "npy"
    if ext in (".nc", ".nc4", ".cdf"):
        return "nc"
    if ext in (".h5", ".hdf5", ".hdf"):
        return "h5"
    raise ValueError(f"Can't tell the format of '{path.name}'. Use .npy, .nc or .h5, or set 'format' on the card.")


def list_variables(path: Path, fmt: str = "auto") -> list[dict]:
    """Numeric arrays inside a file: [{'name', 'shape', 'dtype'}], largest first."""
    fmt = detect_format(path, fmt)
    out: list[dict] = []
    if fmt == "npy":
        a = np.load(path, mmap_mode="r")
        out.append({"name": path.stem, "shape": list(a.shape), "dtype": str(a.dtype)})
    elif fmt == "nc":
        import netCDF4

        with netCDF4.Dataset(path, "r") as ds:
            def walk(g, prefix=""):
                for name, v in g.variables.items():
                    if np.issubdtype(v.dtype, np.number):
                        out.append({"name": prefix + name, "shape": list(v.shape), "dtype": str(v.dtype)})
                for gname, sub in g.groups.items():
                    walk(sub, prefix + gname + "/")
            walk(ds)
    else:
        import h5py

        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                    out.append({"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
            f.visititems(visit)
    out.sort(key=lambda d: -int(np.prod(d["shape"] or [0])))
    return out


class DatasetHandle:
    def __init__(self, card: DatasetCard, path: Path):
        self.card = card
        self.path = Path(path)
        self.fmt = detect_format(self.path, card.format)
        variables = list_variables(self.path, self.fmt)
        if not variables:
            raise ValueError(f"No numeric arrays found in {self.path.name}")
        if card.variable:
            match = [v for v in variables if v["name"] == card.variable]
            if not match:
                names = ", ".join(v["name"] for v in variables)
                raise ValueError(f"Variable '{card.variable}' not in {self.path.name}. Available: {names}")
            self.var = match[0]
        else:
            self.var = variables[0]
        self.raw_shape = tuple(self.var["shape"])
        if len(self.raw_shape) < 3:
            raise ValueError(f"Expected at least (trajectories, time, space...), got shape {self.raw_shape}")
        self.n_traj = self.raw_shape[0]
        self.shape = self._canonical_shape()

    # ---- layout -------------------------------------------------------
    def _canonical_shape(self) -> tuple[int, ...]:
        lay = self.card.layout
        if lay.axes:
            if len(lay.axes) != len(self.raw_shape):
                raise ValueError(f"layout.axes '{lay.axes}' has {len(lay.axes)} letters but the array "
                                 f"has {len(self.raw_shape)} dims {self.raw_shape}")
            sizes = dict(zip(lay.axes, self.raw_shape))
            return tuple(sizes.get(a, 1) for a in CANON)
        N, T = self.raw_shape[:2]
        want = (lay.fields, lay.components, lay.depth, lay.height, lay.width)
        if int(np.prod(self.raw_shape[2:])) != int(np.prod(want)):
            raise ValueError(
                f"Array shape {self.raw_shape} can't be reshaped to F,C,D,H,W = {want}. "
                f"Fix the sizes on the dataset card, or set layout.axes (e.g. 'NTFHW').")
        return (N, T, *want)

    def _to_canonical(self, a: np.ndarray) -> np.ndarray:
        lay = self.card.layout
        n = a.shape[0]
        if lay.axes:
            present = [c for c in CANON if c in lay.axes]
            a = np.transpose(a, [lay.axes.index(c) for c in present])
        return np.ascontiguousarray(a, dtype=np.float32).reshape((n, *self.shape[1:]))

    # ---- reading ------------------------------------------------------
    def read(self, indices: list[int]) -> np.ndarray:
        idx = sorted(int(i) for i in indices)
        if not idx:
            raise ValueError("No trajectories selected")
        if self.fmt == "npy":
            a = np.load(self.path, mmap_mode="r")[idx]
        elif self.fmt == "nc":
            import netCDF4

            with netCDF4.Dataset(self.path, "r") as ds:
                *groups, name = self.var["name"].split("/")
                g = ds
                for grp in groups:
                    g = g.groups[grp]
                v = g.variables[name]
                v.set_auto_mask(False)
                a = np.asarray(v[idx])
        else:
            import h5py

            with h5py.File(self.path, "r") as f:
                a = f[self.var["name"]][idx]
        a = np.asarray(a)
        if not np.all(np.isfinite(a)):
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return self._to_canonical(a)

    def eval_indices(self, limit: Optional[int] = None) -> list[int]:
        ev, N = self.card.eval, self.n_traj
        if ev.mode == "all":
            idx = list(range(N))
        elif ev.mode == "first_n":
            idx = list(range(min(ev.n, N)))
        elif ev.mode == "indices":
            idx = [i for i in ev.indices if 0 <= i < N]
        else:
            k = max(1, int(round(N * ev.fraction)))
            idx = list(range(N - k, N))
        if limit:
            idx = idx[:limit]
        if not idx:
            raise ValueError("The evaluation split selects no trajectories")
        return idx

    def names(self) -> tuple[list[str], list[str]]:
        F, C = self.shape[2], self.shape[3]
        f = list(self.card.field_names[:F]) + [f"field {i + 1}" for i in range(len(self.card.field_names), F)]
        default_c = ["x", "y", "z"] if C > 1 else ["value"]
        c = list(self.card.component_names[:C]) + default_c[len(self.card.component_names):C]
        return f, c

    def info(self) -> dict:
        N, T, F, C, D, H, W = self.shape
        dims = 3 if D > 1 else (2 if H > 1 else 1)
        fields, comps = self.names()
        return {"file": self.path.name, "format": self.fmt, "variable": self.var["name"],
                "raw_shape": list(self.raw_shape), "shape": list(self.shape),
                "trajectories": N, "time_steps": T, "spatial_dims": dims, "grid": [D, H, W],
                "field_names": fields, "component_names": comps}
