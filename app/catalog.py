"""Model cards and dataset cards.

A *card* is a small YAML file in ``catalog/models`` or ``catalog/datasets`` that
says what a model or dataset is, where its files live (Hugging Face, GitHub, a
URL, or a local path) and how to read it. The app lists whatever cards it finds,
so adding a model is just adding a file (or using the "Add model" form).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .config import CATALOG_DIR

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class Link(BaseModel):
    label: str
    url: str


class Source(BaseModel):
    """Where a file comes from."""

    type: Literal["huggingface", "github", "url", "local", "none"] = "none"
    repo_id: Optional[str] = None  # huggingface: "owner/name"; github: "owner/name"
    repo_type: Literal["model", "dataset"] = "model"  # huggingface only
    filename: Optional[str] = None  # file inside the repo (github: path inside the repo)
    subfolder: Optional[str] = None  # huggingface only
    revision: Optional[str] = None  # branch, tag or commit
    url: Optional[str] = None  # url only
    path: Optional[str] = None  # local only

    @model_validator(mode="after")
    def _check(self):
        need = {
            "huggingface": ["repo_id", "filename"],
            "github": ["repo_id", "filename"],
            "url": ["url"],
            "local": ["path"],
            "none": [],
        }[self.type]
        missing = [k for k in need if not getattr(self, k)]
        if missing:
            raise ValueError(f"source type '{self.type}' needs: {', '.join(missing)}")
        return self

    def describe(self) -> str:
        if self.type == "huggingface":
            sub = f"{self.subfolder}/" if self.subfolder else ""
            return f"Hugging Face: {self.repo_id}, {sub}{self.filename}"
        if self.type == "github":
            return f"GitHub: {self.repo_id} ({self.revision or 'default branch'}), {self.filename}"
        if self.type == "url":
            return self.url or ""
        if self.type == "local":
            return f"Local file: {self.path}"
        return "Built in (no files)"

    def web_url(self) -> Optional[str]:
        if self.type == "huggingface":
            base = "https://huggingface.co/" + ("datasets/" if self.repo_type == "dataset" else "")
            return f"{base}{self.repo_id}"
        if self.type == "github":
            return f"https://github.com/{self.repo_id}"
        if self.type == "url":
            return self.url
        return None


class ModelCard(BaseModel):
    id: str
    name: str
    adapter: str = Field(description="Which model type runs this model, e.g. 'morph' or 'pytorch'")
    family: Optional[str] = None
    description: str = ""
    source: Source = Source()
    config: dict[str, Any] = {}
    requires: list[str] = Field(default_factory=list, description="Python packages that must be importable")
    install_hint: Optional[str] = None
    links: list[Link] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    order: int = Field(100, description="Lower numbers are listed first")

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not ID_RE.match(v):
            raise ValueError("id must be lowercase letters, digits, '-', '_' or '.' (max 64)")
        return v


class Layout(BaseModel):
    """How the stored array maps onto (N, T, F, C, D, H, W).

    N trajectories, T time steps, F fields, C components per field, and a
    D x H x W grid (1D data: D=H=1; 2D data: D=1). This is MORPH's "UPTF-7" layout.

    Two ways to describe a file:
      * ``axes`` (recommended): the letters of the stored array's axes in order,
        e.g. "NTFHW" for (traj, time, field, y, x) or "NTHWF" for channels-last.
        Missing letters become size 1.
      * the sizes below: the array is reshaped (C-order) to (N, T, F, C, D, H, W),
        exactly like MORPH's ``--dataset_specs F C D H W``.
    """

    axes: Optional[str] = None
    fields: int = 1
    components: int = 1
    depth: int = 1
    height: int = 1
    width: int = 1

    @field_validator("axes")
    @classmethod
    def _axes(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        v = v.replace(" ", "").upper()
        if not set(v) <= set("NTFCDHW") or len(set(v)) != len(v) or not v.startswith("NT"):
            raise ValueError("axes must start with 'NT' and use each of N,T,F,C,D,H,W at most once")
        return v


class EvalSplit(BaseModel):
    """Which trajectories are used for evaluation."""

    mode: Literal["last_fraction", "first_n", "all", "indices"] = "last_fraction"
    fraction: float = 0.1
    n: int = 10
    indices: list[int] = Field(default_factory=list)


class DatasetCard(BaseModel):
    id: str
    name: str
    description: str = ""
    source: Source
    format: Literal["auto", "npy", "nc", "h5"] = "auto"
    variable: Optional[str] = Field(None, description="Variable inside a .nc/.h5 file; empty = the largest array")
    layout: Layout = Layout()
    field_names: list[str] = Field(default_factory=list)
    component_names: list[str] = Field(default_factory=list)
    eval: EvalSplit = EvalSplit()
    links: list[Link] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    order: int = Field(100, description="Lower numbers are listed first")

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not ID_RE.match(v):
            raise ValueError("id must be lowercase letters, digits, '-', '_' or '.' (max 64)")
        return v


KINDS = {"models": ModelCard, "datasets": DatasetCard}


def _dir(kind: str) -> Path:
    return CATALOG_DIR / kind


def load_all(kind: str) -> tuple[list[BaseModel], list[dict]]:
    """Return (cards, problems). A broken YAML file is reported, not fatal."""
    cls = KINDS[kind]
    cards, problems = [], []
    for f in sorted(_dir(kind).glob("*.y*ml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            cards.append(cls(**data))
        except Exception as e:  # noqa: BLE001 - show the user what is wrong with the file
            problems.append({"file": f.name, "error": str(e)})
    cards.sort(key=lambda c: (c.order, c.name.lower()))
    return cards, problems


def get(kind: str, card_id: str):
    for c in load_all(kind)[0]:
        if c.id == card_id:
            return c
    raise KeyError(f"No {kind[:-1]} with id '{card_id}'")


def save(kind: str, card: BaseModel, overwrite: bool = False) -> Path:
    path = _dir(kind) / f"{card.id}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"A {kind[:-1]} with id '{card.id}' already exists")
    data = card.model_dump(exclude_none=True, exclude_defaults=False)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def delete(kind: str, card_id: str) -> None:
    for f in _dir(kind).glob("*.y*ml"):
        try:
            if (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("id") == card_id:
                f.unlink()
                return
        except Exception:  # noqa: BLE001
            continue
    raise KeyError(f"No {kind[:-1]} with id '{card_id}'")
