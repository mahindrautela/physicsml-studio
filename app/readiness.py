"""Is a model or dataset ready to run? Shown as the status dot next to each card.

States:
  ready          everything is on this machine
  download       files will be downloaded on the first run
  needs_install  a Python package from the card's `requires` is missing
  error          something must be fixed first (e.g. a local file doesn't exist)
"""
from __future__ import annotations

import importlib.util

from .adapters import LOAD_ERRORS, REGISTRY
from .catalog import DatasetCard, ModelCard
from .sources import cached_path


def model_status(card: ModelCard) -> dict:
    if card.adapter not in REGISTRY:
        detail = LOAD_ERRORS.get(card.adapter)
        return _status("error", f"Model type '{card.adapter}' isn't installed" + (f": {detail}" if detail else ""))
    missing = [pkg for pkg in card.requires if importlib.util.find_spec(pkg) is None]
    if missing:
        return _status("needs_install", card.install_hint or f"pip install {' '.join(missing)}")
    if not REGISTRY[card.adapter].needs_weights:
        return _status("ready", "No download needed")
    return _file_status(card, "Weights")


def dataset_status(card: DatasetCard) -> dict:
    return _file_status(card, "File")


def _file_status(card, what: str) -> dict:
    if cached_path(card.source) is not None:
        return _status("ready", f"{what} available on this machine")
    if card.source.type == "local":
        return _status("error", f"{what} not found: {card.source.path}")
    return _status("download", f"{what} download on first run")


def _status(state: str, message: str) -> dict:
    return {"state": state, "message": message}
