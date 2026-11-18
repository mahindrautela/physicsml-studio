"""Adapter registry. Every module in this folder is imported at startup, so a
new adapter only has to live here and use ``@register("name")``."""
from __future__ import annotations

import importlib
import pkgutil
import traceback

from .base import Adapter, IncompatibleData  # noqa: F401

REGISTRY: dict[str, type[Adapter]] = {}
LOAD_ERRORS: dict[str, str] = {}


def register(name: str):
    def deco(cls: type[Adapter]) -> type[Adapter]:
        REGISTRY[name] = cls
        cls.adapter_name = name
        return cls
    return deco


def _discover() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name in ("base", "template") or mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{mod.name}")
        except Exception:  # noqa: BLE001 - one broken adapter must not take the app down
            LOAD_ERRORS[mod.name] = traceback.format_exc(limit=3)


_discover()
