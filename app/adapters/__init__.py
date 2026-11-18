"""Model types ("adapters") and the registry that finds them.

Every module in this folder is imported at startup. A module adds a model type
by decorating its class with ``@register("name")``; model cards then refer to
it with ``adapter: name``. Name the file after the model type (my_model.py ->
"my_model") so a load error can be shown on the right cards.
Files starting with "_" are helpers, and template.py is only an example.
"""
from __future__ import annotations

import importlib
import pkgutil
import traceback

from .base import Adapter, IncompatibleData  # noqa: F401 - re-exported for the rest of the app

REGISTRY: dict[str, type[Adapter]] = {}
LOAD_ERRORS: dict[str, str] = {}


def register(name: str):
    def deco(cls: type[Adapter]) -> type[Adapter]:
        REGISTRY[name] = cls
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
