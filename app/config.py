"""Where PhysicsML Studio keeps its files.

Everything can be moved with environment variables, so the same code works on a
laptop, a lab server, or a cluster node:

    STUDIO_HOME     working folder for downloads and run results (default: ./workspace)
    STUDIO_CATALOG  folder with the model and dataset cards (default: ./catalog)
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CATALOG_DIR = Path(os.environ.get("STUDIO_CATALOG", PROJECT_DIR / "catalog")).resolve()
HOME_DIR = Path(os.environ.get("STUDIO_HOME", PROJECT_DIR / "workspace")).resolve()
CACHE_DIR = HOME_DIR / "cache"
RUNS_DIR = HOME_DIR / "runs"
STATIC_DIR = PROJECT_DIR / "static"

# Cards can write "$STUDIO_HOME/..." for files that live in the working folder (e.g. the demo data).
os.environ.setdefault("STUDIO_HOME", str(HOME_DIR))

for _d in (CATALOG_DIR / "models", CATALOG_DIR / "datasets", CACHE_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def resolve_local(path: str) -> Path:
    """Local paths in cards may be absolute, relative to the project folder, or use $STUDIO_HOME."""
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()
