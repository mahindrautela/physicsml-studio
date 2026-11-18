"""Start PhysicsML Studio.

    python studio.py                 # http://127.0.0.1:8000, opens your browser
    python studio.py --port 8080 --no-browser
    python studio.py --skip-demo     # don't create the demo datasets
    python studio.py --host 0.0.0.0  # let other machines on your network connect

On a remote machine or cluster node, keep the default host and use an SSH tunnel:
    ssh -L 8000:localhost:8000 you@server   then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Packages the server can't start without (import name -> pip name).
REQUIRED = {"fastapi": "fastapi", "uvicorn": "uvicorn[standard]", "yaml": "pyyaml", "numpy": "numpy",
            "httpx": "httpx", "huggingface_hub": "huggingface_hub", "netCDF4": "netCDF4", "h5py": "h5py"}


def check_packages() -> None:
    """Stop early with the exact install command, instead of failing after the demo step."""
    missing = [pip for name, pip in REQUIRED.items() if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit("Some packages are missing from this Python environment:\n"
                 f"    {', '.join(missing)}\n"
                 "Install them with:\n"
                 "    pip install -r requirements.txt\n"
                 f"(this Python is {sys.executable})")


def make_demo_data(home: Path) -> None:
    demo = home / "demo"
    if (demo / "kolmogorov2d.npy").exists() and (demo / "burgers1d.npy").exists():
        return
    print("First start: simulating the demo datasets (10-60 seconds)...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo_data.py"), "--out", str(demo)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="PhysicsML Studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    ap.add_argument("--skip-demo", action="store_true", help="Don't create the demo datasets")
    args = ap.parse_args()

    check_packages()
    if not args.skip_demo:
        make_demo_data(Path(os.environ.get("STUDIO_HOME", ROOT / "workspace")))

    import uvicorn

    url = f"http://127.0.0.1:{args.port}" if args.host in ("127.0.0.1", "0.0.0.0") else f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"PhysicsML Studio is running at {url}  (press Ctrl+C to stop)", flush=True)
    sys.path.insert(0, str(ROOT))
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
