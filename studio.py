"""Start PhysicsML Studio.

    python studio.py                 # http://127.0.0.1:8000, opens your browser
    python studio.py --port 8080 --no-browser
    python studio.py --host 0.0.0.0  # let other machines on your network connect

On a remote machine or cluster node, keep the default host and use an SSH tunnel:
    ssh -L 8000:localhost:8000 you@server   then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="PhysicsML Studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-demo", action="store_true", help="Don't create the demo datasets")
    a = ap.parse_args()

    home = Path(os.environ.get("STUDIO_HOME", ROOT / "workspace"))
    demo = home / "demo"
    if not a.skip_demo and not ((demo / "kolmogorov2d.npy").exists() and (demo / "burgers1d.npy").exists()):
        print("First start: simulating the demo datasets (about 30 seconds)…", flush=True)
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo_data.py"), "--out", str(demo)], check=True)

    import uvicorn

    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"PhysicsML Studio is running at {url}  (Ctrl+C to stop)", flush=True)
    sys.path.insert(0, str(ROOT))
    uvicorn.run("app.main:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
