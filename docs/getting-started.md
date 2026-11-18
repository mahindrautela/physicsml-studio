# Getting started

## Install

You need **Python 3.10 or newer** and **git**. Get the code:

```bash
git clone https://github.com/mahindrautela/physicsml-studio.git
cd physicsml-studio
```

Then install the packages into a venv or a conda environment.

**With a venv** (macOS, Linux):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**With conda** (for example in the Anaconda Prompt on Windows):

```bash
conda create -n physicsml python=3.11
conda activate physicsml
pip install -r requirements.txt
```

Install into the same environment you start the app from. If `python studio.py` reports
missing packages, the environment that's active isn't the one you installed into.

`requirements.txt` also installs MORPH's code from GitHub (the `morph-pde` line, which is why
git is needed). If you don't use MORPH, or GitHub is blocked, delete that line.

Don't reuse the `environment.yml` from the MORPH repo. It pins Python 3.8, which is too old.

### NVIDIA GPU

`pip install torch` gives a build that may not match your CUDA version. Install the matching
PyTorch build **before** `requirements.txt`. For example, for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Pick the right command for your system at https://pytorch.org/get-started/locally/.
The app uses the GPU automatically. The top bar shows which GPUs PyTorch found.

## Start

```bash
python studio.py
```

The first start simulates the two demo datasets, which takes 10 to 60 seconds and shows its
progress. Then the window prints `PhysicsML Studio is running at http://127.0.0.1:8000`, and
your browser opens there. Keep the window open while you use the app; press Ctrl+C in it to stop.

| Option | What it does |
|---|---|
| `--port 8080` | use another port |
| `--no-browser` | don't open a browser tab |
| `--host 0.0.0.0` | let other machines on your network connect. There's no login, so only do this on a trusted network. |
| `--skip-demo` | don't create the demo datasets |

The page is tested in Chrome. It uses only standard web features, which current Firefox, Edge
and Safari also support.

## On a remote server or cluster node

Start the app on the remote machine with the default host. Then forward the port from your
laptop over SSH, and open http://127.0.0.1:8000 on the laptop:

```bash
ssh -L 8000:localhost:8000 you@server
ssh -J you@login-node -L 8000:localhost:8000 you@compute-node    # app on a cluster compute node
```

If the page doesn't open, see [Troubleshooting](troubleshooting.md#the-page-doesnt-open).

## Where files go

| Folder | Contents | Change it with |
|---|---|---|
| `catalog/` | model and dataset cards (YAML) | `STUDIO_CATALOG=/path` |
| `workspace/cache/` | downloaded weights and data | `STUDIO_HOME=/path` |
| `workspace/runs/` | results of every run | `STUDIO_HOME=/path` |
| `workspace/demo/` | the simulated demo datasets | `STUDIO_HOME=/path` |

Set them before starting: `export STUDIO_HOME=/scratch/me/studio` on macOS and Linux, or
`set STUDIO_HOME=D:\studio` in the Windows command prompt. Point `STUDIO_HOME` at scratch
space on a cluster. Point `STUDIO_CATALOG` at a shared folder so a team sees the same models
and datasets.

For private repositories, set `HF_TOKEN` (Hugging Face) or `GITHUB_TOKEN` (GitHub) before you start.
