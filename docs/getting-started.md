# Getting started

## Install

You need **Python 3.10 or newer** and **git**. A venv or a conda environment both work.

```bash
git clone https://github.com/mahindrautela/physicsml-studio.git
cd physicsml-studio
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` also installs MORPH's code from GitHub (the `morph-pde` line, which is why
git is needed). If you don't use MORPH, you can delete that line.

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

Your browser opens at http://localhost:8000. The first start takes about 30 seconds while the two
demo datasets are simulated.

| Option | What it does |
|---|---|
| `--port 8080` | use another port |
| `--no-browser` | don't open a browser tab |
| `--host 0.0.0.0` | let other machines on your network connect. There's no login, so only do this on a trusted network. |
| `--skip-demo` | don't create the demo datasets |

The page is tested in Chrome. It uses only standard web features, which current Firefox, Edge
and Safari also support.

## On a remote server or cluster node

Start it on the remote machine with the default host, then forward the port from your laptop:

```bash
ssh -L 8000:localhost:8000 you@server      # then open http://localhost:8000 on your laptop
```

On a cluster, run it inside an interactive job on a GPU node, and forward the port to that node.

## Where files go

| Folder | Contents | Change it with |
|---|---|---|
| `catalog/` | model and dataset cards (YAML) | `STUDIO_CATALOG=/path` |
| `workspace/cache/` | downloaded weights and data | `STUDIO_HOME=/path` |
| `workspace/runs/` | results of every run | `STUDIO_HOME=/path` |
| `workspace/demo/` | the simulated demo datasets | `STUDIO_HOME=/path` |

Point `STUDIO_HOME` at scratch space on a cluster. Point `STUDIO_CATALOG` at a shared folder so
a team sees the same models and datasets.

For private repositories, set `HF_TOKEN` (Hugging Face) or `GITHUB_TOKEN` (GitHub) before you start.
