"""Fetch model weights and data files from Hugging Face, GitHub, a URL or disk.

Downloads land in ``workspace/cache`` and are reused on later runs.
Private repositories work if you set ``HF_TOKEN`` / ``GITHUB_TOKEN`` before
starting the server.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from .catalog import Source
from .config import CACHE_DIR, resolve_local

Progress = Callable[[str, Optional[float]], None]  # (message, fraction 0..1 or None)

LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def _noop(msg: str, frac: Optional[float] = None) -> None:  # pragma: no cover
    pass


def _github_headers() -> dict:
    tok = os.environ.get("GITHUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _download(url: str, dest: Path, progress: Progress, headers: Optional[dict] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, headers=headers or {}, follow_redirects=True, timeout=60) as r:
        if r.status_code == 404:
            raise FileNotFoundError(f"Not found (404): {url}")
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(part, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                progress(f"Downloading {dest.name} ({done / 1e6:.0f} MB)", done / total if total else None)
    part.replace(dest)
    return dest


def cached_path(src: Source) -> Optional[Path]:
    """Path of the file if it is already available locally, else None."""
    if src.type == "none":
        return None
    if src.type == "local":
        p = resolve_local(src.path or "")
        return p if p.exists() else None
    if src.type == "huggingface":
        try:
            from huggingface_hub import try_to_load_from_cache

            fn = f"{src.subfolder}/{src.filename}" if src.subfolder else src.filename
            p = try_to_load_from_cache(src.repo_id, fn, cache_dir=CACHE_DIR / "hf",
                                       revision=src.revision, repo_type=src.repo_type)
            return Path(p) if isinstance(p, str) else None
        except Exception:  # noqa: BLE001
            return None
    p = _target(src)
    return p if p.exists() else None


def _target(src: Source) -> Path:
    if src.type == "github":
        return CACHE_DIR / "github" / src.repo_id / (src.revision or "HEAD") / src.filename
    if src.type == "url":
        h = hashlib.sha1(src.url.encode()).hexdigest()[:12]
        name = Path(urlparse(src.url).path).name or "download.bin"
        return CACHE_DIR / "url" / h / name
    raise ValueError(src.type)


def fetch(src: Source, progress: Progress = _noop) -> Optional[Path]:
    """Make the file available locally and return its path."""
    try:
        return _fetch(src, progress)
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001 - turn network errors into something actionable
        where = src.describe()
        hint = ""
        text = str(e)
        if "401" in text or "403" in text or "Unauthorized" in text or "Forbidden" in text:
            hint = (" Access was refused: check that the repo and file exist, that this machine can reach the site "
                    "(proxy/firewall), and set HF_TOKEN or GITHUB_TOKEN for private repos.")
        elif "404" in text or "EntryNotFound" in e.__class__.__name__ or "RepositoryNotFound" in e.__class__.__name__:
            hint = " The repo or file wasn't found: check the names and the branch/revision."
        elif isinstance(e, (httpx.ConnectError, httpx.TimeoutException)) or "Connect" in e.__class__.__name__:
            hint = " The site couldn't be reached: check the network connection or proxy settings."
        raise RuntimeError(f"Couldn't download {where}. {e.__class__.__name__}: {text[:200]}.{hint}") from e


def _fetch(src: Source, progress: Progress) -> Optional[Path]:
    if src.type == "none":
        return None
    hit = cached_path(src)
    if hit is not None:
        return hit
    if src.type == "local":
        raise FileNotFoundError(f"Local file not found: {resolve_local(src.path or '')}")
    if src.type == "huggingface":
        from huggingface_hub import hf_hub_download

        progress(f"Downloading {src.filename} from Hugging Face ({src.repo_id})", None)
        return Path(hf_hub_download(repo_id=src.repo_id, filename=src.filename, subfolder=src.subfolder,
                                    repo_type=src.repo_type, revision=src.revision,
                                    cache_dir=CACHE_DIR / "hf"))
    if src.type == "github":
        ref = src.revision or "HEAD"
        dest = _target(src)
        raw = f"https://raw.githubusercontent.com/{src.repo_id}/{ref}/{src.filename}"
        _download(raw, dest, progress, _github_headers())
        with open(dest, "rb") as fh:
            head = fh.read(len(LFS_POINTER))
        if head == LFS_POINTER:  # large file stored with Git LFS: fetch the real content
            media = f"https://media.githubusercontent.com/media/{src.repo_id}/{ref}/{src.filename}"
            _download(media, dest, progress, _github_headers())
        return dest
    if src.type == "url":
        return _download(src.url, _target(src), progress)
    raise ValueError(f"Unknown source type {src.type}")


_HF_RE = re.compile(r"^https?://huggingface\.co/(datasets/)?([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+)$")
_GH_RE = re.compile(r"^https?://github\.com/([^/]+/[^/]+)/(?:blob|raw|tree)/([^/]+)/(.+)$")
_GH_RAW_RE = re.compile(r"^https?://raw\.githubusercontent\.com/([^/]+/[^/]+)/([^/]+)/(.+)$")


def parse_link(link: str) -> Source:
    """Turn a pasted link into a Source.

    Understands Hugging Face file links, GitHub file links (blob/raw), raw
    GitHub links, plain http(s) URLs and local paths.
    """
    link = link.strip()
    m = _HF_RE.match(link)
    if m:
        is_ds, repo, rev, path = m.groups()
        sub, _, fn = path.rpartition("/")
        return Source(type="huggingface", repo_id=repo, repo_type="dataset" if is_ds else "model",
                      revision=None if rev == "main" else rev, subfolder=sub or None, filename=fn)
    m = _GH_RE.match(link) or _GH_RAW_RE.match(link)
    if m:
        repo, ref, path = m.groups()
        return Source(type="github", repo_id=repo, revision=ref, filename=path)
    if link.startswith(("http://", "https://")):
        return Source(type="url", url=link)
    return Source(type="local", path=link)
