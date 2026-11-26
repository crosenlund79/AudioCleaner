"""Resumable, verified downloads of the speech model and (optionally) NVIDIA GPU runtime libraries.

* Downloads go to ``<name>.part`` files and resume with HTTP Range requests after interruption.
* Every file is verified against a pinned SHA-256 before it is moved into place.
* Installed components are marked with a ``verified.json`` so later starts only check sizes.
* Nothing is downloaded during normal processing; this runs only when a component is missing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import __version__
from .constants import ModelSpec
from .paths import gpu_runtime_dir, models_dir

log = logging.getLogger("audiorefine.downloads")

CHUNK = 1 << 20
USER_AGENT = f"AudioRefine/{__version__}"


class DownloadError(Exception):
    pass


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class RemoteFile:
    url: str
    size: int
    sha256: str
    name: str


# NVIDIA CUDA 12 runtime libraries needed by CTranslate2 on Windows (redistributed by NVIDIA on PyPI).
GPU_WHEELS = (
    RemoteFile(
        "https://files.pythonhosted.org/packages/70/61/7d7b3c70186fb651d0fbd35b01dbfc8e755f69fd58f817f3d0f642df20c3/nvidia_cublas_cu12-12.8.4.1-py3-none-win_amd64.whl",
        567544208, "47e9b82132fa8d2b4944e708049229601448aaad7e6f296f630f2d1a32de35af", "nvidia_cublas_cu12-12.8.4.1-py3-none-win_amd64.whl"),
    RemoteFile(
        "https://files.pythonhosted.org/packages/3d/90/0bd6e586701b3a890fd38aa71c387dab4883d619d6e5ad912ccbd05bfd67/nvidia_cudnn_cu12-9.10.2.21-py3-none-win_amd64.whl",
        692992268, "c6288de7d63e6cf62988f0923f96dc339cea362decb1bf5b3141883392a7d65e", "nvidia_cudnn_cu12-9.10.2.21-py3-none-win_amd64.whl"),
)
GPU_RUNTIME_ID = "cublas-12.8.4.1_cudnn-9.10.2.21"


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:  # use the operating system trust store where available (corporate proxies etc.)
        import truststore  # type: ignore

        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass
    return ctx


def sha256_file(path: Path, progress: Callable[[int], None] | None = None, cancel: Callable[[], None] | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK * 4)
            if not b:
                break
            h.update(b)
            if progress:
                progress(len(b))
            if cancel:
                cancel()
    return h.hexdigest()


def download_file(rf: RemoteFile, dest: Path, progress: Callable[[int], None], cancel: Callable[[], None] | None = None,
                  retries: int = 6) -> None:
    """Download ``rf`` to ``dest`` (resuming ``dest.part``), verifying size and SHA-256.

    ``progress`` receives the number of bytes of this file present so far.
    """
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_context()
    attempt = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have > rf.size:
            part.unlink()
            have = 0
        if have < rf.size:
            req = urllib.request.Request(rf.url, headers={"User-Agent": USER_AGENT})
            if have:
                req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                    status = getattr(r, "status", 200)
                    mode = "ab"
                    if have and status != 206:  # server ignored Range: restart
                        mode, have = "wb", 0
                    with open(part, mode) as f:
                        got = have
                        progress(got)
                        while True:
                            if cancel:
                                cancel()
                            b = r.read(CHUNK)
                            if not b:
                                break
                            f.write(b)
                            got += len(b)
                            progress(got)
            except Cancelled:
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                attempt += 1
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                    raise DownloadError(f"The download server refused the request ({e.code}).") from e
                if attempt > retries:
                    raise DownloadError(f"Download failed after {retries} retries: {e}") from e
                wait = min(30, 2 ** attempt)
                log.warning("Download interrupted (%s); retrying in %ds", e, wait)
                for _ in range(wait * 10):
                    if cancel:
                        cancel()
                    time.sleep(0.1)
                continue
        size = part.stat().st_size
        if size != rf.size:
            attempt += 1
            if attempt > retries:
                raise DownloadError(f"{rf.name}: incomplete download ({size} of {rf.size} bytes).")
            continue
        digest = sha256_file(part, cancel=cancel)
        if digest != rf.sha256:
            part.unlink(missing_ok=True)
            attempt += 1
            if attempt > retries:
                raise DownloadError(f"{rf.name}: checksum mismatch; the downloaded file is corrupt.")
            log.warning("Checksum mismatch for %s; downloading again", rf.name)
            continue
        os.replace(part, dest)
        return


# ------------------------------------------------------------------------------ speech model
def model_path(spec: ModelSpec) -> Path:
    return models_dir() / spec.key / spec.revision[:12]


def model_installed(spec: ModelSpec) -> bool:
    d = model_path(spec)
    marker = d / "verified.json"
    if not marker.is_file():
        return False
    for f in spec.files:
        p = d / f.name
        if not p.is_file() or p.stat().st_size != f.size:
            return False
    return True


def missing_model_bytes(spec: ModelSpec) -> int:
    d = model_path(spec)
    total = 0
    for f in spec.files:
        p = d / f.name
        if p.is_file() and p.stat().st_size == f.size:
            continue
        part = p.with_name(p.name + ".part")
        total += f.size - (part.stat().st_size if part.exists() else 0)
    return max(0, total)


def install_model(spec: ModelSpec, progress: Callable[[int, int], None], cancel: Callable[[], None] | None = None) -> Path:
    d = model_path(spec)
    d.mkdir(parents=True, exist_ok=True)
    total = spec.total_size
    done = 0
    for f in spec.files:
        p = d / f.name
        if not (p.is_file() and p.stat().st_size == f.size):
            download_file(RemoteFile(spec.url(f.name), f.size, f.sha256, f.name), p,
                          lambda n, base=done: progress(base + n, total), cancel)
        done += f.size
        progress(done, total)
    # full verification of the installed set
    for f in spec.files:
        if sha256_file(d / f.name, cancel=cancel) != f.sha256:
            (d / f.name).unlink(missing_ok=True)
            raise DownloadError(f"Model file {f.name} failed verification; please retry.")
    (d / "verified.json").write_text(json.dumps({"repo": spec.repo, "revision": spec.revision,
                                                 "files": {f.name: f.sha256 for f in spec.files}}, indent=1))
    return d


# ------------------------------------------------------------------------------ GPU runtime
def gpu_runtime_path() -> Path:
    return gpu_runtime_dir() / GPU_RUNTIME_ID


def gpu_runtime_installed() -> bool:
    d = gpu_runtime_path()
    return (d / "verified.json").is_file() and (d / "cublas64_12.dll").is_file() and (d / "cudnn64_9.dll").is_file()


def gpu_runtime_bytes() -> int:
    return sum(w.size for w in GPU_WHEELS)


def install_gpu_runtime(progress: Callable[[int, int], None], cancel: Callable[[], None] | None = None) -> Path:
    d = gpu_runtime_path()
    dl = gpu_runtime_dir() / "downloads"
    dl.mkdir(parents=True, exist_ok=True)
    total = gpu_runtime_bytes()
    done = 0
    for w in GPU_WHEELS:
        dest = dl / w.name
        if not (dest.is_file() and dest.stat().st_size == w.size and sha256_file(dest) == w.sha256):
            download_file(w, dest, lambda n, base=done: progress(base + n, total), cancel)
        done += w.size
        progress(done, total)
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    names = []
    for w in GPU_WHEELS:
        with zipfile.ZipFile(dl / w.name) as z:
            for info in z.infolist():
                n = info.filename.replace("\\", "/")
                if n.lower().endswith(".dll") and "/bin/" in n:
                    target = tmp / Path(n).name
                    with z.open(info) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    names.append(target.name)
    (tmp / "verified.json").write_text(json.dumps({"id": GPU_RUNTIME_ID, "dlls": sorted(names)}, indent=1))
    shutil.rmtree(d, ignore_errors=True)
    os.replace(tmp, d)
    shutil.rmtree(dl, ignore_errors=True)
    return d
