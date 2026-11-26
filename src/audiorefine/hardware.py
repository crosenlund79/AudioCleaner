"""Hardware detection and speech-model loading with automatic GPU/CPU selection.

GPU acceleration is used only when an NVIDIA GPU is present, the CUDA runtime libraries have
been installed (downloaded automatically on first use), the model loads on the GPU *and* a
short test transcription succeeds. Any failure falls back to the CPU. The user never chooses.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("audiorefine.hardware")


@dataclass
class NvidiaGpu:
    name: str
    memory_mb: int
    driver: str
    compute_cap: str


@dataclass
class HardwareInfo:
    os: str = ""
    cpu: str = ""
    logical_cpus: int = 1
    ram_gb: float = 0.0
    nvidia: list[NvidiaGpu] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _ram_gb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2**30
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 2**20
    except Exception:
        pass
    return 0.0


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        for c in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
                  Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")):
            if c.is_file():
                exe = str(c)
                break
    return exe


def detect() -> HardwareInfo:
    info = HardwareInfo(os=f"{platform.system()} {platform.release()} ({platform.version()})", cpu=_cpu_name(),
                        logical_cpus=os.cpu_count() or 1, ram_gb=round(_ram_gb(), 1))
    smi = _nvidia_smi()
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20,
                                 creationflags=0x08000000 if sys.platform == "win32" else 0)
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    try:
                        mem = int(float(parts[1]))
                    except ValueError:
                        mem = 0
                    info.nvidia.append(NvidiaGpu(parts[0], mem, parts[2], parts[3]))
        except Exception as e:
            log.info("nvidia-smi query failed: %s", e)
    return info


def gpu_candidate(hw: HardwareInfo) -> NvidiaGpu | None:
    """An NVIDIA GPU that could run Whisper large-v3 (>= 4 GB, Turing or newer)."""
    if sys.platform != "win32" and not os.environ.get("AUDIOREFINE_ALLOW_LINUX_GPU"):
        return None
    best = None
    for g in hw.nvidia:
        try:
            cc = float(g.compute_cap)
        except ValueError:
            cc = 0.0
        if g.memory_mb >= 4000 and cc >= 7.5 and (best is None or g.memory_mb > best.memory_mb):
            best = g
    return best


def cpu_threads() -> int:
    n = os.cpu_count() or 2
    # CTranslate2 scales with physical cores; hyper-threads add little.
    return max(1, n // 2 if n >= 8 else n)


@dataclass
class LoadedModel:
    model: object
    device: str
    compute_type: str
    description: str
    load_seconds: float
    gpu_error: str = ""


def _add_gpu_dll_dir(d: Path) -> None:
    if sys.platform == "win32":
        os.add_dll_directory(str(d))
    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def load_model(model_dir: Path, hw: HardwareInfo, allow_gpu: bool = True, gpu_runtime: Path | None = None) -> LoadedModel:
    from faster_whisper import WhisperModel

    gpu = gpu_candidate(hw) if allow_gpu else None
    gpu_error = ""
    if gpu is not None and gpu_runtime is not None and gpu_runtime.is_dir():
        compute = "float16" if gpu.memory_mb >= 6000 else "int8_float16"
        t0 = time.time()
        try:
            _add_gpu_dll_dir(gpu_runtime)
            import ctranslate2

            if ctranslate2.get_cuda_device_count() < 1:
                raise RuntimeError("CUDA reports no usable device (driver too old?)")
            m = WhisperModel(str(model_dir), device="cuda", compute_type=compute)
            # smoke test: a short decode exercises cuBLAS/cuDNN kernels
            noise = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.01).astype(np.float32)
            list(m.transcribe(noise, language="da", beam_size=1, without_timestamps=True)[0])
            return LoadedModel(m, "cuda", compute, f"NVIDIA {gpu.name} ({gpu.memory_mb} MB, {compute})",
                               time.time() - t0)
        except Exception as e:
            gpu_error = f"{type(e).__name__}: {e}"
            log.warning("GPU acceleration unavailable, using CPU: %s", gpu_error)
    elif gpu is not None:
        gpu_error = "GPU runtime libraries not installed"
    t0 = time.time()
    threads = cpu_threads()
    m = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=threads)
    return LoadedModel(m, "cpu", "int8", f"CPU ({hw.cpu or 'unknown'}, {threads} threads, int8)", time.time() - t0,
                       gpu_error=gpu_error)
