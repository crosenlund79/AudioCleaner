# PyInstaller spec: one portable folder with two executables sharing the same runtime.
#   AudioRefine.exe        - the desktop window (no console)
#   AudioRefineEngine.exe  - the processing engine / command line (console subsystem)
# Build:  pyinstaller packaging/audiorefine.spec --noconfirm
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
exe_suffix = ".exe" if sys.platform == "win32" else ""
arfilter = ROOT / "native" / "arfilter" / "target" / "release" / f"arfilter{exe_suffix}"
if not arfilter.is_file():
    raise SystemExit(f"arfilter helper not built: {arfilter}")

datas = []
datas += collect_data_files("faster_whisper")          # Silero VAD model
datas += collect_data_files("av")
datas += collect_data_files("soundfile")
datas += collect_data_files("onnxruntime", includes=["**/*.dll", "**/*.so*", "**/*.pyd"])
datas += [(str(ROOT / "assets" / "audiorefine.ico"), "assets"),
          (str(ROOT / "assets" / "audiorefine.png"), "assets"),
          (str(ROOT / "THIRD_PARTY_NOTICES.txt"), "."),
          (str(ROOT / "LICENSE"), ".")]
binaries = [(str(arfilter), "bin")]
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("av")
binaries += collect_dynamic_libs("soundfile")

hidden = ["audiorefine.cli", "audiorefine.gui.app", "truststore"]
hidden += collect_submodules("audiorefine")
hidden += collect_submodules("faster_whisper")
hidden += collect_submodules("av")

excludes = ["tkinter", "matplotlib", "scipy", "pandas", "torch", "IPython", "pytest", "jiwer", "PIL", "pyarrow",
            "hf_xet", "datasets", "rapidfuzz", "pystoi", "sympy",
            "PySide6.QtNetwork", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtWebEngineCore",
            "PySide6.QtMultimedia", "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
            "PySide6.QtPdf", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtOpenGL", "PySide6.QtSvg"]

a = Analysis(
    [str(ROOT / "packaging" / "gui_entry.py"), str(ROOT / "packaging" / "engine_entry.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

gui_script = [s for s in a.scripts if s[0] == "gui_entry"]
eng_script = [s for s in a.scripts if s[0] == "engine_entry"]
common = [s for s in a.scripts if s[0] not in ("gui_entry", "engine_entry")]

version_file = str(ROOT / "packaging" / "version_info.txt") if sys.platform == "win32" else None
icon = str(ROOT / "assets" / "audiorefine.ico")

gui = EXE(pyz, common + gui_script, [], exclude_binaries=True, name="AudioRefine", console=False,
          icon=icon, version=version_file, upx=False)
engine = EXE(pyz, common + eng_script, [], exclude_binaries=True, name="AudioRefineEngine", console=True,
             icon=icon, version=version_file, upx=False)
coll = COLLECT(gui, engine, a.binaries, a.datas, name="AudioRefine", upx=False)
