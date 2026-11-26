"""Runs the AudioRefine engine as a child process and turns its JSON lines into Qt signals.

Running the heavy work in a separate process keeps the window responsive, makes cancellation
immediate and safe, and protects the window from crashes in native libraries (GPU drivers,
decoders). If the engine dies, the window reports it and nothing is lost: outputs are staged
and originals are only moved at the very end.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from ..paths import app_dir, is_frozen


def engine_command() -> tuple[str, list[str]]:
    if is_frozen():
        exe = Path(sys.executable).with_name("AudioRefineEngine.exe" if sys.platform == "win32" else "AudioRefineEngine")
        return str(exe), []
    return sys.executable, ["-m", "audiorefine.cli"]


class EngineJob(QObject):
    event = Signal(dict)
    finished = Signal(int, dict)        # exit code, last "result" event (may be empty)
    crashed = Signal(str)

    def __init__(self, args: list[str], parent=None):
        super().__init__(parent)
        self.args = args
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        env = QProcessEnvironment.systemEnvironment()
        if not is_frozen():
            src = str(app_dir() / "src")
            env.insert("PYTHONPATH", src + (";" if sys.platform == "win32" else ":") + env.value("PYTHONPATH", ""))
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.readyReadStandardError.connect(self._read_err)
        self.proc.finished.connect(self._done)
        self.proc.errorOccurred.connect(self._error)
        self._buf = b""
        self._stderr = b""
        self.result: dict = {}
        self._kill_timer: QTimer | None = None

    def start(self):
        prog, pre = engine_command()
        self.proc.start(prog, pre + ["--json"] + self.args)

    def running(self) -> bool:
        return self.proc.state() != QProcess.ProcessState.NotRunning

    def cancel(self, grace_ms: int = 30000):
        if not self.running():
            return
        try:
            self.proc.write(b"cancel\n")
        except Exception:
            pass
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self.kill)
        self._kill_timer.start(grace_ms)

    def kill(self):
        if self.running():
            self.proc.kill()

    def _read(self):
        self._buf += bytes(self.proc.readAllStandardOutput())
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                self.result = ev
            self.event.emit(ev)

    def _read_err(self):
        self._stderr = (self._stderr + bytes(self.proc.readAllStandardError()))[-20000:]

    def _done(self, code, status):
        self._read()
        if self._kill_timer:
            self._kill_timer.stop()
        if status == QProcess.ExitStatus.CrashExit and not self.result:
            self.crashed.emit(self._stderr.decode("utf-8", "replace")[-2000:])
        self.finished.emit(int(code), self.result)

    def _error(self, err):
        if err == QProcess.ProcessError.FailedToStart:
            self.crashed.emit("The AudioRefine engine could not be started. The installation may be incomplete "
                              "or blocked by security software.")
            self.finished.emit(-1, {})
