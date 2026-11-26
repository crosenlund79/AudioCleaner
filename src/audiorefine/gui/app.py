"""Entry point of the desktop application.

``AudioRefine.exe --screenshot out.png [--folder DIR]`` opens the window, optionally scans a
folder, saves a screenshot and exits; used by the Windows build to verify the packaged GUI.
"""
from __future__ import annotations

import sys
import time


def _arg(name: str) -> str | None:
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def main() -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from .. import APP_NAME

    if sys.platform == "win32":
        try:  # own taskbar icon/grouping instead of python.exe
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AudioRefine.AudioRefine")
        except Exception:
            pass
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    from .main_window import MainWindow

    w = MainWindow()
    w.show()
    shot = _arg("--screenshot")
    if shot:
        from pathlib import Path

        folder = _arg("--folder")
        t0 = time.time()
        state = {"folder_set": False}

        def poll():
            if folder and not state["folder_set"] and w.status is not None:
                state["folder_set"] = True
                w.set_folder(Path(folder))
            done = w.status is not None and (not folder or (state["folder_set"] and w.scan is not None and not w.busy()))
            if done or time.time() - t0 > 60:
                app.processEvents()
                ok = w.grab().save(shot)
                print(f"screenshot saved: {ok}; status={'ok' if w.status else 'missing'}; "
                      f"files={len((w.scan or {}).get('sources', []))}", flush=True)
                app.exit(0 if ok and w.status is not None else 1)
            else:
                QTimer.singleShot(250, poll)

        QTimer.singleShot(500, poll)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
