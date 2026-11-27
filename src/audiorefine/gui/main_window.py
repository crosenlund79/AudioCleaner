"""The AudioRefine main window: one folder, one button."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..paths import bundle_dir, logs_dir, settings_file
from .engine_client import EngineJob

REFINED = "refined"


def fmt_hms(s: float | None) -> str:
    if s is None:
        return "–"
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt_rec(iso: str | None, estimated: bool | None) -> str:
    if not iso:
        return ""
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso).astimezone()
        return ("≈" if estimated else "") + dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def hours_span(hours: list[str]) -> str:
    """'2025-11-25 08' list -> '08:00–10:00' (or with dates when spanning days)."""
    if not hours:
        return ""
    a, b = hours[0], hours[-1]
    da, ha = a.split(" ")
    db, hb = b.split(" ")
    end = f"{(int(hb) + 1) % 24:02d}:00"
    if len(hours) == 1:
        return f"hour {ha}:00–{end}"
    if da == db:
        return f"hours {ha}:00–{end}"
    return f"{da[5:]} {ha}:00 – {db[5:]} {end}"


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


CODECS = {"mp3float": "MP3", "mp3": "MP3", "aac": "AAC", "opus": "Opus", "vorbis": "Vorbis", "flac": "FLAC",
          "alac": "ALAC", "wmav1": "WMA", "wmav2": "WMA", "wmapro": "WMA Pro", "wmavoice": "WMA Voice",
          "amrnb": "AMR-NB", "amrwb": "AMR-WB", "pcm_s16le": "PCM 16-bit", "pcm_s24le": "PCM 24-bit",
          "pcm_s32le": "PCM 32-bit", "pcm_f32le": "PCM float", "pcm_u8": "PCM 8-bit", "pcm_s16be": "PCM 16-bit",
          "pcm_mulaw": "µ-law", "pcm_alaw": "A-law", "adpcm_ima_wav": "ADPCM", "gsm_ms": "GSM"}


def codec_name(c: str) -> str:
    return CODECS.get(c, c.upper() if len(c) <= 5 else c)


STYLE = """
QWidget#central { background: palette(window); }
QLabel#title { font-size: 22px; font-weight: 600; }
QLabel#subtitle { color: palette(mid); }
QLabel#muted { color: #6b7280; }
QLabel#warn { color: #92400e; background: #fef3c7; border-radius: 6px; padding: 6px 8px; }
QPushButton#primary { background: #2563eb; color: white; border: none; border-radius: 6px;
                      padding: 10px 28px; font-size: 15px; font-weight: 600; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #93c5fd; color: #eff6ff; }
QFrame#banner_ok { background: #dcfce7; border: 1px solid #86efac; border-radius: 6px; }
QFrame#banner_err { background: #fee2e2; border: 1px solid #fca5a5; border-radius: 6px; }
QFrame#banner_info { background: #e0f2fe; border: 1px solid #7dd3fc; border-radius: 6px; }
QProgressBar { min-height: 18px; border-radius: 4px; text-align: center; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        icon = bundle_dir() / "assets" / "audiorefine.ico"
        if icon.is_file():
            self.setWindowIcon(QIcon(str(icon)))
        self.setAcceptDrops(True)
        self.folder: Path | None = None
        self.scan: dict | None = None
        self.status: dict | None = None
        self.job: EngineJob | None = None
        self.job_kind = ""
        self.started = 0.0
        self.last_eta = None
        self.last_batch_dir: str | None = None
        self.master_path: str | None = None
        self.cpu_retry_done = False
        self._close_after_cancel = False
        self._build()
        self.tick = QTimer(self)
        self.tick.timeout.connect(self._on_tick)
        self.tick.start(1000)
        QTimer.singleShot(0, self._startup)

    # ------------------------------------------------------------------ layout
    def _build(self):
        c = QWidget(objectName="central")
        self.setCentralWidget(c)
        c.setStyleSheet(STYLE)
        v = QVBoxLayout(c)
        v.setContentsMargins(22, 18, 22, 16)
        v.setSpacing(10)

        title = QLabel(APP_NAME, objectName="title")
        sub = QLabel("Enhanced speech and timestamped transcripts from a folder of recordings", objectName="muted")
        v.addWidget(title)
        v.addWidget(sub)
        v.addSpacing(6)

        v.addWidget(QLabel("<b>Source folder</b>"))
        row = QHBoxLayout()
        self.folder_edit = QLineEdit(readOnly=True, placeholderText="Choose the folder that contains your recordings")
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self.choose_folder)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(self.browse_btn)
        v.addLayout(row)

        self.summary = QLabel("")
        f = QFont()
        f.setPointSizeF(f.pointSizeF() + 1)
        self.summary.setFont(f)
        v.addWidget(self.summary)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["#", "Recording", "Recorded", "Duration", "Format", "Note"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tree.setAlternatingRowColors(True)
        h = self.tree.header()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setMinimumHeight(170)
        v.addWidget(self.tree, 1)

        self.order_label = QLabel("", objectName="muted")
        self.order_label.setWordWrap(True)
        v.addWidget(self.order_label)
        self.warn_label = QLabel("", objectName="warn")
        self.warn_label.setWordWrap(True)
        self.warn_label.hide()
        v.addWidget(self.warn_label)

        brow = QHBoxLayout()
        brow.addStretch(1)
        self.process_btn = QPushButton("Process folder", objectName="primary")
        self.process_btn.setMinimumWidth(220)
        self.process_btn.clicked.connect(self.on_process)
        brow.addWidget(self.process_btn)
        brow.addStretch(1)
        v.addLayout(brow)

        self.hour_label = QLabel("")
        v.addWidget(self.hour_label)
        self.stage_label = QLabel("")
        v.addWidget(self.stage_label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFormat("%p%")
        self.bar.setValue(0)
        v.addWidget(self.bar)
        self.time_label = QLabel("", objectName="muted")
        v.addWidget(self.time_label)

        self.banner = QFrame(objectName="banner_info")
        bl = QVBoxLayout(self.banner)
        bl.setContentsMargins(12, 8, 12, 8)
        self.banner_text = QLabel("")
        self.banner_text.setWordWrap(True)
        self.banner_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bl.addWidget(self.banner_text)
        self.banner.hide()
        v.addWidget(self.banner)

        bottom = QHBoxLayout()
        self.open_btn = QPushButton("Open refined folder")
        self.open_btn.clicked.connect(self.open_refined)
        self.open_btn.setEnabled(False)
        self.master_btn = QPushButton("Open master transcript")
        self.master_btn.setToolTip("Transcript_Master_Complete.txt: everything transcribed so far, in chronological order")
        self.master_btn.clicked.connect(self.open_master)
        self.master_btn.setEnabled(False)
        self.log_btn = QPushButton("Logs")
        self.log_btn.setToolTip("Open the folder with diagnostic logs (useful when reporting a problem)")
        self.log_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs_dir()))))
        self.about_btn = QPushButton("About")
        self.about_btn.clicked.connect(self.about)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.on_cancel)
        bottom.addWidget(self.open_btn)
        bottom.addWidget(self.master_btn)
        bottom.addWidget(self.log_btn)
        bottom.addWidget(self.about_btn)
        bottom.addStretch(1)
        bottom.addWidget(self.cancel_btn)
        v.addLayout(bottom)
        self.device_label = QLabel("", objectName="muted")
        v.addWidget(self.device_label)
        self.resize(QSize(760, 680))
        self._set_idle()

    # ------------------------------------------------------------------ settings
    def _load_settings(self) -> dict:
        try:
            return json.loads(settings_file().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self, **kw):
        s = self._load_settings()
        s.update(kw)
        try:
            settings_file().write_text(json.dumps(s, indent=1), encoding="utf-8")
        except Exception:
            pass

    def _startup(self):
        self._run_status()
        last = self._load_settings().get("last_folder")
        if last and Path(last).is_dir():
            self.set_folder(Path(last))

    # ------------------------------------------------------------------ folder handling
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and not self.busy():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.is_dir():
                self.set_folder(p)
                return

    def choose_folder(self):
        start = str(self.folder) if self.folder else str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Choose the folder with your recordings", start)
        if d:
            self.set_folder(Path(d))

    def set_folder(self, p: Path):
        self.folder = p
        self.folder_edit.setText(str(p))
        self._save_settings(last_folder=str(p))
        self.banner.hide()
        self.rescan()

    def rescan(self):
        if not self.folder or self.busy():
            return
        self.scan = None
        self.tree.clear()
        self.summary.setText("Looking for recordings…")
        self.order_label.setText("")
        self.warn_label.hide()
        self.process_btn.setEnabled(False)
        self._start_job("scan", ["scan", str(self.folder)])

    def _show_scan(self, ev: dict):
        self.scan = ev
        self.tree.clear()
        srcs = ev.get("sources", [])
        status_note = {"processed": "already processed – will be moved to 'processed'",
                       "duplicate": "already processed earlier – left in the folder",
                       "partial": "partly processed – continues"}
        cols = 6
        for i, s in enumerate(srcs, 1):
            ch = s.get("channels")
            fmt = f"{codec_name(s.get('codec', ''))} · {s.get('sample_rate', 0) / 1000:g} kHz · " \
                  f"{'mono' if ch == 1 else 'stereo' if ch == 2 else str(ch) + ' ch'}"
            rec = fmt_rec(s.get("rec_time"), s.get("rec_estimated"))
            hours_ = s.get("todo_hours") or s.get("all_hours") or []
            note = status_note.get(s.get("status"), "")
            if not note and hours_:
                note = hours_span(hours_)
            it = QTreeWidgetItem([str(i), s["name"], rec, fmt_hms(s.get("duration")), fmt, note])
            it.setTextAlignment(0, Qt.AlignmentFlag.AlignRight)
            it.setToolTip(2, f"Recording time taken from: {s.get('rec_basis', '')}"
                             + (" (estimated)" if s.get("rec_estimated") else "")
                             + (f"\n{s.get('rec_note')}" if s.get("rec_note") else ""))
            if s.get("status") in ("processed", "duplicate"):
                for col in range(cols):
                    it.setForeground(col, Qt.GlobalColor.gray)
            self.tree.addTopLevelItem(it)
        for p in ev.get("pending", []):
            for n in p["names"]:
                it = QTreeWidgetItem(["", n, "", "", "", f"already processed ({p['batch']}); will be archived"])
                for col in range(cols):
                    it.setForeground(col, Qt.GlobalColor.gray)
                self.tree.addTopLevelItem(it)
        for u in ev.get("unreadable", []):
            it = QTreeWidgetItem(["✕", u["name"], "", "", "", u.get("error", "cannot be read")])
            for col in range(cols):
                it.setForeground(col, Qt.GlobalColor.darkRed)
            it.setToolTip(5, u.get("error", ""))
            self.tree.addTopLevelItem(it)
        plan = ev.get("plan") or {}
        todo = [x for x in srcs if x.get("status") in ("new", "partial")]
        total = sum(x.get("duration", 0.0) for x in todo)
        nh, nd = plan.get("hours_to_process", 0), plan.get("days_to_process", 0)
        done_h = plan.get("completed_hours", 0)
        if todo:
            self.summary.setText(f"<b>{len(todo)}</b> new recording{'s' if len(todo) != 1 else ''} · "
                                 f"<b>{nh}</b> hour{'s' if nh != 1 else ''} on <b>{nd}</b> day{'s' if nd != 1 else ''} to process"
                                 f" · Total duration <b>{fmt_hms(total)}</b>"
                                 + (f" &nbsp;<span style='color:#6b7280'>({done_h} hour(s) already finished)</span>" if done_h else ""))
            self.order_label.setText("Processing order: by recording date and time – one clock hour at a time, oldest "
                                     "first. Times marked ≈ are estimated from the file dates.")
        elif srcs or done_h:
            self.summary.setText(f"No new recordings to process. {done_h} hour(s) already finished in this folder.")
            self.order_label.setText("Recordings must be placed directly in the folder (subfolders are not searched).")
        else:
            self.summary.setText("No audio recordings found directly in this folder.")
            self.order_label.setText("Recordings must be placed directly in the folder (subfolders are not searched).")
        warns = list(ev.get("warnings", []))
        if ev.get("unreadable"):
            warns.append(f"{len(ev['unreadable'])} file(s) cannot be read and will be left untouched.")
        if ev.get("busy"):
            warns.append("This folder is currently being processed by another AudioRefine window.")
        if warns:
            self.warn_label.setText("\n".join(warns))
            self.warn_label.show()
        self._set_idle()

    # ------------------------------------------------------------------ jobs
    def busy(self) -> bool:
        return self.job is not None and self.job.running()

    def _start_job(self, kind: str, args: list[str]):
        self.job_kind = kind
        self.job = EngineJob(args, self)
        self.job.event.connect(self._on_event)
        self.job.finished.connect(self._on_finished)
        self.job.crashed.connect(self._on_crashed)
        self.job.start()

    def _run_status(self):
        job = EngineJob(["status"], self)
        job.event.connect(lambda ev: setattr(self, "status", ev) if ev.get("type") == "status" else None)
        job.start()
        self._status_job = job

    def _on_event(self, ev: dict):
        t = ev.get("type")
        if t == "scan":
            self._show_scan(ev)
        elif t == "progress":
            stage = f"<b>{ev.get('label', '')}</b>" + (f" – {ev['detail']}" if ev.get("detail") else "")
            if ev.get("current_hour"):
                self.hour_label.setText(
                    f"Current day: <b>{ev['current_date']}</b> &nbsp; Current hour: <b>{ev['current_hour']}</b>"
                    f" &nbsp;·&nbsp; Days completed: {ev.get('days_done', 0)} of {ev.get('days_total', 0)}"
                    f" &nbsp;·&nbsp; Hours completed: {ev.get('hours_done', 0)} of {ev.get('hours_total', 0)}")
                stage += f" &nbsp;<span style='color:#6b7280'>(current hour {int(ev.get('hour_fraction', 0) * 100)}%)</span>"
            self.stage_label.setText(stage)
            self.bar.setValue(int(ev.get("fraction", 0) * 1000))
            self.last_eta = ev.get("eta")
        elif t == "hour_done":
            self.master_path = ev.get("master")
            self.master_btn.setEnabled(bool(self.master_path))
            self.open_btn.setEnabled(True)
            self.device_label.setText(f"Finished {ev.get('date')} {ev.get('hour')} – results are available in "
                                      f"'refined' while the next hours are processed.")
        elif t == "download":
            done, total = ev.get("done", 0), max(1, ev.get("total", 1))
            el = max(0.1, ev.get("elapsed", 0.1))
            speed = done / el
            self.stage_label.setText(f"<b>Downloading {ev.get('label', '').lower()} (one-time)</b> – "
                                     f"{fmt_size(done)} of {fmt_size(total)} · {fmt_size(speed)}/s")
            self.bar.setValue(int(done / total * 1000))
            self.last_eta = (total - done) / speed if speed > 0 and done > 0 else None
        elif t == "info" and ev.get("device"):
            self.device_label.setText(f"Speech recognition running on: {ev['device']}")
        elif t == "warning":
            self.device_label.setText(ev.get("message", ""))

    def _on_crashed(self, err: str):
        self._crash_text = err

    def _on_finished(self, code: int, result: dict):
        kind = self.job_kind
        self.job = None
        if kind == "scan":
            if not self.scan:
                self.summary.setText("The folder could not be read.")
            self._set_idle()
            return
        if kind == "download":
            st = result.get("status")
            if st == "success":
                self._run_status()
                self._start_processing()
            else:
                self._finish_ui()
                self._banner("err" if st != "cancelled" else "info",
                             result.get("message") or "The download did not complete. Try again; it will resume.")
            return
        if kind == "process":
            st = result.get("status")
            if not result:
                crash = getattr(self, "_crash_text", "")
                if not self.cpu_retry_done and "Loading speech" in self.stage_label.text():
                    # e.g. a GPU driver crash while loading the model: retry once on the CPU
                    self.cpu_retry_done = True
                    self._start_processing(force_cpu=True)
                    return
                self._finish_ui()
                self._banner("err", "The processing engine stopped unexpectedly. Your recordings were not changed. "
                                    "Please try again; if it happens again, open Logs and share the newest log file."
                             + (f"\n\nDetails: {crash[-300:]}" if crash else ""))
                self.rescan_after()
                return
            if st == "model_missing":
                self._finish_ui()
                self._ask_download()
                return
            self._finish_ui()
            msg = result.get("message", "")
            warns = result.get("warnings") or []
            if result.get("master"):
                self.master_path = result["master"]
            if st == "success":
                self.last_batch_dir = result.get("refined_dir")
                extra = ("\n\nNotes:\n• " + "\n• ".join(warns)) if warns else ""
                self._banner("ok", "✔ " + msg + f"\nTotal time {fmt_hms(time.time() - self.started)}." + extra)
            elif st == "archive_incomplete":
                self.last_batch_dir = result.get("refined_dir")
                self._banner("err", msg + ("\n\n• " + "\n• ".join(warns) if warns else ""))
            elif st == "cancelled":
                self._banner("info", msg)
            else:
                self._banner("err", "✕ " + msg + ("\n\n• " + "\n• ".join(warns) if warns else ""))
            if self._close_after_cancel:
                self.close()
                return
            self.rescan_after()

    def rescan_after(self):
        QTimer.singleShot(300, self.rescan)

    # ------------------------------------------------------------------ actions
    def on_process(self):
        if not self.folder or not self.scan or not (self.scan.get("sources") or self.scan.get("pending")):
            return
        st = self.status or {}
        if st and not st.get("model_installed", False):
            self._ask_download()
            return
        self._start_processing()

    def _ask_download(self):
        st = self.status or {}
        need = st.get("model_missing_bytes") or 3_100_000_000
        parts = [f"• Speech recognition model (Whisper large-v3): {fmt_size(need)}"]
        gpu = st.get("gpu")
        use_gpu = bool(gpu) and not st.get("gpu_runtime_installed")
        if use_gpu:
            parts.append(f"• Acceleration for your {gpu['name']} graphics card: {fmt_size(st.get('gpu_runtime_bytes', 0))}")
        r = QMessageBox.question(
            self, "One-time download",
            "AudioRefine needs to download the following once:\n\n" + "\n".join(parts) +
            "\n\nAfter that, everything runs offline on this computer. Your recordings are never uploaded.\n\n"
            "Download now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return
        self._set_busy("download")
        self.stage_label.setText("<b>Preparing download…</b>")
        self._start_job("download", ["download"] + (["--gpu"] if use_gpu else []))

    def _start_processing(self, force_cpu: bool = False):
        if not force_cpu:
            self.cpu_retry_done = False
        self._crash_text = ""
        self._set_busy("process")
        self.stage_label.setText("<b>Starting…</b>")
        args = ["process", str(self.folder)] + (["--cpu"] if force_cpu else [])
        self._start_job("process", args)

    def on_cancel(self):
        if self.busy():
            self.cancel_btn.setEnabled(False)
            self.stage_label.setText("<b>Cancelling…</b> (cleaning up; your recordings are not changed)")
            self.job.cancel()

    def open_refined(self):
        target = None
        if self.last_batch_dir and Path(self.last_batch_dir).is_dir():
            target = Path(self.last_batch_dir)
        elif self.folder and (self.folder / REFINED).is_dir():
            target = self.folder / REFINED
        if target:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def open_master(self):
        p = Path(self.master_path) if getattr(self, "master_path", None) else None
        if (p is None or not p.is_file()) and self.folder:
            p = self.folder / REFINED / "Transcript_Master_Complete.txt"
        if p and p.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def about(self):
        notices = bundle_dir() / "THIRD_PARTY_NOTICES.txt"
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<b>{APP_NAME} {__version__}</b><br>"
                          "Speech enhancement and transcription for folders of recordings.<br><br>"
                          "Runs entirely on this computer. Uses Whisper large-v3 (OpenAI, MIT) via faster-whisper/"
                          "CTranslate2, DeepFilterNet3 (MIT/Apache-2.0), Silero VAD (MIT), FFmpeg (LGPL) via PyAV and "
                          "Qt for Python (LGPL).<br><br>"
                          f"Licence details: {notices if notices.is_file() else 'THIRD_PARTY_NOTICES.txt'}<br>"
                          f"Logs: {logs_dir()}")

    # ------------------------------------------------------------------ state helpers
    def _banner(self, kind: str, text: str):
        self.banner.setObjectName({"ok": "banner_ok", "err": "banner_err"}.get(kind, "banner_info"))
        self.banner.setStyleSheet(STYLE)
        self.banner_text.setText(text)
        self.banner.show()

    def _set_busy(self, kind: str):
        self.started = time.time()
        self.last_eta = None
        self.banner.hide()
        self.process_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.bar.setValue(0)
        self.time_label.setText("Elapsed time: 00:00:00")
        self.setAcceptDrops(False)
        self.hour_label.setText("")
        for wdg in (self.bar, self.stage_label, self.time_label, self.hour_label):
            wdg.setVisible(True)

    def _finish_ui(self):
        self.device_label.setText("")
        self.cancel_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.setAcceptDrops(True)
        self.stage_label.setText("")
        self.bar.setValue(0)
        self.time_label.setText("")
        self._set_idle()

    def _refresh_buttons(self):
        refined = self.folder / REFINED if self.folder else None
        self.open_btn.setEnabled(bool(refined and refined.is_dir()))
        self.master_btn.setEnabled(bool(refined and (refined / "Transcript_Master_Complete.txt").is_file()))

    def _set_idle(self):
        working = self.busy() and self.job_kind in ("process", "download")
        self.bar.setVisible(working)
        self.stage_label.setVisible(working)
        self.time_label.setVisible(working)
        sc = self.scan or {}
        plan = sc.get("plan") or {}
        work = plan.get("hours_to_process") or plan.get("pending_archive") or sc.get("pending")
        can = bool(work) and not self.busy() and not sc.get("busy")
        self.process_btn.setEnabled(can)
        self.hour_label.setVisible(working)
        self._refresh_buttons()

    def _on_tick(self):
        if self.busy() and self.job_kind in ("process", "download"):
            t = f"Elapsed time: {fmt_hms(time.time() - self.started)}"
            if self.last_eta:
                t += f"   ·   About {fmt_hms(self.last_eta)} remaining"
            self.time_label.setText(t)

    def closeEvent(self, e):
        if self.busy() and self.job_kind in ("process", "download"):
            r = QMessageBox.question(self, APP_NAME, "Processing is still running. Cancel it and close AudioRefine?\n\n"
                                     "Your recordings stay unchanged.")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self._close_after_cancel = True
            self.on_cancel()
            e.ignore()
            return
        e.accept()
