"""Sound Explorer: browse, play and review the detected sounds of a folder."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFont, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QRadioButton, QScrollArea,
                               QSlider, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..hourtext import fmt_offset
from ..rectime import fmt_clock, local
from ..sound import explore, store
from ..sound.player import Player, PlayerError

GROUP_COLORS = {
    "Human sounds": "#e11d48", "Impacts and movement": "#ea580c", "Personal care and bathroom": "#0891b2",
    "Kitchen": "#ca8a04", "Household": "#7c3aed", "Entertainment and electronics": "#2563eb",
    "Animals": "#16a34a", "Outdoor and transport": "#64748b",
}
ACTIVITY_COLOR = "#0f766e"
DISCLAIMER = ("Detections are made automatically and are approximate; some will be wrong. A sound that is not "
              "listed may still have occurred. Sounds from a TV, radio, phone or game can be detected as if they "
              "happened in the room.")


def color_of(x: dict) -> QColor:
    if x.get("kind") == "activity":
        return QColor(ACTIVITY_COLOR)
    return QColor(GROUP_COLORS.get(x.get("group", ""), "#475569"))


def span(x: dict) -> str:
    st, en = datetime.fromisoformat(x["start"]), datetime.fromisoformat(x["end"])
    return f"{fmt_clock(st, x.get('estimated', False), with_date=False)} – {local(en).strftime('%H:%M:%S')}"


class Timeline(QWidget):
    """00–24 h overview per date (or one hour in detail); bars show where sounds were detected."""
    itemClicked = Signal(str)
    hourClicked = Signal(str, int)

    ROW = 22
    LEFT = 92

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[dict] = []
        self.dates: list[str] = []
        self.hour: int | None = None
        self.selected: str | None = None
        self._hits: list[tuple[QRectF, str]] = []
        self.setMouseTracking(True)

    def set_data(self, items: list[dict], dates: list[str], hour: int | None, selected: str | None):
        self.items, self.dates, self.hour, self.selected = items, dates, hour, selected
        self.setMinimumHeight(28 + self.ROW * max(1, len(dates)))
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(600, 28 + self.ROW * max(1, len(self.dates)))

    def _x(self, h: float) -> float:
        w = self.width() - self.LEFT - 10
        if self.hour is None:
            return self.LEFT + w * max(0.0, min(24.0, h)) / 24
        return self.LEFT + w * max(0.0, min(1.0, h - self.hour))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        f = QFont(self.font())
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        p.setFont(f)
        top = 20
        ticks = range(0, 25, 3) if self.hour is None else range(0, 61, 10)
        for t in ticks:
            x = self._x(t if self.hour is None else self.hour + t / 60)
            p.setPen(QPen(QColor("#e5e7eb")))
            p.drawLine(int(x), top - 2, int(x), self.height())
            p.setPen(QPen(QColor("#6b7280")))
            label = f"{t:02d}:00" if self.hour is None else f"{self.hour:02d}:{t:02d}"
            p.drawText(QRectF(x - 25, 2, 50, 14), Qt.AlignmentFlag.AlignCenter, label)
        self._hits = []
        for r, d in enumerate(self.dates):
            y = top + r * self.ROW
            p.setPen(QPen(QColor("#374151")))
            p.drawText(QRectF(4, y, self.LEFT - 8, self.ROW), Qt.AlignmentFlag.AlignVCenter, d)
            p.fillRect(QRectF(self.LEFT, y + 4, self.width() - self.LEFT - 10, self.ROW - 8), QColor("#f3f4f6"))
            for x in self.items:
                if explore.item_date(x) != d:
                    continue
                a, b = explore.item_hours(x)
                if self.hour is not None and (b < self.hour or a >= self.hour + 1):
                    continue
                x0, x1 = self._x(a), self._x(b)
                rect = QRectF(x0, y + 5, max(3.0, x1 - x0), self.ROW - 10)
                c = color_of(x)
                if x.get("confidence") != "high":
                    c.setAlpha(150)
                p.fillRect(rect, QBrush(c))
                if x["id"] == self.selected:
                    p.setPen(QPen(QColor("#111827"), 2))
                    p.drawRect(rect.adjusted(-1, -1, 1, 1))
                self._hits.append((rect.adjusted(-2, -3, 2, 3), x["id"]))
        p.end()

    def mousePressEvent(self, e):
        pos = e.position()
        for rect, iid in reversed(self._hits):
            if rect.contains(pos):
                self.itemClicked.emit(iid)
                return
        if self.hour is None and pos.x() > self.LEFT:
            row = int((pos.y() - 20) // self.ROW)
            if 0 <= row < len(self.dates):
                w = self.width() - self.LEFT - 10
                h = int((pos.x() - self.LEFT) / w * 24)
                self.hourClicked.emit(self.dates[row], max(0, min(23, h)))


class SoundExplorer(QMainWindow):
    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.res: explore.Results | None = None
        self.player = Player()
        self.current: dict | None = None
        self.target: explore.PlayTarget | None = None
        self.detail_hour: tuple[str, int] | None = None
        self.reviews_changed = False
        self.setWindowTitle(f"Sound Explorer – {self.root.name}")
        self._build()
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.reload()

    # ------------------------------------------------------------------ layout
    def _build(self):
        c = QWidget()
        self.setCentralWidget(c)
        v = QVBoxLayout(c)
        v.setContentsMargins(14, 10, 14, 10)
        top = QHBoxLayout()
        self.title = QLabel(f"<b>Sound Explorer</b> – {self.root}")
        top.addWidget(self.title, 1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.reload)
        self.report_btn = QPushButton("Open sound report")
        self.report_btn.clicked.connect(lambda: self._open(store.adir(self.root) / store.MASTER_TXT))
        self.annot_btn = QPushButton("Open annotated transcript")
        self.annot_btn.clicked.connect(lambda: self._open(store.adir(self.root) / store.ANNOTATED_TXT))
        for b in (self.refresh_btn, self.report_btn, self.annot_btn):
            top.addWidget(b)
        v.addLayout(top)

        f = QHBoxLayout()
        self.by_sound = QRadioButton("By sound")
        self.by_time = QRadioButton("Chronological")
        self.by_sound.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.by_sound)
        grp.addButton(self.by_time)
        self.by_sound.toggled.connect(self.refresh_view)
        f.addWidget(QLabel("View:"))
        f.addWidget(self.by_sound)
        f.addWidget(self.by_time)
        f.addSpacing(12)
        f.addWidget(QLabel("Date:"))
        self.date_cb = QComboBox()
        self.date_cb.currentIndexChanged.connect(self.refresh_view)
        f.addWidget(self.date_cb)
        f.addWidget(QLabel("Hours:"))
        self.h_from = QComboBox()
        self.h_to = QComboBox()
        for h in range(24):
            self.h_from.addItem(f"{h:02d}:00", h)
        for h in range(1, 25):
            self.h_to.addItem(f"{h:02d}:00", h)
        self.h_to.setCurrentIndex(23)
        self.h_from.currentIndexChanged.connect(self.refresh_view)
        self.h_to.currentIndexChanged.connect(self.refresh_view)
        f.addWidget(self.h_from)
        f.addWidget(QLabel("–"))
        f.addWidget(self.h_to)
        f.addWidget(QLabel("Confidence:"))
        self.conf_cb = QComboBox()
        self.conf_cb.addItem("High and moderate", "all")
        self.conf_cb.addItem("High only", "high")
        self.conf_cb.currentIndexChanged.connect(self.refresh_view)
        f.addWidget(self.conf_cb)
        self.show_wrong = QCheckBox("Show detections marked incorrect")
        self.show_wrong.toggled.connect(self.refresh_view)
        f.addWidget(self.show_wrong)
        f.addStretch(1)
        v.addLayout(f)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("<b>Detected sounds</b>"))
        self.cat_list = QTreeWidget()
        self.cat_list.setHeaderLabels(["Sound", "Count"])
        self.cat_list.setRootIsDecorated(False)
        self.cat_list.setColumnWidth(0, 190)
        self.cat_list.currentItemChanged.connect(lambda *_: self.refresh_events())
        lv.addWidget(self.cat_list, 1)
        split.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        tl_head = QHBoxLayout()
        self.tl_label = QLabel("<b>Timeline</b> (click an hour to see it in detail)")
        self.day_btn = QPushButton("◀ Whole day")
        self.day_btn.clicked.connect(self._back_to_day)
        self.day_btn.hide()
        tl_head.addWidget(self.tl_label, 1)
        tl_head.addWidget(self.day_btn)
        rv.addLayout(tl_head)
        self.timeline = Timeline()
        self.timeline.itemClicked.connect(self.select_id)
        self.timeline.hourClicked.connect(self._zoom_hour)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(self.timeline)
        sa.setFrameShape(QFrame.Shape.StyledPanel)
        self.tl_scroll = sa
        rv.addWidget(sa)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Time", "Sound", "Confidence", "Recording", "Note"])
        self.tree.setColumnWidth(0, 215)
        self.tree.setColumnWidth(1, 180)
        self.tree.setColumnWidth(2, 90)
        self.tree.setColumnWidth(3, 190)
        self.tree.currentItemChanged.connect(self._tree_selected)
        self.tree.itemDoubleClicked.connect(lambda *_: self.play_event())
        rv.addWidget(self.tree, 1)
        split.addWidget(right)
        split.setSizes([250, 750])
        v.addWidget(split, 1)

        pf = QFrame(objectName="player")
        pf.setFrameShape(QFrame.Shape.StyledPanel)
        pv = QVBoxLayout(pf)
        self.sel_label = QLabel("Select a detection to listen to it.")
        self.sel_label.setWordWrap(True)
        pv.addWidget(self.sel_label)
        prow = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play from event")
        self.play_btn.clicked.connect(self.play_event)
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderReleased.connect(self._seek)
        self.pos_label = QLabel("00:00:00 / 00:00:00")
        self.src_orig = QRadioButton("Original")
        self.src_enh = QRadioButton("Enhanced")
        self.src_orig.setChecked(True)
        sg = QButtonGroup(self)
        sg.addButton(self.src_orig)
        sg.addButton(self.src_enh)
        self.src_enh.toggled.connect(lambda on: self._switch_source())
        for w in (self.play_btn, self.pause_btn):
            prow.addWidget(w)
        prow.addWidget(self.slider, 1)
        prow.addWidget(self.pos_label)
        prow.addWidget(self.src_orig)
        prow.addWidget(self.src_enh)
        pv.addLayout(prow)
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Your review:"))
        self.rev_btns = {}
        for st, text in (("confirmed", "✔ Confirmed"), ("incorrect", "✕ Incorrect detection"), ("unreviewed", "Unreviewed")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, s=st: self.review(s))
            self.rev_btns[st] = b
            rrow.addWidget(b)
        rrow.addStretch(1)
        pv.addLayout(rrow)
        v.addWidget(pf)
        note = QLabel(DISCLAIMER)
        note.setWordWrap(True)
        note.setStyleSheet("color:#6b7280")
        v.addWidget(note)
        self._enable_player(False)
        self.resize(QSize(1180, 820))

    # ------------------------------------------------------------------ data
    def reload(self):
        keep = self.current["id"] if self.current else None
        self.res = explore.load(self.root)
        self.date_cb.blockSignals(True)
        cur = self.date_cb.currentData()
        self.date_cb.clear()
        self.date_cb.addItem("All dates", None)
        for d in (self.res.dates if self.res else []):
            self.date_cb.addItem(d, d)
        i = self.date_cb.findData(cur)
        self.date_cb.setCurrentIndex(max(0, i))
        self.date_cb.blockSignals(False)
        self.annot_btn.setEnabled((store.adir(self.root) / store.ANNOTATED_TXT).is_file())
        self.report_btn.setEnabled((store.adir(self.root) / store.MASTER_TXT).is_file())
        status = (self.res.status if self.res else "") or ""
        self.title.setText(f"<b>Sound Explorer</b> – {self.root}"
                           + (f" &nbsp; <span style='color:#6b7280'>({status.split(' –')[0].lower()})</span>"
                              if status else ""))
        self.refresh_view()
        if keep:
            self.select_id(keep)

    def _filters(self) -> dict:
        return {"date": self.date_cb.currentData(), "h_from": self.h_from.currentData(),
                "h_to": max(self.h_to.currentData(), self.h_from.currentData() + 1),
                "conf": self.conf_cb.currentData(), "show_incorrect": self.show_wrong.isChecked()}

    def refresh_view(self):
        if self.res is None:
            self.cat_list.clear()
            self.tree.clear()
            self.sel_label.setText("No sound analysis results in this folder yet. Click 'Analyze sounds' in the "
                                   "main window first.")
            return
        prev = self.cat_list.currentItem().data(0, Qt.ItemDataRole.UserRole) if self.cat_list.currentItem() else None
        self.cat_list.blockSignals(True)
        self.cat_list.clear()
        rows = explore.counts(self.res, **self._filters())
        allitem = QTreeWidgetItem(["All sounds", str(sum(n for _, n, _ in rows))])
        allitem.setData(0, Qt.ItemDataRole.UserRole, None)
        self.cat_list.addTopLevelItem(allitem)
        sel = allitem
        header_done = False
        for label, n, is_act in rows:
            if is_act and not header_done:
                h = QTreeWidgetItem(["Possible activities (inferred)"])
                h.setFlags(Qt.ItemFlag.NoItemFlags)
                f = h.font(0)
                f.setBold(True)
                h.setFont(0, f)
                self.cat_list.addTopLevelItem(h)
                header_done = True
            it = QTreeWidgetItem([label, str(n)])
            it.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            it.setData(0, Qt.ItemDataRole.UserRole, label)
            it.setForeground(0, QBrush(QColor(ACTIVITY_COLOR)) if is_act else it.foreground(0))
            self.cat_list.addTopLevelItem(it)
            if label == prev:
                sel = it
        self.cat_list.setCurrentItem(sel)
        self.cat_list.resizeColumnToContents(1)
        self.cat_list.blockSignals(False)
        self.refresh_events()

    def refresh_events(self):
        if self.res is None:
            return
        it = self.cat_list.currentItem()
        cat = it.data(0, Qt.ItemDataRole.UserRole) if it else None
        items = explore.filter_items(self.res, category=cat, **self._filters())
        self.tree.blockSignals(True)
        self.tree.clear()
        bold = self.tree.font()
        bold.setBold(True)
        if self.by_sound.isChecked() and cat is None:
            groups: dict[str, list[dict]] = {}
            for x in items:
                groups.setdefault(explore.label_of(x), []).append(x)
            for label in sorted(groups, key=lambda k: (groups[k][0].get("kind") == "activity", k)):
                top = QTreeWidgetItem([f"{label} ({len(groups[label])})"])
                top.setFont(0, bold)
                top.setFirstColumnSpanned(True)
                self.tree.addTopLevelItem(top)
                self._add_dated(top, groups[label])
        else:
            self._add_dated(self.tree.invisibleRootItem(), items)
        self.tree.expandToDepth(1 if self.by_sound.isChecked() and cat is None else 0)
        self.tree.blockSignals(False)
        dates = [self.date_cb.currentData()] if self.date_cb.currentData() else sorted({explore.item_date(x) for x in items})
        if self.detail_hour:
            dates = [self.detail_hour[0]]
        self.timeline.set_data(items, dates, self.detail_hour[1] if self.detail_hour else None,
                               self.current["id"] if self.current else None)
        self.tl_scroll.setFixedHeight(min(8, max(1, len(dates))) * Timeline.ROW + 34)
        if not items:
            self.tree.addTopLevelItem(QTreeWidgetItem(["No detections match the selection."]))

    def _add_dated(self, parent, items: list[dict]):
        bold = self.tree.font()
        bold.setBold(True)
        for d, hours in explore.group_by_date_hour(items):
            n = sum(len(v) for _, v in hours)
            di = QTreeWidgetItem([f"{d}  ({n})"])
            di.setFont(0, bold)
            di.setFirstColumnSpanned(True)
            parent.addChild(di) if isinstance(parent, QTreeWidgetItem) else self.tree.addTopLevelItem(di)
            for _, xs in hours:
                for x in xs:
                    self._add_item(di, x)

    def _add_item(self, parent: QTreeWidgetItem, x: dict):
        note = []
        if x.get("kind") == "activity":
            note.append("inferred from: " + ", ".join(x.get("supporting_labels") or []))
        if x.get("during_media"):
            note.append("during music/TV audio")
        if x.get("hits", 1) > 1:
            note.append(f"{x['hits']}× close together")
        if x.get("frequent"):
            note.append("very frequent here – only high-confidence ones shown")
        if x.get("review") == "confirmed":
            note.append("confirmed by you")
        elif x.get("review") == "incorrect":
            note.append("marked incorrect")
        if x.get("estimated"):
            note.append("time estimated (≈)")
        rec = x.get("source_name", "") if x.get("kind") != "activity" else f"{len(x.get('supporting', []))} sounds"
        it = QTreeWidgetItem([span(x), explore.label_of(x), "High" if x["confidence"] == "high" else "Moderate", rec,
                              "; ".join(note)])
        it.setData(0, Qt.ItemDataRole.UserRole, x["id"])
        it.setForeground(1, QBrush(color_of(x)))
        parent.addChild(it)
        if x.get("kind") == "activity" and self.res:
            for s in explore.supporting(self.res, x):
                ch = QTreeWidgetItem([span(s), s["display"], "High" if s["confidence"] == "high" else "Moderate",
                                      s.get("source_name", ""), "supporting sound"])
                ch.setData(0, Qt.ItemDataRole.UserRole, s["id"])
                it.addChild(ch)

    # ------------------------------------------------------------------ selection and playback
    def _tree_selected(self, cur, _prev):
        iid = cur.data(0, Qt.ItemDataRole.UserRole) if cur else None
        if iid and self.res and iid in self.res.by_id:
            self._select(self.res.by_id[iid])

    def select_id(self, iid: str):
        if not self.res or iid not in self.res.by_id:
            return
        it = self._find(self.tree.invisibleRootItem(), iid)
        if it is not None:
            self.tree.setCurrentItem(it)
            self.tree.scrollToItem(it)
        else:
            self._select(self.res.by_id[iid])

    def _find(self, parent, iid):
        for i in range(parent.childCount()):
            ch = parent.child(i)
            if ch.data(0, Qt.ItemDataRole.UserRole) == iid:
                return ch
            r = self._find(ch, iid)
            if r is not None:
                return r
        return None

    def _select(self, x: dict):
        self.player.stop()
        self.current = x
        self.timeline.selected = x["id"]
        self.timeline.update()
        kind = "Possible activity (inferred, not a fact)" if x.get("kind") == "activity" else "Detected sound"
        lines = [f"<b>{x['display']}</b> &nbsp;·&nbsp; {kind} &nbsp;·&nbsp; {'High' if x['confidence'] == 'high' else 'Moderate'} confidence",
                 f"{explore.item_date(x)} {span(x)}" + (" – recording time estimated from the file dates (≈)"
                                                       if x.get("estimated") else " – recording time from "
                                                       + str(x.get("time_basis", "the recording")))]
        if x.get("kind") == "activity":
            lines.append("Based on: " + ", ".join(x.get("supporting_labels") or []) +
                         ". Play starts at the first supporting sound; expand the row to play the others.")
        else:
            lines.append(f"Recording: {x['source_name']} at {fmt_offset(x['src_start'])}"
                         + (f" &nbsp;·&nbsp; enhanced audio: {x['refined']['file']} at {fmt_offset(x['refined']['offset'])}"
                            if x.get("refined") else ""))
            if x.get("during_media"):
                lines.append("<span style='color:#b45309'>Detected while music or TV audio was playing – it may "
                             "come from the playback device.</span>")
        self.sel_label.setText("<br>".join(lines))
        has_enh = x.get("kind") != "activity" and bool(x.get("refined"))
        self.src_enh.setEnabled(has_enh)
        if not has_enh:
            self.src_orig.setChecked(True)
        self._prepare_target()
        for st, b in self.rev_btns.items():
            b.setChecked(x.get("review", "unreviewed") == st)

    def _prepare_target(self):
        if not self.current or not self.res:
            return
        self.target = explore.play_target(self.res, self.current, enhanced=self.src_enh.isChecked())
        if self.target.path is None:
            self._enable_player(False)
            self.pos_label.setText(self.target.note)
            return
        try:
            self.player.open(self.target.path)
            self._enable_player(True)
            self._update_pos(self.target.start)
        except PlayerError as e:
            self._enable_player(False)
            self.pos_label.setText(str(e))

    def _switch_source(self):
        if self.current:
            was = self.player.playing
            self.player.stop()
            self._prepare_target()
            if was:
                self.play_event()

    def play_event(self):
        if not self.target or self.target.path is None:
            return
        try:
            self.player.play(self.target.start)
            self.pause_btn.setText("⏸ Pause")
        except PlayerError as e:          # e.g. no sound output device: shown in the panel, no pop-up
            self.pos_label.setText(f"Cannot play: {e}")

    def toggle_pause(self):
        if self.player.playing:
            self.player.pause()
            self.pause_btn.setText("▶ Continue")
        elif self.target and self.target.path:
            if self.player.finished or self.player._thread is None:
                self.player.play(self.player.position())
            else:
                self.player.resume()
            self.pause_btn.setText("⏸ Pause")

    def _seek(self):
        if not self.target or not self.player.duration:
            return
        t = self.slider.value() / 1000 * self.player.duration
        try:
            self.player.seek(t)
            if self.player.playing:
                self.pause_btn.setText("⏸ Pause")
        except PlayerError:
            pass

    def _tick(self):
        if self.player.path is not None and (self.player.playing or self.player.finished):
            self._update_pos(self.player.position())
            if self.player.finished:
                self.pause_btn.setText("▶ Continue")

    def _update_pos(self, t: float):
        d = max(0.001, self.player.duration)
        if not self.slider.isSliderDown():
            self.slider.setValue(int(t / d * 1000))
        ev = ""
        if self.target and self.target.event_start <= t <= self.target.event_end + 0.5:
            ev = "  ◀ detected sound"
        self.pos_label.setText(f"{fmt_offset(t)} / {fmt_offset(d)}{ev}")

    def _enable_player(self, on: bool):
        for w in (self.play_btn, self.pause_btn, self.slider):
            w.setEnabled(on)

    # ------------------------------------------------------------------ review
    def review(self, state: str):
        if not self.current or not self.res:
            return
        explore.set_review(self.res, self.current, state)
        self.reviews_changed = True
        for st, b in self.rev_btns.items():
            b.setChecked(st == state)
        keep = self.current["id"]
        self.refresh_view()
        self.select_id(keep)

    # ------------------------------------------------------------------ misc
    def _zoom_hour(self, date: str, hour: int):
        self.detail_hour = (date, hour)
        self.day_btn.show()
        self.tl_label.setText(f"<b>Timeline</b> {date} {hour:02d}:00–{(hour + 1) % 24:02d}:00")
        for w in (self.date_cb, self.h_from, self.h_to):
            w.blockSignals(True)
        self.date_cb.setCurrentIndex(max(0, self.date_cb.findData(date)))
        self.h_from.setCurrentIndex(hour)
        self.h_to.setCurrentIndex(hour)
        for w in (self.date_cb, self.h_from, self.h_to):
            w.blockSignals(False)
        self.refresh_view()

    def _back_to_day(self):
        self.detail_hour = None
        self.day_btn.hide()
        self.tl_label.setText("<b>Timeline</b> (click an hour to see it in detail)")
        for w in (self.h_from, self.h_to):
            w.blockSignals(True)
        self.h_from.setCurrentIndex(0)
        self.h_to.setCurrentIndex(23)
        for w in (self.h_from, self.h_to):
            w.blockSignals(False)
        self.refresh_view()

    def _open(self, p: Path):
        if p.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def closeEvent(self, e):
        self.player.stop()
        if self.reviews_changed and not store.analysis_busy(self.root):
            try:
                from ..sound import reports
                from ..sound.vocab import load as load_vocab

                voc = load_vocab()
                fp = store.fingerprint(voc)
                if store.same_fingerprint(self.res.fingerprint if self.res else None, fp):
                    data = reports.write_combined(self.root, fp, voc, self.res.status or "COMPLETE")
                    reports.write_annotated(self.root, data)
            except Exception:
                pass
        e.accept()
