"""Helium level on the dashboard: background reader, status-bar indicator and plot panel.

The reader thread looks at the level meter's folder every 30 s, re-reads
LevelData.nc (from a copy) only when the file changed, and never touches an
instrument, so measurements never wait for it. The level meter itself is a
separate program; see ``levelmeter_monitor.py`` and docs/LEVELMETER.md.
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from levelmeter_monitor import (ATTENTION, DATA_NAME, INACTIVE, LOW_LEVEL_MM, NORMAL, PROBLEM, SLOPE_WINDOW_H,
                                STOP_NAME, levelmeter_dir, read_history, summarize)

POLL_S = 30.0
GAP_S = 3600.0  # readings come every 10 min: a longer gap is drawn as a break
COLORS = {NORMAL: "#047857", ATTENTION: "#b45309", PROBLEM: "#b91c1c", INACTIVE: "#64748b"}


class LevelmeterWorker(QtCore.QThread):
    """Polls the level meter's folder; emits a LevelSummary every ``poll_s`` seconds."""
    updated = QtCore.pyqtSignal(object)

    def __init__(self, folder=None, poll_s=POLL_S, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.poll_s = float(poll_s)
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        stamp, times, levels = None, np.zeros(0), np.zeros(0)
        while not self._stop_requested:
            folder = Path(self.folder) if self.folder else levelmeter_dir()
            error = None
            try:
                current = (folder / DATA_NAME).stat().st_mtime_ns
                if current != stamp:
                    times, levels = read_history(folder / DATA_NAME)
                    stamp = current
            except Exception as exc:  # missing folder, or the file is being written: retry next poll
                error = f"{type(exc).__name__}: {exc}"
            try:
                stop_file = folder / STOP_NAME
                stopped_since = stop_file.stat().st_mtime if stop_file.exists() else None
            except OSError:
                stopped_since = None
            self.updated.emit(summarize(times, levels, now=time.time(), stopped_since=stopped_since,
                                        error=error, folder=folder))
            deadline = time.monotonic() + self.poll_s
            while not self._stop_requested and time.monotonic() < deadline:
                self.msleep(100)


class LevelmeterStatusLabel(QtWidgets.QLabel):
    """Status-bar text on every tab; red when the level meter stopped recording or reads ~0 mm."""
    clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("LHe: reading the level meter...", parent)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.severity = INACTIVE
        self._style(INACTIVE)

    def _style(self, severity):
        base = "font-weight: bold; font-family: Consolas, monospace; font-size: 11px; padding: 0 6px;"
        if severity == PROBLEM:
            self.setStyleSheet(base + f"color: #ffffff; background-color: {COLORS[PROBLEM]};")
        else:
            self.setStyleSheet(base + f"color: {COLORS[severity]};")

    def show_summary(self, summary):
        self.severity = summary.severity
        self.setText(summary.text)
        self.setToolTip(f"{summary.detail}\n\nClick to show the helium plot (Dashboard tab).")
        self._style(summary.severity)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


def with_gaps(times, levels, gap=GAP_S):
    """Insert NaN where readings are more than ``gap`` apart, so outages show as breaks."""
    times, levels = np.asarray(times, dtype=float), np.asarray(levels, dtype=float)
    breaks = np.flatnonzero(np.diff(times) > gap) + 1
    if not breaks.size:
        return times, levels
    return (np.insert(times, breaks, times[breaks - 1] + gap / 2),
            np.insert(levels, breaks, np.nan))


class LevelmeterPanel(QtWidgets.QGroupBox):
    """14-day history and 24-hour trend of the helium level, for the Dashboard tab."""

    def __init__(self, parent=None):
        super().__init__("Liquid helium level", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 16, 10, 10)
        self.summary_label = QtWidgets.QLabel("Reading the level meter...")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary_label)
        self.history_plot = self._plot("Last 14 days")
        self.history_curve = self.history_plot.plot(pen=pg.mkPen("#1769aa", width=1.5), connect="finite")
        self.trend_plot = self._plot("Last 24 hours")
        self.trend_curve = self.trend_plot.plot(pen=pg.mkPen("#1769aa", width=1.5), symbol="o", symbolSize=4,
                                                symbolPen=pg.mkPen("#1769aa"), symbolBrush=pg.mkBrush("#1769aa"),
                                                connect="finite")
        self.fit_curve = self.trend_plot.plot(pen=pg.mkPen("#c2410c", width=2, style=QtCore.Qt.PenStyle.DashLine))
        layout.addWidget(self.history_plot, 2)
        layout.addWidget(self.trend_plot, 1)
        note = QtWidgets.QLabel("Read-only: the level meter is a separate program that keeps running when the "
                                "dashboard is closed. Start, stop and check it with Levelmeter.bat, "
                                "Levelmeter-Stop.bat and Levelmeter-Status.bat in its folder.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #475569; font-size: 11px;")
        layout.addWidget(note)
        self.setMinimumWidth(380)

    @staticmethod
    def _plot(title):
        plot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()}, title=title)
        plot.setBackground("#ffffff")
        plot.setLabel("left", "Level", units="mm")
        plot.getAxis("left").enableAutoSIPrefix(False)
        for side in ("left", "bottom"):
            plot.getAxis(side).setPen(pg.mkPen("#334155"))
            plot.getAxis(side).setTextPen(pg.mkPen("#334155"))
        plot.setMinimumHeight(170)
        # Fill the height the other Dashboard panels leave, rather than asking for more (no scrolling).
        plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored)
        threshold = pg.InfiniteLine(pos=LOW_LEVEL_MM, angle=0,
                                    pen=pg.mkPen("#d97706", width=1.2, style=QtCore.Qt.PenStyle.DashLine),
                                    label=f"low level {LOW_LEVEL_MM:.0f} mm",
                                    labelOpts={"position": 0.08, "color": "#b45309"})
        # Not part of the automatic range: the 24-hour view zooms to the readings.
        plot.addItem(threshold, ignoreBounds=True)
        return plot

    def show_summary(self, summary):
        self.summary_label.setText(summary.detail)
        self.summary_label.setStyleSheet(f"color: {COLORS[summary.severity]}; font-weight: bold;")
        times, levels = summary.times, summary.levels
        if not times.size:
            for curve in (self.history_curve, self.trend_curve, self.fit_curve):
                curve.setData([], [])
            return
        self.history_curve.setData(*with_gaps(times, levels))
        self.history_plot.setYRange(0, max(800.0, float(np.nanmax(levels)) + 20), padding=0)
        recent = times >= summary.latest_time - SLOPE_WINDOW_H * 3600
        self.trend_curve.setData(*with_gaps(times[recent], levels[recent]))
        if math.isfinite(summary.slope) and np.count_nonzero(recent) >= 2:
            hours = (times[recent] - summary.latest_time) / 3600.0
            slope, intercept = np.polyfit(hours, levels[recent], 1)
            self.fit_curve.setData(times[recent], slope * hours + intercept)
            self.trend_plot.setTitle(f"Last 24 hours: {summary.slope:+.2f} mm/h")
        else:
            self.fit_curve.setData([], [])
            self.trend_plot.setTitle("Last 24 hours")
