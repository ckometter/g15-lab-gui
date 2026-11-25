"""The measurement queue: its runner and its panel (docs/MEASUREMENT_QUEUE.md).

The runner starts each entry through its tab's adapter. The adapter puts the
entry's settings in the tab's fields and calls the tab's own Start path, so a
queued run gets the same checks, ramps, autosave, filenames and metadata as
pressing Begin. The runner waits for that run to end and for its thread to
finish before it starts the next entry. The queue itself never sends an
instrument command.
"""
import time
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

import measurement_queue as mq
from sweep_timing import clock_text, format_duration


class QueueAdapter(QtCore.QObject):
    """What a measurement tab provides to join the queue.

    A subclass sets ``key`` (saved in every entry), ``title`` and ``tab_index`` (its
    dashboard tab) and implements the methods below. ``run_ended`` is emitted once
    after every run that ``start()`` began, with the outcome ("finished", "error" or
    "stopped"), a message, and a dict the record keeps (file, copy, timing, ...).
    """
    run_ended = QtCore.pyqtSignal(str, str, object)
    key = ""
    title = ""
    tab_index = None

    def capture(self):
        """The tab's present settings, as a JSON-ready dict."""
        raise NotImplementedError

    def capture_problem(self):
        """Why the tab's settings cannot be captured or shown now, or None."""
        return None

    def problems(self, settings):
        """Why these settings cannot run on the tab now; [] when they can. No instrument commands."""
        raise NotImplementedError

    def summary(self, settings):
        raise NotImplementedError

    def estimate(self, settings, start, end_ramp=False):
        """(seconds or None, gate voltages at the end of the run), for the queue's time estimate."""
        return None, start

    def start_voltages(self):
        return (0.0, 0.0)

    def show(self, settings):
        """Put the settings in the tab's fields, without instrument commands."""
        raise NotImplementedError

    def idle(self):
        """True when no run, ramp or worker thread of the tab is active."""
        raise NotImplementedError

    def start(self, settings, info, end_ramp=False):
        """Start a run as Begin does. None when it runs, else the reason it did not start."""
        raise NotImplementedError

    def stop_now(self):
        """Stop the present run as the tab's own Stop (Reset) does."""
        raise NotImplementedError

    def hold(self, held):
        """Lock (True) or release (False) the tab's inputs while the queue runs."""

    def envelope(self):
        """The tab settings the queue checks entries against, as the record keeps them."""
        return {}

    def progress(self):
        """(percent, paused) of the present run, or None."""
        return None

    def seconds_left(self):
        """Estimated seconds left in the present run, or None."""
        return None

    def data_root(self):
        """The folder that holds data/ and value_storage."""
        raise NotImplementedError

    def abort(self, reason):
        """The dashboard closes during a run. Keep its data; return what the record keeps."""
        return {}


class QueueRunner(QtCore.QObject):
    """Runs scheduled entries one after another, each through its tab's Start path."""
    changed = QtCore.pyqtSignal()
    ended = QtCore.pyqtSignal(str, str)  # outcome ("finished", "failed", "stopped"), message

    # The previous run's thread normally ends within milliseconds of its result.
    IDLE_TIMEOUT_S = 60.0

    def __init__(self, adapters, log=print, parent=None):
        super().__init__(parent)
        self.adapters = adapters
        self.log = log
        self.entries = []
        self.options = dict(mq.DEFAULT_OPTIONS)
        self.current = None
        self.state = "idle"  # "idle", "waiting" (for a tab to be idle) or "running"
        self.stop_requested = False
        self.record = None
        self.last_outcome = None
        self._waiting_since = 0.0
        self._entry_started = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._start_when_idle)
        for adapter in adapters.values():
            adapter.run_ended.connect(self._on_run_ended)

    def running(self):
        return self.state != "idle"

    def number(self, entry):
        return self.entries.index(entry) + 1

    def _adapters_used(self):
        return [self.adapters[key] for key in dict.fromkeys(entry.tab for entry in self.entries)
                if key in self.adapters]

    def start(self, entries, options):
        """Start the scheduled ``entries``; they must have been checked."""
        if self.running():
            raise RuntimeError("The queue is already running.")
        if not entries:
            raise ValueError("There is nothing to run.")
        self.entries = list(entries)
        self.options = dict(options)
        self.current = None
        self.stop_requested = False
        self.last_outcome = None
        for entry in self.entries:
            entry.result = {"status": mq.PENDING}
        used = self._adapters_used()
        started_at = datetime.now().astimezone()
        path = mq.record_path(used[0].data_root(), started_at)
        self.record = mq.QueueRecord(path, self.entries, self.options,
                                     {adapter.key: adapter.envelope() for adapter in used}, log=self.log)
        self.state = "waiting"
        self.record.write()
        self.log(f"Measurement queue: starting {len(self.entries)} "
                 f"entr{'y' if len(self.entries) == 1 else 'ies'}. Record: {path}")
        for adapter in used:
            adapter.hold(True)
        self._waiting_since = time.monotonic()
        self.changed.emit()
        self._timer.start()

    def _next_pending(self):
        return next((entry for entry in self.entries if entry.status == mq.PENDING), None)

    def _start_when_idle(self):
        entry = self._next_pending()
        if entry is None:
            self._finish("finished", f"All {len(self.entries)} entries ran.")
            return
        if self.stop_requested:
            self._finish("stopped", f"Stopped before entry {self.number(entry)}, as asked.")
            return
        adapter = self.adapters.get(entry.tab)
        if adapter is None:
            self._fail(entry, f"No tab here runs {entry.tab!r} entries.")
            return
        if not adapter.idle():
            if time.monotonic() - self._waiting_since > self.IDLE_TIMEOUT_S:
                self._fail(entry, f"The {adapter.title} tab was still busy {self.IDLE_TIMEOUT_S:g} s "
                                  "after the previous run; this entry was not started.")
            return
        self._timer.stop()
        self._start_entry(entry, adapter)

    def _start_entry(self, entry, adapter):
        number, total = self.number(entry), len(self.entries)
        # Checked before the queue started; again here, as Begin will check again too.
        try:
            problems = adapter.problems(entry.settings)
        except Exception as exc:
            problems = [f"The entry could not be checked: {type(exc).__name__}: {exc}"]
        if problems:
            self._fail(entry, " ".join(problems))
            return
        end_ramp = number == total and bool(self.options.get("ramp_to_zero_after_last"))
        # Saved in the run's files as queue_record, queue_entry, ... (save_sweep_data).
        info = {"record": str(self.record.path), "entry": number, "entries": total,
                "note": entry.note, "started_at": self.record.run["started_at"]}
        entry.result = {"status": mq.RUNNING, "started_at": mq.now_text()}
        if end_ramp and not entry.settings.get("ramp_to_zero"):
            entry.result["end_ramp_by_queue"] = True
        self.current = entry
        self.state = "running"
        self._entry_started = time.monotonic()
        self.record.write()
        self.log(f"Measurement queue {number}/{total}: starting {adapter.summary(entry.settings)}")
        self.changed.emit()
        try:
            reason = adapter.start(entry.settings, info, end_ramp=end_ramp)
        except Exception as exc:  # The queue must end rather than wait for a run that never began.
            reason = f"{type(exc).__name__}: {exc}"
        if reason is not None and self.current is entry:
            self.current = None
            self._fail(entry, f"Begin did not start the run: {reason}")

    def _on_run_ended(self, outcome, message, result):
        entry = self.current
        if self.state != "running" or entry is None:
            return
        self.current = None
        number, total = self.number(entry), len(self.entries)
        status = {"finished": mq.DONE, "error": mq.FAILED, "stopped": mq.STOPPED}.get(outcome, mq.FAILED)
        entry.result.update({key: value for key, value in (result or {}).items() if value is not None})
        entry.result.update(status=status, ended_at=mq.now_text(), message=message or "")
        took = format_duration(time.monotonic() - self._entry_started)
        files = [entry.result.get(key) for key in ("copy", "file") if entry.result.get(key)]
        saved = f" Saved {' and '.join(files)}." if files else " No data was saved."
        self.log(f"Measurement queue {number}/{total}: {status} after {took}."
                 + (f" {message}" if message else "") + saved)
        if status != mq.DONE:
            if status == mq.FAILED:
                self._finish("failed", f"Entry {number} failed: {message}")
            else:
                self._finish("stopped", f"Entry {number} was stopped" + (f": {message}" if message else "."))
            return
        if self._next_pending() is None:
            self._finish("finished", f"All {total} entries ran.")
            return
        if self.stop_requested:
            self._finish("stopped", f"Stopped after entry {number}, as asked.")
            return
        self.state = "waiting"
        self.record.write()
        self.adapters[entry.tab].hold(True)
        self._waiting_since = time.monotonic()
        self.changed.emit()
        self._timer.start()

    def _fail(self, entry, message):
        entry.result.update(status=mq.FAILED, ended_at=mq.now_text(), message=message)
        number = self.number(entry)
        self.log(f"Measurement queue {number}/{len(self.entries)}: failed. {message}")
        self._finish("failed", f"Entry {number} failed: {message}")

    def stop_after_current(self):
        """Let the present run finish (with its end ramp), then stop."""
        if self.running():
            self.stop_requested = True
            self.changed.emit()

    def stop_now(self):
        """Stop the present run as its tab's Stop does; between runs, stop at once."""
        if self.state == "running" and self.current is not None:
            self.adapters[self.current.tab].stop_now()
        elif self.state == "waiting":
            entry = self._next_pending()
            where = f" before entry {self.number(entry)}" if entry is not None else ""
            self._finish("stopped", f"Stopped{where} with Stop now.")

    def abort(self, reason):
        """End the queue at once (the dashboard closes); a running entry keeps its data."""
        if not self.running():
            return
        entry = self.current
        if entry is not None:
            self.current = None
            result = self.adapters[entry.tab].abort(reason) or {}
            entry.result.update({key: value for key, value in result.items() if value is not None})
            entry.result.update(status=mq.STOPPED, ended_at=mq.now_text(), message=reason)
        self._finish("stopped", reason)

    def _finish(self, outcome, message):
        self._timer.stop()
        for entry in self.entries:
            if entry.status == mq.RUNNING:
                entry.result["status"] = mq.STOPPED
            elif entry.status == mq.PENDING:
                entry.result["status"] = mq.NOT_RUN
        self.state = "idle"
        self.current = None
        self.stop_requested = False
        self.last_outcome = (outcome, message)
        if self.record is not None:
            self.record.finish(outcome, message)
        for adapter in self._adapters_used():
            adapter.hold(False)
        where = f" Record: {self.record.path}" if self.record is not None else ""
        self.log(f"Measurement queue {outcome}: {message}{where}")
        self.changed.emit()
        self.ended.emit(outcome, message)


def _button(text, slot, tooltip=""):
    button = QtWidgets.QPushButton(text)
    button.clicked.connect(lambda _checked=False: slot())
    if tooltip:
        button.setToolTip(tooltip)
    return button


class MeasurementQueuePanel(QtWidgets.QWidget):
    """The queue's list, buttons and status, shown in a dock of the dashboard."""
    title_changed = QtCore.pyqtSignal(str)

    COLUMNS = ("#", "Tab", "Settings", "Note", "Time", "State")
    NOTE_COLUMN = 3
    COLORS = {mq.DONE: "#047857", mq.FAILED: "#b91c1c", mq.STOPPED: "#b45309",
              mq.NOT_RUN: "#64748b", mq.RUNNING: "#1d4ed8", "problem": "#b45309"}

    def __init__(self, dashboard, adapters, parent=None, log=print):
        super().__init__(parent)
        self.dashboard = dashboard
        self.adapters = dict(adapters)
        self.log = log
        self.entries = []
        self.options = dict(mq.DEFAULT_OPTIONS)
        self.problems = {}   # id(entry): problems from the last check
        self.estimates = {}  # id(entry): seconds
        self.notice = ""     # the last action's result, shown while the queue is idle
        self.add_buttons = {}  # "Add to queue" buttons on the tabs: {button: adapter}
        self._title = ""
        self._filling = False
        self.runner = QueueRunner(self.adapters, log=log, parent=self)
        self.runner.changed.connect(self.refresh)
        self.runner.ended.connect(self.on_queue_ended)
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start()
        self.refresh()

    # --- Layout ---
    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
                                   | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.verticalHeader().hide()
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        for column in range(len(self.COLUMNS)):
            mode = (QtWidgets.QHeaderView.ResizeMode.Stretch if column == 2
                    else QtWidgets.QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(column, mode)
        for column, width in ((0, 30), (1, 110), (3, 150), (4, 80), (5, 300)):
            self.table.setColumnWidth(column, width)
        self.table.setMinimumHeight(110)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.itemSelectionChanged.connect(self.update_buttons)
        layout.addWidget(self.table)

        edit_row = QtWidgets.QHBoxLayout()
        self.add_btn = _button("Add current tab", self.add_current_tab,
                               "Add the open tab's settings as a new entry (Dual-Gate Sweep).")
        self.show_btn = _button("Show on tab", self.show_selected,
                                "Put the selected entry's settings in its tab's fields to look at or edit "
                                "them. Sends nothing to the instruments.")
        self.replace_btn = _button("Replace with tab", self.replace_selected,
                                   "Replace the selected entry with the tab's present settings.")
        self.up_btn = _button("Up", lambda: self.move_selected(-1))
        self.down_btn = _button("Down", lambda: self.move_selected(1))
        self.remove_btn = _button("Remove", self.remove_selected)
        self.open_btn = _button("Open…", self.open_queue_file, "Open a saved queue or a queue record.")
        self.save_btn = _button("Save…", self.save_queue_file, "Save the entries as a queue file (JSON).")
        for button in (self.add_btn, self.show_btn, self.replace_btn, self.up_btn, self.down_btn,
                       self.remove_btn, self.open_btn, self.save_btn):
            edit_row.addWidget(button)
        edit_row.addStretch(1)
        layout.addLayout(edit_row)

        run_row = QtWidgets.QHBoxLayout()
        self.end_ramp_box = QtWidgets.QCheckBox("Ramp gates to 0 V after the last entry")
        self.end_ramp_box.setToolTip(
            "After the last entry's complete sweep, ramp both gates to 0 V together, as its "
            "'Ramp gates to 0 V when complete' would. Stops and errors hold the gates.")
        self.end_ramp_box.setChecked(self.options["ramp_to_zero_after_last"])
        self.end_ramp_box.toggled.connect(self.on_end_ramp_toggled)
        run_row.addWidget(self.end_ramp_box)
        self.status_lbl = QtWidgets.QLabel()
        self.status_lbl.setWordWrap(True)
        run_row.addWidget(self.status_lbl, 1)
        self.run_btn = _button("Run queue", self.run_queue,
                               "Check every entry, then run them one after another through the "
                               "tab's own Begin.")
        self.stop_after_btn = _button("Stop after this run", self.stop_after_run,
                                      "Let the present run finish, then stop the queue.")
        self.stop_now_btn = _button("Stop now", self.stop_now,
                                    "Stop the present run as its tab's Stop (Reset) does: its data "
                                    "is saved and the gates stay where they are.")
        for button in (self.run_btn, self.stop_after_btn, self.stop_now_btn):
            run_row.addWidget(button)
        layout.addLayout(run_row)

    # --- Helpers ---
    def adapter_for(self, entry):
        return self.adapters.get(entry.tab)

    def current_adapter(self):
        tabs = getattr(self.dashboard, "tab_widget", None)
        index = tabs.currentIndex() if tabs is not None else None
        return next((adapter for adapter in self.adapters.values() if adapter.tab_index == index), None)

    def data_root(self):
        adapter = next(iter(self.adapters.values()), None)
        return Path(adapter.data_root()) if adapter is not None else Path.cwd()

    def selected_index(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else None

    def selected_entry(self):
        index = self.selected_index()
        return self.entries[index] if index is not None and index < len(self.entries) else None

    def select(self, index):
        if index is not None and 0 <= index < len(self.entries):
            self.table.selectRow(index)

    def show_tab(self, adapter):
        """Open the adapter's tab first: a tab change reloads that tab's fields from value_storage."""
        tabs = getattr(self.dashboard, "tab_widget", None)
        if tabs is None or adapter.tab_index is None:
            return True
        if tabs.currentIndex() != adapter.tab_index:
            tabs.setCurrentIndex(adapter.tab_index)
        return tabs.currentIndex() == adapter.tab_index

    def busy_elsewhere(self):
        dashboard = self.dashboard
        return dashboard is not None and hasattr(dashboard, "measurement_active") and dashboard.measurement_active()

    def check(self, entry):
        adapter = self.adapter_for(entry)
        problems = ([f"No tab here runs {entry.tab!r} entries."] if adapter is None
                    else adapter.problems(entry.settings))
        self.problems[id(entry)] = problems
        return problems

    def entries_changed(self):
        self.notice = ""
        self.update_estimates()
        self.refresh()

    def set_notice(self, text):
        self.notice = text
        self.refresh_status()

    def update_estimates(self, entries=None):
        """Chain the estimates: each run starts where the previous one leaves the gates."""
        entries = self.entries if entries is None else entries
        self.estimates = {key: value for key, value in self.estimates.items()
                          if key in {id(entry) for entry in self.entries}}
        start = None
        for number, entry in enumerate(entries, 1):
            adapter = self.adapter_for(entry)
            if adapter is None or self.problems.get(id(entry)):
                self.estimates[id(entry)] = None
                continue
            if start is None:
                start = adapter.start_voltages()
            last = number == len(entries) and self.options["ramp_to_zero_after_last"]
            seconds, start = adapter.estimate(entry.settings, start, end_ramp=last)
            self.estimates[id(entry)] = seconds
        return sum(self.estimates.get(id(entry)) or 0.0 for entry in entries)

    # --- Editing ---
    def add_current_tab(self):
        adapter = self.current_adapter()
        if adapter is None:
            QtWidgets.QMessageBox.information(
                self, "Measurement queue",
                "Only Dual-Gate Sweep runs can be queued so far. Open that tab, set up the "
                "sweep and add it from there.")
            return None
        return self.add_from(adapter)

    def add_from(self, adapter, show_dock=False):
        """Add the tab's present settings; refused, with the reason, if Begin would refuse them."""
        if self.runner.running():
            return None
        problem = adapter.capture_problem()
        if problem:
            QtWidgets.QMessageBox.warning(self, "Not added to the queue", problem)
            return None
        settings = adapter.capture()
        problems = adapter.problems(settings)
        if problems:
            QtWidgets.QMessageBox.warning(self, "Not added to the queue",
                                          "Begin would refuse these settings:\n\n" + "\n".join(problems))
            return None
        entry = mq.QueueEntry(adapter.key, settings)
        self.entries.append(entry)
        self.problems[id(entry)] = []
        self.entries_changed()
        self.select(len(self.entries) - 1)
        if show_dock:
            dock = getattr(self.dashboard, "queue_dock", None)
            if dock is not None:
                dock.show()
                dock.raise_()
        return entry

    def show_selected(self):
        entry = self.selected_entry()
        adapter = self.adapter_for(entry) if entry is not None else None
        if adapter is None or self.runner.running():
            return False
        problem = adapter.capture_problem()
        if problem:
            QtWidgets.QMessageBox.warning(self, "Measurement queue", problem)
            return False
        if not self.show_tab(adapter):
            QtWidgets.QMessageBox.warning(self, "Measurement queue",
                                          f"The {adapter.title} tab cannot be opened while another "
                                          "measurement or movement is active.")
            return False
        adapter.show(entry.settings)
        self.set_notice(f"Entry {self.entries.index(entry) + 1} is shown on the {adapter.title} tab. "
                        "Edit it there, then press Replace with tab.")
        return True

    def replace_selected(self):
        entry = self.selected_entry()
        adapter = self.adapter_for(entry) if entry is not None else None
        if adapter is None or self.runner.running():
            return False
        if self.current_adapter() is not adapter:
            QtWidgets.QMessageBox.information(self, "Measurement queue",
                                              f"Open the {adapter.title} tab first: the entry is "
                                              "replaced with that tab's settings.")
            return False
        problem = adapter.capture_problem()
        if problem:
            QtWidgets.QMessageBox.warning(self, "Not replaced", problem)
            return False
        settings = adapter.capture()
        problems = adapter.problems(settings)
        if problems:
            QtWidgets.QMessageBox.warning(self, "Not replaced",
                                          "Begin would refuse these settings:\n\n" + "\n".join(problems))
            return False
        entry.settings = settings
        entry.result = {}
        self.problems[id(entry)] = []
        self.entries_changed()
        return True

    def move_selected(self, delta):
        index = self.selected_index()
        if index is None or self.runner.running():
            return
        target = index + delta
        if 0 <= target < len(self.entries):
            self.entries[index], self.entries[target] = self.entries[target], self.entries[index]
            self.entries_changed()
            self.select(target)

    def remove_selected(self):
        index = self.selected_index()
        if index is None or self.runner.running():
            return
        entry = self.entries.pop(index)
        self.problems.pop(id(entry), None)
        self.estimates.pop(id(entry), None)
        self.entries_changed()
        self.select(min(index, len(self.entries) - 1))

    def on_item_changed(self, item):
        if self._filling or item.column() != self.NOTE_COLUMN or self.runner.running():
            return
        if item.row() < len(self.entries):
            self.entries[item.row()].note = item.text()

    def on_end_ramp_toggled(self, checked):
        self.options["ramp_to_zero_after_last"] = bool(checked)
        self.entries_changed()

    # --- Files ---
    def queue_folder(self):
        folder = self.data_root() / "data" / "queues"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_queue_file(self, path=None):
        if not self.entries:
            QtWidgets.QMessageBox.information(self, "Measurement queue", "The queue is empty.")
            return None
        if path is None:
            default = self.queue_folder() / datetime.now().strftime("%Y_%m_%d_queue.json")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save measurement queue", str(default),
                                                            "Queue files (*.json);;All files (*.*)")
            if not path:
                return None
        try:
            mq.save_queue_file(path, self.entries, self.options)
        except (OSError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Queue not saved", str(exc))
            return None
        self.set_notice(f"Saved {Path(path).name}.")
        return str(path)

    def open_queue_file(self, path=None, replace=None):
        """Open a queue or a record. ``replace``: None asks when the queue has entries."""
        if self.runner.running():
            return None
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open measurement queue",
                                                            str(self.queue_folder()),
                                                            "Queue files (*.json);;All files (*.*)")
            if not path:
                return None
        try:
            entries, options = mq.load_queue_file(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Queue not opened", str(exc))
            return None
        if replace is None:
            replace = True
            if self.entries:
                box = QtWidgets.QMessageBox(self)
                box.setWindowTitle("Open measurement queue")
                box.setText(f"The queue has {len(self.entries)} entries. Replace them with the "
                            f"{len(entries)} entries of {Path(path).name}, or add those after them?")
                replace_button = box.addButton("Replace", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                add_button = box.addButton("Add after", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
                box.exec()
                if box.clickedButton() not in (replace_button, add_button):
                    return None
                replace = box.clickedButton() is replace_button
        if replace:
            self.entries = []
            self.problems = {}
            self.options = options
            with QtCore.QSignalBlocker(self.end_ramp_box):
                self.end_ramp_box.setChecked(options["ramp_to_zero_after_last"])
        self.entries.extend(entries)
        bad = [self.entries.index(entry) + 1 for entry in entries if self.check(entry)]
        self.entries_changed()
        text = f"Opened {Path(path).name}: {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}."
        if len(bad) == 1:
            text += f" Entry {bad[0]} needs a change before it can run (see State)."
        elif bad:
            text += f" Entries {', '.join(map(str, bad))} need changes before they can run (see State)."
        self.set_notice(text)
        return entries

    # --- Running ---
    def run_queue(self, confirm=True):
        """Check every entry that is not done, then start the queue. True when it started."""
        if self.runner.running():
            return False
        if not self.entries:
            QtWidgets.QMessageBox.information(self, "Measurement queue", "The queue is empty.")
            return False
        if self.busy_elsewhere():
            QtWidgets.QMessageBox.warning(self, "Queue not started",
                                          "Finish the active measurement or movement first.")
            return False
        todo = [entry for entry in self.entries if entry.status != mq.DONE]
        if not todo:
            if confirm and QtWidgets.QMessageBox.question(
                    self, "Run the queue again?",
                    "Every entry is done. Run them all again?") != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
            todo = list(self.entries)
        for adapter in {self.adapter_for(entry) for entry in todo} - {None}:
            problem = adapter.capture_problem()  # The tab must be idle and show live settings.
            if problem:
                QtWidgets.QMessageBox.warning(self, "Queue not started", problem)
                return False
        lines = []
        for entry in todo:
            problems = self.check(entry)
            if problems:
                lines.append(f"Entry {self.entries.index(entry) + 1}: {' '.join(problems)}")
        self.update_estimates()
        self.refresh()
        if lines:
            more = f"\n… and {len(lines) - 10} more." if len(lines) > 10 else ""
            QtWidgets.QMessageBox.critical(self, "Queue not started",
                                           "Nothing was started or moved. Change these entries first:\n\n"
                                           + "\n".join(lines[:10]) + more)
            return False
        adapter = self.adapter_for(todo[0])
        if confirm:
            seconds = self.update_estimates(todo)
            known = all(self.estimates.get(id(entry)) is not None for entry in todo)
            numbers = ", ".join(str(self.entries.index(entry) + 1) for entry in todo)
            when = (f", about {format_duration(seconds)}, ending about "
                    f"{clock_text(time.time() + seconds, time.time())}" if known and seconds else "")
            after = ("Both gates ramp to 0 V after the last entry." if self.options["ramp_to_zero_after_last"]
                     else "The gates stay where the last entry leaves them.")
            text = (f"Run {len(todo)} entr{'y' if len(todo) == 1 else 'ies'} ({numbers}) on the "
                    f"{adapter.title} tab{when}?\n\nEach run starts as Begin does, with its checks, current "
                    "limits and protected ramps. Between runs the gates move only by Begin's own ramp to "
                    f"the next first point. {after} An error stops the queue and holds the gates.")
            if QtWidgets.QMessageBox.question(self, "Run the queue?", text) != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        if not self.show_tab(adapter):
            QtWidgets.QMessageBox.warning(self, "Queue not started",
                                          f"The {adapter.title} tab cannot be opened now.")
            return False
        self.notice = ""
        self.runner.start(todo, self.options)
        self.update_estimates(todo)
        return True

    def stop_after_run(self):
        self.runner.stop_after_current()

    def stop_now(self):
        self.runner.stop_now()

    def abort(self, reason):
        self.runner.abort(reason)

    def on_queue_ended(self, outcome, message):
        self.notice = f"Queue {outcome}: {message}"
        self.refresh()
        record = self.runner.record.path if self.runner.record is not None else None
        # Shown after the run's own handlers have returned; nothing waits on these.
        if outcome == "finished":
            QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.information(
                self, "Queue finished", f"{message}\n\nRecord: {record}"))
        elif outcome == "failed":
            QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.critical(
                self, "Queue stopped",
                f"{message}\n\nThe run's data is saved and the gates were held where they were. "
                f"Later entries did not run.\n\nRecord: {record}"))

    # --- Display ---
    def summary(self, entry):
        adapter = self.adapter_for(entry)
        return adapter.summary(entry.settings) if adapter is not None else f"({entry.tab} entry)"

    def time_text(self, entry):
        seconds = self.estimates.get(id(entry))
        return format_duration(seconds) if seconds is not None else ""

    def state_text(self, entry):
        status, result = entry.status, entry.result
        if status == mq.PENDING:
            problems = self.problems.get(id(entry))
            return f"needs a change: {problems[0]}" if problems else "pending"
        if status == mq.RUNNING:
            adapter = self.adapter_for(entry)
            progress = adapter.progress() if adapter is not None else None
            if progress is None:
                return "running"
            percent, paused = progress
            return f"{'paused' if paused else 'running'} {percent} %"
        if status == mq.DONE:
            name = Path(result.get("copy") or result.get("file") or "").name
            return f"done · {name}" if name else "done"
        if status in (mq.FAILED, mq.STOPPED):
            message = result.get("message")
            return f"{status}: {message}" if message else status
        return status

    def state_color(self, entry):
        if entry.status == mq.PENDING:
            return self.COLORS["problem"] if self.problems.get(id(entry)) else None
        return self.COLORS.get(entry.status)

    def state_tooltip(self, entry):
        lines = list(self.problems.get(id(entry)) or []) if entry.status == mq.PENDING else []
        result = entry.result
        for key, label in (("message", "Message"), ("copy", "File"), ("file", "Autosave"),
                           ("timing", "Timing"), ("started_at", "Started"), ("ended_at", "Ended")):
            if result.get(key):
                lines.append(f"{label}: {result[key]}")
        if result.get("gates_after_V"):
            lines.append("Gates afterwards: V1 = {:+.4f} V, V2 = {:+.4f} V".format(*result["gates_after_V"]))
        if result.get("end_ramp_by_queue"):
            lines.append("The queue added the ramp to 0 V after this last entry.")
        return "\n".join(lines)

    def refresh(self):
        selected = self.selected_index()
        running = self.runner.running()
        self._filling = True
        try:
            self.table.setRowCount(len(self.entries))
            for row, entry in enumerate(self.entries):
                adapter = self.adapter_for(entry)
                texts = (str(row + 1), adapter.title if adapter is not None else entry.tab,
                         self.summary(entry), entry.note, self.time_text(entry), self.state_text(entry))
                for column, text in enumerate(texts):
                    item = self.table.item(row, column)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self.table.setItem(row, column, item)
                    item.setText(text)
                    flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
                    if column == self.NOTE_COLUMN and not running:
                        flags |= QtCore.Qt.ItemFlag.ItemIsEditable
                    item.setFlags(flags)
                self.table.item(row, 2).setToolTip("\n".join(f"{key}: {value}"
                                                             for key, value in entry.settings.items()))
                state = self.table.item(row, 5)
                state.setToolTip(self.state_tooltip(entry))
                color = self.state_color(entry)
                state.setForeground(QtGui.QBrush(QtGui.QColor(color)) if color else QtGui.QBrush())
        finally:
            self._filling = False
        if selected is not None:
            self.select(min(selected, len(self.entries) - 1))
        self.refresh_status()

    def queue_end_text(self):
        """'ends about tomorrow 05:10' from the present run's estimate and the later entries'."""
        runner = self.runner
        left = 0.0
        if runner.current is not None:
            adapter = self.adapter_for(runner.current)
            current = adapter.seconds_left() if adapter is not None else None
            if current is None:
                return ""
            left += current
        for entry in runner.entries:
            if entry.status == mq.PENDING:
                seconds = self.estimates.get(id(entry))
                if seconds is None:
                    return ""
                left += seconds
        return f" · ends about {clock_text(time.time() + left, time.time())}"

    def refresh_status(self):
        runner = self.runner
        if runner.running():
            current = runner.current or runner._next_pending()
            number = runner.number(current) if current is not None else len(runner.entries)
            text = f"Running entry {number} of {len(runner.entries)}" + self.queue_end_text()
            if runner.stop_requested:
                text += " · stopping after this run"
            title = f"Measurement queue · running {number} of {len(runner.entries)}"
            # The running entry's progress.
            if runner.current is not None and runner.current in self.entries:
                item = self.table.item(self.entries.index(runner.current), 5)
                if item is not None:
                    item.setText(self.state_text(runner.current))
        else:
            count = len(self.entries)
            if self.notice:
                text = self.notice
            elif count:
                total = sum(self.estimates.get(id(entry)) or 0.0 for entry in self.entries
                            if entry.status != mq.DONE)
                text = f"{count} entr{'y' if count == 1 else 'ies'}" + (
                    f" · about {format_duration(total)}" if total else "")
            else:
                text = "Empty. Set up a Dual-Gate Sweep and press Add to queue on its tab."
            title = "Measurement queue" + (f" · {count}" if count else "")
        if self.status_lbl.text() != text:
            self.status_lbl.setText(text)
        if title != self._title:
            self._title = title
            self.title_changed.emit(title)
        self.update_buttons()

    def update_buttons(self):
        running = self.runner.running()
        selected = self.selected_entry() is not None
        for button in (self.show_btn, self.replace_btn, self.up_btn, self.down_btn, self.remove_btn):
            button.setEnabled(not running and selected)
        adapter = self.current_adapter()
        self.add_btn.setEnabled(not running and adapter is not None)
        self.open_btn.setEnabled(not running)
        self.save_btn.setEnabled(not running and bool(self.entries))
        self.end_ramp_box.setEnabled(not running)
        self.run_btn.setEnabled(not running and bool(self.entries))
        self.stop_after_btn.setEnabled(running and not self.runner.stop_requested)
        self.stop_now_btn.setEnabled(running)
        for button, tab_adapter in self.add_buttons.items():
            button.setEnabled(not running and tab_adapter.capture_problem() is None)
