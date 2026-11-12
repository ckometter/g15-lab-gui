"""Shared window, sweep engine and file helpers for the gate-map tabs.

The Capacitance and Transport tabs sweep the same three electrodes (top gate,
bottom gate, sample) over the same coordinates and differ only in what they
read at each point. ``GateMapWindow`` holds what they share: the measurement,
sweep, run, electrode and live panels, coordinated manual moves, pre-start
validation and preview, the worker threads, maps with cuts and a crosshair,
autosave, saving and reopening files. A subclass supplies its instrument
panel, its worker-side point read and its file format through the hooks
marked "subclass hook" below.

Hardware access happens only in worker threads. Every physical target of the
complete sweep is validated before anything moves; nothing is clipped.
"""
import json
import math
import os
import sys
import time
import traceback

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

try:
    from .utils import EnhancedViewBox, configure_plot_axes
except ImportError:
    _measurements_dir = os.path.dirname(os.path.abspath(__file__))
    if _measurements_dir not in sys.path:
        sys.path.insert(0, _measurements_dir)
    from utils import EnhancedViewBox, configure_plot_axes

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from drivers.stanford_sr830 import OVERLOAD_MASK, UNLOCK_MASK
from electrode_outputs import DEFAULT_WIRING, OUTPUT_CHOICES, ElectrodeSet, MeasurementStopped, RunControl
from electrode_sweep import (ELECTRODES, ELECTRODE_LABELS, ELECTRODE_SYMBOLS, ElectrodeLimits, build_sweep_plan,
                             coordinate_names, density_field, direct_voltages, electrode_voltages,
                             estimate_duration, plan_summary, transform_description)
from gate_safety import GateRampCancelled
from measurement_data_paths import (dated_measurement_filename, dated_measurement_path, measurement_data_dir,
                                    measurement_device_modes, measurement_start_text, new_measurement_start)
from measurement_filenames import axis_token, number_token
from measurements.Dual_gate_sweep_gui import ScientificDoubleSpinBox, ScientificPointSpinBox

pg.setConfigOption('background', '#ffffff')
pg.setConfigOption('foreground', '#1e293b')

# Settings every gate-map tab stores; a subclass adds its own to DEFAULTS.
COMMON_DEFAULTS = {
    "fixed": "bottom",
    "v_fixed": 0.0,
    "delta": 0.0,
    "coordinate_mode": "density",
    "x_start": -1.0, "x_end": 1.0, "x_points": 101,
    "y_start": -1.0, "y_end": 1.0, "y_points": 1,
    "fast_axis": "x",
    "serpentine": False,
    "point_settle": 0.5,
    "line_settle": 3.0,
    "ramp_to_zero": True,
    "wiring": dict(DEFAULT_WIRING),
    "dac_ramp_rate": 1.0,
    "max_abs": 10.0,
    "max_top_sample": 10.0,
    "max_bottom_sample": 10.0,
    "aux_adc": "none",
    "sample_name": "",
    "comment": "",
}
# Per-point readbacks recorded by every gate-map tab (filled in by ``run_plan``).
ELECTRODE_POINT_FIELDS = ("v_top", "v_bottom", "v_sample", "leakage_top", "leakage_bottom", "leakage_sample",
                          "aux_adc", "timestamp")


def finite_levels(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        pad = max(abs(low) * 1e-6, 1e-15)
        return low - pad, high + pad
    return low, high


def image_extent(coords):
    coords = np.asarray(coords, dtype=float)
    low, high = float(np.min(coords)), float(np.max(coords))
    if coords.size > 1 and high > low:
        step = (high - low) / (coords.size - 1)
        return low - step / 2, (high - low) + step
    return low - 0.5, 1.0


def format_si(value, unit, precision=4):
    if value is None or not np.isfinite(value):
        return "--"
    return pg.siFormat(float(value), precision=precision, suffix=unit)


# ---------------------------------------------------------------------------
# Worker-side operations (no GUI access; everything needed is passed in)
# ---------------------------------------------------------------------------

def run_plan(worker, station, settings, plan, electrodes, *, measure, before_line=None):
    """Visit every point of ``plan`` in acquisition order with coordinated moves.

    ``measure(iy, ix, on_pause)`` settles and reads the instruments at one point
    and returns a dict of values; the electrode readbacks, the optional ADC and
    a timestamp are added here. ``before_line(line_number, points, first)``
    runs before each line's first move (e.g. to rebalance). After a complete
    run the driven electrodes optionally ramp to 0 V together.
    """
    control = worker.control
    emit = worker.emit
    moved = lambda values: emit("electrodes", values=values)
    paused = lambda: emit("status", text="Status: Paused")
    aux = settings.get("aux_adc", "none")
    dac = station.components.get("dac") if aux != "none" else None
    total, done = len(plan.order), 0
    for line_number, (_, points) in enumerate(plan.lines()):
        control.wait_if_paused(paused)
        first = plan.targets(*points[0])
        if before_line is not None:
            before_line(line_number, points, first)
        electrodes.move_to(first, control, moved)
        if settings["line_settle"] > 0:
            emit("status", text=f"Status: Line {line_number + 1} settling...")
            control.sleep(settings["line_settle"], paused)
        for iy, ix in points:
            control.wait_if_paused(paused)
            electrodes.move_to(plan.targets(iy, ix), control, moved)
            values = measure(iy, ix, paused)
            readbacks = electrodes.readbacks()
            values.update({
                "v_top": readbacks["top"][0], "v_bottom": readbacks["bottom"][0],
                "v_sample": readbacks["sample"][0], "leakage_top": readbacks["top"][1],
                "leakage_bottom": readbacks["bottom"][1], "leakage_sample": readbacks["sample"][1]})
            values["aux_adc"] = float(dac.adc(int(aux[-1])).voltage()) if dac is not None else math.nan
            values["timestamp"] = time.time()
            done += 1
            emit("point", iy=iy, ix=ix, values=values)
            emit("progress", done=done, total=total)
            if done == 1 or done % 10 == 0 or done == total:
                emit("status", text=f"Status: Sweeping... ({done}/{total})")
    if settings["ramp_to_zero"]:
        emit("status", text="Status: Ramping driven electrodes to 0 V...")
        zero = {role: (0.0 if electrodes.outputs[role].driven else electrodes.outputs[role].setpoint())
                for role in ELECTRODES}
        electrodes.move_to(zero, control, moved)


def run_move(worker, electrodes, targets):
    emit = worker.emit
    emit("status", text="Status: Ramping electrodes...")
    electrodes.preflight()
    electrodes.move_to(targets, worker.control, lambda values: emit("electrodes", values=values))
    emit("electrodes", values=electrodes.setpoints())
    return "Electrodes at the requested voltages."


class GateMapTask(QtCore.QThread):
    """Runs one worker-side function with stop/pause and an event channel."""
    # Not "event": that name would shadow the virtual QObject.event() Qt calls for every event.
    task_event = QtCore.pyqtSignal(object)
    completed = QtCore.pyqtSignal(str, str)
    # Exceptions meaning the user stopped: outputs hold their last setpoints.
    STOP_EXCEPTIONS = (MeasurementStopped,)

    def __init__(self, function, *args, stop_callbacks=(), parent=None):
        super().__init__(parent)
        self.control = RunControl()
        for callback in stop_callbacks:
            self.control.on_stop(callback)
        self._function = function
        self._args = args

    def stop(self):
        self.control.stop()

    def pause(self):
        self.control.pause()

    def resume(self):
        self.control.resume()

    def emit(self, kind, **values):
        values["kind"] = kind
        self.task_event.emit(values)

    def run(self):
        outcome, message = "complete", ""
        try:
            message = self._function(self, *self._args) or ""
        except self.STOP_EXCEPTIONS:
            outcome, message = "stopped", "Stopped by user; outputs hold their last setpoints."
        except GateRampCancelled as exc:
            outcome = "stopped" if self.control.stopped else "error"
            message = "Stopped by user; outputs hold their last setpoints." if self.control.stopped else str(exc)
        except Exception as exc:
            outcome, message = "error", str(exc) or type(exc).__name__
            self.emit("log", text=traceback.format_exc())
        self.completed.emit(outcome, message)


class GateMapLiveWorker(QtCore.QThread):
    """Idle polling of the electrode setpoints plus a subclass's instruments."""
    updated = QtCore.pyqtSignal(object)

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self._stop_requested = False
        self._is_paused = False
        self._actively_polling = False
        self._electrodes = None
        self._last_leakage = 0.0

    def set_electrodes(self, electrodes):
        self._electrodes = electrodes

    def stop(self):
        self._stop_requested = True

    def pause(self):
        self._is_paused = True
        deadline = time.time() + 1.0
        while self._actively_polling and time.time() < deadline:
            time.sleep(0.01)

    def resume(self):
        self._is_paused = False

    def run(self):
        while not self._stop_requested:
            if self._is_paused or getattr(self.station, "live_status_paused", False):
                self.msleep(100)
                continue
            self._actively_polling = True
            try:
                self.updated.emit(self.poll())
            except Exception as exc:
                self.updated.emit({"error": str(exc)})
            finally:
                self._actively_polling = False
            self.msleep(250)

    def poll(self):
        report = {}
        now = time.monotonic()
        self.poll_instruments(report, now)
        electrodes = self._electrodes
        if electrodes is not None:
            values = {}
            leakage = now - self._last_leakage > 1.0
            for role, output in electrodes.outputs.items():
                try:
                    setpoint = output.setpoint()
                    current = output.readback()[1] if leakage and output.kind == "smu" else None
                except Exception:
                    setpoint, current = math.nan, None
                values[role] = (setpoint, current)
            if leakage:
                self._last_leakage = now
            report["electrodes"] = values
        return report

    def poll_instruments(self, report, now):
        """Subclass hook: add the tab's instrument readings to ``report``."""


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def grid_writer(f, plan):
    """Return ``grid(name, values, units, description)`` for [y, x] datasets of ``plan``."""
    x_name, y_name = plan.names

    def grid(name, values, units="", description=""):
        dataset = f.create_dataset(name, data=values)
        dataset.attrs["units"] = units
        if description:
            dataset.attrs["description"] = description
        if np.ndim(values) == 2:
            dataset.dims[0].label = y_name
            dataset.dims[1].label = x_name
        return dataset
    return grid


def write_file_attributes(f, attrs, schema):
    for key, value in attrs.items():
        f.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
    f.attrs["schema"] = schema


def write_plan_datasets(f, plan, grid):
    """Sweep coordinates, requested electrode voltages and (n0, p0) of every target."""
    x_name, y_name = plan.names
    f.create_dataset("x_coords", data=plan.x).attrs.update(units="V", name=x_name)
    f.create_dataset("y_coords", data=plan.y).attrs.update(units="V", name=y_name)
    grid("target_v_top", plan.vt, "V", "Requested top-gate voltage")
    grid("target_v_bottom", plan.vb, "V", "Requested bottom-gate voltage")
    grid("target_v_sample", plan.vs, "V", "Requested sample voltage")
    grid("n0", plan.n0, "V", "Normalized density (1+delta)(Vt-Vs)+(1-delta)(Vb-Vs)")
    grid("p0", plan.p0, "V", "Normalized field (1+delta)(Vt-Vs)-(1-delta)(Vb-Vs)")


def write_metadata(f, settings, value_storage):
    string = h5py.string_dtype(encoding="utf-8")
    meta = f.create_group("metadata")
    meta.create_dataset("settings", data=json.dumps(settings), dtype=string)
    meta.create_dataset("value_storage", data=json.dumps(value_storage), dtype=string)


def read_plan_datasets(f):
    return {"x": np.array(f["x_coords"]), "y": np.array(f["y_coords"]),
            "vt": np.array(f["target_v_top"]), "vb": np.array(f["target_v_bottom"]),
            "vs": np.array(f["target_v_sample"]), "n0": np.array(f["n0"]), "p0": np.array(f["p0"]),
            "order": np.array(f["acquisition_order"])}


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class GateMapWindow(QtWidgets.QMainWindow):
    """Base dashboard tab for maps and line sweeps over the three electrodes."""

    STORAGE_KEY = ""
    DEFAULTS = COMMON_DEFAULTS
    DEFAULT_WIRING = DEFAULT_WIRING
    POINT_FIELDS = ELECTRODE_POINT_FIELDS
    INT_FIELDS = {}                     # name -> fill value of integer per-point arrays
    WINDOW_TITLE = ""
    MEASUREMENT_NAME = "Measurement"    # message boxes: "<name> blocked", "<name> failed"
    TAB_NAME = ""                       # manual-action errors: "<tab> tab"
    DATA_NOUN = "measurement"           # "Save <noun> measurement", "There is no <noun> data"
    DEVICE_LABELS = (("dac", "DAC"), ("smu", "Keysight"))
    REQUIRED_DEVICES = ()               # (component, label) needed before Begin
    TASK_CLASS = GateMapTask
    LIVE_WORKER_CLASS = GateMapLiveWorker
    START_EXPECTED_END = "Expected End: (calculating)"
    FIXED_TOOLTIP = "Electrode held at V_fixed."
    POINT_SETTLE_LABEL = "Point settle"
    POINT_SETTLE_TOOLTIP = "Wait after each point's ramp, before reading."

    def __init__(self, station, embed=False, parent=None):
        super().__init__(parent)
        self.station = station
        self.embed = embed
        self.dashboard = None
        self.plotter_mode = False
        self.is_running = False
        self.is_paused = False
        self.sweep_worker = None
        self.ramp_worker = None
        self.control_worker = None
        self.value_storage = {}
        self.plan = None
        self.data = None
        self.measured_mask = None
        self.run_settings = None
        self.run_outcome = None
        self.run_message = ""
        self.measurement_started_at = None
        self.measurement_finished_at = None
        self.measurement_device_modes = {}
        self.electrode_modes = {}
        self.loaded_path = None
        self._plots_dirty = False
        self._overloads = 0
        self._unlocks = 0
        self._preview_ok = False
        self._last_point = None
        self._autosaved_path = None
        self.sweep_start_time = None
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self._idle_buttons = []         # enabled only when nothing runs
        self._display_controls = []     # stay usable while measuring
        self.init_detector_state()
        self.setWindowTitle(self.WINDOW_TITLE)
        self._init_ui()
        self.apply_theme()
        self.load_storage()
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self.refresh_preview)
        self._plot_timer = QtCore.QTimer(self)
        self._plot_timer.setInterval(100)
        self._plot_timer.timeout.connect(self._refresh_plots_if_dirty)
        self._plot_timer.start()
        self._connect_preview_signals()
        self.refresh_preview()
        self.refresh_device_label()
        self.live_status_worker = self.LIVE_WORKER_CLASS(station, self)
        self.live_status_worker.updated.connect(self.on_live_status)
        self._update_live_electrodes()
        self.live_status_worker.start()
        if not embed:
            from gui_safety import install_parameter_wheel_guard
            install_parameter_wheel_guard(self)

    def centralWidget(self):
        return self

    def data_root(self):
        """The checkout folder holding value_storage and data/.

        Read from the subclass's module, so tests that redirect that module's
        ``parent_dir`` also redirect every file this window writes.
        """
        return sys.modules[type(self).__module__].parent_dir

    # -- subclass hooks ----------------------------------------------------------
    def init_detector_state(self):
        """Subclass hook: per-instance state used by the hooks below."""

    def add_measurement_rows(self, form):
        """Subclass hook: rows above the coordinate settings of the Measurement group."""

    def build_detector_groups(self):
        """Subclass hook: instrument groups for the right-hand settings column."""
        return []

    def add_live_rows(self, form):
        """Subclass hook: instrument readouts at the top of the live group."""

    def build_result_pages(self):
        """Subclass hook: add result pages to ``right_tabs``.

        Must set ``map_items`` (key -> (plot, image, colour bar), with a
        crosshair each), ``cut_items`` ((key, curve, scale), ...) and
        ``cut_plots``; ``display_arrays()`` supplies every key used.
        """
        raise NotImplementedError

    def detector_preview_widgets(self):
        """Subclass hook: settings widgets whose changes refresh the preview."""
        return []

    def collect_detector_settings(self):
        return {}

    def apply_detector_values(self, s):
        """Subclass hook: show stored settings ``s`` (signals are blocked)."""

    def validate_targets(self, settings, plan, electrodes):
        """Subclass hook: check extra targets; the result is passed to the hooks below."""
        return None

    def validate_detector(self, settings):
        """Subclass hook: hardware-free checks of the instrument settings."""

    def preview_lines(self, settings, plan, electrodes, extra):
        return []

    def point_wait_estimate(self, settings):
        """Seconds waited at every point, for the duration estimate."""
        return float(settings["point_settle"])

    def update_detector_labels(self, x_name, y_name, cut_name):
        """Subclass hook: titles, cut and colour-bar labels of the result plots."""

    def prepare_run(self, settings, plan, electrodes, extra):
        """Subclass hook: finalize ``settings`` for the worker and bind simulations."""

    def reset_run_state(self):
        """Subclass hook: clear per-run records before a new sweep."""

    def sweep_task(self, settings, plan, electrodes):
        """Subclass hook: (worker function, args) of the sweep."""
        raise NotImplementedError

    def start_message(self, settings, plan):
        return (f"Starting {self.MEASUREMENT_NAME.lower()}: {len(plan.order)} points; fixed "
                f"{ELECTRODE_LABELS[settings['fixed']]} at {settings['v_fixed']:g} V.")

    def on_detector_event(self, event):
        """Subclass hook: task events other than log/status/electrodes/point/progress."""

    def point_status(self, values):
        """Subclass hook: SR830 LIAS? bits of a point, for the overload/unlock counts."""
        return 0

    def display_arrays(self):
        """Subclass hook: key -> [y, x] array shown in the maps."""
        raise NotImplementedError

    def cut_arrays(self):
        """Arrays for the cut curves; the maps' unless a subclass rescales its maps."""
        return self.display_arrays()

    def on_live_detector(self, report):
        """Subclass hook: show the instrument part of a live report."""

    def filename_tokens(self, settings):
        return [self.DATA_NOUN]

    def detector_file_attributes(self, settings):
        return {}

    def write_file(self, path):
        raise NotImplementedError

    def read_file(self, path):
        raise NotImplementedError

    def apply_loaded(self, loaded):
        """Subclass hook: restore per-run records from an opened file."""

    # -- widget helpers --------------------------------------------------------
    @staticmethod
    def _spin(value=0.0, minimum=-1e6, maximum=1e6, suffix="", tooltip=""):
        box = ScientificDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setKeyboardTracking(False)
        if suffix:
            box.setSuffix(" " + suffix)
        if tooltip:
            box.setToolTip(tooltip)
        return box

    @staticmethod
    def _points(value=11, maximum=100000):
        box = ScientificPointSpinBox()
        box.setRange(1, maximum)
        box.setValue(value)
        box.setKeyboardTracking(False)
        return box

    @staticmethod
    def _combo(items, tooltip=""):
        combo = QtWidgets.QComboBox()
        for label, data in items:
            combo.addItem(label, data)
        # Long entries must not widen the two-column panel; the popup shows them in full.
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(10)
        if tooltip:
            combo.setToolTip(tooltip)
        return combo

    @staticmethod
    def _set_combo(combo, value):
        for index in range(combo.count()):
            data = combo.itemData(index)
            if data == value or (isinstance(data, float) and isinstance(value, (int, float))
                                 and not isinstance(value, bool) and math.isclose(data, value, rel_tol=1e-9)):
                combo.setCurrentIndex(index)
                return True
        return False

    @staticmethod
    def _form(group):
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(5)
        layout.setContentsMargins(10, 16, 10, 10)
        return layout

    @staticmethod
    def _row(*widgets):
        """Container widget for compound form fields (keeps proper minimum heights)."""
        holder = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for widget in widgets:
            layout.addWidget(widget)
        return holder

    @staticmethod
    def _note(text=""):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #475569; font-size: 11px; font-weight: normal;")
        return label

    @staticmethod
    def _mono_label(text=""):
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        label.setWordWrap(True)
        return label

    # -- UI --------------------------------------------------------------------
    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(720)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(8)
        scroll.setWidget(left)
        main_layout.addWidget(scroll)

        self.device_label = QtWidgets.QLabel()
        self.device_label.setWordWrap(True)
        self.device_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        left_layout.addWidget(self.device_label)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(8)
        column_a = QtWidgets.QVBoxLayout()
        column_b = QtWidgets.QVBoxLayout()
        for column in (column_a, column_b):
            column.setSpacing(8)
            holder = QtWidgets.QWidget()
            holder.setLayout(column)
            # Equal columns that never force a horizontal scrollbar.
            holder.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
            columns.addWidget(holder, 1)
        column_a.setContentsMargins(0, 0, 0, 0)
        column_b.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(columns)
        left_layout.addStretch(1)

        column_a.addWidget(self._build_measurement_group())
        column_a.addWidget(self._build_sweep_group())
        column_a.addWidget(self._build_run_group())
        column_a.addStretch(1)
        column_b.addWidget(self._build_electrode_group())
        for group in self.build_detector_groups():
            column_b.addWidget(group)
        column_b.addWidget(self._build_live_group())
        column_b.addStretch(1)

        self.right_tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.right_tabs, 1)
        self._build_plot_tabs()

    def _build_measurement_group(self):
        group = QtWidgets.QGroupBox("Measurement")
        form = self._form(group)
        self.add_measurement_rows(form)
        self.fixed_combo = self._combo([(ELECTRODE_LABELS[name], name) for name in ELECTRODES], self.FIXED_TOOLTIP)
        form.addRow("DC-fixed electrode", self.fixed_combo)
        self.v_fixed_spin = self._spin(0.0, -210, 210, "V")
        form.addRow("V_fixed", self.v_fixed_spin)
        self.delta_spin = self._spin(0.0, -0.99, 0.99, tooltip="Gate asymmetry delta = (Ct - Cb) / (Ct + Cb)")
        form.addRow("Asymmetry delta", self.delta_spin)
        self.mode_combo = self._combo([("n0, p0 (density, field)", "density"),
                                       ("Electrode voltages (direct)", "direct")])
        form.addRow("Coordinates", self.mode_combo)
        self.transform_label = self._note()
        form.addRow(self.transform_label)
        self.wiring_note = self._note()
        form.addRow(self.wiring_note)
        self.sample_name_edit = QtWidgets.QLineEdit()
        self.sample_name_edit.setPlaceholderText("Sample / device name")
        form.addRow("Sample", self.sample_name_edit)
        self.comment_edit = QtWidgets.QLineEdit()
        self.comment_edit.setPlaceholderText("Saved with the data")
        form.addRow("Comment", self.comment_edit)
        return group

    def _build_sweep_group(self):
        group = QtWidgets.QGroupBox("Sweep")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(10, 16, 10, 10)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(5)
        self.x_header = QtWidgets.QLabel("n0")
        self.y_header = QtWidgets.QLabel("p0")
        for column, header in ((1, self.x_header), (2, self.y_header)):
            header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet("font-weight: bold;")
            grid.addWidget(header, 0, column)
        self.x_start_spin, self.x_end_spin = self._spin(-1.0, -210, 210, "V"), self._spin(1.0, -210, 210, "V")
        self.y_start_spin, self.y_end_spin = self._spin(-1.0, -210, 210, "V"), self._spin(1.0, -210, 210, "V")
        self.x_points_spin, self.y_points_spin = self._points(101), self._points(1)
        self.x_step_label, self.y_step_label = QtWidgets.QLabel("--"), QtWidgets.QLabel("--")
        for row, (label, x_widget, y_widget) in enumerate((
                ("Start", self.x_start_spin, self.y_start_spin),
                ("End", self.x_end_spin, self.y_end_spin),
                ("Points", self.x_points_spin, self.y_points_spin),
                ("Step", self.x_step_label, self.y_step_label)), start=1):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(x_widget, row, 1)
            grid.addWidget(y_widget, row, 2)
        layout.addLayout(grid)
        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(5)
        self.fast_combo = self._combo([("x (inner loop), y slow", "x"), ("y (inner loop), x slow", "y")])
        form.addRow("Fast axis", self.fast_combo)
        self.serpentine_check = QtWidgets.QCheckBox("Serpentine")
        self.serpentine_check.setToolTip("Alternate the fast-axis direction on every line.")
        form.addRow(self.serpentine_check)
        self.point_settle_spin = self._spin(0.5, 0, 3600, "s", self.POINT_SETTLE_TOOLTIP)
        form.addRow(self.POINT_SETTLE_LABEL, self.point_settle_spin)
        self.line_settle_spin = self._spin(3.0, 0, 3600, "s", "Extra wait at the start of every line.")
        form.addRow("Line settle", self.line_settle_spin)
        self.ramp_zero_check = QtWidgets.QCheckBox("Ramp to 0 V when complete")
        self.ramp_zero_check.setToolTip("Ramps every driven electrode to 0 V together, only after a complete "
                                        "run; Stop and errors hold the voltages.")
        form.addRow(self.ramp_zero_check)
        layout.addLayout(form)
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self.preview_label)
        return group

    def _build_run_group(self):
        group = QtWidgets.QGroupBox("Run")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(10, 16, 10, 10)
        row = QtWidgets.QHBoxLayout()
        self.btn_begin = QtWidgets.QPushButton("▶ Begin")
        self.btn_begin.setObjectName("btn_begin")
        self.btn_begin.clicked.connect(self.begin_sweep)
        self.btn_pause = QtWidgets.QPushButton("⏸ Interrupt")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop = QtWidgets.QPushButton("■ Stop")
        self.btn_stop.setObjectName("btn_reset")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop after the current step. Data is kept and autosaved; voltages hold.")
        self.btn_stop.clicked.connect(self.stop_sweep)
        for button in (self.btn_begin, self.btn_pause, self.btn_stop):
            row.addWidget(button)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("💾 Save Dataset")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(lambda: self.save_data())
        self.btn_open = QtWidgets.QPushButton("Open saved…")
        self.btn_open.setToolTip(f"View a saved {self.DATA_NOUN} file; nothing is sent to instruments.")
        self.btn_open.clicked.connect(lambda: self.open_saved())
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_open)
        layout.addLayout(row)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.lbl_expected_end = QtWidgets.QLabel("Expected End: --:--:--")
        self.lbl_expected_end.setStyleSheet("font-family: Consolas, monospace; font-weight: bold;")
        layout.addWidget(self.lbl_expected_end)
        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-style: italic; font-weight: bold;")
        layout.addWidget(self.lbl_status)
        return group

    def _build_electrode_group(self):
        group = QtWidgets.QGroupBox("Electrodes and limits")
        form = self._form(group)
        self.wiring_combos = {}
        for role in ELECTRODES:
            combo = self._combo([(label, key) for key, label in OUTPUT_CHOICES],
                                f"DC output wired to the {ELECTRODE_LABELS[role].lower()}.")
            self.wiring_combos[role] = combo
            form.addRow(ELECTRODE_LABELS[role], combo)
        self.dac_rate_spin = self._spin(1.0, 1e-4, 10.0, "V/s", "Ramp rate for uVoltDAQ outputs. Keysight "
                                        "outputs keep the ramp rate set in the Dual-Gate tab.")
        form.addRow("DAC ramp rate", self.dac_rate_spin)
        self.max_abs_spin = self._spin(10.0, 1e-3, 210, "V", "Maximum |V| on any electrode (after the transform).")
        form.addRow("Max |V| electrode", self.max_abs_spin)
        self.max_ts_spin = self._spin(10.0, 1e-3, 420, "V", "Maximum voltage across the top dielectric.")
        form.addRow("Max |Vt - Vs|", self.max_ts_spin)
        self.max_bs_spin = self._spin(10.0, 1e-3, 420, "V", "Maximum voltage across the bottom dielectric.")
        form.addRow("Max |Vb - Vs|", self.max_bs_spin)
        self.aux_combo = self._combo([("None", "none")] + [(f"uVoltDAQ ADC{n}", f"adc{n}") for n in range(4)],
                                     "Optional DC voltage recorded at every point (the scripts' ADC0).")
        form.addRow("Record ADC", self.aux_combo)
        return group

    def _build_live_group(self):
        group = QtWidgets.QGroupBox("Live readout and manual moves")
        form = self._form(group)
        self.add_live_rows(form)
        self.live_electrode_label = self._mono_label("Electrodes: --")
        form.addRow(self.live_electrode_label)
        self.goto_x_spin = self._spin(0.0, -210, 210, "V")
        self.goto_y_spin = self._spin(0.0, -210, 210, "V")
        form.addRow("Go to x, y", self._row(self.goto_x_spin, self.goto_y_spin))
        self.btn_goto = QtWidgets.QPushButton("Ramp to x, y")
        self.btn_goto.clicked.connect(self.goto_point)
        self.btn_zero = QtWidgets.QPushButton("Ramp to 0 V")
        self.btn_zero.setToolTip("Ramp every driven electrode to 0 V together.")
        self.btn_zero.clicked.connect(self.ramp_to_zero)
        self.btn_stop_ramp = QtWidgets.QPushButton("Stop ramp")
        self.btn_stop_ramp.setEnabled(False)
        self.btn_stop_ramp.clicked.connect(self.stop_manual)
        form.addRow(self._row(self.btn_goto, self.btn_zero, self.btn_stop_ramp))
        return group

    def _make_map(self, title):
        plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title=title)
        image = pg.ImageItem(axisOrder="row-major")
        plot.addItem(image)
        bar = pg.ColorBarItem(colorMap="viridis", interactive=False, width=12, pen="#94a3b8")
        bar.setImageItem(image, insert_in=plot.getPlotItem())
        plot.scene().sigMouseClicked.connect(lambda event, p=plot: self._on_map_clicked(event, p))
        return plot, image, bar

    def _make_cut(self, title, color):
        plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title=title)
        curve = plot.plot(pen=pg.mkPen(color, width=2), symbol="o", symbolSize=4,
                          symbolPen=pg.mkPen(color), symbolBrush=pg.mkBrush(color))
        return plot, curve

    def _plot_header(self):
        """Cut selector, follow and autoscale controls for the first result page."""
        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(4, 4, 4, 0)
        header_layout.addWidget(QtWidgets.QLabel("Cut along"))
        self.cut_combo = self._combo([("x", "x"), ("y", "y")],
                                     "Cut along x at the crosshair's y, or along y at its x.")
        self.cut_combo.setMinimumContentsLength(3)
        self.cut_combo.currentIndexChanged.connect(self.on_cut_axis_changed)
        header_layout.addWidget(self.cut_combo)
        self.follow_check = QtWidgets.QCheckBox("Follow acquisition")
        self.follow_check.setChecked(True)
        header_layout.addWidget(self.follow_check)
        header_layout.addStretch(1)
        self.btn_autoscale = QtWidgets.QPushButton("Autoscale")
        self.btn_autoscale.clicked.connect(self.autoscale)
        header_layout.addWidget(self.btn_autoscale)
        self.plot_header_layout = header_layout
        self._display_controls += [self.cut_combo, self.follow_check]
        return header

    def _build_plot_tabs(self):
        self.map_items = {}
        self.cut_items = []
        self.cut_plots = []
        self.build_result_pages()

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid, 1)
        self.vt_map_plot, self.vt_map_img, self.vt_map_bar = self._make_map("Target V_top")
        self.vb_map_plot, self.vb_map_img, self.vb_map_bar = self._make_map("Target V_bottom")
        self.vs_map_plot, self.vs_map_img, self.vs_map_bar = self._make_map("Target V_sample")
        grid.addWidget(self.vt_map_plot, 0, 0)
        grid.addWidget(self.vb_map_plot, 0, 1)
        grid.addWidget(self.vs_map_plot, 1, 0)
        self.targets_note = self._note()
        self.targets_note.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        grid.addWidget(self.targets_note, 1, 1)
        self.right_tabs.addTab(page, "Electrode targets")

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(5000)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.right_tabs.addTab(self.log_view, "Log")

        self.crosshairs = []
        for key, (plot, _image, _bar) in self.map_items.items():
            vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#f97316", width=1.5))
            hline = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen("#f97316", width=1.5))
            plot.addItem(vline)
            plot.addItem(hline)
            vline.sigPositionChanged.connect(lambda line, v=vline, h=hline: self._on_crosshair_dragged(v, h))
            hline.sigPositionChanged.connect(lambda line, v=vline, h=hline: self._on_crosshair_dragged(v, h))
            self.crosshairs.append((vline, hline))
        self.crosshair = (0.0, 0.0)

    def _connect_preview(self, widget):
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            widget.valueChanged.connect(self.schedule_preview)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.currentIndexChanged.connect(self.schedule_preview)
        elif isinstance(widget, QtWidgets.QAbstractButton):
            widget.toggled.connect(self.schedule_preview)

    def _connect_preview_signals(self):
        widgets = (self.v_fixed_spin, self.delta_spin, self.x_start_spin, self.x_end_spin, self.y_start_spin,
                   self.y_end_spin, self.point_settle_spin, self.line_settle_spin, self.dac_rate_spin,
                   self.max_abs_spin, self.max_ts_spin, self.max_bs_spin, self.x_points_spin, self.y_points_spin,
                   self.fixed_combo, self.mode_combo, self.fast_combo, *self.wiring_combos.values(),
                   self.serpentine_check, self.ramp_zero_check, *self.detector_preview_widgets())
        for widget in widgets:
            self._connect_preview(widget)
        for combo in self.wiring_combos.values():
            combo.currentIndexChanged.connect(lambda: self._update_live_electrodes())
        self.fixed_combo.currentIndexChanged.connect(lambda: self._update_live_electrodes())
        self.v_fixed_spin.valueChanged.connect(lambda: self._update_live_electrodes())

    # -- settings --------------------------------------------------------------
    def collect_settings(self):
        settings = {
            "fixed": self.fixed_combo.currentData(),
            "v_fixed": self.v_fixed_spin.value(),
            "delta": self.delta_spin.value(),
            "coordinate_mode": self.mode_combo.currentData(),
            "x_start": self.x_start_spin.value(), "x_end": self.x_end_spin.value(),
            "x_points": self.x_points_spin.value(),
            "y_start": self.y_start_spin.value(), "y_end": self.y_end_spin.value(),
            "y_points": self.y_points_spin.value(),
            "fast_axis": self.fast_combo.currentData(),
            "serpentine": self.serpentine_check.isChecked(),
            "point_settle": self.point_settle_spin.value(),
            "line_settle": self.line_settle_spin.value(),
            "ramp_to_zero": self.ramp_zero_check.isChecked(),
            "wiring": {role: combo.currentData() for role, combo in self.wiring_combos.items()},
            "dac_ramp_rate": self.dac_rate_spin.value(),
            "max_abs": self.max_abs_spin.value(),
            "max_top_sample": self.max_ts_spin.value(),
            "max_bottom_sample": self.max_bs_spin.value(),
            "aux_adc": self.aux_combo.currentData(),
            "sample_name": self.sample_name_edit.text().strip(),
            "comment": self.comment_edit.text().strip(),
        }
        settings.update(self.collect_detector_settings())
        return settings

    def apply_settings(self, settings):
        s = dict(self.DEFAULTS)
        s.update({key: value for key, value in settings.items() if key in self.DEFAULTS})
        wiring = dict(self.DEFAULT_WIRING)
        if isinstance(settings.get("wiring"), dict):
            wiring.update({k: v for k, v in settings["wiring"].items() if k in ELECTRODES})
        widgets = [w for w in self.findChildren(QtWidgets.QWidget)
                   if isinstance(w, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QCheckBox,
                                     QtWidgets.QLineEdit))]
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        try:
            self._apply_values(s, wiring)
        finally:
            for blocker in blockers:
                blocker.unblock()

    def _apply_values(self, s, wiring):
        self.apply_detector_values(s)
        self._set_combo(self.fixed_combo, s["fixed"])
        self.v_fixed_spin.setValue(float(s["v_fixed"]))
        self.delta_spin.setValue(float(s["delta"]))
        self._set_combo(self.mode_combo, s["coordinate_mode"])
        self.x_start_spin.setValue(float(s["x_start"]))
        self.x_end_spin.setValue(float(s["x_end"]))
        self.x_points_spin.setValue(int(s["x_points"]))
        self.y_start_spin.setValue(float(s["y_start"]))
        self.y_end_spin.setValue(float(s["y_end"]))
        self.y_points_spin.setValue(int(s["y_points"]))
        self._set_combo(self.fast_combo, s["fast_axis"])
        self.serpentine_check.setChecked(bool(s["serpentine"]))
        self.point_settle_spin.setValue(float(s["point_settle"]))
        self.line_settle_spin.setValue(float(s["line_settle"]))
        self.ramp_zero_check.setChecked(bool(s["ramp_to_zero"]))
        for role, combo in self.wiring_combos.items():
            self._set_combo(combo, wiring[role])
        self.dac_rate_spin.setValue(float(s["dac_ramp_rate"]))
        self.max_abs_spin.setValue(float(s["max_abs"]))
        self.max_ts_spin.setValue(float(s["max_top_sample"]))
        self.max_bs_spin.setValue(float(s["max_bottom_sample"]))
        self._set_combo(self.aux_combo, s["aux_adc"])
        self.sample_name_edit.setText(str(s["sample_name"]))
        self.comment_edit.setText(str(s["comment"]))

    def load_storage(self):
        storage_path = os.path.join(self.data_root(), "value_storage")
        stored = {}
        try:
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    self.value_storage = json.load(f)
                stored = self.value_storage.get(self.STORAGE_KEY, {})
        except Exception as exc:
            print(f"Error loading value_storage in {self.TAB_NAME} tab: {exc}")
        try:
            self.apply_settings(stored if isinstance(stored, dict) else {})
        except Exception as exc:
            print(f"Saved {self.DATA_NOUN} settings were invalid ({exc}); using defaults.")
            self.apply_settings({})
        self._update_labels()

    def save_storage(self):
        storage_path = os.path.join(self.data_root(), "value_storage")
        try:
            current = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    current = json.load(f)
            current[self.STORAGE_KEY] = self.collect_settings()
            self.value_storage = current
            with open(storage_path, "w") as f:
                json.dump(current, f, indent=4)
        except Exception as exc:
            print(f"Error saving {self.DATA_NOUN} settings to value_storage: {exc}")

    def reload_gui_from_storage(self):
        if not self.busy():
            self.load_storage()
            self.refresh_preview()

    # -- derived settings ------------------------------------------------------
    @staticmethod
    def build_plan(settings):
        return build_sweep_plan(
            mode=settings["coordinate_mode"], fixed=settings["fixed"], delta=settings["delta"],
            v_fixed=settings["v_fixed"], x_start=settings["x_start"], x_end=settings["x_end"],
            x_points=settings["x_points"], y_start=settings["y_start"], y_end=settings["y_end"],
            y_points=settings["y_points"], fast_axis=settings["fast_axis"], serpentine=settings["serpentine"])

    def make_electrodes(self, settings):
        return ElectrodeSet(self.station, settings["wiring"], fixed=settings["fixed"],
                            fixed_voltage=settings["v_fixed"],
                            limits=ElectrodeLimits(settings["max_abs"], settings["max_top_sample"],
                                                   settings["max_bottom_sample"]),
                            dac_ramp_rate=settings["dac_ramp_rate"])

    @staticmethod
    def coordinate_targets(settings, x, y):
        if settings["coordinate_mode"] == "density":
            vt, vb, vs = electrode_voltages(x, y, delta=settings["delta"], v_fixed=settings["v_fixed"],
                                            fixed=settings["fixed"])
        else:
            vt, vb, vs = direct_voltages(x, y, v_fixed=settings["v_fixed"], fixed=settings["fixed"])
        return {"top": float(vt), "bottom": float(vb), "sample": float(vs)}

    def zero_targets(self, electrodes):
        return {role: (0.0 if output.driven else output.setpoint()) for role, output in electrodes.outputs.items()}

    def validate_run(self, settings):
        """Everything checkable without hardware; raises ValueError/RuntimeError."""
        plan = self.build_plan(settings)
        electrodes = self.make_electrodes(settings)
        electrodes.check(plan.vt, plan.vb, plan.vs, context="Sweep")
        extra = self.validate_targets(settings, plan, electrodes)
        if settings["ramp_to_zero"]:
            zero = self.zero_targets(electrodes)
            electrodes.check(zero["top"], zero["bottom"], zero["sample"], context="Ramp to 0 V")
        self.validate_detector(settings)
        return plan, electrodes, extra

    # -- labels and preview ------------------------------------------------------
    def _update_labels(self):
        x_name, y_name = coordinate_names(self.mode_combo.currentData(), self.fixed_combo.currentData())
        self.x_header.setText(x_name)
        self.y_header.setText(y_name)
        cut_name = x_name if self.cut_combo.currentData() == "x" else y_name
        self.update_detector_labels(x_name, y_name, cut_name)
        for plot in [item[0] for item in self.map_items.values()] + [self.vt_map_plot, self.vb_map_plot,
                                                                      self.vs_map_plot]:
            plot.setLabel("bottom", x_name, units="V")
            plot.setLabel("left", y_name, units="V")
        for bar in (self.vt_map_bar, self.vb_map_bar, self.vs_map_bar):
            axis = getattr(bar, "axis", None)
            if axis is not None:
                axis.setLabel(text="", units="V")
        for plot in self._plot_widgets():
            configure_plot_axes(plot)

    def schedule_preview(self, *args):
        self._preview_timer.start()

    def refresh_preview(self):
        self._update_labels()
        try:
            settings = self.collect_settings()
            self.transform_label.setText(transform_description(settings["coordinate_mode"], settings["fixed"],
                                                               settings["delta"], settings["v_fixed"]))
        except Exception as exc:
            self.transform_label.setText(str(exc))
        try:
            plan, electrodes, extra = self.validate_run(settings)
        except Exception as exc:
            self._preview_ok = False
            self.preview_label.setText(f"<span style='color:#b91c1c'><b>Blocked:</b> {exc}</span>")
            self.targets_note.setText(f"Blocked: {exc}")
            self._update_buttons()
            return
        self._preview_ok = True
        for label, coords in ((self.x_step_label, plan.x), (self.y_step_label, plan.y)):
            label.setText(format_si(abs(coords[1] - coords[0]), "V", 4) if coords.size > 1 else "single point")
        summary = plan_summary(plan)
        seconds = estimate_duration(plan, point_settle=self.point_wait_estimate(settings),
                                    line_settle=settings["line_settle"],
                                    ramp_rate=self.ramp_rates(settings, electrodes))
        lines = [f"{summary['points']} points ({len(plan.x)} x {len(plan.y)}), ~{self._duration_text(seconds)}"]
        for role in ELECTRODES:
            low, high = summary[role]
            lines.append(f"{ELECTRODE_SYMBOLS[role]:>9}: {low:+.4f} .. {high:+.4f} V  "
                         f"[{electrodes.outputs[role].label}]")
        lines.append(f"max |Vt-Vs| {summary['top_sample']:.4f} V, max |Vb-Vs| {summary['bottom_sample']:.4f} V")
        lines += self.preview_lines(settings, plan, electrodes, extra)
        text = "\n".join(lines)
        self.preview_label.setText(text)
        self.targets_note.setText(text)
        if not self.is_running:
            for image, bar, grid in ((self.vt_map_img, self.vt_map_bar, plan.vt),
                                     (self.vb_map_img, self.vb_map_bar, plan.vb),
                                     (self.vs_map_img, self.vs_map_bar, plan.vs)):
                self._set_map(image, bar, grid, plan.x, plan.y)
        self._update_buttons()

    @staticmethod
    def ramp_rates(settings, electrodes):
        """Per-electrode ramp rate: DAC from this tab, Keysight from the Dual-Gate setting."""
        rates = {}
        for role, output in electrodes.outputs.items():
            if output.kind == "dac":
                rates[role] = float(settings["dac_ramp_rate"])
            elif output.kind == "smu":
                try:
                    rates[role] = float(output.channel.ramp_rate())
                except Exception:
                    rates[role] = 0.1
            else:
                rates[role] = math.inf
        return rates

    @staticmethod
    def _duration_text(seconds):
        seconds = int(round(seconds))
        hours, rest = divmod(seconds, 3600)
        return f"{hours} h {rest // 60:02d} min" if hours else f"{rest // 60} min {rest % 60:02d} s"

    def refresh_device_label(self):
        parts = []
        for name, label in self.DEVICE_LABELS:
            component = self.station.components.get(name)
            state = "missing" if component is None else ("SIM" if getattr(component, "simulated", False) else "physical")
            color = "#b91c1c" if component is None else ("#7c3aed" if state == "SIM" else "#047857")
            parts.append(f"<span style='color:{color}'><b>{label}</b> {state}</span>")
        self.device_label.setText(" &nbsp;|&nbsp; ".join(parts))

    def on_devices_changed(self):
        self.refresh_device_label()
        self._update_live_electrodes()

    def _update_live_electrodes(self):
        try:
            electrodes = self.make_electrodes(self.collect_settings())
        except Exception:
            electrodes = None
        worker = getattr(self, "live_status_worker", None)
        if worker is not None:
            worker.set_electrodes(electrodes)
        self._live_electrodes = electrodes

    @staticmethod
    def electrode_voltage_reader(electrodes):
        """``() -> (Vt, Vb, Vs)`` for simulated sample models, without instrument queries.

        Simulated or undriven outputs report their setpoints; a physical output
        reports its cached value, so a simulation never adds hardware traffic.
        """
        def voltages():
            values = []
            for role in ELECTRODES:
                output = electrodes.outputs[role]
                try:
                    if not output.driven or output.simulated:
                        values.append(output.setpoint())
                    else:
                        cache = output.channel.volt.cache if output.kind == "smu" else output.channel.voltage.cache
                        values.append(float(cache.get(get_if_invalid=False) or 0.0))
                except Exception:
                    values.append(0.0)
            return tuple(values)
        return voltages

    # -- buttons and state -----------------------------------------------------
    def busy(self):
        return self.is_running or any(worker is not None and worker.isRunning()
                                      for worker in (self.sweep_worker, self.ramp_worker, self.control_worker))

    def _update_buttons(self):
        manual = self.ramp_worker is not None and self.ramp_worker.isRunning()
        control = self.control_worker is not None and self.control_worker.isRunning()
        idle = not self.is_running and not manual and not control
        self.btn_begin.setEnabled((idle and self._preview_ok) or (self.is_running and self.is_paused))
        self.btn_pause.setEnabled(self.is_running)
        self.btn_stop.setEnabled(self.is_running)
        self.btn_save.setEnabled(idle and self.measured_mask is not None and bool(np.any(self.measured_mask)))
        self.btn_open.setEnabled(idle)
        for button in (*self._idle_buttons, self.btn_goto, self.btn_zero):
            button.setEnabled(idle)
        self.btn_stop_ramp.setEnabled(manual or control)

    def set_inputs_enabled(self, enabled):
        for widget in self.findChildren(QtWidgets.QWidget):
            if isinstance(widget, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QCheckBox,
                                   QtWidgets.QLineEdit)) and widget not in self._display_controls:
                widget.setEnabled(enabled)

    def _show_blocked(self, title, exc):
        message = str(exc)
        self.lbl_status.setText(f"Status: Blocked ({message})")
        self.append_log(f"{title}: {message}")
        QtWidgets.QMessageBox.critical(self, title, message)

    def append_log(self, text):
        stamp = time.strftime("%H:%M:%S")
        for line in str(text).rstrip().splitlines() or [""]:
            self.log_view.appendPlainText(f"[{stamp}] {line}")

    def _pause_polling(self):
        self.live_status_worker.pause()
        self.station.live_status_paused = True

    def resume_status_worker(self):
        if self.busy():
            return
        self.station.live_status_paused = False
        if self.dashboard is None:
            self.live_status_worker.resume()

    # -- sweep -----------------------------------------------------------------
    def begin_sweep(self):
        if self.plotter_mode:
            return
        if self.is_running:
            if self.is_paused:
                self.toggle_pause()
            return
        if self.busy():
            return
        settings = self.collect_settings()
        try:
            plan, electrodes, extra = self.validate_run(settings)
            for name, label in self.REQUIRED_DEVICES:
                if self.station.components.get(name) is None:
                    raise RuntimeError(f"The {label} is not initialized.")
        except Exception as exc:
            self._show_blocked(f"{self.MEASUREMENT_NAME} blocked", exc)
            return
        self.save_storage()
        self._pause_polling()
        self.prepare_run(settings, plan, electrodes, extra)
        self.run_settings = settings
        self.plan = plan
        self.loaded_path = None
        self._allocate(plan)
        self.reset_run_state()
        self.run_outcome, self.run_message = None, ""
        self._overloads = self._unlocks = 0
        self._autosaved_path = None
        self.sweep_start_time = time.time()
        self.total_paused_time, self.pause_start_time = 0.0, None
        self.measurement_started_at = new_measurement_start(self.sweep_start_time)
        self.measurement_finished_at = None
        self.measurement_device_modes = measurement_device_modes(self.station)
        self.electrode_modes = electrodes.simulated_modes()
        self.is_running, self.is_paused = True, False
        self.set_inputs_enabled(False)
        self.progress_bar.setValue(0)
        self.lbl_expected_end.setText(self.START_EXPECTED_END)
        self.append_log(self.start_message(settings, plan))
        self.reset_view()
        function, args = self.sweep_task(settings, plan, electrodes)
        worker = self.TASK_CLASS(function, *args, stop_callbacks=[electrodes.request_stop], parent=self)
        worker.task_event.connect(self.on_task_event)
        worker.completed.connect(self.on_sweep_completed)
        self._replace_worker("sweep_worker", worker)
        worker.start()
        self._update_buttons()

    def _replace_worker(self, attribute, worker):
        """Swap in a new worker; free the finished one only after it is unreferenced."""
        old = getattr(self, attribute)
        setattr(self, attribute, worker)
        if old is not None and not old.isRunning():
            old.deleteLater()

    def toggle_pause(self):
        worker = self.sweep_worker
        if not self.is_running or worker is None:
            return
        if self.is_paused:
            self.is_paused = False
            if self.pause_start_time is not None:
                self.total_paused_time += time.time() - self.pause_start_time
                self.pause_start_time = None
            worker.resume()
            self.btn_pause.setText("⏸ Interrupt")
            self.lbl_status.setText("Status: Resuming...")
        else:
            self.is_paused = True
            self.pause_start_time = time.time()
            worker.pause()
            self.btn_pause.setText("▶ Resume")
            self.lbl_status.setText("Status: Pausing after the current step...")
        self._update_buttons()

    def stop_sweep(self):
        if self.sweep_worker is not None and self.sweep_worker.isRunning():
            self.sweep_worker.stop()
            self.lbl_status.setText("Status: Stopping after the current step...")
            self.btn_stop.setEnabled(False)
            self.btn_pause.setEnabled(False)

    def _allocate(self, plan):
        shape = plan.shape
        self.data = {key: np.full(shape, np.nan) for key in self.POINT_FIELDS}
        for key, fill in self.INT_FIELDS.items():
            self.data[key] = np.full(shape, fill, dtype=int)
        self.measured_mask = np.zeros(shape, dtype=bool)
        self.x_coords, self.y_coords = plan.x.copy(), plan.y.copy()
        self._last_point = None
        self._plots_dirty = True

    def on_task_event(self, event):
        kind = event.get("kind")
        if kind == "log":
            self.append_log(event["text"])
        elif kind == "status":
            self.lbl_status.setText(event["text"])
        elif kind == "electrodes":
            self._show_electrodes({role: (value, None) for role, value in event["values"].items()})
        elif kind == "point":
            self.on_point(event["iy"], event["ix"], event["values"])
        elif kind == "progress":
            self.on_progress(event["done"], event["total"])
        else:
            self.on_detector_event(event)

    def on_point(self, iy, ix, values):
        if self.data is None:
            return
        for key, value in values.items():
            self.data[key][iy, ix] = value
        self.measured_mask[iy, ix] = True
        status = int(self.point_status(values))
        if status & OVERLOAD_MASK:
            self._overloads += 1
        if status & UNLOCK_MASK:
            self._unlocks += 1
        self._last_point = (iy, ix)
        self._plots_dirty = True

    def on_progress(self, done, total):
        self.progress_bar.setValue(int(round(100 * done / max(total, 1))))
        if self.sweep_start_time and done:
            elapsed = time.time() - self.sweep_start_time - self.total_paused_time
            remaining = elapsed / done * (total - done)
            self.lbl_expected_end.setText("Expected End: " + time.strftime("%H:%M:%S",
                                                                           time.localtime(time.time() + remaining)))
        if self._overloads or self._unlocks:
            self.lbl_status.setText(f"Status: Sweeping... ({done}/{total}); lock-in overload at "
                                    f"{self._overloads} point(s), reference unlock at {self._unlocks}")

    def on_sweep_completed(self, outcome, message):
        self.run_outcome, self.run_message = outcome, message
        self.measurement_finished_at = new_measurement_start()
        self.is_running = self.is_paused = False
        self.btn_pause.setText("⏸ Interrupt")
        self._plots_dirty = True
        self._refresh_plots_if_dirty()
        measured = self.measured_mask is not None and bool(np.any(self.measured_mask))
        if measured:
            self.trigger_autosave(was_interrupted=outcome != "complete")
        self.set_inputs_enabled(True)
        self.append_log(f"Run {outcome}: {message}")
        if outcome == "complete":
            self.progress_bar.setValue(100)
            self.lbl_expected_end.setText("Sweep finished")
            self.lbl_status.setText("Status: Finished" + (f" ({self._overloads} overloaded point(s))"
                                                          if self._overloads else ""))
        elif outcome == "stopped":
            self.lbl_expected_end.setText("Expected End: --:--:--")
            self.lbl_status.setText("Status: Stopped; voltages hold. " + ("Partial data autosaved." if measured else ""))
        else:
            self.lbl_expected_end.setText("Expected End: --:--:--")
            self.lbl_status.setText(f"Status: Error ({message})")
        self._update_buttons()
        QtCore.QTimer.singleShot(500, self.resume_status_worker)
        if outcome == "complete":
            QtWidgets.QMessageBox.information(self, self.MEASUREMENT_NAME, "Measurement finished.")
        elif outcome == "error":
            QtWidgets.QMessageBox.critical(self, f"{self.MEASUREMENT_NAME} failed",
                                           f"{message}\n\nAcquired points were kept"
                                           + (" and autosaved." if measured else "."))

    # -- manual actions ----------------------------------------------------------
    def _start_manual(self, attribute, function, *args, stop_callbacks=()):
        if self.busy() or self.plotter_mode:
            return None
        self._pause_polling()
        worker = self.TASK_CLASS(function, *args, stop_callbacks=stop_callbacks, parent=self)
        worker.task_event.connect(self.on_task_event)
        worker.completed.connect(lambda outcome, message, a=attribute: self.on_manual_completed(a, outcome, message))
        self._replace_worker(attribute, worker)
        worker.start()
        self._update_buttons()
        return worker

    def on_manual_completed(self, attribute, outcome, message):
        text = {"complete": "Done", "stopped": "Stopped", "error": "Error"}[outcome]
        self.lbl_status.setText(f"Status: {text}{': ' + message if message else ''}")
        self.append_log(f"{text}: {message}")
        if outcome == "error":
            QtWidgets.QMessageBox.critical(self, f"{self.TAB_NAME} tab", message)
        worker = getattr(self, attribute)
        if worker is not None:
            worker.wait(2000)
        self._update_buttons()
        QtCore.QTimer.singleShot(300, self.resume_status_worker)

    def goto_point(self):
        settings = self.collect_settings()
        try:
            electrodes = self.make_electrodes(settings)
            targets = self.coordinate_targets(settings, self.goto_x_spin.value(), self.goto_y_spin.value())
            electrodes.check(targets["top"], targets["bottom"], targets["sample"], context="Go to")
        except Exception as exc:
            self._show_blocked("Move blocked", exc)
            return
        self.append_log(f"Ramping to Vt {targets['top']:+.4f}, Vb {targets['bottom']:+.4f}, "
                        f"Vs {targets['sample']:+.4f} V.")
        self._start_manual("ramp_worker", run_move, electrodes, targets, stop_callbacks=[electrodes.request_stop])

    def ramp_to_zero(self):
        settings = self.collect_settings()
        try:
            electrodes = self.make_electrodes(settings)
            targets = self.zero_targets(electrodes)
            electrodes.check(targets["top"], targets["bottom"], targets["sample"], context="Ramp to 0 V")
        except Exception as exc:
            self._show_blocked("Move blocked", exc)
            return
        self._start_manual("ramp_worker", run_move, electrodes, targets, stop_callbacks=[electrodes.request_stop])

    def stop_manual(self):
        for worker in (self.ramp_worker, self.control_worker):
            if worker is not None and worker.isRunning():
                worker.stop()
        self.lbl_status.setText("Status: Stopping...")

    # -- live readout ------------------------------------------------------------
    def on_live_status(self, report):
        if self.is_running:
            return
        self.on_live_detector(report)
        if "electrodes" in report:
            self._show_electrodes(report["electrodes"])

    def _show_electrodes(self, values):
        lines = []
        currents = getattr(self, "_last_currents", {})
        for role in ELECTRODES:
            if role not in values:
                continue
            setpoint, current = values[role]
            if current is not None:
                currents[role] = current
            leak = currents.get(role)
            leak_text = f"  I {format_si(leak, 'A', 3)}" if leak is not None and np.isfinite(leak) else ""
            lines.append(f"{ELECTRODE_SYMBOLS[role]:>9} {setpoint:+.5f} V{leak_text}")
        self._last_currents = currents
        try:
            vt, vb, vs = (values[r][0] for r in ELECTRODES)
            n0, p0 = density_field(vt, vb, vs, delta=self.delta_spin.value())
            lines.append(f"       n0 {n0:+.5f} V   p0 {p0:+.5f} V")
        except Exception:
            pass
        self.live_electrode_label.setText("\n".join(lines))

    # -- plots -----------------------------------------------------------------
    def _plot_widgets(self):
        return [value for value in vars(self).values() if isinstance(value, pg.PlotWidget)]

    def _set_map(self, image, bar, values, xs, ys):
        values = np.asarray(values, dtype=float)
        if xs[0] > xs[-1]:
            values = values[:, ::-1]
        if ys[0] > ys[-1]:
            values = values[::-1, :]
        low, high = finite_levels(values)
        image.setImage(values, autoLevels=False, levels=(low, high))
        x0, width = image_extent(xs)
        y0, height = image_extent(ys)
        image.setRect(QtCore.QRectF(x0, y0, width, height))
        bar.setLevels((low, high))

    def _refresh_plots_if_dirty(self):
        if not self._plots_dirty or self.data is None:
            return
        self._plots_dirty = False
        arrays = self.display_arrays()
        for key, (plot, image, bar) in self.map_items.items():
            self._set_map(image, bar, arrays[key], self.x_coords, self.y_coords)
        if self.follow_check.isChecked() and self._last_point is not None:
            iy, ix = self._last_point
            self.set_crosshair(self.x_coords[ix], self.y_coords[iy])
        else:
            self.refresh_cuts()

    def set_crosshair(self, x, y):
        self.crosshair = (float(x), float(y))
        for vline, hline in self.crosshairs:
            with QtCore.QSignalBlocker(vline), QtCore.QSignalBlocker(hline):
                vline.setValue(x)
                hline.setValue(y)
        self.refresh_cuts()

    def _on_crosshair_dragged(self, vline, hline):
        self.follow_check.setChecked(False)
        self.set_crosshair(vline.value(), hline.value())

    def _on_map_clicked(self, event, plot):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        view = plot.getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        point = view.mapSceneToView(event.scenePos())
        self.follow_check.setChecked(False)
        self.set_crosshair(point.x(), point.y())

    def refresh_cuts(self):
        if self.data is None:
            return
        arrays = self.cut_arrays()
        x, y = self.crosshair
        along_x = self.cut_combo.currentData() == "x"
        if along_x:
            index = int(np.argmin(np.abs(self.y_coords - y)))
            coords = self.x_coords
            pick = lambda array: array[index, :]
            mask = self.measured_mask[index, :]
        else:
            index = int(np.argmin(np.abs(self.x_coords - x)))
            coords = self.y_coords
            pick = lambda array: array[:, index]
            mask = self.measured_mask[:, index]
        for key, curve, scale in self.cut_items:
            values = np.where(mask, pick(arrays[key]) * scale, np.nan)
            if np.isfinite(values).any():
                curve.setData(coords, values, connect="finite")
            else:
                # Nothing measured on this cut yet; all-NaN data makes pyqtgraph warn on every refresh.
                curve.setData([], [])

    def on_cut_axis_changed(self):
        self._update_labels()
        self.refresh_cuts()

    def reset_view(self):
        if self.data is None:
            return
        x0, width = image_extent(self.x_coords)
        y0, height = image_extent(self.y_coords)
        for plot, _image, _bar in self.map_items.values():
            plot.setRange(xRange=(x0, x0 + width), yRange=(y0, y0 + height), padding=0)
        for plot in self.cut_plots:
            plot.enableAutoRange()
        self.set_crosshair(self.x_coords[0], self.y_coords[0])

    def autoscale(self):
        for plot in self._plot_widgets():
            plot.enableAutoRange()

    def apply_theme(self):
        from theme import SCIENTIFIC_THEME_QSS
        # The theme draws spin buttons with zero width. Unlike other tabs, do not also
        # select NoButtons: with Qt >= 6.12 that combination leaves a 1 px edit field.
        self.setStyleSheet(SCIENTIFIC_THEME_QSS)
        for plot in self._plot_widgets():
            configure_plot_axes(plot)
            plot.setBackground("#ffffff")
            for side in ("left", "bottom"):
                plot.getAxis(side).setPen(pg.mkPen("#334155"))
                plot.getAxis(side).setTextPen(pg.mkPen("#334155"))
            plot.showGrid(x=False, y=False)

    # -- save / load -------------------------------------------------------------
    def default_filename(self, prefix="", suffix=""):
        """``yyyy_mm_dd_[prefix]<tokens>_<ranges>_<shape>_HHMMSS[suffix].h5``; the date is the run start."""
        settings = self.run_settings or self.collect_settings()
        plan = self.plan
        parts = self.filename_tokens(settings)
        if plan is not None:
            for name, coords in zip(plan.names, (plan.x, plan.y)):
                if len(coords) > 1:
                    parts.append(f"{axis_token(name)}_{number_token(coords[0])}-{number_token(coords[-1])}V")
            parts.append(f"{len(plan.x)}x{len(plan.y)}")
        started = self.measurement_started_at or new_measurement_start()
        parts.append(started.strftime("%H%M%S"))
        return dated_measurement_filename(prefix + "_".join(parts) + suffix + ".h5", started)

    def file_attributes(self):
        settings = self.run_settings or {}
        x_name, y_name = self.plan.names
        attrs = {
            "fixed_electrode": self.plan.fixed, "v_fixed_V": self.plan.v_fixed, "delta": self.plan.delta,
            "coordinate_mode": self.plan.mode, "x_name": x_name, "y_name": y_name,
            "fast_axis": self.plan.fast_axis, "serpentine": self.plan.serpentine,
            "wiring": settings.get("wiring", {}), "electrode_modes": self.electrode_modes,
            "device_modes": self.measurement_device_modes,
            "measurement_started_at": measurement_start_text(self.measurement_started_at),
            "measurement_finished_at": (self.measurement_finished_at.isoformat(timespec="seconds")
                                        if self.measurement_finished_at else ""),
            "status": self.run_outcome or "in progress", "status_message": self.run_message,
            "sample_name": settings.get("sample_name", ""), "comment": settings.get("comment", ""),
            "lockin_overload_points": int(self._overloads), "reference_unlock_points": int(self._unlocks),
        }
        attrs.update(self.detector_file_attributes(settings))
        return attrs

    def save_data(self, file_path=None):
        if self.plan is None or self.measured_mask is None or not np.any(self.measured_mask):
            if file_path is None:
                QtWidgets.QMessageBox.warning(self, "No data", f"There is no {self.DATA_NOUN} data to save.")
            return None
        try:
            if file_path is None:
                directory = measurement_data_dir(self.data_root(), self.measurement_started_at)
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, f"Save {self.DATA_NOUN} measurement", os.path.join(directory, self.default_filename()),
                    "HDF5 files (*.h5);;All files (*.*)")
                if not filename:
                    return None
                filename = dated_measurement_path(filename, self.measurement_started_at)
            else:
                filename = file_path
            self.write_file(filename)
            if file_path is None:
                QtWidgets.QMessageBox.information(self, "Saved", f"Data saved to {filename}")
            else:
                print(f"Auto-saved {self.DATA_NOUN} data to {filename}")
            self.append_log(f"Saved {filename}")
            return filename
        except Exception as exc:
            self.append_log(f"Save failed: {exc}")
            if file_path is None:
                QtWidgets.QMessageBox.critical(self, "Save error", str(exc))
            else:
                print(f"{self.DATA_NOUN.capitalize()} autosave failed: {exc}")
            return None

    def trigger_autosave(self, was_interrupted=False):
        try:
            directory = measurement_data_dir(self.data_root(), self.measurement_started_at, autosave=True)
            name = self.default_filename("autosave_", "_interrupted" if was_interrupted else "")
            self._autosaved_path = self.save_data(file_path=os.path.join(directory, name))
        except Exception as exc:
            print(f"{self.DATA_NOUN.capitalize()} autosave failed: {exc}")

    def open_saved(self, path=None):
        if self.busy():
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, f"Open {self.DATA_NOUN} measurement", measurement_data_dir(self.data_root()),
                "HDF5 files (*.h5);;All files (*.*)")
            if not path:
                return
        try:
            loaded = self.read_file(path)
            settings = loaded["settings"]
            plan = self.build_plan(settings)
            if plan.shape != loaded["mask"].shape:
                raise ValueError("The saved grid does not match its recorded sweep settings.")
            plan.vt, plan.vb, plan.vs = loaded["vt"], loaded["vb"], loaded["vs"]
            plan.n0, plan.p0, plan.x, plan.y = loaded["n0"], loaded["p0"], loaded["x"], loaded["y"]
            plan.order = loaded["order"]
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", f"{path}\n\n{exc}")
            return
        self.plan, self.run_settings = plan, settings
        self.data, self.measured_mask = loaded["data"], loaded["mask"]
        self.x_coords, self.y_coords = plan.x, plan.y
        attrs = loaded["attrs"]
        self.run_outcome = attrs.get("status", "")
        self.run_message = attrs.get("status_message", "")
        self.loaded_path = path
        self._last_point = None
        self.apply_loaded(loaded)
        self.follow_check.setChecked(False)
        self.reset_view()
        self._plots_dirty = True
        self._refresh_plots_if_dirty()
        for image, bar, grid in ((self.vt_map_img, self.vt_map_bar, plan.vt), (self.vb_map_img, self.vb_map_bar, plan.vb),
                                 (self.vs_map_img, self.vs_map_bar, plan.vs)):
            self._set_map(image, bar, grid, plan.x, plan.y)
        self.lbl_status.setText(f"Status: Viewing {os.path.basename(path)} ({self.run_outcome}); "
                                "settings in the panel are unchanged.")
        self.append_log(f"Opened {path}")
        self._update_buttons()

    # -- shutdown ----------------------------------------------------------------
    def shutdown(self):
        """Stop polling and workers (called by the dashboard on close)."""
        self._plot_timer.stop()
        for worker in (self.sweep_worker, self.ramp_worker, self.control_worker):
            if worker is not None and worker.isRunning():
                worker.stop()
                worker.wait(10000)
        if self.live_status_worker.isRunning():
            self.live_status_worker.stop()
            self.live_status_worker.wait(3000)
        self.save_storage()

    def closeEvent(self, event):
        self.shutdown()
        if not self.embed:
            from gui_safety import uninstall_parameter_wheel_guard
            uninstall_parameter_wheel_guard(self)
        event.accept()
