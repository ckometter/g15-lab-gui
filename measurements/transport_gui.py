"""Four-probe transport maps with two SR830 lock-ins for the lab dashboard.

Ports the qcodes-qpg ``transport_measurement_v2.py`` workflow: one lock-in
reads the sample current, the other the four-probe voltage, while the gates
sweep a map (top gate x bottom gate, or n0 x p0). At every point the tab
records both lock-ins' X and Y and derives R4 = V_x / I_x and the
two-terminal R2. See ``docs/TRANSPORT_MEASUREMENTS.md``.

The electrode panels, sweep engine, plots and file handling are shared with
the Capacitance tab (``measurements/gate_map_gui.py``). Hardware access
happens only in worker threads. Every physical target of the sweep is
validated before anything moves, and both lock-ins' input modes are checked
before the first electrode move.
"""
import json
import math
import os
import sys

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from drivers.stanford_sr830 import TIME_CONSTANTS, describe_status, lockin_wait_time
from measurements.gate_map_gui import (COMMON_DEFAULTS, ELECTRODE_POINT_FIELDS, GateMapLiveWorker, GateMapTask,
                                       GateMapWindow, format_si, grid_writer, read_plan_datasets, run_plan,
                                       write_file_attributes, write_metadata, write_plan_datasets)
from transport_lockin import (CURRENT_MODES, LOCKIN_LABELS, LOCKINS, SimulatedTransportSample,
                              SimulatedTransportSignals, check_lockin_inputs, current_from_reading,
                              four_probe_resistance, lockin_roles, point_wait, sensitivity_for,
                              two_probe_resistance, validate_transport_settings, voltage_from_reading)

STORAGE_KEY = "transport"
FILE_SCHEMA = "g15/transport_lockin/1"
# transport_measurement_v2.py: back gate on uVoltDAQ dac0, top gate on dac1; the
# sample is grounded through the lock-in input (or current preamp).
DEFAULT_WIRING = {"top": "dac:1", "bottom": "dac:0", "sample": "none"}
# Before the first live reading, assume this time constant for the duration estimate.
ASSUMED_TIME_CONSTANT_S = 0.3

DEFAULTS = dict(COMMON_DEFAULTS, **{
    "fixed": "sample",
    "v_fixed": 0.0,
    "coordinate_mode": "direct",
    # The script's grid: top gate -0.8..0 V (20 points, slow), back gate -4.5..0 V (40, fast).
    "x_start": -0.8, "x_end": 0.0, "x_points": 20,
    "y_start": -4.5, "y_end": 0.0, "y_points": 40,
    "fast_axis": "y",
    "point_settle": 0.0,
    "line_settle": 1.0,
    "wiring": dict(DEFAULT_WIRING),
    "dac_ramp_rate": 0.5,
    # The script's station limits: top gate within +-0.8 V, back gate +-10 V (sample at 0 V).
    "max_top_sample": 0.8,
    "current_lockin": "lockin",
    "current_mode": "direct",
    "preamp_sensitivity": 1e-6,
    "voltage_gain": 1.0,
    "excitation_divider": 1.0,
    "series_resistance": 0.0,
    "time_constant": "keep",
    "settle_tc": 3.0,
    "autorange_current": False,
    "autorange_voltage": False,
    "configure_excitation": False,
    "frequency": 17.777,
    "amplitude": 0.1,
    "log_scale": False,
})

POINT_FIELDS = ("current_x", "current_y", "voltage_x", "voltage_y", "r4", "r2",
                "lockin_current_x", "lockin_current_y", "lockin_voltage_x", "lockin_voltage_y",
                "sensitivity_current", "sensitivity_voltage") + ELECTRODE_POINT_FIELDS
INT_FIELDS = {"status_current": 0, "status_voltage": 0}
# Dataset name, units and description of every per-point field. The raw units of
# the current lock-in ("A or V") are fixed when writing, from the current input.
FIELD_INFO = {
    "current_x": ("current_x", "A", "In-phase sample current"),
    "current_y": ("current_y", "A", "Quadrature sample current"),
    "voltage_x": ("voltage_x", "V", "In-phase four-probe voltage at the sample (after the preamp gain)"),
    "voltage_y": ("voltage_y", "V", "Quadrature four-probe voltage at the sample"),
    "r4": ("resistance_4pt", "Ohm", "Four-probe resistance voltage_x / current_x"),
    "r2": ("resistance_2pt", "Ohm", "Two-terminal resistance excitation / current_x - series resistor"),
    "lockin_current_x": ("lockin_current_x", None, "Current lock-in X as read"),
    "lockin_current_y": ("lockin_current_y", None, "Current lock-in Y as read"),
    "lockin_voltage_x": ("lockin_voltage_x", "V", "Voltage lock-in X as read"),
    "lockin_voltage_y": ("lockin_voltage_y", "V", "Voltage lock-in Y as read"),
    "sensitivity_current": ("sensitivity_current", None, "Current lock-in sensitivity"),
    "sensitivity_voltage": ("sensitivity_voltage", "V", "Voltage lock-in sensitivity"),
    "v_top": ("v_top", "V", "Top-gate readback"),
    "v_bottom": ("v_bottom", "V", "Bottom-gate readback"),
    "v_sample": ("v_sample", "V", "Sample readback"),
    "leakage_top": ("leakage_top", "A", "Keysight current of the top-gate output"),
    "leakage_bottom": ("leakage_bottom", "A", "Keysight current of the bottom-gate output"),
    "leakage_sample": ("leakage_sample", "A", "Keysight current of the sample output"),
    "aux_adc": ("aux_adc", "V", "Recorded uVoltDAQ ADC"),
    "timestamp": ("timestamp", "s", "Unix time of the read"),
}


# ---------------------------------------------------------------------------
# Worker-side operations (no GUI access; everything needed is passed in)
# ---------------------------------------------------------------------------

class TransportSession:
    """Operations on the two lock-ins of one task, honouring stop and pause."""

    def __init__(self, station, settings, control, log):
        self.station = station
        self.s = settings
        self.control = control
        self.log = log
        self.roles = lockin_roles(settings["current_lockin"])
        self.wait = float(settings["point_settle"])
        self.initial_wait = 0.0
        self.excitation_v = math.nan
        self.sensitivities = {"current": math.nan, "voltage": math.nan}

    def lockin(self, role):
        name = self.roles[role]
        component = self.station.components.get(name)
        if component is None:
            raise RuntimeError(f"{LOCKIN_LABELS[name]} ({role}) is not initialized. Connect it in the Device Manager.")
        return component

    @property
    def simulated(self):
        return all(getattr(self.lockin(role), "simulated", False) for role in ("current", "voltage"))

    def read_info(self, role):
        lockin = self.lockin(role)
        name = self.roles[role]
        return {"name": name, "label": LOCKIN_LABELS[name], "role": role,
                "input_config": lockin.input_config(), "reference_source": lockin.reference_source(),
                "frequency": float(lockin.frequency()), "amplitude": float(lockin.amplitude()),
                "phase": float(lockin.phase()), "time_constant": float(lockin.time_constant()),
                "filter_slope": int(lockin.filter_slope()), "sensitivity": float(lockin.sensitivity()),
                "sensitivity_unit": lockin.sensitivity.unit,
                "simulated": bool(getattr(lockin, "simulated", False))}

    def check_instruments(self):
        info = {role: self.read_info(role) for role in ("current", "voltage")}
        warnings = check_lockin_inputs(info["current"], info["voltage"], self.s)
        for role in ("current", "voltage"):
            item = info[role]
            self.log(f"{item['label']} ({role}): input {item['input_config']}, {item['reference_source']} "
                     f"reference at {item['frequency']:.6g} Hz, τ {item['time_constant']:g} s, "
                     f"{item['filter_slope']} dB/oct, sensitivity {item['sensitivity']:g} {item['sensitivity_unit']}.")
        for text in warnings:
            self.log("WARNING: " + text)
        return info, warnings

    def configure(self):
        """Optional excitation and time constant, then the per-point wait."""
        s = self.s
        current, voltage = self.lockin("current"), self.lockin("voltage")
        if s["configure_excitation"]:
            self.control.check()
            current.reference_source("internal")
            current.frequency(float(s["frequency"]))
            current.amplitude(float(s["amplitude"]))
            voltage.reference_source("external")
            if getattr(voltage, "simulated", False):
                voltage.frequency(float(s["frequency"]))  # a real lock-in follows its reference input
            frequency, amplitude = float(current.frequency()), float(current.amplitude())
            if not (math.isclose(frequency, float(s["frequency"]), rel_tol=1e-4)
                    and math.isclose(amplitude, float(s["amplitude"]), rel_tol=1e-2, abs_tol=2e-3)):
                raise RuntimeError(f"The excitation readback ({frequency:g} Hz, {amplitude:g} V) does not match "
                                   f"the request ({s['frequency']:g} Hz, {s['amplitude']:g} V).")
            self.log(f"Excitation set: {LOCKIN_LABELS[self.roles['current']]} internal reference, "
                     f"{frequency:.6g} Hz, {amplitude:g} V rms; {LOCKIN_LABELS[self.roles['voltage']]} "
                     "External reference.")
        if s["time_constant"] != "keep":
            for lockin in (current, voltage):
                self.control.check()
                lockin.time_constant(float(s["time_constant"]))
        taus = [float(lockin.time_constant()) for lockin in (current, voltage)]
        slopes = [int(lockin.filter_slope()) for lockin in (current, voltage)]
        self.excitation_v = float(current.amplitude()) * float(s["excitation_divider"])
        self.sensitivities = {"current": float(current.sensitivity()), "voltage": float(voltage.sensitivity())}
        if self.simulated:
            # The simulation has no filter: only the extra settle is waited.
            self.wait, self.initial_wait = float(s["point_settle"]), 0.0
            self.log(f"Simulated lock-ins have no filter: {self.wait:.3g} s per point (the extra settle).")
        else:
            self.wait = point_wait(taus, s["settle_tc"], s["point_settle"])
            self.initial_wait = max(lockin_wait_time(t, slope) for t, slope in zip(taus, slopes))
            self.log(f"Settle {s['settle_tc']:g} τ (τ = {max(taus):g} s) + {s['point_settle']:g} s = "
                     f"{self.wait:.3g} s per point; {self.initial_wait:.3g} s before the first point.")
        self.log(f"R2 uses V_exc = {self.excitation_v:.4g} V rms (sine output x divider).")
        return {"time_constants": taus, "filter_slopes": slopes, "point_wait_s": self.wait,
                "initial_wait_s": self.initial_wait, "excitation_V_rms": self.excitation_v}

    def clear_status(self):
        """Read (and clear) both status bytes: overloads latched while moving are dropped."""
        for role in ("current", "voltage"):
            self.lockin(role).status_byte()

    def read_point(self, on_pause):
        s = self.s
        current, voltage = self.lockin("current"), self.lockin("voltage")
        changed = False
        for role, lockin, enabled in (("current", current, s["autorange_current"]),
                                      ("voltage", voltage, s["autorange_voltage"])):
            if enabled:
                before = self.sensitivities[role]
                lockin.autorange(3)
                self.sensitivities[role] = float(lockin.sensitivity())
                changed = changed or self.sensitivities[role] != before
        if changed:
            # A new range needs the full settle again before the read.
            self.clear_status()
            self.control.sleep(self.wait, on_pause)
        # Read both lock-ins back to back (each with one coherent SNAP).
        current_x, current_y = current.snap_xy()
        voltage_x, voltage_y = voltage.snap_xy()
        status_current, status_voltage = int(current.status_byte()), int(voltage.status_byte())
        i_x, i_y = (float(current_from_reading(value, s["current_mode"], s["preamp_sensitivity"]))
                    for value in (current_x, current_y))
        v_x, v_y = (float(voltage_from_reading(value, s["voltage_gain"])) for value in (voltage_x, voltage_y))
        return {"lockin_current_x": current_x, "lockin_current_y": current_y,
                "lockin_voltage_x": voltage_x, "lockin_voltage_y": voltage_y,
                "current_x": i_x, "current_y": i_y, "voltage_x": v_x, "voltage_y": v_y,
                "r4": float(four_probe_resistance(v_x, i_x)),
                "r2": float(two_probe_resistance(self.excitation_v, i_x, s["series_resistance"])),
                "sensitivity_current": self.sensitivities["current"],
                "sensitivity_voltage": self.sensitivities["voltage"],
                "status_current": status_current, "status_voltage": status_voltage}


def run_transport(worker, station, settings, plan, electrodes):
    control = worker.control
    emit = worker.emit
    session = TransportSession(station, settings, control, lambda text: emit("log", text=text))
    moved = lambda values: emit("electrodes", values=values)
    emit("status", text="Status: Checking outputs and lock-ins...")
    electrodes.preflight()
    info, warnings = session.check_instruments()
    emit("lockin_info", info=info, warnings=warnings)
    emit("status", text="Status: Preparing the lock-ins...")
    emit("lockin_config", config=session.configure())

    def before_line(line_number, points, first):
        if line_number:
            return
        # Like the script's before_run(): reach the first point, then let the filters settle.
        emit("status", text="Status: Ramping to the first point...")
        electrodes.move_to(first, control, moved)
        if session.initial_wait > 0:
            emit("status", text="Status: Letting the lock-in filters settle...")
            control.sleep(session.initial_wait, lambda: emit("status", text="Status: Paused"))

    def measure(iy, ix, paused):
        session.clear_status()
        control.sleep(session.wait, paused)
        return session.read_point(paused)

    run_plan(worker, station, settings, plan, electrodes, measure=measure, before_line=before_line)
    return "Transport measurement complete."


class TransportLiveWorker(GateMapLiveWorker):
    """Idle polling of both lock-ins (configuration every 2 s) and the electrode setpoints."""

    def __init__(self, station, parent=None):
        super().__init__(station, parent)
        self._last_config = 0.0

    def poll_instruments(self, report, now):
        slow = now - self._last_config > 2.0
        if slow:
            self._last_config = now
        for name in LOCKINS:
            lockin = self.station.components.get(name)
            if lockin is None:
                continue
            try:
                x, y = lockin.snap_xy()
                entry = {"x": x, "y": y, "status": int(lockin.status_byte()),
                         "sensitivity": float(lockin.sensitivity())}
                if slow:
                    entry.update(input_config=lockin.input_config(), reference_source=lockin.reference_source(),
                                 frequency=float(lockin.frequency()), amplitude=float(lockin.amplitude()),
                                 time_constant=float(lockin.time_constant()))
                report[name] = entry
            except Exception as exc:
                report[name] = {"error": str(exc)}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def write_transport_file(path, *, plan, data, mask, attrs, settings, value_storage):
    raw_current_units = "A" if settings.get("current_mode") == "direct" else "V"
    with h5py.File(path, "w") as f:
        write_file_attributes(f, attrs, FILE_SCHEMA)
        grid = grid_writer(f, plan)
        write_plan_datasets(f, plan, grid)
        for key in POINT_FIELDS:
            name, units, description = FIELD_INFO[key]
            grid(name, data[key], raw_current_units if units is None else units, description)
        grid("status_current", data["status_current"], "", "SR830 LIAS? bits of the current lock-in")
        grid("status_voltage", data["status_voltage"], "", "SR830 LIAS? bits of the voltage lock-in")
        grid("measured_mask", mask, "", "True once the point was saved")
        f.create_dataset("acquisition_order", data=plan.order)
        write_metadata(f, settings, value_storage)


def read_transport_file(path):
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != FILE_SCHEMA:
            raise ValueError("This file is not a four-probe transport measurement saved by this dashboard.")
        data = {key: np.array(f[FIELD_INFO[key][0]]) for key in POINT_FIELDS}
        for key in INT_FIELDS:
            data[key] = np.array(f[key])
        return {"settings": json.loads(f["metadata/settings"][()]), "data": data,
                "mask": np.array(f["measured_mask"], dtype=bool), **read_plan_datasets(f),
                "attrs": {key: f.attrs[key] for key in f.attrs}}


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class TransportWindow(GateMapWindow):
    """Dashboard tab for four-probe lock-in transport maps and line sweeps."""

    STORAGE_KEY = STORAGE_KEY
    DEFAULTS = DEFAULTS
    DEFAULT_WIRING = DEFAULT_WIRING
    POINT_FIELDS = POINT_FIELDS
    INT_FIELDS = INT_FIELDS
    WINDOW_TITLE = "Four-probe transport"
    MEASUREMENT_NAME = "Transport measurement"
    TAB_NAME = "Transport"
    DATA_NOUN = "transport"
    DEVICE_LABELS = (("lockin", "Lock-in 1"), ("lockin2", "Lock-in 2"), ("dac", "DAC"), ("smu", "Keysight"))
    REQUIRED_DEVICES = (("lockin", "SR830 lock-in 1"), ("lockin2", "SR830 lock-in 2"))
    TASK_CLASS = GateMapTask
    LIVE_WORKER_CLASS = TransportLiveWorker
    START_EXPECTED_END = "Expected End: (after checking the lock-ins)"
    FIXED_TOOLTIP = ("Electrode held at V_fixed: normally the sample, grounded through the lock-in "
                     "input or current preamp.")
    POINT_SETTLE_LABEL = "Extra point settle"
    POINT_SETTLE_TOOLTIP = "Added to the settle in time constants (Lock-ins group) at every point."

    def init_detector_state(self):
        self.lockin_info = {}
        self.lockin_warnings = []
        self.lockin_config = {}
        self._live_lockins = {name: {} for name in LOCKINS}

    # -- UI --------------------------------------------------------------------
    def build_detector_groups(self):
        return [self._build_lockin_group()]

    def _build_lockin_group(self):
        group = QtWidgets.QGroupBox("Lock-ins: current and four-probe voltage")
        form = self._form(group)
        self.current_lockin_combo = self._combo(
            [(LOCKIN_LABELS[name], name) for name in LOCKINS],
            "Lock-in whose input reads the sample current; its sine output is the excitation. The other "
            "lock-in reads the four-probe voltage. Device Manager: 'SR830 lock-in 1/2'.")
        form.addRow("Current read by", self.current_lockin_combo)
        self.voltage_lockin_label = QtWidgets.QLabel()
        form.addRow("Voltage read by", self.voltage_lockin_label)
        self.current_mode_combo = self._combo([(label, key) for key, label in CURRENT_MODES.items()],
                                              "How the current lock-in's X and Y become the sample current.")
        form.addRow("Current input", self.current_mode_combo)
        self.preamp_spin = self._spin(1e-6, 1e-15, 1.0, "A/V", "Current preamp sensitivity: sample current per "
                                      "volt at the lock-in input. Used only with a current preamp.")
        form.addRow("Current preamp", self.preamp_spin)
        self.voltage_gain_spin = self._spin(1.0, 1e-3, 1e6, "V/V", "Gain of a voltage preamp before the voltage "
                                            "lock-in; 1 without one (transport_measurement.py used 100).")
        form.addRow("Voltage preamp gain", self.voltage_gain_spin)
        self.divider_spin = self._spin(1.0, 1e-9, 1e3, "", "Excitation reaching the sample per volt of sine "
                                       "output (0.01 for a 1:100 divider). Only R2 uses it.")
        form.addRow("Excitation divider", self.divider_spin)
        self.series_spin = self._spin(0.0, 0.0, 1e12, "Ω", "Current-limiting series resistor, subtracted from R2.")
        form.addRow("Series resistor", self.series_spin)
        constants = [("Keep front-panel value", "keep")]
        constants += [(format_si(t, "s", 3), float(t)) for t in TIME_CONSTANTS if 1e-3 <= t <= 30]
        self.tc_combo = self._combo(constants, "Time constant applied to both lock-ins at Begin.")
        form.addRow("Time constant τ", self.tc_combo)
        self.settle_tc_spin = self._spin(3.0, 0.0, 100.0, "τ", "Wait at every point in units of the longer time "
                                         "constant (the script used 3 τ). The extra point settle is added.")
        form.addRow("Point settle", self.settle_tc_spin)
        self.autorange_current_check = QtWidgets.QCheckBox("Autorange the current lock-in at every point")
        self.autorange_voltage_check = QtWidgets.QCheckBox("Autorange the voltage lock-in at every point")
        for check in (self.autorange_current_check, self.autorange_voltage_check):
            check.setToolTip("Keeps the signal on range where the resistance spans decades. A range change "
                             "repeats the point settle; the sensitivity is saved per point.")
            form.addRow(check)
        self.configure_excitation_check = QtWidgets.QCheckBox("Set the excitation at Begin")
        self.configure_excitation_check.setToolTip(
            "Current lock-in: internal reference, the frequency and sine amplitude below; voltage lock-in: "
            "External reference. Unchecked, the front-panel settings are kept and only read.")
        form.addRow(self.configure_excitation_check)
        self.frequency_spin = self._spin(17.777, 1e-3, 102e3, "Hz")
        self.amplitude_spin = self._spin(0.1, 0.004, 5.0, "V", "SR830 sine output amplitude (V rms).")
        form.addRow("Frequency, sine", self._row(self.frequency_spin, self.amplitude_spin))
        self.lockin_status_label = self._mono_label("Lock-in check: runs at Begin")
        form.addRow(self.lockin_status_label)
        self.current_lockin_combo.currentIndexChanged.connect(self._update_lockin_controls)
        self.current_mode_combo.currentIndexChanged.connect(self._update_lockin_controls)
        self.configure_excitation_check.toggled.connect(self._update_lockin_controls)
        return group

    def _update_lockin_controls(self, *args):
        roles = lockin_roles(self.current_lockin_combo.currentData())
        self.voltage_lockin_label.setText(LOCKIN_LABELS[roles["voltage"]])
        if not self.is_running:
            self.preamp_spin.setEnabled(self.current_mode_combo.currentData() == "preamp")
            for spin in (self.frequency_spin, self.amplitude_spin):
                spin.setEnabled(self.configure_excitation_check.isChecked())

    def set_inputs_enabled(self, enabled):
        super().set_inputs_enabled(enabled)
        if enabled:
            self._update_lockin_controls()

    def add_live_rows(self, form):
        self.live_lockin_label = self._mono_label("Lock-ins: --")
        form.addRow(self.live_lockin_label)

    def build_result_pages(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        header = self._plot_header()
        self.log_scale_check = QtWidgets.QCheckBox("log₁₀ |R| maps")
        self.log_scale_check.setToolTip("Show the resistance maps on a logarithmic colour scale; cuts stay linear.")
        self.log_scale_check.toggled.connect(self.on_log_scale_toggled)
        self.plot_header_layout.insertWidget(self.plot_header_layout.count() - 1, self.log_scale_check)
        self._display_controls.append(self.log_scale_check)
        layout.addWidget(header)
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid, 1)
        self.r4_map_plot, self.r4_map_img, self.r4_map_bar = self._make_map("Four-probe R4 = Vx / Ix")
        self.r2_map_plot, self.r2_map_img, self.r2_map_bar = self._make_map("Two-terminal R2")
        self.r4_cut_plot, self.r4_cut_curve = self._make_cut("R4 cut", "#2563eb")
        self.r2_cut_plot, self.r2_cut_curve = self._make_cut("R2 cut", "#e11d48")
        grid.addWidget(self.r4_map_plot, 0, 0)
        grid.addWidget(self.r2_map_plot, 0, 1)
        grid.addWidget(self.r4_cut_plot, 1, 0)
        grid.addWidget(self.r2_cut_plot, 1, 1)
        self.right_tabs.addTab(page, "Resistance")

        page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(page)
        self.i_map_plot, self.i_map_img, self.i_map_bar = self._make_map("Current Ix")
        self.v_map_plot, self.v_map_img, self.v_map_bar = self._make_map("Four-probe voltage Vx")
        self.i_cut_plot, self.ix_cut_curve = self._make_cut("Current cut: X solid, Y dashed", "#0f766e")
        self.v_cut_plot, self.vx_cut_curve = self._make_cut("Voltage cut: X solid, Y dashed", "#7c3aed")
        dashed = QtCore.Qt.PenStyle.DashLine
        self.iy_cut_curve = self.i_cut_plot.plot(pen=pg.mkPen("#0f766e", width=1.5, style=dashed))
        self.vy_cut_curve = self.v_cut_plot.plot(pen=pg.mkPen("#7c3aed", width=1.5, style=dashed))
        grid.addWidget(self.i_map_plot, 0, 0)
        grid.addWidget(self.v_map_plot, 0, 1)
        grid.addWidget(self.i_cut_plot, 1, 0)
        grid.addWidget(self.v_cut_plot, 1, 1)
        self.right_tabs.addTab(page, "Lock-in signals")

        self.map_items = {
            "r4": (self.r4_map_plot, self.r4_map_img, self.r4_map_bar),
            "r2": (self.r2_map_plot, self.r2_map_img, self.r2_map_bar),
            "ix": (self.i_map_plot, self.i_map_img, self.i_map_bar),
            "vx": (self.v_map_plot, self.v_map_img, self.v_map_bar),
        }
        # Voltages are plotted in uV: the dashboard keeps "V" axes prefix-free for gate voltages.
        self.cut_items = [("r4", self.r4_cut_curve, 1.0), ("r2", self.r2_cut_curve, 1.0),
                          ("ix", self.ix_cut_curve, 1.0), ("iy", self.iy_cut_curve, 1.0),
                          ("vx", self.vx_cut_curve, 1e6), ("vy", self.vy_cut_curve, 1e6)]
        self.cut_plots = [self.r4_cut_plot, self.r2_cut_plot, self.i_cut_plot, self.v_cut_plot]

    def detector_preview_widgets(self):
        return [self.current_lockin_combo, self.current_mode_combo, self.preamp_spin, self.voltage_gain_spin,
                self.divider_spin, self.series_spin, self.tc_combo, self.settle_tc_spin,
                self.configure_excitation_check, self.frequency_spin, self.amplitude_spin]

    # -- settings --------------------------------------------------------------
    def collect_detector_settings(self):
        return {
            "current_lockin": self.current_lockin_combo.currentData(),
            "current_mode": self.current_mode_combo.currentData(),
            "preamp_sensitivity": self.preamp_spin.value(),
            "voltage_gain": self.voltage_gain_spin.value(),
            "excitation_divider": self.divider_spin.value(),
            "series_resistance": self.series_spin.value(),
            "time_constant": self.tc_combo.currentData(),
            "settle_tc": self.settle_tc_spin.value(),
            "autorange_current": self.autorange_current_check.isChecked(),
            "autorange_voltage": self.autorange_voltage_check.isChecked(),
            "configure_excitation": self.configure_excitation_check.isChecked(),
            "frequency": self.frequency_spin.value(),
            "amplitude": self.amplitude_spin.value(),
            "log_scale": self.log_scale_check.isChecked(),
        }

    def apply_detector_values(self, s):
        self._set_combo(self.current_lockin_combo, s["current_lockin"])
        self._set_combo(self.current_mode_combo, s["current_mode"])
        self.preamp_spin.setValue(float(s["preamp_sensitivity"]))
        self.voltage_gain_spin.setValue(float(s["voltage_gain"]))
        self.divider_spin.setValue(float(s["excitation_divider"]))
        self.series_spin.setValue(float(s["series_resistance"]))
        tc = s["time_constant"] if s["time_constant"] == "keep" else float(s["time_constant"])
        self._set_combo(self.tc_combo, tc) or self._set_combo(self.tc_combo, "keep")
        self.settle_tc_spin.setValue(float(s["settle_tc"]))
        self.autorange_current_check.setChecked(bool(s["autorange_current"]))
        self.autorange_voltage_check.setChecked(bool(s["autorange_voltage"]))
        self.configure_excitation_check.setChecked(bool(s["configure_excitation"]))
        self.frequency_spin.setValue(float(s["frequency"]))
        self.amplitude_spin.setValue(float(s["amplitude"]))
        self.log_scale_check.setChecked(bool(s["log_scale"]))
        self._update_lockin_controls()

    def validate_detector(self, settings):
        validate_transport_settings(settings)

    # -- labels and preview ------------------------------------------------------
    def _assumed_tau(self, settings):
        """Time constant for the estimate: the requested one, else the slower live reading."""
        if settings["time_constant"] != "keep":
            return float(settings["time_constant"])
        taus = [entry["time_constant"] for entry in self._live_lockins.values() if "time_constant" in entry]
        return max(taus) if taus else ASSUMED_TIME_CONSTANT_S

    def _lockins_simulated(self, settings):
        roles = lockin_roles(settings["current_lockin"])
        components = [self.station.components.get(roles[role]) for role in ("current", "voltage")]
        return all(component is not None and getattr(component, "simulated", False) for component in components)

    def point_wait_estimate(self, settings):
        if self._lockins_simulated(settings):
            return float(settings["point_settle"])
        return point_wait([self._assumed_tau(settings)], settings["settle_tc"], settings["point_settle"])

    def preview_lines(self, settings, plan, electrodes, extra):
        roles = lockin_roles(settings["current_lockin"])
        current = LOCKIN_LABELS[roles["current"]]
        source = ("current input" if settings["current_mode"] == "direct"
                  else f"preamp {settings['preamp_sensitivity']:g} A/V")
        tau = "simulated, no filter" if self._lockins_simulated(settings) else f"τ ≈ {self._assumed_tau(settings):g} s"
        return [f"I: {current} ({source}); V: {LOCKIN_LABELS[roles['voltage']]} (gain {settings['voltage_gain']:g})",
                f"Settle {settings['settle_tc']:g} τ ({tau}) + {settings['point_settle']:g} s per point"]

    def update_detector_labels(self, x_name, y_name, cut_name):
        roles = lockin_roles(self.current_lockin_combo.currentData())
        current, voltage = LOCKIN_LABELS[roles["current"]], LOCKIN_LABELS[roles["voltage"]]
        self.wiring_note.setText(f"{current}: sine output excites the sample, input reads the current. {voltage}: "
                                 f"reads the inner-probe voltage, External reference from {current}.")
        log = self.log_scale_check.isChecked()
        self.r4_map_plot.setTitle("log₁₀ |R4 / Ω|, R4 = Vx / Ix" if log else "Four-probe R4 = Vx / Ix")
        self.r2_map_plot.setTitle("log₁₀ |R2 / Ω|, two-terminal" if log else "Two-terminal R2 = V_exc / Ix − R_series")
        for plot, label, units in ((self.r4_cut_plot, "R4", "Ω"), (self.r2_cut_plot, "R2", "Ω"),
                                   (self.i_cut_plot, "I", "A"), (self.v_cut_plot, "V", "µV")):
            plot.setLabel("bottom", cut_name, units="V")
            plot.setLabel("left", label, units=units)
            if units == "µV":
                plot.getAxis("left").enableAutoSIPrefix(False)
        for bar, units in ((self.r4_map_bar, "" if log else "Ω"), (self.r2_map_bar, "" if log else "Ω"),
                           (self.i_map_bar, "A"), (self.v_map_bar, "V")):
            axis = getattr(bar, "axis", None)
            if axis is not None:
                axis.setLabel(text="", units=units)

    def on_log_scale_toggled(self, checked):
        self._update_labels()
        self._plots_dirty = True

    # -- simulation ------------------------------------------------------------
    def bind_simulation(self, settings, electrodes, plan=None):
        """Connect a simulated lock-in pair to the sample model; physical lock-ins are never touched."""
        roles = lockin_roles(settings["current_lockin"])
        current = self.station.components.get(roles["current"])
        voltage = self.station.components.get(roles["voltage"])
        if not all(component is not None and getattr(component, "simulated", False)
                   for component in (current, voltage)):
            return
        direct = settings["current_mode"] == "direct"
        # Match the simulated front panels to the selected wiring: the current lock-in is the oscillator.
        current.input_config("I 1M" if direct else "a")
        current.reference_source("internal")
        voltage.input_config("a-b")
        voltage.reference_source("external")
        voltage.frequency(float(current.frequency()))
        voltage.time_constant(float(current.time_constant()))
        reader = self.electrode_voltage_reader(electrodes)
        signals = SimulatedTransportSignals(SimulatedTransportSample(delta=settings["delta"]),
                                            lambda: float(current.amplitude()), reader, settings, noise=1e-3)
        current.signal_source = signals.current_input
        voltage.signal_source = signals.voltage_input
        # Ranges with headroom over the whole map (or the present voltages), as an operator would set them.
        targets = (plan.vt, plan.vb, plan.vs) if plan is not None else tuple(np.array([v]) for v in reader())
        excitation = float(settings["amplitude"]) if settings["configure_excitation"] else None
        peak_current, peak_voltage = signals.peak_readings(*targets, excitation)
        current.sensitivity(sensitivity_for(1.25 * peak_current, direct))
        voltage.sensitivity(sensitivity_for(1.25 * peak_voltage, False))

    def reload_gui_from_storage(self):
        super().reload_gui_from_storage()
        # Simulated lock-ins follow the sample model while the tab is shown (physical ones are untouched).
        if not self.busy():
            try:
                settings = self.collect_settings()
                self.bind_simulation(settings, self.make_electrodes(settings))
            except Exception:
                pass

    # -- sweep -----------------------------------------------------------------
    def prepare_run(self, settings, plan, electrodes, extra):
        self.bind_simulation(settings, electrodes, plan)

    def reset_run_state(self):
        self.lockin_info, self.lockin_warnings, self.lockin_config = {}, [], {}

    def sweep_task(self, settings, plan, electrodes):
        return run_transport, (self.station, settings, plan, electrodes)

    def on_detector_event(self, event):
        kind = event.get("kind")
        if kind == "lockin_info":
            self.lockin_info = event["info"]
            self.lockin_warnings = list(event.get("warnings", []))
            self._show_lockin_check()
        elif kind == "lockin_config":
            self.lockin_config = event["config"]
            self._show_lockin_check()

    def _show_lockin_check(self):
        lines = []
        for role in ("current", "voltage"):
            item = self.lockin_info.get(role)
            if item:
                lines.append(f"{item['label']} ({role}): {item['input_config']}, {item['reference_source']} ref "
                             f"{item['frequency']:.6g} Hz, τ {item['time_constant']:g} s")
        if self.lockin_config:
            lines.append(f"{self.lockin_config['point_wait_s']:.3g} s per point; "
                         f"V_exc {self.lockin_config['excitation_V_rms']:.4g} V rms")
        lines += [f"⚠ {text}" for text in self.lockin_warnings]
        self.lockin_status_label.setText("\n".join(lines) or "Lock-in check: runs at Begin")
        self.lockin_status_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: "
                                               + ("#b45309;" if self.lockin_warnings else "#0f172a;"))

    def point_status(self, values):
        return int(values["status_current"]) | int(values["status_voltage"])

    # -- live readout ------------------------------------------------------------
    def on_live_detector(self, report):
        for name in LOCKINS:
            if name in report:
                entry = report[name]
                if "error" in entry:
                    self._live_lockins[name] = {"error": entry["error"]}
                else:
                    self._live_lockins[name] = {**{k: v for k, v in self._live_lockins[name].items()
                                                   if k != "error"}, **entry}
        if not any(name in report for name in LOCKINS):
            return
        try:
            settings = self.collect_settings()
            roles = lockin_roles(settings["current_lockin"])
        except Exception:
            return
        current, voltage = self._live_lockins[roles["current"]], self._live_lockins[roles["voltage"]]
        lines, flags = [], []
        for role, entry in (("current", current), ("voltage", voltage)):
            label = LOCKIN_LABELS[roles[role]]
            if "error" in entry:
                lines.append(f"{role[0].upper()} {label}: read failed ({entry['error']})")
            elif "x" in entry:
                flags += [f"{label}: {flag}" for flag in describe_status(entry.get("status", 0))]
        if "x" in current and "x" in voltage:
            i_x, i_y = (float(current_from_reading(current[key], settings["current_mode"],
                                                   settings["preamp_sensitivity"])) for key in ("x", "y"))
            v_x, v_y = (float(voltage_from_reading(voltage[key], settings["voltage_gain"])) for key in ("x", "y"))
            r4 = float(four_probe_resistance(v_x, i_x))
            excitation = current.get("amplitude", math.nan) * settings["excitation_divider"]
            r2 = float(two_probe_resistance(excitation, i_x, settings["series_resistance"]))
            lines.insert(0, f"I  X {format_si(i_x, 'A')}  Y {format_si(i_y, 'A')}")
            lines.insert(1, f"V  X {format_si(v_x, 'V')}  Y {format_si(v_y, 'V')}")
            lines.insert(2, f"R4 {format_si(r4, 'Ω')}   R2 {format_si(r2, 'Ω')}")
        if "time_constant" in current and "time_constant" in voltage:
            lines.append(f"τ {format_si(current['time_constant'], 's', 3)} / "
                         f"{format_si(voltage['time_constant'], 's', 3)}  f {current['frequency']:.5g} / "
                         f"{voltage['frequency']:.5g} Hz  ref {current['reference_source'][:3]} / "
                         f"{voltage['reference_source'][:3]}")
            try:
                flags += check_lockin_inputs({"label": LOCKIN_LABELS[roles["current"]], **current},
                                             {"label": LOCKIN_LABELS[roles["voltage"]], **voltage}, settings)
            except RuntimeError as exc:
                flags.append(str(exc).replace(" Nothing was moved.", ""))
        text = "\n".join(lines) or "Lock-ins: --"
        if flags:
            text += "\n⚠ " + "\n⚠ ".join(flags)
        self.live_lockin_label.setText(text)
        self.live_lockin_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: "
                                             + ("#b45309;" if flags else "#0f172a;"))

    # -- plots -----------------------------------------------------------------
    def cut_arrays(self):
        data = self.data
        return {"r4": data["r4"], "r2": data["r2"], "ix": data["current_x"], "iy": data["current_y"],
                "vx": data["voltage_x"], "vy": data["voltage_y"]}

    def display_arrays(self):
        arrays = self.cut_arrays()
        if self.log_scale_check.isChecked():
            for key in ("r4", "r2"):
                magnitude = np.abs(arrays[key])
                with np.errstate(divide="ignore", invalid="ignore"):
                    arrays[key] = np.where(magnitude > 0, np.log10(magnitude), np.nan)
        return arrays

    # -- save / load -------------------------------------------------------------
    def filename_tokens(self, settings):
        return ["transport", "4pt"]

    def detector_file_attributes(self, settings):
        roles = lockin_roles(settings.get("current_lockin", "lockin"))
        config = self.lockin_config
        return {
            "measurement_label": "Four-probe transport with two SR830 lock-ins",
            "current_lockin": roles["current"], "voltage_lockin": roles["voltage"],
            "current_mode": settings.get("current_mode", ""),
            "current_preamp_A_per_V": float(settings.get("preamp_sensitivity", math.nan)),
            "voltage_gain": float(settings.get("voltage_gain", math.nan)),
            "excitation_divider": float(settings.get("excitation_divider", math.nan)),
            "series_resistance_ohm": float(settings.get("series_resistance", math.nan)),
            "excitation_V_rms": float(config.get("excitation_V_rms", math.nan)),
            "settle_tc": float(settings.get("settle_tc", math.nan)),
            "point_wait_s": float(config.get("point_wait_s", math.nan)),
            "time_constants_s": list(config.get("time_constants", [])),
            "lockin_info": self.lockin_info, "lockin_warnings": list(self.lockin_warnings),
            "resistance_definitions": "resistance_4pt = voltage_x / current_x; resistance_2pt = "
                                      "excitation_V_rms / current_x - series_resistance_ohm",
        }

    def write_file(self, path):
        write_transport_file(path, plan=self.plan, data=self.data, mask=self.measured_mask,
                             attrs=self.file_attributes(), settings=self.run_settings or self.collect_settings(),
                             value_storage=self.value_storage)

    def read_file(self, path):
        return read_transport_file(path)

    def apply_loaded(self, loaded):
        attrs = loaded["attrs"]
        try:
            self.lockin_info = json.loads(attrs.get("lockin_info", "{}"))
            self.lockin_warnings = json.loads(attrs.get("lockin_warnings", "[]"))
        except (TypeError, ValueError):
            self.lockin_info, self.lockin_warnings = {}, []
        self.lockin_config = {key: float(attrs[name]) for key, name in
                              (("point_wait_s", "point_wait_s"), ("excitation_V_rms", "excitation_V_rms"))
                              if name in attrs}
        if len(self.lockin_config) < 2:
            self.lockin_config = {}
        self._show_lockin_check()


if __name__ == "__main__":
    from station_devices import create_station, ensure_simulated_devices
    app = QtWidgets.QApplication(sys.argv)
    station = create_station(os.path.join(parent_dir, "my_measurement_data.db"))
    ensure_simulated_devices(station)
    window = TransportWindow(station)
    window.resize(1600, 950)
    window.show()
    sys.exit(app.exec())
