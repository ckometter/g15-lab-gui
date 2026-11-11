"""Capacitance-bridge measurements (C_P, C_T, C_B, C_S) for the lab dashboard.

Ports the qcodes-qpg ``capacitance_measurement.py`` workflow into the dashboard:
balance an AC capacitance bridge (SR830 lock-in, two-channel Siglent AWG), then
sweep the normalized density n0 and field p0 (or two electrode voltages
directly) and convert the off-balance lock-in signal into capacitance and
dissipation at every point. See ``docs/CAPACITANCE_MEASUREMENTS.md``.

The electrode panels, sweep engine, plots and file handling are shared with
the Transport tab (``measurements/gate_map_gui.py``). Hardware access happens
only in worker threads. Every physical target of the complete sweep is
validated before anything moves; nothing is clipped.
"""
import json
import math
import os
import sys
import time

import h5py
import numpy as np
from PyQt6 import QtCore, QtWidgets

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from capacitance_bridge import (BalanceCancelled, MEASUREMENT_TYPES, LinearBridge, SimulatedBridgeSignal,
                                SimulatedSample, ac_scale_factor, vector_phase_deg)
from drivers.stanford_sr830 import TIME_CONSTANTS, VOLTAGE_SENSITIVITIES, describe_status, lockin_wait_time
from electrode_outputs import DEFAULT_WIRING, MeasurementStopped
from electrode_sweep import ELECTRODE_LABELS, density_field
from measurement_filenames import axis_token
from measurements.gate_map_gui import (COMMON_DEFAULTS, GateMapLiveWorker, GateMapTask, GateMapWindow,  # noqa: F401
                                       finite_levels, format_si, grid_writer, image_extent, read_plan_datasets,
                                       run_move, run_plan, write_file_attributes, write_metadata,
                                       write_plan_datasets)

STORAGE_KEY = "capacitance"
FILE_SCHEMA = "g15/capacitance_bridge/1"
# Legacy qcodes-qpg mapping: balancing sets the relative phase on CH1 (sample
# excitation, also the lock-in reference) and the amplitude of CH2 (reference arm).
SAMPLE_CHANNEL = 1
REFERENCE_CHANNEL = 2

DEFAULTS = dict(COMMON_DEFAULTS, **{
    "measurement_type": "penetration",
    "frequency": 7777.7,
    "sample_amplitude": 0.1,
    "reference_amplitude": 0.3,
    "sample_atten": 40.0,
    "ref_atten": 20.0,
    "max_reference_amplitude": 1.0,
    "configure_awg": True,
    "balance_tc": 0.3,
    "measure_tc": 0.3,
    "sensitivity": 1e-4,
    "tolerance": 1.5e-5,
    "iterations": 10,
    "s1": [0.1, 0.1],
    "s2": [-0.1, -0.1],
    "balance_at": "start",
    "balance_x": -1.0,
    "balance_y": -1.0,
    "rebalance": "never",
    "c_ref": 0.0,
})

POINT_FIELDS = ("lockin_x", "lockin_y", "capacitance", "dissipation", "v_top", "v_bottom", "v_sample",
                "leakage_top", "leakage_bottom", "leakage_sample", "sensitivity", "aux_adc", "timestamp")
INT_FIELDS = {"lockin_status": 0, "balance_index": -1}


# ---------------------------------------------------------------------------
# Worker-side operations (no GUI access; everything needed is passed in)
# ---------------------------------------------------------------------------

class BridgeSession:
    """Lock-in and AWG operations of one task, honouring stop and pause."""

    def __init__(self, station, settings, control, log):
        self.station = station
        self.s = settings
        self.control = control
        self.log = log

    def _component(self, name, label):
        component = self.station.components.get(name)
        if component is None:
            raise RuntimeError(f"The {label} is not initialized. Connect it in the Device Manager.")
        return component

    @property
    def lockin(self):
        return self._component("lockin", "SR830 lock-in")

    @property
    def awg(self):
        return self._component("awg", "Siglent AWG")

    def wait_filter(self, time_constant=None):
        """Original wait_time(): 5-10 time constants; the simulation has no filter."""
        lockin = self.lockin
        if lockin.simulated:
            return
        time_constant = float(lockin.time_constant()) if time_constant is None else float(time_constant)
        self.control.sleep(lockin_wait_time(time_constant, lockin.filter_slope()))

    def check_instruments(self):
        lockin, awg = self.lockin, self.awg
        info = {"input_config": lockin.input_config(), "reference_source": lockin.reference_source(),
                "frequency": float(lockin.frequency()), "filter_slope": int(lockin.filter_slope()),
                "lockin_simulated": bool(lockin.simulated), "awg_simulated": bool(awg.simulated)}
        if self.s["sensitivity"] not in ("keep", "auto") and info["input_config"] not in ("a", "a-b"):
            raise RuntimeError("The lock-in is in current-input mode; a fixed voltage sensitivity "
                               "cannot be applied. Choose 'Keep after balance' or 'Autorange each point'.")
        self.log(f"Lock-in: input {info['input_config']}, {info['reference_source']} reference at "
                 f"{info['frequency']:.6g} Hz, {info['filter_slope']} dB/oct.")
        return info

    def warn_reference(self, info):
        if info["lockin_simulated"] or info["reference_source"] != "external":
            return
        expected = float(self.s["frequency"])
        if expected > 0 and abs(info["frequency"] - expected) > 0.01 * expected:
            self.log(f"WARNING: the lock-in reads {info['frequency']:.6g} Hz but the excitation is "
                     f"{expected:.6g} Hz. Check that the lock-in reference follows AWG CH{SAMPLE_CHANNEL}.")

    def configure_excitation(self):
        """Original init_awg(): sine on both channels, set amplitudes, zero phases, outputs on."""
        awg, s = self.awg, self.s
        for channel in (awg.ch1, awg.ch2):
            self.control.check()
            channel.wave_type("SINE")
            channel.frequency(float(s["frequency"]))
        getattr(awg, f"ch{SAMPLE_CHANNEL}").amplitude(float(s["sample_amplitude"]))
        getattr(awg, f"ch{REFERENCE_CHANNEL}").amplitude(float(s["reference_amplitude"]))
        awg.ch1.phase(0.0)
        awg.ch2.phase(0.0)
        awg.ch1.enabled(True)
        awg.ch2.enabled(True)
        lockin = self.station.components.get("lockin")
        if lockin is not None and lockin.simulated:
            lockin.frequency(float(s["frequency"]))
        if not awg.simulated:
            self.control.sleep(1.0)  # the original script waited 1 s after configuring
        states = self.read_awg()
        sample = states[SAMPLE_CHANNEL]
        if not (sample["enabled"] and math.isclose(sample["frequency"], float(s["frequency"]), rel_tol=1e-6)
                and math.isclose(sample["amplitude"], float(s["sample_amplitude"]), rel_tol=1e-3)):
            raise RuntimeError(f"AWG readback does not match the requested excitation: {states}")
        self.log(f"AWG configured: {s['frequency']:.6g} Hz, CH{SAMPLE_CHANNEL} sample "
                 f"{s['sample_amplitude']:g} Vpp, CH{REFERENCE_CHANNEL} reference "
                 f"{s['reference_amplitude']:g} Vpp, outputs on.")
        return states

    def read_awg(self):
        awg = self.awg
        return {number: getattr(awg, f"ch{number}").state() for number in (1, 2)}

    def excite(self, vector):
        awg = self.awg
        getattr(awg, f"ch{SAMPLE_CHANNEL}").phase(vector_phase_deg(vector))
        getattr(awg, f"ch{REFERENCE_CHANNEL}").amplitude(float(np.linalg.norm(vector)))

    def measure_for_balance(self):
        """Original measure(): wait, autorange (max 5 changes), read X and Y."""
        self.wait_filter()
        self.control.check()
        lockin = self.lockin
        lockin.autorange(5)
        return lockin.snap_xy()

    def balance(self, previous=None):
        s, lockin = self.s, self.lockin
        lockin.time_constant(float(s["balance_tc"]))
        self.wait_filter(float(s["balance_tc"]))
        if previous is None:
            lockin.autorange(8)
        bridge = LinearBridge(
            self.excite, self.measure_for_balance, s1=s["s1"], s2=s["s2"],
            tolerance=float(s["tolerance"]), iterations=int(s["iterations"]),
            ac_scale=ac_scale_factor(s["ref_atten"], s["sample_atten"], s["sample_amplitude"]),
            max_amplitude=float(s["max_reference_amplitude"]), log=self.log,
            keep_running=self.control.keep_running)
        result = bridge.balance() if previous is None else bridge.rebalance(previous)
        c_coeff, d_coeff = result.offbalance_coefficients
        self.log(f"Off-balance factors: c' = {c_coeff:.6g} /V, d' = {d_coeff:.6g} /V "
                 f"(reference {result.reference_amplitude:.4g} Vpp at {result.reference_phase:.2f} deg).")
        return result

    def apply_measurement_settings(self):
        s, lockin = self.s, self.lockin
        if s["sensitivity"] not in ("keep", "auto"):
            lockin.sensitivity(float(s["sensitivity"]))
        lockin.time_constant(float(s["measure_tc"]))
        self.wait_filter(float(s["measure_tc"]))
        return float(lockin.sensitivity())

    def read_point(self, sensitivity):
        lockin = self.lockin
        if self.s["sensitivity"] == "auto":
            lockin.autorange(3)
            sensitivity = float(lockin.sensitivity())
        x, y = lockin.snap_xy()
        return x, y, lockin.status_byte(), sensitivity


def balance_event(result, index, location, targets):
    record = result.as_dict()
    record.update(index=index, x=float(location[0]), y=float(location[1]), targets=dict(targets),
                  time=time.time(), reference_amplitude=result.reference_amplitude,
                  reference_phase=result.reference_phase, magnitude=result.magnitude,
                  phase_deg=result.phase_deg)
    return record


def run_sweep(worker, station, settings, plan, electrodes):
    control = worker.control
    emit = worker.emit
    session = BridgeSession(station, settings, control, lambda text: emit("log", text=text))
    moved = lambda values: emit("electrodes", values=values)
    emit("status", text="Status: Checking instruments...")
    electrodes.preflight()
    info = session.check_instruments()
    emit("lockin_info", info=info)
    if settings["configure_awg"]:
        emit("status", text="Status: Configuring AWG...")
        emit("awg", states=session.configure_excitation())
    else:
        emit("awg", states=session.read_awg())
    session.warn_reference(info)
    balance_targets = settings["balance_targets"]
    emit("status", text="Status: Ramping to the balance point...")
    electrodes.move_to(balance_targets, control, moved)
    emit("status", text="Status: Balancing bridge...")
    state = {"result": session.balance(), "index": 0}
    emit("balance", record=balance_event(state["result"], 0, settings["balance_location"], balance_targets))
    emit("awg", states=session.read_awg())
    if not state["result"].converged:
        emit("log", text="WARNING: balance did not reach the tolerance; the off-balance calibration "
                         "still corrects the residual (as in the original script).")
    state["sensitivity"] = session.apply_measurement_settings()

    def before_line(line_number, points, first):
        if not (line_number and settings["rebalance"] == "line"):
            return
        electrodes.move_to(first, control, moved)
        emit("status", text=f"Status: Rebalancing (line {line_number + 1})...")
        state["result"] = session.balance(previous=state["result"])
        state["index"] += 1
        location = (plan.x[points[0][1]], plan.y[points[0][0]])
        emit("balance", record=balance_event(state["result"], state["index"], location, first))
        emit("awg", states=session.read_awg())
        state["sensitivity"] = session.apply_measurement_settings()

    def measure(iy, ix, paused):
        session.lockin.status_byte()  # clear overloads latched while moving
        control.sleep(settings["point_settle"], paused)
        x, y, status, state["sensitivity"] = session.read_point(state["sensitivity"])
        capacitance, dissipation = state["result"].offbalance(x, y)
        return {"lockin_x": x, "lockin_y": y, "capacitance": float(capacitance),
                "dissipation": float(dissipation), "sensitivity": state["sensitivity"],
                "lockin_status": int(status), "balance_index": state["index"]}

    run_plan(worker, station, settings, plan, electrodes, measure=measure, before_line=before_line)
    return "Capacitance measurement complete."


def run_balance(worker, station, settings, electrodes):
    emit = worker.emit
    session = BridgeSession(station, settings, worker.control, lambda text: emit("log", text=text))
    emit("status", text="Status: Checking instruments...")
    info = session.check_instruments()
    if settings["configure_awg"]:
        emit("awg", states=session.configure_excitation())
    session.warn_reference(info)
    setpoints = electrodes.setpoints()
    emit("electrodes", values=setpoints)
    emit("status", text="Status: Balancing bridge at the present electrode voltages...")
    result = session.balance()
    emit("balance", record=balance_event(result, -1, settings["here"], setpoints))
    emit("awg", states=session.read_awg())
    session.apply_measurement_settings()
    return result.message


def run_awg(worker, station, settings, action):
    emit = worker.emit
    session = BridgeSession(station, settings, worker.control, lambda text: emit("log", text=text))
    if action == "apply":
        states = session.configure_excitation()
    else:
        if action == "off":
            for channel in (session.awg.ch1, session.awg.ch2):
                channel.enabled(False)
            emit("log", text="AWG outputs switched off.")
        states = session.read_awg()
    emit("awg", states=states)
    return {"apply": "Excitation applied.", "off": "AWG outputs off.", "read": "AWG read."}[action]


class CapacitanceTask(GateMapTask):
    """A gate-map task that also treats a stopped bridge balance as a user stop."""
    STOP_EXCEPTIONS = (MeasurementStopped, BalanceCancelled)


class CapacitanceLiveWorker(GateMapLiveWorker):
    """Idle polling of the lock-in, electrode setpoints and (slowly) the AWG."""

    def __init__(self, station, parent=None):
        super().__init__(station, parent)
        self._last_awg = 0.0

    def poll_instruments(self, report, now):
        lockin = self.station.components.get("lockin")
        if lockin is not None:
            try:
                x, y = lockin.snap_xy()
                report.update(x=x, y=y, status=lockin.status_byte(),
                              sensitivity=float(lockin.sensitivity()))
            except Exception as exc:
                report["lockin_error"] = str(exc)
        awg = self.station.components.get("awg")
        if awg is not None and now - self._last_awg > 3.0:
            self._last_awg = now
            try:
                report["awg"] = {number: getattr(awg, f"ch{number}").state() for number in (1, 2)}
            except Exception as exc:
                report["awg_error"] = str(exc)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def write_capacitance_file(path, *, plan, data, mask, balances, attrs, settings, value_storage):
    string = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        write_file_attributes(f, attrs, FILE_SCHEMA)
        grid = grid_writer(f, plan)
        write_plan_datasets(f, plan, grid)
        units = {"lockin_x": "V", "lockin_y": "V", "capacitance": "", "dissipation": "",
                 "v_top": "V", "v_bottom": "V", "v_sample": "V", "leakage_top": "A",
                 "leakage_bottom": "A", "leakage_sample": "A", "sensitivity": "V", "aux_adc": "V",
                 "timestamp": "s"}
        names = {"capacitance": "capacitance_ratio", "dissipation": "dissipation_ratio",
                 "sensitivity": "lockin_sensitivity"}
        for key in POINT_FIELDS:
            grid(names.get(key, key), data[key], units[key])
        f["capacitance_ratio"].attrs["description"] = "C_sample / C_ref from the off-balance calibration"
        f["dissipation_ratio"].attrs["description"] = "D_sample / C_ref from the off-balance calibration"
        c_ref = float(settings.get("c_ref", 0.0))
        if c_ref > 0:
            grid("capacitance_F", data["capacitance"] * c_ref, "F")
            grid("dissipation_F", data["dissipation"] * c_ref, "F")
        grid("lockin_status", data["lockin_status"], "", "SR830 LIAS? bits during the read")
        grid("balance_index", data["balance_index"], "", "Index into /balance used for the point")
        grid("measured_mask", mask, "", "True once the point was saved")
        f.create_dataset("acquisition_order", data=plan.order)
        group = f.create_group("balance")
        if balances:
            for key in ("matrix", "offset", "balance_vector"):
                group.create_dataset(key, data=np.array([b[key] for b in balances], dtype=float))
            for key in ("capacitance_ratio", "dissipation_ratio", "c_coefficient", "d_coefficient",
                        "residual_V", "x", "y", "time", "reference_amplitude", "reference_phase"):
                group.create_dataset(key, data=np.array([b[key] for b in balances], dtype=float))
            group.create_dataset("iterations", data=np.array([b["iterations"] for b in balances], dtype=int))
            group.create_dataset("converged", data=np.array([b["converged"] for b in balances], dtype=bool))
        # A dataset, not an attribute: per-line rebalancing can exceed the 64 kB attribute limit.
        group.create_dataset("records_json", data=json.dumps(balances), dtype=string)
        write_metadata(f, settings, value_storage)


def read_capacitance_file(path):
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != FILE_SCHEMA:
            raise ValueError("This file is not a capacitance-bridge measurement saved by this dashboard.")
        settings = json.loads(f["metadata/settings"][()])
        names = {"capacitance": "capacitance_ratio", "dissipation": "dissipation_ratio",
                 "sensitivity": "lockin_sensitivity"}
        data = {key: np.array(f[names.get(key, key)]) for key in POINT_FIELDS}
        for key in INT_FIELDS:
            data[key] = np.array(f[key])
        result = {
            "settings": settings, "data": data, "mask": np.array(f["measured_mask"], dtype=bool),
            **read_plan_datasets(f),
            "balances": json.loads(f["balance/records_json"][()]),
            "attrs": {key: f.attrs[key] for key in f.attrs},
        }
    return result


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class CapacitanceWindow(GateMapWindow):
    """Dashboard tab for capacitance-bridge maps and line sweeps."""

    STORAGE_KEY = STORAGE_KEY
    DEFAULTS = DEFAULTS
    DEFAULT_WIRING = DEFAULT_WIRING
    POINT_FIELDS = POINT_FIELDS
    INT_FIELDS = INT_FIELDS
    WINDOW_TITLE = "Capacitance bridge"
    MEASUREMENT_NAME = "Capacitance measurement"
    TAB_NAME = "Capacitance"
    DATA_NOUN = "capacitance"
    DEVICE_LABELS = (("lockin", "Lock-in"), ("awg", "AWG"), ("dac", "DAC"), ("smu", "Keysight"))
    REQUIRED_DEVICES = (("lockin", "SR830 lock-in"), ("awg", "Siglent AWG"))
    TASK_CLASS = CapacitanceTask
    LIVE_WORKER_CLASS = CapacitanceLiveWorker
    START_EXPECTED_END = "Expected End: (after balancing)"
    FIXED_TOOLTIP = "Electrode held at V_fixed, normally the bridge measurement node."
    POINT_SETTLE_TOOLTIP = "Wait after each point's ramp, before reading X/Y."

    def init_detector_state(self):
        self.balances = []
        self.last_balance = None
        self.lockin_info = {}
        self.awg_states = {}

    # -- UI --------------------------------------------------------------------
    def add_measurement_rows(self, form):
        self.type_combo = self._combo([(spec["label"], key) for key, spec in MEASUREMENT_TYPES.items()],
                                      "Selecting a type sets its usual DC-fixed (measurement) electrode.")
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        form.addRow("Type", self.type_combo)

    def build_detector_groups(self):
        return [self._build_awg_group(), self._build_balance_group()]

    def _build_awg_group(self):
        group = QtWidgets.QGroupBox("AC excitation (Siglent AWG)")
        form = self._form(group)
        self.freq_spin = self._spin(7777.7, 1e-3, 1e5, "Hz")
        form.addRow("Frequency", self.freq_spin)
        self.sample_amp_spin = self._spin(0.1, 2e-3, 20, "Vpp", f"CH{SAMPLE_CHANNEL}: sample excitation (fixed).")
        form.addRow(f"CH{SAMPLE_CHANNEL} sample", self.sample_amp_spin)
        self.ref_amp_spin = self._spin(0.3, 2e-3, 20, "Vpp", f"CH{REFERENCE_CHANNEL}: reference amplitude "
                                       "before balancing.")
        form.addRow(f"CH{REFERENCE_CHANNEL} reference", self.ref_amp_spin)
        self.sample_atten_spin = self._spin(40.0, 0, 200, "dB")
        form.addRow("Sample attenuation", self.sample_atten_spin)
        self.ref_atten_spin = self._spin(20.0, 0, 200, "dB")
        form.addRow("Reference attenuation", self.ref_atten_spin)
        self.max_ref_spin = self._spin(1.0, 1e-3, 20, "V", "Balance fails rather than exceed this reference "
                                       "component (original limit: 1 V).")
        form.addRow("Max balance component", self.max_ref_spin)
        self.configure_awg_check = QtWidgets.QCheckBox("Configure AWG at start")
        self.configure_awg_check.setToolTip("Before balancing: sine on both channels, frequency, amplitudes, "
                                            "zero phases, outputs on (the script's init_awg).")
        form.addRow(self.configure_awg_check)
        self.btn_awg_apply = QtWidgets.QPushButton("Apply")
        self.btn_awg_apply.setToolTip("Write frequency, amplitudes and zero phases, then enable both outputs.")
        self.btn_awg_apply.clicked.connect(lambda: self.run_awg_action("apply"))
        self.btn_awg_off = QtWidgets.QPushButton("Outputs off")
        self.btn_awg_off.clicked.connect(lambda: self.run_awg_action("off"))
        self.btn_awg_read = QtWidgets.QPushButton("Read")
        self.btn_awg_read.clicked.connect(lambda: self.run_awg_action("read"))
        form.addRow(self._row(self.btn_awg_apply, self.btn_awg_off, self.btn_awg_read))
        self.awg_label = self._note("AWG: not read")
        self.awg_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        form.addRow(self.awg_label)
        self._idle_buttons += [self.btn_awg_apply, self.btn_awg_off, self.btn_awg_read]
        return group

    def _build_balance_group(self):
        group = QtWidgets.QGroupBox("Lock-in and bridge balance")
        form = self._form(group)
        constants = [(format_si(t, "s", 3), float(t)) for t in TIME_CONSTANTS if 1e-3 <= t <= 30]
        self.balance_tc_combo = self._combo(constants, "Time constant while balancing.")
        form.addRow("Balance τ", self.balance_tc_combo)
        self.measure_tc_combo = self._combo(constants, "Time constant during the sweep.")
        form.addRow("Measurement τ", self.measure_tc_combo)
        sensitivities = [("Keep after balance", "keep"), ("Autorange each point", "auto")]
        sensitivities += [(format_si(v, "V", 3), float(v)) for v in VOLTAGE_SENSITIVITIES]
        self.sensitivity_combo = self._combo(sensitivities, "Sensitivity during the sweep (original: fixed).")
        form.addRow("Sensitivity", self.sensitivity_combo)
        self.tolerance_spin = self._spin(1.5e-5, 1e-12, 1.0, "V", "Balanced when |X + iY| < tolerance.")
        form.addRow("Tolerance", self.tolerance_spin)
        self.iterations_spin = self._points(10, 100)
        form.addRow("Max iterations", self.iterations_spin)
        self.probe_spins = []
        for index, name in enumerate(("s1", "s2")):
            pair = []
            for _ in range(2):
                spin = self._spin(0.1 if index == 0 else -0.1, -20, 20, "V")
                spin.setToolTip("Probe excitation vector (x, y) used to measure the response matrix.")
                pair.append(spin)
            self.probe_spins.append(pair)
            form.addRow(f"Probe {name} (x, y)", self._row(*pair))
        self.balance_at_combo = self._combo([("First sweep point", "start"), ("Custom point (x, y)", "custom")])
        form.addRow("Balance at", self.balance_at_combo)
        self.balance_x_spin = self._spin(-1.0, -210, 210, "V")
        self.balance_y_spin = self._spin(-1.0, -210, 210, "V")
        form.addRow("Balance x, y", self._row(self.balance_x_spin, self.balance_y_spin))
        self.rebalance_combo = self._combo([("Never (single balance)", "never"), ("At the start of every line", "line")])
        form.addRow("Rebalance", self.rebalance_combo)
        self.c_ref_spin = self._spin(0.0, 0.0, 1e9, "pF", "Reference capacitor. 0 reports C/C_ref ratios.")
        form.addRow("C_ref", self.c_ref_spin)
        self.btn_balance = QtWidgets.QPushButton("Balance now (present voltages)")
        self.btn_balance.setToolTip("Balance without moving electrodes; useful to check the bridge before a run.")
        self.btn_balance.clicked.connect(self.balance_now)
        form.addRow(self.btn_balance)
        self.balance_label = QtWidgets.QLabel("No balance yet.")
        self.balance_label.setWordWrap(True)
        self.balance_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.balance_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        form.addRow(self.balance_label)
        self._idle_buttons.append(self.btn_balance)
        return group

    def add_live_rows(self, form):
        self.live_lockin_label = self._mono_label("Lock-in: --")
        form.addRow(self.live_lockin_label)

    def build_result_pages(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_header())
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid, 1)
        self.c_map_plot, self.c_map_img, self.c_map_bar = self._make_map("Capacitance C/C_ref")
        self.d_map_plot, self.d_map_img, self.d_map_bar = self._make_map("Dissipation D/C_ref")
        self.c_cut_plot, self.c_cut_curve = self._make_cut("Capacitance cut", "#2563eb")
        self.d_cut_plot, self.d_cut_curve = self._make_cut("Dissipation cut", "#e11d48")
        grid.addWidget(self.c_map_plot, 0, 0)
        grid.addWidget(self.d_map_plot, 0, 1)
        grid.addWidget(self.c_cut_plot, 1, 0)
        grid.addWidget(self.d_cut_plot, 1, 1)
        self.right_tabs.addTab(page, "Capacitance")

        page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(page)
        self.x_map_plot, self.x_map_img, self.x_map_bar = self._make_map("Lock-in X")
        self.y_map_plot, self.y_map_img, self.y_map_bar = self._make_map("Lock-in Y")
        self.x_cut_plot, self.x_cut_curve = self._make_cut("X cut", "#0f766e")
        self.y_cut_plot, self.y_cut_curve = self._make_cut("Y cut", "#7c3aed")
        grid.addWidget(self.x_map_plot, 0, 0)
        grid.addWidget(self.y_map_plot, 0, 1)
        grid.addWidget(self.x_cut_plot, 1, 0)
        grid.addWidget(self.y_cut_plot, 1, 1)
        self.right_tabs.addTab(page, "Lock-in X/Y")

        self.map_items = {
            "c": (self.c_map_plot, self.c_map_img, self.c_map_bar),
            "d": (self.d_map_plot, self.d_map_img, self.d_map_bar),
            "x": (self.x_map_plot, self.x_map_img, self.x_map_bar),
            "y": (self.y_map_plot, self.y_map_img, self.y_map_bar),
        }
        self.cut_items = [("c", self.c_cut_curve, 1.0), ("d", self.d_cut_curve, 1.0),
                          ("x", self.x_cut_curve, 1e6), ("y", self.y_cut_curve, 1e6)]
        self.cut_plots = [self.c_cut_plot, self.d_cut_plot, self.x_cut_plot, self.y_cut_plot]

    def detector_preview_widgets(self):
        return [self.balance_x_spin, self.balance_y_spin, self.sample_amp_spin, self.ref_atten_spin,
                self.sample_atten_spin, self.c_ref_spin, self.balance_at_combo, self.rebalance_combo]

    # -- settings --------------------------------------------------------------
    def collect_detector_settings(self):
        return {
            "measurement_type": self.type_combo.currentData(),
            "frequency": self.freq_spin.value(),
            "sample_amplitude": self.sample_amp_spin.value(),
            "reference_amplitude": self.ref_amp_spin.value(),
            "sample_atten": self.sample_atten_spin.value(),
            "ref_atten": self.ref_atten_spin.value(),
            "max_reference_amplitude": self.max_ref_spin.value(),
            "configure_awg": self.configure_awg_check.isChecked(),
            "balance_tc": self.balance_tc_combo.currentData(),
            "measure_tc": self.measure_tc_combo.currentData(),
            "sensitivity": self.sensitivity_combo.currentData(),
            "tolerance": self.tolerance_spin.value(),
            "iterations": self.iterations_spin.value(),
            "s1": [spin.value() for spin in self.probe_spins[0]],
            "s2": [spin.value() for spin in self.probe_spins[1]],
            "balance_at": self.balance_at_combo.currentData(),
            "balance_x": self.balance_x_spin.value(),
            "balance_y": self.balance_y_spin.value(),
            "rebalance": self.rebalance_combo.currentData(),
            "c_ref": self.c_ref_spin.value() * 1e-12,
        }

    def apply_detector_values(self, s):
        self._set_combo(self.type_combo, s["measurement_type"])
        self.freq_spin.setValue(float(s["frequency"]))
        self.sample_amp_spin.setValue(float(s["sample_amplitude"]))
        self.ref_amp_spin.setValue(float(s["reference_amplitude"]))
        self.sample_atten_spin.setValue(float(s["sample_atten"]))
        self.ref_atten_spin.setValue(float(s["ref_atten"]))
        self.max_ref_spin.setValue(float(s["max_reference_amplitude"]))
        self.configure_awg_check.setChecked(bool(s["configure_awg"]))
        self._set_combo(self.balance_tc_combo, float(s["balance_tc"])) or self._set_combo(self.balance_tc_combo, 0.3)
        self._set_combo(self.measure_tc_combo, float(s["measure_tc"])) or self._set_combo(self.measure_tc_combo, 0.3)
        sensitivity = s["sensitivity"] if s["sensitivity"] in ("keep", "auto") else float(s["sensitivity"])
        self._set_combo(self.sensitivity_combo, sensitivity) or self._set_combo(self.sensitivity_combo, 1e-4)
        self.tolerance_spin.setValue(float(s["tolerance"]))
        self.iterations_spin.setValue(int(s["iterations"]))
        for spins, vector in zip(self.probe_spins, (s["s1"], s["s2"])):
            for spin, value in zip(spins, vector):
                spin.setValue(float(value))
        self._set_combo(self.balance_at_combo, s["balance_at"])
        self.balance_x_spin.setValue(float(s["balance_x"]))
        self.balance_y_spin.setValue(float(s["balance_y"]))
        self._set_combo(self.rebalance_combo, s["rebalance"])
        self.c_ref_spin.setValue(float(s["c_ref"]) * 1e12)

    # -- derived settings ------------------------------------------------------
    def balance_location(self, settings, plan):
        if settings["balance_at"] == "custom":
            return float(settings["balance_x"]), float(settings["balance_y"])
        iy, ix = plan.order[0]
        return float(plan.x[ix]), float(plan.y[iy])

    def validate_targets(self, settings, plan, electrodes):
        location = self.balance_location(settings, plan)
        balance_targets = self.coordinate_targets(settings, *location)
        electrodes.check(balance_targets["top"], balance_targets["bottom"], balance_targets["sample"],
                         context="Balance point")
        return location, balance_targets

    def validate_detector(self, settings):
        LinearBridge(lambda s: None, lambda: (0, 0), s1=settings["s1"], s2=settings["s2"],
                     tolerance=settings["tolerance"], iterations=settings["iterations"],
                     max_amplitude=settings["max_reference_amplitude"])
        ac_scale_factor(settings["ref_atten"], settings["sample_atten"], settings["sample_amplitude"])

    # -- labels and preview ------------------------------------------------------
    def on_type_changed(self):
        key = self.type_combo.currentData()
        self._set_combo(self.fixed_combo, MEASUREMENT_TYPES[key]["fixed"])
        self._update_labels()
        self.schedule_preview()

    def update_detector_labels(self, x_name, y_name, cut_name):
        spec = MEASUREMENT_TYPES[self.type_combo.currentData()]
        self.wiring_note.setText(f"Wiring for this type: AC on {spec['excited']}, bridge on {spec['measured']}. "
                                 f"Lock-in referenced to AWG CH{SAMPLE_CHANNEL}.")
        unit = "F" if self.current_c_ref() > 0 else ""
        c_name = "C" if unit else "C/C_ref"
        d_name = "D" if unit else "D/C_ref"
        self.c_map_plot.setTitle(f"{spec['symbol']}: capacitance {c_name}")
        self.d_map_plot.setTitle(f"{spec['symbol']}: dissipation {d_name}")
        # Lock-in cuts are plotted in uV: the dashboard keeps "V" axes prefix-free for gate voltages.
        for plot, label, units in ((self.c_cut_plot, c_name, unit), (self.d_cut_plot, d_name, unit),
                                   (self.x_cut_plot, "X", "µV"), (self.y_cut_plot, "Y", "µV")):
            plot.setLabel("bottom", cut_name, units="V")
            plot.setLabel("left", label, units=units)
            if units == "µV":
                plot.getAxis("left").enableAutoSIPrefix(False)
        # Colour scales with SI prefixes (e.g. fF, uV); ratios stay unitless.
        for bar, units in ((self.c_map_bar, unit), (self.d_map_bar, unit), (self.x_map_bar, "V"),
                           (self.y_map_bar, "V")):
            axis = getattr(bar, "axis", None)
            if axis is not None:
                axis.setLabel(text="", units=units)

    def preview_lines(self, settings, plan, electrodes, extra):
        location, balance_targets = extra
        return [f"Balance at ({location[0]:+.4g}, {location[1]:+.4g}): Vt {balance_targets['top']:+.4f}, "
                f"Vb {balance_targets['bottom']:+.4f}, Vs {balance_targets['sample']:+.4f} V"]

    # -- simulation ------------------------------------------------------------
    def bind_simulation(self, settings, electrodes):
        """Connect a simulated lock-in to the bridge model (no effect on hardware)."""
        lockin = self.station.components.get("lockin")
        if lockin is None or not getattr(lockin, "simulated", False):
            return
        # The simulated lock-in is shared with the Transport tab, which may have left it on a current input.
        if lockin.input_config() != "a":
            lockin.input_config("a")

        def excitation():
            awg = self.station.components.get("awg")
            return awg.excitation_state() if awg is not None else {}

        attenuation = (float(settings["sample_atten"]), float(settings["ref_atten"]))
        kind = "penetration" if settings["measurement_type"] == "penetration" else "gate"
        lockin.signal_source = SimulatedBridgeSignal(excitation, self.electrode_voltage_reader(electrodes),
                                                     lambda: attenuation, sample=SimulatedSample(kind),
                                                     noise_rms=2e-9)

    # -- sweep -----------------------------------------------------------------
    def prepare_run(self, settings, plan, electrodes, extra):
        location, balance_targets = extra
        self.bind_simulation(settings, electrodes)
        settings["balance_location"] = list(location)
        settings["balance_targets"] = balance_targets

    def reset_run_state(self):
        self.balances = []

    def sweep_task(self, settings, plan, electrodes):
        return run_sweep, (self.station, settings, plan, electrodes)

    def start_message(self, settings, plan):
        return (f"Starting {MEASUREMENT_TYPES[settings['measurement_type']]['label']}: "
                f"{len(plan.order)} points; fixed {ELECTRODE_LABELS[settings['fixed']]} at "
                f"{settings['v_fixed']:g} V.")

    def on_detector_event(self, event):
        kind = event.get("kind")
        if kind == "balance":
            self.on_balance(event["record"])
        elif kind == "awg":
            self.on_awg_states(event["states"])
        elif kind == "lockin_info":
            self.lockin_info = event["info"]

    def point_status(self, values):
        return values["lockin_status"]

    def on_balance(self, record):
        self.last_balance = record
        if record["index"] >= 0 and self.is_running:
            self.balances.append(record)
        c_ref = self.current_c_ref() if record["index"] >= 0 else self.c_ref_spin.value() * 1e-12
        lines = [
            f"{'Converged' if record['converged'] else 'NOT converged'} after {record['iterations']} "
            f"iteration(s); residual {format_si(record['residual_V'], 'V')}",
            f"C/C_ref = {record['capacitance_ratio']:.6g}   D/C_ref = {record['dissipation_ratio']:.6g}",
        ]
        if c_ref > 0:
            lines.append(f"C = {format_si(record['capacitance_ratio'] * c_ref, 'F')}   "
                         f"D = {format_si(record['dissipation_ratio'] * c_ref, 'F')}")
        lines += [
            f"c' = {record['c_coefficient']:.5g} /V   d' = {record['d_coefficient']:.5g} /V",
            f"|M| = {record['magnitude']:.4g}, arg M = {record['phase_deg']:.2f} deg",
            f"Reference {record['reference_amplitude']:.4g} Vpp, phase {record['reference_phase']:.2f} deg",
            f"Balanced at x = {record['x']:.4g} V, y = {record['y']:.4g} V",
        ]
        self.balance_label.setText("\n".join(lines))
        self.balance_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: "
                                         + ("#047857;" if record["converged"] else "#b45309;"))

    def on_awg_states(self, states):
        self.awg_states = states
        rows = []
        for number in (1, 2):
            s = states.get(number) or states.get(str(number))
            if s:
                rows.append(f"CH{number} {'ON ' if s['enabled'] else 'OFF'} {s['frequency']:.6g} Hz "
                            f"{s['amplitude']:.4g} Vpp {s['phase']:.1f} deg")
        self.awg_label.setText("\n".join(rows) or "AWG: no readback")

    # -- manual actions ----------------------------------------------------------
    def run_awg_action(self, action):
        if self.station.components.get("awg") is None:
            self._show_blocked("AWG", "The Siglent AWG is not initialized.")
            return
        self.lbl_status.setText(f"Status: AWG {action}...")
        self._start_manual("control_worker", run_awg, self.station, self.collect_settings(), action)

    def balance_now(self):
        settings = self.collect_settings()
        try:
            electrodes = self.make_electrodes(settings)
            LinearBridge(lambda s: None, lambda: (0, 0), s1=settings["s1"], s2=settings["s2"],
                         tolerance=settings["tolerance"], iterations=settings["iterations"],
                         max_amplitude=settings["max_reference_amplitude"])
            for name in ("lockin", "awg"):
                if self.station.components.get(name) is None:
                    raise RuntimeError(f"The {name} is not initialized.")
        except Exception as exc:
            self._show_blocked("Balance blocked", exc)
            return
        settings["here"] = [math.nan, math.nan]
        try:
            setpoints = electrodes.setpoints()
            n0, p0 = density_field(setpoints["top"], setpoints["bottom"], setpoints["sample"],
                                   delta=settings["delta"])
            settings["here"] = [float(n0), float(p0)]
        except Exception:
            pass
        self.bind_simulation(settings, electrodes)
        self._start_manual("control_worker", run_balance, self.station, settings, electrodes)

    # -- live readout ------------------------------------------------------------
    def on_live_detector(self, report):
        if "x" in report:
            r = math.hypot(report["x"], report["y"])
            theta = math.degrees(math.atan2(report["y"], report["x"]))
            flags = describe_status(report.get("status", 0))
            text = (f"X {format_si(report['x'], 'V')}  Y {format_si(report['y'], 'V')}\n"
                    f"R {format_si(r, 'V')}  θ {theta:+.1f} deg  sens {format_si(report['sensitivity'], 'V', 3)}")
            if flags:
                text += "\n⚠ " + ", ".join(flags)
            self.live_lockin_label.setText(text)
            self.live_lockin_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: "
                                                 + ("#b45309;" if flags else "#0f172a;"))
        elif "lockin_error" in report:
            self.live_lockin_label.setText(f"Lock-in read failed: {report['lockin_error']}")
        if "awg" in report:
            self.on_awg_states(report["awg"])

    # -- plots -----------------------------------------------------------------
    def current_c_ref(self):
        """C_ref in farads: the run's value while showing a run, else the panel value."""
        if self.run_settings is not None:
            return float(self.run_settings.get("c_ref", 0.0))
        return self.c_ref_spin.value() * 1e-12

    def display_arrays(self):
        c_ref = self.current_c_ref()
        scale = c_ref if c_ref > 0 else 1.0
        return {"c": self.data["capacitance"] * scale, "d": self.data["dissipation"] * scale,
                "x": self.data["lockin_x"], "y": self.data["lockin_y"]}

    # -- save / load -------------------------------------------------------------
    def filename_tokens(self, settings):
        symbol = MEASUREMENT_TYPES.get(settings["measurement_type"], MEASUREMENT_TYPES["custom"])["symbol"]
        return ["capacitance", axis_token(symbol)]

    def detector_file_attributes(self, settings):
        spec = MEASUREMENT_TYPES.get(settings.get("measurement_type", "custom"), MEASUREMENT_TYPES["custom"])
        return {
            "measurement_type": settings.get("measurement_type", "custom"),
            "measurement_label": spec["label"], "measurement_symbol": spec["symbol"],
            "ac_excitation_on": spec["excited"], "bridge_node": spec["measured"],
            "sample_channel": SAMPLE_CHANNEL, "reference_channel": REFERENCE_CHANNEL,
            "ac_scale": ac_scale_factor(settings.get("ref_atten", 0), settings.get("sample_atten", 0),
                                        settings.get("sample_amplitude", 1)) if settings else math.nan,
            "c_ref_F": float(settings.get("c_ref", 0.0)),
            "lockin_info": self.lockin_info,
            "awg_states": {str(k): v for k, v in self.awg_states.items()},
        }

    def write_file(self, path):
        write_capacitance_file(path, plan=self.plan, data=self.data, mask=self.measured_mask,
                               balances=self.balances, attrs=self.file_attributes(),
                               settings=self.run_settings or self.collect_settings(),
                               value_storage=self.value_storage)

    def read_file(self, path):
        return read_capacitance_file(path)

    def apply_loaded(self, loaded):
        self.balances = loaded["balances"]
        if self.balances:
            self.on_balance(self.balances[-1])


if __name__ == "__main__":
    from station_devices import create_station, ensure_simulated_devices
    app = QtWidgets.QApplication(sys.argv)
    station = create_station(os.path.join(parent_dir, "my_measurement_data.db"))
    ensure_simulated_devices(station)
    window = CapacitanceWindow(station)
    window.resize(1600, 950)
    window.show()
    sys.exit(app.exec())
