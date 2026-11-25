"""Dual-Gate Sweep runs in the measurement queue (docs/MEASUREMENT_QUEUE.md).

An entry holds the tab's sweep settings: what to sweep and how to acquire. The
gate outputs (wiring), voltage windows, current limits, source modes, Output
switches, ramp rates and the Single Spectra settings (frames per point, grating,
centre wavelength) stay as they are set on the tab. The queue never changes
them, and it checks every entry against them. A queued run puts the entry's
values in the tab's fields and calls the tab's own ``begin_sweep()``, as
pressing Begin does.
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PyQt6 import QtCore

import measurement_queue as mq
from dgs_gate_outputs import gate_output_label, is_dac_output, normalize_gate_outputs
from measurement_data_paths import dated_measurement_filename, measurement_data_dir, measurement_device_modes
from measurements import Dual_gate_sweep_gui as dgs
from measurements.queue_panel import QueueAdapter
from sweep_timing import SweepEstimate, default_point_overhead, stored_point_overhead

DGS_TAB = "dual_gate"
SWEEP_MODES = ("Hold C1, sweep C2", "Hold C2, sweep C1")

# Each number field's (lowest, highest, decimals), equal to the tab's own field
# (tests/test_measurement_queue.py checks this). A spin box would clamp or round any
# other value without a word, for example an exposure of 20 s to 10 s.
NUMBER_FIELDS = {
    "c1_start": (-210.0, 210.0, 4), "c1_end": (-210.0, 210.0, 4),
    "c2_start": (-210.0, 210.0, 4), "c2_end": (-210.0, 210.0, 4),
    "coeff_a": (-100.0, 100.0, 4), "coeff_b": (-100.0, 100.0, 4),
    "coeff_c": (-100.0, 100.0, 4), "coeff_d": (-100.0, 100.0, 4),
    "exposure_time": (0.001, 10.0, 3), "settling_time": (0.0, 10.0, 3),
    "filename_wavelength_nm": (1.0, 100000.0, 12),
}
POINT_FIELDS = {"c1_pts": (1, 1000), "c2_pts": (1, 1000)}
UNITS = {"c1_start": " V", "c1_end": " V", "c2_start": " V", "c2_end": " V",
         "exposure_time": " s", "settling_time": " s", "filename_wavelength_nm": " nm"}

# Entry setting: the Dual-Gate window's field that holds it, in the order files list them.
FIELD_WIDGETS = {
    "c1_name": "c1_name_edit", "c2_name": "c2_name_edit",
    "c1_start": "c1_start", "c1_end": "c1_end", "c1_pts": "c1_pts",
    "c2_start": "c2_start", "c2_end": "c2_end", "c2_pts": "c2_pts",
    "coeff_a": "coeff_a", "coeff_b": "coeff_b", "coeff_c": "coeff_c", "coeff_d": "coeff_d",
    "sweep_mode": "sweep_mode_combo", "serpentine": "serpentine_box", "ramp_to_zero": "ramp_zero_box",
    "exposure_time": "exposure_box", "settling_time": "settling_box",
    "filename_wavelength_nm": "filename_wavelength_box",
}
# "gate_outputs" is recorded only to check that the tab still drives the gates the same way.
SETTING_KEYS = (*FIELD_WIDGETS, "gate_outputs")
LABELS = {"c1_name": "Virtual gate 1 name", "c2_name": "Virtual gate 2 name",
          "coeff_a": "Matrix a", "coeff_b": "Matrix b", "coeff_c": "Matrix c", "coeff_d": "Matrix d",
          "sweep_mode": "Variable order", "serpentine": "Serpentine",
          "ramp_to_zero": "Ramp gates to 0 V when complete", "exposure_time": "Exposure",
          "settling_time": "Settling time", "filename_wavelength_nm": "Filename wavelength",
          "gate_outputs": "Gate outputs"}


def gate_name(settings, axis):
    name = settings.get(f"c{axis}_name") if isinstance(settings, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else f"Vc{axis}"


def field_label(key, settings):
    if key[:3] in ("c1_", "c2_") and key[3:] in ("start", "end", "pts"):
        part = {"start": "start", "end": "end", "pts": "points"}[key[3:]]
        return f"{gate_name(settings, int(key[1]))} {part}"
    return LABELS.get(key, key)


def _number(value):
    """A finite number that is not true/false, as a float; otherwise None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _fits_decimals(value, decimals):
    return abs(round(value, decimals) - value) <= 1e-9 * max(1.0, abs(value))


def field_problems(settings):
    """Every value must be one the tab's field holds exactly: type, range and decimals."""
    problems = []
    missing = [key for key in SETTING_KEYS if key not in settings]
    if missing:
        problems.append(f"Missing setting{'s' if len(missing) > 1 else ''}: {', '.join(missing)}.")
    unknown = sorted(str(key) for key in settings if key not in SETTING_KEYS)
    if unknown:
        problems.append(f"Unknown setting{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}.")
    for key, (lowest, highest, decimals) in NUMBER_FIELDS.items():
        if key not in settings:
            continue
        label, unit, value = field_label(key, settings), UNITS.get(key, ""), _number(settings[key])
        if value is None:
            problems.append(f"{label} must be a finite number, not {settings[key]!r}.")
        elif not lowest <= value <= highest:
            problems.append(f"{label}: {value:g}{unit} is outside its field's range, "
                            f"{lowest:g} to {highest:g}{unit}.")
        elif not _fits_decimals(value, decimals):
            problems.append(f"{label}: {value!r}{unit} has more than {decimals} decimals; "
                            f"the field would hold {round(value, decimals):g}{unit}.")
    for key, (lowest, highest) in POINT_FIELDS.items():
        if key not in settings:
            continue
        value = settings[key]
        if _number(value) is None or not float(value).is_integer():
            problems.append(f"{field_label(key, settings)} must be a whole number, not {value!r}.")
        elif not lowest <= value <= highest:
            problems.append(f"{field_label(key, settings)}: {int(value)} is outside {lowest} to {highest}.")
    for key in ("c1_name", "c2_name"):
        if key in settings and not isinstance(settings[key], str):
            problems.append(f"{field_label(key, settings)} must be text.")
    for key in ("serpentine", "ramp_to_zero"):
        if key in settings and not isinstance(settings[key], bool):
            problems.append(f"{field_label(key, settings)} must be true or false.")
    if "sweep_mode" in settings and settings["sweep_mode"] not in SWEEP_MODES:
        problems.append(f"Variable order must be {SWEEP_MODES[0]!r} or {SWEEP_MODES[1]!r}, "
                        f"not {settings['sweep_mode']!r}.")
    return problems


def sweep_parameters(settings, gate_outputs, gate_limits):
    """The parameters Begin passes to validate_dgs_parameters, for these settings."""
    return {
        'a': float(settings['coeff_a']), 'b': float(settings['coeff_b']),
        'c': float(settings['coeff_c']), 'd': float(settings['coeff_d']),
        'c1_start': float(settings['c1_start']), 'c1_end': float(settings['c1_end']),
        'c1_pts': int(settings['c1_pts']),
        'c2_start': float(settings['c2_start']), 'c2_end': float(settings['c2_end']),
        'c2_pts': int(settings['c2_pts']),
        'gate_outputs': dict(gate_outputs), 'gate_limits': gate_limits,
    }


def entry_problems(settings, *, gate_outputs, gate_limits, source_modes=("VOLT", "VOLT"),
                   dac_connected=True):
    """Why this entry cannot run on the tab as it is set now; [] when it can.

    These are Begin's own checks, made before the first run: the whole sweep with
    validate_dgs_parameters (the SMU's +/-210 V, the uVoltDAQ's +/-10 V, each gate's
    voltage window, a singular matrix), voltage-source mode and a connected uVoltDAQ.
    Also every value must fit its field exactly, and the tab must drive the gates
    from the outputs the entry was made with. Nothing here touches an instrument.
    """
    if not isinstance(settings, dict):
        return ["The entry has no settings."]
    problems = field_problems(settings)
    if problems:
        return problems
    try:
        recorded = normalize_gate_outputs(settings["gate_outputs"])
    except (AttributeError, TypeError, ValueError) as exc:
        return [f"Gate outputs: {exc}"]
    outputs = normalize_gate_outputs(gate_outputs)
    for axis in (1, 2):
        if recorded[axis] != outputs[axis]:
            problems.append(f"The entry was made with V{axis} on {gate_output_label(recorded[axis])}, but the "
                            f"tab drives V{axis} with {gate_output_label(outputs[axis])}. Check the wiring, "
                            "then set the tab's Output or replace the entry.")
    if tuple(source_modes) != ("VOLT", "VOLT"):
        problems.append("Gate sweeps require both DC sources in voltage mode.")
    if any(is_dac_output(outputs[axis]) for axis in (1, 2)) and not dac_connected:
        problems.append("The uVoltDAQ is not initialized. Connect it in the Device Manager.")
    try:
        dgs.validate_dgs_parameters(sweep_parameters(settings, outputs, gate_limits))
    except (RuntimeError, TypeError, ValueError) as exc:
        problems.append(str(exc))
    return problems


def entry_summary(settings):
    """One line: the two axes, the matrix if it is not the identity, order, timing, end ramp."""
    try:
        def axis(number):
            key = f"c{number}"
            start, end, points = (float(settings[f"{key}_start"]), float(settings[f"{key}_end"]),
                                  int(settings[f"{key}_pts"]))
            name = gate_name(settings, number)
            return f"{name} {start:g} V" if points == 1 else f"{name} {start:g} to {end:g} V ({points})"
        parts = [f"{axis(1)} × {axis(2)}"]
        matrix = tuple(float(settings[f"coeff_{key}"]) for key in "abcd")
        if matrix != (1.0, 0.0, 0.0, 1.0):
            parts.append("matrix ({:g}, {:g}; {:g}, {:g})".format(*matrix))
        held = gate_name(settings, 1 if settings["sweep_mode"] == SWEEP_MODES[0] else 2)
        parts.append(f"hold {held}" + (", serpentine" if settings.get("serpentine") else ""))
        parts.append(f"{float(settings['exposure_time']):g} s exposure, "
                     f"{float(settings['settling_time']):g} s settling")
        if settings.get("ramp_to_zero"):
            parts.append("then 0 V")
        return " · ".join(parts)
    except (KeyError, TypeError, ValueError):
        return "(these settings cannot be read)"


def _axis_values(settings, key):
    points, start = int(settings[f"{key}_pts"]), float(settings[f"{key}_start"])
    return np.linspace(start, float(settings[f"{key}_end"]), points) if points > 1 else np.array([start])


def sweep_targets(settings):
    """The physical (V1, V2) of every point, in the order the sweep worker visits them."""
    c1, c2 = _axis_values(settings, "c1"), _axis_values(settings, "c2")
    hold_c2 = settings["sweep_mode"] == "Hold C2, sweep C1"
    outer, inner = (c2, c1) if hold_c2 else (c1, c2)
    index = np.tile(np.arange(len(inner)), (len(outer), 1))
    if settings.get("serpentine"):
        index[1::2] = index[1::2, ::-1]
    inner_values, outer_values = inner[index].ravel(), np.repeat(outer, len(inner))
    c1_values, c2_values = (inner_values, outer_values) if hold_c2 else (outer_values, inner_values)
    v1, v2 = dgs.transform_c_to_v(c1_values, c2_values, *(float(settings[f"coeff_{key}"]) for key in "abcd"))
    return np.column_stack((np.broadcast_to(v1, c1_values.shape), np.broadcast_to(v2, c1_values.shape)))


def entry_estimate(settings, start, rates, *, frames, overhead, end_ramp=False):
    """(seconds, (V1, V2) afterwards): the Expected End model of sweep_timing, plus the end ramp."""
    targets = sweep_targets(settings)
    estimate = SweepEstimate(targets, start, rates, exposure=float(settings["exposure_time"]),
                             frames=frames, settling=float(settings["settling_time"]), overhead=overhead)
    seconds = estimate.total()
    end = tuple(float(value) for value in targets[-1])
    if end_ramp or settings.get("ramp_to_zero"):
        # Both gates move together, each at its own rate: |V| / rate per gate.
        seconds += sum(abs(value) / rate for value, rate in zip(end, rates) if rate > 0)
        end = (0.0, 0.0)
    return seconds, end


def frames_per_point(data_root):
    """Frames per point, which Begin reads from the Single Spectra settings in value_storage."""
    try:
        with open(os.path.join(data_root, "value_storage"), "r") as f:
            stored = json.load(f)
        return max(1, int(stored.get("single_spectra", {}).get("num_spectra", stored.get("num_spectra", 1))))
    except Exception:
        return 1


class DualGateQueueAdapter(QueueAdapter):
    """The Dual-Gate Sweep tab in the queue. Begin's own path does the measuring."""
    key = DGS_TAB
    title = "Dual-Gate Sweep"

    def __init__(self, window, tab_index=1, parent=None):
        super().__init__(parent)
        self.window = window
        self.tab_index = tab_index
        self._running = False
        window.run_ended.connect(self._on_window_run_ended)

    def data_root(self):
        """The Dual-Gate module's folder for data/ and value_storage (tests redirect it)."""
        return dgs.parent_dir

    # --- Settings ---
    def capture(self):
        w = self.window
        # Text typed in a number field counts once it is read, as Enter or leaving the field does.
        for key in (*NUMBER_FIELDS, *POINT_FIELDS):
            getattr(w, FIELD_WIDGETS[key]).interpretText()
        settings = {}
        for key, name in FIELD_WIDGETS.items():
            widget = getattr(w, name)
            if key in ("c1_name", "c2_name"):
                settings[key] = widget.text()
            elif key in POINT_FIELDS:
                settings[key] = int(widget.value())
            elif key in NUMBER_FIELDS:
                settings[key] = float(widget.value())
            elif key == "sweep_mode":
                settings[key] = widget.currentText()
            else:
                settings[key] = bool(widget.isChecked())
        settings["gate_outputs"] = {str(axis): w.gate_outputs[axis] for axis in (1, 2)}
        return settings

    def capture_problem(self):
        w = self.window
        if getattr(w, "plotter_mode", False):
            return "The Dual-Gate tab shows a saved file."
        if w.is_running or (w.ramp_worker is not None and w.ramp_worker.isRunning()):
            return "The Dual-Gate tab is measuring or ramping. Use the queue while it is idle."
        return None

    def problems(self, settings):
        w = self.window
        return entry_problems(settings, gate_outputs=w.gate_outputs, gate_limits=w.gate_limits(),
                              source_modes=(w.ch1_mode.currentText(), w.ch2_mode.currentText()),
                              dac_connected=getattr(w.station, "dac", None) is not None)

    def summary(self, settings):
        return entry_summary(settings)

    def start_voltages(self):
        return (float(self.window.ch1_setpoint.value()), float(self.window.ch2_setpoint.value()))

    def estimate(self, settings, start, end_ramp=False):
        try:
            root = self.data_root()
            frames = frames_per_point(root)
            stored = stored_point_overhead(os.path.join(root, "value_storage"))
            overhead = default_point_overhead(frames) if stored is None else stored
            rates = (float(self.window.ch1_ramp_rate.value()), float(self.window.ch2_ramp_rate.value()))
            return entry_estimate(settings, start, rates, frames=frames, overhead=overhead, end_ramp=end_ramp)
        except Exception:
            return None, start

    def show(self, settings):
        """Put the settings in the fields as if typed, then save them as the fields' own handlers do.

        Only these fields change: no ramp rate, current limit, output, source mode or window,
        and no instrument command.
        """
        w = self.window
        widgets = [getattr(w, name) for name in FIELD_WIDGETS.values()]
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        try:
            for key, name in FIELD_WIDGETS.items():
                widget, value = getattr(w, name), settings[key]
                if key in ("c1_name", "c2_name"):
                    widget.setText(value)
                elif key in POINT_FIELDS:
                    widget.setValue(int(value))
                elif key in NUMBER_FIELDS:
                    widget.setValue(float(value))
                elif key == "sweep_mode":
                    widget.setCurrentText(value)
                else:
                    widget.setChecked(bool(value))
        finally:
            for blocker in blockers:
                blocker.unblock()
        # What on_timing_params_changed and on_sweep_range_changed do after an edit.
        w.stored_settings["exposure_time"] = w.exposure_box.value()
        w.stored_settings["settling_time"] = w.settling_box.value()
        w.update_step_calculations()
        w.update_variable_names()
        w.update_coordinates_from_ui()
        w.reset_plot_scaling(reset_spectrum=False)
        w.save_storage()

    # --- Running ---
    def idle(self):
        w = self.window
        return not (w.is_running or w.is_paused
                    or (w.sweep_worker is not None and w.sweep_worker.isRunning())
                    or (w.ramp_worker is not None and w.ramp_worker.isRunning()))

    def start(self, settings, info, end_ramp=False):
        w = self.window
        if not self.idle():
            return "The Dual-Gate tab is busy."
        settings = dict(settings)
        if end_ramp:
            settings["ramp_to_zero"] = True
        self.show(settings)
        w.queued_run = dict(info)
        w.begin_blocked_reason = None
        try:
            w.begin_sweep()
        except Exception as exc:
            w.queued_run = None
            return f"{type(exc).__name__}: {exc}"
        if not w.is_running:
            w.queued_run = None
            return w.begin_blocked_reason or w.status_lbl.text() or "Begin did not start a run."
        self._running = True
        return None

    def stop_now(self):
        self.window.reset_sweep()

    def hold(self, held):
        w = self.window
        if held:
            w.set_inputs_enabled(False)
            w.begin_btn.setEnabled(False)
        elif not w.is_running:
            w.set_inputs_enabled(True)
            w.update_det_label()

    def progress(self):
        w = self.window
        return (int(w.progress_bar.value()), bool(w.is_paused)) if w.is_running else None

    def seconds_left(self):
        w = self.window
        return getattr(w, "sweep_seconds_left", None) if w.is_running else None

    def envelope(self):
        w = self.window
        smu = getattr(w.station, "smu", None)
        current_limits = {}
        for axis in (1, 2):
            channel = None if w.gate_is_dac(axis) else getattr(smu, f"ch{axis}", None)
            current_limits[str(axis)] = getattr(channel, "current_compliance_target", None)
        return {
            "gate_outputs": {str(axis): w.gate_outputs[axis] for axis in (1, 2)},
            "gate_limits_V": {str(axis): list(window) for axis, window in w.gate_limits().items()},
            "ramp_rates_V_per_s": {"1": w.ch1_ramp_rate.value(), "2": w.ch2_ramp_rate.value()},
            "current_limits_A": current_limits,
            "source_modes": {"1": w.ch1_mode.currentText(), "2": w.ch2_mode.currentText()},
            "frames_per_point": frames_per_point(self.data_root()),
            "device_modes": measurement_device_modes(w.station),
        }

    def save_named_copy(self, complete):
        """The run under the name Save Dataset proposes, plus its start time, in data/<date>/."""
        w = self.window
        if not np.any(w.data_measured_mask):
            return None
        started = getattr(w, "measurement_started_at", None)
        directory = Path(measurement_data_dir(self.data_root(), started))
        name = dated_measurement_filename(w.default_sweep_filename(), started)
        base = name[:-3] if name.endswith(".h5") else name
        stamp = started.strftime("%H%M%S") if started is not None else time.strftime("%H%M%S")
        path = mq.unique_path(directory / f"{base.rstrip('_')}_{stamp}{'' if complete else '_interrupted'}.h5")
        return w.save_sweep_data(file_path=str(path))

    def _result(self):
        w = self.window
        targets = getattr(w, "last_gate_targets", None)
        return {"file": getattr(w, "last_autosave_path", None),
                "timing": getattr(w, "last_timing_text", None),
                "gates_after_V": list(targets) if targets is not None else None}

    def _on_window_run_ended(self, outcome, message):
        if not self._running:
            return  # A run started with the tab's own Begin.
        self._running = False
        w = self.window
        result = self._result()
        try:
            result["copy"] = self.save_named_copy(complete=outcome == "finished")
        except Exception as exc:  # The runner must still hear that the run ended.
            print(f"Measurement queue: the named copy was not saved ({exc}); the autosave holds the data.")
        finally:
            w.queued_run = None
        self.run_ended.emit(outcome, message, result)

    def abort(self, reason):
        """The dashboard closes during the run: autosave its data, as Stop would."""
        if not self._running:
            return {}
        self._running = False
        w = self.window
        try:
            w.trigger_autosave(was_interrupted=True)
            return self._result()
        finally:
            w.queued_run = None
