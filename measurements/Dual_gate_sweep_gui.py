import os
import sys
import json
import time
import math
import re
import h5py
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
try:
    from .utils import EnhancedViewBox, calculate_expected_duration, configure_plot_axes, update_current_slice
except ImportError:
    import sys
    import os
    _measurements_dir = os.path.dirname(os.path.abspath(__file__))
    if _measurements_dir not in sys.path:
        sys.path.insert(0, _measurements_dir)
    from utils import EnhancedViewBox, calculate_expected_duration, configure_plot_axes, update_current_slice

# Ensure parent directory is in path for importing station_setup
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from station_setup import initialize_station, simulated_devices
from gate_safety import (GATE_CURRENT_COMPLIANCE_A, GATE_MAX_ABS_VOLTAGE_V,
                         GateRampCancelled, validate_gate_targets, validate_gate_voltage,
                         validate_current_compliance)
from gui_safety import install_parameter_wheel_guard
from measurement_data_paths import dated_measurement_filename, dated_measurement_path
from measurement_data_paths import measurement_data_dir, measurement_start_text, new_measurement_start, measurement_device_modes
from measurement_acquisition import acquire_sweep_spectrum

# Setup pyqtgraph themes (Light Theme)
pg.setConfigOption('background', '#ffffff')
pg.setConfigOption('foreground', '#1e293b')

class ScientificPointSpinBox(QtWidgets.QSpinBox):
    """A spin box that accepts integers in scientific notation like 1e3."""

    _sci_pattern = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?$')

    def valueFromText(self, text):
        try:
            if text.strip() in ("", "-", "+"):
                return int(max(self.minimum(), min(self.maximum(), 0)))
            value = float(text)
            if not math.isfinite(value):
                raise ValueError
            if not value.is_integer():
                raise ValueError
            return int(value)
        except Exception:
            return int(max(self.minimum(), min(self.maximum(), 0)))

    def validate(self, text, pos):
        if text.strip() == "":
            return QtGui.QValidator.State.Intermediate, text, pos
        if text.strip() in ("-", "+"):
            return QtGui.QValidator.State.Intermediate, text, pos
        if not self._sci_pattern.match(text):
            return QtGui.QValidator.State.Invalid, text, pos
        if len(text) > 0 and text[-1] in 'eE+-':
            return QtGui.QValidator.State.Intermediate, text, pos
        try:
            value = float(text)
            if not math.isfinite(value):
                return QtGui.QValidator.State.Invalid, text, pos
            if not value.is_integer():
                return QtGui.QValidator.State.Invalid, text, pos
            if self.minimum() <= int(value) <= self.maximum():
                return QtGui.QValidator.State.Acceptable, text, pos
            return QtGui.QValidator.State.Intermediate, text, pos
        except ValueError:
            return QtGui.QValidator.State.Invalid, text, pos


_OriginalQDoubleSpinBox = QtWidgets.QDoubleSpinBox

class ScientificDoubleSpinBox(_OriginalQDoubleSpinBox):
    """A double spin box that displays and accepts float values in scientific notation with high precision."""
    
    _sci_pattern = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?$')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDecimals(12)  # Maintain high precision internally
        self.setRange(-1e30, 1e30) # Default large range, can be overridden

    def textFromValue(self, value):
        if value == 0:
            return "0"
        abs_val = abs(value)
        if abs_val < 1e-3 or abs_val >= 1e5:
            s = f"{value:.8e}"
            parts = s.split('e')
            if len(parts) == 2:
                mantissa, exponent = parts
                if '.' in mantissa:
                    mantissa = mantissa.rstrip('0')
                    if mantissa.endswith('.'):
                        mantissa += '0'
                exp_sign = exponent[0]
                exp_val = exponent[1:].lstrip('0')
                if not exp_val:
                    exp_val = '0'
                exponent = exp_sign + exp_val
                s = f"{mantissa}e{exponent}"
            return s
        else:
            s = f"{value:.8f}"
            if '.' in s:
                s = s.rstrip('0')
                if s.endswith('.'):
                    s += '0'
            return s

    def valueFromText(self, text):
        try:
            if text.strip() in ("", "-", "+", "-.", "+.", "."):
                return max(self.minimum(), min(self.maximum(), 0.0))
            value = float(text)
            if not math.isfinite(value):
                raise ValueError
            return value
        except Exception:
            return max(self.minimum(), min(self.maximum(), 0.0))

    def validate(self, text, pos):
        if text.strip() == "":
            return QtGui.QValidator.State.Intermediate, text, pos
        if text.strip() in ("-", "+", "-.", "+.", "."):
            return QtGui.QValidator.State.Intermediate, text, pos
        if not self._sci_pattern.match(text):
            return QtGui.QValidator.State.Invalid, text, pos
        if len(text) > 0 and text[-1] in 'eE+-':
            return QtGui.QValidator.State.Intermediate, text, pos
        try:
            value = float(text)
            if not math.isfinite(value):
                return QtGui.QValidator.State.Invalid, text, pos
            if self.minimum() <= value <= self.maximum():
                return QtGui.QValidator.State.Acceptable, text, pos
            return QtGui.QValidator.State.Intermediate, text, pos
        except ValueError:
            return QtGui.QValidator.State.Invalid, text, pos

# Override the default QDoubleSpinBox class dynamically
QtWidgets.QDoubleSpinBox = ScientificDoubleSpinBox


def transform_c_to_v(vg1, vg2, a, b, c, d):
    """
    Convert virtual gate coordinates (vg1, vg2) to physical channel voltages (ch1, ch2).

    The matrix M is defined by the user as:
      VG1 = a * ch1 + b * ch2
      VG2 = c * ch1 + d * ch2

    To sweep at a given (vg1, vg2) we need the inverse:
      ch1 = ( d * vg1 - b * vg2) / det
      ch2 = (-c * vg1 + a * vg2) / det
    where det = a*d - b*c.

    Raises ValueError if the matrix is singular (det ≈ 0).
    """
    if not np.isfinite([a, b, c, d]).all():
        raise ValueError('Gate transformation coefficients must be finite.')
    det = a * d - b * c
    if abs(det) < 1e-9:
        raise ValueError(
            f"Gate transformation matrix is singular (det={det:.2e}). "
            "Adjust coefficients so that a·d − b·c ≠ 0."
        )
    ch1 = ( d * vg1 - b * vg2) / det
    ch2 = (-c * vg1 + a * vg2) / det
    return ch1, ch2


def finite_plot_levels(values):
    """Missing saved-data pixels must not produce NaN axes or colour limits."""
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    return (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)


def validate_dgs_parameters(params):
    """Linear transform extrema occur at corners; preflight before any motion."""
    x = [params['c1_start'], params['c1_end'] if params['c1_pts'] > 1 else params['c1_start']]
    y = [params['c2_start'], params['c2_end'] if params['c2_pts'] > 1 else params['c2_start']]
    xx, yy = np.meshgrid(x, y)
    validate_gate_targets(*transform_c_to_v(xx, yy, *(params[k] for k in ('a', 'b', 'c', 'd'))))


def require_voltage_source_modes(smu):
    for axis in (1, 2):
        channel = getattr(smu, f'ch{axis}')
        if channel.source_mode() != 'VOLT':
            raise RuntimeError(f'Channel {axis} is not in voltage-source mode. Gate sweep blocked.')
        validate_gate_voltage(channel.volt(), axis)


def read_compliance_values(smu):
    readings = {}
    for axis in (1, 2):
        try:
            channel = getattr(smu, f'ch{axis}')
            mode = channel.source_mode()
            value = channel.volt_limit() if mode == 'CURR' else channel.curr_limit()
            readings[axis] = (mode, float(value))
        except Exception:
            readings[axis] = None
    return readings


class SingleRampWorker(QtCore.QThread):
    """
    Lightweight background worker to ramp a single channel to a voltage or current setpoint
    without freezing the main GUI thread.
    """
    progress = QtCore.pyqtSignal(str, float, float)
    output_updated = QtCore.pyqtSignal(str, bool)
    compliance_updated = QtCore.pyqtSignal(int, float)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, channel, target_val, param_name="volt", ch_name="Channel 1", parent=None):
        super().__init__(parent)
        self.channel = channel
        self.target_val = target_val
        self.param_name = param_name
        self.ch_name = ch_name
        self._stop_requested = False
        self.channel.ramp_stop_requested = False

    def stop(self):
        self._stop_requested = True
        self.channel.ramp_stop_requested = True

    def run(self):
        # A failed manual operation is reported; do not replay an uncertain write.
        max_retries = 1
        
        def progress_callback(val):
            curr_val = 0.0
            volt_val = val
            try:
                if self.param_name == "curr":
                    curr_val = val
                    try:
                        volt_val = self.channel.volt()
                    except Exception:
                        volt_val = 0.0
                else:
                    try:
                        curr_val = self.channel.measure_curr()
                    except Exception:
                        curr_val = 0.0
            except Exception:
                pass
            self.progress.emit(self.ch_name, volt_val, curr_val)
            
        for attempt in range(1, max_retries + 1):
            try:
                if self._stop_requested:
                    raise GateRampCancelled("Manual ramp stopped")
                if self.param_name == "output":
                    self.channel.output(bool(self.target_val))
                    self.output_updated.emit(self.ch_name, bool(self.channel.output()))
                elif self.param_name == 'curr_limit':
                    self.channel.curr_limit(self.target_val)
                    self.compliance_updated.emit(self.channel.channel_num, float(self.target_val))
                elif self.param_name == "curr":
                    self.channel.set_r.curr(self.target_val, callback=progress_callback)
                else:
                    self.channel.set_r.volt(self.target_val, callback=progress_callback)
                break
            except GateRampCancelled:
                self.report_output_state()
                self.finished.emit()
                return
            except Exception as e:
                if attempt == max_retries or self.param_name == "output":
                    self.report_output_state()
                    self.error.emit(str(e))
                    return
                try:
                    self.channel.parent.visa_handle.clear()
                except Exception:
                    pass
                time.sleep(0.1)
        try:
            # Release front-panel remote lock
            self.channel.parent.return_to_local()
        except Exception:
            pass
        self.finished.emit()

    def report_output_state(self):
        if self.param_name == 'output':
            try:
                self.output_updated.emit(self.ch_name, bool(self.channel.output()))
            except Exception:
                pass


class SweepWorker(QtCore.QThread):
    """
    Background worker thread that runs the gate sweep measurement.
    Communicates back to the main thread via Qt signals.
    """
    point_measured = QtCore.pyqtSignal(int, int, float, float, float, float, np.ndarray, float, float, float, float, float, np.ndarray)
    progress_updated = QtCore.pyqtSignal(int, int)
    status_updated = QtCore.pyqtSignal(str)
    voltage_updated = QtCore.pyqtSignal(float, float, float, float)
    finished = QtCore.pyqtSignal()
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, station, params, parent=None):
        super().__init__(parent)
        self.station = station
        self.params = params
        self._is_running = True
        self._is_paused = False
        self.mutex = QtCore.QMutex()
        self.pause_cond = QtCore.QWaitCondition()
        self._control = self
        self._cancel_generation = 0

    def pause(self):
        self.mutex.lock()
        self._is_paused = True
        self._cancel_generation += 1
        try:
            self.station.smu.ch1.ramp_stop_requested = True
        except Exception:
            pass
        try:
            self.station.smu.ch2.ramp_stop_requested = True
        except Exception:
            pass
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self._is_paused = False
        self.pause_cond.wakeAll()
        self.mutex.unlock()

    def stop(self):
        self.mutex.lock()
        self._is_running = False
        self._is_paused = False
        self._cancel_generation += 1
        try:
            self.station.smu.ch1.ramp_stop_requested = True
        except Exception:
            pass
        try:
            self.station.smu.ch2.ramp_stop_requested = True
        except Exception:
            pass
        self.pause_cond.wakeAll()
        self.mutex.unlock()

    def prepare_gate_ramp(self):
        """Clear an earlier pause only while holding the stop/pause mutex."""
        control = self._control
        control.mutex.lock()
        try:
            while control._is_paused:
                control.status_updated.emit('Status: Paused')
                control.pause_cond.wait(control.mutex)
            if not control._is_running:
                return None
            self.station.smu.ch1.ramp_stop_requested = False
            self.station.smu.ch2.ramp_stop_requested = False
            return control._cancel_generation
        finally:
            control.mutex.unlock()

    def check_pause_and_stop(self):
        self.mutex.lock()
        while self._is_paused:
            self.status_updated.emit("Status: Paused")
            self.pause_cond.wait(self.mutex)
        if not self._is_running:
            self.mutex.unlock()
            try:
                self.station.smu.return_to_local()
            except Exception:
                pass
            return False
        self.mutex.unlock()
        return True

    def run(self):
        smu = getattr(self.station, "smu", None)
        try:
            try:
                self._do_run()
            finally:
                if smu is not None:
                    try:
                        smu.return_to_local()
                    except Exception:
                        pass
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _do_run(self):
        try:
            validate_dgs_parameters(self.params)
            smu = self.station.smu
            require_voltage_source_modes(smu)
            smu.enforce_current_compliance()
            camera = self.station.camera
            mono = self.station.monochromator

            a = self.params['a']
            b = self.params['b']
            c = self.params['c']
            d = self.params['d']

            c1_start = self.params['c1_start']
            c1_end = self.params['c1_end']
            c1_pts = self.params['c1_pts']

            c2_start = self.params['c2_start']
            c2_end = self.params['c2_end']
            c2_pts = self.params['c2_pts']

            sweep_mode = self.params['sweep_mode']
            serpentine = self.params['serpentine']
            exposure_time = self.params['exposure_time']
            settling_time = self.params['settling_time']
            num_pixels = self.params.get('num_pixels', 1340)

            # --- Read multi-exposure settings from value_storage ---
            num_spectra   = 1
            accum_method  = "Sum"
            do_average    = False
            storage_path  = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "value_storage"
            )
            if os.path.exists(storage_path):
                try:
                    with open(storage_path, "r") as f:
                        vs = json.load(f)
                    single_data = vs.get("single_spectra", {})
                    num_spectra  = int(single_data.get("num_spectra",  vs.get("num_spectra", 1)))
                    accum_method = str(single_data.get("accum_method", vs.get("accum_method", "Sum")))
                    do_average   = str(single_data.get("average",      vs.get("average",      "No"))) == "Yes"
                except Exception as e:
                    print(f"SweepWorker: could not read value_storage ({e}), using defaults")

            # Use the same frame count as the preallocated data array.
            num_spectra = int(self.params.get("num_spectra", num_spectra))

            # Set CCD exposure and native frames count directly on the remote camera
            camera.exposure_time(exposure_time)
            try:
                camera.num_frames(num_spectra)
            except Exception as e:
                print(f"SweepWorker: Error setting remote num_frames: {e}")

            # Generate sweep coordinate arrays (always full 2D grid)
            c1_arr = np.linspace(c1_start, c1_end, c1_pts) if c1_pts > 1 else np.array([c1_start])
            c2_arr = np.linspace(c2_start, c2_end, c2_pts) if c2_pts > 1 else np.array([c2_start])

            total_points = len(c1_arr) * len(c2_arr)
            point_count = 0

            # Outer loop holds one gate fixed while the inner loop sweeps the other.
            if sweep_mode == "Hold C2, sweep C1":
                outer_arr = c2_arr
                inner_arr = c1_arr
                is_c2_outer = True
            else:
                outer_arr = c1_arr
                inner_arr = c2_arr
                is_c2_outer = False

            for idx_outer, val_outer in enumerate(outer_arr):
                # Handle serpentine sweeping direction
                if serpentine and (idx_outer % 2 == 1):
                    inner_indices = list(range(len(inner_arr) - 1, -1, -1))
                else:
                    inner_indices = list(range(len(inner_arr)))

                for idx_inner in inner_indices:
                    # Thread pause / stop safety check
                    if not self.check_pause_and_stop():
                        return

                    val_inner = inner_arr[idx_inner]

                    # Assign variables based on mode
                    if is_c2_outer:
                        c2 = val_outer
                        c1 = val_inner
                        i = idx_inner
                        j = idx_outer
                    else:
                        c1 = val_outer
                        c2 = val_inner
                        i = idx_outer
                        j = idx_inner

                    # 1. Transform to physical voltages
                    v1, v2 = transform_c_to_v(c1, c2, a, b, c, d)

                    # 2. Ramp physical voltages (blocking hardware set call, with interrupt retry loop)
                    self.voltage_updated.emit(c1, c2, v1, v2)
                    while True:
                        if not self.check_pause_and_stop():
                            return
                        
                        generation = self.prepare_gate_ramp()
                        if generation is None:
                            return
                        try:
                            smu.ch1.set_r.volt(v1)
                            if not self.check_pause_and_stop():
                                return
                            smu.ch2.set_r.volt(v2)
                        except GateRampCancelled:
                            # Resume retries this entire point, with a fresh ramp
                            # readback, before starting its full settling delay.
                            if not self._control._is_running:
                                return
                            if self._control._cancel_generation == generation:
                                raise
                            continue
                        if getattr(smu.ch1, 'ramp_stop_requested', False) or getattr(smu.ch2, 'ramp_stop_requested', False):
                            if self._control._cancel_generation == generation:
                                raise GateRampCancelled('Gate output cancelled during sweep')
                            continue
                        break

                    # 3. Wait settling time
                    if settling_time > 0:
                        sleep_end = time.time() + settling_time
                        while time.time() < sleep_end:
                            if not self.check_pause_and_stop():
                                return
                            time.sleep(0.05)

                    if not self.check_pause_and_stop():
                        return

                    # 4. Read actual SMU voltage and current (before spectrum, so values
                    #    reflect the settled state, not the state after a long exposure)
                    try:
                        ch1_on = bool(smu.ch1.output())
                        curr_ch1 = smu.ch1.measure_curr() if ch1_on else 0.0
                        meas_v1 = smu.ch1.measure_volt() if ch1_on else 0.0
                    except Exception:
                        curr_ch1 = 0.0
                        meas_v1 = 0.0

                    try:
                        ch2_on = bool(smu.ch2.output())
                        curr_ch2 = smu.ch2.measure_curr() if ch2_on else 0.0
                        meas_v2 = smu.ch2.measure_volt() if ch2_on else 0.0
                    except Exception:
                        curr_ch2 = 0.0
                        meas_v2 = 0.0

                    meas_b = 0.0
                    magnet = getattr(self.station, 'magnet', None)
                    if magnet is not None:
                        try:
                            meas_b = magnet.output_field()
                        except Exception:
                            try:
                                meas_b = magnet.field()
                            except Exception:
                                pass

                    # 5. Acquire spectrum
                    if not self.check_pause_and_stop():
                        return
                    try:
                        spectrum, raw_frames = acquire_sweep_spectrum(
                            camera, pixels=num_pixels, frames=num_spectra,
                            average=do_average or accum_method == "Average",
                            wavelengths=self.params.get('wl_axis'),
                        )
                    except Exception:
                        if not self._control._is_running:
                            return
                        raise

                    if not self.check_pause_and_stop():
                        return

                    # 6. Emit results
                    point_count += 1
                    self.point_measured.emit(i, j, c1, c2, v1, v2, spectrum, curr_ch1, curr_ch2, meas_v1, meas_v2, meas_b, raw_frames)
                    self.progress_updated.emit(point_count, total_points)

            self.finished.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))


class ViewControlPopup(QtWidgets.QFrame):
    """Dropdown style floating popup widget containing controls for autoscaling plots."""
    def __init__(self, parent_window, has_grid=True):
        super().__init__(parent_window, QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint)
        self.parent_window = parent_window
        self.has_grid = has_grid
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.setFixedWidth(160)
        self.hide()
        self._init_ui()
        
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
            }
            QLabel {
                font-weight: bold;
                color: #1e293b;
                border: none;
                background-color: transparent;
            }
        """)
        
        # 1. 2D Layout Selector Grid
        if self.has_grid:
            layout.addWidget(QtWidgets.QLabel("Target Plots:"))
            self.grid_layout = QtWidgets.QGridLayout()
            self.grid_layout.setSpacing(4)
            
            self.btn_grid_A = QtWidgets.QPushButton("")
            self.btn_grid_B = QtWidgets.QPushButton("")
            self.btn_grid_C = QtWidgets.QPushButton("")
            self.btn_grid_D = QtWidgets.QPushButton("")
            self.btn_grid_AC = QtWidgets.QPushButton("")
            
            self.btn_grid_A.setToolTip("Top-Left: 2D Gate Map (Integrated Counts)")
            self.btn_grid_B.setToolTip("Top-Right: Vertical Slice spectra")
            self.btn_grid_C.setToolTip("Bottom-Left: Horizontal Slice spectra")
            self.btn_grid_D.setToolTip("Bottom-Right: CCD Spectrum")
            self.btn_grid_AC.setToolTip("Left: Top Gate Current Heatmap")
            
            btn_style = """
                QPushButton {
                    background-color: #f8fafc;
                    border: 1px solid #cbd5e1;
                    font-weight: bold;
                    border-radius: 4px;
                    min-width: 32px;
                    min-height: 32px;
                    color: #1e293b;
                }
                QPushButton:hover {
                    background-color: #f1f5f9;
                }
                QPushButton:checked {
                    background-color: #3b82f6;
                    color: white;
                    border-color: #2563eb;
                }
            """
            for btn in (self.btn_grid_A, self.btn_grid_B, self.btn_grid_C, self.btn_grid_D, self.btn_grid_AC):
                btn.setCheckable(True)
                btn.setChecked(True)
                btn.setStyleSheet(btn_style)
            
            # Initial setup: assume tab 0 (4-plot layout)
            self.grid_layout.addWidget(self.btn_grid_A, 0, 0)
            self.grid_layout.addWidget(self.btn_grid_B, 0, 1)
            self.grid_layout.addWidget(self.btn_grid_C, 1, 0)
            self.grid_layout.addWidget(self.btn_grid_D, 1, 1)
            self.btn_grid_AC.hide()
            
            layout.addLayout(self.grid_layout)

        
        # 2. Axis Selection Buttons
        layout.addWidget(QtWidgets.QLabel("Autoscale Axis:"))
        axis_layout = QtWidgets.QHBoxLayout()
        axis_layout.setSpacing(2)
        
        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.setExclusive(True)
        
        self.btn_axis_both = QtWidgets.QToolButton()
        self.btn_axis_both.setText("✥")
        self.btn_axis_both.setToolTip("Autoscale both axes")
        
        self.btn_axis_x = QtWidgets.QToolButton()
        self.btn_axis_x.setText("↔")
        self.btn_axis_x.setToolTip("Autoscale X axis only")
        
        self.btn_axis_y = QtWidgets.QToolButton()
        self.btn_axis_y.setText("↕")
        self.btn_axis_y.setToolTip("Autoscale Y axis only")
        
        axis_btn_style = """
            QToolButton {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                font-weight: bold;
                border-radius: 4px;
                min-width: 30px;
                min-height: 30px;
                font-size: 14px;
                color: #1e293b;
            }
            QToolButton:hover {
                background-color: #f1f5f9;
            }
            QToolButton:checked {
                background-color: #3b82f6;
                color: white;
                border-color: #2563eb;
            }
        """
        
        for btn in (self.btn_axis_both, self.btn_axis_x, self.btn_axis_y):
            btn.setCheckable(True)
            btn.setStyleSheet(axis_btn_style)
            self.btn_group.addButton(btn)
            axis_layout.addWidget(btn)
            
        self.btn_axis_both.setChecked(True)
        layout.addLayout(axis_layout)
        
        # 3. Plot Source Selector
        layout.addWidget(QtWidgets.QLabel("Plot Source:"))
        self.combo_source = QtWidgets.QComboBox()
        self.combo_source.addItems(["Signal", "Signal - Background", "(S-R)/(R-B)"])
        self.combo_source.setStyleSheet("""
            QComboBox {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px;
                color: #1e293b;
            }
            QComboBox:hover {
                background-color: #f1f5f9;
            }
        """)
        self.combo_source.currentIndexChanged.connect(self.parent_window.on_plot_source_changed)
        layout.addWidget(self.combo_source)
        
        # 4. Autoscale Now Button
        self.btn_scale_now = QtWidgets.QPushButton("Autoscale Now")
        self.btn_scale_now.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: 1px solid #4f46e5;
                padding: 6px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        self.btn_scale_now.clicked.connect(self.parent_window.perform_autoscale)
        layout.addWidget(self.btn_scale_now)

    def get_source_option(self):
        return self.combo_source.currentText()

    def get_axis_mode(self):
        if self.btn_axis_both.isChecked():
            return "Both"
        elif self.btn_axis_x.isChecked():
            return "X"
        elif self.btn_axis_y.isChecked():
            return "Y"
        else:
            return "None"

    def update_grid_layout(self, tab_idx):
        if not self.has_grid:
            return
        # Remove all widgets
        for btn in (self.btn_grid_A, self.btn_grid_B, self.btn_grid_C, self.btn_grid_D, self.btn_grid_AC):
            self.grid_layout.removeWidget(btn)
            btn.hide()
            
        if tab_idx == 0:
            # 4-plot layout: A (0,0), B (0,1), C (1,0), D (1,1)
            self.btn_grid_A.setToolTip("Top-Left: 2D Gate Map (Integrated Counts)")
            self.btn_grid_B.setToolTip("Top-Right: Vertical Slice spectra")
            self.btn_grid_C.setToolTip("Bottom-Left: Horizontal Slice spectra")
            self.btn_grid_D.setToolTip("Bottom-Right: CCD Spectrum")
            
            self.btn_grid_A.show()
            self.btn_grid_B.show()
            self.btn_grid_C.show()
            self.btn_grid_D.show()
            self.grid_layout.addWidget(self.btn_grid_A, 0, 0)
            self.grid_layout.addWidget(self.btn_grid_B, 0, 1)
            self.grid_layout.addWidget(self.btn_grid_C, 1, 0)
            self.grid_layout.addWidget(self.btn_grid_D, 1, 1)
        elif tab_idx == 1:
            # 3-plot layout (Current tabs): AC on left (spanning 2 rows), B on top-right, D on bottom-right
            self.btn_grid_AC.setToolTip("Left: Top Gate Current Heatmap")
            self.btn_grid_B.setToolTip("Top-Right: Slice at fixed C2")
            self.btn_grid_D.setToolTip("Bottom-Right: Slice at fixed C1")
            
            self.btn_grid_AC.show()
            self.btn_grid_B.show()
            self.btn_grid_D.show()
            self.grid_layout.addWidget(self.btn_grid_AC, 0, 0, 2, 1)
            self.grid_layout.addWidget(self.btn_grid_B, 0, 1)
            self.grid_layout.addWidget(self.btn_grid_D, 1, 1)
        elif tab_idx == 2:
            # 3-plot layout (Current tabs): AC on left (spanning 2 rows), B on top-right, D on bottom-right
            self.btn_grid_AC.setToolTip("Left: Bottom Gate Current Heatmap")
            self.btn_grid_B.setToolTip("Top-Right: Slice at fixed C2")
            self.btn_grid_D.setToolTip("Bottom-Right: Slice at fixed C1")
            
            self.btn_grid_AC.show()
            self.btn_grid_B.show()
            self.btn_grid_D.show()
            self.grid_layout.addWidget(self.btn_grid_AC, 0, 0, 2, 1)
            self.grid_layout.addWidget(self.btn_grid_B, 0, 1)
            self.grid_layout.addWidget(self.btn_grid_D, 1, 1)

class LiveStatusWorker(QtCore.QThread):
    """
    Background worker thread to continuously poll B, V1, V2 at 5 Hz.
    """
    status_updated = QtCore.pyqtSignal(float, float, float)
    compliance_updated = QtCore.pyqtSignal(object)

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self._stop_requested = False
        self._is_paused = False
        self._actively_polling = False
        self._last_compliance_poll = 0.0

    def stop(self):
        self._stop_requested = True

    def pause(self):
        self._is_paused = True
        import time
        t0 = time.time()
        while self._actively_polling and time.time() < t0 + 1.0:
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
                # Read magnet field
                b_val = 0.0
                magnet = getattr(self.station, 'magnet', None)
                if magnet is not None:
                    try:
                        b_val = magnet.field()
                    except Exception:
                        try:
                            b_val = magnet.output_field()
                        except Exception:
                            pass
                
                # Read SMU voltages
                v1_val = 0.0
                v2_val = 0.0
                smu = getattr(self.station, 'smu', None)
                if smu is not None:
                    ch1_ramping = getattr(smu.ch1, "ramping", False)
                    ch2_ramping = getattr(smu.ch2, "ramping", False)
                    if ch1_ramping or ch2_ramping:
                        v1_val = smu.ch1.volt.get_latest()
                        if v1_val is None:
                            v1_val = 0.0
                        v2_val = smu.ch2.volt.get_latest()
                        if v2_val is None:
                            v2_val = 0.0
                    else:
                        if getattr(smu, "simulated", False):
                            try:
                                v1_val = smu.ch1.volt()
                            except Exception:
                                pass
                            try:
                                v2_val = smu.ch2.volt()
                            except Exception:
                                pass
                        else:
                            old_timeout = None
                            try:
                                if hasattr(smu, "_lock"):
                                    with smu._lock:
                                        old_timeout = smu.timeout()
                                        smu.timeout(0.5)
                                else:
                                    old_timeout = smu.timeout()
                                    smu.timeout(0.5)
                            except Exception:
                                old_timeout = None

                            try:
                                v1_val = smu.ch1.volt()
                            except Exception:
                                try:
                                    smu.visa_handle.clear()
                                except Exception:
                                    pass
                            
                            try:
                                v2_val = smu.ch2.volt()
                            except Exception:
                                try:
                                    smu.visa_handle.clear()
                                except Exception:
                                    pass

                            if old_timeout is not None:
                                try:
                                    if hasattr(smu, "_lock"):
                                        with smu._lock:
                                            smu.timeout(old_timeout)
                                    else:
                                        smu.timeout(old_timeout)
                                except Exception:
                                    pass
                
                if smu is not None and not (getattr(smu.ch1, 'ramping', False) or getattr(smu.ch2, 'ramping', False)):
                    if time.monotonic() - self._last_compliance_poll >= 1.0:
                        self.compliance_updated.emit(read_compliance_values(smu))
                        self._last_compliance_poll = time.monotonic()
                self.status_updated.emit(b_val, v1_val, v2_val)
            except Exception:
                pass
            finally:
                self._actively_polling = False
            
            self.msleep(200) # 5 Hz

class MainWindow(QtWidgets.QMainWindow):
    editable_current_compliance = True

    def __init__(self, station, embed=False):
        super().__init__()
        self.station = station
        self._embed = embed
        self.plotter_mode = False
        self._compliance_readbacks = {}
        self._compliance_edits = set()
        self._applying_compliance = False
        install_parameter_wheel_guard(self)
        self.setWindowTitle("Electrostatic Gate Control & 2D Sweep Spectrometer")
        self.setMinimumSize(1280, 850)

        # Application State
        self.is_running = False
        self.is_paused = False
        self.reset_slice_levels = True
        self.sweep_worker = None
        self.ramp_worker = None
        self.single_spectra_window = None
        
        # Time tracking for expected end calculation
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None

        # Data grids
        self.c1_coords = np.array([])
        self.c2_coords = np.array([])
        self.v1_grid = np.array([])
        self.v2_grid = np.array([])
        
        # 3D spectra storage: (c2_len, c1_len, 512)
        self.data_spectra = np.array([])
        self.data_raw_spectra = np.array([])
        self.data_current_ch1 = np.array([])
        self.data_current_ch2 = np.array([])
        self.data_measured_mask = np.array([]) # Tracks which points are measured
        self.data_measured_b = np.array([])

        # Energy calibration axis
        self.energy_axis = np.linspace(1.5, 3.0, 512) # Dummy range initialized at startup
        self.linear_energy_axis = self.energy_axis
        self.energy_sort_idx = np.arange(512)
        self.selected_energy = 2.25

        # Gate map point coordinates
        self.point_gatemap_1 = 0.0
        self.point_gatemap_2 = 0.0

        # Load stored settings or defaults
        self.load_storage()

        self.setup_ui()
        self._update_compliance_controls()
        # Manual naming metadata only: never sends a monochromator command.
        self.filename_wavelength_box = QtWidgets.QDoubleSpinBox()
        self.filename_wavelength_box.setRange(1, 100000)
        self.filename_wavelength_box.setValue(1090)
        self.filename_wavelength_box.setSuffix(" nm")
        self.filename_wavelength_box.setToolTip("Central wavelength for the saved filename; entered manually, not a hardware setting.")
        naming_layout = QtWidgets.QFormLayout()
        naming_layout.addRow("Filename wavelength:", self.filename_wavelength_box)
        self.save_btn.parentWidget().layout().addLayout(naming_layout)
        self.apply_theme()
        
        # Read live output states from physical SMU at startup to sync buttons
        try:
            smu = self.station.smu
            ch1_out = bool(smu.ch1.output())
            ch2_out = bool(smu.ch2.output())
            
            self.ch1_out_btn.blockSignals(True)
            self.ch1_out_btn.setChecked(ch1_out)
            self.ch1_out_btn.setText(f"{'ON' if ch1_out else 'OFF'}")
            self.ch1_out_btn.setStyleSheet(f"background-color: {'#10b981' if ch1_out else '#f1f5f9'}; color: {'white' if ch1_out else '#475569'}; border: 1px solid {'#10b981' if ch1_out else '#cbd5e1'}; font-weight: bold;")
            self.ch1_out_btn.blockSignals(False)

            self.ch2_out_btn.blockSignals(True)
            self.ch2_out_btn.setChecked(ch2_out)
            self.ch2_out_btn.setText(f"{'ON' if ch2_out else 'OFF'}")
            self.ch2_out_btn.setStyleSheet(f"background-color: {'#10b981' if ch2_out else '#f1f5f9'}; color: {'white' if ch2_out else '#475569'}; border: 1px solid {'#10b981' if ch2_out else '#cbd5e1'}; font-weight: bold;")
            self.ch2_out_btn.blockSignals(False)

            # Sync source modes at startup
            ch1_mode = smu.ch1.source_mode()
            ch2_mode = smu.ch2.source_mode()
            
            self.ch1_mode.blockSignals(True)
            self.ch1_mode.setCurrentText(ch1_mode)
            self.ch1_mode.blockSignals(False)
            
            self.ch2_mode.blockSignals(True)
            self.ch2_mode.setCurrentText(ch2_mode)
            self.ch2_mode.blockSignals(False)
            
            # Trigger label/range adjustments to match active modes
            self.on_mode_changed(write_hardware=False)
            
            self.on_compliance_readback(read_compliance_values(smu))

            # Release front panel lock back to local control
            try:
                smu.return_to_local()
            except Exception:
                pass
        except Exception as e:
            print(f"Startup warning: could not sync SMU states ({e})")
        
        self.view_control_popup = ViewControlPopup(self, has_grid=True)

        # Start background 5 Hz status polling worker
        self.live_status_worker = LiveStatusWorker(self.station, parent=self)
        self.live_status_worker.status_updated.connect(self.on_live_status_updated)
        self.live_status_worker.compliance_updated.connect(self.on_compliance_readback)
        self.live_status_worker.start()

    def setup_histogram_lut(self, plot_win, image_item, interactive=True):
        """Helper to create a HistogramLUTItem with the colorbar positioned between the plot and the scale."""
        hist = pg.HistogramLUTItem(gradientPosition='left')
        hist.setImageItem(image_item)
        hist.gradient.loadPreset('viridis')
        
        # Rearrange layout to place colorbar between plot (2D heatmap) and scale (histogram + axis)
        hist.layout.removeItem(hist.gradient)
        hist.layout.removeItem(hist.vb)
        hist.layout.removeItem(hist.axis)
        hist.layout.addItem(hist.gradient, 0, 0) # Left column: Colorbar (Gradient)
        hist.layout.addItem(hist.vb, 0, 1)       # Middle column: Histogram Plot
        hist.layout.addItem(hist.axis, 0, 2)     # Right column: Scale (Axis)
        
        # Set compact widths and layout margins/spacings to prevent squeezing/clipping
        hist.layout.setContentsMargins(0, 0, 0, 0)
        hist.layout.setSpacing(2)
        
        # Make the intensity plot (vb) wider so that the curves are clearly visible
        hist.layout.setColumnFixedWidth(0, 12)  # Gradient colorbar
        hist.layout.setColumnFixedWidth(1, 45)  # ViewBox (intensity histogram plot)
        
        hist.gradient.setMinimumWidth(0)
        hist.gradient.setMaximumWidth(12)
        hist.vb.setMinimumWidth(0)
        hist.vb.setMaximumWidth(45)
        
        # Allow the axis and the overall widget to grow dynamically to fit any text length
        hist.setMinimumWidth(80)
        
        # Hide the gradient editor ticks for a cleaner and more professional look
        for tick in list(hist.gradient.ticks.keys()):
            tick.hide()
        hist.gradient.allowAdd = False
        
        if not interactive:
            hist.region.setMovable(False)
            for line in hist.region.lines:
                line.setMovable(False)
            hist.gradient.setEnabled(False)
            hist.vb.setMouseEnabled(x=False, y=True)
            
        # Adjust plot title span to span all 6 columns (0 to 5) so it doesn't force Column 1 to be too wide
        pi = plot_win.getPlotItem()
        pi.layout.removeItem(pi.titleLabel)
        pi.layout.addItem(pi.titleLabel, 0, 0, 1, 6)
            
        plot_win.getPlotItem().layout.addItem(hist, 2, 5)
        plot_win.getPlotItem().layout.setColumnFixedWidth(4, 3)
        return hist

    # --- Value Storage Persistence ---
    def load_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        defaults = {
            "exposure_time": 0.1,
            "settling_time": 0.05,
            "ramprate1": 0.1,
            "ramprate2": 0.1,
            "ramp_step": 0.05,
            "c1_name": "Vc1",
            "c2_name": "Vc2",
            "v1_compliance": GATE_CURRENT_COMPLIANCE_A,
            "v1_voltage_compliance": 10.0,
            "v2_compliance": GATE_CURRENT_COMPLIANCE_A,
            "v2_voltage_compliance": 10.0,
            "c1_start": -0.005,
            "c1_end": 0.005,
            "c1_pts": 21,
            "c2_start": -0.005,
            "c2_end": 0.005,
            "c2_pts": 21,
            "coeff_a": 1.0,
            "coeff_b": 0.0,
            "coeff_c": 0.0,
            "coeff_d": 1.0,
            "sweep_mode": "Hold C1, sweep C2",
            "serpentine": True
        }
        self.stored_settings = defaults.copy()
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    raw_data = json.load(f)
                
                shared_data = raw_data.get("shared", {})
                dg_data = raw_data.get("dual_gate", {})
                
                self.stored_settings["exposure_time"] = shared_data.get("exposure_time", raw_data.get("exposure_time", defaults["exposure_time"]))
                self.stored_settings["settling_time"] = shared_data.get("settling_time", raw_data.get("settling_time", defaults["settling_time"]))
                
                for key in defaults:
                    if key not in ("exposure_time", "settling_time"):
                        if key == "v1_compliance":
                            self.stored_settings[key] = GATE_CURRENT_COMPLIANCE_A
                        elif key == "v2_compliance":
                            self.stored_settings[key] = GATE_CURRENT_COMPLIANCE_A
                        elif key == "c1_name":
                            self.stored_settings[key] = shared_data.get("c1_name", dg_data.get("c1_name", raw_data.get("c1_name", defaults["c1_name"])))
                        elif key == "c2_name":
                            self.stored_settings[key] = shared_data.get("c2_name", dg_data.get("c2_name", raw_data.get("c2_name", defaults["c2_name"])))
                        elif key in ("coeff_a", "coeff_b", "coeff_c", "coeff_d"):
                            legacy_key = f"dg_coeff_{key[-1]}"
                            self.stored_settings[key] = dg_data.get(key, dg_data.get(legacy_key,
                                raw_data.get(legacy_key, shared_data.get(key, defaults[key]))))
                        else:
                            self.stored_settings[key] = dg_data.get(key, raw_data.get(key, defaults[key]))

                # Migrate old ramp_rate keys if present
                if "ramp_rate" in raw_data:
                    old_rate = raw_data["ramp_rate"]
                    self.stored_settings.setdefault("ramprate1", old_rate)
                    self.stored_settings.setdefault("ramprate2", old_rate)
                if "ramp_rate_1" in raw_data:
                    self.stored_settings.setdefault("ramprate1", raw_data["ramp_rate_1"])
                if "ramp_rate_2" in raw_data:
                    self.stored_settings.setdefault("ramprate2", raw_data["ramp_rate_2"])
                if str(self.stored_settings.get("c1_name", "")).strip().lower() in ("c1", ""):
                    self.stored_settings["c1_name"] = "Vc1"
                if str(self.stored_settings.get("c2_name", "")).strip().lower() in ("c2", ""):
                    self.stored_settings["c2_name"] = "Vc2"
            except Exception as e:
                print(f"Error loading value_storage: {e}")

    def save_storage(self):
        if getattr(self, "plotter_mode", False):
            return
        storage_path = os.path.join(parent_dir, "value_storage")
        if hasattr(self, "c1_name_edit"):
            self.stored_settings["c1_name"] = self.c1_name_edit.text()
        if hasattr(self, "c2_name_edit"):
            self.stored_settings["c2_name"] = self.c2_name_edit.text()
        if hasattr(self, "coeff_a"):
            self.stored_settings["coeff_a"] = self.coeff_a.value()
        if hasattr(self, "coeff_b"):
            self.stored_settings["coeff_b"] = self.coeff_b.value()
        if hasattr(self, "coeff_c"):
            self.stored_settings["coeff_c"] = self.coeff_c.value()
        if hasattr(self, "coeff_d"):
            self.stored_settings["coeff_d"] = self.coeff_d.value()
        try:
            current_data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    try:
                        current_data = json.load(f)
                    except Exception:
                        pass
            if "shared" not in current_data:
                current_data["shared"] = {}
            if "dual_gate" not in current_data:
                current_data["dual_gate"] = {}
            current_data["shared"]["exposure_time"] = self.stored_settings.get("exposure_time", 0.1)
            current_data["shared"]["settling_time"] = self.stored_settings.get("settling_time", 0.05)
            current_data["shared"]["c1_name"] = self.stored_settings.get("c1_name", "Vc1")
            current_data["shared"]["c2_name"] = self.stored_settings.get("c2_name", "Vc2")
            current_data["shared"]["coeff_a"] = self.stored_settings.get("coeff_a", 1.0)
            current_data["shared"]["coeff_b"] = self.stored_settings.get("coeff_b", 0.0)
            current_data["shared"]["coeff_c"] = self.stored_settings.get("coeff_c", 0.0)
            current_data["shared"]["coeff_d"] = self.stored_settings.get("coeff_d", 1.0)
            # DGS owns its matrix. Shared values remain a compatibility mirror
            # for map panels; B/V owns its independent control coefficients.
            for key in ('coeff_a', 'coeff_b', 'coeff_c', 'coeff_d'):
                current_data["dual_gate"][key] = self.stored_settings[key]
            camera = getattr(self.station, "camera", None)
            current_data["shared"]["spectrometer"] = getattr(camera, "spectrometer_type", "lightfieldG15")
            current_data["dual_gate"]["ramprate1"] = self.stored_settings.get("ramprate1", 0.1)
            current_data["dual_gate"]["ramprate2"] = self.stored_settings.get("ramprate2", 0.1)
            current_data["dual_gate"]["ramp_step"] = self.stored_settings.get("ramp_step", 0.0005)
            current_data["dual_gate"]["c1_name"] = self.stored_settings.get("c1_name", "Vc1")
            current_data["dual_gate"]["c2_name"] = self.stored_settings.get("c2_name", "Vc2")
            
            # New inputs
            if hasattr(self, "ch1_compliance"):
                current_data["dual_gate"]["v1_compliance"] = self._compliance_setting(1)
            if hasattr(self, "ch2_compliance"):
                current_data["dual_gate"]["v2_compliance"] = self._compliance_setting(2)
            if hasattr(self, "c1_start"):
                current_data["dual_gate"]["c1_start"] = self.c1_start.value()
            if hasattr(self, "c1_end"):
                current_data["dual_gate"]["c1_end"] = self.c1_end.value()
            if hasattr(self, "c1_pts"):
                current_data["dual_gate"]["c1_pts"] = self.c1_pts.value()
                c1_step = abs(self.c1_end.value() - self.c1_start.value()) / (self.c1_pts.value() - 1) if self.c1_pts.value() > 1 else 0.0
                current_data["dual_gate"]["c1_step"] = c1_step
            if hasattr(self, "c2_start"):
                current_data["dual_gate"]["c2_start"] = self.c2_start.value()
            if hasattr(self, "c2_end"):
                current_data["dual_gate"]["c2_end"] = self.c2_end.value()
            if hasattr(self, "c2_pts"):
                current_data["dual_gate"]["c2_pts"] = self.c2_pts.value()
                c2_step = abs(self.c2_end.value() - self.c2_start.value()) / (self.c2_pts.value() - 1) if self.c2_pts.value() > 1 else 0.0
                current_data["dual_gate"]["c2_step"] = c2_step
            if hasattr(self, "sweep_mode_combo"):
                current_data["dual_gate"]["sweep_mode"] = self.sweep_mode_combo.currentText()
            if hasattr(self, "serpentine_box"):
                current_data["dual_gate"]["serpentine"] = self.serpentine_box.isChecked()

            # Purge deprecated root-level keys and legacy c1/c2 compliance keys
            deprecated_keys = [
                "exposure_time", "settling_time", "ramprate1", "ramprate2", "ramp_step", 
                "c1_name", "c2_name", "grating", "wavelength", "target_temp", "num_spectra", 
                "average", "accum_method", "plot_mode", "x_axis_scale", "plot_modes", 
                "b_coord_name", "b_start", "b_end", "b_pts", "v_start", "v_end", "v_pts", 
                "v_fixed", "variable_order", "serpentine", "faraday_comp", "faraday_rot", 
                "end_ramp_zero", "end_ramp_ground", "bv_coeff_a", "bv_coeff_b", "bv_coeff_c", 
                "bv_coeff_d", "pol_exc_vertical", "pol_exc_horizontal", "pol_exc_manual", 
                "pol_coll_vertical", "pol_coll_horizontal", "pol_coll_manual",
                "c1_compliance", "c2_compliance",
                "dg_coeff_a", "dg_coeff_b", "dg_coeff_c", "dg_coeff_d"
            ]
            for key in deprecated_keys:
                current_data.pop(key, None)
                if "dual_gate" in current_data:
                    current_data["dual_gate"].pop("c1_compliance", None)
                    current_data["dual_gate"].pop("c2_compliance", None)
                    current_data["dual_gate"].pop("dg_coeff_a", None)
                    current_data["dual_gate"].pop("dg_coeff_b", None)
                    current_data["dual_gate"].pop("dg_coeff_c", None)
                    current_data["dual_gate"].pop("dg_coeff_d", None)
                
            with open(storage_path, "w") as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Error saving to value_storage: {e}")

    def reload_gui_from_storage(self):
        if getattr(self, "plotter_mode", False):
            return
        self.load_storage()
        
        self.c1_name_edit.blockSignals(True)
        self.c1_name_edit.setText(self.stored_settings.get("c1_name", "Vc1"))
        self.c1_name_edit.blockSignals(False)

        self.c2_name_edit.blockSignals(True)
        self.c2_name_edit.setText(self.stored_settings.get("c2_name", "Vc2"))
        self.c2_name_edit.blockSignals(False)

        self.ch1_ramp_rate.blockSignals(True)
        self.ch1_ramp_rate.setValue(self.stored_settings.get("ramprate1", 0.1))
        self.ch1_ramp_rate.blockSignals(False)

        self.ch2_ramp_rate.blockSignals(True)
        self.ch2_ramp_rate.setValue(self.stored_settings.get("ramprate2", 0.1))
        self.ch2_ramp_rate.blockSignals(False)

        self.exposure_box.blockSignals(True)
        self.exposure_box.setValue(self.stored_settings.get("exposure_time", 0.1))
        self.exposure_box.blockSignals(False)

        self.settling_box.blockSignals(True)
        self.settling_box.setValue(self.stored_settings.get("settling_time", 0.05))
        self.settling_box.blockSignals(False)

        # Compliance
        self.ch1_compliance.blockSignals(True)
        self.ch1_compliance.setValue(self._compliance_setting(1))
        self.ch1_compliance.blockSignals(False)

        self.ch2_compliance.blockSignals(True)
        self.ch2_compliance.setValue(self._compliance_setting(2))
        self.ch2_compliance.blockSignals(False)

        # Sweep range C1
        self.c1_start.blockSignals(True)
        self.c1_start.setValue(self.stored_settings.get("c1_start", -0.005))
        self.c1_start.blockSignals(False)

        self.c1_end.blockSignals(True)
        self.c1_end.setValue(self.stored_settings.get("c1_end", 0.005))
        self.c1_end.blockSignals(False)

        self.c1_pts.blockSignals(True)
        self.c1_pts.setValue(self.stored_settings.get("c1_pts", 21))
        self.c1_pts.blockSignals(False)

        # Sweep range C2
        self.c2_start.blockSignals(True)
        self.c2_start.setValue(self.stored_settings.get("c2_start", -0.005))
        self.c2_start.blockSignals(False)

        self.c2_end.blockSignals(True)
        self.c2_end.setValue(self.stored_settings.get("c2_end", 0.005))
        self.c2_end.blockSignals(False)

        self.c2_pts.blockSignals(True)
        self.c2_pts.setValue(self.stored_settings.get("c2_pts", 21))
        self.c2_pts.blockSignals(False)

        # Basis change coefficients
        self.coeff_a.blockSignals(True)
        self.coeff_a.setValue(self.stored_settings.get("coeff_a", 1.0))
        self.coeff_a.blockSignals(False)

        self.coeff_b.blockSignals(True)
        self.coeff_b.setValue(self.stored_settings.get("coeff_b", 0.0))
        self.coeff_b.blockSignals(False)

        self.coeff_c.blockSignals(True)
        self.coeff_c.setValue(self.stored_settings.get("coeff_c", 0.0))
        self.coeff_c.blockSignals(False)

        self.coeff_d.blockSignals(True)
        self.coeff_d.setValue(self.stored_settings.get("coeff_d", 1.0))
        self.coeff_d.blockSignals(False)

        # Sweep mode and serpentine
        self.sweep_mode_combo.blockSignals(True)
        self.sweep_mode_combo.setCurrentText(self.stored_settings.get("sweep_mode", "Hold C1, sweep C2"))
        self.sweep_mode_combo.blockSignals(False)

        self.serpentine_box.blockSignals(True)
        self.serpentine_box.setChecked(self.stored_settings.get("serpentine", True))
        self.serpentine_box.blockSignals(False)

        self.update_step_calculations()
        self.update_variable_names()
        self.update_coordinates_from_ui()
        self.on_compliance_changed()

    def create_scientific_validator(self, minimum, maximum, decimals):
        validator = QtGui.QDoubleValidator(minimum, maximum, decimals, self)
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        return validator

    def enable_scientific_input(self, spinbox):
        if hasattr(spinbox, "lineEdit") and spinbox.lineEdit() is not None:
            spinbox.lineEdit().setPlaceholderText("e.g. 1e-3")

    # --- UI Building ---
    def setup_ui(self):
        # Initialize Status Bar and Connection Status Indicator (only when not embedded)
        if not getattr(self, "_embed", False):
            status_bar = self.statusBar()
            self.conn_status_lbl = QtWidgets.QLabel()
            if self.station.smu.simulated:
                self.conn_status_lbl.setText("Keysight SMU: SIMULATED (Connection Locked/Failed)  ")
                self.conn_status_lbl.setStyleSheet("color: #d97706; font-weight: bold; font-family: Consolas, monospace;")
            else:
                self.conn_status_lbl.setText("Keysight SMU: CONNECTED (Physical)  ")
                self.conn_status_lbl.setStyleSheet("color: #059669; font-weight: bold; font-family: Consolas, monospace;")
            status_bar.addPermanentWidget(self.conn_status_lbl)
        else:
            # When embedded, still create the label for potential internal use but do not add to status bar
            self.conn_status_lbl = QtWidgets.QLabel()
            if self.station.smu.simulated:
                self.conn_status_lbl.setText("Keysight SMU: SIMULATED (Connection Locked/Failed)  ")
                self.conn_status_lbl.setStyleSheet("color: #d97706; font-weight: bold; font-family: Consolas, monospace;")
            else:
                self.conn_status_lbl.setText("Keysight SMU: CONNECTED (Physical)  ")
                self.conn_status_lbl.setStyleSheet("color: #059669; font-weight: bold; font-family: Consolas, monospace;")
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left Column (Controls Panel) - Broadened to 680 to fit 2-column layout
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(680)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        scroll_area.setWidget(left_widget)
        main_layout.addWidget(scroll_area)

        # Right Column (Tabs Panel)
        self.right_panel = QtWidgets.QTabWidget()
        main_layout.addWidget(self.right_panel, 1)

        # --- TWO-COLUMN CONTROLS LAYOUT (Left and Right Stacked Columns) ---
        primary_columns_layout = QtWidgets.QHBoxLayout()
        primary_columns_layout.setSpacing(10)

        # --- COLUMN 1 (Left Channel + Sweep Controls) ---
        col1_layout = QtWidgets.QVBoxLayout()
        col1_layout.setSpacing(10)
        col1_layout.setContentsMargins(0, 0, 0, 0)

        # Keysight Channel 1 SMU Controls
        ch1_box = QtWidgets.QGroupBox("Keysight channel 1")
        ch1_layout = QtWidgets.QVBoxLayout(ch1_box)
        ch1_layout.setSpacing(8)

        grid1 = QtWidgets.QGridLayout()
        grid1.setSpacing(6)
        
        grid1.addWidget(QtWidgets.QLabel("Output on"), 0, 0)
        self.ch1_out_btn = QtWidgets.QCheckBox()
        self.ch1_out_btn.toggled.connect(self.toggle_ch1_output)
        grid1.addWidget(self.ch1_out_btn, 0, 1)
        
        self.ch1_setpoint_lbl = QtWidgets.QLabel("Setpoint:")
        grid1.addWidget(self.ch1_setpoint_lbl, 1, 0)
        self.ch1_setpoint = QtWidgets.QDoubleSpinBox()
        self.ch1_setpoint.setRange(-210.0, 210.0)
        self.ch1_setpoint.setDecimals(4)
        self.ch1_setpoint.setSingleStep(0.001)
        self.ch1_setpoint.setKeyboardTracking(False)
        self.ch1_setpoint.lineEdit().returnPressed.connect(self.on_ch1_setpoint_enter)
        self.ch1_setpoint.setSuffix(" V")
        grid1.addWidget(self.ch1_setpoint, 1, 1)

        grid1.addWidget(QtWidgets.QLabel("Source mode:"), 2, 0)
        self.ch1_mode = QtWidgets.QComboBox()
        self.ch1_mode.addItems(["VOLT", "CURR"])
        self.ch1_mode.currentTextChanged.connect(self.on_mode_changed)
        grid1.addWidget(self.ch1_mode, 2, 1)

        grid1.addWidget(QtWidgets.QLabel("Ramp rate:"), 3, 0)
        self.ch1_ramp_rate = QtWidgets.QDoubleSpinBox()
        self.ch1_ramp_rate.setRange(1e-6, 100.0)
        self.ch1_ramp_rate.setSingleStep(1e-6)
        self.ch1_ramp_rate.setValue(self.stored_settings.get("ramprate1", 0.1))
        self.ch1_ramp_rate.setKeyboardTracking(False)
        self.ch1_ramp_rate.editingFinished.connect(self.on_ch1_ramping_params_changed)
        self.ch1_ramp_rate.setSuffix(" V/s")
        grid1.addWidget(self.ch1_ramp_rate, 3, 1)

        self.ch1_compliance_lbl = QtWidgets.QLabel("Compliance:")
        grid1.addWidget(self.ch1_compliance_lbl, 4, 0)
        self.ch1_compliance = QtWidgets.QDoubleSpinBox()
        self.ch1_compliance.setRange(1e-9, 3.0)
        self.ch1_compliance.setDecimals(9)
        self.ch1_compliance.setValue(self._compliance_setting(1))
        self.ch1_compliance.setSingleStep(0.001)
        self.ch1_compliance.setKeyboardTracking(False)
        self.ch1_compliance.valueChanged.connect(lambda _value: self._compliance_edits.add(1))
        self.ch1_compliance.lineEdit().textEdited.connect(lambda _text: self._compliance_edits.add(1))
        self.ch1_compliance.editingFinished.connect(lambda: self.on_compliance_changed(1))
        self.ch1_compliance.setSuffix(" A")
        grid1.addWidget(self.ch1_compliance, 4, 1)
        ch1_layout.addLayout(grid1)

        ch1_read_box = QtWidgets.QGroupBox("Last Reading")
        ch1_read_layout = QtWidgets.QVBoxLayout(ch1_read_box)
        ch1_read_layout.setSpacing(4)
        
        self.ch1_live_volt_lbl = QtWidgets.QLabel("Voltage: None")
        self.ch1_live_volt_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch1_read_layout.addWidget(self.ch1_live_volt_lbl)
        
        self.ch1_live_curr_lbl = QtWidgets.QLabel("Current: None")
        self.ch1_live_curr_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch1_read_layout.addWidget(self.ch1_live_curr_lbl)
        
        btn_row1 = QtWidgets.QHBoxLayout()
        btn_row1.setSpacing(6)
        btn_row1.setContentsMargins(0, 0, 0, 0)
        
        self.ch1_read_btn = QtWidgets.QPushButton("Read")
        self.ch1_read_btn.clicked.connect(self.read_channel_one)
        btn_row1.addWidget(self.ch1_read_btn)
        
        self.ch1_stop_btn = QtWidgets.QPushButton("Stop Ramp")
        self.ch1_stop_btn.setEnabled(False)
        self.ch1_stop_btn.clicked.connect(self.on_ch1_stop_clicked)
        btn_row1.addWidget(self.ch1_stop_btn)
        ch1_read_layout.addLayout(btn_row1)
        ch1_layout.addWidget(ch1_read_box)
        col1_layout.addWidget(ch1_box)

        # Control Variable 1 (Sweep settings)
        c1_default_name = self.stored_settings.get("c1_name", "Virtual Gate 1")
        self.ctrl_c1_box = QtWidgets.QGroupBox(f"{c1_default_name}")
        ctrl_c1_layout = QtWidgets.QVBoxLayout(self.ctrl_c1_box)
        
        c1_grid = QtWidgets.QGridLayout()
        c1_grid.setSpacing(6)
        
        c1_grid.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
        self.c1_start = QtWidgets.QDoubleSpinBox()
        self.c1_start.setRange(-210.0, 210.0)
        self.c1_start.setValue(self.stored_settings.get("c1_start", -0.005))
        self.c1_start.setDecimals(4)
        self.c1_start.setKeyboardTracking(False)
        c1_grid.addWidget(self.c1_start, 0, 1)

        c1_grid.addWidget(QtWidgets.QLabel("End:"), 1, 0)
        self.c1_end = QtWidgets.QDoubleSpinBox()
        self.c1_end.setRange(-210.0, 210.0)
        self.c1_end.setValue(self.stored_settings.get("c1_end", 0.005))
        self.c1_end.setDecimals(4)
        self.c1_end.setKeyboardTracking(False)
        c1_grid.addWidget(self.c1_end, 1, 1)

        c1_grid.addWidget(QtWidgets.QLabel("No. points:"), 2, 0)
        self.c1_pts = ScientificPointSpinBox()
        self.c1_pts.setRange(1, 1000)
        self.c1_pts.setValue(self.stored_settings.get("c1_pts", 21))
        self.c1_pts.setKeyboardTracking(True)
        self.c1_pts.lineEdit().setPlaceholderText("e.g. 1e2")
        c1_grid.addWidget(self.c1_pts, 2, 1)

        c1_grid.addWidget(QtWidgets.QLabel("Step size:"), 3, 0)
        self.c1_step_lbl = QtWidgets.QLabel("0.0 V")
        c1_grid.addWidget(self.c1_step_lbl, 3, 1)
        
        ctrl_c1_layout.addLayout(c1_grid)
        col1_layout.addWidget(self.ctrl_c1_box)
        primary_columns_layout.addLayout(col1_layout)

        # --- COLUMN 2 (Right Channel + Sweep Controls) ---
        col2_layout = QtWidgets.QVBoxLayout()
        col2_layout.setSpacing(10)
        col2_layout.setContentsMargins(0, 0, 0, 0)

        # Keysight Channel 2 SMU Controls
        ch2_box = QtWidgets.QGroupBox("Keysight channel 2")
        ch2_layout = QtWidgets.QVBoxLayout(ch2_box)
        ch2_layout.setSpacing(8)

        grid2 = QtWidgets.QGridLayout()
        grid2.setSpacing(6)
        
        grid2.addWidget(QtWidgets.QLabel("Output on"), 0, 0)
        self.ch2_out_btn = QtWidgets.QCheckBox()
        self.ch2_out_btn.toggled.connect(self.toggle_ch2_output)
        grid2.addWidget(self.ch2_out_btn, 0, 1)
        
        self.ch2_setpoint_lbl = QtWidgets.QLabel("Setpoint:")
        grid2.addWidget(self.ch2_setpoint_lbl, 1, 0)
        self.ch2_setpoint = QtWidgets.QDoubleSpinBox()
        self.ch2_setpoint.setRange(-210.0, 210.0)
        self.ch2_setpoint.setDecimals(4)
        self.ch2_setpoint.setSingleStep(0.001)
        self.ch2_setpoint.setKeyboardTracking(False)
        self.ch2_setpoint.lineEdit().returnPressed.connect(self.on_ch2_setpoint_enter)
        self.ch2_setpoint.setSuffix(" V")
        grid2.addWidget(self.ch2_setpoint, 1, 1)

        grid2.addWidget(QtWidgets.QLabel("Source mode:"), 2, 0)
        self.ch2_mode = QtWidgets.QComboBox()
        self.ch2_mode.addItems(["VOLT", "CURR"])
        self.ch2_mode.currentTextChanged.connect(self.on_mode_changed)
        grid2.addWidget(self.ch2_mode, 2, 1)

        grid2.addWidget(QtWidgets.QLabel("Ramp rate:"), 3, 0)
        self.ch2_ramp_rate = QtWidgets.QDoubleSpinBox()
        self.ch2_ramp_rate.setRange(1e-6, 100.0)
        self.ch2_ramp_rate.setSingleStep(1e-6)
        self.ch2_ramp_rate.setValue(self.stored_settings.get("ramprate2", 0.1))
        self.ch2_ramp_rate.setKeyboardTracking(False)
        self.ch2_ramp_rate.editingFinished.connect(self.on_ch2_ramping_params_changed)
        self.ch2_ramp_rate.setSuffix(" V/s")
        grid2.addWidget(self.ch2_ramp_rate, 3, 1)

        self.ch2_compliance_lbl = QtWidgets.QLabel("Compliance:")
        grid2.addWidget(self.ch2_compliance_lbl, 4, 0)
        self.ch2_compliance = QtWidgets.QDoubleSpinBox()
        self.ch2_compliance.setRange(1e-9, 3.0)
        self.ch2_compliance.setDecimals(9)
        self.ch2_compliance.setValue(self._compliance_setting(2))
        self.ch2_compliance.setSingleStep(0.001)
        self.ch2_compliance.setKeyboardTracking(False)
        self.ch2_compliance.valueChanged.connect(lambda _value: self._compliance_edits.add(2))
        self.ch2_compliance.lineEdit().textEdited.connect(lambda _text: self._compliance_edits.add(2))
        self.ch2_compliance.editingFinished.connect(lambda: self.on_compliance_changed(2))
        self.ch2_compliance.setSuffix(" A")
        grid2.addWidget(self.ch2_compliance, 4, 1)
        ch2_layout.addLayout(grid2)

        ch2_read_box = QtWidgets.QGroupBox("Last Reading")
        ch2_read_layout = QtWidgets.QVBoxLayout(ch2_read_box)
        ch2_read_layout.setSpacing(4)
        
        self.ch2_live_volt_lbl = QtWidgets.QLabel("Voltage: None")
        self.ch2_live_volt_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch2_read_layout.addWidget(self.ch2_live_volt_lbl)
        
        self.ch2_live_curr_lbl = QtWidgets.QLabel("Current: None")
        self.ch2_live_curr_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch2_read_layout.addWidget(self.ch2_live_curr_lbl)
        
        btn_row2 = QtWidgets.QHBoxLayout()
        btn_row2.setSpacing(6)
        btn_row2.setContentsMargins(0, 0, 0, 0)
        
        self.ch2_read_btn = QtWidgets.QPushButton("Read")
        self.ch2_read_btn.clicked.connect(self.read_channel_two)
        btn_row2.addWidget(self.ch2_read_btn)
        
        self.ch2_stop_btn = QtWidgets.QPushButton("Stop Ramp")
        self.ch2_stop_btn.setEnabled(False)
        self.ch2_stop_btn.clicked.connect(self.on_ch2_stop_clicked)
        btn_row2.addWidget(self.ch2_stop_btn)
        ch2_read_layout.addLayout(btn_row2)
        ch2_layout.addWidget(ch2_read_box)
        col2_layout.addWidget(ch2_box)

        # Control Variable 2 (Sweep settings)
        c2_default_name = self.stored_settings.get("c2_name", "Virtual Gate 2")
        self.ctrl_c2_box = QtWidgets.QGroupBox(f"{c2_default_name}")
        ctrl_c2_layout = QtWidgets.QVBoxLayout(self.ctrl_c2_box)
        
        c2_grid = QtWidgets.QGridLayout()
        c2_grid.setSpacing(6)
        
        c2_grid.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
        self.c2_start = QtWidgets.QDoubleSpinBox()
        self.c2_start.setRange(-210.0, 210.0)
        self.c2_start.setValue(self.stored_settings.get("c2_start", -0.005))
        self.c2_start.setDecimals(4)
        self.c2_start.setKeyboardTracking(False)
        c2_grid.addWidget(self.c2_start, 0, 1)

        c2_grid.addWidget(QtWidgets.QLabel("End:"), 1, 0)
        self.c2_end = QtWidgets.QDoubleSpinBox()
        self.c2_end.setRange(-210.0, 210.0)
        self.c2_end.setValue(self.stored_settings.get("c2_end", 0.005))
        self.c2_end.setDecimals(4)
        self.c2_end.setKeyboardTracking(False)
        c2_grid.addWidget(self.c2_end, 1, 1)

        c2_grid.addWidget(QtWidgets.QLabel("No. points:"), 2, 0)
        self.c2_pts = ScientificPointSpinBox()
        self.c2_pts.setRange(1, 1000)
        self.c2_pts.setValue(self.stored_settings.get("c2_pts", 21))
        self.c2_pts.setKeyboardTracking(True)
        self.c2_pts.lineEdit().setPlaceholderText("e.g. 1e2")
        c2_grid.addWidget(self.c2_pts, 2, 1)

        c2_grid.addWidget(QtWidgets.QLabel("Step size:"), 3, 0)
        self.c2_step_lbl = QtWidgets.QLabel("0.0 V")
        c2_grid.addWidget(self.c2_step_lbl, 3, 1)
        
        ctrl_c2_layout.addLayout(c2_grid)
        col2_layout.addWidget(self.ctrl_c2_box)
        primary_columns_layout.addLayout(col2_layout)

        left_layout.addLayout(primary_columns_layout)

        # Bind events to update step sizes and plot scaling dynamically
        for box in (self.c1_start, self.c1_end, self.c1_pts):
            box.editingFinished.connect(self.on_sweep_range_changed)
        for box in (self.c2_start, self.c2_end, self.c2_pts):
            box.editingFinished.connect(self.on_sweep_range_changed)
        self.update_step_calculations()

        # --- ROW 3: Gate Variable Matrix (full width) ---
        gate_box = QtWidgets.QGroupBox("Gate Variable")
        gate_layout = QtWidgets.QVBoxLayout(gate_box)
        gate_layout.setSpacing(10)

        # Name fields
        names_layout = QtWidgets.QGridLayout()
        names_layout.setSpacing(10)
        names_layout.addWidget(QtWidgets.QLabel("Virtual Gate 1 name:"), 0, 0)
        self.c1_name_edit = QtWidgets.QLineEdit(c1_default_name)
        names_layout.addWidget(self.c1_name_edit, 0, 1)
        
        names_layout.addWidget(QtWidgets.QLabel("Virtual Gate 2 name:"), 1, 0)
        self.c2_name_edit = QtWidgets.QLineEdit(c2_default_name)
        names_layout.addWidget(self.c2_name_edit, 1, 1)
        gate_layout.addLayout(names_layout)

        # 2x2 grid for formulas
        matrix_grid = QtWidgets.QGridLayout()
        matrix_grid.setSpacing(6)

        # Row 1: VG1 = coeff_a * ch1 + coeff_b * ch2
        self.v1_formula_label = QtWidgets.QLabel(f"{c1_default_name} =")
        matrix_grid.addWidget(self.v1_formula_label, 0, 0)

        self.coeff_a = QtWidgets.QDoubleSpinBox()
        self.coeff_a.setValue(self.stored_settings.get("coeff_a", 1.0))
        self.coeff_a.setRange(-100, 100)
        self.coeff_a.setDecimals(4)
        self.coeff_a.setSingleStep(0.01)
        self.coeff_a.setKeyboardTracking(False)
        matrix_grid.addWidget(self.coeff_a, 0, 1)

        self.v1_c1_label = QtWidgets.QLabel(" * ch1  + ")
        matrix_grid.addWidget(self.v1_c1_label, 0, 2)

        self.coeff_b = QtWidgets.QDoubleSpinBox()
        self.coeff_b.setValue(self.stored_settings.get("coeff_b", 0.0))
        self.coeff_b.setRange(-100, 100)
        self.coeff_b.setDecimals(4)
        self.coeff_b.setSingleStep(0.01)
        self.coeff_b.setKeyboardTracking(False)
        matrix_grid.addWidget(self.coeff_b, 0, 3)

        self.v1_c2_label = QtWidgets.QLabel(" * ch2")
        matrix_grid.addWidget(self.v1_c2_label, 0, 4)

        # Row 2: VG2 = coeff_c * ch1 + coeff_d * ch2
        self.v2_formula_label = QtWidgets.QLabel(f"{c2_default_name} =")
        matrix_grid.addWidget(self.v2_formula_label, 1, 0)

        self.coeff_c = QtWidgets.QDoubleSpinBox()
        self.coeff_c.setValue(self.stored_settings.get("coeff_c", 0.0))
        self.coeff_c.setRange(-100, 100)
        self.coeff_c.setDecimals(4)
        self.coeff_c.setSingleStep(0.01)
        self.coeff_c.setKeyboardTracking(False)
        matrix_grid.addWidget(self.coeff_c, 1, 1)

        self.v2_c1_label = QtWidgets.QLabel(" * ch1  + ")
        matrix_grid.addWidget(self.v2_c1_label, 1, 2)

        self.coeff_d = QtWidgets.QDoubleSpinBox()
        self.coeff_d.setValue(self.stored_settings.get("coeff_d", 1.0))
        self.coeff_d.setRange(-100, 100)
        self.coeff_d.setDecimals(4)
        self.coeff_d.setSingleStep(0.01)
        self.coeff_d.setKeyboardTracking(False)
        matrix_grid.addWidget(self.coeff_d, 1, 3)

        self.v2_c2_label = QtWidgets.QLabel(" * ch2")
        matrix_grid.addWidget(self.v2_c2_label, 1, 4)

        gate_layout.addLayout(matrix_grid)

        # Determinant indicator — shown below the matrix
        self.det_label = QtWidgets.QLabel("det = 1.0000  ✓")
        self.det_label.setStyleSheet("color: #16a34a; font-weight: bold; font-size: 11px;")
        gate_layout.addWidget(self.det_label)

        left_layout.addWidget(gate_box)

        # Connect name change events to update layouts and axes
        self.c1_name_edit.textChanged.connect(self.update_variable_names)
        self.c2_name_edit.textChanged.connect(self.update_variable_names)
        self.c1_name_edit.editingFinished.connect(self.save_storage)
        self.c2_name_edit.editingFinished.connect(self.save_storage)

        # Connect coefficient updates to automatically recalculate coordinates and det
        self.coeff_a.editingFinished.connect(self.on_sweep_range_changed)
        self.coeff_b.editingFinished.connect(self.on_sweep_range_changed)
        self.coeff_c.editingFinished.connect(self.on_sweep_range_changed)
        self.coeff_d.editingFinished.connect(self.on_sweep_range_changed)
        self.coeff_a.editingFinished.connect(self.update_det_label)
        self.coeff_b.editingFinished.connect(self.update_det_label)
        self.coeff_c.editingFinished.connect(self.update_det_label)
        self.coeff_d.editingFinished.connect(self.update_det_label)

        # --- ROW 4: Other Settings & Measurement Control next to each other ---
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(10)

        # Left bottom column (Other Settings & Plot Options stacked)
        left_bottom_layout = QtWidgets.QVBoxLayout()
        left_bottom_layout.setSpacing(10)
        left_bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Other Settings Box
        other_box = QtWidgets.QGroupBox("Other")
        other_layout = QtWidgets.QVBoxLayout(other_box)
        other_layout.setSpacing(6)
        
        other_layout.addWidget(QtWidgets.QLabel("Variable order:"))
        self.sweep_mode_combo = QtWidgets.QComboBox()
        self.sweep_mode_combo.addItems(["Hold C1, sweep C2", "Hold C2, sweep C1"])
        self.sweep_mode_combo.setCurrentText(self.stored_settings.get("sweep_mode", "Hold C1, sweep C2"))
        self.sweep_mode_combo.currentTextChanged.connect(self.on_sweep_range_changed)
        other_layout.addWidget(self.sweep_mode_combo)

        self.serpentine_box = QtWidgets.QCheckBox("Serpentine")
        self.serpentine_box.setChecked(self.stored_settings.get("serpentine", True))
        self.serpentine_box.stateChanged.connect(self.save_storage)
        other_layout.addWidget(self.serpentine_box)

        # Timing Grid
        timing_grid = QtWidgets.QGridLayout()
        timing_grid.setSpacing(6)
        timing_grid.addWidget(QtWidgets.QLabel("Exposure:"), 0, 0)
        self.exposure_box = QtWidgets.QDoubleSpinBox()
        self.exposure_box.setRange(0.001, 10.0)
        self.exposure_box.setDecimals(3)
        self.exposure_box.setValue(self.stored_settings.get("exposure_time", 0.1))
        self.exposure_box.setKeyboardTracking(False)
        self.exposure_box.editingFinished.connect(self.on_timing_params_changed)
        self.exposure_box.setSuffix(" s")
        timing_grid.addWidget(self.exposure_box, 0, 1)

        timing_grid.addWidget(QtWidgets.QLabel("Settling time:"), 1, 0)
        self.settling_box = QtWidgets.QDoubleSpinBox()
        self.settling_box.setRange(0.0, 10.0)
        self.settling_box.setDecimals(3)
        self.settling_box.setValue(self.stored_settings.get("settling_time", 0.05))
        self.settling_box.setKeyboardTracking(False)
        self.settling_box.editingFinished.connect(self.on_timing_params_changed)
        self.settling_box.setSuffix(" s")
        timing_grid.addWidget(self.settling_box, 1, 1)
        other_layout.addLayout(timing_grid)
        left_bottom_layout.addWidget(other_box)



        bottom_row.addLayout(left_bottom_layout, 0)

        # Enable scientific notation entry on numeric spin boxes
        for spin in (
            self.ch1_setpoint,
            self.ch1_ramp_rate,
            self.ch1_compliance,
            self.ch2_setpoint,
            self.ch2_ramp_rate,
            self.ch2_compliance,
            self.c1_start,
            self.c1_end,
            self.c2_start,
            self.c2_end,
            self.coeff_a,
            self.coeff_b,
            self.coeff_c,
            self.coeff_d,
            self.exposure_box,
            self.settling_box,
        ):
            self.enable_scientific_input(spin)

        # Measurement Control Box
        meas_box = QtWidgets.QGroupBox("Measurement control")
        meas_layout = QtWidgets.QVBoxLayout(meas_box)
        meas_layout.setSpacing(6)
        
        self.begin_btn = QtWidgets.QPushButton("► Begin")
        self.begin_btn.setObjectName("begin_btn")
        self.begin_btn.clicked.connect(self.begin_sweep)
        meas_layout.addWidget(self.begin_btn)

        self.pause_btn = QtWidgets.QPushButton("⏸ Interrupt")
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.on_interrupt_clicked)
        meas_layout.addWidget(self.pause_btn)

        self.reset_btn = QtWidgets.QPushButton("↺ Reset")
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.clicked.connect(self.reset_sweep)
        meas_layout.addWidget(self.reset_btn)

        self.save_btn = QtWidgets.QPushButton("💾 Save Dataset")
        self.save_btn.setObjectName("save_btn")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_sweep_data)
        meas_layout.addWidget(self.save_btn)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        meas_layout.addWidget(self.progress_bar)

        self.lbl_expected_end = QtWidgets.QLabel("Expected End: --:--:--")
        self.lbl_expected_end.setStyleSheet("color: #0F172A; font-weight: bold;")
        meas_layout.addWidget(self.lbl_expected_end)

        self.status_lbl = QtWidgets.QLabel("Status: Idle")
        self.status_lbl.setStyleSheet("color: #0F172A; font-style: italic; font-weight: bold;")
        meas_layout.addWidget(self.status_lbl)

        self.lbl_target_vc1 = QtWidgets.QLabel("V1 = --‑‑ V")
        self.lbl_target_vc1.setStyleSheet("color: #0F172A; font-weight: bold;")
        meas_layout.addWidget(self.lbl_target_vc1)

        self.lbl_target_vc2 = QtWidgets.QLabel("V2 = --‑‑ V")
        self.lbl_target_vc2.setStyleSheet("color: #0F172A; font-weight: bold;")
        meas_layout.addWidget(self.lbl_target_vc2)

        self.lbl_current_b = QtWidgets.QLabel("B = --‑‑ T")
        self.lbl_current_b.setStyleSheet("color: #0F172A; font-weight: bold;")
        meas_layout.addWidget(self.lbl_current_b)

        bottom_row.addWidget(meas_box, 1)
        left_layout.addLayout(bottom_row)

        # --- RIGHT TAB PANEL SECTIONS ---
        self.tab1 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab1, "Optical signal")
        self.setup_tab1()

        self.tab2 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab2, "IV channel 1")
        self.setup_tab2()

        self.tab3 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab3, "IV channel 2")
        self.setup_tab3()

        self.btn_view_control = QtWidgets.QPushButton("View Control ▾")
        self.btn_view_control.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 0px;
                padding: 6px 12px;
                font-weight: bold;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        self.right_panel.setCornerWidget(self.btn_view_control, QtCore.Qt.Corner.TopRightCorner)
        self.btn_view_control.clicked.connect(self.toggle_view_control_popup)
        self.right_panel.currentChanged.connect(self.on_tab_changed)

        # Update initial variable names
        self.update_variable_names()
        self.update_coordinates_from_ui()
        self.reset_plot_scaling(reset_spectrum=False)
    def setup_tab1(self):
        """Setup 2x2 grid of plots for Gating vs Energy Spectra in Tab 1"""
        layout = QtWidgets.QGridLayout(self.tab1)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # 1. Gate Map Plot (Top Left)
        self.gatemap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="2D Gate Map (Integrated Counts)")
        self.gatemap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.gatemap_win.getViewBox().enableAutoRange(enable=False)
        self.gatemap_img = pg.ImageItem()
        self.gatemap_img.setOpts(axisOrder='row-major')
        self.gatemap_win.addItem(self.gatemap_img)
        self.gatemap_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.gatemap_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        
        # Color bar / Histogram
        self.gatemap_colorbar = self.setup_histogram_lut(self.gatemap_win, self.gatemap_img, interactive=False)
        
        # Crosshair lines
        self.cross1_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.cross1_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.gatemap_win.addItem(self.cross1_v)
        self.gatemap_win.addItem(self.cross1_h)
        self.cross1_v.sigPositionChanged.connect(self.on_crosshair_moved)
        self.cross1_h.sigPositionChanged.connect(self.on_crosshair_moved)
        
        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
        viewbox1 = self.gatemap_win.getViewBox()
        viewbox1.originalMouseClickEvent = viewbox1.mouseClickEvent
        def gatemap_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox1.mapSceneToView(ev.scenePos())
                c1_val = mouse_point.x()
                c2_val = mouse_point.y()
                self.update_crosshair_values(c1_val, c2_val, self.selected_energy)
                self.refresh_slice_plots(c1_val, c2_val)
                ev.accept()
            else:
                viewbox1.originalMouseClickEvent(ev)
        viewbox1.mouseClickEvent = gatemap_click_event

        layout.addWidget(self.gatemap_win, 0, 0)

        # 2. Vertical Slice (Top Right)
        self.vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Vertical Slice spectra")
        self.vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.vslice_win.getViewBox().enableAutoRange(enable=False)
        self.vslice_img = pg.ImageItem()
        self.vslice_img.setOpts(axisOrder='row-major')
        self.vslice_win.addItem(self.vslice_img)
        self.vslice_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.vslice_win.setLabel('bottom', 'Energy', units='eV')
        
        self.vslice_colorbar = self.setup_histogram_lut(self.vslice_win, self.vslice_img, interactive=True)

        # Draggable crosshair lines
        self.vslice_cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.vslice_cross_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        # Add crosshair lines to Vertical Slice plot (Plot B)
        self.vslice_win.addItem(self.vslice_cross_v)
        self.vslice_win.addItem(self.vslice_cross_h)
        self.vslice_cross_v.sigPositionChanged.connect(self.on_vslice_crosshair_moved)
        self.vslice_cross_h.sigPositionChanged.connect(self.on_vslice_crosshair_moved)

        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
        viewbox_vslice = self.vslice_win.getViewBox()
        viewbox_vslice.originalMouseClickEvent = viewbox_vslice.mouseClickEvent
        def vslice_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox_vslice.mapSceneToView(ev.scenePos())
                c1_val = self.cross1_v.value()
                c2_val = mouse_point.y()
                energy_val = self.selected_energy # Keep current selected energy instead of changing it
                self.update_crosshair_values(c1_val, c2_val, energy_val)
                self.refresh_slice_plots(c1_val, c2_val)
                ev.accept()
            else:
                viewbox_vslice.originalMouseClickEvent(ev)
        viewbox_vslice.mouseClickEvent = vslice_click_event

        layout.addWidget(self.vslice_win, 0, 1)

        # 3. Horizontal Slice (Bottom Left)
        self.hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Horizontal Slice spectra")
        self.hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.hslice_win.getViewBox().enableAutoRange(enable=False)
        self.hslice_img = pg.ImageItem()
        self.hslice_img.setOpts(axisOrder='row-major')
        self.hslice_win.addItem(self.hslice_img)
        self.hslice_win.setLabel('left', 'Virtual Gate C1', units='V')
        self.hslice_win.setLabel('bottom', 'Energy', units='eV')

        self.hslice_colorbar = self.setup_histogram_lut(self.hslice_win, self.hslice_img, interactive=True)

        # Draggable crosshair lines
        self.hslice_cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.hslice_cross_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        # Do not add crosshair lines to Horizontal Slice plot (Plot C) per user request
        # self.hslice_win.addItem(self.hslice_cross_v)
        # self.hslice_win.addItem(self.hslice_cross_h)
        # self.hslice_cross_v.sigPositionChanged.connect(self.on_hslice_crosshair_moved)
        # self.hslice_cross_h.sigPositionChanged.connect(self.on_hslice_crosshair_moved)

        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
        viewbox_hslice = self.hslice_win.getViewBox()
        viewbox_hslice.originalMouseClickEvent = viewbox_hslice.mouseClickEvent
        def hslice_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox_hslice.mapSceneToView(ev.scenePos())
                c1_val = mouse_point.y()
                c2_val = self.cross1_h.value()
                energy_val = mouse_point.x()
                self.update_crosshair_values(c1_val, c2_val, energy_val)
                self.refresh_slice_plots(c1_val, c2_val)
                ev.accept()
            else:
                viewbox_hslice.originalMouseClickEvent(ev)
        viewbox_hslice.mouseClickEvent = hslice_click_event

        layout.addWidget(self.hslice_win, 1, 0)

        # 4. Spectrum at Cross (Bottom Right)
        self.spectrum_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="CCD Spectrum at Gating Crosshair")
        self.spectrum_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.spectrum_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.spectrum_win.getViewBox().enableAutoRange(axis='y', enable=False)
        self.spectrum_curve = self.spectrum_win.plot(
            pen=pg.mkPen('#000000', width=2)
        )
        self.spectrum_win.setLabel('left', 'Counts')
        self.spectrum_win.setLabel('bottom', 'Energy', units='eV')
        self.spectrum_win.showGrid(x=False, y=False)

        # Draggable vertical lines (spectrum_v1 and spectrum_v2) to highlight region (X-axis integration window)
        self.spectrum_v_region = pg.LinearRegionItem(
            orientation='vertical',
            values=(2.4, 2.6),
            brush=pg.mkBrush(16, 185, 129, 30), # green transparent fill
            pen=pg.mkPen('#10b981', width=1.5)  # continuous green line
        )
        self.spectrum_win.addItem(self.spectrum_v_region, ignoreBounds=True)
        self.spectrum_v1 = self.spectrum_v_region.lines[0]
        self.spectrum_v2 = self.spectrum_v_region.lines[1]
        self.spectrum_v_region.sigRegionChanged.connect(self.on_integration_region_changed)

        # Draggable horizontal lines (spectrum_h1 and spectrum_h2) to highlight region (Y-axis counts window)
        self.spectrum_h_region = pg.LinearRegionItem(
            orientation='horizontal',
            values=(1000.0, 5000.0),
            brush=pg.mkBrush(16, 185, 129, 30), # green transparent fill
            pen=pg.mkPen('#10b981', width=1.5)  # continuous green line
        )
        self.spectrum_win.addItem(self.spectrum_h_region, ignoreBounds=True)
        self.spectrum_h1 = self.spectrum_h_region.lines[0]
        self.spectrum_h2 = self.spectrum_h_region.lines[1]
        self.spectrum_h_region.sigRegionChanged.connect(self.on_integration_region_changed)

        # Position crosshair markers on the spectrum plot
        self.spectrum_cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#10b981', width=2.5))
        self.spectrum_cross_v.setValue(self.selected_energy)
        # Do not add the energy crosshair vertical line to the spectrum plot per user request
        # self.spectrum_win.addItem(self.spectrum_cross_v)
        # self.spectrum_cross_v.sigPositionChanged.connect(self.on_spectrum_crosshair_moved)


        layout.addWidget(self.spectrum_win, 1, 1)

    def setup_tab2(self):
        """Setup 2x2 layout for Ch1 Current vs coordinates"""
        layout = QtWidgets.QGridLayout(self.tab2)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # Heatmap (Top Left)
        self.t2_heatmap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Top Gate Current Heatmap")
        self.t2_heatmap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_heatmap_win.getViewBox().enableAutoRange(enable=False)
        self.t2_heatmap_img = pg.ImageItem()
        self.t2_heatmap_img.setOpts(axisOrder='row-major')
        self.t2_heatmap_win.addItem(self.t2_heatmap_img)
        self.t2_heatmap_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.t2_heatmap_win.setLabel('bottom', 'Virtual Gate C1', units='V')

        self.t2_colorbar = self.setup_histogram_lut(self.t2_heatmap_win, self.t2_heatmap_img, interactive=True)

        # Draggable crosshair
        self.cross2_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#ea580c', width=2))
        self.cross2_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#ea580c', width=2))
        self.t2_heatmap_win.addItem(self.cross2_v)
        self.t2_heatmap_win.addItem(self.cross2_h)
        self.cross2_v.sigPositionChanged.connect(self.on_tab2_crosshair_moved)
        self.cross2_h.sigPositionChanged.connect(self.on_tab2_crosshair_moved)
        
        # Monkey-patch ViewBox to intercept left clicks
        viewbox2 = self.t2_heatmap_win.getViewBox()
        viewbox2.originalMouseClickEvent = viewbox2.mouseClickEvent
        def t2_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox2.mapSceneToView(ev.scenePos())
                self.cross2_v.blockSignals(True)
                self.cross2_h.blockSignals(True)
                self.cross2_v.setValue(mouse_point.x())
                self.cross2_h.setValue(mouse_point.y())
                self.cross2_v.blockSignals(False)
                self.cross2_h.blockSignals(False)
                self.on_tab2_crosshair_moved()
                ev.accept()
            else:
                viewbox2.originalMouseClickEvent(ev)
        viewbox2.mouseClickEvent = t2_click_event

        layout.addWidget(self.t2_heatmap_win, 0, 0)

        # Vertical Slice (Top Right)
        self.t2_vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C1" if True else "Slice at fixed V")
        self.t2_vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_vslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t2_vslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t2_vslice_curve = self.t2_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t2_vslice_win.setLabel('left', 'Current', units='A')
        self.t2_vslice_win.setLabel('bottom', 'Virtual Gate C2', units='V')
        self.t2_vslice_win.showGrid(x=False, y=False)
        layout.addWidget(self.t2_vslice_win, 0, 1)

        # Horizontal Slice (Bottom Left)
        self.t2_hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C2" if True else "Slice at fixed B")
        self.t2_hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_hslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t2_hslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t2_hslice_curve = self.t2_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t2_hslice_win.setLabel('left', 'Current', units='A')
        self.t2_hslice_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        self.t2_hslice_win.showGrid(x=False, y=False)
        layout.addWidget(self.t2_hslice_win, 1, 0)

        # Empty (Bottom Right)
        dummy = QtWidgets.QWidget()
        layout.addWidget(dummy, 1, 1)

    def setup_tab3(self):
        """Setup 2x2 layout for Ch2 Current vs coordinates"""
        layout = QtWidgets.QGridLayout(self.tab3)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # Heatmap (Top Left)
        self.t3_heatmap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Bottom Gate Current Heatmap")
        self.t3_heatmap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_heatmap_win.getViewBox().enableAutoRange(enable=False)
        self.t3_heatmap_img = pg.ImageItem()
        self.t3_heatmap_img.setOpts(axisOrder='row-major')
        self.t3_heatmap_win.addItem(self.t3_heatmap_img)
        self.t3_heatmap_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.t3_heatmap_win.setLabel('bottom', 'Virtual Gate C1', units='V')

        self.t3_colorbar = self.setup_histogram_lut(self.t3_heatmap_win, self.t3_heatmap_img, interactive=True)

        # Draggable crosshair
        self.cross3_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#16a34a', width=2))
        self.cross3_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#16a34a', width=2))
        self.t3_heatmap_win.addItem(self.cross3_v)
        self.t3_heatmap_win.addItem(self.cross3_h)
        self.cross3_v.sigPositionChanged.connect(self.on_tab3_crosshair_moved)
        self.cross3_h.sigPositionChanged.connect(self.on_tab3_crosshair_moved)
        
        # Monkey-patch ViewBox to intercept left clicks
        viewbox3 = self.t3_heatmap_win.getViewBox()
        viewbox3.originalMouseClickEvent = viewbox3.mouseClickEvent
        def t3_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox3.mapSceneToView(ev.scenePos())
                self.cross3_v.blockSignals(True)
                self.cross3_h.blockSignals(True)
                self.cross3_v.setValue(mouse_point.x())
                self.cross3_h.setValue(mouse_point.y())
                self.cross3_v.blockSignals(False)
                self.cross3_h.blockSignals(False)
                self.on_tab3_crosshair_moved()
                ev.accept()
            else:
                viewbox3.originalMouseClickEvent(ev)
        viewbox3.mouseClickEvent = t3_click_event

        layout.addWidget(self.t3_heatmap_win, 0, 0)

        # Vertical Slice (Top Right)
        self.t3_vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C1" if True else "Slice at fixed V")
        self.t3_vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_vslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t3_vslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t3_vslice_curve = self.t3_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t3_vslice_win.setLabel('left', 'Current', units='A')
        self.t3_vslice_win.setLabel('bottom', 'Virtual Gate C2', units='V')
        self.t3_vslice_win.showGrid(x=False, y=False)
        layout.addWidget(self.t3_vslice_win, 0, 1)

        # Horizontal Slice (Bottom Left)
        self.t3_hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C2" if True else "Slice at fixed B")
        self.t3_hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_hslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t3_hslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t3_hslice_curve = self.t3_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t3_hslice_win.setLabel('left', 'Current', units='A')
        self.t3_hslice_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        self.t3_hslice_win.showGrid(x=False, y=False)
        layout.addWidget(self.t3_hslice_win, 1, 0)

        # Empty (Bottom Right)
        dummy = QtWidgets.QWidget()
        layout.addWidget(dummy, 1, 1)

    def setup_histogram_lut(self, plot_win, image_item, interactive=True):
        """Helper to create a HistogramLUTItem with the colorbar positioned between the plot and the scale."""
        hist = pg.HistogramLUTItem(gradientPosition='left')
        hist.setImageItem(image_item)
        hist.gradient.loadPreset('viridis')
        
        # Rearrange layout to place colorbar between plot (2D heatmap) and scale (histogram + axis)
        hist.layout.removeItem(hist.gradient)
        hist.layout.removeItem(hist.vb)
        hist.layout.removeItem(hist.axis)
        hist.layout.addItem(hist.gradient, 0, 0) # Left column: Colorbar (Gradient)
        hist.layout.addItem(hist.vb, 0, 1)       # Middle column: Histogram Plot
        hist.layout.addItem(hist.axis, 0, 2)     # Right column: Scale (Axis)
        
        # Set compact widths and layout margins/spacings to prevent squeezing/clipping
        hist.layout.setContentsMargins(0, 0, 0, 0)
        hist.layout.setSpacing(2)
        
        # Make the intensity plot (vb) wider so that the curves are clearly visible
        hist.layout.setColumnFixedWidth(0, 12)  # Gradient colorbar
        hist.layout.setColumnFixedWidth(1, 45)  # ViewBox (intensity histogram plot)
        
        hist.gradient.setMinimumWidth(0)
        hist.gradient.setMaximumWidth(12)
        hist.vb.setMinimumWidth(0)
        hist.vb.setMaximumWidth(45)
        
        # Allow the axis and the overall widget to grow dynamically to fit any text length
        hist.setMinimumWidth(80)
        
        # Hide the gradient editor ticks for a cleaner and more professional look
        for tick in list(hist.gradient.ticks.keys()):
            tick.hide()
        hist.gradient.allowAdd = False
        
        if not interactive:
            hist.region.setMovable(False)
            for line in hist.region.lines:
                line.setMovable(False)
            hist.gradient.setEnabled(False)
            hist.vb.setMouseEnabled(x=False, y=True)
            
        # Adjust plot title span to span all 6 columns (0 to 5) so it doesn't force Column 1 to be too wide
        pi = plot_win.getPlotItem()
        pi.layout.removeItem(pi.titleLabel)
        pi.layout.addItem(pi.titleLabel, 0, 0, 1, 6)
            
        plot_win.getPlotItem().layout.addItem(hist, 2, 5)
        plot_win.getPlotItem().layout.setColumnFixedWidth(4, 3)
        return hist

    # --- Value Storage Persistence ---
    def load_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        defaults = {
            "exposure_time": 0.1,
            "settling_time": 0.05,
            "ramprate1": 0.1,
            "ramprate2": 0.1,
            "ramp_step": 0.05,
            "c1_name": "Vc1",
            "c2_name": "Vc2",
            "v1_compliance": GATE_CURRENT_COMPLIANCE_A,
            "v1_voltage_compliance": 10.0,
            "v2_compliance": GATE_CURRENT_COMPLIANCE_A,
            "v2_voltage_compliance": 10.0,
            "c1_start": -0.005,
            "c1_end": 0.005,
            "c1_pts": 21,
            "c2_start": -0.005,
            "c2_end": 0.005,
            "c2_pts": 21,
            "coeff_a": 1.0,
            "coeff_b": 0.0,
            "coeff_c": 0.0,
            "coeff_d": 1.0,
            "sweep_mode": "Hold C1, sweep C2",
            "serpentine": True
        }
        self.stored_settings = defaults.copy()
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    raw_data = json.load(f)
                
                shared_data = raw_data.get("shared", {})
                dg_data = raw_data.get("dual_gate", {})
                
                self.stored_settings["exposure_time"] = shared_data.get("exposure_time", raw_data.get("exposure_time", defaults["exposure_time"]))
                self.stored_settings["settling_time"] = shared_data.get("settling_time", raw_data.get("settling_time", defaults["settling_time"]))
                
                for key in defaults:
                    if key not in ("exposure_time", "settling_time"):
                        if key == "v1_compliance":
                            self.stored_settings[key] = GATE_CURRENT_COMPLIANCE_A
                        elif key == "v2_compliance":
                            self.stored_settings[key] = GATE_CURRENT_COMPLIANCE_A
                        elif key == "c1_name":
                            self.stored_settings[key] = shared_data.get("c1_name", dg_data.get("c1_name", raw_data.get("c1_name", defaults["c1_name"])))
                        elif key == "c2_name":
                            self.stored_settings[key] = shared_data.get("c2_name", dg_data.get("c2_name", raw_data.get("c2_name", defaults["c2_name"])))
                        elif key in ("coeff_a", "coeff_b", "coeff_c", "coeff_d"):
                            legacy_key = f"dg_coeff_{key[-1]}"
                            self.stored_settings[key] = dg_data.get(key, dg_data.get(legacy_key,
                                raw_data.get(legacy_key, shared_data.get(key, defaults[key]))))
                        else:
                            self.stored_settings[key] = dg_data.get(key, raw_data.get(key, defaults[key]))

                # Migrate old ramp_rate keys if present
                if "ramp_rate" in raw_data:
                    old_rate = raw_data["ramp_rate"]
                    self.stored_settings.setdefault("ramprate1", old_rate)
                    self.stored_settings.setdefault("ramprate2", old_rate)
                if "ramp_rate_1" in raw_data:
                    self.stored_settings.setdefault("ramprate1", raw_data["ramp_rate_1"])
                if "ramp_rate_2" in raw_data:
                    self.stored_settings.setdefault("ramprate2", raw_data["ramp_rate_2"])
                if str(self.stored_settings.get("c1_name", "")).strip().lower() in ("c1", ""):
                    self.stored_settings["c1_name"] = "Vc1"
                if str(self.stored_settings.get("c2_name", "")).strip().lower() in ("c2", ""):
                    self.stored_settings["c2_name"] = "Vc2"
            except Exception as e:
                print(f"Error loading value_storage: {e}")

    def save_storage(self):
        if getattr(self, "plotter_mode", False):
            return
        storage_path = os.path.join(parent_dir, "value_storage")
        if hasattr(self, "c1_name_edit"):
            self.stored_settings["c1_name"] = self.c1_name_edit.text()
        if hasattr(self, "c2_name_edit"):
            self.stored_settings["c2_name"] = self.c2_name_edit.text()
        if hasattr(self, "coeff_a"):
            self.stored_settings["coeff_a"] = self.coeff_a.value()
        if hasattr(self, "coeff_b"):
            self.stored_settings["coeff_b"] = self.coeff_b.value()
        if hasattr(self, "coeff_c"):
            self.stored_settings["coeff_c"] = self.coeff_c.value()
        if hasattr(self, "coeff_d"):
            self.stored_settings["coeff_d"] = self.coeff_d.value()
        try:
            current_data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    try:
                        current_data = json.load(f)
                    except Exception:
                        pass
            if "shared" not in current_data:
                current_data["shared"] = {}
            if "dual_gate" not in current_data:
                current_data["dual_gate"] = {}
            current_data["shared"]["exposure_time"] = self.stored_settings.get("exposure_time", 0.1)
            current_data["shared"]["settling_time"] = self.stored_settings.get("settling_time", 0.05)
            current_data["shared"]["c1_name"] = self.stored_settings.get("c1_name", "Vc1")
            current_data["shared"]["c2_name"] = self.stored_settings.get("c2_name", "Vc2")
            current_data["shared"]["coeff_a"] = self.stored_settings.get("coeff_a", 1.0)
            current_data["shared"]["coeff_b"] = self.stored_settings.get("coeff_b", 0.0)
            current_data["shared"]["coeff_c"] = self.stored_settings.get("coeff_c", 0.0)
            current_data["shared"]["coeff_d"] = self.stored_settings.get("coeff_d", 1.0)
            # DGS owns its matrix. Shared values remain a compatibility mirror
            # for map panels; B/V owns its independent control coefficients.
            for key in ('coeff_a', 'coeff_b', 'coeff_c', 'coeff_d'):
                current_data["dual_gate"][key] = self.stored_settings[key]
            camera = getattr(self.station, "camera", None)
            current_data["shared"]["spectrometer"] = getattr(camera, "spectrometer_type", "lightfieldG15")
            current_data["dual_gate"]["ramprate1"] = self.stored_settings.get("ramprate1", 0.1)
            current_data["dual_gate"]["ramprate2"] = self.stored_settings.get("ramprate2", 0.1)
            current_data["dual_gate"]["ramp_step"] = self.stored_settings.get("ramp_step", 0.0005)
            current_data["dual_gate"]["c1_name"] = self.stored_settings.get("c1_name", "Vc1")
            current_data["dual_gate"]["c2_name"] = self.stored_settings.get("c2_name", "Vc2")
            
            # New inputs
            if hasattr(self, "ch1_compliance"):
                current_data["dual_gate"]["v1_compliance"] = self._compliance_setting(1)
            if hasattr(self, "ch2_compliance"):
                current_data["dual_gate"]["v2_compliance"] = self._compliance_setting(2)
            if hasattr(self, "c1_start"):
                current_data["dual_gate"]["c1_start"] = self.c1_start.value()
            if hasattr(self, "c1_end"):
                current_data["dual_gate"]["c1_end"] = self.c1_end.value()
            if hasattr(self, "c1_pts"):
                current_data["dual_gate"]["c1_pts"] = self.c1_pts.value()
                c1_step = abs(self.c1_end.value() - self.c1_start.value()) / (self.c1_pts.value() - 1) if self.c1_pts.value() > 1 else 0.0
                current_data["dual_gate"]["c1_step"] = c1_step
            if hasattr(self, "c2_start"):
                current_data["dual_gate"]["c2_start"] = self.c2_start.value()
            if hasattr(self, "c2_end"):
                current_data["dual_gate"]["c2_end"] = self.c2_end.value()
            if hasattr(self, "c2_pts"):
                current_data["dual_gate"]["c2_pts"] = self.c2_pts.value()
                c2_step = abs(self.c2_end.value() - self.c2_start.value()) / (self.c2_pts.value() - 1) if self.c2_pts.value() > 1 else 0.0
                current_data["dual_gate"]["c2_step"] = c2_step
            if hasattr(self, "sweep_mode_combo"):
                current_data["dual_gate"]["sweep_mode"] = self.sweep_mode_combo.currentText()
            if hasattr(self, "serpentine_box"):
                current_data["dual_gate"]["serpentine"] = self.serpentine_box.isChecked()

            # Purge deprecated root-level keys and legacy c1/c2 compliance keys
            deprecated_keys = [
                "exposure_time", "settling_time", "ramprate1", "ramprate2", "ramp_step", 
                "c1_name", "c2_name", "grating", "wavelength", "target_temp", "num_spectra", 
                "average", "accum_method", "plot_mode", "x_axis_scale", "plot_modes", 
                "b_coord_name", "b_start", "b_end", "b_pts", "v_start", "v_end", "v_pts", 
                "v_fixed", "variable_order", "serpentine", "faraday_comp", "faraday_rot", 
                "end_ramp_zero", "end_ramp_ground", "bv_coeff_a", "bv_coeff_b", "bv_coeff_c", 
                "bv_coeff_d", "pol_exc_vertical", "pol_exc_horizontal", "pol_exc_manual", 
                "pol_coll_vertical", "pol_coll_horizontal", "pol_coll_manual",
                "c1_compliance", "c2_compliance",
                "dg_coeff_a", "dg_coeff_b", "dg_coeff_c", "dg_coeff_d"
            ]
            for key in deprecated_keys:
                current_data.pop(key, None)
                if "dual_gate" in current_data:
                    current_data["dual_gate"].pop("c1_compliance", None)
                    current_data["dual_gate"].pop("c2_compliance", None)
                    current_data["dual_gate"].pop("dg_coeff_a", None)
                    current_data["dual_gate"].pop("dg_coeff_b", None)
                    current_data["dual_gate"].pop("dg_coeff_c", None)
                    current_data["dual_gate"].pop("dg_coeff_d", None)
                
            with open(storage_path, "w") as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Error saving to value_storage: {e}")

    def reload_gui_from_storage(self):
        if getattr(self, "plotter_mode", False):
            return
        self.load_storage()
        
        self.c1_name_edit.blockSignals(True)
        self.c1_name_edit.setText(self.stored_settings.get("c1_name", "Vc1"))
        self.c1_name_edit.blockSignals(False)

        self.c2_name_edit.blockSignals(True)
        self.c2_name_edit.setText(self.stored_settings.get("c2_name", "Vc2"))
        self.c2_name_edit.blockSignals(False)

        self.ch1_ramp_rate.blockSignals(True)
        self.ch1_ramp_rate.setValue(self.stored_settings.get("ramprate1", 0.1))
        self.ch1_ramp_rate.blockSignals(False)

        self.ch2_ramp_rate.blockSignals(True)
        self.ch2_ramp_rate.setValue(self.stored_settings.get("ramprate2", 0.1))
        self.ch2_ramp_rate.blockSignals(False)

        self.exposure_box.blockSignals(True)
        self.exposure_box.setValue(self.stored_settings.get("exposure_time", 0.1))
        self.exposure_box.blockSignals(False)

        self.settling_box.blockSignals(True)
        self.settling_box.setValue(self.stored_settings.get("settling_time", 0.05))
        self.settling_box.blockSignals(False)

        # Compliance
        self.ch1_compliance.blockSignals(True)
        self.ch1_compliance.setValue(self._compliance_setting(1))
        self.ch1_compliance.blockSignals(False)

        self.ch2_compliance.blockSignals(True)
        self.ch2_compliance.setValue(self._compliance_setting(2))
        self.ch2_compliance.blockSignals(False)

        # Sweep range C1
        self.c1_start.blockSignals(True)
        self.c1_start.setValue(self.stored_settings.get("c1_start", -0.005))
        self.c1_start.blockSignals(False)

        self.c1_end.blockSignals(True)
        self.c1_end.setValue(self.stored_settings.get("c1_end", 0.005))
        self.c1_end.blockSignals(False)

        self.c1_pts.blockSignals(True)
        self.c1_pts.setValue(self.stored_settings.get("c1_pts", 21))
        self.c1_pts.blockSignals(False)

        # Sweep range C2
        self.c2_start.blockSignals(True)
        self.c2_start.setValue(self.stored_settings.get("c2_start", -0.005))
        self.c2_start.blockSignals(False)

        self.c2_end.blockSignals(True)
        self.c2_end.setValue(self.stored_settings.get("c2_end", 0.005))
        self.c2_end.blockSignals(False)

        self.c2_pts.blockSignals(True)
        self.c2_pts.setValue(self.stored_settings.get("c2_pts", 21))
        self.c2_pts.blockSignals(False)

        # Basis change coefficients
        self.coeff_a.blockSignals(True)
        self.coeff_a.setValue(self.stored_settings.get("coeff_a", 1.0))
        self.coeff_a.blockSignals(False)

        self.coeff_b.blockSignals(True)
        self.coeff_b.setValue(self.stored_settings.get("coeff_b", 0.0))
        self.coeff_b.blockSignals(False)

        self.coeff_c.blockSignals(True)
        self.coeff_c.setValue(self.stored_settings.get("coeff_c", 0.0))
        self.coeff_c.blockSignals(False)

        self.coeff_d.blockSignals(True)
        self.coeff_d.setValue(self.stored_settings.get("coeff_d", 1.0))
        self.coeff_d.blockSignals(False)

        # Sweep mode and serpentine
        self.sweep_mode_combo.blockSignals(True)
        self.sweep_mode_combo.setCurrentText(self.stored_settings.get("sweep_mode", "Hold C1, sweep C2"))
        self.sweep_mode_combo.blockSignals(False)

        self.serpentine_box.blockSignals(True)
        self.serpentine_box.setChecked(self.stored_settings.get("serpentine", True))
        self.serpentine_box.blockSignals(False)

        self.update_step_calculations()
        self.update_variable_names()
        self.update_coordinates_from_ui()
        self.on_compliance_changed()

    def create_scientific_validator(self, minimum, maximum, decimals):
        validator = QtGui.QDoubleValidator(minimum, maximum, decimals, self)
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        return validator

    def enable_scientific_input(self, spinbox):
        if hasattr(spinbox, "lineEdit") and spinbox.lineEdit() is not None:
            spinbox.lineEdit().setPlaceholderText("e.g. 1e-3")

    # --- UI Building ---
    def setup_ui(self):
        # Initialize Status Bar and Connection Status Indicator (only when not embedded)
        if not getattr(self, "_embed", False):
            status_bar = self.statusBar()
            self.conn_status_lbl = QtWidgets.QLabel()
            if self.station.smu.simulated:
                self.conn_status_lbl.setText("Keysight SMU: SIMULATED (Connection Locked/Failed)  ")
                self.conn_status_lbl.setStyleSheet("color: #d97706; font-weight: bold; font-family: Consolas, monospace;")
            else:
                self.conn_status_lbl.setText("Keysight SMU: CONNECTED (Physical)  ")
                self.conn_status_lbl.setStyleSheet("color: #059669; font-weight: bold; font-family: Consolas, monospace;")
            status_bar.addPermanentWidget(self.conn_status_lbl)
        else:
            # When embedded, still create the label for potential internal use but do not add to status bar
            self.conn_status_lbl = QtWidgets.QLabel()
            if self.station.smu.simulated:
                self.conn_status_lbl.setText("Keysight SMU: SIMULATED (Connection Locked/Failed)  ")
                self.conn_status_lbl.setStyleSheet("color: #d97706; font-weight: bold; font-family: Consolas, monospace;")
            else:
                self.conn_status_lbl.setText("Keysight SMU: CONNECTED (Physical)  ")
                self.conn_status_lbl.setStyleSheet("color: #059669; font-weight: bold; font-family: Consolas, monospace;")
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left Column (Controls Panel) - Broadened to 680 to fit 2-column layout
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(680)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        scroll_area.setWidget(left_widget)
        main_layout.addWidget(scroll_area)

        # Right Column (Tabs Panel)
        self.right_panel = QtWidgets.QTabWidget()
        main_layout.addWidget(self.right_panel, 1)

        # --- TWO-COLUMN CONTROLS LAYOUT (Left and Right Stacked Columns) ---
        primary_columns_layout = QtWidgets.QHBoxLayout()
        primary_columns_layout.setSpacing(10)

        # --- COLUMN 1 (Left Channel + Sweep Controls) ---
        col1_layout = QtWidgets.QVBoxLayout()
        col1_layout.setSpacing(10)
        col1_layout.setContentsMargins(0, 0, 0, 0)

        # Keysight Channel 1 SMU Controls
        ch1_box = QtWidgets.QGroupBox("Keysight channel 1")
        ch1_layout = QtWidgets.QVBoxLayout(ch1_box)
        ch1_layout.setSpacing(8)

        grid1 = QtWidgets.QGridLayout()
        grid1.setSpacing(6)
        
        grid1.addWidget(QtWidgets.QLabel("Output on"), 0, 0)
        self.ch1_out_btn = QtWidgets.QCheckBox()
        self.ch1_out_btn.toggled.connect(self.toggle_ch1_output)
        grid1.addWidget(self.ch1_out_btn, 0, 1)
        
        self.ch1_setpoint_lbl = QtWidgets.QLabel("Setpoint:")
        grid1.addWidget(self.ch1_setpoint_lbl, 1, 0)
        self.ch1_setpoint = QtWidgets.QDoubleSpinBox()
        self.ch1_setpoint.setRange(-210.0, 210.0)
        self.ch1_setpoint.setDecimals(4)
        self.ch1_setpoint.setSingleStep(0.001)
        self.ch1_setpoint.setKeyboardTracking(False)
        self.ch1_setpoint.lineEdit().returnPressed.connect(self.on_ch1_setpoint_enter)
        self.ch1_setpoint.setSuffix(" V")
        grid1.addWidget(self.ch1_setpoint, 1, 1)

        grid1.addWidget(QtWidgets.QLabel("Source mode:"), 2, 0)
        self.ch1_mode = QtWidgets.QComboBox()
        self.ch1_mode.addItems(["VOLT", "CURR"])
        self.ch1_mode.currentTextChanged.connect(self.on_mode_changed)
        grid1.addWidget(self.ch1_mode, 2, 1)

        grid1.addWidget(QtWidgets.QLabel("Ramp rate:"), 3, 0)
        self.ch1_ramp_rate = QtWidgets.QDoubleSpinBox()
        self.ch1_ramp_rate.setRange(1e-6, 100.0)
        self.ch1_ramp_rate.setSingleStep(1e-6)
        self.ch1_ramp_rate.setValue(self.stored_settings.get("ramprate1", 0.1))
        self.ch1_ramp_rate.setKeyboardTracking(False)
        self.ch1_ramp_rate.editingFinished.connect(self.on_ch1_ramping_params_changed)
        self.ch1_ramp_rate.setSuffix(" V/s")
        grid1.addWidget(self.ch1_ramp_rate, 3, 1)

        self.ch1_compliance_lbl = QtWidgets.QLabel("Compliance:")
        grid1.addWidget(self.ch1_compliance_lbl, 4, 0)
        self.ch1_compliance = QtWidgets.QDoubleSpinBox()
        self.ch1_compliance.setRange(1e-9, 3.0)
        self.ch1_compliance.setDecimals(9)
        self.ch1_compliance.setValue(self._compliance_setting(1))
        self.ch1_compliance.setSingleStep(0.001)
        self.ch1_compliance.setKeyboardTracking(False)
        self.ch1_compliance.valueChanged.connect(lambda _value: self._compliance_edits.add(1))
        self.ch1_compliance.lineEdit().textEdited.connect(lambda _text: self._compliance_edits.add(1))
        self.ch1_compliance.editingFinished.connect(lambda: self.on_compliance_changed(1))
        self.ch1_compliance.setSuffix(" A")
        grid1.addWidget(self.ch1_compliance, 4, 1)
        ch1_layout.addLayout(grid1)

        ch1_read_box = QtWidgets.QGroupBox("Last Reading")
        ch1_read_layout = QtWidgets.QVBoxLayout(ch1_read_box)
        ch1_read_layout.setSpacing(4)
        
        self.ch1_live_volt_lbl = QtWidgets.QLabel("Voltage: None")
        self.ch1_live_volt_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch1_read_layout.addWidget(self.ch1_live_volt_lbl)
        
        self.ch1_live_curr_lbl = QtWidgets.QLabel("Current: None")
        self.ch1_live_curr_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch1_read_layout.addWidget(self.ch1_live_curr_lbl)
        
        btn_row1 = QtWidgets.QHBoxLayout()
        btn_row1.setSpacing(6)
        btn_row1.setContentsMargins(0, 0, 0, 0)
        
        self.ch1_read_btn = QtWidgets.QPushButton("Read")
        self.ch1_read_btn.clicked.connect(self.read_channel_one)
        btn_row1.addWidget(self.ch1_read_btn)
        
        self.ch1_stop_btn = QtWidgets.QPushButton("Stop Ramp")
        self.ch1_stop_btn.setEnabled(False)
        self.ch1_stop_btn.clicked.connect(self.on_ch1_stop_clicked)
        btn_row1.addWidget(self.ch1_stop_btn)
        ch1_read_layout.addLayout(btn_row1)
        ch1_layout.addWidget(ch1_read_box)
        col1_layout.addWidget(ch1_box)

        # Control Variable 1 (Sweep settings)
        c1_default_name = self.stored_settings.get("c1_name", "Virtual Gate 1")
        self.ctrl_c1_box = QtWidgets.QGroupBox(f"{c1_default_name}")
        ctrl_c1_layout = QtWidgets.QVBoxLayout(self.ctrl_c1_box)
        
        c1_grid = QtWidgets.QGridLayout()
        c1_grid.setSpacing(6)
        
        c1_grid.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
        self.c1_start = QtWidgets.QDoubleSpinBox()
        self.c1_start.setRange(-210.0, 210.0)
        self.c1_start.setValue(self.stored_settings.get("c1_start", -0.005))
        self.c1_start.setDecimals(4)
        self.c1_start.setKeyboardTracking(False)
        c1_grid.addWidget(self.c1_start, 0, 1)

        c1_grid.addWidget(QtWidgets.QLabel("End:"), 1, 0)
        self.c1_end = QtWidgets.QDoubleSpinBox()
        self.c1_end.setRange(-210.0, 210.0)
        self.c1_end.setValue(self.stored_settings.get("c1_end", 0.005))
        self.c1_end.setDecimals(4)
        self.c1_end.setKeyboardTracking(False)
        c1_grid.addWidget(self.c1_end, 1, 1)

        c1_grid.addWidget(QtWidgets.QLabel("No. points:"), 2, 0)
        self.c1_pts = ScientificPointSpinBox()
        self.c1_pts.setRange(1, 1000)
        self.c1_pts.setValue(self.stored_settings.get("c1_pts", 21))
        self.c1_pts.setKeyboardTracking(True)
        self.c1_pts.lineEdit().setPlaceholderText("e.g. 1e2")
        c1_grid.addWidget(self.c1_pts, 2, 1)

        c1_grid.addWidget(QtWidgets.QLabel("Step size:"), 3, 0)
        self.c1_step_lbl = QtWidgets.QLabel("0.0 V")
        c1_grid.addWidget(self.c1_step_lbl, 3, 1)
        
        ctrl_c1_layout.addLayout(c1_grid)
        col1_layout.addWidget(self.ctrl_c1_box)
        primary_columns_layout.addLayout(col1_layout)

        # --- COLUMN 2 (Right Channel + Sweep Controls) ---
        col2_layout = QtWidgets.QVBoxLayout()
        col2_layout.setSpacing(10)
        col2_layout.setContentsMargins(0, 0, 0, 0)

        # Keysight Channel 2 SMU Controls
        ch2_box = QtWidgets.QGroupBox("Keysight channel 2")
        ch2_layout = QtWidgets.QVBoxLayout(ch2_box)
        ch2_layout.setSpacing(8)

        grid2 = QtWidgets.QGridLayout()
        grid2.setSpacing(6)
        
        grid2.addWidget(QtWidgets.QLabel("Output on"), 0, 0)
        self.ch2_out_btn = QtWidgets.QCheckBox()
        self.ch2_out_btn.toggled.connect(self.toggle_ch2_output)
        grid2.addWidget(self.ch2_out_btn, 0, 1)
        
        self.ch2_setpoint_lbl = QtWidgets.QLabel("Setpoint:")
        grid2.addWidget(self.ch2_setpoint_lbl, 1, 0)
        self.ch2_setpoint = QtWidgets.QDoubleSpinBox()
        self.ch2_setpoint.setRange(-210.0, 210.0)
        self.ch2_setpoint.setDecimals(4)
        self.ch2_setpoint.setSingleStep(0.001)
        self.ch2_setpoint.setKeyboardTracking(False)
        self.ch2_setpoint.lineEdit().returnPressed.connect(self.on_ch2_setpoint_enter)
        self.ch2_setpoint.setSuffix(" V")
        grid2.addWidget(self.ch2_setpoint, 1, 1)

        grid2.addWidget(QtWidgets.QLabel("Source mode:"), 2, 0)
        self.ch2_mode = QtWidgets.QComboBox()
        self.ch2_mode.addItems(["VOLT", "CURR"])
        self.ch2_mode.currentTextChanged.connect(self.on_mode_changed)
        grid2.addWidget(self.ch2_mode, 2, 1)

        grid2.addWidget(QtWidgets.QLabel("Ramp rate:"), 3, 0)
        self.ch2_ramp_rate = QtWidgets.QDoubleSpinBox()
        self.ch2_ramp_rate.setRange(1e-6, 100.0)
        self.ch2_ramp_rate.setSingleStep(1e-6)
        self.ch2_ramp_rate.setValue(self.stored_settings.get("ramprate2", 0.1))
        self.ch2_ramp_rate.setKeyboardTracking(False)
        self.ch2_ramp_rate.editingFinished.connect(self.on_ch2_ramping_params_changed)
        self.ch2_ramp_rate.setSuffix(" V/s")
        grid2.addWidget(self.ch2_ramp_rate, 3, 1)

        self.ch2_compliance_lbl = QtWidgets.QLabel("Compliance:")
        grid2.addWidget(self.ch2_compliance_lbl, 4, 0)
        self.ch2_compliance = QtWidgets.QDoubleSpinBox()
        self.ch2_compliance.setRange(1e-9, 3.0)
        self.ch2_compliance.setDecimals(9)
        self.ch2_compliance.setValue(self._compliance_setting(2))
        self.ch2_compliance.setSingleStep(0.001)
        self.ch2_compliance.setKeyboardTracking(False)
        self.ch2_compliance.valueChanged.connect(lambda _value: self._compliance_edits.add(2))
        self.ch2_compliance.lineEdit().textEdited.connect(lambda _text: self._compliance_edits.add(2))
        self.ch2_compliance.editingFinished.connect(lambda: self.on_compliance_changed(2))
        self.ch2_compliance.setSuffix(" A")
        grid2.addWidget(self.ch2_compliance, 4, 1)
        ch2_layout.addLayout(grid2)

        ch2_read_box = QtWidgets.QGroupBox("Last Reading")
        ch2_read_layout = QtWidgets.QVBoxLayout(ch2_read_box)
        ch2_read_layout.setSpacing(4)
        
        self.ch2_live_volt_lbl = QtWidgets.QLabel("Voltage: None")
        self.ch2_live_volt_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch2_read_layout.addWidget(self.ch2_live_volt_lbl)
        
        self.ch2_live_curr_lbl = QtWidgets.QLabel("Current: None")
        self.ch2_live_curr_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch2_read_layout.addWidget(self.ch2_live_curr_lbl)
        
        btn_row2 = QtWidgets.QHBoxLayout()
        btn_row2.setSpacing(6)
        btn_row2.setContentsMargins(0, 0, 0, 0)
        
        self.ch2_read_btn = QtWidgets.QPushButton("Read")
        self.ch2_read_btn.clicked.connect(self.read_channel_two)
        btn_row2.addWidget(self.ch2_read_btn)
        
        self.ch2_stop_btn = QtWidgets.QPushButton("Stop Ramp")
        self.ch2_stop_btn.setEnabled(False)
        self.ch2_stop_btn.clicked.connect(self.on_ch2_stop_clicked)
        btn_row2.addWidget(self.ch2_stop_btn)
        ch2_read_layout.addLayout(btn_row2)
        ch2_layout.addWidget(ch2_read_box)
        col2_layout.addWidget(ch2_box)

        # Control Variable 2 (Sweep settings)
        c2_default_name = self.stored_settings.get("c2_name", "Virtual Gate 2")
        self.ctrl_c2_box = QtWidgets.QGroupBox(f"{c2_default_name}")
        ctrl_c2_layout = QtWidgets.QVBoxLayout(self.ctrl_c2_box)
        
        c2_grid = QtWidgets.QGridLayout()
        c2_grid.setSpacing(6)
        
        c2_grid.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
        self.c2_start = QtWidgets.QDoubleSpinBox()
        self.c2_start.setRange(-210.0, 210.0)
        self.c2_start.setValue(self.stored_settings.get("c2_start", -0.005))
        self.c2_start.setDecimals(4)
        self.c2_start.setKeyboardTracking(False)
        c2_grid.addWidget(self.c2_start, 0, 1)

        c2_grid.addWidget(QtWidgets.QLabel("End:"), 1, 0)
        self.c2_end = QtWidgets.QDoubleSpinBox()
        self.c2_end.setRange(-210.0, 210.0)
        self.c2_end.setValue(self.stored_settings.get("c2_end", 0.005))
        self.c2_end.setDecimals(4)
        self.c2_end.setKeyboardTracking(False)
        c2_grid.addWidget(self.c2_end, 1, 1)

        c2_grid.addWidget(QtWidgets.QLabel("No. points:"), 2, 0)
        self.c2_pts = ScientificPointSpinBox()
        self.c2_pts.setRange(1, 1000)
        self.c2_pts.setValue(self.stored_settings.get("c2_pts", 21))
        self.c2_pts.setKeyboardTracking(True)
        self.c2_pts.lineEdit().setPlaceholderText("e.g. 1e2")
        c2_grid.addWidget(self.c2_pts, 2, 1)

        c2_grid.addWidget(QtWidgets.QLabel("Step size:"), 3, 0)
        self.c2_step_lbl = QtWidgets.QLabel("0.0 V")
        c2_grid.addWidget(self.c2_step_lbl, 3, 1)
        
        ctrl_c2_layout.addLayout(c2_grid)
        col2_layout.addWidget(self.ctrl_c2_box)
        primary_columns_layout.addLayout(col2_layout)

        left_layout.addLayout(primary_columns_layout)

        # Bind events to update step sizes and plot scaling dynamically
        for box in (self.c1_start, self.c1_end, self.c1_pts):
            box.editingFinished.connect(self.on_sweep_range_changed)
        for box in (self.c2_start, self.c2_end, self.c2_pts):
            box.editingFinished.connect(self.on_sweep_range_changed)
        self.update_step_calculations()

        # --- ROW 3: Gate Variable Matrix (full width) ---
        gate_box = QtWidgets.QGroupBox("Gate Variable")
        gate_layout = QtWidgets.QVBoxLayout(gate_box)
        gate_layout.setSpacing(10)

        # Name fields
        names_layout = QtWidgets.QGridLayout()
        names_layout.setSpacing(10)
        names_layout.addWidget(QtWidgets.QLabel("Virtual Gate 1 name:"), 0, 0)
        self.c1_name_edit = QtWidgets.QLineEdit(c1_default_name)
        names_layout.addWidget(self.c1_name_edit, 0, 1)
        
        names_layout.addWidget(QtWidgets.QLabel("Virtual Gate 2 name:"), 1, 0)
        self.c2_name_edit = QtWidgets.QLineEdit(c2_default_name)
        names_layout.addWidget(self.c2_name_edit, 1, 1)
        gate_layout.addLayout(names_layout)

        # 2x2 grid for formulas
        matrix_grid = QtWidgets.QGridLayout()
        matrix_grid.setSpacing(6)

        # Row 1: V1 = coeff_a * C1 + coeff_b * C2
        self.v1_formula_label = QtWidgets.QLabel("Voltage 1 =")
        matrix_grid.addWidget(self.v1_formula_label, 0, 0)

        self.coeff_a = QtWidgets.QDoubleSpinBox()
        self.coeff_a.setValue(self.stored_settings.get("coeff_a", 1.0))
        self.coeff_a.setRange(-100, 100)
        self.coeff_a.setDecimals(4)
        self.coeff_a.setSingleStep(0.01)
        self.coeff_a.setKeyboardTracking(False)
        matrix_grid.addWidget(self.coeff_a, 0, 1)

        self.v1_c1_label = QtWidgets.QLabel(f" * {c1_default_name}  + ")
        matrix_grid.addWidget(self.v1_c1_label, 0, 2)

        self.coeff_b = QtWidgets.QDoubleSpinBox()
        self.coeff_b.setValue(self.stored_settings.get("coeff_b", 0.0))
        self.coeff_b.setRange(-100, 100)
        self.coeff_b.setDecimals(4)
        self.coeff_b.setSingleStep(0.01)
        self.coeff_b.setKeyboardTracking(False)
        matrix_grid.addWidget(self.coeff_b, 0, 3)

        self.v1_c2_label = QtWidgets.QLabel(f" * {c2_default_name}")
        matrix_grid.addWidget(self.v1_c2_label, 0, 4)

        # Row 2: V2 = coeff_c * C1 + coeff_d * C2
        self.v2_formula_label = QtWidgets.QLabel("Voltage 2 =")
        matrix_grid.addWidget(self.v2_formula_label, 1, 0)

        self.coeff_c = QtWidgets.QDoubleSpinBox()
        self.coeff_c.setValue(self.stored_settings.get("coeff_c", 0.0))
        self.coeff_c.setRange(-100, 100)
        self.coeff_c.setDecimals(4)
        self.coeff_c.setSingleStep(0.01)
        self.coeff_c.setKeyboardTracking(False)
        matrix_grid.addWidget(self.coeff_c, 1, 1)

        self.v2_c1_label = QtWidgets.QLabel(f" * {c1_default_name}  + ")
        matrix_grid.addWidget(self.v2_c1_label, 1, 2)

        self.coeff_d = QtWidgets.QDoubleSpinBox()
        self.coeff_d.setValue(self.stored_settings.get("coeff_d", 1.0))
        self.coeff_d.setRange(-100, 100)
        self.coeff_d.setDecimals(4)
        self.coeff_d.setSingleStep(0.01)
        self.coeff_d.setKeyboardTracking(False)
        matrix_grid.addWidget(self.coeff_d, 1, 3)

        self.v2_c2_label = QtWidgets.QLabel(f" * {c2_default_name}")
        matrix_grid.addWidget(self.v2_c2_label, 1, 4)

        gate_layout.addLayout(matrix_grid)
        self.det_label = QtWidgets.QLabel()
        self.det_label.setWordWrap(True)
        gate_layout.addWidget(self.det_label)
        left_layout.addWidget(gate_box)

        # Connect name change events to update layouts and axes
        self.c1_name_edit.textChanged.connect(self.update_variable_names)
        self.c2_name_edit.textChanged.connect(self.update_variable_names)
        self.c1_name_edit.editingFinished.connect(self.save_storage)
        self.c2_name_edit.editingFinished.connect(self.save_storage)

        # Connect coefficient updates to automatically recalculate coordinates
        self.coeff_a.editingFinished.connect(self.on_sweep_range_changed)
        self.coeff_b.editingFinished.connect(self.on_sweep_range_changed)
        self.coeff_c.editingFinished.connect(self.on_sweep_range_changed)
        self.coeff_d.editingFinished.connect(self.on_sweep_range_changed)

        # --- ROW 4: Other Settings & Measurement Control next to each other ---
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(10)

        # Left bottom column (Other Settings & Plot Options stacked)
        left_bottom_layout = QtWidgets.QVBoxLayout()
        left_bottom_layout.setSpacing(10)
        left_bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Other Settings Box
        other_box = QtWidgets.QGroupBox("Other")
        other_layout = QtWidgets.QVBoxLayout(other_box)
        other_layout.setSpacing(6)
        
        other_layout.addWidget(QtWidgets.QLabel("Variable order:"))
        self.sweep_mode_combo = QtWidgets.QComboBox()
        self.sweep_mode_combo.addItems(["Hold C1, sweep C2", "Hold C2, sweep C1"])
        self.sweep_mode_combo.setCurrentText(self.stored_settings.get("sweep_mode", "Hold C1, sweep C2"))
        self.sweep_mode_combo.currentTextChanged.connect(self.on_sweep_range_changed)
        other_layout.addWidget(self.sweep_mode_combo)

        self.serpentine_box = QtWidgets.QCheckBox("Serpentine")
        self.serpentine_box.setChecked(self.stored_settings.get("serpentine", True))
        self.serpentine_box.stateChanged.connect(self.save_storage)
        other_layout.addWidget(self.serpentine_box)

        # Timing Grid
        timing_grid = QtWidgets.QGridLayout()
        timing_grid.setSpacing(6)
        timing_grid.addWidget(QtWidgets.QLabel("Exposure:"), 0, 0)
        self.exposure_box = QtWidgets.QDoubleSpinBox()
        self.exposure_box.setRange(0.001, 10.0)
        self.exposure_box.setDecimals(3)
        self.exposure_box.setValue(self.stored_settings.get("exposure_time", 0.1))
        self.exposure_box.setKeyboardTracking(False)
        self.exposure_box.editingFinished.connect(self.on_timing_params_changed)
        self.exposure_box.setSuffix(" s")
        timing_grid.addWidget(self.exposure_box, 0, 1)

        timing_grid.addWidget(QtWidgets.QLabel("Settling time:"), 1, 0)
        self.settling_box = QtWidgets.QDoubleSpinBox()
        self.settling_box.setRange(0.0, 10.0)
        self.settling_box.setDecimals(3)
        self.settling_box.setValue(self.stored_settings.get("settling_time", 0.05))
        self.settling_box.setKeyboardTracking(False)
        self.settling_box.editingFinished.connect(self.on_timing_params_changed)
        self.settling_box.setSuffix(" s")
        timing_grid.addWidget(self.settling_box, 1, 1)
        other_layout.addLayout(timing_grid)
        left_bottom_layout.addWidget(other_box)



        bottom_row.addLayout(left_bottom_layout, 0)

        # Enable scientific notation entry on numeric spin boxes
        for spin in (
            self.ch1_setpoint,
            self.ch1_ramp_rate,
            self.ch1_compliance,
            self.ch2_setpoint,
            self.ch2_ramp_rate,
            self.ch2_compliance,
            self.c1_start,
            self.c1_end,
            self.c2_start,
            self.c2_end,
            self.coeff_a,
            self.coeff_b,
            self.coeff_c,
            self.coeff_d,
            self.exposure_box,
            self.settling_box,
        ):
            self.enable_scientific_input(spin)

        # Measurement Control Box
        meas_box = QtWidgets.QGroupBox("Measurement control")
        meas_layout = QtWidgets.QVBoxLayout(meas_box)
        meas_layout.setSpacing(6)
        
        self.begin_btn = QtWidgets.QPushButton("► Begin")
        self.begin_btn.setObjectName("begin_btn")
        self.begin_btn.clicked.connect(self.begin_sweep)
        meas_layout.addWidget(self.begin_btn)

        self.pause_btn = QtWidgets.QPushButton("⏸ Interrupt")
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.on_interrupt_clicked)
        meas_layout.addWidget(self.pause_btn)

        self.reset_btn = QtWidgets.QPushButton("↺ Reset")
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.clicked.connect(self.reset_sweep)
        meas_layout.addWidget(self.reset_btn)

        self.save_btn = QtWidgets.QPushButton("💾 Save Dataset")
        self.save_btn.setObjectName("save_btn")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_sweep_data)
        meas_layout.addWidget(self.save_btn)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        meas_layout.addWidget(self.progress_bar)

        self.lbl_expected_end = QtWidgets.QLabel("Expected End: --:--:--")
        self.lbl_expected_end.setStyleSheet("color: #0F172A; font-weight: bold;")
        meas_layout.addWidget(self.lbl_expected_end)

        self.status_lbl = QtWidgets.QLabel("Status: Idle")
        self.status_lbl.setStyleSheet("color: #0F172A; font-style: italic; font-weight: bold;")
        meas_layout.addWidget(self.status_lbl)

        self.lbl_target_vc1 = QtWidgets.QLabel("V1 = --‑‑ V")
        self.lbl_target_vc1.setStyleSheet("color: #0F172A; font-weight: bold;")
        meas_layout.addWidget(self.lbl_target_vc1)

        self.lbl_target_vc2 = QtWidgets.QLabel("V2 = --‑‑ V")
        self.lbl_target_vc2.setStyleSheet("color: #0F172A; font-weight: bold;")
        meas_layout.addWidget(self.lbl_target_vc2)

        self.lbl_current_b = QtWidgets.QLabel("B = --‑‑ T")
        self.lbl_current_b.setStyleSheet("color: #0F172A; font-weight: bold;")
        meas_layout.addWidget(self.lbl_current_b)

        bottom_row.addWidget(meas_box, 1)
        left_layout.addLayout(bottom_row)

        # --- RIGHT TAB PANEL SECTIONS ---
        self.tab1 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab1, "Optical signal")
        self.setup_tab1()

        self.tab2 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab2, "IV channel 1")
        self.setup_tab2()

        self.tab3 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab3, "IV channel 2")
        self.setup_tab3()

        self.btn_view_control = QtWidgets.QPushButton("View Control ▾")
        self.btn_view_control.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 0px;
                padding: 6px 12px;
                font-weight: bold;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        self.right_panel.setCornerWidget(self.btn_view_control, QtCore.Qt.Corner.TopRightCorner)
        self.btn_view_control.clicked.connect(self.toggle_view_control_popup)
        self.right_panel.currentChanged.connect(self.on_tab_changed)

        # Update initial variable names
        self.update_variable_names()
        self.update_coordinates_from_ui()
        self.reset_plot_scaling(reset_spectrum=False)
    def setup_tab1(self):
        """Setup 2x2 grid of plots for Gating vs Energy Spectra in Tab 1"""
        layout = QtWidgets.QGridLayout(self.tab1)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # 1. Gate Map Plot (Top Left)
        self.gatemap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="2D Gate Map (Integrated Counts)")
        self.gatemap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.gatemap_win.getViewBox().enableAutoRange(enable=False)
        self.gatemap_img = pg.ImageItem()
        self.gatemap_img.setOpts(axisOrder='row-major')
        self.gatemap_win.addItem(self.gatemap_img)
        self.gatemap_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.gatemap_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        
        # Color bar / Histogram
        self.gatemap_colorbar = self.setup_histogram_lut(self.gatemap_win, self.gatemap_img, interactive=False)
        
        # Crosshair lines
        self.cross1_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.cross1_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.gatemap_win.addItem(self.cross1_v)
        self.gatemap_win.addItem(self.cross1_h)
        self.cross1_v.sigPositionChanged.connect(self.on_crosshair_moved)
        self.cross1_h.sigPositionChanged.connect(self.on_crosshair_moved)
        
        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
        viewbox1 = self.gatemap_win.getViewBox()
        viewbox1.originalMouseClickEvent = viewbox1.mouseClickEvent
        def gatemap_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox1.mapSceneToView(ev.scenePos())
                c1_val = mouse_point.x()
                c2_val = mouse_point.y()
                self.update_crosshair_values(c1_val, c2_val, self.selected_energy)
                self.refresh_slice_plots(c1_val, c2_val)
                ev.accept()
            else:
                viewbox1.originalMouseClickEvent(ev)
        viewbox1.mouseClickEvent = gatemap_click_event

        layout.addWidget(self.gatemap_win, 0, 0)

        # 2. Vertical Slice (Top Right)
        self.vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Vertical Slice spectra")
        self.vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.vslice_win.getViewBox().enableAutoRange(enable=False)
        self.vslice_img = pg.ImageItem()
        self.vslice_img.setOpts(axisOrder='row-major')
        self.vslice_win.addItem(self.vslice_img)
        self.vslice_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.vslice_win.setLabel('bottom', 'Energy', units='eV')
        
        self.vslice_colorbar = self.setup_histogram_lut(self.vslice_win, self.vslice_img, interactive=True)

        # Draggable crosshair lines
        self.vslice_cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.vslice_cross_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        # Add crosshair lines to Vertical Slice plot (Plot B)
        self.vslice_win.addItem(self.vslice_cross_v)
        self.vslice_win.addItem(self.vslice_cross_h)
        self.vslice_cross_v.sigPositionChanged.connect(self.on_vslice_crosshair_moved)
        self.vslice_cross_h.sigPositionChanged.connect(self.on_vslice_crosshair_moved)

        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
        viewbox_vslice = self.vslice_win.getViewBox()
        viewbox_vslice.originalMouseClickEvent = viewbox_vslice.mouseClickEvent
        def vslice_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox_vslice.mapSceneToView(ev.scenePos())
                c1_val = self.cross1_v.value()
                c2_val = mouse_point.y()
                energy_val = self.selected_energy # Keep current selected energy instead of changing it
                self.update_crosshair_values(c1_val, c2_val, energy_val)
                self.refresh_slice_plots(c1_val, c2_val)
                ev.accept()
            else:
                viewbox_vslice.originalMouseClickEvent(ev)
        viewbox_vslice.mouseClickEvent = vslice_click_event

        layout.addWidget(self.vslice_win, 0, 1)

        # 3. Horizontal Slice (Bottom Left)
        self.hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Horizontal Slice spectra")
        self.hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.hslice_win.getViewBox().enableAutoRange(enable=False)
        self.hslice_img = pg.ImageItem()
        self.hslice_img.setOpts(axisOrder='row-major')
        self.hslice_win.addItem(self.hslice_img)
        self.hslice_win.setLabel('left', 'Virtual Gate C1', units='V')
        self.hslice_win.setLabel('bottom', 'Energy', units='eV')

        self.hslice_colorbar = self.setup_histogram_lut(self.hslice_win, self.hslice_img, interactive=True)

        # Draggable crosshair lines
        self.hslice_cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.hslice_cross_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.hslice_win.addItem(self.hslice_cross_v)
        self.hslice_win.addItem(self.hslice_cross_h)
        self.hslice_cross_v.sigPositionChanged.connect(self.on_hslice_crosshair_moved)
        self.hslice_cross_h.sigPositionChanged.connect(self.on_hslice_crosshair_moved)

        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
        viewbox_hslice = self.hslice_win.getViewBox()
        viewbox_hslice.originalMouseClickEvent = viewbox_hslice.mouseClickEvent
        def hslice_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox_hslice.mapSceneToView(ev.scenePos())
                c1_val = mouse_point.y()
                c2_val = self.cross1_h.value()
                energy_val = mouse_point.x()
                self.update_crosshair_values(c1_val, c2_val, energy_val)
                self.refresh_slice_plots(c1_val, c2_val)
                ev.accept()
            else:
                viewbox_hslice.originalMouseClickEvent(ev)
        viewbox_hslice.mouseClickEvent = hslice_click_event

        layout.addWidget(self.hslice_win, 1, 0)

        # 4. Spectrum at Cross (Bottom Right)
        self.spectrum_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="CCD Spectrum at Gating Crosshair")
        self.spectrum_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.spectrum_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.spectrum_win.getViewBox().enableAutoRange(axis='y', enable=False)
        self.spectrum_curve = self.spectrum_win.plot(
            pen=pg.mkPen('#000000', width=2)
        )
        self.spectrum_win.setLabel('left', 'Counts')
        self.spectrum_win.setLabel('bottom', 'Energy', units='eV')
        self.spectrum_win.showGrid(x=False, y=False)

        # Draggable vertical lines (spectrum_v1 and spectrum_v2) to highlight region (X-axis integration window)
        self.spectrum_v_region = pg.LinearRegionItem(
            orientation='vertical',
            values=(2.4, 2.6),
            brush=pg.mkBrush(16, 185, 129, 30), # green transparent fill
            pen=pg.mkPen('#10b981', width=1.5)  # continuous green line
        )
        self.spectrum_win.addItem(self.spectrum_v_region, ignoreBounds=True)
        self.spectrum_v1 = self.spectrum_v_region.lines[0]
        self.spectrum_v2 = self.spectrum_v_region.lines[1]
        self.spectrum_v_region.sigRegionChanged.connect(self.on_integration_region_changed)

        # Draggable horizontal lines (spectrum_h1 and spectrum_h2) to highlight region (Y-axis counts window)
        self.spectrum_h_region = pg.LinearRegionItem(
            orientation='horizontal',
            values=(1000.0, 5000.0),
            brush=pg.mkBrush(16, 185, 129, 30), # green transparent fill
            pen=pg.mkPen('#10b981', width=1.5)  # continuous green line
        )
        self.spectrum_win.addItem(self.spectrum_h_region, ignoreBounds=True)
        self.spectrum_h1 = self.spectrum_h_region.lines[0]
        self.spectrum_h2 = self.spectrum_h_region.lines[1]
        self.spectrum_h_region.sigRegionChanged.connect(self.on_integration_region_changed)

        # Position crosshair markers on the spectrum plot
        self.spectrum_cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#10b981', width=2.5))
        self.spectrum_cross_v.setValue(self.selected_energy)
        # Do not add the energy crosshair vertical line to the spectrum plot per user request
        # self.spectrum_win.addItem(self.spectrum_cross_v)
        # self.spectrum_cross_v.sigPositionChanged.connect(self.on_spectrum_crosshair_moved)


        layout.addWidget(self.spectrum_win, 1, 1)

    def setup_tab2(self):
        """Setup 2x2 layout for Ch1 Current vs coordinates"""
        layout = QtWidgets.QGridLayout(self.tab2)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # Heatmap (Top Left)
        self.t2_heatmap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Top Gate Current Heatmap")
        self.t2_heatmap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_heatmap_win.getViewBox().enableAutoRange(enable=False)
        self.t2_heatmap_img = pg.ImageItem()
        self.t2_heatmap_img.setOpts(axisOrder='row-major')
        self.t2_heatmap_win.addItem(self.t2_heatmap_img)
        self.t2_heatmap_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.t2_heatmap_win.setLabel('bottom', 'Virtual Gate C1', units='V')

        self.t2_colorbar = self.setup_histogram_lut(self.t2_heatmap_win, self.t2_heatmap_img, interactive=True)

        # Draggable crosshair
        self.cross2_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#ea580c', width=2))
        self.cross2_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#ea580c', width=2))
        self.t2_heatmap_win.addItem(self.cross2_v)
        self.t2_heatmap_win.addItem(self.cross2_h)
        self.cross2_v.sigPositionChanged.connect(self.on_tab2_crosshair_moved)
        self.cross2_h.sigPositionChanged.connect(self.on_tab2_crosshair_moved)
        
        # Monkey-patch ViewBox to intercept left clicks
        viewbox2 = self.t2_heatmap_win.getViewBox()
        viewbox2.originalMouseClickEvent = viewbox2.mouseClickEvent
        def t2_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox2.mapSceneToView(ev.scenePos())
                self.cross2_v.blockSignals(True)
                self.cross2_h.blockSignals(True)
                self.cross2_v.setValue(mouse_point.x())
                self.cross2_h.setValue(mouse_point.y())
                self.cross2_v.blockSignals(False)
                self.cross2_h.blockSignals(False)
                self.on_tab2_crosshair_moved()
                ev.accept()
            else:
                viewbox2.originalMouseClickEvent(ev)
        viewbox2.mouseClickEvent = t2_click_event

        layout.addWidget(self.t2_heatmap_win, 0, 0)

        # Vertical Slice (Top Right)
        self.t2_vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C1")
        self.t2_vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_vslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t2_vslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t2_vslice_curve = self.t2_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t2_vslice_win.setLabel('left', 'Current', units='A')
        self.t2_vslice_win.setLabel('bottom', 'Virtual Gate C2', units='V')
        self.t2_vslice_win.showGrid(x=False, y=False)
        layout.addWidget(self.t2_vslice_win, 0, 1)

        # Horizontal Slice (Bottom Left)
        self.t2_hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C2")
        self.t2_hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_hslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t2_hslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t2_hslice_curve = self.t2_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t2_hslice_win.setLabel('left', 'Current', units='A')
        self.t2_hslice_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        self.t2_hslice_win.showGrid(x=False, y=False)
        layout.addWidget(self.t2_hslice_win, 1, 0)

        # Empty (Bottom Right)
        dummy = QtWidgets.QWidget()
        layout.addWidget(dummy, 1, 1)

    def setup_tab3(self):
        """Setup 2x2 layout for Ch2 Current vs coordinates"""
        layout = QtWidgets.QGridLayout(self.tab3)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # Heatmap (Top Left)
        self.t3_heatmap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Bottom Gate Current Heatmap")
        self.t3_heatmap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_heatmap_win.getViewBox().enableAutoRange(enable=False)
        self.t3_heatmap_img = pg.ImageItem()
        self.t3_heatmap_img.setOpts(axisOrder='row-major')
        self.t3_heatmap_win.addItem(self.t3_heatmap_img)
        self.t3_heatmap_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.t3_heatmap_win.setLabel('bottom', 'Virtual Gate C1', units='V')

        self.t3_colorbar = self.setup_histogram_lut(self.t3_heatmap_win, self.t3_heatmap_img, interactive=True)

        # Draggable crosshair
        self.cross3_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#16a34a', width=2))
        self.cross3_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#16a34a', width=2))
        self.t3_heatmap_win.addItem(self.cross3_v)
        self.t3_heatmap_win.addItem(self.cross3_h)
        self.cross3_v.sigPositionChanged.connect(self.on_tab3_crosshair_moved)
        self.cross3_h.sigPositionChanged.connect(self.on_tab3_crosshair_moved)
        
        # Monkey-patch ViewBox to intercept left clicks
        viewbox3 = self.t3_heatmap_win.getViewBox()
        viewbox3.originalMouseClickEvent = viewbox3.mouseClickEvent
        def t3_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox3.mapSceneToView(ev.scenePos())
                self.cross3_v.blockSignals(True)
                self.cross3_h.blockSignals(True)
                self.cross3_v.setValue(mouse_point.x())
                self.cross3_h.setValue(mouse_point.y())
                self.cross3_v.blockSignals(False)
                self.cross3_h.blockSignals(False)
                self.on_tab3_crosshair_moved()
                ev.accept()
            else:
                viewbox3.originalMouseClickEvent(ev)
        viewbox3.mouseClickEvent = t3_click_event

        layout.addWidget(self.t3_heatmap_win, 0, 0)

        # Vertical Slice (Top Right)
        self.t3_vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C1")
        self.t3_vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_vslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t3_vslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t3_vslice_curve = self.t3_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t3_vslice_win.setLabel('left', 'Current', units='A')
        self.t3_vslice_win.setLabel('bottom', 'Virtual Gate C2', units='V')
        self.t3_vslice_win.showGrid(x=False, y=False)
        layout.addWidget(self.t3_vslice_win, 0, 1)

        # Horizontal Slice (Bottom Left)
        self.t3_hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C2")
        self.t3_hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_hslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t3_hslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t3_hslice_curve = self.t3_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t3_hslice_win.setLabel('left', 'Current', units='A')
        self.t3_hslice_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        self.t3_hslice_win.showGrid(x=False, y=False)
        layout.addWidget(self.t3_hslice_win, 1, 0)

        # Empty (Bottom Right)
        dummy = QtWidgets.QWidget()
        layout.addWidget(dummy, 1, 1)

    def apply_theme(self):
        from theme import SCIENTIFIC_THEME_QSS
        self.setStyleSheet(SCIENTIFIC_THEME_QSS)

        # Hide up/down arrows on all spinboxes
        for spin in self.findChildren(QtWidgets.QAbstractSpinBox):
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)

        # Apply unified dark styling and gridlines to all PlotWidgets
        import pyqtgraph as pg
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, pg.PlotWidget):
                configure_plot_axes(attr)
                attr.setBackground('#ffffff')
                attr.getAxis('left').setPen(pg.mkPen('#334155'))
                attr.getAxis('bottom').setPen(pg.mkPen('#334155'))
                attr.getAxis('left').setTextPen(pg.mkPen('#334155'))
                attr.getAxis('bottom').setTextPen(pg.mkPen('#334155'))
                attr.showGrid(x=False, y=False)

    def pause_status_polling(self):
        """Temporarily pause the live status worker and wait to ensure no active VISA query is running."""
        was_paused = getattr(self.station, "live_status_paused", False)
        self.station.live_status_paused = True
        if hasattr(self, "live_status_worker") and self.live_status_worker:
            self.live_status_worker.pause()
        time.sleep(0.06) # Wait 60ms to let any active query finish
        return was_paused

    def resume_status_polling(self, was_paused):
        """Restore the live status worker state."""
        if not was_paused:
            self.station.live_status_paused = False
            if hasattr(self, "live_status_worker") and self.live_status_worker:
                self.live_status_worker.resume()

    # --- Live Parameter Readout ---
    # --- Live Parameter Readout (On Demand) ---
    def read_channel_one(self):
        if getattr(self, "plotter_mode", False):
            return
        was_paused = self.pause_status_polling()
        try:
            smu = self.station.smu
            v = smu.ch1.volt()
            i = smu.ch1.measure_curr()
            try:
                smu.return_to_local()
            except Exception:
                pass
            
            def fmt(curr):
                if abs(curr) < 1e-6:
                    return f"{curr*1e9:.2f} nA"
                elif abs(curr) < 1e-3:
                    return f"{curr*1e6:.2f} uA"
                else:
                    return f"{curr*1e3:.2f} mA"

            self.ch1_live_volt_lbl.setText(f"Voltage: {v:.4f} V")
            self.ch1_live_curr_lbl.setText(f"Current: {fmt(i)}")
        except Exception as e:
            self.ch1_live_volt_lbl.setText("Voltage: Error")
            self.ch1_live_curr_lbl.setText("Current: Error")
            err_msg = str(e)
            if "VI_ERROR_RSRC_LOCKED" in err_msg or "locked" in err_msg.lower():
                QtWidgets.QMessageBox.warning(
                    self, "Resource Locked",
                    "Failed to read Channel 1:\nThe instrument connection is locked by another process.\n\n"
                    "Please ensure no other Python instances, Jupyter notebooks, or software programs are communicating with the Keysight SMU, then try again."
                )
            else:
                QtWidgets.QMessageBox.warning(self, "Read Error", f"Failed to read Channel 1:\n{e}")
        finally:
            self.resume_status_polling(was_paused)

    def read_channel_two(self):
        if getattr(self, "plotter_mode", False):
            return
        was_paused = self.pause_status_polling()
        try:
            smu = self.station.smu
            v = smu.ch2.volt()
            i = smu.ch2.measure_curr()
            try:
                smu.return_to_local()
            except Exception:
                pass
            
            def fmt(curr):
                if abs(curr) < 1e-6:
                    return f"{curr*1e9:.2f} nA"
                elif abs(curr) < 1e-3:
                    return f"{curr*1e6:.2f} uA"
                else:
                    return f"{curr*1e3:.2f} mA"

            self.ch2_live_volt_lbl.setText(f"Voltage: {v:.4f} V")
            self.ch2_live_curr_lbl.setText(f"Current: {fmt(i)}")
        except Exception as e:
            self.ch2_live_volt_lbl.setText("Voltage: Error")
            self.ch2_live_curr_lbl.setText("Current: Error")
            err_msg = str(e)
            if "VI_ERROR_RSRC_LOCKED" in err_msg or "locked" in err_msg.lower():
                QtWidgets.QMessageBox.warning(
                    self, "Resource Locked",
                    "Failed to read Channel 2:\nThe instrument connection is locked by another process.\n\n"
                    "Please ensure no other Python instances, Jupyter notebooks, or software programs are communicating with the Keysight SMU, then try again."
                )
            else:
                QtWidgets.QMessageBox.warning(self, "Read Error", f"Failed to read Channel 2:\n{e}")
        finally:
            self.resume_status_polling(was_paused)

    def on_live_status_updated(self, b_val, v1_val, v2_val):
        self._last_polled_b = b_val
        self.lbl_current_b.setText(f"B = {b_val:.4f} T")

        # Forward mapping: VG = M * CH  →  vg1 = a*v1 + b*v2,  vg2 = c*v1 + d*v2
        a = self.coeff_a.value()
        b = self.coeff_b.value()
        c = self.coeff_c.value()
        d = self.coeff_d.value()

        c1_val = a * v1_val + b * v2_val
        c2_val = c * v1_val + d * v2_val

        c1_label = self.c1_name_edit.text() if hasattr(self, "c1_name_edit") and self.c1_name_edit.text() else "C1"
        c2_label = self.c2_name_edit.text() if hasattr(self, "c2_name_edit") and self.c2_name_edit.text() else "C2"

        self.lbl_target_vc1.setText(f"{c1_label} = {c1_val:.4f} V (V1 = {v1_val:.4f} V)")
        self.lbl_target_vc2.setText(f"{c2_label} = {c2_val:.4f} V (V2 = {v2_val:.4f} V)")

    # --- Event Handlers & Input Syncing ---
    def update_step_calculations(self):
        """Calculate and display step sizes for gating variables dynamically."""
        for axis in ('c1', 'c2'):
            start = getattr(self, f"{axis}_start").value()
            end = getattr(self, f"{axis}_end").value()
            pts = getattr(self, f"{axis}_pts").value()
            if pts > 1:
                step = (end - start) / (pts - 1)
                getattr(self, f"{axis}_step_lbl").setText(f"{step:.5f} V")
            else:
                getattr(self, f"{axis}_step_lbl").setText("N/A")

    def update_coordinates_from_ui(self):
        """Build coordinates and grids from UI control values."""
        c1_start = self.c1_start.value()
        c1_end = self.c1_end.value()
        c1_pts = self.c1_pts.value()

        c2_start = self.c2_start.value()
        c2_end = self.c2_end.value()
        c2_pts = self.c2_pts.value()

        c1_pts_actual = c1_pts
        c2_pts_actual = c2_pts

        self.c1_coords = np.linspace(c1_start, c1_end, c1_pts_actual) if c1_pts_actual > 1 else np.array([c1_start])
        self.c2_coords = np.linspace(c2_start, c2_end, c2_pts_actual) if c2_pts_actual > 1 else np.array([c2_start])

        a = self.coeff_a.value()
        b = self.coeff_b.value()
        c = self.coeff_c.value()
        d = self.coeff_d.value()

        try:
            c1, c2 = np.meshgrid(self.c1_coords, self.c2_coords)
            self.v1_grid, self.v2_grid = transform_c_to_v(c1, c2, a, b, c, d)
        except ValueError:
            # Saved or temporarily edited singular matrices must not prevent
            # opening the GUI. Undefined targets cannot be used for a sweep.
            shape = (c2_pts_actual, c1_pts_actual)
            self.v1_grid = np.full(shape, np.nan)
            self.v2_grid = np.full(shape, np.nan)
        self.update_det_label()

    def on_sweep_range_changed(self):
        """Called dynamically when sweep parameters are changed in the UI."""
        self.update_step_calculations()
        if not self.is_running:
            self.update_coordinates_from_ui()
            self.reset_plot_scaling(reset_spectrum=False)
            self.save_storage()

    def update_det_label(self):
        """Recompute and display the determinant of the gate transformation matrix."""
        a = self.coeff_a.value()
        b = self.coeff_b.value()
        c = self.coeff_c.value()
        d = self.coeff_d.value()
        det = a * d - b * c
        try:
            transform_c_to_v(0, 0, a, b, c, d)
            valid = True
            text = f"det = {det:.4g}"
        except ValueError as error:
            valid = False
            text = f"Sweep disabled: {error}"
        if hasattr(self, 'det_label'):
            self.det_label.setText(text)
            color = '#16a34a' if valid else '#dc2626'
            self.det_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
        can_begin = valid and not getattr(self, 'plotter_mode', False)
        ramp = getattr(self, 'ramp_worker', None)
        if hasattr(self, 'begin_btn') and not self.is_running and not (ramp and ramp.isRunning()):
            self.begin_btn.setEnabled(can_begin)
        return can_begin

    def update_variable_names(self):
        """Dynamically update control group box titles, labels, equations and plot axes on user edit."""
        c1_name = self.c1_name_edit.text().strip() or "Vc1"
        c2_name = self.c2_name_edit.text().strip() or "Vc2"

        # Update group box titles
        self.ctrl_c1_box.setTitle(f"{c1_name}")
        self.ctrl_c2_box.setTitle(f"{c2_name}")

        # Update formula row labels (channel names are fixed; VG names update the row prefix)
        self.v1_formula_label.setText(f"{c1_name} =")
        self.v2_formula_label.setText(f"{c2_name} =")
        self.v1_c1_label.setText(" * ch1  + ")
        self.v1_c2_label.setText(" * ch2")
        self.v2_c1_label.setText(" * ch1  + ")
        self.v2_c2_label.setText(" * ch2")
        self.update_det_label()

        # Format axes label strings
        c1_label = f"Vc1 ({c1_name})" if c1_name.lower() != "vc1" else "Vc1"
        c2_label = f"Vc2 ({c2_name})" if c2_name.lower() != "vc2" else "Vc2"

        # Update plot axis labels if they exist
        if hasattr(self, 'gatemap_win'):
            self.gatemap_win.setLabel('bottom', c1_label, units='V')
            self.gatemap_win.setLabel('left', c2_label, units='V')
        if hasattr(self, 'vslice_win'):
            self.vslice_win.setLabel('left', c2_label, units='V')
        if hasattr(self, 'hslice_win'):
            self.hslice_win.setLabel('left', c1_label, units='V')
        if hasattr(self, 't2_heatmap_win'):
            self.t2_heatmap_win.setLabel('bottom', c1_label, units='V')
            self.t2_heatmap_win.setLabel('left', c2_label, units='V')
        if hasattr(self, 't2_hslice_win'):
            self.t2_hslice_win.setLabel('bottom', c1_label, units='V')
        if hasattr(self, 't2_vslice_win'):
            self.t2_vslice_win.setLabel('bottom', c2_label, units='V')
        if hasattr(self, 't3_heatmap_win'):
            self.t3_heatmap_win.setLabel('bottom', c1_label, units='V')
            self.t3_heatmap_win.setLabel('left', c2_label, units='V')
        if hasattr(self, 't3_hslice_win'):
            self.t3_hslice_win.setLabel('bottom', c1_label, units='V')
        if hasattr(self, 't3_vslice_win'):
            self.t3_vslice_win.setLabel('bottom', c2_label, units='V')

        # Re-trigger step, coordinate updates, and storage save
        self.on_sweep_range_changed()

    def on_ch1_setpoint_enter(self):
        """Enter key pressed in Setpoint 1 - execute background ramping."""
        val = self.ch1_setpoint.value()
        mode = self.ch1_mode.currentText().lower() # 'volt' or 'curr'
        self.ramp_channel(self.station.smu.ch1, val, mode, "Channel 1")

    def on_ch2_setpoint_enter(self):
        """Enter key pressed in Setpoint 2 - execute background ramping."""
        val = self.ch2_setpoint.value()
        mode = self.ch2_mode.currentText().lower() # 'volt' or 'curr'
        self.ramp_channel(self.station.smu.ch2, val, mode, "Channel 2")

    def ramp_channel(self, channel, target, param_name, ch_name):
        if getattr(self, "plotter_mode", False) or self.is_running:
            return
        if self.ramp_worker is not None and self.ramp_worker.isRunning():
            return
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.pause()
        self.station.live_status_paused = True
        if self.ramp_worker is not None and self.ramp_worker.isRunning():
            return
        unit = "A" if param_name in ('curr', 'curr_limit') else "V"
        self.status_lbl.setText(f"Status: Enabling {ch_name} with a voltage ramp..." if param_name == "output" else f"Status: Ramping {ch_name} to {target} {unit}...")
        if param_name == 'curr_limit':
            self.status_lbl.setText(f'Status: Setting {ch_name} current compliance to {target:g} A...')
        self.set_inputs_enabled(False)
        self.begin_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.pause_btn.setEnabled(param_name != 'curr_limit')
        if ch_name == "Channel 1" and param_name != 'curr_limit':
            self.ch1_stop_btn.setEnabled(True)
        elif param_name != 'curr_limit':
            self.ch2_stop_btn.setEnabled(True)
        
        self.ramp_worker = SingleRampWorker(channel, target, param_name, ch_name)
        self.ramp_worker.output_updated.connect(self.on_output_updated)
        self.ramp_worker.compliance_updated.connect(self.on_compliance_applied)
        self.ramp_worker.progress.connect(self.on_ramp_progress)
        self.ramp_worker.finished.connect(self.on_ramp_finished)
        self.ramp_worker.error.connect(self.on_ramp_error)
        self.ramp_worker.start()
        # Lock competing tabs in this click handler, before another click can
        # arrive; the periodic dashboard refresh alone leaves a race window.
        dashboard = getattr(self, 'dashboard', None)
        if dashboard is not None:
            dashboard.check_measurement_status()

    def on_ramp_finished(self):
        self._applying_compliance = False
        self.station.live_status_paused = False
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        if self.ramp_worker and getattr(self.ramp_worker.channel, "ramp_stop_requested", False):
            self.status_lbl.setText("Status: Ramping interrupted.")
        else:
            self.status_lbl.setText("Status: Ramping complete.")
        if self.ramp_worker and self.ramp_worker.param_name == 'curr_limit':
            self.status_lbl.setText('Status: Current compliance applied and verified.')
        self.set_inputs_enabled(True)
        self.begin_btn.setEnabled(self.update_det_label())
        self.reset_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.ch1_stop_btn.setEnabled(False)
        self.ch2_stop_btn.setEnabled(False)

    def on_ramp_error(self, err_msg):
        compliance_failed = self._applying_compliance
        self._applying_compliance = False
        self.station.live_status_paused = False
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        self.status_lbl.setText(f"Status: Ramping error")
        self.set_inputs_enabled(True)
        self.begin_btn.setEnabled(self.update_det_label())
        self.reset_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.ch1_stop_btn.setEnabled(False)
        self.ch2_stop_btn.setEnabled(False)
        if compliance_failed:
            axis = self.ramp_worker.channel.channel_num
            self._compliance_readbacks[axis] = None
            self._compliance_edits.discard(axis)
            self._update_compliance_controls(force=True)
            self.status_lbl.setText('Status: Current compliance update failed.')
        QtWidgets.QMessageBox.critical(self, 'Compliance Error' if compliance_failed else 'Ramping Error', err_msg)

    def on_ramp_progress(self, ch_name, volt_val, curr_val):
        def fmt(curr):
            if abs(curr) < 1e-6:
                return f"{curr*1e9:.2f} nA"
            elif abs(curr) < 1e-3:
                return f"{curr*1e6:.2f} uA"
            else:
                return f"{curr*1e3:.2f} mA"

        if ch_name == "Channel 1":
            self.ch1_live_volt_lbl.setText(f"Voltage: {volt_val:.4f} V")
            self.ch1_live_curr_lbl.setText(f"Current: {fmt(curr_val)}")
        else:
            self.ch2_live_volt_lbl.setText(f"Voltage: {volt_val:.4f} V")
            self.ch2_live_curr_lbl.setText(f"Current: {fmt(curr_val)}")

    def on_ch1_stop_clicked(self):
        if self.ramp_worker and self.ramp_worker.isRunning() and self.ramp_worker.ch_name == "Channel 1":
            self.ramp_worker.stop()
            self.status_lbl.setText("Status: Interrupting Channel 1 ramp...")
            self.ch1_stop_btn.setEnabled(False)

    def on_ch2_stop_clicked(self):
        if self.ramp_worker and self.ramp_worker.isRunning() and self.ramp_worker.ch_name == "Channel 2":
            self.ramp_worker.stop()
            self.status_lbl.setText("Status: Interrupting Channel 2 ramp...")
            self.ch2_stop_btn.setEnabled(False)

    def on_output_updated(self, ch_name, enabled):
        axis = 1 if ch_name == 'Channel 1' else 2
        button = getattr(self, f'ch{axis}_out_btn')
        with QtCore.QSignalBlocker(button):
            button.setChecked(enabled)
            button.setText('ON' if enabled else 'OFF')
            button.setStyleSheet(f"background-color: {'#10b981' if enabled else '#f1f5f9'}; color: {'white' if enabled else '#475569'}; font-weight: bold;")

    def _toggle_output(self, axis, enabled):
        if self.plotter_mode:
            return
        channel = getattr(self.station.smu, f'ch{axis}')
        if enabled:
            self.ramp_channel(channel, True, 'output', f'Channel {axis}')
        else:
            # OFF remains immediate; do not wait behind a running ramp.
            try:
                channel.output(False)
                self.on_output_updated(f'Channel {axis}', False)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, 'Hardware error', str(exc))

    def toggle_ch1_output(self, enabled):
        self._toggle_output(1, enabled)

    def toggle_ch2_output(self, enabled):
        self._toggle_output(2, enabled)

    def on_compliance_changed(self, axis=None):
        # Only an edited current-limit field commits a hardware change. Calls
        # from loading settings, mode changes and display refresh are read-only.
        if axis is None or axis not in self._compliance_edits:
            if axis is None:
                self._compliance_edits.clear()
            self._update_compliance_controls(force=axis is None)
            return
        self._compliance_edits.discard(axis)
        if (self.plotter_mode or not self.editable_current_compliance
                or self.is_running or self._applying_compliance
                or (self.ramp_worker and self.ramp_worker.isRunning())
                or getattr(self, f'ch{axis}_mode').currentText() != 'VOLT'):
            self._update_compliance_controls(force=True)
            return
        dashboard = getattr(self, 'dashboard', None)
        if dashboard and dashboard.measurement_active():
            self._update_compliance_controls(force=True)
            return
        try:
            value = validate_current_compliance(getattr(self, f'ch{axis}_compliance').value())
        except ValueError as error:
            self._update_compliance_controls(force=True)
            QtWidgets.QMessageBox.warning(self, 'Invalid current compliance', str(error))
            return
        channel = getattr(self.station.smu, f'ch{axis}')
        self._applying_compliance = True
        self.ramp_channel(channel, value, 'curr_limit', f'Channel {axis}')

    def on_compliance_applied(self, axis, value):
        self._compliance_readbacks[axis] = ('VOLT', value)
        self._compliance_edits.discard(axis)
        self._update_compliance_controls(force=True)
        self.save_storage()

    def _compliance_setting(self, axis):
        reading = self._compliance_readbacks.get(axis)
        mode = getattr(self, f'ch{axis}_mode').currentText()
        return reading[1] if reading is not None and reading[0] == mode else -1.0

    def on_compliance_readback(self, readings):
        if self.plotter_mode or self._applying_compliance:
            return
        self._compliance_readbacks = readings
        self._update_compliance_controls()

    def _update_compliance_controls(self, force=False):
        for axis in (1, 2):
            box = getattr(self, f'ch{axis}_compliance')
            current_mode = getattr(self, f'ch{axis}_mode').currentText() == 'CURR'
            value = self._compliance_setting(axis)
            editable = self.editable_current_compliance and not current_mode and not self.plotter_mode
            # Polling must not replace text while the user is entering a limit.
            if editable and not force and (box.hasFocus() or axis in self._compliance_edits):
                continue
            with QtCore.QSignalBlocker(box):
                box.setReadOnly(not editable)
                box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                box.setDecimals(12)
                box.setRange(-1.0, 1e12)
                box.setSpecialValueText('Readback unavailable')
                box.setSuffix(' V' if current_mode else ' A')
                box.setValue(value)
                box.setSingleStep(1e-9 if not current_mode else .1)
                box.setToolTip('Current compliance in amperes (1 nA to 3 A). Press Enter or leave the edited field to apply and verify.'
                               if editable else 'Read-only DC-source compliance readback.')
                channel = getattr(self.station.smu, f'ch{axis}', None)
                expected = getattr(channel, 'current_compliance_target', GATE_CURRENT_COMPLIANCE_A)
                mismatch = value < 0 or (not current_mode and not math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-12)) or (current_mode and value > GATE_MAX_ABS_VOLTAGE_V)
                box.setStyleSheet('color: #b91c1c; font-weight: bold;' if mismatch else '')
            getattr(self, f'ch{axis}_compliance_lbl').setText('Voltage compliance:' if current_mode else 'Current limit:')

    def on_mode_changed(self, *args, write_hardware=True):
        if getattr(self, "plotter_mode", False):
            return
        was_paused = self.pause_status_polling()
        try:
            if write_hardware:
                self.station.smu.ch1.source_mode(self.ch1_mode.currentText())
                self.station.smu.ch2.source_mode(self.ch2_mode.currentText())
        except Exception as exc:
            # A blocked mode change must not leave the GUI claiming another mode.
            for axis in (1, 2):
                box = getattr(self, f'ch{axis}_mode')
                with QtCore.QSignalBlocker(box):
                    box.setCurrentText(getattr(self.station.smu, f'ch{axis}').source_mode())
            QtWidgets.QMessageBox.critical(self, 'Gate protection', str(exc))
            write_hardware = False
        finally:
            self.resume_status_polling(was_paused)

        self._compliance_edits.clear()
        for axis in (1, 2):
            current_mode = getattr(self, f'ch{axis}_mode').currentText() == 'CURR'
            box = getattr(self, f'ch{axis}_setpoint')
            getattr(self, f'ch{axis}_setpoint_lbl').setText('Setpoint (A):' if current_mode else 'Setpoint (V):')
            box.setDecimals(9 if current_mode else 4)
            box.setRange(-GATE_CURRENT_COMPLIANCE_A, GATE_CURRENT_COMPLIANCE_A) if current_mode else box.setRange(-GATE_MAX_ABS_VOLTAGE_V, GATE_MAX_ABS_VOLTAGE_V)
            box.setSingleStep(1e-9 if current_mode else 0.001)
        self._update_compliance_controls(force=True)
        if write_hardware:
            self.on_compliance_changed()

    def on_ch1_ramping_params_changed(self):
        if getattr(self, "plotter_mode", False):
            return
        """Save Channel 1 settings to value_storage and sync with hardware driver."""
        rate = self.ch1_ramp_rate.value()
        self.stored_settings["ramprate1"] = rate
        self.save_storage()
        was_paused = self.pause_status_polling()
        try:
            self.station.smu.ch1.ramp_rate(rate)
        except Exception:
            pass
        finally:
            self.resume_status_polling(was_paused)

    def on_ch2_ramping_params_changed(self):
        if getattr(self, "plotter_mode", False):
            return
        """Save Channel 2 settings to value_storage and sync with hardware driver."""
        rate = self.ch2_ramp_rate.value()
        self.stored_settings["ramprate2"] = rate
        self.save_storage()
        was_paused = self.pause_status_polling()
        try:
            self.station.smu.ch2.ramp_rate(rate)
        except Exception:
            pass
        finally:
            self.resume_status_polling(was_paused)


    def on_timing_params_changed(self):
        self.stored_settings["exposure_time"] = self.exposure_box.value()
        self.stored_settings["settling_time"] = self.settling_box.value()
        self.save_storage()

    def set_inputs_enabled(self, enabled: bool):
        """Enable or disable all configuration inputs and manual control buttons during sweeps."""
        # Keysight controls
        self.ch1_out_btn.setEnabled(enabled)
        self.ch2_out_btn.setEnabled(enabled)
        self.ch1_setpoint.setEnabled(enabled)
        self.ch2_setpoint.setEnabled(enabled)
        self.ch1_mode.setEnabled(enabled)
        self.ch2_mode.setEnabled(enabled)
        self.ch1_ramp_rate.setEnabled(enabled)
        self.ch2_ramp_rate.setEnabled(enabled)
        self.ch1_compliance.setEnabled(enabled)
        self.ch2_compliance.setEnabled(enabled)
        self.ch1_read_btn.setEnabled(enabled)
        self.ch2_read_btn.setEnabled(enabled)
        
        # Gating coefficients
        self.coeff_a.setEnabled(enabled)
        self.coeff_b.setEnabled(enabled)
        self.coeff_c.setEnabled(enabled)
        self.coeff_d.setEnabled(enabled)
        
        # Control Variables
        self.c1_start.setEnabled(enabled)
        self.c1_end.setEnabled(enabled)
        self.c1_pts.setEnabled(enabled)
        self.c2_start.setEnabled(enabled)
        self.c2_end.setEnabled(enabled)
        self.c2_pts.setEnabled(enabled)
        
        # Timing / Sweep settings
        self.exposure_box.setEnabled(enabled)
        self.settling_box.setEnabled(enabled)
        self.sweep_mode_combo.setEnabled(enabled)
        self.serpentine_box.setEnabled(enabled)

    # --- Sweep Measurement Control ---
    def begin_sweep(self):
        if getattr(self, "plotter_mode", False):
            return
        if self.ramp_worker is not None and self.ramp_worker.isRunning():
            return
        if self.is_paused:
            # Resume sweep
            self.is_paused = False
            self.pause_btn.setText("⏸ Interrupt")
            self.pause_btn.setEnabled(True)
            self.begin_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.status_lbl.setText("Status: Sweeping...")
            self.sweep_worker.resume()
            return

        if self.is_running:
            return

        try:
            validate_dgs_parameters({
                'a': self.coeff_a.value(), 'b': self.coeff_b.value(),
                'c': self.coeff_c.value(), 'd': self.coeff_d.value(),
                'c1_start': self.c1_start.value(), 'c1_end': self.c1_end.value(), 'c1_pts': self.c1_pts.value(),
                'c2_start': self.c2_start.value(), 'c2_end': self.c2_end.value(), 'c2_pts': self.c2_pts.value(),
            })
            if self.ch1_mode.currentText() != 'VOLT' or self.ch2_mode.currentText() != 'VOLT':
                raise ValueError('Gate sweeps require both DC sources in voltage mode.')
        except Exception as exc:
            self.status_lbl.setText(f'Sweep blocked: {exc}')
            QtWidgets.QMessageBox.critical(self, 'Gate protection', str(exc))
            return
        self.reset_slice_levels = True

        # Pause the status worker first to avoid any race conditions during monochromator/camera queries
        self.live_status_worker.pause()
        self.station.live_status_paused = True

        # Clear the SMU VISA session to discard any unread response or dirty buffer from active status workers
        smu = getattr(self.station, "smu", None)
        if smu is not None and not smu.simulated:
            try:
                smu.visa_handle.clear()
                smu.write('*CLS')
            except Exception:
                pass

        if smu is not None:
            try:
                smu.enforce_current_compliance()
                self.measurement_current_compliance = {
                    axis: getattr(getattr(smu, f'ch{axis}'), 'current_compliance_target', GATE_CURRENT_COMPLIANCE_A)
                    for axis in (1, 2)
                }
            except Exception as e:
                self.station.live_status_paused = False
                self.live_status_worker.resume()
                self.status_lbl.setText(f'Sweep blocked: cannot confirm current compliance: {e}')
                QtWidgets.QMessageBox.critical(self, 'Gate protection', self.status_lbl.text())
                return

        # Prepare parameters and validate
        self.update_coordinates_from_ui()
        c1_pts_actual = len(self.c1_coords)
        c2_pts_actual = len(self.c2_coords)

        # Determine number of pixels and wavelengths dynamically from camera
        num_pixels = 512 # fallback
        wavelengths = None
        
        # Dynamically retrieve monochromator wavelength to calibrate Energy axis
        try:
            mono_wl = self.station.monochromator.wavelength()
        except Exception:
            mono_wl = 500.0 # fallback

        # Try to get wavelengths from cache first (fast, non-blocking)
        try:
            wls = self.station.camera.remote_lf.get_wavelengths_cache()
            if wls is not None and len(wls) > 0:
                wavelengths = np.array(wls)
                num_pixels = len(wavelengths)
        except Exception as e:
            print(f"Could not get wavelengths cache: {e}")
                
        if wavelengths is None:
            # Fallback to manual calibration around center wavelength
            pixels_arr = np.arange(num_pixels)
            wavelengths = mono_wl + (pixels_arr - num_pixels // 2) * 0.05

        # Load num_spectra from value_storage
        num_spectra = 1
        try:
            storage_path = os.path.join(parent_dir, "value_storage")
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f_vs:
                    vs = json.load(f_vs)
                single_data = vs.get("single_spectra", {})
                num_spectra = int(single_data.get("num_spectra", vs.get("num_spectra", 1)))
        except Exception:
            num_spectra = 1

        self.data_spectra = np.zeros((c2_pts_actual, c1_pts_actual, num_pixels))
        self.data_raw_spectra = np.zeros((c2_pts_actual, c1_pts_actual, num_spectra, num_pixels))
        self.data_current_ch1 = np.zeros((c2_pts_actual, c1_pts_actual))
        self.data_current_ch2 = np.zeros((c2_pts_actual, c1_pts_actual))
        self.data_measured_volt_ch1 = np.zeros((c2_pts_actual, c1_pts_actual))
        self.data_measured_volt_ch2 = np.zeros((c2_pts_actual, c1_pts_actual))
        self.data_measured_b = np.zeros((c2_pts_actual, c1_pts_actual))
        self.data_measured_mask = np.zeros((c2_pts_actual, c1_pts_actual), dtype=bool)

        # Initialize heatmap images with zero arrays of the correct size
        # so pyqtgraph can resolve their size when setRect is called
        self.gatemap_img.setImage(np.zeros((c2_pts_actual, c1_pts_actual)))
        self.vslice_img.setImage(np.zeros((c2_pts_actual, num_pixels)))
        self.hslice_img.setImage(np.zeros((c1_pts_actual, num_pixels)))
        self.t2_heatmap_img.setImage(np.zeros((c2_pts_actual, c1_pts_actual)))
        self.t3_heatmap_img.setImage(np.zeros((c1_pts_actual, c2_pts_actual)))

        # Convert nm to eV
        energy_raw = 1239.84193 / wavelengths
        self.energy_sort_idx = np.argsort(energy_raw)
        self.energy_axis = energy_raw[self.energy_sort_idx]
        self.linear_energy_axis = np.linspace(self.energy_axis[0], self.energy_axis[-1], len(self.energy_axis))
        self.wavelength_axis = wavelengths[self.energy_sort_idx]

        # Reset plots and set axes ranges
        self.reset_plot_scaling(reset_spectrum=True)

        # Start sweep thread — guard against singular transformation matrix
        _a = self.coeff_a.value()
        _b = self.coeff_b.value()
        _c = self.coeff_c.value()
        _d = self.coeff_d.value()
        _det = _a * _d - _b * _c
        if abs(_det) < 1e-9:
            QtWidgets.QMessageBox.warning(
                self, "Singular Matrix",
                f"The gate transformation matrix is singular (det = {_det:.2e}).\n"
                "Virtual gates cannot be converted to physical voltages.\n"
                "Please adjust coefficients so that a·d − b·c ≠ 0."
            )
            return

        params = {
            'a': self.coeff_a.value(),
            'b': self.coeff_b.value(),
            'c': self.coeff_c.value(),
            'd': self.coeff_d.value(),
            'c1_start': self.c1_start.value(),
            'c1_end': self.c1_end.value(),
            'c1_pts': self.c1_pts.value(),
            'c2_start': self.c2_start.value(),
            'c2_end': self.c2_end.value(),
            'c2_pts': self.c2_pts.value(),
            'sweep_mode': self.sweep_mode_combo.currentText(),
            'serpentine': self.serpentine_box.isChecked(),
            'exposure_time': self.exposure_box.value(),
            'settling_time': self.settling_box.value(),
            'num_pixels': num_pixels,
            'wl_axis': wavelengths,
        }

        # Calculate total expected duration
        try:
            magnet = getattr(self.station, "magnet", None)
            curr_b = magnet.output_field() if magnet is not None else 0.0
        except Exception:
            curr_b = 0.0

        try:
            smu = getattr(self.station, "smu", None)
            prev_v1 = 0.0
            prev_v2 = 0.0
            if smu is not None and not smu.simulated:
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        prev_v1 = smu.ch1.volt()
                        break
                    except Exception as ex:
                        if attempt == max_retries:
                            raise ex
                        try:
                            smu.visa_handle.clear()
                        except Exception:
                            pass
                        time.sleep(0.1)
                for attempt in range(1, max_retries + 1):
                    try:
                        prev_v2 = smu.ch2.volt()
                        break
                    except Exception as ex:
                        if attempt == max_retries:
                            raise ex
                        try:
                            smu.visa_handle.clear()
                        except Exception:
                            pass
                        time.sleep(0.1)
        except Exception:
            prev_v1 = 0.0
            prev_v2 = 0.0

        v_points = []
        is_c2_outer = (self.sweep_mode_combo.currentText() == "Hold C2, sweep C1")
        serpentine = self.serpentine_box.isChecked()

        if is_c2_outer:
            outer_arr = self.c2_coords
            inner_arr = self.c1_coords
        else:
            outer_arr = self.c1_coords
            inner_arr = self.c2_coords

        a = self.coeff_a.value()
        b = self.coeff_b.value()
        c = self.coeff_c.value()
        d = self.coeff_d.value()

        for idx_outer, val_outer in enumerate(outer_arr):
            if serpentine and (idx_outer % 2 == 1):
                inner_indices = list(range(len(inner_arr) - 1, -1, -1))
            else:
                inner_indices = list(range(len(inner_arr)))
            for idx_inner in inner_indices:
                val_inner = inner_arr[idx_inner]
                if is_c2_outer:
                    c2_val = val_outer
                    c1_val = val_inner
                else:
                    c1_val = val_outer
                    c2_val = val_inner
                v1_target, v2_target = transform_c_to_v(c1_val, c2_val, a, b, c, d)
                v_points.append((v1_target, v2_target))

        num_spectra = 1
        storage_path = os.path.join(parent_dir, "value_storage")
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f_vs:
                    vs = json.load(f_vs)
                single_data = vs.get("single_spectra", {})
                num_spectra = int(single_data.get("num_spectra", vs.get("num_spectra", 1)))
            except Exception:
                num_spectra = 1

        self.sweep_total_duration = calculate_expected_duration(
            v_points=v_points,
            b_points=[curr_b] * len(v_points),
            exposure_time=self.exposure_box.value(),
            settling_time=self.settling_box.value(),
            num_spectra=num_spectra,
            sweep_rate_b=0.5 / 60.0,
            initial_b=curr_b,
            initial_v1=prev_v1,
            initial_v2=prev_v2,
            ramp_rate_v1=self.ch1_ramp_rate.value(),
            ramp_rate_v2=self.ch2_ramp_rate.value()
        )
        self.sweep_v_points = v_points
        self.sweep_b_points = [curr_b] * len(v_points)
        self.sweep_num_spectra = num_spectra
        self.sweep_rate_b = 0.5 / 60.0
        self.sweep_initial_v1 = prev_v1
        self.sweep_initial_v2 = prev_v2
        self.sweep_initial_b = curr_b
        self.sweep_start_time = time.time()
        self.measurement_started_at = new_measurement_start(self.sweep_start_time)
        self.measurement_device_modes = measurement_device_modes(self.station)
        self.total_paused_time = 0.0
        self.pause_start_time = None

        self.lbl_expected_end.setText("Expected End: (waiting for measurements)")

        self.is_running = True
        self.is_paused = False
        self._autosaved_this_run = False
        
        # Pause background status polling during active sweep to avoid SMU PyVISA collisions
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.pause()
        self.station.live_status_paused = True

        # Clear SMU VISA session to discard any unread response or dirty buffer from active status workers
        smu = getattr(self.station, "smu", None)
        if smu is not None and not getattr(smu, "simulated", False):
            try:
                smu.visa_handle.clear()
                smu.write('*CLS')
            except Exception:
                pass

        self.begin_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ Interrupt")
        self.reset_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.set_inputs_enabled(False)

        params["num_spectra"] = self.data_raw_spectra.shape[2]
        self._filename_context = {
            "exposure": params["exposure_time"], "acquisitions": params["num_spectra"],
            "axes": [(self.c1_name_edit.text(), self.c1_coords.copy()),
                     (self.c2_name_edit.text(), self.c2_coords.copy())],
            "field": curr_b,
        }
        self.sweep_worker = self.create_sweep_worker(params)
        self.sweep_worker.point_measured.connect(self.on_point_measured)
        self.sweep_worker.progress_updated.connect(self.on_progress_updated)
        self.sweep_worker.status_updated.connect(self.status_lbl.setText)
        self.sweep_worker.voltage_updated.connect(self.on_sweep_voltage_updated)
        self.sweep_worker.finished.connect(self.on_sweep_finished)
        self.sweep_worker.error_occurred.connect(self.on_sweep_error)
        self.sweep_worker.start()

    def create_sweep_worker(self, params):
        return SweepWorker(self.station, params)

    def default_sweep_filename(self):
        from measurement_filenames import dgs_filename
        context = getattr(self, "_filename_context", None) or {
            "exposure": self.exposure_box.value(),
            "acquisitions": self.data_raw_spectra.shape[2] if self.data_raw_spectra.size else 1,
            "axes": [(self.c1_name_edit.text(), self.c1_coords),
                     (self.c2_name_edit.text(), self.c2_coords)],
            "field": getattr(self, "_last_polled_b", 0.0),
        }
        return dgs_filename(**context, wavelength=self.filename_wavelength_box.value())

    def on_interrupt_clicked(self):
        if self.ramp_worker and self.ramp_worker.isRunning():
            self.ramp_worker.stop()
            self.status_lbl.setText("Status: Interrupting ramp...")
            self.pause_btn.setEnabled(False)
        elif self.is_running:
            self.pause_sweep()

    def pause_sweep(self):
        if not self.is_running:
            return
        if self.is_paused:
            # Already paused, resume it
            self.is_paused = False
            self.pause_btn.setText("⏸ Interrupt")
            self.begin_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.status_lbl.setText("Status: Sweeping...")
            if getattr(self, "pause_start_time", None) is not None:
                self.total_paused_time += time.time() - self.pause_start_time
                self.pause_start_time = None
            if hasattr(self, 'live_status_worker') and self.live_status_worker:
                self.live_status_worker.pause()
            self.station.live_status_paused = True
            self.sweep_worker.resume()
        else:
            self.is_paused = True
            self.pause_btn.setText("▶ Resume")
            self.begin_btn.setEnabled(self.update_det_label())
            self.save_btn.setEnabled(True if np.any(self.data_measured_mask) else False)
            self.status_lbl.setText("Status: Paused (Interrupt)")
            self.pause_start_time = time.time()
            self.station.live_status_paused = True
            self.sweep_worker.pause()

    def reset_sweep(self):
        was_running = self.is_running
        if self.is_running:
            self.trigger_autosave(was_interrupted=True)
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        self.station.live_status_paused = False
        
        self.is_running = False
        self.is_paused = False
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self.begin_btn.setEnabled(self.update_det_label())
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ Interrupt")
        self.save_btn.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.set_inputs_enabled(True)
        
        self.progress_bar.setValue(0)
        self.lbl_expected_end.setText("Expected End: --:--:--")
        self.status_lbl.setText("Status: Reset complete")

    def on_point_measured(self, i, j, c1, c2, v1, v2, spectrum, curr_ch1, curr_ch2, meas_v1, meas_v2, meas_b, raw_frames):
        # Sort spectrum to align with increasing energy values
        sorted_spectrum = spectrum[self.energy_sort_idx]
        sorted_raw_frames = raw_frames[:, self.energy_sort_idx]
        
        # Save measurement point
        self.data_spectra[j, i, :] = sorted_spectrum
        if getattr(self, "data_raw_spectra", None) is not None:
            self.data_raw_spectra[j, i, :, :] = sorted_raw_frames
        self.data_current_ch1[j, i] = curr_ch1
        self.data_current_ch2[j, i] = curr_ch2
        self.data_measured_volt_ch1[j, i] = meas_v1
        self.data_measured_volt_ch2[j, i] = meas_v2
        self.data_measured_b[j, i] = meas_b
        self.data_measured_mask[j, i] = True

        # Redraw plots
        self.update_heatmaps_and_slices()
        
        # Force refreshing current slices based on active crosshairs during sweep
        self.on_crosshair_moved()
        self.on_tab2_crosshair_moved()
        self.on_tab3_crosshair_moved()

        # Update panel labels with current sweep values
        self._last_polled_b = meas_b
        self.lbl_current_b.setText(f"B = {meas_b:.4f} T")
        c1_label = self.c1_name_edit.text() if hasattr(self, "c1_name_edit") and self.c1_name_edit.text() else "C1"
        c2_label = self.c2_name_edit.text() if hasattr(self, "c2_name_edit") and self.c2_name_edit.text() else "C2"
        self.lbl_target_vc1.setText(f"{c1_label} = {c1:.4f} V (V1 = {meas_v1:.4f} V)")
        self.lbl_target_vc2.setText(f"{c2_label} = {c2:.4f} V (V2 = {meas_v2:.4f} V)")

    def on_progress_updated(self, current, total):
        pct = int(100 * current / total)
        self.progress_bar.setValue(pct)
        
        if getattr(self, "sweep_start_time", None) is not None:
            if current < 3:
                self.lbl_expected_end.setText("Expected End: (waiting for measurements)")
                return

            current_pause_overhead = 0.0
            if self.is_paused and getattr(self, "pause_start_time", None) is not None:
                current_pause_overhead = time.time() - self.pause_start_time
                
            total_elapsed_paused = self.total_paused_time + current_pause_overhead
            active_elapsed_time = time.time() - self.sweep_start_time - total_elapsed_paused

            # Get remaining points
            remaining_v = self.sweep_v_points[current:]
            remaining_b = self.sweep_b_points[current:]

            if current > 0 and current <= len(self.sweep_v_points):
                last_v1, last_v2 = self.sweep_v_points[current - 1]
                last_b = self.sweep_b_points[current - 1]
            else:
                last_v1, last_v2 = self.sweep_initial_v1, self.sweep_initial_v2
                last_b = self.sweep_initial_b

            predicted_remaining = calculate_expected_duration(
                v_points=remaining_v,
                b_points=remaining_b,
                exposure_time=self.exposure_box.value(),
                settling_time=self.settling_box.value(),
                num_spectra=self.sweep_num_spectra,
                sweep_rate_b=self.sweep_rate_b,
                initial_b=last_b,
                initial_v1=last_v1,
                initial_v2=last_v2,
                ramp_rate_v1=self.ch1_ramp_rate.value(),
                ramp_rate_v2=self.ch2_ramp_rate.value()
            )

            predicted_elapsed = self.sweep_total_duration - predicted_remaining
            if predicted_elapsed > 0 and active_elapsed_time > 0:
                scale_factor = active_elapsed_time / predicted_elapsed
                scale_factor = max(0.5, min(2.0, scale_factor))
            else:
                scale_factor = 1.0

            dynamic_remaining = predicted_remaining * scale_factor
            expected_end = time.time() + dynamic_remaining
            
            time_str = time.strftime("%H:%M:%S", time.localtime(expected_end))
            self.lbl_expected_end.setText(f"Expected End: {time_str}")
        else:
            self.lbl_expected_end.setText("Expected End: --:--:--")

    def on_sweep_voltage_updated(self, c1, c2, v1, v2):
        self.ch1_setpoint.setValue(v1)
        self.ch2_setpoint.setValue(v2)
        c1_label = self.c1_name_edit.text() if hasattr(self, "c1_name_edit") and self.c1_name_edit.text() else "C1"
        c2_label = self.c2_name_edit.text() if hasattr(self, "c2_name_edit") and self.c2_name_edit.text() else "C2"
        self.lbl_target_vc1.setText(f"{c1_label} = {c1:.4f} V (V1 = {v1:.4f} V)")
        self.lbl_target_vc2.setText(f"{c2_label} = {c2:.4f} V (V2 = {v2:.4f} V)")

    def on_sweep_finished(self):
        if not self.is_running:
            return
        self.trigger_autosave(was_interrupted=False)
        self.is_running = False
        self.begin_btn.setEnabled(self.update_det_label())
        self.pause_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.set_inputs_enabled(True)
        self.status_lbl.setText("Status: Sweep Finished (Data ready to save)")
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        self.station.live_status_paused = False
        QtWidgets.QMessageBox.information(self, "Sweep Complete", "Electrostatic gate sweep finished successfully!")

    def on_sweep_error(self, err_msg):
        self.trigger_autosave(was_interrupted=True)
        self.is_running = False
        self.begin_btn.setEnabled(self.update_det_label())
        self.pause_btn.setEnabled(False)
        self.save_btn.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.set_inputs_enabled(True)
        self.status_lbl.setText("Status: Error occurred")
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        self.station.live_status_paused = False
        QtWidgets.QMessageBox.critical(self, "Sweep Error", f"Measurement aborted due to error:\n{err_msg}")

    def prompt_save_on_interrupt(self):
        if not np.any(self.data_measured_mask):
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Save Interrupted Sweep",
            "The sweep was interrupted. Would you like to save the swept data?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.save_sweep_data()

    # --- Plot Sizing & Coordinates Setup ---
    def reset_plot_scaling(self, reset_spectrum=True):
        self.reset_slice_levels = True
        """Set initial geometric sizes and labels for heatmap plots."""
        # Tab 1: Gate Map (integrated spectra)
        width1 = self.c1_coords[-1] - self.c1_coords[0]
        x_start1 = self.c1_coords[0]
        c1_min = min(self.c1_coords[0], self.c1_coords[-1])
        c1_max = max(self.c1_coords[0], self.c1_coords[-1])
        if len(self.c1_coords) <= 1 or width1 == 0:
            width1 = 1.0
            x_start1 = self.c1_coords[0] - 0.5
            c1_min = x_start1
            c1_max = x_start1 + 1.0

        height1 = self.c2_coords[-1] - self.c2_coords[0]
        y_start1 = self.c2_coords[0]
        c2_min = min(self.c2_coords[0], self.c2_coords[-1])
        c2_max = max(self.c2_coords[0], self.c2_coords[-1])
        if len(self.c2_coords) <= 1 or height1 == 0:
            height1 = 1.0
            y_start1 = self.c2_coords[0] - 0.5
            c2_min = y_start1
            c2_max = y_start1 + 1.0
        # Determine energy axis limits for slice and spectrum plots
        energy_min = min(self.energy_axis[0], self.energy_axis[-1])
        energy_max = max(self.energy_axis[0], self.energy_axis[-1])

        # Set limits for plots (matching Vector Map style)
        self.gatemap_win.setLimits(xMin=c1_min, xMax=c1_max, yMin=c2_min, yMax=c2_max)
        self.vslice_win.setLimits(xMin=energy_min, xMax=energy_max, yMin=c2_min, yMax=c2_max)
        self.hslice_win.setLimits(xMin=energy_min, xMax=energy_max, yMin=c1_min, yMax=c1_max)
        if reset_spectrum:
            self.spectrum_win.setLimits(xMin=energy_min, xMax=energy_max)

        rect1 = QtCore.QRectF(x_start1, y_start1, width1, height1)
        if self.gatemap_img.image is not None:
            self.gatemap_img.setRect(rect1)
        
        vb1 = self.gatemap_win.getViewBox()
        vb1.invertX(self.c1_coords[0] > self.c1_coords[-1])
        vb1.invertY(self.c2_coords[0] > self.c2_coords[-1])
        self.gatemap_win.setRange(xRange=(c1_min, c1_max), yRange=(c2_min, c2_max), padding=0)

        # Tab 1: Vertical Slice (Y axis is C2, X axis is Energy)
        # Pixel-centre-aligned energy bounds: expand the rect by half a pixel on each side
        # so that image column k is centred at energy_axis[k], matching the 1D spectrum
        # curve which places data points at exact energy_axis[k] positions.
        n_e = len(self.energy_axis)
        if n_e > 1:
            d_e = (self.energy_axis[-1] - self.energy_axis[0]) / (n_e - 1)
            width_vslice = self.energy_axis[-1] - self.energy_axis[0] + d_e
            x_start_vslice = self.energy_axis[0] - d_e / 2
        else:
            d_e = 1.0
            width_vslice = 1.0
            x_start_vslice = self.energy_axis[0] - 0.5

        height_vslice = self.c2_coords[-1] - self.c2_coords[0]
        y_start_vslice = self.c2_coords[0]
        if len(self.c2_coords) <= 1 or height_vslice == 0:
            height_vslice = 1.0
            y_start_vslice = self.c2_coords[0] - 0.5

        rect_vslice = QtCore.QRectF(x_start_vslice, y_start_vslice, width_vslice, height_vslice)
        if self.vslice_img.image is not None:
            self.vslice_img.setRect(rect_vslice)

        vb_vslice = self.vslice_win.getViewBox()
        vb_vslice.invertY(self.c2_coords[0] > self.c2_coords[-1])
        self.vslice_win.setRange(xRange=(energy_min, energy_max), yRange=(c2_min, c2_max), padding=0)

        # Tab 1: Horizontal Slice (X axis is Energy, Y axis is C1)
        width_hslice = self.c1_coords[-1] - self.c1_coords[0]
        x_start_hslice = self.c1_coords[0]
        if len(self.c1_coords) <= 1 or width_hslice == 0:
            width_hslice = 1.0
            x_start_hslice = self.c1_coords[0] - 0.5

        # Apply same pixel-centre-aligned energy bounds as vslice (d_e computed above)
        if n_e > 1:
            height_hslice = self.energy_axis[-1] - self.energy_axis[0] + d_e
            y_start_hslice = self.energy_axis[0] - d_e / 2
        else:
            height_hslice = 1.0
            y_start_hslice = self.energy_axis[0] - 0.5

        rect_hslice = QtCore.QRectF(y_start_hslice, x_start_hslice, height_hslice, width_hslice)
        if self.hslice_img.image is not None:
            self.hslice_img.setRect(rect_hslice)
        
        vb_hslice = self.hslice_win.getViewBox()
        vb_hslice.invertY(self.c1_coords[0] > self.c1_coords[-1])
        self.hslice_win.setRange(xRange=(energy_min, energy_max), yRange=(c1_min, c1_max), padding=0)

        # Block region signals during scaling reset to avoid cascading updates/crashes
        self.spectrum_v_region.blockSignals(True)
        self.spectrum_h_region.blockSignals(True)

        if reset_spectrum:
            # Tab 1: Spectrum plot Energy range
            self.spectrum_win.setRange(xRange=(energy_min, energy_max), padding=0)
            self.spectrum_cross_v.setValue(self.selected_energy)

            # Center the vertical region lines dynamically inside the active energy range (30% to 70% of the span) if needed
            e_range = energy_max - energy_min
            curr_v_reg = self.spectrum_v_region.getRegion()
            v_reg_min, v_reg_max = curr_v_reg
            if v_reg_min < energy_min or v_reg_max > energy_max or v_reg_min >= v_reg_max or (v_reg_min == 0.0 and v_reg_max == 1.0):
                self.spectrum_v_region.setRegion((energy_min + 0.3 * e_range, energy_min + 0.7 * e_range))

        # Center the horizontal region lines dynamically inside the active counts range (30% to 70% of the span) if needed
        if np.any(self.data_measured_mask):
            measured_spectra = self.data_spectra[self.data_measured_mask]
            y_min, y_max = finite_plot_levels(measured_spectra)
        else:
            y_min = 200.0
            y_max = 15000.0

        y_range = y_max - y_min
        if y_range <= 0:
            y_range = 1.0

        if reset_spectrum:
            curr_h_reg = self.spectrum_h_region.getRegion()
            h_reg_min, h_reg_max = curr_h_reg
            if h_reg_min < 0 or h_reg_max <= h_reg_min or (h_reg_min == 0.0 and h_reg_max == 1.0):
                self.spectrum_h_region.setRegion((y_min + 0.3 * y_range, y_min + 0.7 * y_range))
            self.spectrum_win.setRange(yRange=(y_min - 0.1 * y_range, y_max + 0.1 * y_range), padding=0)

        self.spectrum_v_region.blockSignals(False)
        self.spectrum_h_region.blockSignals(False)

        # Tab 2: Top Gate Current Heatmap (X axis is C1, Y axis is C2)
        rect2 = QtCore.QRectF(x_start1, y_start1, width1, height1)
        if self.t2_heatmap_img.image is not None:
            self.t2_heatmap_img.setRect(rect2)
        
        vb_t2 = self.t2_heatmap_win.getViewBox()
        vb_t2.invertX(self.c1_coords[0] > self.c1_coords[-1])
        vb_t2.invertY(self.c2_coords[0] > self.c2_coords[-1])
        vb_t2.setRange(xRange=(c1_min, c1_max), yRange=(c2_min, c2_max), padding=0)

        # Tab 2: Top Gate Slice X ranges
        vb_t2_hslice = self.t2_hslice_win.getViewBox()
        vb_t2_hslice.invertX(self.c1_coords[0] > self.c1_coords[-1])
        vb_t2_hslice.setRange(xRange=(c1_min, c1_max), padding=0.05)
        
        vb_t2_vslice = self.t2_vslice_win.getViewBox()
        vb_t2_vslice.invertX(self.c2_coords[0] > self.c2_coords[-1])
        vb_t2_vslice.setRange(xRange=(c2_min, c2_max), padding=0.05)

        # Update bounds for Tab 2 plots:
        # Heatmap: X=C1, Y=C2
        self.t2_heatmap_win.getViewBox().update_bounds(c1_min, c1_max, c2_min, c2_max)
        # fixed C2 Slice: X=C1, Y=Current
        self.t2_hslice_win.getViewBox().update_bounds(c1_min, c1_max, None, None)
        # fixed C1 Slice: X=C2, Y=Current
        self.t2_vslice_win.getViewBox().update_bounds(c2_min, c2_max, None, None)

        # Tab 3: Bottom Gate Current Heatmap (X axis is C1, Y axis is C2)
        rect3 = QtCore.QRectF(x_start1, y_start1, width1, height1)
        if self.t3_heatmap_img.image is not None:
            self.t3_heatmap_img.setRect(rect3)
        
        vb_t3 = self.t3_heatmap_win.getViewBox()
        vb_t3.invertX(self.c1_coords[0] > self.c1_coords[-1])
        vb_t3.invertY(self.c2_coords[0] > self.c2_coords[-1])
        vb_t3.setRange(xRange=(c1_min, c1_max), yRange=(c2_min, c2_max), padding=0)

        # Tab 3: Bottom Gate Slice X ranges
        vb_t3_hslice = self.t3_hslice_win.getViewBox()
        vb_t3_hslice.invertX(self.c1_coords[0] > self.c1_coords[-1])
        vb_t3_hslice.setRange(xRange=(c1_min, c1_max), padding=0.05)
        
        vb_t3_vslice = self.t3_vslice_win.getViewBox()
        vb_t3_vslice.invertX(self.c2_coords[0] > self.c2_coords[-1])
        vb_t3_vslice.setRange(xRange=(c2_min, c2_max), padding=0.05)

        # Update bounds for Tab 3 plots:
        # Heatmap: X=C1, Y=C2
        self.t3_heatmap_win.getViewBox().update_bounds(c1_min, c1_max, c2_min, c2_max)
        # fixed C2 Slice: X=C1, Y=Current
        self.t3_hslice_win.getViewBox().update_bounds(c1_min, c1_max, None, None)
        # fixed C1 Slice: X=C2, Y=Current
        self.t3_vslice_win.getViewBox().update_bounds(c2_min, c2_max, None, None)

        # Initialize crosshairs to current point_gatemap values, or center of ranges if out of bounds
        p1 = self.point_gatemap_1
        p2 = self.point_gatemap_2
        
        # Check if stored values are within current coordinate ranges
        c1_range_min = min(self.c1_coords[0], self.c1_coords[-1])
        c1_range_max = max(self.c1_coords[0], self.c1_coords[-1])
        c2_range_min = min(self.c2_coords[0], self.c2_coords[-1])
        c2_range_max = max(self.c2_coords[0], self.c2_coords[-1])
        
        if not (c1_range_min <= p1 <= c1_range_max):
            p1 = (self.c1_coords[0] + self.c1_coords[-1]) / 2
        if not (c2_range_min <= p2 <= c2_range_max):
            p2 = (self.c2_coords[0] + self.c2_coords[-1]) / 2

        self.selected_energy = (self.energy_axis[0] + self.energy_axis[-1]) / 2
        self.update_crosshair_values(p1, p2, self.selected_energy)
        
        self.cross2_v.setValue(p1)
        self.cross2_h.setValue(p2)
        self.cross3_v.setValue(p1)
        self.cross3_h.setValue(p2)

        # All maps and slices need fresh ticks for the visible zoom range.
        for plot in (self.gatemap_win, self.vslice_win, self.hslice_win,
                     self.spectrum_win, self.t2_heatmap_win, self.t2_hslice_win,
                     self.t2_vslice_win, self.t3_heatmap_win, self.t3_hslice_win,
                     self.t3_vslice_win):
            configure_plot_axes(plot)

    def scale_plots(self):
        self.reset_slice_levels = True
        """Scale all plots: zoom completely out on coordinates, auto-range 1D plots, and set colorbar levels to full data range."""
        # 1. Reset all viewbox ranges to coordinate limits (c1, c2, energy) by calling reset_plot_scaling
        self.reset_plot_scaling(reset_spectrum=False)

        # 2. Scale interactive colorbars to full data range (min to max of the active slice/current data)
        # Restore the default Viridis palette for the main map and slices.
        self.gatemap_colorbar.gradient.loadPreset('viridis')
        self.vslice_colorbar.gradient.loadPreset('viridis')
        self.hslice_colorbar.gradient.loadPreset('viridis')
        self.t2_colorbar.gradient.loadPreset('viridis')
        self.t3_colorbar.gradient.loadPreset('viridis')

        if self.vslice_img.image is not None and self.vslice_img.image.size > 0:
            vslice_data = self.vslice_img.image
            min_v, max_v = finite_plot_levels(vslice_data)
            if min_v == max_v:
                max_v += 1.0
            self.vslice_colorbar.setLevels(min_v, max_v)

        if self.hslice_img.image is not None and self.hslice_img.image.size > 0:
            hslice_data = self.hslice_img.image
            min_h, max_h = finite_plot_levels(hslice_data)
            if min_h == max_h:
                max_h += 1.0
            self.hslice_colorbar.setLevels(min_h, max_h)

        if np.any(self.data_measured_mask):
            c1_meas = self.data_current_ch1[self.data_measured_mask]
            if len(c1_meas) > 0:
                min_c1, max_c1 = np.min(c1_meas), np.max(c1_meas)
                if min_c1 == max_c1:
                    max_c1 += 1e-12
                self.t2_colorbar.setLevels(min_c1, max_c1)

        if np.any(self.data_measured_mask):
            c2_meas = self.data_current_ch2[self.data_measured_mask]
            if len(c2_meas) > 0:
                min_c2, max_c2 = np.min(c2_meas), np.max(c2_meas)
                if min_c2 == max_c2:
                    max_c2 += 1e-12
                self.t3_colorbar.setLevels(min_c2, max_c2)

        # 3. Reset Y-axis for 1D slice curves and spectrum curve using manual scaling helper
        def manually_scale_y(plot_win, curve):
            y_data = curve.yData
            if y_data is not None:
                y_data = np.asarray(y_data)[np.isfinite(y_data)]
            if y_data is not None and len(y_data) > 0:
                y_min = np.min(y_data)
                y_max = np.max(y_data)
                y_range = y_max - y_min
                if y_range <= 0:
                    y_range = max(abs(y_min) * .1, 1e-12) if plot_win.getAxis('left').labelUnits == 'A' else 1.0
                plot_win.getViewBox().setRange(yRange=(y_min - 0.05 * y_range, y_max + 0.05 * y_range), padding=0)

        self.t2_hslice_win.getViewBox()._current_user_zoom = False
        manually_scale_y(self.t2_hslice_win, self.t2_hslice_curve)
        self.t2_vslice_win.getViewBox()._current_user_zoom = False
        manually_scale_y(self.t2_vslice_win, self.t2_vslice_curve)
        self.t3_hslice_win.getViewBox()._current_user_zoom = False
        manually_scale_y(self.t3_hslice_win, self.t3_hslice_curve)
        self.t3_vslice_win.getViewBox()._current_user_zoom = False
        manually_scale_y(self.t3_vslice_win, self.t3_vslice_curve)
        manually_scale_y(self.spectrum_win, self.spectrum_curve)

    def toggle_view_control_popup(self):
        if self.view_control_popup.isVisible():
            self.view_control_popup.hide()
        else:
            tab_idx = self.right_panel.currentIndex()
            self.view_control_popup.update_grid_layout(tab_idx)
            pos = self.btn_view_control.mapToGlobal(QtCore.QPoint(self.btn_view_control.width() - self.view_control_popup.width(), self.btn_view_control.height()))
            self.view_control_popup.move(pos)
            self.view_control_popup.show()
            self.view_control_popup.raise_()

    def on_tab_changed(self, tab_idx):
        if self.view_control_popup.isVisible():
            self.view_control_popup.update_grid_layout(tab_idx)

    def perform_autoscale(self):
        popup = self.view_control_popup
        mode = popup.get_axis_mode()
        if mode == "None":
            return
            
        tab_idx = self.right_panel.currentIndex()
        targets = []
        if tab_idx == 0:
            if popup.btn_grid_A.isChecked(): targets.append(self.gatemap_win)
            if popup.btn_grid_B.isChecked(): targets.append(self.vslice_win)
            if popup.btn_grid_C.isChecked(): targets.append(self.hslice_win)
            if popup.btn_grid_D.isChecked(): targets.append(self.spectrum_win)
        elif tab_idx == 1:
            if popup.btn_grid_AC.isChecked(): targets.append(self.t2_heatmap_win)
            if popup.btn_grid_B.isChecked(): targets.append(self.t2_hslice_win)
            if popup.btn_grid_D.isChecked(): targets.append(self.t2_vslice_win)
        elif tab_idx == 2:
            if popup.btn_grid_AC.isChecked(): targets.append(self.t3_heatmap_win)
            if popup.btn_grid_B.isChecked(): targets.append(self.t3_hslice_win)
            if popup.btn_grid_D.isChecked(): targets.append(self.t3_vslice_win)
            
        for plot in targets:
            vb = plot.getViewBox()
            if mode == "X":
                vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
                if tab_idx != 0:
                    vb.autoRange(axis=pg.ViewBox.XAxis)
            elif mode == "Y":
                vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                if tab_idx != 0:
                    vb.autoRange(axis=pg.ViewBox.YAxis)
            elif mode == "Both":
                vb.enableAutoRange(enable=True)
                if tab_idx != 0:
                    vb.autoRange()

    def get_processed_spectra(self):
        raw_spectra = self.data_spectra
        if raw_spectra is None or raw_spectra.size == 0:
            return raw_spectra
            
        option = "Signal"
        if hasattr(self, "view_control_popup"):
            option = self.view_control_popup.get_source_option()
            
        if option == "Signal":
            return raw_spectra
            
        single_win = getattr(self, "single_spectra_window", None)
        if single_win is None:
            return raw_spectra
            
        bg_x = getattr(single_win, "bg_x", None)
        bg_y = getattr(single_win, "bg_y", None)
        ref_x = getattr(single_win, "ref_x", None)
        ref_y = getattr(single_win, "ref_y", None)
        
        sweep_wavelengths = 1239.84193 / self.energy_axis
        
        def get_interp_spectrum(spec_x, spec_y):
            if spec_x is None or spec_y is None or len(spec_x) == 0:
                return np.zeros_like(sweep_wavelengths)
            x_arr = np.asarray(spec_x)
            y_arr = np.asarray(spec_y)
            sort_idx = np.argsort(x_arr)
            return np.interp(sweep_wavelengths, x_arr[sort_idx], y_arr[sort_idx])
            
        bg_interp = get_interp_spectrum(bg_x, bg_y)
        ref_interp = get_interp_spectrum(ref_x, ref_y)
        
        if option in ("Signal - BG", "Signal - Background"):
            return raw_spectra - bg_interp
        elif option == "(S-R)/(R-B)":
            denom = ref_interp - bg_interp
            denom = np.where(denom == 0.0, 1e-9, denom)
            return (raw_spectra - ref_interp) / denom
            
        return raw_spectra

    def on_plot_source_changed(self):
        if not np.any(self.data_measured_mask):
            return
        self.update_heatmaps_and_slices()
        self.on_crosshair_moved()

    # --- Live Rendering updates ---
    def update_heatmaps_and_slices(self):
        """Calculates integrated counts and currents to update heatmaps and slices."""
        if not np.any(self.data_measured_mask):
            return

        # 1. Gate Map Integration
        e_min, e_max = sorted(self.spectrum_v_region.getRegion())
        h_min, h_max = sorted(self.spectrum_h_region.getRegion())
        # Find indices corresponding to this energy window
        idx_range = np.where((self.energy_axis >= e_min) & (self.energy_axis <= e_max))[0]
        
        if len(idx_range) > 0:
            # Clip/clamp the intensity spectra within the horizontal region limits and integrate
            sub_spectra = self.get_processed_spectra()[:, :, idx_range]
            clamped_spectra = np.clip(sub_spectra, h_min, h_max) - h_min
            integrated_map = np.sum(clamped_spectra, axis=2)
        else:
            integrated_map = np.zeros((len(self.c2_coords), len(self.c1_coords)))

        # Update Gate Map Image
        self.gatemap_img.setImage(integrated_map)
        
        # Calculate dynamic color ranges
        if np.any(self.data_measured_mask):
            measured_values = integrated_map[self.data_measured_mask]
            min_val, max_val = np.min(measured_values), np.max(measured_values)
            if min_val == max_val:
                max_val += 1
            self.gatemap_colorbar.setLevels(min_val, max_val)

        # Update Tab 2 and Tab 3 current heatmaps
        self.t2_heatmap_img.setImage(self.data_current_ch1)
        if np.any(self.data_measured_mask):
            c1_meas = self.data_current_ch1[self.data_measured_mask]
            self.t2_colorbar.setLevels(np.min(c1_meas), np.max(c1_meas))

        self.t3_heatmap_img.setImage(self.data_current_ch2)
        if np.any(self.data_measured_mask):
            c2_meas = self.data_current_ch2[self.data_measured_mask]
            self.t3_colorbar.setLevels(np.min(c2_meas), np.max(c2_meas))



    def on_integration_region_changed(self):
        """Called dynamically as the shaded spectral region is dragged."""
        self.update_heatmaps_and_slices()



    def update_crosshair_values(self, c1_val, c2_val, energy_val):
        """Helper to synchronize all Tab 1 crosshair values with signals blocked to avoid feedback loops."""
        self.selected_energy = energy_val
        self.point_gatemap_1 = c1_val
        self.point_gatemap_2 = c2_val

        # Block signals to avoid infinite recursion loops
        self.cross1_v.blockSignals(True)
        self.cross1_h.blockSignals(True)
        self.vslice_cross_v.blockSignals(True)
        self.vslice_cross_h.blockSignals(True)
        self.hslice_cross_v.blockSignals(True)
        self.hslice_cross_h.blockSignals(True)
        self.spectrum_cross_v.blockSignals(True)

        self.cross1_v.setValue(c1_val)
        self.cross1_h.setValue(c2_val)
        
        self.vslice_cross_v.setValue(energy_val)
        self.vslice_cross_h.setValue(c2_val)
        
        self.hslice_cross_v.setValue(energy_val)
        self.hslice_cross_h.setValue(c1_val)

        # Unblock signals
        self.cross1_v.blockSignals(False)
        self.cross1_h.blockSignals(False)
        self.vslice_cross_v.blockSignals(False)
        self.vslice_cross_h.blockSignals(False)
        self.hslice_cross_v.blockSignals(False)
        self.hslice_cross_h.blockSignals(False)
        self.spectrum_cross_v.blockSignals(False)

    def on_crosshair_moved(self):
        """Update Tab 1 slice cuts and spectrum plot when crosshairs are dragged."""
        if not np.any(self.data_measured_mask):
            return

        c1_val = self.cross1_v.value()
        c2_val = self.cross1_h.value()
        energy_val = self.selected_energy

        self.update_crosshair_values(c1_val, c2_val, energy_val)
        self.refresh_slice_plots(c1_val, c2_val)

    def on_vslice_crosshair_moved(self):
        """Called when crosshairs in vslice_win (top right) are dragged."""
        if not np.any(self.data_measured_mask):
            return
        c1_val = self.cross1_v.value()
        c2_val = self.vslice_cross_h.value()
        energy_val = self.vslice_cross_v.value()
        
        self.update_crosshair_values(c1_val, c2_val, energy_val)
        self.refresh_slice_plots(c1_val, c2_val)

    def on_hslice_crosshair_moved(self):
        """Called when crosshairs in hslice_win (bottom left) are dragged."""
        if not np.any(self.data_measured_mask):
            return
        energy_val = self.hslice_cross_v.value()
        c2_val = self.cross1_h.value()
        c1_val = self.hslice_cross_h.value()
        
        self.update_crosshair_values(c1_val, c2_val, energy_val)
        self.refresh_slice_plots(c1_val, c2_val)

    def on_spectrum_crosshair_moved(self):
        """Called when the energy crosshair on the spectrum plot is dragged."""
        if not np.any(self.data_measured_mask):
            return
        energy_val = self.spectrum_cross_v.value()
        c1_val = self.cross1_v.value()
        c2_val = self.cross1_h.value()
        
        self.update_crosshair_values(c1_val, c2_val, energy_val)
        self.refresh_slice_plots(c1_val, c2_val)

    def refresh_slice_plots(self, c1_val, c2_val):
        """Redraws slice plots based on updated coordinates."""
        if not np.any(self.data_measured_mask) or self.data_spectra.ndim < 3 or self.c1_coords.size == 0 or self.c2_coords.size == 0:
            return
        # Find closest grid indexes
        idx_c1 = np.argmin(np.abs(self.c1_coords - c1_val))
        idx_c2 = np.argmin(np.abs(self.c2_coords - c2_val))

        processed_spectra = self.get_processed_spectra()

        # 1. Update 1D Spectrum at crosshair intersection
        spectrum = processed_spectra[idx_c2, idx_c1, :]
        self.spectrum_curve.setData(self.energy_axis, spectrum)

        # Update crosshair markers on the spectrum plot
        self.spectrum_cross_v.setValue(self.selected_energy)
        energy_idx = np.argmin(np.abs(self.energy_axis - self.selected_energy))
        if energy_idx < len(spectrum):
            y_val = spectrum[energy_idx]

        
        # Manually scale the Y-axis to fit only the spectrum curve if autoRange is still enabled
        vb_spec = self.spectrum_win.getViewBox()
        finite_spectrum = spectrum[np.isfinite(spectrum)]
        if vb_spec.state['autoRange'][1] and finite_spectrum.size:
            y_min = np.min(finite_spectrum)
            y_max = np.max(finite_spectrum)
            y_range = y_max - y_min
            if y_range <= 0:
                y_range = 1.0
            vb_spec.setRange(yRange=(y_min - 0.1 * y_range, y_max + 0.1 * y_range), padding=0)

        # 2. Update Vertical Slice (all C2 coordinates at fixed C1)
        vslice_data = processed_spectra[:, idx_c1, :]
        vslice_data_linear = np.zeros_like(vslice_data)
        if vslice_data.size > 0:
            for idx in range(vslice_data.shape[0]):
                vslice_data_linear[idx, :] = np.interp(self.linear_energy_axis, self.energy_axis, vslice_data[idx, :])
        auto_levels = getattr(self, "reset_slice_levels", True)
        self.vslice_img.setImage(vslice_data_linear, autoLevels=auto_levels)
        finite_vslice = vslice_data[np.isfinite(vslice_data)]
        if auto_levels and finite_vslice.size > 0:
            min_v, max_v = np.min(finite_vslice), np.max(finite_vslice)
            if min_v == max_v:
                max_v += 1
            self.vslice_colorbar.setLevels(min_v, max_v)

        # 3. Update Horizontal Slice (all C1 coordinates at fixed C2)
        hslice_data = processed_spectra[idx_c2, :, :]
        hslice_data_linear = np.zeros_like(hslice_data)
        if hslice_data.size > 0:
            for idx in range(hslice_data.shape[0]):
                hslice_data_linear[idx, :] = np.interp(self.linear_energy_axis, self.energy_axis, hslice_data[idx, :])
        self.hslice_img.setImage(hslice_data_linear, autoLevels=auto_levels)
        finite_hslice = hslice_data[np.isfinite(hslice_data)]
        if auto_levels and finite_hslice.size > 0:
            min_h, max_h = np.min(finite_hslice), np.max(finite_hslice)
            if min_h == max_h:
                max_h += 1
            self.hslice_colorbar.setLevels(min_h, max_h)
            
        # Update spectrum plot title to show the gate voltage coordinates at the current crosshair
        c1_label = self.c1_name_edit.text() if hasattr(self, 'c1_name_edit') and self.c1_name_edit.text() else 'C1'
        c2_label = self.c2_name_edit.text() if hasattr(self, 'c2_name_edit') and self.c2_name_edit.text() else 'C2'
        self.spectrum_win.setTitle(
            f"<span style='font-size:9pt'>CCD Spectrum &nbsp;&mdash;&nbsp; "
            f"{c1_label} = {c1_val:.4f} V &nbsp;|&nbsp; {c2_label} = {c2_val:.4f} V</span>"
        )
        self.reset_slice_levels = False

    def on_tab2_crosshair_moved(self):
        """Update Tab 2 slices when its crosshair is dragged."""
        if not np.any(self.data_measured_mask):
            return

        c1_val = self.cross2_v.value()
        c2_val = self.cross2_h.value()

        # Find closest indices in grid
        idx_c1 = np.argmin(np.abs(self.c1_coords - c1_val))
        idx_c2 = np.argmin(np.abs(self.c2_coords - c2_val))

        # Horizontal slice: Current vs C1 (at fixed C2)
        curr_vs_c1 = self.data_current_ch1[idx_c2, :]
        update_current_slice(self.t2_hslice_win, self.t2_hslice_curve, self.c1_coords, curr_vs_c1, self.data_measured_mask[idx_c2, :])

        # Vertical slice: Current vs C2 (at fixed C1)
        curr_vs_c2 = self.data_current_ch1[:, idx_c1]
        update_current_slice(self.t2_vslice_win, self.t2_vslice_curve, self.c2_coords, curr_vs_c2, self.data_measured_mask[:, idx_c1])

    def on_tab3_crosshair_moved(self):
        """Update Tab 3 slices when its crosshair is dragged."""
        if not np.any(self.data_measured_mask):
            return

        c1_val = self.cross3_v.value()
        c2_val = self.cross3_h.value()

        # Find closest indices in grid
        idx_c1 = np.argmin(np.abs(self.c1_coords - c1_val))
        idx_c2 = np.argmin(np.abs(self.c2_coords - c2_val))

        # Horizontal slice: Current vs C1 (at fixed C2)
        curr_vs_c1 = self.data_current_ch2[idx_c2, :]
        update_current_slice(self.t3_hslice_win, self.t3_hslice_curve, self.c1_coords, curr_vs_c1, self.data_measured_mask[idx_c2, :])

        # Vertical slice: Current vs C2 (at fixed C1)
        curr_vs_c2 = self.data_current_ch2[:, idx_c1]
        update_current_slice(self.t3_vslice_win, self.t3_vslice_curve, self.c2_coords, curr_vs_c2, self.data_measured_mask[:, idx_c1])

    # --- HDF5 Data Storage Saving ---
    def save_sweep_data(self, file_path=None):
        if getattr(self, "plotter_mode", False):
            return
        """Save the measured dataset (spectra and currents) to a structured HDF5 file."""
        if isinstance(file_path, bool):
            file_path = None
        if not np.any(self.data_measured_mask):
            if file_path is None:
                QtWidgets.QMessageBox.warning(self, "No Data", "There is no sweep data to save.")
            return

        try:
            data_dir = measurement_data_dir(
                parent_dir, getattr(self, "measurement_started_at", None)
            )
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_filename = self.default_sweep_filename()
            default_filename = dated_measurement_filename(default_filename, getattr(self, "measurement_started_at", None))

            if file_path is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save Measurement Data",
                    os.path.join(data_dir, default_filename),
                    "HDF5 files (*.h5);;All files (*.*)"
                )
                if not filename:
                    return
                filename = dated_measurement_path(filename, getattr(self, "measurement_started_at", None))
            else:
                filename = file_path

            with h5py.File(filename, "w") as f:
                f.attrs["device_modes"] = json.dumps(getattr(self, "measurement_device_modes", {}))
                f.attrs["filename_central_wavelength_nm"] = self.filename_wavelength_box.value()
                f.attrs["filename_wavelength_source"] = "manual"
                # Save coordinates and grid geometry
                f.create_dataset("c1_coords", data=self.c1_coords)
                f.create_dataset("c2_coords", data=self.c2_coords)
                f.create_dataset("v1_grid", data=self.v1_grid)
                f.create_dataset("v2_grid", data=self.v2_grid)
                f.create_dataset("energy_axis", data=self.energy_axis)

                # Save measured datasets
                f.create_dataset("wavelength_axis", data=self.wavelength_axis)
                f.create_dataset("spectra", data=self.data_spectra)
                if getattr(self, "data_raw_spectra", None) is not None and self.data_raw_spectra.size > 0:
                    f.create_dataset("raw_intensities", data=self.data_raw_spectra)
                f.create_dataset("current_ch1", data=self.data_current_ch1)
                f.create_dataset("current_ch2", data=self.data_current_ch2)
                f.create_dataset("measured_volt_ch1", data=self.data_measured_volt_ch1)
                f.create_dataset("measured_volt_ch2", data=self.data_measured_volt_ch2)
                if self.data_measured_b is not None and self.data_measured_b.size > 0:
                    f.create_dataset("measured_b", data=self.data_measured_b)
                f.create_dataset("measured_mask", data=self.data_measured_mask)

                # Save background and reference if available
                single_win = getattr(self, "single_spectra_window", None)
                if single_win is not None:
                    if getattr(single_win, "bg_x", None) is not None and getattr(single_win, "bg_y", None) is not None:
                        bg_grp = f.create_group("background")
                        bg_grp.create_dataset("wavelength", data=single_win.bg_x)
                        bg_grp.create_dataset("intensity", data=single_win.bg_y)
                        bg_raw = getattr(single_win, "bg_y_raw", None)
                        if bg_raw is None:
                            bg_raw = np.array([single_win.bg_y])
                        bg_grp.create_dataset("raw_intensities", data=bg_raw)
                    if getattr(single_win, "ref_x", None) is not None and getattr(single_win, "ref_y", None) is not None:
                        ref_grp = f.create_group("reference")
                        ref_grp.create_dataset("wavelength", data=single_win.ref_x)
                        ref_grp.create_dataset("intensity", data=single_win.ref_y)
                        ref_raw = getattr(single_win, "ref_y_raw", None)
                        if ref_raw is None:
                            ref_raw = np.array([single_win.ref_y])
                        ref_grp.create_dataset("raw_intensities", data=ref_raw)

                # Save coefficients and metadata
                f.attrs["coeff_a"] = self.coeff_a.value()
                f.attrs["coeff_b"] = self.coeff_b.value()
                f.attrs["coeff_c"] = self.coeff_c.value()
                f.attrs["coeff_d"] = self.coeff_d.value()
                limits = getattr(self, 'measurement_current_compliance', {})
                for axis in (1, 2):
                    f.attrs[f'current_compliance_ch{axis}_A'] = limits.get(axis, self._compliance_setting(axis))
                # Store full value_storage JSON as a dataset for full state recovery
                try:
                    storage_path = os.path.join(parent_dir, "value_storage")
                    if os.path.exists(storage_path):
                        with open(storage_path, "r") as f_vs:
                            vs_data = json.load(f_vs)
                        
                        # Create metadata group/folder
                        meta_grp = f.create_group("metadata")
                        meta_grp.create_dataset("value_storage", data=json.dumps(vs_data), dtype=h5py.string_dtype(encoding="utf-8"))
                        
                        # Add exposure time, grating, and central wavelength as separate datasets
                        shared = vs_data.get("shared", {})
                        single_spec = vs_data.get("single_spectra", {})
                        exposure_time = shared.get("exposure_time", 0.0)
                        grating = single_spec.get("grating", "")
                        central_wavelength = single_spec.get("wavelength", 0.0)
                        
                        meta_grp.create_dataset("exposure_time", data=exposure_time)
                        meta_grp.create_dataset("grating", data=grating, dtype=h5py.string_dtype(encoding="utf-8"))
                        meta_grp.create_dataset("central_wavelength", data=central_wavelength)
                except Exception as e:
                    print(f"Failed to embed value_storage in HDF5: {e}")
                f.attrs["c1_name"] = self.c1_name_edit.text()
                f.attrs["c2_name"] = self.c2_name_edit.text()
                f.attrs["exposure_time"] = self.exposure_box.value()
                f.attrs["settling_time"] = self.settling_box.value()
                f.attrs["timestamp"] = timestamp
                f.attrs["measurement_started_at"] = measurement_start_text(
                    getattr(self, "measurement_started_at", None)
                )

            self.status_lbl.setText(f"Status: Saved {os.path.basename(filename)}")
            if file_path is None:
                QtWidgets.QMessageBox.information(self, "Success", f"Data saved successfully to:\n{filename}")
            else:
                print(f"Auto-saved data to {filename}")
            self.save_storage()
        except Exception as e:
            if file_path is None:
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save dataset:\n{e}")
            else:
                print(f"Auto-save failed: {e}")

    def trigger_autosave(self, was_interrupted=False):
        if getattr(self, "plotter_mode", False):
            return
        if getattr(self, "_autosaved_this_run", False):
            return
        if not np.any(self.data_measured_mask):
            return
        autosave_dir = measurement_data_dir(
            parent_dir, getattr(self, "measurement_started_at", None), autosave=True
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_interrupted" if was_interrupted else ""
        filename = f"autosave_gate_sweep_{timestamp[-6:]}{suffix}.h5"
        filename = dated_measurement_filename(filename, getattr(self, "measurement_started_at", None))
        file_path = os.path.join(autosave_dir, filename)
        try:
            self.save_sweep_data(file_path=file_path)
            self._autosaved_this_run = True
        except Exception as e:
            print(f"Auto-save trigger failed: {e}")

    def closeEvent(self, event):
        """Cleanup thread objects on exit to avoid dangling process exceptions."""
        if hasattr(self, 'live_status_worker') and self.live_status_worker and self.live_status_worker.isRunning():
            self.live_status_worker.stop()
            self.live_status_worker.wait()
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        if self.ramp_worker and self.ramp_worker.isRunning():
            self.ramp_worker.stop()
            self.ramp_worker.wait()
        self.save_storage()
        from gui_safety import uninstall_parameter_wheel_guard
        uninstall_parameter_wheel_guard(self)
        event.accept()


if __name__ == '__main__':
    print("Initializing physical station for sweep GUI...")
    station = initialize_station(simulated=False, simulated_devices=simulated_devices)
    
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(station)
    win.show()
    sys.exit(app.exec())
