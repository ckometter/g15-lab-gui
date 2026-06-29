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
    from .utils import EnhancedViewBox, calculate_expected_duration
except ImportError:
    import sys
    import os
    _measurements_dir = os.path.dirname(os.path.abspath(__file__))
    if _measurements_dir not in sys.path:
        sys.path.insert(0, _measurements_dir)
    from utils import EnhancedViewBox, calculate_expected_duration

# Ensure parent directory is in path for importing station_setup
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from station_setup import initialize_station, simulated_devices

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


def transform_c_to_v(c1, c2, a, b, c, d):
    """
    Convert virtual gate coordinates (c1, c2) to physical voltages (v1, v2)
    using the linear mapping:
      v1 = a * c1 + b * c2
      v2 = c * c1 + d * c2
    """
    v1 = a * c1 + b * c2
    v2 = c * c1 + d * c2
    return v1, v2


class SingleRampWorker(QtCore.QThread):
    """
    Lightweight background worker to ramp a single channel to a voltage or current setpoint
    without freezing the main GUI thread.
    """
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, channel, target_val, param_name="volt", parent=None):
        super().__init__(parent)
        self.channel = channel
        self.target_val = target_val
        self.param_name = param_name

    def stop(self):
        self.channel.ramp_stop_requested = True

    def run(self):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self.channel.ramp_stop_requested = False
                if self.param_name == "curr":
                    self.channel.set_r.curr(self.target_val)
                else:
                    self.channel.set_r.volt(self.target_val)
                break
            except Exception as e:
                if attempt == max_retries:
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


class SweepWorker(QtCore.QThread):
    """
    Background worker thread that runs the gate sweep measurement.
    Communicates back to the main thread via Qt signals.
    """
    point_measured = QtCore.pyqtSignal(int, int, float, float, float, float, np.ndarray, float, float, float, float, float)
    progress_updated = QtCore.pyqtSignal(int, int)
    status_updated = QtCore.pyqtSignal(str)
    voltage_updated = QtCore.pyqtSignal(float, float)
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

    def pause(self):
        self.mutex.lock()
        self._is_paused = True
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
            smu = self.station.smu
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
                    self.voltage_updated.emit(v1, v2)
                    while True:
                        if not self.check_pause_and_stop():
                            return
                        
                        smu.ch1.ramp_stop_requested = False
                        smu.ch2.ramp_stop_requested = False
                        
                        max_retries = 3
                        for attempt in range(1, max_retries + 1):
                            try:
                                smu.ch1.set_r.volt(v1)
                                break
                            except Exception as ex:
                                if attempt == max_retries:
                                    raise ex
                                print(f"SweepWorker: Retrying SMU1 set voltage (attempt {attempt+1}/{max_retries}) due to error: {ex}")
                                try:
                                    smu.visa_handle.clear()
                                except Exception:
                                    pass
                                time.sleep(0.1)
                                
                        for attempt in range(1, max_retries + 1):
                            try:
                                smu.ch2.set_r.volt(v2)
                                break
                            except Exception as ex:
                                if attempt == max_retries:
                                    raise ex
                                print(f"SweepWorker: Retrying SMU2 set voltage (attempt {attempt+1}/{max_retries}) due to error: {ex}")
                                try:
                                    smu.visa_handle.clear()
                                except Exception:
                                    pass
                                time.sleep(0.1)
                        
                        # If ramping was interrupted, loop again to handle pause/stop check
                        if getattr(smu.ch1, "ramp_stop_requested", False) or getattr(smu.ch2, "ramp_stop_requested", False):
                            continue
                        break

                    # 3. Wait settling time
                    if settling_time > 0:
                        sleep_end = time.time() + settling_time
                        while time.time() < sleep_end:
                            if not self.check_pause_and_stop():
                                return
                            time.sleep(0.05)

                    # 4. Acquire spectrum with multi-exposure accumulation
                    self.status_updated.emit(
                        f"Status: Acquiring spectrum ({num_spectra}x, {accum_method})..."
                    )
                    spectrum = None
                    if not self.check_pause_and_stop():
                        return
                    try:
                        res = camera.remote_lf.get_spectrum()
                        if res is not None:
                            intensities, _ = res
                            intensities_arr = np.array(intensities, dtype=float)
                            if intensities_arr.ndim == 2:
                                if do_average or accum_method == "Average":
                                    intensities_arr = np.mean(intensities_arr, axis=0)
                                else:
                                    intensities_arr = np.sum(intensities_arr, axis=0)
                            spectrum = intensities_arr
                    except Exception as e:
                        if self._is_running:
                            print(f"SweepWorker error fetching spectrum: {e}")
                    
                    if spectrum is None:
                        spectrum = np.zeros(num_pixels)

                    if not self.check_pause_and_stop():
                        return

                    # Query hardware output states before measuring current and voltage
                    # to prevent automated query-based power-on
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

                    point_count += 1
                    self.point_measured.emit(i, j, c1, c2, v1, v2, spectrum, curr_ch1, curr_ch2, meas_v1, meas_v2, meas_b)
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

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self._stop_requested = False
        self._is_paused = False
        self._actively_polling = False

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
                
                self.status_updated.emit(b_val, v1_val, v2_val)
            except Exception:
                pass
            finally:
                self._actively_polling = False
            
            self.msleep(200) # 5 Hz

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, station, embed=False):
        super().__init__()
        self.station = station
        self._embed = embed
        self.setWindowTitle("Electrostatic Gate Control & 2D Sweep Spectrometer")
        self.setMinimumSize(1280, 850)

        # Application State
        self.is_running = False
        self.is_paused = False
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
        self.data_current_ch1 = np.array([])
        self.data_current_ch2 = np.array([])
        self.data_measured_mask = np.array([]) # Tracks which points are measured
        self.data_measured_b = np.array([])

        # Energy calibration axis
        self.energy_axis = np.linspace(1.5, 3.0, 512) # Dummy range initialized at startup
        self.energy_sort_idx = np.arange(512)
        self.selected_energy = 2.25

        # Gate map point coordinates
        self.point_gatemap_1 = 0.0
        self.point_gatemap_2 = 0.0

        # Load stored settings or defaults
        self.load_storage()

        self.setup_ui()
        self.apply_theme()
        
        # Read live output states from physical SMU at startup to sync buttons
        try:
            smu = self.station.smu
            ch1_out = bool(smu.ch1.output())
            ch2_out = bool(smu.ch2.output())
            
            self.ch1_out_btn.blockSignals(True)
            self.ch1_out_btn.setChecked(ch1_out)
            self.ch1_out_btn.setText(f"Ch1 Output: {'ON' if ch1_out else 'OFF'}")
            self.ch1_out_btn.setStyleSheet(f"background-color: {'#10b981' if ch1_out else '#f1f5f9'}; color: {'white' if ch1_out else '#475569'}; border: 1px solid {'#10b981' if ch1_out else '#cbd5e1'}; font-weight: bold;")
            self.ch1_out_btn.blockSignals(False)

            self.ch2_out_btn.blockSignals(True)
            self.ch2_out_btn.setChecked(ch2_out)
            self.ch2_out_btn.setText(f"Ch2 Output: {'ON' if ch2_out else 'OFF'}")
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
            self.on_mode_changed()
            
            # Sync compliance values as well
            if ch1_mode == "CURR":
                self.ch1_compliance.setValue(smu.ch1.volt_limit())
            else:
                self.ch1_compliance.setValue(smu.ch1.curr_limit())
                
            if ch2_mode == "CURR":
                self.ch2_compliance.setValue(smu.ch2.volt_limit())
            else:
                self.ch2_compliance.setValue(smu.ch2.curr_limit())
                
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
        hist.layout.setColumnFixedWidth(2, 35)  # Axis (numbers scale)
        
        hist.gradient.setMinimumWidth(0)
        hist.gradient.setMaximumWidth(12)
        hist.vb.setMinimumWidth(0)
        hist.vb.setMaximumWidth(45)
        hist.axis.setMinimumWidth(0)
        hist.axis.setMaximumWidth(35)
        
        hist.setMinimumWidth(92)
        hist.setMaximumWidth(100)
        
        if not interactive:
            hist.region.setMovable(False)
            for line in hist.region.lines:
                line.setMovable(False)
            hist.gradient.setEnabled(False)
            hist.vb.setMouseEnabled(x=False, y=False)
            
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
            "ramp_step": 0.0005,
            "c1_name": "Vc1",
            "c2_name": "Vc2",
            "v1_compliance": 0.01,
            "v2_compliance": 0.01,
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
                            self.stored_settings[key] = dg_data.get("v1_compliance", dg_data.get("c1_compliance", raw_data.get("v1_compliance", raw_data.get("c1_compliance", defaults[key]))))
                        elif key == "v2_compliance":
                            self.stored_settings[key] = dg_data.get("v2_compliance", dg_data.get("c2_compliance", raw_data.get("v2_compliance", raw_data.get("c2_compliance", defaults[key]))))
                        elif key == "c1_name":
                            self.stored_settings[key] = shared_data.get("c1_name", dg_data.get("c1_name", raw_data.get("c1_name", defaults["c1_name"])))
                        elif key == "c2_name":
                            self.stored_settings[key] = shared_data.get("c2_name", dg_data.get("c2_name", raw_data.get("c2_name", defaults["c2_name"])))
                        elif key in ("coeff_a", "coeff_b", "coeff_c", "coeff_d"):
                            legacy_key = f"dg_coeff_{key[-1]}"
                            self.stored_settings[key] = shared_data.get(key, dg_data.get(legacy_key, raw_data.get(legacy_key, defaults[key])))
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
            camera = getattr(self.station, "camera", None)
            current_data["shared"]["spectrometer"] = getattr(camera, "spectrometer_type", "lightfieldG15")
            current_data["dual_gate"]["ramprate1"] = self.stored_settings.get("ramprate1", 0.1)
            current_data["dual_gate"]["ramprate2"] = self.stored_settings.get("ramprate2", 0.1)
            current_data["dual_gate"]["ramp_step"] = self.stored_settings.get("ramp_step", 0.0005)
            current_data["dual_gate"]["c1_name"] = self.stored_settings.get("c1_name", "Vc1")
            current_data["dual_gate"]["c2_name"] = self.stored_settings.get("c2_name", "Vc2")
            
            # New inputs
            if hasattr(self, "ch1_compliance"):
                current_data["dual_gate"]["v1_compliance"] = self.ch1_compliance.value()
            if hasattr(self, "ch2_compliance"):
                current_data["dual_gate"]["v2_compliance"] = self.ch2_compliance.value()
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
        self.ch1_compliance.setValue(self.stored_settings.get("v1_compliance", 0.01))
        self.ch1_compliance.blockSignals(False)

        self.ch2_compliance.blockSignals(True)
        self.ch2_compliance.setValue(self.stored_settings.get("v2_compliance", 0.01))
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
                self.conn_status_lbl.setStyleSheet("color: #f9e2af; font-weight: bold; font-family: Consolas, monospace;")
            else:
                self.conn_status_lbl.setText("Keysight SMU: CONNECTED (Physical)  ")
                self.conn_status_lbl.setStyleSheet("color: #a6e3a1; font-weight: bold; font-family: Consolas, monospace;")
            status_bar.addPermanentWidget(self.conn_status_lbl)
        else:
            # When embedded, still create the label for potential internal use but do not add to status bar
            self.conn_status_lbl = QtWidgets.QLabel()
            if self.station.smu.simulated:
                self.conn_status_lbl.setText("Keysight SMU: SIMULATED (Connection Locked/Failed)  ")
                self.conn_status_lbl.setStyleSheet("color: #f9e2af; font-weight: bold; font-family: Consolas, monospace;")
            else:
                self.conn_status_lbl.setText("Keysight SMU: CONNECTED (Physical)  ")
                self.conn_status_lbl.setStyleSheet("color: #a6e3a1; font-weight: bold; font-family: Consolas, monospace;")
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

        # --- ROW 1: Keysight Channel 1 & Keysight Channel 2 next to each other ---
        smu_row = QtWidgets.QHBoxLayout()
        smu_row.setSpacing(10)

        # Channel 1 Box
        ch1_box = QtWidgets.QGroupBox("[ Keysight Channel 1 ]")
        ch1_layout = QtWidgets.QVBoxLayout(ch1_box)
        ch1_layout.setSpacing(8)

        # 1. Output ON/OFF
        self.ch1_out_btn = QtWidgets.QPushButton("Ch1 Output: OFF")
        self.ch1_out_btn.setCheckable(True)
        self.ch1_out_btn.toggled.connect(self.toggle_ch1_output)
        ch1_layout.addWidget(self.ch1_out_btn)

        # Grid for Setpoint, Source Mode, Ramp Rate, Compliance
        grid1 = QtWidgets.QGridLayout()
        grid1.setSpacing(6)
        
        # 2. Setpoint
        self.ch1_setpoint_lbl = QtWidgets.QLabel("Setpoint (V):")
        grid1.addWidget(self.ch1_setpoint_lbl, 0, 0)
        self.ch1_setpoint = QtWidgets.QDoubleSpinBox()
        self.ch1_setpoint.setRange(-210.0, 210.0)
        self.ch1_setpoint.setDecimals(4)
        self.ch1_setpoint.setSingleStep(0.001)
        self.ch1_setpoint.setKeyboardTracking(False)
        self.ch1_setpoint.lineEdit().returnPressed.connect(self.on_ch1_setpoint_enter)
        grid1.addWidget(self.ch1_setpoint, 0, 1)

        # 3. Source Mode
        grid1.addWidget(QtWidgets.QLabel("Source Mode:"), 1, 0)
        self.ch1_mode = QtWidgets.QComboBox()
        self.ch1_mode.addItems(["VOLT", "CURR"])
        self.ch1_mode.currentTextChanged.connect(self.on_mode_changed)
        grid1.addWidget(self.ch1_mode, 1, 1)

        # 4. Ramp Rate
        grid1.addWidget(QtWidgets.QLabel("Ramp Rate (V/s):"), 2, 0)
        self.ch1_ramp_rate = QtWidgets.QDoubleSpinBox()
        self.ch1_ramp_rate.setRange(1e-6, 100.0)
        self.ch1_ramp_rate.setSingleStep(1e-6)
        self.ch1_ramp_rate.setValue(self.stored_settings.get("ramprate1", 0.1))
        self.ch1_ramp_rate.setKeyboardTracking(False)
        self.ch1_ramp_rate.editingFinished.connect(self.on_ch1_ramping_params_changed)
        grid1.addWidget(self.ch1_ramp_rate, 2, 1)

        # 5. Compliance
        self.ch1_compliance_lbl = QtWidgets.QLabel("Compliance (A):")
        grid1.addWidget(self.ch1_compliance_lbl, 3, 0)
        self.ch1_compliance = QtWidgets.QDoubleSpinBox()
        self.ch1_compliance.setRange(1e-9, 3.0)
        self.ch1_compliance.setDecimals(6)
        self.ch1_compliance.setValue(self.stored_settings.get("v1_compliance", 0.01))
        self.ch1_compliance.setSingleStep(0.001)
        self.ch1_compliance.setKeyboardTracking(False)
        self.ch1_compliance.editingFinished.connect(self.on_compliance_changed)
        grid1.addWidget(self.ch1_compliance, 3, 1)
        
        ch1_layout.addLayout(grid1)

        # 6. Last Reading Subbox
        ch1_read_box = QtWidgets.QGroupBox("Last Reading")
        ch1_read_layout = QtWidgets.QVBoxLayout(ch1_read_box)
        ch1_read_layout.setSpacing(4)
        
        self.ch1_live_volt_lbl = QtWidgets.QLabel("Voltage: None")
        self.ch1_live_volt_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch1_read_layout.addWidget(self.ch1_live_volt_lbl)
        
        self.ch1_live_curr_lbl = QtWidgets.QLabel("Current: None")
        self.ch1_live_curr_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch1_read_layout.addWidget(self.ch1_live_curr_lbl)
        
        self.ch1_read_btn = QtWidgets.QPushButton("Read Channel One")
        self.ch1_read_btn.clicked.connect(self.read_channel_one)
        ch1_read_layout.addWidget(self.ch1_read_btn)
        
        ch1_layout.addWidget(ch1_read_box)
        smu_row.addWidget(ch1_box)

        # Channel 2 Box
        ch2_box = QtWidgets.QGroupBox("[ Keysight Channel 2 ]")
        ch2_layout = QtWidgets.QVBoxLayout(ch2_box)
        ch2_layout.setSpacing(8)

        # 1. Output ON/OFF
        self.ch2_out_btn = QtWidgets.QPushButton("Ch2 Output: OFF")
        self.ch2_out_btn.setCheckable(True)
        self.ch2_out_btn.toggled.connect(self.toggle_ch2_output)
        ch2_layout.addWidget(self.ch2_out_btn)

        # Grid for Setpoint, Source Mode, Ramp Rate, Compliance
        grid2 = QtWidgets.QGridLayout()
        grid2.setSpacing(6)
        
        # 2. Setpoint
        self.ch2_setpoint_lbl = QtWidgets.QLabel("Setpoint (V):")
        grid2.addWidget(self.ch2_setpoint_lbl, 0, 0)
        self.ch2_setpoint = QtWidgets.QDoubleSpinBox()
        self.ch2_setpoint.setRange(-210.0, 210.0)
        self.ch2_setpoint.setDecimals(4)
        self.ch2_setpoint.setSingleStep(0.001)
        self.ch2_setpoint.setKeyboardTracking(False)
        self.ch2_setpoint.lineEdit().returnPressed.connect(self.on_ch2_setpoint_enter)
        grid2.addWidget(self.ch2_setpoint, 0, 1)

        # 3. Source Mode
        grid2.addWidget(QtWidgets.QLabel("Source Mode:"), 1, 0)
        self.ch2_mode = QtWidgets.QComboBox()
        self.ch2_mode.addItems(["VOLT", "CURR"])
        self.ch2_mode.currentTextChanged.connect(self.on_mode_changed)
        grid2.addWidget(self.ch2_mode, 1, 1)

        # 4. Ramp Rate
        grid2.addWidget(QtWidgets.QLabel("Ramp Rate (V/s):"), 2, 0)
        self.ch2_ramp_rate = QtWidgets.QDoubleSpinBox()
        self.ch2_ramp_rate.setRange(1e-6, 100.0)
        self.ch2_ramp_rate.setSingleStep(1e-6)
        self.ch2_ramp_rate.setValue(self.stored_settings.get("ramprate2", 0.1))
        self.ch2_ramp_rate.setKeyboardTracking(False)
        self.ch2_ramp_rate.editingFinished.connect(self.on_ch2_ramping_params_changed)
        grid2.addWidget(self.ch2_ramp_rate, 2, 1)

        # 5. Compliance
        self.ch2_compliance_lbl = QtWidgets.QLabel("Compliance (A):")
        grid2.addWidget(self.ch2_compliance_lbl, 3, 0)
        self.ch2_compliance = QtWidgets.QDoubleSpinBox()
        self.ch2_compliance.setRange(1e-9, 3.0)
        self.ch2_compliance.setDecimals(6)
        self.ch2_compliance.setValue(self.stored_settings.get("v2_compliance", 0.01))
        self.ch2_compliance.setSingleStep(0.001)
        self.ch2_compliance.setKeyboardTracking(False)
        self.ch2_compliance.editingFinished.connect(self.on_compliance_changed)
        grid2.addWidget(self.ch2_compliance, 3, 1)
        
        ch2_layout.addLayout(grid2)

        # 6. Last Reading Subbox
        ch2_read_box = QtWidgets.QGroupBox("Last Reading")
        ch2_read_layout = QtWidgets.QVBoxLayout(ch2_read_box)
        ch2_read_layout.setSpacing(4)
        
        self.ch2_live_volt_lbl = QtWidgets.QLabel("Voltage: None")
        self.ch2_live_volt_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch2_read_layout.addWidget(self.ch2_live_volt_lbl)
        
        self.ch2_live_curr_lbl = QtWidgets.QLabel("Current: None")
        self.ch2_live_curr_lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
        ch2_read_layout.addWidget(self.ch2_live_curr_lbl)
        
        self.ch2_read_btn = QtWidgets.QPushButton("Read Channel Two")
        self.ch2_read_btn.clicked.connect(self.read_channel_two)
        ch2_read_layout.addWidget(self.ch2_read_btn)
        
        ch2_layout.addWidget(ch2_read_box)
        smu_row.addWidget(ch2_box)

        left_layout.addLayout(smu_row)

        # --- ROW 2: Control Variable 1 & Control Variable 2 next to each other ---
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(10)

        # Control 1
        # Load name from storage if available, else default to "Virtual Gate 1"
        c1_default_name = self.stored_settings.get("c1_name", "Virtual Gate 1")
        self.ctrl_c1_box = QtWidgets.QGroupBox(f"[ {c1_default_name} ]")
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

        c1_grid.addWidget(QtWidgets.QLabel("Points:"), 2, 0)
        self.c1_pts = ScientificPointSpinBox()
        self.c1_pts.setRange(1, 1000)
        self.c1_pts.setValue(self.stored_settings.get("c1_pts", 21))
        self.c1_pts.setKeyboardTracking(True)
        self.c1_pts.lineEdit().setPlaceholderText("e.g. 1e2")
        c1_grid.addWidget(self.c1_pts, 2, 1)

        c1_grid.addWidget(QtWidgets.QLabel("Step:"), 3, 0)
        self.c1_step_lbl = QtWidgets.QLabel("0.0 V")
        c1_grid.addWidget(self.c1_step_lbl, 3, 1)
        
        ctrl_c1_layout.addLayout(c1_grid)
        ctrl_row.addWidget(self.ctrl_c1_box)

        # Control 2
        # Load name from storage if available, else default to "Virtual Gate 2"
        c2_default_name = self.stored_settings.get("c2_name", "Virtual Gate 2")
        self.ctrl_c2_box = QtWidgets.QGroupBox(f"[ {c2_default_name} ]")
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

        c2_grid.addWidget(QtWidgets.QLabel("Points:"), 2, 0)
        self.c2_pts = ScientificPointSpinBox()
        self.c2_pts.setRange(1, 1000)
        self.c2_pts.setValue(self.stored_settings.get("c2_pts", 21))
        self.c2_pts.setKeyboardTracking(True)
        self.c2_pts.lineEdit().setPlaceholderText("e.g. 1e2")
        c2_grid.addWidget(self.c2_pts, 2, 1)

        c2_grid.addWidget(QtWidgets.QLabel("Step:"), 3, 0)
        self.c2_step_lbl = QtWidgets.QLabel("0.0 V")
        c2_grid.addWidget(self.c2_step_lbl, 3, 1)

        ctrl_c2_layout.addLayout(c2_grid)
        ctrl_row.addWidget(self.ctrl_c2_box)
        
        left_layout.addLayout(ctrl_row)

        # Bind events to update step sizes and plot scaling dynamically
        for box in (self.c1_start, self.c1_end, self.c1_pts):
            box.editingFinished.connect(self.on_sweep_range_changed)
        for box in (self.c2_start, self.c2_end, self.c2_pts):
            box.editingFinished.connect(self.on_sweep_range_changed)
        self.update_step_calculations()

        # --- ROW 3: Gate Variables (full width) ---
        gate_box = QtWidgets.QGroupBox("[ Gate Variables ]")
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

        # Formula 1 layout: Voltage 1 = a * C1 + b * C2
        f1_layout = QtWidgets.QHBoxLayout()
        f1_layout.setSpacing(5)
        self.v1_formula_label = QtWidgets.QLabel("Voltage 1 = ")
        f1_layout.addWidget(self.v1_formula_label)

        self.coeff_a = QtWidgets.QDoubleSpinBox()
        self.coeff_a.setValue(self.stored_settings.get("coeff_a", 1.0))
        self.coeff_a.setRange(-100, 100)
        self.coeff_a.setDecimals(4)
        self.coeff_a.setSingleStep(0.01)
        self.coeff_a.setKeyboardTracking(False)
        f1_layout.addWidget(self.coeff_a)

        self.v1_c1_label = QtWidgets.QLabel(f" * {c1_default_name}  + ")
        f1_layout.addWidget(self.v1_c1_label)

        self.coeff_b = QtWidgets.QDoubleSpinBox()
        self.coeff_b.setValue(self.stored_settings.get("coeff_b", 0.0))
        self.coeff_b.setRange(-100, 100)
        self.coeff_b.setDecimals(4)
        self.coeff_b.setSingleStep(0.01)
        self.coeff_b.setKeyboardTracking(False)
        f1_layout.addWidget(self.coeff_b)

        self.v1_c2_label = QtWidgets.QLabel(f" * {c2_default_name}")
        f1_layout.addWidget(self.v1_c2_label)
        f1_layout.addStretch()

        gate_layout.addLayout(f1_layout)

        # Formula 2 layout: Voltage 2 = c * C1 + d * C2
        f2_layout = QtWidgets.QHBoxLayout()
        f2_layout.setSpacing(5)
        self.v2_formula_label = QtWidgets.QLabel("Voltage 2 = ")
        f2_layout.addWidget(self.v2_formula_label)

        self.coeff_c = QtWidgets.QDoubleSpinBox()
        self.coeff_c.setValue(self.stored_settings.get("coeff_c", 0.0))
        self.coeff_c.setRange(-100, 100)
        self.coeff_c.setDecimals(4)
        self.coeff_c.setSingleStep(0.01)
        self.coeff_c.setKeyboardTracking(False)
        f2_layout.addWidget(self.coeff_c)

        self.v2_c1_label = QtWidgets.QLabel(f" * {c1_default_name}  + ")
        f2_layout.addWidget(self.v2_c1_label)

        self.coeff_d = QtWidgets.QDoubleSpinBox()
        self.coeff_d.setValue(self.stored_settings.get("coeff_d", 1.0))
        self.coeff_d.setRange(-100, 100)
        self.coeff_d.setDecimals(4)
        self.coeff_d.setSingleStep(0.01)
        self.coeff_d.setKeyboardTracking(False)
        f2_layout.addWidget(self.coeff_d)

        self.v2_c2_label = QtWidgets.QLabel(f" * {c2_default_name}")
        f2_layout.addWidget(self.v2_c2_label)
        f2_layout.addStretch()

        gate_layout.addLayout(f2_layout)
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

        # Other Settings Box
        other_box = QtWidgets.QGroupBox("[ Other ]")
        other_layout = QtWidgets.QVBoxLayout(other_box)
        other_layout.setSpacing(6)
        
        other_layout.addWidget(QtWidgets.QLabel("Sweep Configuration:"))
        self.sweep_mode_combo = QtWidgets.QComboBox()
        self.sweep_mode_combo.addItems(["Hold C1, sweep C2", "Hold C2, sweep C1"])
        self.sweep_mode_combo.setCurrentText(self.stored_settings.get("sweep_mode", "Hold C1, sweep C2"))
        self.sweep_mode_combo.currentTextChanged.connect(self.on_sweep_range_changed)
        other_layout.addWidget(self.sweep_mode_combo)

        self.serpentine_box = QtWidgets.QCheckBox("Serpentine (Snake) Sweep")
        self.serpentine_box.setChecked(self.stored_settings.get("serpentine", True))
        self.serpentine_box.stateChanged.connect(self.save_storage)
        other_layout.addWidget(self.serpentine_box)

        # Timing and Ramping Grid
        timing_ramp_grid = QtWidgets.QGridLayout()
        timing_ramp_grid.setSpacing(6)
        timing_ramp_grid.addWidget(QtWidgets.QLabel("Exposure (s):"), 0, 0)
        self.exposure_box = QtWidgets.QDoubleSpinBox()
        self.exposure_box.setRange(0.001, 10.0)
        self.exposure_box.setDecimals(3)
        self.exposure_box.setValue(self.stored_settings.get("exposure_time", 0.1))
        self.exposure_box.setKeyboardTracking(False)
        self.exposure_box.editingFinished.connect(self.on_timing_params_changed)
        timing_ramp_grid.addWidget(self.exposure_box, 0, 1)

        timing_ramp_grid.addWidget(QtWidgets.QLabel("Settling (s):"), 1, 0)
        self.settling_box = QtWidgets.QDoubleSpinBox()
        self.settling_box.setRange(0.0, 10.0)
        self.settling_box.setDecimals(3)
        self.settling_box.setValue(self.stored_settings.get("settling_time", 0.05))
        self.settling_box.setKeyboardTracking(False)
        self.settling_box.editingFinished.connect(self.on_timing_params_changed)
        timing_ramp_grid.addWidget(self.settling_box, 1, 1)

        # Ramp Step is managed only in value_storage and cannot be changed in the GUI
        
        other_layout.addLayout(timing_ramp_grid)

        bottom_row.addWidget(other_box)

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
        meas_box = QtWidgets.QGroupBox("[ Measurement Control ]")
        meas_layout = QtWidgets.QVBoxLayout(meas_box)
        meas_layout.setSpacing(6)
        
        self.begin_btn = QtWidgets.QPushButton("▶ Begin Sweep")
        self.begin_btn.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; border: 1px solid #10b981;")
        self.begin_btn.clicked.connect(self.begin_sweep)
        meas_layout.addWidget(self.begin_btn)

        self.pause_btn = QtWidgets.QPushButton("⏸ Interrupt")
        self.pause_btn.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold; border: 1px solid #ef4444;")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.on_interrupt_clicked)
        meas_layout.addWidget(self.pause_btn)

        self.reset_btn = QtWidgets.QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self.reset_sweep)
        meas_layout.addWidget(self.reset_btn)

        self.save_btn = QtWidgets.QPushButton("💾 Save Dataset")
        self.save_btn.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; border: 1px solid #3b82f6;")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_sweep_data)
        meas_layout.addWidget(self.save_btn)

        self.status_lbl = QtWidgets.QLabel("Status: Idle")
        self.status_lbl.setStyleSheet("color: #4f46e5; font-style: italic; font-weight: bold;")
        meas_layout.addWidget(self.status_lbl)

        self.lbl_current_b = QtWidgets.QLabel("B = --‑‑ T")
        self.lbl_current_b.setStyleSheet("color: #1e293b; font-weight: bold;")
        meas_layout.addWidget(self.lbl_current_b)

        self.lbl_target_vc1 = QtWidgets.QLabel("Vc1 = --‑‑ V")
        self.lbl_target_vc1.setStyleSheet("color: #1e293b; font-weight: bold;")
        meas_layout.addWidget(self.lbl_target_vc1)

        self.lbl_target_vc2 = QtWidgets.QLabel("Vc2 = --‑‑ V")
        self.lbl_target_vc2.setStyleSheet("color: #1e293b; font-weight: bold;")
        meas_layout.addWidget(self.lbl_target_vc2)

        self.lbl_expected_end = QtWidgets.QLabel("Expected End: --:--:--")
        self.lbl_expected_end.setStyleSheet("color: #1e293b; font-weight: bold;")
        meas_layout.addWidget(self.lbl_expected_end)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        meas_layout.addWidget(self.progress_bar)

        bottom_row.addWidget(meas_box)
        left_layout.addLayout(bottom_row)

        # --- RIGHT TAB PANEL SECTIONS ---

        # Tab 1: Gate Sweep (CCD Spectra)
        self.tab1 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab1, "Gate Sweep (Spectra)")
        self.setup_tab1()

        # Tab 2: Top Gate current
        self.tab2 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab2, "Top Gate (Ch1 Current)")
        self.setup_tab2()

        # Tab 3: Bottom Gate current
        self.tab3 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab3, "Bottom Gate (Ch2 Current)")
        self.setup_tab3()

        # Add View Control button as corner widget of QTabWidget
        self.btn_view_control = QtWidgets.QPushButton("View Control ▾")
        self.btn_view_control.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
                color: #1e293b;
            }
            QPushButton:hover {
                background-color: #f8fafc;
                border-color: #94a3b8;
            }
        """)
        self.right_panel.setCornerWidget(self.btn_view_control, QtCore.Qt.Corner.TopRightCorner)
        self.btn_view_control.clicked.connect(self.toggle_view_control_popup)
        self.right_panel.currentChanged.connect(self.on_tab_changed)

        # Update initial variable names
        self.update_variable_names()
        self.update_coordinates_from_ui()
        self.reset_plot_scaling()

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
        # Do not add crosshair lines to Vertical Slice plot (Plot B) per user request
        # self.vslice_win.addItem(self.vslice_cross_v)
        # self.vslice_win.addItem(self.vslice_cross_h)
        # self.vslice_cross_v.sigPositionChanged.connect(self.on_vslice_crosshair_moved)
        # self.vslice_cross_h.sigPositionChanged.connect(self.on_vslice_crosshair_moved)

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
        self.spectrum_curve = self.spectrum_win.plot(pen=pg.mkPen('#7c3aed', width=2))
        self.spectrum_win.setLabel('left', 'Counts')
        self.spectrum_win.setLabel('bottom', 'Energy', units='eV')
        self.spectrum_win.showGrid(x=True, y=True, alpha=0.3)

        # Draggable vertical lines (spectrum_v1 and spectrum_v2) to highlight region (X-axis integration window)
        self.spectrum_v_region = pg.LinearRegionItem(
            orientation='vertical',
            values=(2.4, 2.6),
            brush=pg.mkBrush(16, 185, 129, 40), # green transparent fill
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
            brush=pg.mkBrush(16, 185, 129, 40), # green transparent fill
            pen=pg.mkPen('#10b981', width=1.5)  # continuous green line
        )
        self.spectrum_win.addItem(self.spectrum_h_region, ignoreBounds=True)
        self.spectrum_h1 = self.spectrum_h_region.lines[0]
        self.spectrum_h2 = self.spectrum_h_region.lines[1]
        self.spectrum_h_region.sigRegionChanged.connect(self.on_integration_region_changed)


        layout.addWidget(self.spectrum_win, 1, 1)

    def setup_tab2(self):
        """Setup layout for Ch1 Current vs physical Top Gate and virtual C2"""
        layout = QtWidgets.QHBoxLayout(self.tab2)
        layout.setContentsMargins(5, 5, 5, 5)

        # Left: Heatmap
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
        
        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
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

        layout.addWidget(self.t2_heatmap_win, 3)

        # Right: Slice plots (vertical layout)
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(6)

        self.t2_hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C2")
        self.t2_hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_hslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t2_hslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t2_hslice_curve = self.t2_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t2_hslice_win.setLabel('left', 'Current', units='A')
        self.t2_hslice_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        self.t2_hslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t2_hslice_win)

        self.t2_vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C1")
        self.t2_vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_vslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t2_vslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t2_vslice_curve = self.t2_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t2_vslice_win.setLabel('left', 'Current', units='A')
        self.t2_vslice_win.setLabel('bottom', 'Virtual Gate C2', units='V')
        self.t2_vslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t2_vslice_win)

        layout.addLayout(right_layout, 2)

    def setup_tab3(self):
        """Setup layout for Ch2 Current vs physical Bottom Gate and virtual C1"""
        layout = QtWidgets.QHBoxLayout(self.tab3)
        layout.setContentsMargins(5, 5, 5, 5)

        # Left: Heatmap
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
        
        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
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

        layout.addWidget(self.t3_heatmap_win, 3)

        # Right: Slice plots (vertical layout)
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(6)

        self.t3_hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C2")
        self.t3_hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_hslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t3_hslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t3_hslice_curve = self.t3_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t3_hslice_win.setLabel('left', 'Current', units='A')
        self.t3_hslice_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        self.t3_hslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t3_hslice_win)

        self.t3_vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed C1")
        self.t3_vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_vslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t3_vslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t3_vslice_curve = self.t3_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t3_vslice_win.setLabel('left', 'Current', units='A')
        self.t3_vslice_win.setLabel('bottom', 'Virtual Gate C2', units='V')
        self.t3_vslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t3_vslice_win)

        layout.addLayout(right_layout, 2)

    def apply_theme(self):
        """Premium light theme styling (white backgrounds, dark text, clean borders)"""
        qss = """
        QMainWindow, QWidget {
            background-color: #ffffff;
            color: #1e293b;
            font-family: "Segoe UI", Arial;
            font-size: 13px;
        }
        QGroupBox {
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            margin-top: 14px;
            font-weight: bold;
            color: #4f46e5;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 2px 4px;
        }
        QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
            background-color: #ffffff;
            color: #1e293b;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
            padding: 4px;
        }
        QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
            border: 1px solid #6366f1;
        }
        QPushButton {
            background-color: #f1f5f9;
            color: #1e293b;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #e2e8f0;
            border-color: #94a3b8;
        }
        QPushButton:pressed {
            background-color: #cbd5e1;
        }
        QTabWidget::pane {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background-color: #ffffff;
            padding: 5px;
        }
        QTabBar::tab {
            background-color: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
            border-bottom-color: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            color: #4f46e5;
            border-color: #cbd5e1;
            font-weight: bold;
        }
        QTabBar::tab:hover {
            background-color: #e2e8f0;
            color: #1e293b;
        }
        QScrollBar:vertical {
            border: none;
            background: #f1f5f9;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #cbd5e1;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """
        self.setStyleSheet(qss)

    # --- Live Parameter Readout ---
    # --- Live Parameter Readout (On Demand) ---
    def read_channel_one(self):
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

    def read_channel_two(self):
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

    def on_live_status_updated(self, b_val, v1_val, v2_val):
        self._last_polled_b = b_val
        self.lbl_current_b.setText(f"B = {b_val:.4f} T")
        self.lbl_target_vc1.setText(f"V1 = {v1_val:.4f} V")
        self.lbl_target_vc2.setText(f"V2 = {v2_val:.4f} V")

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

        self.v1_grid = np.zeros((c2_pts_actual, c1_pts_actual))
        self.v2_grid = np.zeros((c2_pts_actual, c1_pts_actual))
        for j, c2_val in enumerate(self.c2_coords):
            for i, c1_val in enumerate(self.c1_coords):
                v1, v2 = transform_c_to_v(c1_val, c2_val, a, b, c, d)
                self.v1_grid[j, i] = v1
                self.v2_grid[j, i] = v2

    def on_sweep_range_changed(self):
        """Called dynamically when sweep parameters are changed in the UI."""
        self.update_step_calculations()
        if not self.is_running:
            self.update_coordinates_from_ui()
            self.reset_plot_scaling()
            self.save_storage()

    def update_variable_names(self):
        """Dynamically update control group box titles, labels, equations and plot axes on user edit."""
        c1_name = self.c1_name_edit.text().strip() or "Vc1"
        c2_name = self.c2_name_edit.text().strip() or "Vc2"

        # Update group box titles
        self.ctrl_c1_box.setTitle(f"[ {c1_name} ]")
        self.ctrl_c2_box.setTitle(f"[ {c2_name} ]")

        # Update formulas labels
        self.v1_c1_label.setText(f" * {c1_name}  + ")
        self.v1_c2_label.setText(f" * {c2_name}")
        self.v2_c1_label.setText(f" * {c1_name}  + ")
        self.v2_c2_label.setText(f" * {c2_name}")

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
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.pause()
        self.station.live_status_paused = True
        unit = "A" if param_name == "curr" else "V"
        self.status_lbl.setText(f"Status: Ramping {ch_name} to {target} {unit}...")
        self.set_inputs_enabled(False)
        self.pause_btn.setEnabled(True)
        
        self.ramp_worker = SingleRampWorker(channel, target, param_name)
        self.ramp_worker.finished.connect(self.on_ramp_finished)
        self.ramp_worker.error.connect(self.on_ramp_error)
        self.ramp_worker.start()

    def on_ramp_finished(self):
        self.station.live_status_paused = False
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        if self.ramp_worker and getattr(self.ramp_worker.channel, "ramp_stop_requested", False):
            self.status_lbl.setText("Status: Ramping interrupted.")
        else:
            self.status_lbl.setText("Status: Ramping complete.")
        self.set_inputs_enabled(True)
        self.pause_btn.setEnabled(False)

    def on_ramp_error(self, err_msg):
        self.station.live_status_paused = False
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        self.status_lbl.setText(f"Status: Ramping error")
        self.set_inputs_enabled(True)
        self.pause_btn.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Ramping Error", f"Error ramping channel:\n{err_msg}")

    def toggle_ch1_output(self, enabled):
        try:
            self.station.smu.ch1.output(enabled)
            self.ch1_out_btn.setText(f"Ch1 Output: {'ON' if enabled else 'OFF'}")
            self.ch1_out_btn.setStyleSheet(f"background-color: {'#10b981' if enabled else '#f1f5f9'}; color: {'white' if enabled else '#475569'}; border: 1px solid {'#10b981' if enabled else '#cbd5e1'}; font-weight: bold;")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hardware Error", f"Failed to toggle Ch1 Output:\n{e}")

    def toggle_ch2_output(self, enabled):
        try:
            self.station.smu.ch2.output(enabled)
            self.ch2_out_btn.setText(f"Ch2 Output: {'ON' if enabled else 'OFF'}")
            self.ch2_out_btn.setStyleSheet(f"background-color: {'#10b981' if enabled else '#f1f5f9'}; color: {'white' if enabled else '#475569'}; border: 1px solid {'#10b981' if enabled else '#cbd5e1'}; font-weight: bold;")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hardware Error", f"Failed to toggle Ch2 Output:\n{e}")

    def on_compliance_changed(self):
        try:
            # Channel 1
            if self.ch1_mode.currentText() == "CURR":
                self.station.smu.ch1.volt_limit(self.ch1_compliance.value())
            else:
                self.station.smu.ch1.curr_limit(self.ch1_compliance.value())
        except Exception:
            pass

        try:
            # Channel 2
            if self.ch2_mode.currentText() == "CURR":
                self.station.smu.ch2.volt_limit(self.ch2_compliance.value())
            else:
                self.station.smu.ch2.curr_limit(self.ch2_compliance.value())
        except Exception:
            pass

        self.save_storage()

    def on_mode_changed(self):
        # Update physical instrument modes
        try:
            self.station.smu.ch1.source_mode(self.ch1_mode.currentText())
            self.station.smu.ch2.source_mode(self.ch2_mode.currentText())
        except Exception:
            pass

        # Update Channel 1 GUI labels, ranges, and steps
        if self.ch1_mode.currentText() == "CURR":
            self.ch1_setpoint_lbl.setText("Setpoint (A):")
            self.ch1_setpoint.setRange(-3.0, 3.0)
            self.ch1_setpoint.setDecimals(6)
            self.ch1_setpoint.setSingleStep(1e-6)
            self.ch1_compliance_lbl.setText("Compliance (V):")
            self.ch1_compliance.setRange(1e-9, 210.0)
            self.ch1_compliance.setDecimals(4)
            self.ch1_compliance.setSingleStep(0.1)
        else:
            self.ch1_setpoint_lbl.setText("Setpoint (V):")
            self.ch1_setpoint.setRange(-210.0, 210.0)
            self.ch1_setpoint.setDecimals(4)
            self.ch1_setpoint.setSingleStep(0.001)
            self.ch1_compliance_lbl.setText("Compliance (A):")
            self.ch1_compliance.setRange(1e-9, 3.0)
            self.ch1_compliance.setDecimals(6)
            self.ch1_compliance.setSingleStep(0.001)

        # Update Channel 2 GUI labels, ranges, and steps
        if self.ch2_mode.currentText() == "CURR":
            self.ch2_setpoint_lbl.setText("Setpoint (A):")
            self.ch2_setpoint.setRange(-3.0, 3.0)
            self.ch2_setpoint.setDecimals(6)
            self.ch2_setpoint.setSingleStep(1e-6)
            self.ch2_compliance_lbl.setText("Compliance (V):")
            self.ch2_compliance.setRange(1e-9, 210.0)
            self.ch2_compliance.setDecimals(4)
            self.ch2_compliance.setSingleStep(0.1)
        else:
            self.ch2_setpoint_lbl.setText("Setpoint (V):")
            self.ch2_setpoint.setRange(-210.0, 210.0)
            self.ch2_setpoint.setDecimals(4)
            self.ch2_setpoint.setSingleStep(0.001)
            self.ch2_compliance_lbl.setText("Compliance (A):")
            self.ch2_compliance.setRange(1e-9, 3.0)
            self.ch2_compliance.setDecimals(6)
            self.ch2_compliance.setSingleStep(0.001)

        # Sync compliance limits with new modes
        self.on_compliance_changed()

    def on_ch1_ramping_params_changed(self):
        """Save Channel 1 settings to value_storage and sync with hardware driver."""
        rate = self.ch1_ramp_rate.value()
        self.stored_settings["ramprate1"] = rate
        self.save_storage()
        try:
            self.station.smu.ch1.ramp_rate(rate)
        except Exception:
            pass

    def on_ch2_ramping_params_changed(self):
        """Save Channel 2 settings to value_storage and sync with hardware driver."""
        rate = self.ch2_ramp_rate.value()
        self.stored_settings["ramprate2"] = rate
        self.save_storage()
        try:
            self.station.smu.ch2.ramp_rate(rate)
        except Exception:
            pass


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

        try:
            wls = self.station.camera.remote_lf.get_wavelengths_cache()
            if wls is not None and len(wls) > 0:
                wavelengths = np.array(wls)
                num_pixels = len(wavelengths)
        except Exception as e:
            print(f"Could not get wavelengths cache: {e}")
            
        if wavelengths is None:
            try:
                res = self.station.camera.remote_lf.get_spectrum()
                if res is not None and len(res) >= 2 and len(res[0]) >= 1:
                    num_pixels = len(res[0][0])
                    if len(res[1]) > 0:
                        wavelengths = np.array(res[1])
            except Exception as e:
                print(f"Could not fetch test spectrum for pixel count: {e}")
                
        if wavelengths is None:
            # Fallback to manual calibration around center wavelength
            pixels_arr = np.arange(num_pixels)
            wavelengths = mono_wl + (pixels_arr - num_pixels // 2) * 0.05

        self.data_spectra = np.zeros((c2_pts_actual, c1_pts_actual, num_pixels))
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

        # Reset plots and set axes ranges
        self.reset_plot_scaling()

        # Start sweep thread
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
            'num_pixels': num_pixels
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
        self.total_paused_time = 0.0
        self.pause_start_time = None

        self.lbl_expected_end.setText("Expected End: (waiting for measurements)")

        self.is_running = True
        self.is_paused = False
        self._autosaved_this_run = False
        
        self.begin_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ Interrupt")
        self.reset_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.set_inputs_enabled(False)

        self.sweep_worker = SweepWorker(self.station, params)
        self.sweep_worker.point_measured.connect(self.on_point_measured)
        self.sweep_worker.progress_updated.connect(self.on_progress_updated)
        self.sweep_worker.status_updated.connect(self.status_lbl.setText)
        self.sweep_worker.voltage_updated.connect(self.on_sweep_voltage_updated)
        self.sweep_worker.finished.connect(self.on_sweep_finished)
        self.sweep_worker.error_occurred.connect(self.on_sweep_error)
        self.sweep_worker.start()

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
            self.sweep_worker.resume()
        else:
            self.is_paused = True
            self.pause_btn.setText("▶ Resume")
            self.begin_btn.setEnabled(True)
            self.save_btn.setEnabled(True if np.any(self.data_measured_mask) else False)
            self.status_lbl.setText("Status: Paused (Interrupt)")
            self.pause_start_time = time.time()
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
        self.begin_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ Interrupt")
        self.save_btn.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.set_inputs_enabled(True)
        
        self.progress_bar.setValue(0)
        self.lbl_expected_end.setText("Expected End: --:--:--")
        self.status_lbl.setText("Status: Reset complete")

    def on_point_measured(self, i, j, c1, c2, v1, v2, spectrum, curr_ch1, curr_ch2, meas_v1, meas_v2, meas_b):
        # Sort spectrum to align with increasing energy values
        sorted_spectrum = spectrum[self.energy_sort_idx]
        
        # Save measurement point
        self.data_spectra[j, i, :] = sorted_spectrum
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
        self.lbl_target_vc1.setText(f"V1 = {meas_v1:.4f} V")
        self.lbl_target_vc2.setText(f"V2 = {meas_v2:.4f} V")

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

    def on_sweep_voltage_updated(self, v1, v2):
        self.ch1_setpoint.setValue(v1)
        self.ch2_setpoint.setValue(v2)

    def on_sweep_finished(self):
        self.trigger_autosave(was_interrupted=False)
        self.is_running = False
        self.begin_btn.setEnabled(True)
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
        self.begin_btn.setEnabled(True)
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
    def reset_plot_scaling(self):
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
        self.spectrum_win.setLimits(xMin=energy_min, xMax=energy_max)

        rect1 = QtCore.QRectF(x_start1, y_start1, width1, height1)
        if self.gatemap_img.image is not None:
            self.gatemap_img.setRect(rect1)
        
        vb1 = self.gatemap_win.getViewBox()
        vb1.invertX(self.c1_coords[0] > self.c1_coords[-1])
        vb1.invertY(self.c2_coords[0] > self.c2_coords[-1])
        self.gatemap_win.setRange(xRange=(c1_min, c1_max), yRange=(c2_min, c2_max), padding=0)

        # Tab 1: Vertical Slice (Y axis is C2, X axis is Energy)
        width_vslice = self.energy_axis[-1] - self.energy_axis[0]
        x_start_vslice = self.energy_axis[0]
        if len(self.energy_axis) <= 1 or width_vslice == 0:
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

        height_hslice = self.energy_axis[-1] - self.energy_axis[0]
        y_start_hslice = self.energy_axis[0]
        if len(self.energy_axis) <= 1 or height_hslice == 0:
            height_hslice = 1.0
            y_start_hslice = self.energy_axis[0] - 0.5

        rect_hslice = QtCore.QRectF(y_start_hslice, x_start_hslice, height_hslice, width_hslice)
        if self.hslice_img.image is not None:
            self.hslice_img.setRect(rect_hslice)
        
        vb_hslice = self.hslice_win.getViewBox()
        vb_hslice.invertY(self.c1_coords[0] > self.c1_coords[-1])
        self.hslice_win.setRange(xRange=(energy_min, energy_max), yRange=(c1_min, c1_max), padding=0)

        # Tab 1: Spectrum plot Energy range
        self.spectrum_win.setRange(xRange=(energy_min, energy_max), padding=0)
        
        # Block region signals during scaling reset to avoid cascading updates/crashes
        self.spectrum_v_region.blockSignals(True)
        self.spectrum_h_region.blockSignals(True)

        # Center the vertical region lines dynamically inside the active energy range (30% to 70% of the span) if needed
        e_range = energy_max - energy_min
        curr_v_reg = self.spectrum_v_region.getRegion()
        v_reg_min, v_reg_max = curr_v_reg
        if v_reg_min < energy_min or v_reg_max > energy_max or v_reg_min >= v_reg_max or (v_reg_min == 0.0 and v_reg_max == 1.0):
            self.spectrum_v_region.setRegion((energy_min + 0.3 * e_range, energy_min + 0.7 * e_range))

        # Center the horizontal region lines dynamically inside the active counts range (30% to 70% of the span) if needed
        if np.any(self.data_measured_mask):
            measured_spectra = self.data_spectra[self.data_measured_mask]
            y_min = np.min(measured_spectra)
            y_max = np.max(measured_spectra)
        else:
            y_min = 200.0
            y_max = 15000.0

        y_range = y_max - y_min
        if y_range <= 0:
            y_range = 1.0

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

        # Set custom tick labels showing start, end, and intermediate coordinate values
        def generate_ticks(start, end):
            if start == end:
                return [(start, f"{start:.4f}".rstrip('0').rstrip('.'))]
            vals = np.linspace(start, end, 5)
            ticks = []
            for val in vals:
                ticks.append((val, f"{val:.4f}".rstrip('0').rstrip('.')))
            return ticks

        ticks_c1 = generate_ticks(self.c1_coords[0], self.c1_coords[-1])
        ticks_c2 = generate_ticks(self.c2_coords[0], self.c2_coords[-1])

        self.gatemap_win.getAxis('bottom').setTicks([ticks_c1])
        self.gatemap_win.getAxis('left').setTicks([ticks_c2])
        self.vslice_win.getAxis('left').setTicks([ticks_c2])
        self.hslice_win.getAxis('bottom').setTicks([ticks_c1])
        self.t2_heatmap_win.getAxis('bottom').setTicks([ticks_c1])
        self.t2_heatmap_win.getAxis('left').setTicks([ticks_c2])
        self.t2_hslice_win.getAxis('bottom').setTicks([ticks_c1])
        self.t2_vslice_win.getAxis('bottom').setTicks([ticks_c2])
        self.t3_heatmap_win.getAxis('bottom').setTicks([ticks_c1])
        self.t3_heatmap_win.getAxis('left').setTicks([ticks_c2])
        self.t3_hslice_win.getAxis('bottom').setTicks([ticks_c1])
        self.t3_vslice_win.getAxis('bottom').setTicks([ticks_c2])

    def scale_plots(self):
        """Scale all plots: zoom completely out on coordinates, auto-range 1D plots, and set colorbar levels to full data range."""
        # 1. Reset all viewbox ranges to coordinate limits (c1, c2, energy) by calling reset_plot_scaling
        self.reset_plot_scaling()

        # 2. Scale interactive colorbars to full data range (min to max of the active slice/current data)
        # Ensure gradients are ordered normally (yellow green blue)
        self.vslice_colorbar.gradient.loadPreset('viridis')
        self.hslice_colorbar.gradient.loadPreset('viridis')
        self.t2_colorbar.gradient.loadPreset('viridis')
        self.t3_colorbar.gradient.loadPreset('viridis')

        if self.vslice_img.image is not None and self.vslice_img.image.size > 0:
            vslice_data = self.vslice_img.image
            min_v, max_v = np.min(vslice_data), np.max(vslice_data)
            if min_v == max_v:
                max_v += 1.0
            self.vslice_colorbar.setLevels(min_v, max_v)

        if self.hslice_img.image is not None and self.hslice_img.image.size > 0:
            hslice_data = self.hslice_img.image
            min_h, max_h = np.min(hslice_data), np.max(hslice_data)
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
            if y_data is not None and len(y_data) > 0:
                y_min = np.min(y_data)
                y_max = np.max(y_data)
                y_range = y_max - y_min
                if y_range <= 0:
                    y_range = 1.0
                plot_win.getViewBox().setRange(yRange=(y_min - 0.05 * y_range, y_max + 0.05 * y_range), padding=0)

        manually_scale_y(self.t2_hslice_win, self.t2_hslice_curve)
        manually_scale_y(self.t2_vslice_win, self.t2_vslice_curve)
        manually_scale_y(self.t3_hslice_win, self.t3_hslice_curve)
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
        
        if option == "Signal - BG":
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
        
        # Manually scale the Y-axis to fit only the spectrum curve if autoRange is still enabled
        vb_spec = self.spectrum_win.getViewBox()
        if vb_spec.state['autoRange'][1]:
            y_min = np.min(spectrum)
            y_max = np.max(spectrum)
            y_range = y_max - y_min
            if y_range <= 0:
                y_range = 1.0
            vb_spec.setRange(yRange=(y_min - 0.1 * y_range, y_max + 0.1 * y_range), padding=0)

        # 2. Update Vertical Slice (all C2 coordinates at fixed C1)
        vslice_data = processed_spectra[:, idx_c1, :]
        self.vslice_img.setImage(vslice_data)
        if vslice_data.size > 0:
            min_v, max_v = np.min(vslice_data), np.max(vslice_data)
            if min_v == max_v:
                max_v += 1
            self.vslice_colorbar.setLevels(min_v, max_v)

        # 3. Update Horizontal Slice (all C1 coordinates at fixed C2)
        hslice_data = processed_spectra[idx_c2, :, :]
        self.hslice_img.setImage(hslice_data)
        if hslice_data.size > 0:
            min_h, max_h = np.min(hslice_data), np.max(hslice_data)
            if min_h == max_h:
                max_h += 1
            self.hslice_colorbar.setLevels(min_h, max_h)

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
        self.t2_hslice_curve.setData(self.c1_coords, curr_vs_c1)

        # Vertical slice: Current vs C2 (at fixed C1)
        curr_vs_c2 = self.data_current_ch1[:, idx_c1]
        self.t2_vslice_curve.setData(self.c2_coords, curr_vs_c2)

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
        self.t3_hslice_curve.setData(self.c1_coords, curr_vs_c1)

        # Vertical slice: Current vs C2 (at fixed C1)
        curr_vs_c2 = self.data_current_ch2[:, idx_c1]
        self.t3_vslice_curve.setData(self.c2_coords, curr_vs_c2)

    # --- HDF5 Data Storage Saving ---
    def save_sweep_data(self, file_path=None):
        """Save the measured dataset (spectra and currents) to a structured HDF5 file."""
        if isinstance(file_path, bool):
            file_path = None
        if not np.any(self.data_measured_mask):
            if file_path is None:
                QtWidgets.QMessageBox.warning(self, "No Data", "There is no sweep data to save.")
            return

        try:
            data_dir = os.path.join(parent_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_filename = f"gate_sweep_{timestamp}.h5"

            if file_path is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save Measurement Data",
                    os.path.join(data_dir, default_filename),
                    "HDF5 files (*.h5);;All files (*.*)"
                )
                if not filename:
                    return
            else:
                filename = file_path

            with h5py.File(filename, "w") as f:
                # Save coordinates and grid geometry
                f.create_dataset("c1_coords", data=self.c1_coords)
                f.create_dataset("c2_coords", data=self.c2_coords)
                f.create_dataset("v1_grid", data=self.v1_grid)
                f.create_dataset("v2_grid", data=self.v2_grid)
                f.create_dataset("energy_axis", data=self.energy_axis)

                # Save measured datasets
                f.create_dataset("spectra", data=self.data_spectra)
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
        if getattr(self, "_autosaved_this_run", False):
            return
        if not np.any(self.data_measured_mask):
            return
        autosave_dir = os.path.join(parent_dir, "data", "autosave")
        os.makedirs(autosave_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_interrupted" if was_interrupted else ""
        filename = f"autosave_gate_sweep_{timestamp}{suffix}.h5"
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
        self.save_storage()
        event.accept()


if __name__ == '__main__':
    print("Initializing physical station for sweep GUI...")
    station = initialize_station(simulated=False, simulated_devices=simulated_devices)
    
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(station)
    win.show()
    sys.exit(app.exec())
