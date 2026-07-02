# -*- coding: utf-8 -*-
"""
PyQt6 GUI for the B/V Sweep (Magnetic Field and Voltage Sweep).
"""
import os
import sys
import json
import time
import math
import numpy as np
import h5py
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
try:
    from .utils import EnhancedViewBox, calculate_expected_duration
except ImportError:
    # If run directly as a script, ensure measurements directory is in path
    import sys
    import os
    _measurements_dir = os.path.dirname(os.path.abspath(__file__))
    if _measurements_dir not in sys.path:
        sys.path.insert(0, _measurements_dir)
    from utils import EnhancedViewBox, calculate_expected_duration

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from station_setup import initialize_station

# Setup pyqtgraph themes (Light Theme)
pg.setConfigOption('background', '#ffffff')
pg.setConfigOption('foreground', '#1e293b')

class ScientificPointSpinBox(QtWidgets.QSpinBox):
    """A spin box that accepts integers in scientific notation like 1e3."""
    def valueFromText(self, text):
        try:
            if text.strip() in ("", "-", "+"):
                return int(max(self.minimum(), min(self.maximum(), 0)))
            value = float(text)
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError
            return int(value)
        except Exception:
            return int(max(self.minimum(), min(self.maximum(), 0)))

    def validate(self, text, pos):
        if text.strip() in ("", "-", "+", "-.", "+.", "."):
            return QtGui.QValidator.State.Intermediate, text, pos
        try:
            value = float(text)
            if not math.isfinite(value) or not value.is_integer():
                return QtGui.QValidator.State.Invalid, text, pos
            if self.minimum() <= int(value) <= self.maximum():
                return QtGui.QValidator.State.Acceptable, text, pos
            return QtGui.QValidator.State.Intermediate, text, pos
        except ValueError:
            return QtGui.QValidator.State.Invalid, text, pos


class CollapsibleSection(QtWidgets.QWidget):
    """
    A premium collapsible container widget. Clicking the header toggles visibility of the contents.
    """
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.header = QtWidgets.QPushButton(f"▼ {title}")
        self.header.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                font-size: 13px;
                color: #000000;
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 0px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #eeeeee;
                border-color: #999999;
            }
        """)
        layout.addWidget(self.header)

        self.content = QtWidgets.QFrame()
        self.content.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.content.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 0px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.content)

        self.header.clicked.connect(self.toggle)
        self.is_expanded = True

    def toggle(self):
        self.is_expanded = not self.is_expanded
        self.content.setVisible(self.is_expanded)
        title_text = self.header.text()[2:]
        if self.is_expanded:
            self.header.setText(f"▼ {title_text}")
        else:
            self.header.setText(f"▶ {title_text}")


class BVSweepWorker(QtCore.QThread):
    """
    Background worker thread that runs the B/V sweep measurement.
    """
    point_measured = QtCore.pyqtSignal(int, int, float, float, float, float, np.ndarray, float, float, float, float, float)
    progress_updated = QtCore.pyqtSignal(int, int, float) # current, total, expected_remaining_seconds
    status_updated = QtCore.pyqtSignal(str)
    voltage_updated = QtCore.pyqtSignal(float, float)
    field_updated = QtCore.pyqtSignal(float)
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
        finally:
            self.finished.emit()

    def _do_run(self):
        try:
            smu = self.station.smu
            camera = self.station.camera
            magnet = self.station.magnet
            
            # Rotators for Faraday compensation
            rotator1 = getattr(self.station, "rotator1", None)
            rotator2 = getattr(self.station, "rotator2", None)

            # Coefficients
            a = self.params['a']
            b = self.params['b']
            c = self.params['c']
            d = self.params['d']

            # Voltage sweep parameters
            v_start = self.params['v_start']
            v_end = self.params['v_end']
            v_pts = self.params['v_pts']
            v_fixed = self.params['v_fixed']

            # Magnetic sweep parameters
            b_start = self.params['b_start']
            b_end = self.params['b_end']
            b_pts = self.params['b_pts']

            # Miscellaneous sweep settings
            variable_order = self.params['variable_order']
            serpentine = self.params['serpentine']
            settling_time = self.params['settling_time']
            exposure_time = self.params['exposure_time']
            num_pixels = self.params.get('num_pixels', 1340)
            
            faraday_comp = self.params['faraday_comp']
            faraday_rot = self.params['faraday_rot']
            end_ramp_zero = self.params['end_ramp_zero']
            end_ramp_ground = self.params['end_ramp_ground']
            #print(f"BVSweepWorker: Starting sweep thread. end_ramp_zero={end_ramp_zero}, end_ramp_ground={end_ramp_ground}")


            # Read multi-exposure settings from storage
            num_spectra = 1
            accum_method = "Sum"
            do_average = False
            comp1 = 0.01
            comp2 = 0.01
            mag_rate = 0.5
            storage_path = os.path.join(parent_dir, "value_storage")
            if os.path.exists(storage_path):
                try:
                    with open(storage_path, "r") as f:
                        vs = json.load(f)
                    single_data = vs.get("single_spectra", {})
                    num_spectra = int(single_data.get("num_spectra", vs.get("num_spectra", 1)))
                    accum_method = str(single_data.get("accum_method", vs.get("accum_method", "Sum")))
                    do_average = str(single_data.get("average", vs.get("average", "No"))) == "Yes"
                    
                    dg_data = vs.get("dual_gate", {})
                    comp1 = float(dg_data.get("v1_compliance", vs.get("v1_compliance", 0.01)))
                    comp2 = float(dg_data.get("v2_compliance", vs.get("v2_compliance", 0.01)))
                    
                    pos_data = vs.get("position", {})
                    mag_rate = float(pos_data.get("magnet_ramp_rate", 0.0083))
                except Exception as e:
                    print(f"BVSweepWorker: error reading value_storage ({e})")

            # Configure magnet sweep rate
            if magnet is not None:
                try:
                    magnet.field_sweep_rate(mag_rate * 60.0)
                except Exception as e:
                    print(f"BVSweepWorker: Error setting magnet sweep rate: {e}")

            # Explicitly configure SMU mode, compliance, and turn outputs ON
            smu.ch1.source_mode("VOLT")
            smu.ch2.source_mode("VOLT")
            smu.ch1.curr_limit(comp1)
            smu.ch2.curr_limit(comp2)
            smu.ch1.output(True)
            smu.ch2.output(True)

            camera.exposure_time(exposure_time)
            try:
                camera.num_frames(num_spectra)
            except Exception as e:
                print(f"BVSweepWorker: Error setting remote num_frames: {e}")

            # Generate sweep axes
            v_arr = np.linspace(v_start, v_end, v_pts) if v_pts > 1 else np.array([v_start])
            b_arr = np.linspace(b_start, b_end, b_pts) if b_pts > 1 else np.array([b_start])

            total_points = len(v_arr) * len(b_arr)
            point_count = 0

            # Store baseline rotator positions before sweep if Faraday compensation is enabled
            base_rot1 = 0.0
            base_rot2 = 0.0
            if faraday_comp:
                try:
                    if rotator1:
                        base_rot1 = rotator1.angle()
                    if rotator2:
                        base_rot2 = rotator2.angle()
                except Exception as e:
                    print(f"BVSweepWorker: Warning: failed to read baseline rotator position ({e})")


            # Sweep configuration
            if variable_order == "Hold B, scan V":
                outer_arr = b_arr
                inner_arr = v_arr
                is_b_outer = True
            else:
                outer_arr = v_arr
                inner_arr = b_arr
                is_b_outer = False

            # Magnet sweep rate (T/s)
            sweep_rate = magnet.field_sweep_rate() / 60.0 if not magnet.simulated else magnet._sim_sweep_rate / 60.0
            if sweep_rate <= 0:
                sweep_rate = 1.05 / 60.0

            # --- Helper functions -------------------------------------------------
            def _ramp_magnet(target, allow_abort=True):
                """Ramp the magnet to `target` Tesla, checking for pause/stop.
                Returns when the ramp finishes or the sweep is aborted.
                """
                # Check switch heater state and match leads if fitted and heater is off
                if getattr(magnet, "SWITCH_HEATER_FITTED", False):
                    try:
                        heater = magnet.switch_heater()
                        if heater.startswith('off'):
                            output_f = magnet.output_field()
                            persistent_f = magnet.persistent_magnet_field()
                            # If leads do not match coil field, match them first
                            if not np.isclose(output_f, persistent_f, atol=0.01):
                                self.status_updated.emit(f"Status: Matching leads to persistent field ({persistent_f:.4f} T)...")
                                magnet.target_field_param(persistent_f)
                                magnet.goto_set()
                                while magnet.is_sweeping():
                                    if allow_abort:
                                        self.mutex.lock()
                                        if not self._is_running:
                                            self.mutex.unlock()
                                            return False
                                        self.mutex.unlock()
                                    self.msleep(100)
                                    self.field_updated.emit(magnet.output_field())
                                magnet.hold()
                            
                            self.status_updated.emit("Status: Turning switch heater ON...")
                            magnet.switch_heater('on')
                            time.sleep(1.0) # wait minimum heater transition time
                    except Exception as e:
                        print(f"BVSweepWorker: Error managing switch heater: {e}")

                self.status_updated.emit(f"Status: Ramping magnet to {target:.4f} T...")
                magnet.target_field_param(target)
                magnet.goto_set()
                while magnet.is_sweeping():
                    # Allow pause/stop during magnet ramp
                    if allow_abort:
                        self.mutex.lock()
                        if not self._is_running:
                            magnet.hold()
                            self.mutex.unlock()
                            return False
                        if self._is_paused:
                            self.status_updated.emit("Status: Paused during magnet ramp")
                            magnet.hold()
                            self.pause_cond.wait(self.mutex)
                        self.mutex.unlock()
                    self.msleep(100)
                    self.field_updated.emit(magnet.output_field())
                return True

            def _ramp_smu(chan, target):
                """Ramp SMU channel using driver's software ramp."""
                chan.ramp_stop_requested = False
                try:
                    chan.set_r.volt(target)
                except Exception as e:
                    print(f"BVSweepWorker: Error in _ramp_smu: {e}")

            def _cleanup_on_abort():
                """Safely ramp down instruments on user abort/reset."""
                magnet.hold()
                if end_ramp_ground:
                    _ramp_smu(smu.ch1, 0.0)
                    _ramp_smu(smu.ch2, 0.0)
                    smu.ch1.output(False)
                    smu.ch2.output(False)
                try:
                    smu.return_to_local()
                except Exception:
                    pass
                if end_ramp_zero:
                    self.status_updated.emit("Status: Aborted. Ramping B to zero...")
                    _ramp_magnet(0.0, allow_abort=False)
                else:
                    try:
                        curr_f = magnet.output_field()
                        self.status_updated.emit(f"Status: Aborted. Leaving B-field at {curr_f:.4f} T.")
                    except Exception:
                        self.status_updated.emit("Status: Aborted. Leaving B-field intact.")
                if getattr(magnet, "SWITCH_HEATER_FITTED", False) and end_ramp_zero:
                    try:
                        magnet.switch_heater('off')
                        self.status_updated.emit("Status: Aborted. Ramping leads to zero...")
                        magnet.target_field_param(0.0)
                        magnet.goto_set()
                        while magnet.is_sweeping():
                            self.msleep(100)
                        magnet.hold()
                    except Exception:
                        pass

            # Initial magnet ramp to first B point (if needed)
            start_b = b_arr[0]
            curr_b = magnet.output_field()
            initial_ramp_time = abs(start_b - curr_b) / sweep_rate
            if initial_ramp_time > 0:
                if not _ramp_magnet(start_b, allow_abort=True):
                    # Sweep was aborted during initial ramp
                    _cleanup_on_abort()
                    return

            # Perform 2D Sweep
            for idx_outer, val_outer in enumerate(outer_arr):
                # Serpentine sweep direction
                if serpentine and (idx_outer % 2 == 1):
                    inner_indices = list(range(len(inner_arr) - 1, -1, -1))
                else:
                    inner_indices = list(range(len(inner_arr)))

                for idx_inner in inner_indices:
                    # Thread pause / stop safety check
                    self.mutex.lock()
                    while self._is_paused:
                        self.status_updated.emit("Status: Paused")
                        magnet.hold()
                        self.pause_cond.wait(self.mutex)
                    if not self._is_running:
                        self.mutex.unlock()
                        _cleanup_on_abort()
                        return
                    self.mutex.unlock()

                    val_inner = inner_arr[idx_inner]

                    # Assign variables based on mode
                    if is_b_outer:
                        b_field = val_outer
                        voltage = val_inner
                        idx_v = idx_inner
                        idx_b = idx_outer
                    else:
                        voltage = val_outer
                        b_field = val_inner
                        idx_v = idx_outer
                        idx_b = idx_inner

                    # 1. Ramp Magnetic Field (Non-blocking poll loop)
                    if magnet.output_field() != b_field:
                        if not _ramp_magnet(b_field, allow_abort=True):
                            _cleanup_on_abort()
                            return

                    # Faraday rotation compensation
                    if faraday_comp:
                        # Angle offset in degrees
                        offset = faraday_rot * b_field
                        if rotator1:
                            rotator1.angle(base_rot1 + offset)
                        if rotator2:
                            rotator2.angle(base_rot2 + offset)

                    # 2. Transform voltage coordinates to physical voltages
                    # Vc1 = C1 (voltage), Vc2 = C2 (fixed voltage)
                    v1 = a * voltage + b * v_fixed
                    v2 = c * voltage + d * v_fixed

                    # Ramp physical voltages
                    self.voltage_updated.emit(v1, v2)
                    # Gradually ramp voltages to avoid timeout on large steps
                    # Gradually ramp voltages using driver's software ramp
                    def _ramp(chan, target):
                        chan.ramp_stop_requested = False
                        max_retries = 3
                        for attempt in range(1, max_retries + 1):
                            try:
                                chan.set_r.volt(target)
                                break
                            except Exception as ex:
                                if attempt == max_retries:
                                    raise ex
                                try:
                                    chan.parent.visa_handle.clear()
                                except Exception:
                                    pass
                                time.sleep(0.1)
                    _ramp(smu.ch1, v1)
                    _ramp(smu.ch2, v2)

                    # 3. Wait settling time
                    if settling_time > 0:
                        sleep_end = time.time() + settling_time
                        while time.time() < sleep_end:
                            # Thread pause / stop safety check
                            self.mutex.lock()
                            while self._is_paused:
                                self.status_updated.emit("Status: Paused")
                                magnet.hold()
                                self.pause_cond.wait(self.mutex)
                            if not self._is_running:
                                self.mutex.unlock()
                                _cleanup_on_abort()
                                return
                            self.mutex.unlock()
                            time.sleep(0.05)

                    # 4. Acquire spectrum
                    self.status_updated.emit(f"Status: Acquiring spectrum ({num_spectra}x)...")
                    spectrum = None
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
                            print(f"BVSweepWorker error fetching spectrum: {e}")
                    
                    if spectrum is None:
                        spectrum = np.zeros(num_pixels)

                    # Read currents
                    try:
                        ch1_on = bool(smu.ch1.output())
                        curr_ch1 = smu.ch1.measure_curr() if ch1_on else 0.0
                        meas_v1 = smu.ch1.measure_volt() if ch1_on else 0.0
                    except Exception:
                        curr_ch1, meas_v1 = 0.0, 0.0

                    try:
                        ch2_on = bool(smu.ch2.output())
                        curr_ch2 = smu.ch2.measure_curr() if ch2_on else 0.0
                        meas_v2 = smu.ch2.measure_volt() if ch2_on else 0.0
                    except Exception:
                        curr_ch2, meas_v2 = 0.0, 0.0

                    # Read actual measured B field
                    try:
                        meas_b = magnet.output_field() if magnet is not None else b_field
                    except Exception:
                        meas_b = b_field

                    # Calculate expected remaining time
                    point_count += 1
                    remaining_pts = total_points - point_count
                    pts_time = remaining_pts * (exposure_time + settling_time)
                    
                    # Remaining B ramp time
                    remaining_b_time = 0.0
                    if is_b_outer:
                        remaining_b_steps = len(outer_arr) - (idx_outer + 1)
                        # Prevent division by zero if len(b_arr) is 1
                        b_step_size = abs(b_end - b_start) / (len(b_arr) - 1) if len(b_arr) > 1 else 0.0
                        remaining_b_time = remaining_b_steps * b_step_size / sweep_rate
                    else:
                        # Inner loop B: sweeps start to end for each remaining outer V step
                        remaining_v_steps = len(outer_arr) - (idx_outer + 1)
                        one_b_sweep_time = abs(b_end - b_start) / sweep_rate
                        
                        # Add remaining B sweep time for the current outer step
                        target_b_end = inner_arr[inner_indices[-1]]
                        current_b_remaining_time = abs(target_b_end - b_field) / sweep_rate
                        
                        remaining_b_time = (remaining_v_steps * one_b_sweep_time) + current_b_remaining_time
                    
                    remaining_seconds = pts_time + remaining_b_time

                    self.point_measured.emit(idx_v, idx_b, voltage, b_field, v1, v2, spectrum, curr_ch1, curr_ch2, meas_v1, meas_v2, meas_b)
                    self.progress_updated.emit(point_count, total_points, remaining_seconds)

            # Ramp back to zero or ground if configured at normal completion
            if end_ramp_ground:
                _ramp_smu(smu.ch1, 0.0)
                _ramp_smu(smu.ch2, 0.0)
                smu.ch1.output(False)
                smu.ch2.output(False)
            try:
                smu.return_to_local()
            except Exception:
                pass

            if end_ramp_zero:
                self.status_updated.emit("Status: Ramping magnet back to zero field...")
                if not _ramp_magnet(0.0):
                    return
            else:
                try:
                    curr_f = magnet.output_field()
                    self.status_updated.emit(f"Status: Sweep finished. Leaving B-field at {curr_f:.4f} T.")
                except Exception:
                    self.status_updated.emit("Status: Sweep finished. Leaving B-field intact.")

            # Disable switch heater and take leads to zero if switch heater is fitted
            if getattr(magnet, "SWITCH_HEATER_FITTED", False) and end_ramp_zero:
                try:
                    self.status_updated.emit("Status: Turning switch heater OFF...")
                    magnet.switch_heater('off')
                    self.status_updated.emit("Status: Ramping leads to zero...")
                    magnet.target_field_param(0.0)
                    magnet.goto_set()
                    while magnet.is_sweeping():
                        self.msleep(100)
                    magnet.hold()
                except Exception as e:
                    print(f"BVSweepWorker: Error in switch heater cleanup: {e}")

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
            
            self.btn_grid_A.setToolTip("Top-Left: 2D Map (Integrated Counts)")
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
        self.combo_source.addItems(["Signal", "Signal - BG", "(S-R)/(R-B)"])
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
            self.btn_grid_A.setToolTip("Top-Left: 2D Map (Integrated Counts)")
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
            self.btn_grid_B.setToolTip("Top-Right: Slice at fixed B")
            self.btn_grid_D.setToolTip("Bottom-Right: Slice at fixed V")
            
            self.btn_grid_AC.show()
            self.btn_grid_B.show()
            self.btn_grid_D.show()
            self.grid_layout.addWidget(self.btn_grid_AC, 0, 0, 2, 1)
            self.grid_layout.addWidget(self.btn_grid_B, 0, 1)
            self.grid_layout.addWidget(self.btn_grid_D, 1, 1)
        elif tab_idx == 2:
            # 3-plot layout (Current tabs): AC on left (spanning 2 rows), B on top-right, D on bottom-right
            self.btn_grid_AC.setToolTip("Left: Bottom Gate Current Heatmap")
            self.btn_grid_B.setToolTip("Top-Right: Slice at fixed B")
            self.btn_grid_D.setToolTip("Bottom-Right: Slice at fixed V")
            
            self.btn_grid_AC.show()
            self.btn_grid_B.show()
            self.btn_grid_D.show()
            self.grid_layout.addWidget(self.btn_grid_AC, 0, 0, 2, 1)
            self.grid_layout.addWidget(self.btn_grid_B, 0, 1)
            self.grid_layout.addWidget(self.btn_grid_D, 1, 1)

class BVSweepWindow(QtWidgets.QMainWindow):
    def __init__(self, station, embed=False):
        super().__init__()
        self.station = station
        self._embed = embed
        self.setWindowTitle("Magnetic Field vs Voltage Sweep Spectrometer")
        self.setMinimumSize(1280, 850)

        # Application states
        self.is_running = False
        self.is_paused = False
        self.reset_slice_levels = True
        self.sweep_worker = None
        self.was_aborted = False
        self.had_error = False

        # Time estimation states
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None

        # Data grids
        self.v_coords = np.array([])
        self.b_coords = np.array([])
        self.v1_grid = np.array([])
        self.v2_grid = np.array([])

        self.data_spectra = np.array([])
        self.data_current_ch1 = np.array([])
        self.data_current_ch2 = np.array([])
        self.data_measured_v1 = np.array([])
        self.data_measured_v2 = np.array([])
        self.data_measured_b = np.array([])
        self.data_measured_mask = np.array([])

        # Energy Calibration
        self.energy_axis = np.linspace(1.5, 3.0, 512)
        self.energy_sort_idx = np.arange(512)
        self.selected_energy = 2.25

        self.point_gatemap_v = 0.0
        self.point_gatemap_b = 0.0

        self.load_storage()
        self.setup_ui()
        self.apply_theme()
        self.view_control_popup = ViewControlPopup(self, has_grid=True)
        self.single_spectra_window = None

    def load_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        defaults = {
            "b_coord_name": "B",
            "b_start": -0.5,
            "b_end": 0.5,
            "b_pts": 21,
            
            
            "v_start": -0.05,
            "v_end": 0.05,
            "v_pts": 21,
            "v_fixed": 0.0,
            "variable_order": "Hold B, scan V",
            "serpentine": True,
            "settling_time": 0.05,
            "exposure_time": 0.1,
            "faraday_comp": False,
            "faraday_rot": 0.0,
            "end_ramp_zero": True,
            "end_ramp_ground": True,
            "coeff_a": 1.0,
            "coeff_b": 0.0,
            "coeff_c": 0.0,
            "coeff_d": 1.0
        }
        self.stored_settings = defaults.copy()
        self.stored_settings["c1_name"] = "V1"
        self.stored_settings["c2_name"] = "V2"
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    raw_data = json.load(f)
                shared_data = raw_data.get("shared", {})
                bv_data = raw_data.get("bv_sweep", {})
                self.stored_settings["settling_time"] = shared_data.get("settling_time", raw_data.get("settling_time", defaults["settling_time"]))
                self.stored_settings["exposure_time"] = shared_data.get("exposure_time", raw_data.get("exposure_time", defaults["exposure_time"]))
                self.stored_settings["c1_name"] = shared_data.get("c1_name", "V1")
                self.stored_settings["c2_name"] = shared_data.get("c2_name", "V2")
                for key in defaults:
                    if key not in ("settling_time", "exposure_time"):
                        if key in ("coeff_a", "coeff_b", "coeff_c", "coeff_d"):
                            legacy_key = f"bv_coeff_{key[-1]}"
                            self.stored_settings[key] = shared_data.get(key, bv_data.get(legacy_key, raw_data.get(legacy_key, defaults[key])))
                        else:
                            self.stored_settings[key] = bv_data.get(key, raw_data.get(key, defaults[key]))
            except Exception as e:
                print(f"Error loading value_storage: {e}")
        self.value_storage = self.stored_settings.copy()

    def save_storage(self, *args):
        storage_path = os.path.join(parent_dir, "value_storage")
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
                    current_data = json.load(f)
            
            if "shared" not in current_data:
                current_data["shared"] = {}
            if "bv_sweep" not in current_data:
                current_data["bv_sweep"] = {}
            
            current_data["shared"]["settling_time"] = self.settling_box.value()
            current_data["shared"]["exposure_time"] = self.exposure_box.value()
            current_data["shared"]["c1_name"] = self.stored_settings.get("c1_name", "V1")
            current_data["shared"]["c2_name"] = self.stored_settings.get("c2_name", "V2")
            current_data["shared"]["coeff_a"] = self.stored_settings.get("coeff_a", 1.0)
            current_data["shared"]["coeff_b"] = self.stored_settings.get("coeff_b", 0.0)
            current_data["shared"]["coeff_c"] = self.stored_settings.get("coeff_c", 0.0)
            current_data["shared"]["coeff_d"] = self.stored_settings.get("coeff_d", 1.0)
            camera = getattr(self.station, "camera", None)
            current_data["shared"]["spectrometer"] = getattr(camera, "spectrometer_type", "lightfieldG15")
            
            current_data["bv_sweep"]["b_coord_name"] = self.b_coord_name.text()
            current_data["bv_sweep"]["b_start"] = self.b_start.value()
            current_data["bv_sweep"]["b_end"] = self.b_end.value()
            current_data["bv_sweep"]["b_pts"] = self.b_pts.value()
            current_data["bv_sweep"]["v_start"] = self.v_start.value()
            current_data["bv_sweep"]["v_end"] = self.v_end.value()
            current_data["bv_sweep"]["v_pts"] = self.v_pts.value()
            current_data["bv_sweep"]["v_fixed"] = self.v_fixed.value()
            current_data["bv_sweep"]["variable_order"] = self.variable_order.currentText()
            current_data["bv_sweep"]["serpentine"] = self.serpentine.isChecked()
            current_data["bv_sweep"]["faraday_comp"] = self.faraday_comp.isChecked()
            current_data["bv_sweep"]["faraday_rot"] = self.faraday_rot.value()
            current_data["bv_sweep"]["end_ramp_zero"] = self.end_ramp_zero.isChecked()
            current_data["bv_sweep"]["end_ramp_ground"] = self.end_ramp_ground.isChecked()
            
            
            
            

            # Purge deprecated root-level keys
            deprecated_keys = [
                "exposure_time", "settling_time", "ramprate1", "ramprate2", "ramp_step", 
                "c1_name", "c2_name", "grating", "wavelength", "target_temp", "num_spectra", 
                "average", "accum_method", "plot_mode", "x_axis_scale", "plot_modes", 
                "b_coord_name", "b_start", "b_end", "b_pts", "v_start", "v_end", "v_pts", 
                "v_fixed", "variable_order", "serpentine", "faraday_comp", "faraday_rot", 
                "end_ramp_zero", "end_ramp_ground", "bv_coeff_a", "bv_coeff_b", "bv_coeff_c", 
                "bv_coeff_d", "pol_exc_vertical", "pol_exc_horizontal", "pol_exc_manual", 
                "pol_coll_vertical", "pol_coll_horizontal", "pol_coll_manual"
            ]
            for key in deprecated_keys:
                current_data.pop(key, None)
                if "bv_sweep" in current_data:
                    current_data["bv_sweep"].pop("bv_coeff_a", None)
                    current_data["bv_sweep"].pop("bv_coeff_b", None)
                    current_data["bv_sweep"].pop("bv_coeff_c", None)
                    current_data["bv_sweep"].pop("bv_coeff_d", None)

            self.value_storage = current_data["bv_sweep"]
            with open(storage_path, "w") as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Error saving to value_storage: {e}")

    def reload_gui_from_storage(self):
        self.load_storage()
        
        self.b_coord_name.blockSignals(True)
        self.b_coord_name.setText(self.stored_settings.get("b_coord_name", "B"))
        self.b_coord_name.blockSignals(False)

        self.b_start.blockSignals(True)
        self.b_start.setValue(self.stored_settings.get("b_start", -0.5))
        self.b_start.blockSignals(False)

        self.b_end.blockSignals(True)
        self.b_end.setValue(self.stored_settings.get("b_end", 0.5))
        self.b_end.blockSignals(False)

        self.b_pts.blockSignals(True)
        self.b_pts.setValue(self.stored_settings.get("b_pts", 21))
        self.b_pts.blockSignals(False)

        self.v_start.blockSignals(True)
        self.v_start.setValue(self.stored_settings.get("v_start", -0.05))
        self.v_start.blockSignals(False)

        self.v_end.blockSignals(True)
        self.v_end.setValue(self.stored_settings.get("v_end", 0.05))
        self.v_end.blockSignals(False)

        self.v_pts.blockSignals(True)
        self.v_pts.setValue(self.stored_settings.get("v_pts", 21))
        self.v_pts.blockSignals(False)

        self.v_fixed.blockSignals(True)
        self.v_fixed.setValue(self.stored_settings.get("v_fixed", 0.0))
        self.v_fixed.blockSignals(False)

        self.variable_order.blockSignals(True)
        self.variable_order.setCurrentText(self.stored_settings.get("variable_order", "Hold B, scan V"))
        self.variable_order.blockSignals(False)

        self.serpentine.blockSignals(True)
        self.serpentine.setChecked(self.stored_settings.get("serpentine", True))
        self.serpentine.blockSignals(False)

        self.settling_box.blockSignals(True)
        self.settling_box.setValue(self.stored_settings.get("settling_time", 0.05))
        self.settling_box.blockSignals(False)

        self.exposure_box.blockSignals(True)
        self.exposure_box.setValue(self.stored_settings.get("exposure_time", 0.1))
        self.exposure_box.blockSignals(False)

        self.faraday_comp.blockSignals(True)
        self.faraday_comp.setChecked(self.stored_settings.get("faraday_comp", False))
        self.faraday_comp.blockSignals(False)

        self.faraday_rot.blockSignals(True)
        self.faraday_rot.setValue(self.stored_settings.get("faraday_rot", 0.0))
        self.faraday_rot.blockSignals(False)

        self.end_ramp_zero.blockSignals(True)
        self.end_ramp_zero.setChecked(self.stored_settings.get("end_ramp_zero", True))
        self.end_ramp_zero.blockSignals(False)

        self.end_ramp_ground.blockSignals(True)
        self.end_ramp_ground.setChecked(self.stored_settings.get("end_ramp_ground", True))
        self.end_ramp_ground.blockSignals(False)

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

        self.update_step_calculations()
        self.update_variable_names()
        self.update_coordinates_from_ui()

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left controls panel
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(480)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        scroll_area.setWidget(left_widget)
        main_layout.addWidget(scroll_area)

        # Right plots tabs
        self.right_panel = QtWidgets.QTabWidget()
        main_layout.addWidget(self.right_panel, 1)

        # Build collapsible menus
        self.setup_magnetic_field_menu(left_layout)
        self.setup_voltage_sweep_menu(left_layout)
        self.setup_miscellaneous_menu(left_layout)
        self.setup_bottom_status_panel(left_layout)
        left_layout.addStretch(1)

        # Build plot tabs
        self.tab1 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab1, "Spectra Slices")
        self.setup_tab1()

        self.tab2 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab2, "Ch1 Current (Top Gate)")
        self.setup_tab2()

        self.tab3 = QtWidgets.QWidget()
        self.right_panel.addTab(self.tab3, "Ch2 Current (Bottom Gate)")
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

        self.update_variable_names()
        self.update_coordinates_from_ui()

    def setup_magnetic_field_menu(self, layout):
        sec = CollapsibleSection("1. Magnetic field sweep")
        grid = QtWidgets.QGridLayout(sec.content)
        grid.setSpacing(6)

        grid.addWidget(QtWidgets.QLabel("Coordinate Name:"), 0, 0)
        self.b_coord_name = QtWidgets.QLineEdit(self.stored_settings.get("b_coord_name", "B"))
        self.b_coord_name.editingFinished.connect(self.save_storage)
        self.b_coord_name.textChanged.connect(self.update_variable_names)
        grid.addWidget(self.b_coord_name, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Start (T):"), 1, 0)
        self.b_start = QtWidgets.QDoubleSpinBox()
        self.b_start.setRange(-15.0, 15.0)
        self.b_start.setDecimals(4)
        self.b_start.setValue(self.stored_settings.get("b_start", -0.5))
        self.b_start.editingFinished.connect(self.on_sweep_range_changed)
        grid.addWidget(self.b_start, 1, 1)

        grid.addWidget(QtWidgets.QLabel("End (T):"), 2, 0)
        self.b_end = QtWidgets.QDoubleSpinBox()
        self.b_end.setRange(-15.0, 15.0)
        self.b_end.setDecimals(4)
        self.b_end.setValue(self.stored_settings.get("b_end", 0.5))
        self.b_end.editingFinished.connect(self.on_sweep_range_changed)
        grid.addWidget(self.b_end, 2, 1)

        grid.addWidget(QtWidgets.QLabel("N Steps:"), 3, 0)
        self.b_pts = ScientificPointSpinBox()
        self.b_pts.setRange(1, 1000)
        self.b_pts.setValue(self.stored_settings.get("b_pts", 21))
        self.b_pts.editingFinished.connect(self.on_sweep_range_changed)
        grid.addWidget(self.b_pts, 3, 1)

        grid.addWidget(QtWidgets.QLabel("Stepsize (T):"), 4, 0)
        self.b_step_lbl = QtWidgets.QLabel("0.0")
        self.b_step_lbl.setStyleSheet("font-weight: bold; color: #475569;")
        grid.addWidget(self.b_step_lbl, 4, 1)

        
        
        
        
        
        

        
        
        
        
        
        

        layout.addWidget(sec)

    def setup_voltage_sweep_menu(self, layout):
        sec = CollapsibleSection("2. Voltage sweep")
        grid = QtWidgets.QGridLayout(sec.content)
        grid.setSpacing(6)

        grid.addWidget(QtWidgets.QLabel("Start (V):"), 0, 0)
        self.v_start = QtWidgets.QDoubleSpinBox()
        self.v_start.setRange(-210.0, 210.0)
        self.v_start.setDecimals(4)
        self.v_start.setValue(self.stored_settings.get("v_start", -0.05))
        self.v_start.editingFinished.connect(self.on_sweep_range_changed)
        grid.addWidget(self.v_start, 0, 1)

        grid.addWidget(QtWidgets.QLabel("End (V):"), 1, 0)
        self.v_end = QtWidgets.QDoubleSpinBox()
        self.v_end.setRange(-210.0, 210.0)
        self.v_end.setDecimals(4)
        self.v_end.setValue(self.stored_settings.get("v_end", 0.05))
        self.v_end.editingFinished.connect(self.on_sweep_range_changed)
        grid.addWidget(self.v_end, 1, 1)

        grid.addWidget(QtWidgets.QLabel("N Steps:"), 2, 0)
        self.v_pts = ScientificPointSpinBox()
        self.v_pts.setRange(1, 1000)
        self.v_pts.setValue(self.stored_settings.get("v_pts", 21))
        self.v_pts.editingFinished.connect(self.on_sweep_range_changed)
        grid.addWidget(self.v_pts, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Stepsize (V):"), 3, 0)
        self.v_step_lbl = QtWidgets.QLabel("0.0")
        self.v_step_lbl.setStyleSheet("font-weight: bold; color: #475569;")
        grid.addWidget(self.v_step_lbl, 3, 1)

        self.lbl_v_fixed = QtWidgets.QLabel("V2 Fixed (V):")
        grid.addWidget(self.lbl_v_fixed, 4, 0)
        self.v_fixed = QtWidgets.QDoubleSpinBox()
        self.v_fixed.setRange(-210.0, 210.0)
        self.v_fixed.setDecimals(4)
        self.v_fixed.setValue(self.stored_settings.get("v_fixed", 0.0))
        self.v_fixed.editingFinished.connect(self.save_storage)
        grid.addWidget(self.v_fixed, 4, 1)

        # Reset buttons row
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_ch1_only = QtWidgets.QPushButton("Channel 1 only")
        self.btn_ch1_only.clicked.connect(self.reset_to_ch1_only)
        self.btn_ch2_only = QtWidgets.QPushButton("Channel 2 only")
        self.btn_ch2_only.clicked.connect(self.reset_to_ch2_only)
        btn_layout.addWidget(self.btn_ch1_only)
        btn_layout.addWidget(self.btn_ch2_only)
        grid.addLayout(btn_layout, 5, 0, 1, 2)

        # Basis exchange parameters
        f1_layout = QtWidgets.QHBoxLayout()
        self.lbl_vc1_eq_start = QtWidgets.QLabel("Voltage 1 = ")
        f1_layout.addWidget(self.lbl_vc1_eq_start)
        self.coeff_a = QtWidgets.QDoubleSpinBox()
        self.coeff_a.setRange(-100.0, 100.0)
        self.coeff_a.setDecimals(4)
        self.coeff_a.setValue(self.stored_settings.get("coeff_a", 1.0))
        self.coeff_a.editingFinished.connect(self.on_sweep_range_changed)
        f1_layout.addWidget(self.coeff_a)
        self.lbl_vc1_eq_mid = QtWidgets.QLabel(" * V1_swept + ")
        f1_layout.addWidget(self.lbl_vc1_eq_mid)
        self.coeff_b = QtWidgets.QDoubleSpinBox()
        self.coeff_b.setRange(-100.0, 100.0)
        self.coeff_b.setDecimals(4)
        self.coeff_b.setValue(self.stored_settings.get("coeff_b", 0.0))
        self.coeff_b.editingFinished.connect(self.on_sweep_range_changed)
        f1_layout.addWidget(self.coeff_b)
        self.lbl_vc1_eq_end = QtWidgets.QLabel(" * V2_fixed")
        f1_layout.addWidget(self.lbl_vc1_eq_end)
        grid.addLayout(f1_layout, 6, 0, 1, 2)

        f2_layout = QtWidgets.QHBoxLayout()
        self.lbl_vc2_eq_start = QtWidgets.QLabel("Voltage 2 = ")
        f2_layout.addWidget(self.lbl_vc2_eq_start)
        self.coeff_c = QtWidgets.QDoubleSpinBox()
        self.coeff_c.setRange(-100.0, 100.0)
        self.coeff_c.setDecimals(4)
        self.coeff_c.setValue(self.stored_settings.get("coeff_c", 0.0))
        self.coeff_c.editingFinished.connect(self.on_sweep_range_changed)
        f2_layout.addWidget(self.coeff_c)
        self.lbl_vc2_eq_mid = QtWidgets.QLabel(" * V1_swept + ")
        f2_layout.addWidget(self.lbl_vc2_eq_mid)
        self.coeff_d = QtWidgets.QDoubleSpinBox()
        self.coeff_d.setRange(-100.0, 100.0)
        self.coeff_d.setDecimals(4)
        self.coeff_d.setValue(self.stored_settings.get("coeff_d", 1.0))
        self.coeff_d.editingFinished.connect(self.on_sweep_range_changed)
        f2_layout.addWidget(self.coeff_d)
        self.lbl_vc2_eq_end = QtWidgets.QLabel(" * V2_fixed")
        f2_layout.addWidget(self.lbl_vc2_eq_end)
        grid.addLayout(f2_layout, 7, 0, 1, 2)

        layout.addWidget(sec)

    def setup_miscellaneous_menu(self, layout):
        sec = CollapsibleSection("3. Miscellaneous")
        grid = QtWidgets.QGridLayout(sec.content)
        grid.setSpacing(6)

        grid.addWidget(QtWidgets.QLabel("Variable Order:"), 0, 0)
        self.variable_order = QtWidgets.QComboBox()
        self.variable_order.addItems(["Hold B, scan V", "Hold V, scan B"])
        self.variable_order.setCurrentText(self.stored_settings.get("variable_order", "Hold B, scan V"))
        self.variable_order.currentTextChanged.connect(self.on_sweep_range_changed)
        grid.addWidget(self.variable_order, 0, 1)

        self.serpentine = QtWidgets.QCheckBox("Serpentine Sweep")
        self.serpentine.setChecked(self.stored_settings.get("serpentine", True))
        self.serpentine.stateChanged.connect(self.save_storage)
        grid.addWidget(self.serpentine, 1, 0, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Settling Time (s):"), 2, 0)
        self.settling_box = QtWidgets.QDoubleSpinBox()
        self.settling_box.setRange(0.0, 100.0)
        self.settling_box.setDecimals(3)
        self.settling_box.setValue(self.stored_settings.get("settling_time", 0.05))
        self.settling_box.editingFinished.connect(self.save_storage)
        grid.addWidget(self.settling_box, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Exposure Time (s):"), 3, 0)
        self.exposure_box = QtWidgets.QDoubleSpinBox()
        self.exposure_box.setRange(0.001, 100.0)
        self.exposure_box.setDecimals(3)
        self.exposure_box.setValue(self.stored_settings.get("exposure_time", 0.1))
        self.exposure_box.editingFinished.connect(self.save_storage)
        grid.addWidget(self.exposure_box, 3, 1)

        # Faraday controls
        self.faraday_comp = QtWidgets.QCheckBox("Faraday Compensation")
        self.faraday_comp.setChecked(self.stored_settings.get("faraday_comp", False))
        self.faraday_comp.stateChanged.connect(self.save_storage)
        grid.addWidget(self.faraday_comp, 4, 0, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Faraday Rotation (deg/T):"), 5, 0)
        self.faraday_rot = QtWidgets.QDoubleSpinBox()
        self.faraday_rot.setRange(-360.0, 360.0)
        self.faraday_rot.setDecimals(3)
        self.faraday_rot.setValue(self.stored_settings.get("faraday_rot", 0.0))
        self.faraday_rot.editingFinished.connect(self.save_storage)
        grid.addWidget(self.faraday_rot, 5, 1)

        self.end_ramp_zero = QtWidgets.QCheckBox("End ramp to zero field")
        self.end_ramp_zero.setChecked(self.stored_settings.get("end_ramp_zero", True))
        self.end_ramp_zero.stateChanged.connect(self.save_storage)
        grid.addWidget(self.end_ramp_zero, 6, 0, 1, 2)

        self.end_ramp_ground = QtWidgets.QCheckBox("End ramp to ground")
        self.end_ramp_ground.setChecked(self.stored_settings.get("end_ramp_ground", True))
        self.end_ramp_ground.stateChanged.connect(self.save_storage)
        grid.addWidget(self.end_ramp_ground, 7, 0, 1, 2)

        layout.addWidget(sec)

    def setup_bottom_status_panel(self, layout):
        panel = QtWidgets.QGroupBox("Measurement Controls")
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setSpacing(6)

        # Play/Pause control row
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_begin = QtWidgets.QPushButton("▶ Begin Sweep")
        self.btn_begin.setStyleSheet("background-color: #10b981; color: white; font-weight: bold;")
        self.btn_begin.clicked.connect(self.begin_sweep)
        ctrl_layout.addWidget(self.btn_begin)

        self.btn_pause = QtWidgets.QPushButton("⏸ Interrupt")
        self.btn_pause.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold;")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        ctrl_layout.addWidget(self.btn_pause)
        vbox.addLayout(ctrl_layout)

        # Reset / Save row
        btn_layout2 = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("↺ Reset")
        self.btn_reset.clicked.connect(self.reset_sweep)
        btn_layout2.addWidget(self.btn_reset)

        self.btn_save = QtWidgets.QPushButton("💾 Save Dataset")
        self.btn_save.setStyleSheet("background-color: #555555; color: white; font-weight: bold; border-radius: 0px;")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_sweep_data)
        btn_layout2.addWidget(self.btn_save)
        vbox.addLayout(btn_layout2)

        # Progress / expected end time
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        vbox.addWidget(self.progress_bar)

        self.lbl_expected_end = QtWidgets.QLabel("Expected End: --:--:--")
        self.lbl_expected_end.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; font-weight: bold; color: #000000;")
        vbox.addWidget(self.lbl_expected_end)

        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("color: #000000; font-style: italic; font-weight: bold;")
        vbox.addWidget(self.lbl_status)

        self.lbl_target_vc1 = QtWidgets.QLabel("V1 = --‑‑ V")
        self.lbl_target_vc1.setStyleSheet("color: #000000; font-weight: bold;")
        vbox.addWidget(self.lbl_target_vc1)

        self.lbl_target_vc2 = QtWidgets.QLabel("V2 = --‑‑ V")
        self.lbl_target_vc2.setStyleSheet("color: #000000; font-weight: bold;")
        vbox.addWidget(self.lbl_target_vc2)

        self.lbl_current_b = QtWidgets.QLabel("B = --‑‑ T")
        self.lbl_current_b.setStyleSheet("color: #000000; font-weight: bold;")
        vbox.addWidget(self.lbl_current_b)

        layout.addWidget(panel)

    def reset_to_ch1_only(self):
        self.coeff_a.setValue(1.0)
        self.coeff_b.setValue(0.0)
        self.coeff_c.setValue(0.0)
        self.coeff_d.setValue(0.0)
        self.on_sweep_range_changed()

    def reset_to_ch2_only(self):
        self.coeff_a.setValue(0.0)
        self.coeff_b.setValue(0.0)
        self.coeff_c.setValue(1.0)
        self.coeff_d.setValue(0.0)
        self.on_sweep_range_changed()

    def update_step_calculations(self):
        # Magnetic Field Sweep step
        b_s = self.b_start.value()
        b_e = self.b_end.value()
        b_n = self.b_pts.value()
        if b_n > 1:
            self.b_step_lbl.setText(f"{(b_e - b_s)/(b_n - 1):.5f} T")
        else:
            self.b_step_lbl.setText("N/A")

        # Voltage Sweep step
        v_s = self.v_start.value()
        v_e = self.v_end.value()
        v_n = self.v_pts.value()
        if v_n > 1:
            self.v_step_lbl.setText(f"{(v_e - v_s)/(v_n - 1):.5f} V")
        else:
            self.v_step_lbl.setText("N/A")

    def update_variable_names(self):
        b_name = self.b_coord_name.text().strip() or "B"
        c1_name = self.stored_settings.get("c1_name", "V1")
        c2_name = self.stored_settings.get("c2_name", "V2")

        # Dynamic formula labels
        if hasattr(self, 'lbl_v_fixed'):
            self.lbl_v_fixed.setText(f"{c2_name} Fixed (V):")
        if hasattr(self, 'lbl_vc1_eq_start'):
            self.lbl_vc1_eq_start.setText("Voltage 1 = ")
        if hasattr(self, 'lbl_vc1_eq_mid'):
            self.lbl_vc1_eq_mid.setText(f" * {c1_name}_swept + ")
        if hasattr(self, 'lbl_vc1_eq_end'):
            self.lbl_vc1_eq_end.setText(f" * {c2_name}_fixed")
        if hasattr(self, 'lbl_vc2_eq_start'):
            self.lbl_vc2_eq_start.setText("Voltage 2 = ")
        if hasattr(self, 'lbl_vc2_eq_mid'):
            self.lbl_vc2_eq_mid.setText(f" * {c1_name}_swept + ")
        if hasattr(self, 'lbl_vc2_eq_end'):
            self.lbl_vc2_eq_end.setText(f" * {c2_name}_fixed")

        # Format axes label strings
        c1_label = f"V1 ({c1_name})" if c1_name.lower() != "v1" else "V1"

        # Update axes in plots
        if hasattr(self, 'gatemap_win'):
            self.gatemap_win.setLabel('left', f"Magnetic Field ({b_name})", units='T')
            self.gatemap_win.setLabel('bottom', f"Voltage ({c1_label})", units='V')
        if hasattr(self, 'vslice_win'):
            self.vslice_win.setLabel('left', f"Magnetic Field ({b_name})", units='T')
        if hasattr(self, 'hslice_win'):
            self.hslice_win.setLabel('left', f"Voltage ({c1_label})", units='V')
        if hasattr(self, 't2_heatmap_win'):
            self.t2_heatmap_win.setLabel('left', f"Magnetic Field ({b_name})", units='T')
            self.t2_heatmap_win.setLabel('bottom', f"Voltage ({c1_label})", units='V')
        if hasattr(self, 't2_hslice_win'):
            self.t2_hslice_win.setLabel('bottom', f"Voltage ({c1_label})", units='V')
        if hasattr(self, 't2_vslice_win'):
            self.t2_vslice_win.setLabel('bottom', f"Magnetic Field ({b_name})", units='T')
        if hasattr(self, 't3_heatmap_win'):
            self.t3_heatmap_win.setLabel('left', f"Magnetic Field ({b_name})", units='T')
            self.t3_heatmap_win.setLabel('bottom', f"Voltage ({c1_label})", units='V')
        if hasattr(self, 't3_hslice_win'):
            self.t3_hslice_win.setLabel('bottom', f"Voltage ({c1_label})", units='V')
        if hasattr(self, 't3_vslice_win'):
            self.t3_vslice_win.setLabel('bottom', f"Magnetic Field ({b_name})", units='T')

    def update_coordinates_from_ui(self):
        v_s = self.v_start.value()
        v_e = self.v_end.value()
        v_n = self.v_pts.value()
        self.v_coords = np.linspace(v_s, v_e, v_n) if v_n > 1 else np.array([v_s])

        b_s = self.b_start.value()
        b_e = self.b_end.value()
        b_n = self.b_pts.value()
        self.b_coords = np.linspace(b_s, b_e, b_n) if b_n > 1 else np.array([b_s])

        a = self.coeff_a.value()
        b = self.coeff_b.value()
        c = self.coeff_c.value()
        d = self.coeff_d.value()
        v_fixed = self.v_fixed.value()

        self.v1_grid = np.zeros((b_n, v_n))
        self.v2_grid = np.zeros((b_n, v_n))
        for j, b_val in enumerate(self.b_coords):
            for i, v_val in enumerate(self.v_coords):
                self.v1_grid[j, i] = a * v_val + b * v_fixed
                self.v2_grid[j, i] = c * v_val + d * v_fixed

    def on_sweep_range_changed(self):
        self.update_step_calculations()
        if not self.is_running:
            self.update_coordinates_from_ui()
            self.reset_plot_scaling()
            self.save_storage()

    # Layout setups
    def setup_tab1(self):
        layout = QtWidgets.QGridLayout(self.tab1)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # 1. Gate Map Plot (Top Left)
        self.gatemap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="2D Map (Integrated Counts)")
        self.gatemap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.gatemap_win.getViewBox().enableAutoRange(enable=False)
        self.gatemap_img = pg.ImageItem()
        self.gatemap_img.setOpts(axisOrder='row-major')
        self.gatemap_win.addItem(self.gatemap_img)
        self.gatemap_win.setLabel('left', 'Magnetic Field', units='T')
        self.gatemap_win.setLabel('bottom', 'Voltage (V1)', units='V')

        self.gatemap_colorbar = self.setup_histogram_lut(self.gatemap_win, self.gatemap_img, interactive=False)

        # Crosshairs
        self.cross1_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.cross1_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.gatemap_win.addItem(self.cross1_v)
        self.gatemap_win.addItem(self.cross1_h)
        self.cross1_v.sigPositionChanged.connect(self.on_crosshair_moved)
        self.cross1_h.sigPositionChanged.connect(self.on_crosshair_moved)

        vb = self.gatemap_win.getViewBox()
        def gatemap_click(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                pt = vb.mapSceneToView(ev.scenePos())
                self.update_crosshair_values(pt.x(), pt.y(), self.selected_energy)
                self.refresh_slice_plots(pt.x(), pt.y())
                ev.accept()
            else:
                vb.originalMouseClickEvent(ev)
        vb.originalMouseClickEvent = vb.mouseClickEvent
        vb.mouseClickEvent = gatemap_click

        layout.addWidget(self.gatemap_win, 0, 0)

        # 2. Vertical Slice (Top Right)
        self.vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Vertical Slice spectra")
        self.vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.vslice_win.getViewBox().enableAutoRange(enable=False)
        self.vslice_img = pg.ImageItem()
        self.vslice_img.setOpts(axisOrder='row-major')
        self.vslice_win.addItem(self.vslice_img)
        self.vslice_win.setLabel('left', 'Magnetic Field', units='T')
        self.vslice_win.setLabel('bottom', 'Energy', units='eV')
        self.vslice_colorbar = self.setup_histogram_lut(self.vslice_win, self.vslice_img, interactive=True)

        self.vslice_cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.vslice_cross_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))

        vb_v = self.vslice_win.getViewBox()
        def vslice_click(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                pt = vb_v.mapSceneToView(ev.scenePos())
                self.update_crosshair_values(self.cross1_v.value(), pt.y(), self.selected_energy)
                self.refresh_slice_plots(self.cross1_v.value(), pt.y())
                ev.accept()
            else:
                vb_v.originalMouseClickEvent(ev)
        vb_v.originalMouseClickEvent = vb_v.mouseClickEvent
        vb_v.mouseClickEvent = vslice_click

        layout.addWidget(self.vslice_win, 0, 1)

        # 3. Horizontal Slice (Bottom Left)
        self.hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Horizontal Slice spectra")
        self.hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.hslice_win.getViewBox().enableAutoRange(enable=False)
        self.hslice_img = pg.ImageItem()
        self.hslice_img.setOpts(axisOrder='row-major')
        self.hslice_win.addItem(self.hslice_img)
        self.hslice_win.setLabel('left', 'Voltage (V1)', units='V')
        self.hslice_win.setLabel('bottom', 'Energy', units='eV')
        self.hslice_colorbar = self.setup_histogram_lut(self.hslice_win, self.hslice_img, interactive=True)

        self.hslice_cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.hslice_cross_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))

        vb_h = self.hslice_win.getViewBox()
        def hslice_click(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                pt = vb_h.mapSceneToView(ev.scenePos())
                self.update_crosshair_values(pt.y(), self.cross1_h.value(), pt.x())
                self.refresh_slice_plots(pt.y(), self.cross1_h.value())
                ev.accept()
            else:
                vb_h.originalMouseClickEvent(ev)
        vb_h.originalMouseClickEvent = vb_h.mouseClickEvent
        vb_h.mouseClickEvent = hslice_click

        layout.addWidget(self.hslice_win, 1, 0)

        # 4. CCD Spectrum at Crosshair (Bottom Right)
        self.spectrum_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="CCD Spectrum at Gating Crosshair")
        self.spectrum_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.spectrum_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.spectrum_win.getViewBox().enableAutoRange(axis='y', enable=False)
        self.spectrum_curve = self.spectrum_win.plot(pen=pg.mkPen('#7c3aed', width=2))
        self.spectrum_win.setLabel('left', 'Counts')
        self.spectrum_win.setLabel('bottom', 'Energy', units='eV')
        self.spectrum_win.showGrid(x=True, y=True, alpha=0.3)

        self.spectrum_v_region = pg.LinearRegionItem(
            orientation='vertical',
            values=(2.4, 2.6),
            brush=pg.mkBrush(16, 185, 129, 40),
            pen=pg.mkPen('#10b981', width=1.5)
        )
        self.spectrum_win.addItem(self.spectrum_v_region, ignoreBounds=True)
        self.spectrum_v1 = self.spectrum_v_region.lines[0]
        self.spectrum_v2 = self.spectrum_v_region.lines[1]
        self.spectrum_v_region.sigRegionChanged.connect(self.on_integration_region_changed)

        self.spectrum_h_region = pg.LinearRegionItem(
            orientation='horizontal',
            values=(1000.0, 5000.0),
            brush=pg.mkBrush(16, 185, 129, 40),
            pen=pg.mkPen('#10b981', width=1.5)
        )
        self.spectrum_win.addItem(self.spectrum_h_region, ignoreBounds=True)
        self.spectrum_h1 = self.spectrum_h_region.lines[0]
        self.spectrum_h2 = self.spectrum_h_region.lines[1]
        self.spectrum_h_region.sigRegionChanged.connect(self.on_integration_region_changed)

        layout.addWidget(self.spectrum_win, 1, 1)

    def setup_tab2(self):
        layout = QtWidgets.QHBoxLayout(self.tab2)
        layout.setContentsMargins(5, 5, 5, 5)

        self.t2_heatmap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Top Gate Current Heatmap")
        self.t2_heatmap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_heatmap_win.getViewBox().enableAutoRange(enable=False)
        self.t2_heatmap_img = pg.ImageItem()
        self.t2_heatmap_img.setOpts(axisOrder='row-major')
        self.t2_heatmap_win.addItem(self.t2_heatmap_img)
        self.t2_heatmap_win.setLabel('left', 'Magnetic Field', units='T')
        self.t2_heatmap_win.setLabel('bottom', 'Voltage (V1)', units='V')
        self.t2_colorbar = self.setup_histogram_lut(self.t2_heatmap_win, self.t2_heatmap_img, interactive=True)

        self.cross2_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#ea580c', width=2))
        self.cross2_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#ea580c', width=2))
        self.t2_heatmap_win.addItem(self.cross2_v)
        self.t2_heatmap_win.addItem(self.cross2_h)
        self.cross2_v.sigPositionChanged.connect(self.on_tab2_crosshair_moved)
        self.cross2_h.sigPositionChanged.connect(self.on_tab2_crosshair_moved)

        vb = self.t2_heatmap_win.getViewBox()
        def t2_click(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                pt = vb.mapSceneToView(ev.scenePos())
                self.cross2_v.blockSignals(True)
                self.cross2_h.blockSignals(True)
                self.cross2_v.setValue(pt.x())
                self.cross2_h.setValue(pt.y())
                self.cross2_v.blockSignals(False)
                self.cross2_h.blockSignals(False)
                self.on_tab2_crosshair_moved()
                ev.accept()
            else:
                vb.originalMouseClickEvent(ev)
        vb.originalMouseClickEvent = vb.mouseClickEvent
        vb.mouseClickEvent = t2_click

        layout.addWidget(self.t2_heatmap_win, 3)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(6)

        self.t2_hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed B")
        self.t2_hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_hslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t2_hslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t2_hslice_curve = self.t2_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t2_hslice_win.setLabel('left', 'Current', units='A')
        self.t2_hslice_win.setLabel('bottom', 'Voltage (V1)', units='V')
        self.t2_hslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t2_hslice_win)

        self.t2_vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed V")
        self.t2_vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t2_vslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t2_vslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t2_vslice_curve = self.t2_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t2_vslice_win.setLabel('left', 'Current', units='A')
        self.t2_vslice_win.setLabel('bottom', 'Magnetic Field', units='T')
        self.t2_vslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t2_vslice_win)

        layout.addLayout(right_layout, 2)

    def setup_tab3(self):
        layout = QtWidgets.QHBoxLayout(self.tab3)
        layout.setContentsMargins(5, 5, 5, 5)

        self.t3_heatmap_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Bottom Gate Current Heatmap")
        self.t3_heatmap_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_heatmap_win.getViewBox().enableAutoRange(enable=False)
        self.t3_heatmap_img = pg.ImageItem()
        self.t3_heatmap_img.setOpts(axisOrder='row-major')
        self.t3_heatmap_win.addItem(self.t3_heatmap_img)
        self.t3_heatmap_win.setLabel('left', 'Magnetic Field', units='T')
        self.t3_heatmap_win.setLabel('bottom', 'Voltage (V1)', units='V')
        self.t3_colorbar = self.setup_histogram_lut(self.t3_heatmap_win, self.t3_heatmap_img, interactive=True)

        self.cross3_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#16a34a', width=2))
        self.cross3_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#16a34a', width=2))
        self.t3_heatmap_win.addItem(self.cross3_v)
        self.t3_heatmap_win.addItem(self.cross3_h)
        self.cross3_v.sigPositionChanged.connect(self.on_tab3_crosshair_moved)
        self.cross3_h.sigPositionChanged.connect(self.on_tab3_crosshair_moved)

        vb = self.t3_heatmap_win.getViewBox()
        def t3_click(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                pt = vb.mapSceneToView(ev.scenePos())
                self.cross3_v.blockSignals(True)
                self.cross3_h.blockSignals(True)
                self.cross3_v.setValue(pt.x())
                self.cross3_h.setValue(pt.y())
                self.cross3_v.blockSignals(False)
                self.cross3_h.blockSignals(False)
                self.on_tab3_crosshair_moved()
                ev.accept()
            else:
                vb.originalMouseClickEvent(ev)
        vb.originalMouseClickEvent = vb.mouseClickEvent
        vb.mouseClickEvent = t3_click

        layout.addWidget(self.t3_heatmap_win, 3)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(6)

        self.t3_hslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed B")
        self.t3_hslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_hslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t3_hslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t3_hslice_curve = self.t3_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t3_hslice_win.setLabel('left', 'Current', units='A')
        self.t3_hslice_win.setLabel('bottom', 'Voltage (V1)', units='V')
        self.t3_hslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t3_hslice_win)

        self.t3_vslice_win = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Slice at fixed V")
        self.t3_vslice_win.getViewBox().setMouseEnabled(x=True, y=True)
        self.t3_vslice_win.getViewBox().enableAutoRange(axis='x', enable=False)
        self.t3_vslice_win.getViewBox().enableAutoRange(axis='y', enable=True)
        self.t3_vslice_curve = self.t3_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t3_vslice_win.setLabel('left', 'Current', units='A')
        self.t3_vslice_win.setLabel('bottom', 'Magnetic Field', units='T')
        self.t3_vslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t3_vslice_win)

        layout.addLayout(right_layout, 2)

    def setup_histogram_lut(self, plot_win, image_item, interactive=True):
        hist = pg.HistogramLUTItem(gradientPosition='left')
        hist.setImageItem(image_item)
        hist.gradient.loadPreset('viridis')
        
        hist.layout.removeItem(hist.gradient)
        hist.layout.removeItem(hist.vb)
        hist.layout.removeItem(hist.axis)
        hist.layout.addItem(hist.gradient, 0, 0)
        hist.layout.addItem(hist.vb, 0, 1)
        hist.layout.addItem(hist.axis, 0, 2)
        
        hist.layout.setContentsMargins(0, 0, 0, 0)
        hist.layout.setSpacing(2)
        hist.layout.setColumnFixedWidth(0, 12)
        hist.layout.setColumnFixedWidth(1, 45)
        
        hist.gradient.setMinimumWidth(0)
        hist.gradient.setMaximumWidth(12)
        hist.vb.setMinimumWidth(0)
        hist.vb.setMaximumWidth(45)
        hist.setMinimumWidth(80)
        
        if not interactive:
            hist.region.setMovable(False)
            for line in hist.region.lines:
                line.setMovable(False)
            hist.gradient.setEnabled(False)
            hist.vb.setMouseEnabled(x=False, y=True)
            
        pi = plot_win.getPlotItem()
        pi.layout.removeItem(pi.titleLabel)
        pi.layout.addItem(pi.titleLabel, 0, 0, 1, 6)
        pi.layout.addItem(hist, 2, 5)
        pi.layout.setColumnFixedWidth(4, 3)
        return hist

    def reset_plot_scaling(self):
        self.reset_slice_levels = True
        v_min, v_max = min(self.v_coords), max(self.v_coords)
        b_min, b_max = min(self.b_coords), max(self.b_coords)
        
        if len(self.v_coords) <= 1 or v_min == v_max:
            v_min -= 0.5
            v_max += 0.5
        if len(self.b_coords) <= 1 or b_min == b_max:
            b_min -= 0.5
            b_max += 0.5

        # 2D Heatmap bounds
        self.gatemap_win.setLimits(xMin=v_min, xMax=v_max, yMin=b_min, yMax=b_max)
        if self.gatemap_img.image is not None:
            self.gatemap_img.setRect(QtCore.QRectF(self.v_coords[0], self.b_coords[0], self.v_coords[-1] - self.v_coords[0], self.b_coords[-1] - self.b_coords[0]))
        
        vb = self.gatemap_win.getViewBox()
        vb.invertX(self.v_coords[0] > self.v_coords[-1])
        vb.invertY(self.b_coords[0] > self.b_coords[-1])
        self.gatemap_win.setRange(xRange=(v_min, v_max), yRange=(b_min, b_max), padding=0)

        # Slice bounds
        energy_min, energy_max = min(self.energy_axis), max(self.energy_axis)
        self.vslice_win.setLimits(xMin=energy_min, xMax=energy_max, yMin=b_min, yMax=b_max)
        if self.vslice_img.image is not None:
            self.vslice_img.setRect(QtCore.QRectF(self.energy_axis[0], self.b_coords[0], self.energy_axis[-1] - self.energy_axis[0], self.b_coords[-1] - self.b_coords[0]))
        self.vslice_win.setRange(xRange=(energy_min, energy_max), yRange=(b_min, b_max), padding=0)

        self.hslice_win.setLimits(xMin=energy_min, xMax=energy_max, yMin=v_min, yMax=v_max)
        if self.hslice_img.image is not None:
            self.hslice_img.setRect(QtCore.QRectF(self.energy_axis[0], self.v_coords[0], self.energy_axis[-1] - self.energy_axis[0], self.v_coords[-1] - self.v_coords[0]))
        self.hslice_win.setRange(xRange=(energy_min, energy_max), yRange=(v_min, v_max), padding=0)

        self.spectrum_win.setLimits(xMin=energy_min, xMax=energy_max)
        self.spectrum_win.setRange(xRange=(energy_min, energy_max), padding=0)

        # Current Heatmaps
        self.t2_heatmap_win.getViewBox().update_bounds(v_min, v_max, b_min, b_max)
        if self.t2_heatmap_img.image is not None:
            self.t2_heatmap_img.setRect(QtCore.QRectF(self.v_coords[0], self.b_coords[0], self.v_coords[-1] - self.v_coords[0], self.b_coords[-1] - self.b_coords[0]))
        self.t2_heatmap_win.getViewBox().setRange(xRange=(v_min, v_max), yRange=(b_min, b_max), padding=0)

        self.t2_hslice_win.getViewBox().update_bounds(v_min, v_max, None, None)
        self.t2_hslice_win.getViewBox().setRange(xRange=(v_min, v_max), padding=0.05)
        self.t2_vslice_win.getViewBox().update_bounds(b_min, b_max, None, None)
        self.t2_vslice_win.getViewBox().setRange(xRange=(b_min, b_max), padding=0.05)

        self.t3_heatmap_win.getViewBox().update_bounds(v_min, v_max, b_min, b_max)
        if self.t3_heatmap_img.image is not None:
            self.t3_heatmap_img.setRect(QtCore.QRectF(self.v_coords[0], self.b_coords[0], self.v_coords[-1] - self.v_coords[0], self.b_coords[-1] - self.b_coords[0]))
        self.t3_heatmap_win.getViewBox().setRange(xRange=(v_min, v_max), yRange=(b_min, b_max), padding=0)

        self.t3_hslice_win.getViewBox().update_bounds(v_min, v_max, None, None)
        self.t3_hslice_win.getViewBox().setRange(xRange=(v_min, v_max), padding=0.05)
        self.t3_vslice_win.getViewBox().update_bounds(b_min, b_max, None, None)
        self.t3_vslice_win.getViewBox().setRange(xRange=(b_min, b_max), padding=0.05)

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

        # Set crosshairs
        self.selected_energy = (self.energy_axis[0] + self.energy_axis[-1]) / 2.0
        p_v = (self.v_coords[0] + self.v_coords[-1]) / 2.0
        p_b = (self.b_coords[0] + self.b_coords[-1]) / 2.0
        self.update_crosshair_values(p_v, p_b, self.selected_energy)

        self.cross2_v.setValue(p_v)
        self.cross2_h.setValue(p_b)
        self.cross3_v.setValue(p_v)
        self.cross3_h.setValue(p_b)

    def scale_plots(self):
        self.reset_slice_levels = True
        """Scale all plots: zoom completely out on coordinates, auto-range 1D plots, and set colorbar levels to full data range."""
        # 1. Reset all viewbox ranges to coordinate limits (v, b, energy) by calling reset_plot_scaling
        self.reset_plot_scaling()

        # 2. Scale interactive colorbars to full data range (min to max of the active slice/current data)
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

    def update_crosshair_values(self, v_val, b_val, energy_val):
        self.selected_energy = energy_val
        self.point_gatemap_v = v_val
        self.point_gatemap_b = b_val

        self.cross1_v.blockSignals(True)
        self.cross1_h.blockSignals(True)
        self.vslice_cross_v.blockSignals(True)
        self.vslice_cross_h.blockSignals(True)
        self.hslice_cross_v.blockSignals(True)
        self.hslice_cross_h.blockSignals(True)

        self.cross1_v.setValue(v_val)
        self.cross1_h.setValue(b_val)
        self.vslice_cross_v.setValue(energy_val)
        self.vslice_cross_h.setValue(b_val)
        self.hslice_cross_v.setValue(energy_val)
        self.hslice_cross_h.setValue(v_val)

        self.cross1_v.blockSignals(False)
        self.cross1_h.blockSignals(False)
        self.vslice_cross_v.blockSignals(False)
        self.vslice_cross_h.blockSignals(False)
        self.hslice_cross_v.blockSignals(False)
        self.hslice_cross_h.blockSignals(False)

    def on_crosshair_moved(self):
        if not np.any(self.data_measured_mask): return
        self.update_crosshair_values(self.cross1_v.value(), self.cross1_h.value(), self.selected_energy)
        self.refresh_slice_plots(self.cross1_v.value(), self.cross1_h.value())

    def on_tab2_crosshair_moved(self):
        if not np.any(self.data_measured_mask): return
        idx_v = np.argmin(np.abs(self.v_coords - self.cross2_v.value()))
        idx_b = np.argmin(np.abs(self.b_coords - self.cross2_h.value()))
        self.t2_hslice_curve.setData(self.v_coords, self.data_current_ch1[idx_b, :])
        self.t2_vslice_curve.setData(self.b_coords, self.data_current_ch1[:, idx_v])

    def on_tab3_crosshair_moved(self):
        if not np.any(self.data_measured_mask): return
        idx_v = np.argmin(np.abs(self.v_coords - self.cross3_v.value()))
        idx_b = np.argmin(np.abs(self.b_coords - self.cross3_h.value()))
        self.t3_hslice_curve.setData(self.v_coords, self.data_current_ch2[idx_b, :])
        self.t3_vslice_curve.setData(self.b_coords, self.data_current_ch2[:, idx_v])

    def refresh_slice_plots(self, v_val, b_val):
        if not np.any(self.data_measured_mask): return
        idx_v = np.argmin(np.abs(self.v_coords - v_val))
        idx_b = np.argmin(np.abs(self.b_coords - b_val))

        processed_spectra = self.get_processed_spectra()

        spectrum = processed_spectra[idx_b, idx_v, :]
        self.spectrum_curve.setData(self.energy_axis, spectrum)

        vslice_data = processed_spectra[:, idx_v, :]
        auto_levels = getattr(self, "reset_slice_levels", True)
        self.vslice_img.setImage(vslice_data, autoLevels=auto_levels)
        if auto_levels and vslice_data.size > 0:
            min_v, max_v = np.min(vslice_data), np.max(vslice_data)
            if min_v == max_v:
                max_v += 1.0
            self.vslice_colorbar.setLevels(min_v, max_v)

        hslice_data = processed_spectra[idx_b, :, :]
        self.hslice_img.setImage(hslice_data, autoLevels=auto_levels)
        if auto_levels and hslice_data.size > 0:
            min_h, max_h = np.min(hslice_data), np.max(hslice_data)
            if min_h == max_h:
                max_h += 1.0
            self.hslice_colorbar.setLevels(min_h, max_h)

        self.reset_slice_levels = False

    def on_integration_region_changed(self):
        self.update_heatmaps_and_slices()

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

    def update_heatmaps_and_slices(self):
        if not np.any(self.data_measured_mask): return

        e_min, e_max = sorted(self.spectrum_v_region.getRegion())
        h_min, h_max = sorted(self.spectrum_h_region.getRegion())
        idx_range = np.where((self.energy_axis >= e_min) & (self.energy_axis <= e_max))[0]

        if len(idx_range) > 0:
            sub = self.get_processed_spectra()[:, :, idx_range]
            clamped = np.clip(sub, h_min, h_max) - h_min
            integrated = np.sum(clamped, axis=2)
        else:
            integrated = np.zeros((len(self.b_coords), len(self.v_coords)))

        self.gatemap_img.setImage(integrated)
        measured = integrated[self.data_measured_mask]
        self.gatemap_colorbar.setLevels(np.min(measured), np.max(measured))

        self.t2_heatmap_img.setImage(self.data_current_ch1)
        self.t2_colorbar.setLevels(np.min(self.data_current_ch1[self.data_measured_mask]), np.max(self.data_current_ch1[self.data_measured_mask]))

        self.t3_heatmap_img.setImage(self.data_current_ch2)
        self.t3_colorbar.setLevels(np.min(self.data_current_ch2[self.data_measured_mask]), np.max(self.data_current_ch2[self.data_measured_mask]))

        self.on_tab2_crosshair_moved()
        self.on_tab3_crosshair_moved()

    def apply_theme(self):
        """Premium light theme styling matching Single Spectra (rectangular, strict monochromatic white/grey/black, no blue tint)"""
        qss = """
        QMainWindow, QWidget {
            background-color: #ffffff;
            color: #000000;
        }
        QGroupBox {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 0px;
            margin-top: 2.5ex;
            color: #000000;
            font-weight: bold;
            font-size: 13px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            background-color: transparent;
        }
        QLabel {
            color: #000000;
            font-size: 12px;
            background-color: transparent;
        }
        QPushButton {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #cccccc;
            border-radius: 0px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #eeeeee;
            border-color: #999999;
        }
        QPushButton:pressed {
            background-color: #dddddd;
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 0px;
            color: #000000;
            padding: 5px;
            font-family: Consolas, monospace;
            font-size: 12px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QProgressBar {
            background-color: #eeeeee;
            border: 1px solid #cccccc;
            border-radius: 0px;
            text-align: center;
            color: #000000;
            font-weight: bold;
            height: 18px;
        }
        QProgressBar::chunk {
            background-color: #333333;
            border-radius: 0px;
        }
        QCheckBox {
            background-color: transparent;
        }
        """
        self.setStyleSheet(qss)

        # Apply unified styling (white background and slate axes) to all PlotWidgets
        import pyqtgraph as pg
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, pg.PlotWidget):
                attr.setBackground('w')
                attr.getAxis('left').setPen(pg.mkPen('#334155'))
                attr.getAxis('bottom').setPen(pg.mkPen('#334155'))
                attr.getAxis('left').setTextPen(pg.mkPen('#334155'))
                attr.getAxis('bottom').setTextPen(pg.mkPen('#334155'))

    def set_inputs_enabled(self, enabled: bool):
        self.b_coord_name.setEnabled(enabled)
        self.b_start.setEnabled(enabled)
        self.b_end.setEnabled(enabled)
        self.b_pts.setEnabled(enabled)
        self.v_start.setEnabled(enabled)
        self.v_end.setEnabled(enabled)
        self.v_pts.setEnabled(enabled)
        self.v_fixed.setEnabled(enabled)
        self.coeff_a.setEnabled(enabled)
        self.coeff_b.setEnabled(enabled)
        self.coeff_c.setEnabled(enabled)
        self.coeff_d.setEnabled(enabled)
        self.btn_ch1_only.setEnabled(enabled)
        self.btn_ch2_only.setEnabled(enabled)
        self.variable_order.setEnabled(enabled)
        self.serpentine.setEnabled(enabled)
        self.settling_box.setEnabled(enabled)
        self.exposure_box.setEnabled(enabled)
        self.faraday_comp.setEnabled(enabled)
        self.faraday_rot.setEnabled(enabled)
        self.end_ramp_zero.setEnabled(enabled)
        self.end_ramp_ground.setEnabled(enabled)

    # Sweep execution
    def begin_sweep(self):
        if self.is_paused:
            self.is_paused = False
            self.btn_pause.setText("⏸ Interrupt")
            self.btn_begin.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.lbl_status.setText("Status: Sweeping...")
            self.sweep_worker.resume()
            return

        if self.is_running: return

        self.reset_slice_levels = True

        self._autosaved_this_run = False
        self.was_aborted = False
        self.had_error = False
        self.station.live_status_paused = True
        self.progress_bar.setValue(0)

        # Clear the SMU VISA session to discard any unread response or dirty buffer from active status workers
        smu = getattr(self.station, "smu", None)
        if smu is not None and not smu.simulated:
            try:
                smu.visa_handle.clear()
                smu.write('*CLS')
            except Exception:
                pass
        self.update_coordinates_from_ui()
        v_len = len(self.v_coords)
        b_len = len(self.b_coords)

        # Determine number of pixels dynamically from camera spectrometer type
        spec_type = getattr(self.station.camera, "spectrometer_type", "lightfieldG15")
        if "G11" in spec_type or "nirvana" in spec_type.lower():
            num_pixels = 1024
        else:
            num_pixels = 1340

        # Retrieve center wavelength for calibration
        try:
            mono_wl = self.station.monochromator.wavelength()
        except Exception:
            mono_wl = 500.0

        # Try to get wavelengths from cache
        wavelengths = None
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

        # Allocate arrays
        self.data_spectra = np.zeros((b_len, v_len, num_pixels))
        self.data_current_ch1 = np.zeros((b_len, v_len))
        self.data_current_ch2 = np.zeros((b_len, v_len))
        self.data_measured_v1 = np.zeros((b_len, v_len))
        self.data_measured_v2 = np.zeros((b_len, v_len))
        self.data_measured_b = np.zeros((b_len, v_len))
        self.data_measured_mask = np.zeros((b_len, v_len), dtype=bool)

        # Initialize heatmap images with zero arrays of the correct size
        # so pyqtgraph can resolve their size when setRect is called
        self.gatemap_img.setImage(np.zeros((b_len, v_len)))
        self.vslice_img.setImage(np.zeros((b_len, num_pixels)))
        self.hslice_img.setImage(np.zeros((v_len, num_pixels)))
        self.t2_heatmap_img.setImage(np.zeros((b_len, v_len)))
        self.t3_heatmap_img.setImage(np.zeros((b_len, v_len)))

        energy_raw = 1239.84193 / wavelengths
        self.energy_sort_idx = np.argsort(energy_raw)
        self.energy_axis = energy_raw[self.energy_sort_idx]

        self.reset_plot_scaling()
        self.set_inputs_enabled(False)
        self.is_running = True
        self.is_paused = False

        self.btn_begin.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Interrupt")
        self.btn_save.setEnabled(False)

        # Calculate total expected duration
        v_len = len(self.v_coords)
        b_len = len(self.b_coords)
        v_arr = self.v_coords
        b_arr = self.b_coords

        # 1. Magnet sweep rate (T/s) and B-field Ramping Time
        try:
            magnet = getattr(self.station, "magnet", None)
            if magnet is not None:
                sweep_rate = magnet.field_sweep_rate() / 60.0 if not magnet.simulated else magnet._sim_sweep_rate / 60.0
            else:
                sweep_rate = 0.5 / 60.0
            if sweep_rate <= 0:
                sweep_rate = 1.05 / 60.0
        except Exception:
            sweep_rate = 0.5 / 60.0

        try:
            curr_b = magnet.output_field() if magnet is not None else self.b_start.value()
        except Exception:
            curr_b = self.b_start.value()

        # 2. SMU initial voltages
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

        a = self.coeff_a.value()
        b = self.coeff_b.value()
        c = self.coeff_c.value()
        d = self.coeff_d.value()
        v_fixed = self.v_fixed.value()

        # Generate sequence of points
        v_points = []
        b_points = []
        is_b_outer = (self.variable_order.currentText() == "Hold B, scan V")
        serpentine = self.serpentine.isChecked()

        if is_b_outer:
            outer_arr = b_arr
            inner_arr = v_arr
        else:
            outer_arr = v_arr
            inner_arr = b_arr

        for idx_outer, val_outer in enumerate(outer_arr):
            if serpentine and (idx_outer % 2 == 1):
                inner_indices = list(range(len(inner_arr) - 1, -1, -1))
            else:
                inner_indices = list(range(len(inner_arr)))
            for idx_inner in inner_indices:
                val_inner = inner_arr[idx_inner]
                voltage = val_inner if is_b_outer else val_outer
                b_val = val_outer if is_b_outer else val_inner
                
                v1_target = a * voltage + b * v_fixed
                v2_target = c * voltage + d * v_fixed
                v_points.append((v1_target, v2_target))
                b_points.append(b_val)

        if self.end_ramp_zero.isChecked():
            b_points.append(0.0)

        # 3. Measurement points duration (including multi-exposure setting)
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
            b_points=b_points,
            exposure_time=self.exposure_box.value(),
            settling_time=self.settling_box.value(),
            num_spectra=num_spectra,
            sweep_rate_b=sweep_rate,
            initial_b=curr_b,
            initial_v1=prev_v1,
            initial_v2=prev_v2
        )
        self.sweep_v_points = v_points
        self.sweep_b_points = b_points
        self.sweep_num_spectra = num_spectra
        self.sweep_rate = sweep_rate
        self.sweep_initial_v1 = prev_v1
        self.sweep_initial_v2 = prev_v2
        self.sweep_initial_b = curr_b
        self.sweep_start_time = time.time()
        self.total_paused_time = 0.0
        self.pause_start_time = None

        self.lbl_expected_end.setText("Expected End: (waiting for measurements)")

        # Construct params dictionary
        params = {
            'a': self.coeff_a.value(),
            'b': self.coeff_b.value(),
            'c': self.coeff_c.value(),
            'd': self.coeff_d.value(),
            'v_start': self.v_start.value(),
            'v_end': self.v_end.value(),
            'v_pts': self.v_pts.value(),
            'v_fixed': self.v_fixed.value(),
            'b_start': self.b_start.value(),
            'b_end': self.b_end.value(),
            'b_pts': self.b_pts.value(),
            'variable_order': self.variable_order.currentText(),
            'serpentine': self.serpentine.isChecked(),
            'settling_time': self.settling_box.value(),
            'exposure_time': self.exposure_box.value(),
            'faraday_comp': self.faraday_comp.isChecked(),
            'faraday_rot': self.faraday_rot.value(),
            'end_ramp_zero': self.end_ramp_zero.isChecked(),
            'end_ramp_ground': self.end_ramp_ground.isChecked(),
            'num_pixels': num_pixels
        }

        self.sweep_worker = BVSweepWorker(self.station, params)
        self.sweep_worker.point_measured.connect(self.on_point_measured)
        self.sweep_worker.progress_updated.connect(self.on_progress_updated)
        self.sweep_worker.status_updated.connect(self.lbl_status.setText)
        self.sweep_worker.finished.connect(self.on_sweep_finished)
        self.sweep_worker.error_occurred.connect(self.on_sweep_error)
        
        # Connect visual parameter updates
        self.sweep_worker.field_updated.connect(self.on_field_updated)
        self.sweep_worker.voltage_updated.connect(self.on_voltage_updated)
        self.sweep_worker.start()

    def on_field_updated(self, val):
        """Update current B field label."""
        self.lbl_current_b.setText(f"B = {val:.4f} T")
        self.lbl_status.setText(f"Status: Ramping B to {val:.4f} T...")

    def on_voltage_updated(self, vc1, vc2):
        """Update target voltage labels for channel 1 and channel 2."""
        self.lbl_target_vc1.setText(f"V1 = {vc1:.4f} V")
        self.lbl_target_vc2.setText(f"V2 = {vc2:.4f} V")

    def toggle_pause(self):
        if not self.is_running: return
        if self.is_paused:
            self.is_paused = False
            self.btn_pause.setText("⏸ Interrupt")
            self.btn_save.setEnabled(False)
            self.lbl_status.setText("Status: Sweeping...")
            
            # End pause time tracking
            if getattr(self, "pause_start_time", None) is not None:
                self.total_paused_time += time.time() - self.pause_start_time
                self.pause_start_time = None
                
            self.sweep_worker.resume()
        else:
            self.is_paused = True
            self.btn_pause.setText("▶ Resume")
            self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
            self.lbl_status.setText("Status: Paused (Interrupt)")
            
            # Start pause time tracking
            self.pause_start_time = time.time()
            
            self.sweep_worker.pause()

    def reset_sweep(self):
        """Abort the sweep and reset UI without blocking the main thread.
        If running, we let the worker thread run its safe ramp-down, and reset the UI
        in on_sweep_finished when the thread finally exits.
        """
        if self.is_running:
            self.was_aborted = True
            self.trigger_autosave(was_interrupted=True)
            
            # Stop the worker first so it knows it is aborted and sets _is_running = False
            if self.sweep_worker and self.sweep_worker.isRunning():
                self.lbl_status.setText("Status: Aborting... Ramping down magnet/SMU...")
                self.btn_begin.setEnabled(False)
                self.btn_pause.setEnabled(False)
                self.btn_reset.setEnabled(False)
                # Request stop; the worker will exit its run loop gracefully.
                self.sweep_worker.stop()
                
            # Abort camera acquisition to wake up get_spectrum()
            try:
                self.station.camera.remote_lf.abort_acquisition()
            except Exception as e:
                print(f"BVSweepWindow: Error calling abort_acquisition: {e}")
                
            if self.sweep_worker and self.sweep_worker.isRunning():
                return  # We wait for the finished/error signal to clean up

        # Reset UI state immediately if not running
        self.is_running = False
        self.station.live_status_paused = False
        self.is_paused = False
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self.btn_begin.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Interrupt")
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.set_inputs_enabled(True)
        self.progress_bar.setValue(0)
        self.lbl_expected_end.setText("Expected End: --:--:--")
        self.lbl_status.setText("Status: Sweep Reset Complete")

    def on_point_measured(self, idx_v, idx_b, v_val, b_val, v1, v2, spectrum, curr_ch1, curr_ch2, meas_v1, meas_v2, meas_b):
        sorted_spectrum = spectrum[self.energy_sort_idx]
        self.data_spectra[idx_b, idx_v, :] = sorted_spectrum
        self.data_current_ch1[idx_b, idx_v] = curr_ch1
        self.data_current_ch2[idx_b, idx_v] = curr_ch2
        self.data_measured_v1[idx_b, idx_v] = meas_v1
        self.data_measured_v2[idx_b, idx_v] = meas_v2
        self.data_measured_b[idx_b, idx_v] = meas_b
        self.data_measured_mask[idx_b, idx_v] = True
        self.update_heatmaps_and_slices()
        self.on_crosshair_moved()

    def on_progress_updated(self, current, total, expected_remaining_seconds):
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
                sweep_rate_b=self.sweep_rate,
                initial_b=last_b,
                initial_v1=last_v1,
                initial_v2=last_v2
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

    def on_sweep_finished(self):
        self.is_paused = False
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None

        if getattr(self, "was_aborted", False):
            self.trigger_autosave(was_interrupted=True)
            self.is_running = False
            self.btn_begin.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_pause.setText("⏸ Interrupt")
            self.btn_reset.setEnabled(True)
            self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
            self.set_inputs_enabled(True)
            self.lbl_status.setText("Status: Sweep Aborted and Reset")
            self.station.live_status_paused = False
            self.progress_bar.setValue(0)
            self.lbl_expected_end.setText("Expected End: --:--:--")
            self.was_aborted = False
            return

        if getattr(self, "had_error", False):
            self.had_error = False
            return

        self.trigger_autosave(was_interrupted=False)
        self.is_running = False
        self.btn_begin.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Interrupt")
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.set_inputs_enabled(True)
        self.lbl_status.setText("Status: Sweep Finished (Ready to Save)")
        self.station.live_status_paused = False
        QtWidgets.QMessageBox.information(self, "Sweep Complete", "Magnetic Field vs Voltage sweep finished successfully!")

    def on_sweep_error(self, err):
        if getattr(self, "was_aborted", False):
            return
        self.had_error = True
        self.trigger_autosave(was_interrupted=True)
        self.is_running = False
        self.is_paused = False
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self.btn_begin.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Interrupt")
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.set_inputs_enabled(True)
        self.lbl_status.setText("Status: Error occurred")
        self.station.live_status_paused = False
        QtWidgets.QMessageBox.critical(self, "Sweep Error", f"Measurement aborted due to error:\n{err}")

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

    # Dataset saving (HDF5)
    def save_sweep_data(self, file_path=None):
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
            default_filename = f"bv_sweep_{timestamp}.h5"

            if file_path is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save Measurement Data",
                    os.path.join(data_dir, default_filename),
                    "HDF5 files (*.h5);;All files (*.*)"
                )
                if not filename: return
            else:
                filename = file_path

            import h5py
            with h5py.File(filename, "w") as f:
                f.create_dataset("v_coords", data=self.v_coords)
                f.create_dataset("b_coords", data=self.b_coords)
                f.create_dataset("v1_grid", data=self.v1_grid)
                f.create_dataset("v2_grid", data=self.v2_grid)
                f.create_dataset("energy_axis", data=self.energy_axis)
                f.create_dataset("spectra", data=self.data_spectra)
                f.create_dataset("current_ch1", data=self.data_current_ch1)
                f.create_dataset("current_ch2", data=self.data_current_ch2)
                f.create_dataset("measured_v1", data=self.data_measured_v1)
                f.create_dataset("measured_v2", data=self.data_measured_v2)
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

                # Metadata
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
                f.attrs["v_fixed"] = self.v_fixed.value()
                f.attrs["faraday_comp"] = self.faraday_comp.isChecked()
                f.attrs["faraday_rot"] = self.faraday_rot.value()
                f.attrs["timestamp"] = timestamp

                # Duplicate value_storage handling removed; full value_storage already stored as attribute above.
                self.lbl_status.setText(f"Status: Saved {os.path.basename(filename)}")
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
        filename = f"autosave_bv_sweep_{timestamp}{suffix}.h5"
        file_path = os.path.join(autosave_dir, filename)
        try:
            self.save_sweep_data(file_path=file_path)
            self._autosaved_this_run = True
        except Exception as e:
            print(f"Auto-save trigger failed: {e}")

    def closeEvent(self, event):
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        self.save_storage()
        event.accept()


if __name__ == '__main__':
    print("Running B/V Sweep GUI standalone...")
    # Standalone physical/simulation initialization
    station = initialize_station(simulated=False)
    app = QtWidgets.QApplication(sys.argv)
    win = BVSweepWindow(station)
    win.show()
    sys.exit(app.exec())
