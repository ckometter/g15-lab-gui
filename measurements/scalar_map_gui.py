from scanner_safety import move_scanner, validate_map_scanner_targets
import os
import sys
import json
import time
import numpy as np
import h5py
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

try:
    from .utils import EnhancedViewBox, calculate_expected_duration, configure_plot_axes
except ImportError:
    import sys
    import os
    _measurements_dir = os.path.dirname(os.path.abspath(__file__))
    if _measurements_dir not in sys.path:
        sys.path.insert(0, _measurements_dir)
    from utils import EnhancedViewBox, calculate_expected_duration, configure_plot_axes

# Ensure parent directory is in path for importing station_setup
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from gate_safety import GateRampCancelled, validate_gate_voltage, validate_map_gate_targets
from measurement_data_paths import dated_measurement_filename, dated_measurement_path
from measurement_data_paths import measurement_data_dir, measurement_start_text, new_measurement_start, measurement_device_modes

pg.setConfigOption('background', '#ffffff')
pg.setConfigOption('foreground', '#1e293b')

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


class ScalarMapWorker(QtCore.QThread):
    """
    Background worker that performs a scalar map sweep using the powermeter.
    Emits point_measured with (ix, iy, x_val, y_val, power_val, v1, v2, c1, c2, b).
    """
    point_measured = QtCore.pyqtSignal(int, int, float, float, float, float, float, float, float, float)
    progress_updated = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, station, params, parent=None):
        super().__init__(parent)
        self.station = station
        self.params = params
        self._running = True
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
        self._running = False
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

    def set_piezo(self, axis, val):
        if not move_scanner(getattr(self.station, 'piezo300', None), {axis: val},
                            keep_running=self.check_pause_and_stop):
            raise GateRampCancelled('Map scanner ramp stopped.')

    def check_pause_and_stop(self):
        self.mutex.lock()
        try:
            while self._is_paused and self._running:
                self.pause_cond.wait(self.mutex)
            return self._running
        finally:
            self.mutex.unlock()

    def preflight_gate_scan(self, x_vals, y_vals):
        validate_map_scanner_targets(getattr(self.station, 'piezo300', None), self.params, x_vals, y_vals)
        channels = validate_map_gate_targets(self.params, x_vals, y_vals)
        if channels:
            smu = getattr(self.station, 'smu', None)
            if smu is None:
                raise RuntimeError('A gate scan requires a connected SMU.')
            for number in sorted(channels):
                channel = getattr(smu, f'ch{number}')
                if channel.source_mode() != 'VOLT':
                    raise RuntimeError(f'Channel {number} must be in voltage-source mode for a gate scan.')
                validate_gate_voltage(channel.volt(), number)
            if not self.check_pause_and_stop():
                raise GateRampCancelled('Map stopped before gate preparation.')
            smu.enforce_current_compliance()

    def set_smu_volt(self, ch, val):
        validate_gate_voltage(val, ch)
        if not self.check_pause_and_stop():
            raise GateRampCancelled('Map stopped before gate ramp.')
        channel = getattr(self.station.smu, f'ch{ch}')
        self.mutex.lock()
        try:
            if not self._running:
                raise GateRampCancelled('Map stopped before gate ramp.')
            channel.ramp_stop_requested = False
        finally:
            self.mutex.unlock()
        # A failed/cancelled ramp must propagate and prevent acquisition.
        channel.set_r.volt(val)
        if channel.ramp_stop_requested or not self.check_pause_and_stop():
            raise GateRampCancelled('Map gate ramp cancelled.')

    def set_magnet(self, val):
        try:
            if hasattr(self.station, 'magnet'):
                self.station.magnet.field(val)
        except Exception as e:
            print(f"Error setting magnet field: {e}")

    def apply_sweep_values(self, px, vx, py, vy):
        piezo_updated = False
        # 1. Magnet (B)
        if px == "B":
            self.set_magnet(vx)
        if py == "B":
            self.set_magnet(vy)
            
        # 2. Attocubes (Position X, Position Y)
        if px == "position X":
            self.set_piezo(1, vx)
            piezo_updated = True
        if py == "position X":
            self.set_piezo(1, vy)
            piezo_updated = True
            
        if px == "position Y":
            if piezo_updated:
                time.sleep(0.02)
            self.set_piezo(2, vx)
        if py == "position Y":
            if piezo_updated:
                time.sleep(0.02)
            self.set_piezo(2, vy)
            
        # 3. SMU Channels (V1, V2)
        if px == "V1":
            self.set_smu_volt(1, vx)
        if py == "V1":
            self.set_smu_volt(1, vy)
            
        if px == "V2":
            self.set_smu_volt(2, vx)
        if py == "V2":
            self.set_smu_volt(2, vy)
            
        # 4. Virtual Gates (Vc1, Vc2)
        c1_name = self.params.get('c1_name', 'Vc1')
        c2_name = self.params.get('c2_name', 'Vc2')
        if px in ("Vc1", "Vc2", c1_name, c2_name) or py in ("Vc1", "Vc2", c1_name, c2_name):
            vc1 = 0.0
            vc2 = 0.0
            if px in ("Vc1", c1_name):
                vc1 = vx
            elif px in ("Vc2", c2_name):
                vc2 = vx
            if py in ("Vc1", c1_name):
                vc1 = vy
            elif py in ("Vc2", c2_name):
                vc2 = vy
                 
            a = self.params.get('coeff_a', 1.0)
            b = self.params.get('coeff_b', 0.0)
            c = self.params.get('coeff_c', 0.0)
            d = self.params.get('coeff_d', 1.0)
            
            v1 = a * vc1 + b * vc2
            v2 = c * vc1 + d * vc2
            self.set_smu_volt(1, v1)
            self.set_smu_volt(2, v2)

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
        except GateRampCancelled as e:
            if self._running:
                self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _do_run(self):
        x_start = self.params['x_start']
        x_end = self.params['x_end']
        x_pts = self.params['x_pts']
        y_start = self.params['y_start']
        y_end = self.params['y_end']
        y_pts = self.params['y_pts']
        x_param = self.params['x_param']
        y_param = self.params['y_param']

        x_vals = np.linspace(x_start, x_end, x_pts)
        y_vals = np.linspace(y_start, y_end, y_pts)
        self.preflight_gate_scan(x_vals, y_vals)
        total = x_pts * y_pts
        cnt = 0

        pm = getattr(self.station, 'powermeter', None)

        for ix, xv in enumerate(x_vals):
            if not self._running:
                return
            for iy, yv in enumerate(y_vals):
                # Thread pause / stop safety check
                self.mutex.lock()
                while self._is_paused:
                    self.pause_cond.wait(self.mutex)
                if not self._running:
                    self.mutex.unlock()
                    return
                self.mutex.unlock()

                # Apply physical values for chosen sweep parameter
                self.apply_sweep_values(x_param, xv, y_param, yv)

                # Settling time delay (0.1 s default)
                settling_time = self.params.get('settling_time', 0.1)
                if settling_time > 0:
                    sleep_end = time.time() + settling_time
                    while time.time() < sleep_end:
                        self.mutex.lock()
                        while self._is_paused:
                            self.pause_cond.wait(self.mutex)
                        if not self._running:
                            self.mutex.unlock()
                            return
                        self.mutex.unlock()
                        time.sleep(0.01)

                if not self.check_pause_and_stop():
                    return

                # Acquire power value
                power_val = None
                if pm is not None:
                    try:
                        power_val = pm.power()
                    except Exception as e:
                        if self._running:
                            print(f"ScalarMapWorker error acquiring power: {e}")
                        power_val = None

                if power_val is None:
                    # Simulated powermeter peak fallback
                    peak_center_x = (x_start + x_end) / 2.0
                    peak_center_y = (y_start + y_end) / 2.0
                    width_x = (x_end - x_start) / 4.0 if x_end != x_start else 1.0
                    width_y = (y_end - y_start) / 4.0 if y_end != y_start else 1.0
                    dist_sq = ((xv - peak_center_x) / width_x)**2 + ((yv - peak_center_y) / width_y)**2
                    amplitude = 1.5e-5
                    base = 1e-6
                    power_val = amplitude * np.exp(-dist_sq / 2.0) + base + np.random.normal(0, 1e-8)

                # Read SMU currents and voltages if SMU is present
                curr_ch1, meas_v1 = 0.0, 0.0
                curr_ch2, meas_v2 = 0.0, 0.0
                smu = getattr(self.station, 'smu', None)
                if smu:
                    try:
                        ch1_on = bool(smu.ch1.output())
                        curr_ch1 = smu.ch1.measure_curr() if ch1_on else 0.0
                        meas_v1 = smu.ch1.measure_volt() if ch1_on else 0.0
                    except Exception as e:
                        print(f"ScalarMapWorker error measuring SMU ch1: {e}")
                    try:
                        ch2_on = bool(smu.ch2.output())
                        curr_ch2 = smu.ch2.measure_curr() if ch2_on else 0.0
                        meas_v2 = smu.ch2.measure_volt() if ch2_on else 0.0
                    except Exception as e:
                        print(f"ScalarMapWorker error measuring SMU ch2: {e}")

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

                self.point_measured.emit(ix, iy, xv, yv, power_val, meas_v1, meas_v2, curr_ch1, curr_ch2, meas_b)
                cnt += 1
                self.progress_updated.emit(cnt, total)


class ViewControlPopup(QtWidgets.QFrame):
    """Dropdown style floating popup widget containing controls for autoscaling layout plots."""
    def __init__(self, parent_window):
        super().__init__(parent_window, QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint)
        self.parent_window = parent_window
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
        
        layout.addWidget(QtWidgets.QLabel("Target Plots:"))
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(4)
        
        self.btn_grid_A = QtWidgets.QPushButton("")
        self.btn_grid_B = QtWidgets.QPushButton("")
        self.btn_grid_C = QtWidgets.QPushButton("")
        
        self.btn_grid_A.setToolTip("Top-Left: 2D Power Map")
        self.btn_grid_B.setToolTip("Top-Right: Slice at Fixed X")
        self.btn_grid_C.setToolTip("Bottom-Left: Slice at Fixed Y")
        
        for btn in (self.btn_grid_A, self.btn_grid_B, self.btn_grid_C):
            btn.setCheckable(True)
            btn.setChecked(True)
            
        btn_style = """
            QPushButton {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                font-weight: bold;
                color: #1e293b;
                height: 22px;
                width: 22px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
            QPushButton:checked {
                background-color: #3b82f6;
                color: #ffffff;
                border-color: #3b82f6;
            }
        """
        self.btn_grid_A.setStyleSheet(btn_style)
        self.btn_grid_B.setStyleSheet(btn_style)
        self.btn_grid_C.setStyleSheet(btn_style)
        
        grid.addWidget(self.btn_grid_A, 0, 0)
        grid.addWidget(self.btn_grid_B, 0, 1)
        grid.addWidget(self.btn_grid_C, 1, 0)
        
        layout.addLayout(grid)
        
        # 2. Autoscale Axes options
        layout.addWidget(QtWidgets.QLabel("Autoscale Axis:"))
        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.setExclusive(True)
        
        self.btn_axis_both = QtWidgets.QToolButton()
        self.btn_axis_both.setText("↔↕")
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
                color: #1e293b;
                border-radius: 4px;
                padding: 4px 6px;
            }
            QToolButton:hover {
                background-color: #f1f5f9;
            }
            QToolButton:checked {
                background-color: #3b82f6;
                color: #ffffff;
                border-color: #3b82f6;
            }
        """
        for btn in (self.btn_axis_both, self.btn_axis_x, self.btn_axis_y):
            btn.setCheckable(True)
            btn.setStyleSheet(axis_btn_style)
            self.btn_group.addButton(btn)
            
        self.btn_axis_both.setChecked(True)
        
        row_axis = QtWidgets.QHBoxLayout()
        row_axis.setSpacing(4)
        row_axis.addWidget(self.btn_axis_both)
        row_axis.addWidget(self.btn_axis_x)
        row_axis.addWidget(self.btn_axis_y)
        layout.addLayout(row_axis)
        
        self.btn_scale_now = QtWidgets.QPushButton("Autoscale Now")
        self.btn_scale_now.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                font-weight: bold;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 4px 0px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.btn_scale_now.clicked.connect(self.parent_window.perform_autoscale)
        layout.addWidget(self.btn_scale_now)
        
    def get_selected_axes(self):
        if self.btn_axis_both.isChecked():
            return 'both'
        elif self.btn_axis_x.isChecked():
            return 'x'
        elif self.btn_axis_y.isChecked():
            return 'y'
        return 'both'
        
    def focusOutEvent(self, event):
        self.hide()
        super().focusOutEvent(event)


class ScalarMapWindow(QtWidgets.QMainWindow):
    """
    GUI window tab for performing a 2D scalar map sweep with the powermeter.
    """
    def __init__(self, station, embed=False, parent=None):
        super().__init__(parent)
        self.station = station
        self.embed = embed
        self.is_running = False
        self.is_paused = False
        self.was_aborted = False
        self.sweep_worker = None
        self._autosaved_this_run = False
        
        self.x_coords = None
        self.y_coords = None
        self.data_power = None
        self.data_measured_v1 = None
        self.data_measured_v2 = None
        self.data_measured_b = None
        self.data_current_ch1 = None
        self.data_current_ch2 = None
        self.data_measured_mask = None
        
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None

        self._init_ui()
        self.apply_theme()
        self.load_storage()
        self.update_coordinates_from_ui()
        self.reset_plot_bounds()
        x_center = (self.x_coords[0] + self.x_coords[-1]) / 2.0
        y_center = (self.y_coords[0] + self.y_coords[-1]) / 2.0
        self.update_crosshair_values(x_center, y_center)

        # Initialize status updates
        self.live_status_worker = LiveStatusWorker(self.station)
        self.live_status_worker.status_updated.connect(self.update_live_labels)
        self.live_status_worker.start()

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ----- Left Controls Panel -----
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        param_group = QtWidgets.QGroupBox("Parameter setup")
        param_layout = QtWidgets.QFormLayout(param_group)
        left_layout.addWidget(param_group)

        self.x_param_cb = QtWidgets.QComboBox()
        self.y_param_cb = QtWidgets.QComboBox()
        param_list = ["position X", "position Y", "V1", "V2", getattr(self, "c1_name", "Vc1"), getattr(self, "c2_name", "Vc2"), "B"]
        self.x_param_cb.addItems(param_list)
        self.y_param_cb.addItems(param_list)
        param_layout.addRow("X parameter", self.x_param_cb)
        param_layout.addRow("Y parameter", self.y_param_cb)

        self.scan_group = QtWidgets.QGroupBox("Scan area")
        scan_layout = QtWidgets.QGridLayout(self.scan_group)
        scan_layout.setContentsMargins(12, 12, 12, 12)
        scan_layout.setSpacing(8)
        left_layout.addWidget(self.scan_group)

        self.x_start_spin = QtWidgets.QDoubleSpinBox(); self.x_start_spin.setRange(-1e6, 1e6); self.x_start_spin.setDecimals(3)
        self.x_end_spin   = QtWidgets.QDoubleSpinBox(); self.x_end_spin.setRange(-1e6, 1e6);   self.x_end_spin.setDecimals(3)
        self.x_pts_spin   = QtWidgets.QSpinBox();       self.x_pts_spin.setRange(1, 10000)
        self.y_start_spin = QtWidgets.QDoubleSpinBox(); self.y_start_spin.setRange(-1e6, 1e6); self.y_start_spin.setDecimals(3)
        self.y_end_spin   = QtWidgets.QDoubleSpinBox(); self.y_end_spin.setRange(-1e6, 1e6);   self.y_end_spin.setDecimals(3)
        self.y_pts_spin   = QtWidgets.QSpinBox();       self.y_pts_spin.setRange(1, 10000)
        
        self.pixel_x_spin = QtWidgets.QDoubleSpinBox(); self.pixel_x_spin.setRange(0, 1e6);     self.pixel_x_spin.setDecimals(3)
        self.pixel_y_spin = QtWidgets.QDoubleSpinBox(); self.pixel_y_spin.setRange(0, 1e6);     self.pixel_y_spin.setDecimals(3)

        self.x_start_spin.valueChanged.connect(self.recalc_pixel_x)
        self.x_end_spin.valueChanged.connect(self.recalc_pixel_x)
        self.x_pts_spin.valueChanged.connect(self.recalc_pixel_x)
        self.y_start_spin.valueChanged.connect(self.recalc_pixel_y)
        self.y_end_spin.valueChanged.connect(self.recalc_pixel_y)
        self.y_pts_spin.valueChanged.connect(self.recalc_pixel_y)
        self.pixel_x_spin.valueChanged.connect(self.recalc_pts_x)
        self.pixel_y_spin.valueChanged.connect(self.recalc_pts_y)

        # Header labels
        lbl_hdr_x = QtWidgets.QLabel("X")
        lbl_hdr_x.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_hdr_x.setStyleSheet("font-weight: bold; color: #000000;")
        lbl_hdr_y = QtWidgets.QLabel("Y")
        lbl_hdr_y.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_hdr_y.setStyleSheet("font-weight: bold; color: #000000;")

        scan_layout.addWidget(lbl_hdr_x, 0, 1)
        scan_layout.addWidget(lbl_hdr_y, 0, 2)

        # Start Row
        lbl_start = QtWidgets.QLabel("Start")
        lbl_start.setStyleSheet("font-weight: bold; color: #000000;")
        scan_layout.addWidget(lbl_start, 1, 0)
        scan_layout.addWidget(self.x_start_spin, 1, 1)
        scan_layout.addWidget(self.y_start_spin, 1, 2)

        # End Row
        lbl_end = QtWidgets.QLabel("End")
        lbl_end.setStyleSheet("font-weight: bold; color: #000000;")
        scan_layout.addWidget(lbl_end, 2, 0)
        scan_layout.addWidget(self.x_end_spin, 2, 1)
        scan_layout.addWidget(self.y_end_spin, 2, 2)

        # No. points Row
        lbl_pts = QtWidgets.QLabel("No. points")
        lbl_pts.setStyleSheet("font-weight: bold; color: #000000;")
        scan_layout.addWidget(lbl_pts, 3, 0)
        scan_layout.addWidget(self.x_pts_spin, 3, 1)
        scan_layout.addWidget(self.y_pts_spin, 3, 2)

        # Pixel size Row
        lbl_pixel = QtWidgets.QLabel("Pixel size")
        lbl_pixel.setStyleSheet("font-weight: bold; color: #000000;")
        scan_layout.addWidget(lbl_pixel, 4, 0)
        scan_layout.addWidget(self.pixel_x_spin, 4, 1)
        scan_layout.addWidget(self.pixel_y_spin, 4, 2)

        meas_box = QtWidgets.QGroupBox("Measurement Controls")
        meas_layout = QtWidgets.QVBoxLayout(meas_box)
        meas_layout.setSpacing(6)
        
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_begin = QtWidgets.QPushButton("▶ Begin Sweep")
        self.btn_begin.clicked.connect(self.begin_sweep)
        ctrl_layout.addWidget(self.btn_begin)

        self.btn_pause = QtWidgets.QPushButton("⏸ Interrupt")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.on_interrupt_clicked)
        ctrl_layout.addWidget(self.btn_pause)
        meas_layout.addLayout(ctrl_layout)

        btn_layout2 = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("↺ Reset")
        self.btn_reset.setEnabled(True)
        self.btn_reset.clicked.connect(self.reset_sweep)
        btn_layout2.addWidget(self.btn_reset)

        self.btn_save = QtWidgets.QPushButton("💾 Save Dataset")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_data)
        btn_layout2.addWidget(self.btn_save)
        meas_layout.addLayout(btn_layout2)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        meas_layout.addWidget(self.progress_bar)

        self.lbl_expected_end = QtWidgets.QLabel("Expected End: --:--:--")
        self.lbl_expected_end.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; font-weight: bold; color: #000000;")
        meas_layout.addWidget(self.lbl_expected_end)

        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("color: #000000; font-style: italic; font-weight: bold;")
        meas_layout.addWidget(self.lbl_status)

        self.lbl_target_vc1 = QtWidgets.QLabel("V1 = --‑‑ V")
        self.lbl_target_vc1.setStyleSheet("color: #000000; font-weight: bold;")
        meas_layout.addWidget(self.lbl_target_vc1)

        self.lbl_target_vc2 = QtWidgets.QLabel("V2 = --‑‑ V")
        self.lbl_target_vc2.setStyleSheet("color: #000000; font-weight: bold;")
        meas_layout.addWidget(self.lbl_target_vc2)

        self.lbl_current_b = QtWidgets.QLabel("B = --‑‑ T")
        self.lbl_current_b.setStyleSheet("color: #000000; font-weight: bold;")
        meas_layout.addWidget(self.lbl_current_b)

        left_layout.addWidget(meas_box)
        left_layout.addStretch(1)

        # ----- Right side – Header layout + 3-panel plot layout -----
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        splitter.addWidget(right_widget)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(5, 5, 5, 0)
        header_layout.addStretch()

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
            }
        """)
        header_layout.addWidget(self.btn_view_control)
        right_layout.addLayout(header_layout)

        self.view_control_popup = ViewControlPopup(self)
        self.btn_view_control.clicked.connect(self.toggle_view_control_popup)

        # 3-panel Grid Layout (Plot D is omitted)
        grid_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_widget)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout.setSpacing(8)
        right_layout.addWidget(grid_widget)

        self.map_plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title="XY Map (Power)")
        self.slice_y_plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Vertical Slice")
        self.slice_x_plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Horizontal Slice")

        grid_layout.addWidget(self.map_plot, 0, 0)
        grid_layout.addWidget(self.slice_y_plot, 0, 1)
        grid_layout.addWidget(self.slice_x_plot, 1, 0)

        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        # Apply unified styling (white background and slate axes)
        for p in [self.map_plot, self.slice_y_plot, self.slice_x_plot]:
            p.setBackground('w')
            p.getAxis('left').setPen(pg.mkPen('#334155'))
            p.getAxis('bottom').setPen(pg.mkPen('#334155'))
            p.getAxis('left').setTextPen(pg.mkPen('#334155'))
            p.getAxis('bottom').setTextPen(pg.mkPen('#334155'))

        # Setup Image and curves
        self.map_img = pg.ImageItem()
        self.map_plot.addItem(self.map_img)

        # Colorbar for map
        self.map_colorbar = self.setup_histogram_lut(self.map_plot, self.map_img, interactive=False)

        # 1D slice curves
        self.slice_y_curve = self.slice_y_plot.plot(pen=pg.mkPen('#e11d48', width=2))
        self.slice_x_curve = self.slice_x_plot.plot(pen=pg.mkPen('#2563eb', width=2))

        # Enable mouse scaling for slice plots
        self.slice_y_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.slice_x_plot.getViewBox().setMouseEnabled(x=True, y=True)

        # Draggable crosshair lines in Plot A
        self.cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.cross_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.map_plot.addItem(self.cross_v)
        self.map_plot.addItem(self.cross_h)

        self.cross_v.sigPositionChanged.connect(self.on_crosshair_moved)
        self.cross_h.sigPositionChanged.connect(self.on_crosshair_moved)

        # Monkey-patch ViewBox to intercept left clicks and reposition crosshair lines
        viewbox_map = self.map_plot.getViewBox()
        viewbox_map.originalMouseClickEvent = viewbox_map.mouseClickEvent
        def map_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox_map.mapSceneToView(ev.scenePos())
                self.update_crosshair_values(mouse_point.x(), mouse_point.y())
                ev.accept()
            else:
                viewbox_map.originalMouseClickEvent(ev)
        viewbox_map.mouseClickEvent = map_click_event

        # Load values to spinboxes and combos
        self.load_storage()

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
        
        # Allow the axis and the overall widget to grow dynamically to fit any text length
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
            
        plot_win.getPlotItem().layout.addItem(hist, 2, 5)
        plot_win.getPlotItem().layout.setColumnFixedWidth(4, 3)
                # Hide the gradient editor ticks for a cleaner look
        for tick in list(hist.gradient.ticks.keys()):
            tick.hide()
        hist.gradient.allowAdd = False
        
        return hist

    def toggle_view_control_popup(self):
        if self.view_control_popup.isVisible():
            self.view_control_popup.hide()
        else:
            pos = self.btn_view_control.mapToGlobal(QtCore.QPoint(self.btn_view_control.width() - self.view_control_popup.width(), self.btn_view_control.height()))
            self.view_control_popup.move(pos)
            self.view_control_popup.show()
            self.view_control_popup.setFocus()

    def perform_autoscale(self):
        target_axes = self.view_control_popup.get_selected_axes()
        
        # Plot A
        if self.view_control_popup.btn_grid_A.isChecked():
            if target_axes == 'both':
                self.map_plot.enableAutoRange(axis='x', enable=True)
                self.map_plot.enableAutoRange(axis='y', enable=True)
            elif target_axes == 'x':
                self.map_plot.enableAutoRange(axis='x', enable=True)
            elif target_axes == 'y':
                self.map_plot.enableAutoRange(axis='y', enable=True)
                
        # Plot B
        if self.view_control_popup.btn_grid_B.isChecked():
            if target_axes == 'both':
                self.slice_y_plot.enableAutoRange(axis='x', enable=True)
                self.slice_y_plot.enableAutoRange(axis='y', enable=True)
            elif target_axes == 'x':
                self.slice_y_plot.enableAutoRange(axis='x', enable=True)
            elif target_axes == 'y':
                self.slice_y_plot.enableAutoRange(axis='y', enable=True)
                
        # Plot C
        if self.view_control_popup.btn_grid_C.isChecked():
            if target_axes == 'both':
                self.slice_x_plot.enableAutoRange(axis='x', enable=True)
                self.slice_x_plot.enableAutoRange(axis='y', enable=True)
            elif target_axes == 'x':
                self.slice_x_plot.enableAutoRange(axis='x', enable=True)
            elif target_axes == 'y':
                self.slice_x_plot.enableAutoRange(axis='y', enable=True)



    def update_crosshair_values(self, x_val, y_val):
        self.cross_v.blockSignals(True)
        self.cross_h.blockSignals(True)
        self.cross_v.setValue(x_val)
        self.cross_h.setValue(y_val)
        self.cross_v.blockSignals(False)
        self.cross_h.blockSignals(False)
        self.on_crosshair_moved()

    def on_crosshair_moved(self):
        self.refresh_slice_plots(self.cross_v.value(), self.cross_h.value())

    def refresh_slice_plots(self, x_val, y_val):
        if self.data_power is None or not np.any(self.data_measured_mask):
            return
            
        idx_x = np.argmin(np.abs(self.x_coords - x_val))
        idx_y = np.argmin(np.abs(self.y_coords - y_val))

        # 1. Update Plot B (Y Slice at fixed X)
        power_vs_y = self.data_power[:, idx_x]
        self.slice_y_curve.setData(self.y_coords, power_vs_y)
        
        vb_y = self.slice_y_plot.getViewBox()
        if vb_y.state['autoRange'][1]:
            y_min, y_max = np.min(power_vs_y), np.max(power_vs_y)
            y_range = y_max - y_min if y_max != y_min else 1.0
            vb_y.setRange(yRange=(y_min - 0.1 * y_range, y_max + 0.1 * y_range), padding=0)

        # 2. Update Plot C (X Slice at fixed Y)
        power_vs_x = self.data_power[idx_y, :]
        self.slice_x_curve.setData(self.x_coords, power_vs_x)
        
        vb_x = self.slice_x_plot.getViewBox()
        if vb_x.state['autoRange'][1]:
            x_min, x_max = np.min(power_vs_x), np.max(power_vs_x)
            x_range = x_max - x_min if x_max != x_min else 1.0
            vb_x.setRange(yRange=(x_min - 0.1 * x_range, x_max + 0.1 * x_range), padding=0)

    # --- Table Calculation Slots ---
    def recalc_pixel_x(self):
        start = self.x_start_spin.value()
        end = self.x_end_spin.value()
        pts = self.x_pts_spin.value()
        self.pixel_x_spin.blockSignals(True)
        if pts > 1:
            self.pixel_x_spin.setValue(abs(end - start) / (pts - 1))
        else:
            self.pixel_x_spin.setValue(0.0)
        self.pixel_x_spin.blockSignals(False)
        self.on_sweep_range_changed()

    def recalc_pixel_y(self):
        start = self.y_start_spin.value()
        end = self.y_end_spin.value()
        pts = self.y_pts_spin.value()
        self.pixel_y_spin.blockSignals(True)
        if pts > 1:
            self.pixel_y_spin.setValue(abs(end - start) / (pts - 1))
        else:
            self.pixel_y_spin.setValue(0.0)
        self.pixel_y_spin.blockSignals(False)
        self.on_sweep_range_changed()

    def recalc_pts_x(self):
        start = self.x_start_spin.value()
        end = self.x_end_spin.value()
        pixel = self.pixel_x_spin.value()
        self.x_pts_spin.blockSignals(True)
        if pixel > 0:
            pts = int(round(abs(end - start) / pixel)) + 1
            self.x_pts_spin.setValue(max(1, pts))
        self.x_pts_spin.blockSignals(False)
        self.on_sweep_range_changed()

    def recalc_pts_y(self):
        start = self.y_start_spin.value()
        end = self.y_end_spin.value()
        pixel = self.pixel_y_spin.value()
        self.y_pts_spin.blockSignals(True)
        if pixel > 0:
            pts = int(round(abs(end - start) / pixel)) + 1
            self.y_pts_spin.setValue(max(1, pts))
        self.y_pts_spin.blockSignals(False)
        self.on_sweep_range_changed()

    # --- Live status label updates ---
    def update_live_labels(self, b_val, v1_val, v2_val):
        self.lbl_current_b.setText(f"B = {b_val:.4f} T")
        self.lbl_target_vc1.setText(f"{getattr(self, 'c1_name', 'Vc1')} = {v1_val:.4f} V")
        self.lbl_target_vc2.setText(f"{getattr(self, 'c2_name', 'Vc2')} = {v2_val:.4f} V")

    # --- Sweep Control Logic ---
    def begin_sweep(self):
        if self.is_paused:
            self.is_paused = False
            self.btn_pause.setText("⏸ Interrupt")
            self.btn_begin.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.lbl_status.setText("Status: Sweeping...")
            if getattr(self, "pause_start_time", None) is not None:
                self.total_paused_time += time.time() - self.pause_start_time
                self.pause_start_time = None
            if self.sweep_worker:
                self.sweep_worker.resume()
            return
        if self.is_running:
            return
        self.start_sweep()

    def on_interrupt_clicked(self):
        self.pause_sweep()

    def start_sweep(self):
        # Pause status worker
        self.live_status_worker.pause()
        self.station.live_status_paused = True

        # Clear SMU Visa handles if present
        smu = getattr(self.station, "smu", None)
        if smu is not None and not smu.simulated:
            try:
                smu.visa_handle.clear()
                smu.write('*CLS')
            except Exception:
                pass

        self._autosaved_this_run = False

        # Load virtual gate coefficients
        a, b, c, d = 1.0, 0.0, 0.0, 1.0
        try:
            storage_path = os.path.join(parent_dir, "value_storage")
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    vs = json.load(f)
                shared_data = vs.get("shared", {})
                dg_data = vs.get("dual_gate", {})
                bv_data = vs.get("bv_sweep", {})
                a = shared_data.get("coeff_a", dg_data.get("dg_coeff_a", bv_data.get("bv_coeff_a", 1.0)))
                b = shared_data.get("coeff_b", dg_data.get("dg_coeff_b", bv_data.get("bv_coeff_b", 0.0)))
                c = shared_data.get("coeff_c", dg_data.get("dg_coeff_c", bv_data.get("bv_coeff_c", 0.0)))
                d = shared_data.get("coeff_d", dg_data.get("dg_coeff_d", bv_data.get("bv_coeff_d", 1.0)))
        except Exception:
            pass

        params = {
            'x_param': self.x_param_cb.currentText(),
            'y_param': self.y_param_cb.currentText(),
            'x_start': self.x_start_spin.value(),
            'x_end'  : self.x_end_spin.value(),
            'x_pts'  : self.x_pts_spin.value(),
            'y_start': self.y_start_spin.value(),
            'y_end'  : self.y_end_spin.value(),
            'y_pts'  : self.y_pts_spin.value(),
            'coeff_a': a,
            'coeff_b': b,
            'coeff_c': c,
            'coeff_d': d,
            'settling_time': self.value_storage.get("shared", {}).get("settling_time", self.value_storage.get("settling_time", 0.1)) if isinstance(self.value_storage, dict) else 0.1,
            'c1_name': getattr(self, 'c1_name', 'Vc1'),
            'c2_name': getattr(self, 'c2_name', 'Vc2'),
        }
        self.value_storage['params'] = params

        self.x_coords = np.linspace(params['x_start'], params['x_end'], params['x_pts'])
        self.y_coords = np.linspace(params['y_start'], params['y_end'], params['y_pts'])

        # Initialize data grids
        self.data_power = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_measured_v1 = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_measured_v2 = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_measured_b = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_current_ch1 = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_current_ch2 = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_measured_mask = np.zeros((len(self.y_coords), len(self.x_coords)), dtype=bool)

        # Clear maps
        self.map_img.setImage(np.zeros((len(self.y_coords), len(self.x_coords))), autoLevels=True)
        self.slice_y_curve.clear()
        self.slice_x_curve.clear()

        # Set bounds
        self.reset_plot_bounds()

        self.is_running = True
        self.is_paused = False
        self.was_aborted = False
        self.lbl_status.setText("Status: Sweeping...")
        self.btn_begin.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Interrupt")
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.set_inputs_enabled(False)
        self.progress_bar.setValue(0)

        # Estimate remaining time
        self.sweep_total_duration = len(self.x_coords) * len(self.y_coords) * (params['settling_time'] + 0.05)
        self.sweep_start_time = time.time()
        self.measurement_started_at = new_measurement_start(self.sweep_start_time)
        self.measurement_device_modes = measurement_device_modes(self.station)
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self.lbl_expected_end.setText("Expected End: Calculating...")

        self.sweep_worker = ScalarMapWorker(self.station, params)
        self.sweep_worker.point_measured.connect(self._on_point_measured)
        self.sweep_worker.progress_updated.connect(self._on_progress_updated)
        self.sweep_worker.finished.connect(self._on_sweep_finished)
        self.sweep_worker.error.connect(self._on_sweep_error)
        self.sweep_worker.start()

    def pause_sweep(self):
        if not self.is_running:
            return
        if self.is_paused:
            # Resume
            self.is_paused = False
            self.btn_pause.setText("⏸ Interrupt")
            self.btn_begin.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.lbl_status.setText("Status: Sweeping...")
            if getattr(self, "pause_start_time", None) is not None:
                self.total_paused_time += time.time() - self.pause_start_time
                self.pause_start_time = None
            if self.sweep_worker:
                self.sweep_worker.resume()
        else:
            # Pause
            self.is_paused = True
            self.btn_pause.setText("▶ Resume")
            self.btn_begin.setEnabled(True)
            self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
            self.lbl_status.setText("Status: Paused (Interrupt)")
            self.pause_start_time = time.time()
            if self.sweep_worker:
                self.sweep_worker.pause()

    def reset_sweep(self):
        if self.is_running:
            self.was_aborted = True
            self.trigger_autosave(was_interrupted=True)
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.lbl_status.setText("Status: Aborting...")
            self.btn_begin.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_reset.setEnabled(False)
            self.sweep_worker.stop()
            return

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

        self.progress_bar.setValue(0)
        self.lbl_expected_end.setText("Expected End: --:--:--")
        self.lbl_status.setText("Status: Idle")
        QtCore.QTimer.singleShot(500, self.resume_status_worker)

    def resume_status_worker(self):
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        self.station.live_status_paused = False

    def _on_point_measured(self, ix, iy, xv, yv, power_val, meas_v1, meas_v2, curr_ch1, curr_ch2, meas_b):
        self.data_power[iy, ix] = power_val
        self.data_measured_v1[iy, ix] = meas_v1
        self.data_measured_v2[iy, ix] = meas_v2
        self.data_measured_b[iy, ix] = meas_b
        self.data_current_ch1[iy, ix] = curr_ch1
        self.data_current_ch2[iy, ix] = curr_ch2
        self.data_measured_mask[iy, ix] = True

        self.lbl_current_b.setText(f"B = {meas_b:.4f} T")
        self.lbl_target_vc1.setText(f"{getattr(self, 'c1_name', 'Vc1')} = {meas_v1:.4f} V")
        self.lbl_target_vc2.setText(f"{getattr(self, 'c2_name', 'Vc2')} = {meas_v2:.4f} V")

        self.update_map_plot()

        # Update crosshair position on first point measured
        if np.sum(self.data_measured_mask) == 1:
            self.update_crosshair_values(xv, yv)
        else:
            self.refresh_slice_plots(self.cross_v.value(), self.cross_h.value())

    def _on_progress_updated(self, cnt, total):
        pct = int(round(cnt / total * 100))
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(f"Status: Sweeping... ({cnt}/{total})")

        # Dynamically calculate remaining time
        if self.sweep_start_time is not None:
            elapsed = time.time() - self.sweep_start_time - self.total_paused_time
            if cnt > 0:
                avg_time = elapsed / cnt
                remaining = avg_time * (total - cnt)
                expected_end = time.time() + remaining
                time_str = time.strftime("%H:%M:%S", time.localtime(expected_end))
                self.lbl_expected_end.setText(f"Expected End: {time_str}")
        else:
            self.lbl_expected_end.setText("Expected End: --:--:--")



    def _on_sweep_finished(self):
        if getattr(self, "was_aborted", False):
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
            self.progress_bar.setValue(0)
            self.lbl_expected_end.setText("Expected End: --:--:--")
            self.lbl_status.setText("Status: Idle")
            QtCore.QTimer.singleShot(500, self.resume_status_worker)
            self.was_aborted = False
            return

        if not self.is_running:
            return

        self.trigger_autosave(was_interrupted=False)
        self.is_running = False
        self.btn_begin.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.lbl_expected_end.setText("Sweep finished")
        self.lbl_status.setText("Status: Finished")
        self.update_map_plot()
        self.set_inputs_enabled(True)
        QtCore.QTimer.singleShot(500, self.resume_status_worker)
        QtWidgets.QMessageBox.information(self, "Sweep Complete", "Scalar map sweep finished successfully!")

    def _on_sweep_error(self, msg):
        if getattr(self, "was_aborted", False):
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
            self.progress_bar.setValue(0)
            self.lbl_expected_end.setText("Expected End: --:--:--")
            self.lbl_status.setText("Status: Idle")
            QtCore.QTimer.singleShot(500, self.resume_status_worker)
            self.was_aborted = False
            return

        self.is_running = False
        self.btn_begin.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.lbl_status.setText(f"Status: Error ({msg})")
        self.set_inputs_enabled(True)
        QtCore.QTimer.singleShot(500, self.resume_status_worker)

    def update_map_plot(self):
        if self.data_power is None:
            return
        map_data = self.data_power
        self.map_img.setImage(map_data, autoLevels=True)
        if map_data.size > 0:
            min_v, max_v = np.min(map_data), np.max(map_data)
            if min_v == max_v:
                max_v += 1e-9
            self.map_colorbar.setLevels(min_v, max_v)

    def update_coordinates_from_ui(self):
        """Build coordinates and grids from UI control values."""
        x_start = self.x_start_spin.value()
        x_end = self.x_end_spin.value()
        x_pts = self.x_pts_spin.value()

        y_start = self.y_start_spin.value()
        y_end = self.y_end_spin.value()
        y_pts = self.y_pts_spin.value()

        self.x_coords = np.linspace(x_start, x_end, x_pts) if x_pts > 1 else np.array([x_start])
        self.y_coords = np.linspace(y_start, y_end, y_pts) if y_pts > 1 else np.array([y_start])

    def on_sweep_range_changed(self):
        """Called dynamically when sweep parameters are changed in the UI."""
        if not self.is_running:
            self.update_coordinates_from_ui()
            self.reset_plot_bounds()
            self.save_storage()

    def reset_plot_bounds(self):
        if self.x_coords is None or self.y_coords is None:
            return
            
        x_min, x_max = min(self.x_coords), max(self.x_coords)
        y_min, y_max = min(self.y_coords), max(self.y_coords)
        
        x_param = self.x_param_cb.currentText()
        y_param = self.y_param_cb.currentText()
        
        # Update labels dynamically
        self.map_plot.setLabel('bottom', x_param)
        self.map_plot.setLabel('left', y_param)
        
        self.slice_y_plot.setLabel('left', 'Power', units='W')
        self.slice_y_plot.setLabel('bottom', y_param)
        
        self.slice_x_plot.setLabel('left', 'Power', units='W')
        self.slice_x_plot.setLabel('bottom', x_param)
        
        width_map = x_max - x_min if x_max != x_min else 1.0
        height_map = y_max - y_min if y_max != y_min else 1.0
        rect_map = QtCore.QRectF(x_min, y_min, width_map, height_map)
        if self.map_img.image is not None:
            self.map_img.setRect(rect_map)
            
        self.map_plot.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)
        self.map_plot.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0)
        
        self.slice_y_plot.setLimits(xMin=y_min, xMax=y_max)
        self.slice_y_plot.setRange(xRange=(y_min, y_max), padding=0)
        
        self.slice_x_plot.setLimits(xMin=x_min, xMax=x_max)
        self.slice_x_plot.setRange(xRange=(x_min, x_max), padding=0)

    # --- Save / AutoSave ---
    def trigger_autosave(self, was_interrupted=False):
        try:
            autosave_dir = measurement_data_dir(
                parent_dir, getattr(self, "measurement_started_at", None), autosave=True
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            suffix = "_interrupted" if was_interrupted else ""
            auto_path = os.path.join(autosave_dir, dated_measurement_filename(f"autosave_scalar_map_{timestamp[-6:]}{suffix}.h5", getattr(self, "measurement_started_at", None)))
            self.save_data(file_path=auto_path)
        except Exception as e:
            print(f"ScalarMapWindow Autosave failed: {e}")

    def save_data(self, file_path=None):
        if isinstance(file_path, bool):
            file_path = None
        if not np.any(self.data_measured_mask):
            if file_path is None:
                QtWidgets.QMessageBox.warning(self, "No Data", "There is no sweep data to save.")
            return

        try:
            self.save_storage()
            data_dir = measurement_data_dir(
                parent_dir, getattr(self, "measurement_started_at", None)
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"scalar_map_{timestamp[-6:]}.h5"
            default_name = dated_measurement_filename(default_name, getattr(self, "measurement_started_at", None))
            if file_path is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save Scalar Map",
                    os.path.join(data_dir, default_name),
                    "HDF5 files (*.h5);;All files (*.*)",
                )
                if not filename:
                    return
                filename = dated_measurement_path(filename, getattr(self, "measurement_started_at", None))
            else:
                filename = file_path

            with h5py.File(filename, "w") as f:
                f.attrs["device_modes"] = json.dumps(getattr(self, "measurement_device_modes", {}))
                f.attrs["measurement_started_at"] = measurement_start_text(
                    getattr(self, "measurement_started_at", None)
                )
                f.create_dataset("x_coords", data=self.x_coords)
                f.create_dataset("y_coords", data=self.y_coords)
                f.create_dataset("power", data=self.data_power)
                f.create_dataset("measured_v1", data=self.data_measured_v1)
                f.create_dataset("measured_v2", data=self.data_measured_v2)
                if self.data_measured_b is not None and self.data_measured_b.size > 0:
                    f.create_dataset("measured_b", data=self.data_measured_b)
                f.create_dataset("current_ch1", data=self.data_current_ch1)
                f.create_dataset("current_ch2", data=self.data_current_ch2)
                f.create_dataset("measured_mask", data=self.data_measured_mask)
                
                dtype = h5py.string_dtype(encoding="utf-8")
                meta_grp = f.create_group("metadata")
                meta_grp.create_dataset("value_storage", data=json.dumps(self.value_storage), dtype=dtype)
                
                vs_data = self.value_storage
                shared = vs_data.get("shared", {})
                settling_time = shared.get("settling_time", 0.05)
                meta_grp.create_dataset("settling_time", data=settling_time)

            if file_path is None:
                QtWidgets.QMessageBox.information(self, "Saved", f"Data saved to {filename}")
            else:
                print(f"Auto-saved data to {filename}")
        except Exception as e:
            if file_path is None:
                QtWidgets.QMessageBox.critical(self, "Save error", str(e))

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

    def load_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        defaults = {
            "x_param": "position X",
            "y_param": "position Y",
            "x_start": -5.0,
            "x_end": 5.0,
            "x_pts": 11,
            "y_start": -5.0,
            "y_end": 5.0,
            "y_pts": 11,
            "pixel_x": 1.0,
            "pixel_y": 1.0,
        }
        self.stored_settings = defaults.copy()
        c1_name = "Vc1"
        c2_name = "Vc2"
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    raw_data = json.load(f)
                self.value_storage = raw_data
                sm_data = raw_data.get("scalar_map", {})
                shared_data = raw_data.get("shared", {})
                c1_name = shared_data.get("c1_name", "Vc1")
                c2_name = shared_data.get("c2_name", "Vc2")
                for key in defaults:
                    self.stored_settings[key] = sm_data.get(key, raw_data.get(key, defaults[key]))
            except Exception as e:
                print(f"Error loading value_storage in ScalarMap: {e}")
        
        self.c1_name = c1_name
        self.c2_name = c2_name

        self.x_param_cb.blockSignals(True)
        self.y_param_cb.blockSignals(True)
        self.x_param_cb.clear()
        self.y_param_cb.clear()
        param_list = ["position X", "position Y", "V1", "V2", c1_name, c2_name, "B"]
        self.x_param_cb.addItems(param_list)
        self.y_param_cb.addItems(param_list)

        x_val = self.stored_settings["x_param"]
        if x_val == "Vc1": x_val = c1_name
        elif x_val == "Vc2": x_val = c2_name
        y_val = self.stored_settings["y_param"]
        if y_val == "Vc1": y_val = c1_name
        elif y_val == "Vc2": y_val = c2_name

        self.x_param_cb.setCurrentText(x_val)
        self.y_param_cb.setCurrentText(y_val)
        self.x_param_cb.blockSignals(False)
        self.y_param_cb.blockSignals(False)

        self.x_start_spin.setValue(self.stored_settings["x_start"])
        self.x_end_spin.setValue(self.stored_settings["x_end"])
        self.x_pts_spin.setValue(self.stored_settings["x_pts"])
        self.y_start_spin.setValue(self.stored_settings["y_start"])
        self.y_end_spin.setValue(self.stored_settings["y_end"])
        self.y_pts_spin.setValue(self.stored_settings["y_pts"])
        self.pixel_x_spin.setValue(self.stored_settings["pixel_x"])
        self.pixel_y_spin.setValue(self.stored_settings["pixel_y"])
        
        if isinstance(self.value_storage, dict):
            if "scalar_map" not in self.value_storage:
                self.value_storage["scalar_map"] = {}
            for key, val in self.stored_settings.items():
                self.value_storage["scalar_map"][key] = val
        else:
            self.value_storage = self.stored_settings.copy()

    def save_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        try:
            current_data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    current_data = json.load(f)
            
            if "scalar_map" not in current_data:
                current_data["scalar_map"] = {}
                
            current_data["scalar_map"]["x_param"] = self.x_param_cb.currentText()
            current_data["scalar_map"]["y_param"] = self.y_param_cb.currentText()
            current_data["scalar_map"]["x_start"] = self.x_start_spin.value()
            current_data["scalar_map"]["x_end"] = self.x_end_spin.value()
            current_data["scalar_map"]["x_pts"] = self.x_pts_spin.value()
            current_data["scalar_map"]["y_start"] = self.y_start_spin.value()
            current_data["scalar_map"]["y_end"] = self.y_end_spin.value()
            current_data["scalar_map"]["y_pts"] = self.y_pts_spin.value()
            current_data["scalar_map"]["pixel_x"] = self.pixel_x_spin.value()
            current_data["scalar_map"]["pixel_y"] = self.pixel_y_spin.value()
            
            self.value_storage = current_data
            with open(storage_path, "w") as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Error saving value_storage: {e}")

    def closeEvent(self, event):
        if hasattr(self, 'live_status_worker') and self.live_status_worker and self.live_status_worker.isRunning():
            self.live_status_worker.stop()
            self.live_status_worker.wait()
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        self.save_storage()
        event.accept()

    def reload_gui_from_storage(self):
        self.load_storage()

    def set_inputs_enabled(self, enabled: bool):
        self.x_param_cb.setEnabled(enabled)
        self.y_param_cb.setEnabled(enabled)
        self.scan_group.setEnabled(enabled)

    def centralWidget(self):
        return self


if __name__ == '__main__':
    from station_setup import initialize_station
    station = initialize_station(simulated=True)
    app = QtWidgets.QApplication(sys.argv)
    w = ScalarMapWindow(station)
    w.show()
    sys.exit(app.exec())
