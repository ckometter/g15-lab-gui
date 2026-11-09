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
from measurement_acquisition import acquire_sweep_spectrum

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


class PLMapWorker(QtCore.QThread):
    """Background worker that performs a photoluminescence map sweep.

    Emits `point_measured` with (ix, iy, x_val, y_val, spectrum).
    Emits `progress_updated` with (current, total).
    """
    point_measured = QtCore.pyqtSignal(int, int, float, float, np.ndarray, float, float, float, float, float, np.ndarray)
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
            
        # 4. Virtual Gates (Vc1, Vc2 or their custom names)
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
        try:
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

            camera = getattr(self.station, 'camera', None)
            mono = getattr(self.station, 'monochromator', None)
            
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
                    print(f"PLMapWorker: could not read value_storage ({e}), using defaults")

            # Use the same frame count as the preallocated data array.
            num_spectra = int(self.params.get("num_spectra", num_spectra))

            if camera:
                try:
                    camera.num_frames(num_spectra)
                except Exception as e:
                    print(f"PLMapWorker: Error setting remote num_frames: {e}")
            
            # Retrieve center wavelength
            try:
                mono_wl = mono.wavelength() if mono else 800.0
            except Exception:
                mono_wl = 800.0

            pixels = self.params.get('num_pixels', 512)
            wl_axis = self.params.get('wl_axis', None)
            if wl_axis is None:
                wl_axis = mono_wl + (np.arange(pixels) - pixels // 2) * 0.05

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

                    # Settling time delay (0.1 s delay after setting voltages before taking spectrum)
                    settling_time = self.params.get('settling_time', 0.1)
                    if settling_time > 0:
                        sleep_end = time.time() + settling_time
                        while time.time() < sleep_end:
                            # Thread pause / stop safety check
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

                    try:
                        spectrum, raw_frames = acquire_sweep_spectrum(
                            camera, pixels=pixels, frames=num_spectra,
                            average=do_average or accum_method == "Average",
                            wavelengths=wl_axis,
                        )
                    except Exception:
                        if not self._running:
                            return
                        raise

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
                            print(f"PLMapWorker error measuring SMU ch1: {e}")
                        try:
                            ch2_on = bool(smu.ch2.output())
                            curr_ch2 = smu.ch2.measure_curr() if ch2_on else 0.0
                            meas_v2 = smu.ch2.measure_volt() if ch2_on else 0.0
                        except Exception as e:
                            print(f"PLMapWorker error measuring SMU ch2: {e}")

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

                    self.point_measured.emit(ix, iy, xv, yv, spectrum, meas_v1, meas_v2, curr_ch1, curr_ch2, meas_b, raw_frames)
                    cnt += 1
                    self.progress_updated.emit(cnt, total)
        except GateRampCancelled as e:
            if self._running:
                self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))


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
        
        # 1. 2D Layout Selector Grid
        layout.addWidget(QtWidgets.QLabel("Target Plots:"))
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(4)
        
        self.btn_grid_A = QtWidgets.QPushButton("")
        self.btn_grid_B = QtWidgets.QPushButton("")
        self.btn_grid_C = QtWidgets.QPushButton("")
        self.btn_grid_D = QtWidgets.QPushButton("")
        
        self.btn_grid_A.setToolTip("Top-Left: X-Y Map")
        self.btn_grid_C.setToolTip("Top-Right: Wavelength vs X")
        self.btn_grid_D.setToolTip("Bottom-Left: CCD Spectrum")
        self.btn_grid_B.setToolTip("Bottom-Right: Wavelength vs Y")
        
        for btn in (self.btn_grid_A, self.btn_grid_B, self.btn_grid_C, self.btn_grid_D):
            btn.setCheckable(True)
            btn.setChecked(True)
            
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
        self.btn_grid_A.setStyleSheet(btn_style)
        self.btn_grid_B.setStyleSheet(btn_style)
        self.btn_grid_C.setStyleSheet(btn_style)
        self.btn_grid_D.setStyleSheet(btn_style)
        
        grid.addWidget(self.btn_grid_A, 0, 0)
        grid.addWidget(self.btn_grid_C, 0, 1)
        grid.addWidget(self.btn_grid_D, 1, 0)
        grid.addWidget(self.btn_grid_B, 1, 1)
        layout.addLayout(grid)
        
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


class PhotoluminescenceMapWindow(QtWidgets.QWidget):
    """Photoluminescence Map measurement tab.

    Layout matches the Dual-Gate Sweep UI (with Panel B and D switched):
        A: X vs Y heatmap (Top Left)
        B: Wavelength vs Y vertical slice at selected X position (Top Right)
        C: Wavelength vs X horizontal slice at selected Y position (Bottom Left)
        D: CCD spectrum at selected crosshair position (Bottom Right)
    """
    def __init__(self, station, embed=False):
        super().__init__()
        self.station = station
        self.embed = embed
        self.reset_slice_levels = True
        
        # Data containers
        self.x_coords = None
        self.y_coords = None
        try:
            self.wavelengths = self.station.camera.wavelengths()
        except Exception:
            self.wavelengths = np.linspace(500, 600, 512)
        self.data_spectra = None
        self.data_raw_spectra = None
        self.data_measured_v1 = None
        self.data_measured_v2 = None
        self.data_measured_b = None
        self.data_current_ch1 = None
        self.data_current_ch2 = None
        self.data_measured_mask = None
        
        self._init_ui()
        self.apply_theme()
        self.is_running = False
        self.is_paused = False
        self.was_aborted = False
        self.sweep_worker = None
        
        # Time tracking for expected end calculation
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None
        
        self.value_storage = {}
        self.single_spectra_window = None
        
        # Load persistent configuration
        self.load_storage()
        self.update_coordinates_from_ui()
        self.reset_plot_bounds()
        x_center = (self.x_coords[0] + self.x_coords[-1]) / 2.0
        y_center = (self.y_coords[0] + self.y_coords[-1]) / 2.0
        self.update_crosshair_values(x_center, y_center)

        # Start background 5 Hz status polling worker
        self.live_status_worker = LiveStatusWorker(self.station, parent=self)
        self.live_status_worker.status_updated.connect(self.on_live_status_updated)
        self.live_status_worker.start()

    def on_live_status_updated(self, b_val, v1_val, v2_val):
        self.lbl_current_b.setText(f"B = {b_val:.4f} T")
        self.lbl_target_vc1.setText(f"V1 = {v1_val:.4f} V")
        self.lbl_target_vc2.setText(f"V2 = {v2_val:.4f} V")

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ----- Left side – controls -----
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        # Parameter setup box
        param_group = QtWidgets.QGroupBox("Parameter setup")
        param_layout = QtWidgets.QFormLayout(param_group)
        left_layout.addWidget(param_group)
        self.x_param_cb = QtWidgets.QComboBox()
        self.x_param_cb.addItems(["position X", "position Y", "V1", "V2", "Vc1", "Vc2", "B"])
        self.y_param_cb = QtWidgets.QComboBox()
        self.y_param_cb.addItems(["position X", "position Y", "V1", "V2", "Vc1", "Vc2", "B"])
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

        # Measurement Control Box (matching Dual_gate_sweep_gui.py / B_V_sweep_gui.py style)
        meas_box = QtWidgets.QGroupBox("Measurement Controls")
        meas_layout = QtWidgets.QVBoxLayout(meas_box)
        meas_layout.setSpacing(6)
        
        # Play/Pause control row
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("▶ Begin Sweep")
        self.btn_start.clicked.connect(self.begin_sweep)
        ctrl_layout.addWidget(self.btn_start)

        self.btn_pause = QtWidgets.QPushButton("⏸ Interrupt")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.on_interrupt_clicked)
        ctrl_layout.addWidget(self.btn_pause)
        meas_layout.addLayout(ctrl_layout)

        # Reset / Save row
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

        # Progress / expected end time
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

        # ----- Right side – Header layout + 4-panel plot layout -----
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        splitter.addWidget(right_widget)

        # Header for the View Control button
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
                border-color: #94a3b8;
            }
        """)
        header_layout.addWidget(self.btn_view_control)
        right_layout.addLayout(header_layout)

        # The view control dropdown popup
        self.view_control_popup = ViewControlPopup(self)
        self.btn_view_control.clicked.connect(self.toggle_view_control_popup)

        # Grid widget for plots
        grid_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_widget)
        grid_layout.setContentsMargins(5, 0, 5, 5)
        grid_layout.setSpacing(6)
        right_layout.addWidget(grid_widget)

        # Panel A: X vs Y heatmap (Top Left)
        self.map_plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title="2D Map (Integrated Counts)")
        self.map_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.map_img = pg.ImageItem()
        self.map_img.setOpts(axisOrder='row-major')
        self.map_plot.addItem(self.map_img)
        self.map_plot.setLabel('left', 'Y coordinate')
        self.map_plot.setLabel('bottom', 'X coordinate')
        grid_layout.addWidget(self.map_plot, 0, 0)
        self.map_colorbar = self.setup_histogram_lut(self.map_plot, self.map_img, interactive=False)

        # Crosshair lines in Panel A
        self.cross_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.cross_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.map_plot.addItem(self.cross_v)
        self.map_plot.addItem(self.cross_h)
        self.cross_v.sigPositionChanged.connect(self.on_crosshair_moved)
        self.cross_h.sigPositionChanged.connect(self.on_crosshair_moved)

        # Connect mouse click to reposition crosshair
        viewbox_map = self.map_plot.getViewBox()
        viewbox_map.originalMouseClickEvent = viewbox_map.mouseClickEvent
        def map_click_event(ev):
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                mouse_point = viewbox_map.mapSceneToView(ev.scenePos())
                self.update_crosshair_values(mouse_point.x(), mouse_point.y())
                self.refresh_slice_plots(mouse_point.x(), mouse_point.y())
                ev.accept()
            else:
                viewbox_map.originalMouseClickEvent(ev)
        viewbox_map.mouseClickEvent = map_click_event

        # Panel B: Y vs Energy (Top Right)
        self.wl_vs_y_plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Vertical Slice spectra")
        self.wl_vs_y_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.wl_vs_y_img = pg.ImageItem()
        self.wl_vs_y_img.setOpts(axisOrder='row-major')
        self.wl_vs_y_plot.addItem(self.wl_vs_y_img)
        self.wl_vs_y_plot.setLabel('left', 'Energy', units='eV')
        self.wl_vs_y_plot.setLabel('bottom', 'Y coordinate')
        grid_layout.addWidget(self.wl_vs_y_plot, 0, 1)
        self.wl_vs_y_colorbar = self.setup_histogram_lut(self.wl_vs_y_plot, self.wl_vs_y_img, interactive=True)

        # Panel C: X vs Energy (Bottom Left)
        self.wl_vs_x_plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title="Horizontal Slice spectra")
        self.wl_vs_x_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.wl_vs_x_img = pg.ImageItem()
        self.wl_vs_x_img.setOpts(axisOrder='row-major')
        self.wl_vs_x_plot.addItem(self.wl_vs_x_img)
        self.wl_vs_x_plot.setLabel('left', 'X coordinate')
        self.wl_vs_x_plot.setLabel('bottom', 'Energy', units='eV')
        grid_layout.addWidget(self.wl_vs_x_plot, 1, 0)
        self.wl_vs_x_colorbar = self.setup_histogram_lut(self.wl_vs_x_plot, self.wl_vs_x_img, interactive=True)

        # Panel D: Counts vs Energy (Bottom Right)
        self.spectrum_plot = pg.PlotWidget(viewBox=EnhancedViewBox(), title="CCD Spectrum at Crosshair")
        self.spectrum_plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen('#7c3aed', width=2))
        self.spectrum_plot.setLabel('left', 'Counts')
        self.spectrum_plot.setLabel('bottom', 'Energy', units='eV')
        grid_layout.addWidget(self.spectrum_plot, 1, 1)

        # Apply unified styling (white background and slate axes)
        for p in [self.map_plot, self.wl_vs_y_plot, self.wl_vs_x_plot, self.spectrum_plot]:
            p.setBackground('w')
            p.getAxis('left').setPen(pg.mkPen('#334155'))
            p.getAxis('bottom').setPen(pg.mkPen('#334155'))
            p.getAxis('left').setTextPen(pg.mkPen('#334155'))
            p.getAxis('bottom').setTextPen(pg.mkPen('#334155'))

        # Vertical integration window region in Panel D
        self.integration_region = pg.LinearRegionItem(
            orientation='vertical',
            brush=pg.mkBrush(16, 185, 129, 40),
            pen=pg.mkPen('#10b981', width=1.5)
        )
        self.spectrum_plot.addItem(self.integration_region, ignoreBounds=True)
        self.integration_region.sigRegionChanged.connect(self.update_map_plot)

        # Horizontal integration window region in Panel D
        self.integration_h_region = pg.LinearRegionItem(
            orientation='horizontal',
            brush=pg.mkBrush(16, 185, 129, 40),
            pen=pg.mkPen('#10b981', width=1.5)
        )
        self.spectrum_plot.addItem(self.integration_h_region, ignoreBounds=True)
        self.integration_h_region.sigRegionChanged.connect(self.update_map_plot)

    def toggle_view_control_popup(self):
        if self.view_control_popup.isVisible():
            self.view_control_popup.hide()
        else:
            pos = self.btn_view_control.mapToGlobal(QtCore.QPoint(self.btn_view_control.width() - self.view_control_popup.width(), self.btn_view_control.height()))
            self.view_control_popup.move(pos)
            self.view_control_popup.show()
            self.view_control_popup.raise_()

    def perform_autoscale(self):
        popup = self.view_control_popup
        mode = popup.get_axis_mode()
        if mode == "None":
            return
            
        targets = []
        if popup.btn_grid_A.isChecked(): targets.append(self.map_plot)
        if popup.btn_grid_B.isChecked(): targets.append(self.wl_vs_y_plot)
        if popup.btn_grid_C.isChecked(): targets.append(self.wl_vs_x_plot)
        if popup.btn_grid_D.isChecked(): targets.append(self.spectrum_plot)
        
        for plot in targets:
            vb = plot.getViewBox()
            if mode == "X":
                vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
            elif mode == "Y":
                vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            elif mode == "Both":
                vb.enableAutoRange(enable=True)

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
        
        sweep_wavelengths = self.wavelengths
        
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
        self.update_map_plot()
        self.on_crosshair_moved()

    def recalc_pixel_x(self):
        start = self.x_start_spin.value()
        end = self.x_end_spin.value()
        pts = self.x_pts_spin.value()
        if pts > 1:
            val = abs(end - start) / (pts - 1)
        else:
            val = 0.0
        self.pixel_x_spin.blockSignals(True)
        self.pixel_x_spin.setValue(val)
        self.pixel_x_spin.blockSignals(False)
        self.on_sweep_range_changed()

    def recalc_pixel_y(self):
        start = self.y_start_spin.value()
        end = self.y_end_spin.value()
        pts = self.y_pts_spin.value()
        if pts > 1:
            val = abs(end - start) / (pts - 1)
        else:
            val = 0.0
        self.pixel_y_spin.blockSignals(True)
        self.pixel_y_spin.setValue(val)
        self.pixel_y_spin.blockSignals(False)
        self.on_sweep_range_changed()

    def recalc_pts_x(self):
        start = self.x_start_spin.value()
        end = self.x_end_spin.value()
        pixel = self.pixel_x_spin.value()
        if pixel > 0:
            pts = int(round(abs(end - start) / pixel)) + 1
            pts = max(1, min(10000, pts))
            self.x_pts_spin.blockSignals(True)
            self.x_pts_spin.setValue(pts)
            self.x_pts_spin.blockSignals(False)
            self.on_sweep_range_changed()

    def recalc_pts_y(self):
        start = self.y_start_spin.value()
        end = self.y_end_spin.value()
        pixel = self.pixel_y_spin.value()
        if pixel > 0:
            pts = int(round(abs(end - start) / pixel)) + 1
            pts = max(1, min(10000, pts))
            self.y_pts_spin.blockSignals(True)
            self.y_pts_spin.setValue(pts)
            self.y_pts_spin.blockSignals(False)
            self.on_sweep_range_changed()

    def setup_histogram_lut(self, plot_win, image_item, interactive=True):
        """Helper to create a HistogramLUTItem matching the Dual-Gate Sweep style."""
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

    def start_sweep(self):
        if self.is_paused:
            # Resume sweep
            self.is_paused = False
            self.btn_pause.setText("⏸ Interrupt")
            self.btn_start.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.lbl_status.setText("Status: Sweeping...")
            if self.sweep_worker:
                self.sweep_worker.resume()
            return

        if self.is_running:
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
        
        self._autosaved_this_run = False
            
        # Read virtual gate coefficients from value_storage
        a, b, c, d = 1.0, 0.0, 0.0, 1.0
        try:
            storage_path = os.path.join(parent_dir, "value_storage")
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    raw_data = json.load(f)
                shared_data = raw_data.get("shared", {})
                dg_data = raw_data.get("dual_gate", {})
                bv_data = raw_data.get("bv_sweep", {})
                a = shared_data.get("coeff_a", dg_data.get("dg_coeff_a", bv_data.get("bv_coeff_a", 1.0)))
                b = shared_data.get("coeff_b", dg_data.get("dg_coeff_b", bv_data.get("bv_coeff_b", 0.0)))
                c = shared_data.get("coeff_c", dg_data.get("dg_coeff_c", bv_data.get("bv_coeff_c", 0.0)))
                d = shared_data.get("coeff_d", dg_data.get("dg_coeff_d", bv_data.get("bv_coeff_d", 1.0)))
        except Exception:
            pass

        # Determine number of pixels and wavelengths dynamically from camera
        num_pixels = 512 # fallback
        wavelengths = None
        
        # Read center wavelength for spectrum grid setup
        try:
            mono_wl = self.station.monochromator.wavelength()
        except Exception:
            mono_wl = 800.0

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
            
        self.wavelengths = wavelengths

        params = {
            'x_param': self.x_param_cb.currentText(),
            'y_param': self.y_param_cb.currentText(),
            'x_start': self.x_start_spin.value(),
            'x_end'  : self.x_end_spin.value(),
            'x_pts'  : self.x_pts_spin.value(),
            'y_start': self.y_start_spin.value(),
            'y_end'  : self.y_end_spin.value(),
            'y_pts'  : self.y_pts_spin.value(),
            'pixel_x': self.pixel_x_spin.value(),
            'pixel_y': self.pixel_y_spin.value(),
            'coeff_a': a,
            'coeff_b': b,
            'coeff_c': c,
            'coeff_d': d,
            'settling_time': self.value_storage.get("shared", {}).get("settling_time", self.value_storage.get("settling_time", 0.1)) if isinstance(self.value_storage, dict) else 0.1,
            'c1_name': getattr(self, 'c1_name', 'Vc1'),
            'c2_name': getattr(self, 'c2_name', 'Vc2'),
            'num_pixels': num_pixels,
            'wl_axis': wavelengths
        }
        self.value_storage['params'] = params
        
        # Set up grids and coordinates
        self.x_coords = np.linspace(params['x_start'], params['x_end'], params['x_pts'])
        self.y_coords = np.linspace(params['y_start'], params['y_end'], params['y_pts'])

        # Shaded integration region default range if needed
        e_center = 1239.84193 / mono_wl if mono_wl > 0 else 1.5
        
        curr_region = self.integration_region.getRegion()
        reg_min, reg_max = curr_region
        energies = 1239.84193 / self.wavelengths
        e_min, e_max = min(energies), max(energies)
        if reg_min < e_min or reg_max > e_max or reg_min >= reg_max or (reg_min == 0.0 and reg_max == 1.0):
            self.integration_region.blockSignals(True)
            self.integration_region.setRegion((e_center - 0.005, e_center + 0.005))
            self.integration_region.blockSignals(False)

        curr_h_region = self.integration_h_region.getRegion()
        h_reg_min, h_reg_max = curr_h_region
        if h_reg_min < 0 or h_reg_max <= h_reg_min or (h_reg_min == 0.0 and h_reg_max == 1.0):
            self.integration_h_region.blockSignals(True)
            self.integration_h_region.setRegion((200.0, 1500.0))
            self.integration_h_region.blockSignals(False)

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

        # Clear/initialize spectrum arrays
        self.data_spectra = np.zeros((len(self.y_coords), len(self.x_coords), num_pixels))
        self.data_raw_spectra = np.zeros((len(self.y_coords), len(self.x_coords), num_spectra, num_pixels))
        self.data_measured_v1 = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_measured_v2 = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_measured_b = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_current_ch1 = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_current_ch2 = np.zeros((len(self.y_coords), len(self.x_coords)))
        self.data_measured_mask = np.zeros((len(self.y_coords), len(self.x_coords)), dtype=bool)

        # Pre-initialize images to shape-defined zero matrices so setRect works immediately
        self.map_img.setImage(np.zeros((len(self.y_coords), len(self.x_coords))), autoLevels=True)
        self.wl_vs_y_img.setImage(np.zeros((len(self.y_coords), num_pixels)), autoLevels=True)
        self.wl_vs_x_img.setImage(np.zeros((len(self.x_coords), num_pixels)), autoLevels=True)

        # Set image bounds
        self.reset_plot_bounds()

        self.is_running = True
        self.is_paused = False
        self.was_aborted = False
        self.lbl_status.setText("Status: Sweeping...")
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Interrupt")
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.set_inputs_enabled(False)
        self.progress_bar.setValue(0)
        
        # Calculate total expected duration
        try:
            magnet = getattr(self.station, "magnet", None)
            init_b = magnet.output_field() if magnet is not None else 0.0
            if magnet is not None:
                sweep_rate = magnet.field_sweep_rate() / 60.0 if not magnet.simulated else magnet._sim_sweep_rate / 60.0
            else:
                sweep_rate = 0.5 / 60.0
            if sweep_rate <= 0:
                sweep_rate = 1.05 / 60.0
        except Exception:
            init_b = 0.0
            sweep_rate = 0.5 / 60.0

        try:
            smu = getattr(self.station, "smu", None)
            init_v1 = smu.ch1.volt() if smu is not None and not smu.simulated else 0.0
            init_v2 = smu.ch2.volt() if smu is not None and not smu.simulated else 0.0
        except Exception:
            init_v1 = 0.0
            init_v2 = 0.0

        try:
            exposure_time = self.station.camera.exposure_time()
        except Exception:
            exposure_time = 0.1
            try:
                storage_path = os.path.join(parent_dir, "value_storage")
                if os.path.exists(storage_path):
                    with open(storage_path, "r") as f:
                        vs = json.load(f)
                    exposure_time = vs.get("shared", {}).get("exposure_time", vs.get("exposure_time", 0.1))
            except Exception:
                pass

        num_spectra = 1
        try:
            storage_path = os.path.join(parent_dir, "value_storage")
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    vs = json.load(f)
                single_data = vs.get("single_spectra", {})
                num_spectra = int(single_data.get("num_spectra", vs.get("num_spectra", 1)))
        except Exception:
            num_spectra = 1

        ramp_rate_v1 = 0.1
        ramp_rate_v2 = 0.1
        try:
            storage_path = os.path.join(parent_dir, "value_storage")
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    vs = json.load(f)
                dg_data = vs.get("dual_gate", {})
                ramp_rate_v1 = dg_data.get("ramprate1", vs.get("ramprate1", 0.1))
                ramp_rate_v2 = dg_data.get("ramprate2", vs.get("ramprate2", 0.1))
        except Exception:
            pass

        v_points = []
        b_points = []
        c1_name = getattr(self, 'c1_name', 'Vc1')
        c2_name = getattr(self, 'c2_name', 'Vc2')

        for xv in self.x_coords:
            for yv in self.y_coords:
                v1_target = init_v1
                v2_target = init_v2
                b_target = init_b

                # X parameter changes
                if params['x_param'] == "B":
                    b_target = xv
                elif params['x_param'] == "V1":
                    v1_target = xv
                elif params['x_param'] == "V2":
                    v2_target = xv

                # Y parameter changes
                if params['y_param'] == "B":
                    b_target = yv
                elif params['y_param'] == "V1":
                    v1_target = yv
                elif params['y_param'] == "V2":
                    v2_target = yv

                # Virtual Gates check
                if params['x_param'] in ("Vc1", "Vc2", c1_name, c2_name) or params['y_param'] in ("Vc1", "Vc2", c1_name, c2_name):
                    vc1 = 0.0
                    vc2 = 0.0
                    if params['x_param'] in ("Vc1", c1_name):
                        vc1 = xv
                    elif params['x_param'] in ("Vc2", c2_name):
                        vc2 = xv
                    if params['y_param'] in ("Vc1", c1_name):
                        vc1 = yv
                    elif params['y_param'] in ("Vc2", c2_name):
                        vc2 = yv
                    v1_target = a * vc1 + b * vc2
                    v2_target = c * vc1 + d * vc2

                v_points.append((v1_target, v2_target))
                b_points.append(b_target)

        self.sweep_total_duration = calculate_expected_duration(
            v_points=v_points,
            b_points=b_points,
            exposure_time=exposure_time,
            settling_time=params['settling_time'],
            num_spectra=num_spectra,
            sweep_rate_b=sweep_rate,
            initial_b=init_b,
            initial_v1=init_v1,
            initial_v2=init_v2,
            ramp_rate_v1=ramp_rate_v1,
            ramp_rate_v2=ramp_rate_v2
        )
        self.sweep_v_points = v_points
        self.sweep_b_points = b_points
        self.sweep_exposure_time = exposure_time
        self.sweep_settling_time = params['settling_time']
        self.sweep_num_spectra = num_spectra
        self.sweep_rate_b = sweep_rate
        self.sweep_ramp_rate_v1 = ramp_rate_v1
        self.sweep_ramp_rate_v2 = ramp_rate_v2
        self.sweep_initial_v1 = init_v1
        self.sweep_initial_v2 = init_v2
        self.sweep_initial_b = init_b
        self.sweep_start_time = time.time()
        self.measurement_started_at = new_measurement_start(self.sweep_start_time)
        self.measurement_device_modes = measurement_device_modes(self.station)
        self.total_paused_time = 0.0
        self.pause_start_time = None

        self.lbl_expected_end.setText("Expected End: (waiting for measurements)")

        params["num_spectra"] = self.data_raw_spectra.shape[2]
        self.sweep_worker = PLMapWorker(self.station, params)
        self.sweep_worker.point_measured.connect(self._on_point_measured)
        self.sweep_worker.progress_updated.connect(self._on_progress_updated)
        self.sweep_worker.finished.connect(self._on_sweep_finished)
        self.sweep_worker.error.connect(self._on_sweep_error)
        self.sweep_worker.start()

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
        self.reset_slice_levels = True
        if self.x_coords is None or self.y_coords is None or self.wavelengths is None:
            return
            
        x_min, x_max = min(self.x_coords), max(self.x_coords)
        y_min, y_max = min(self.y_coords), max(self.y_coords)
        
        # Convert wavelengths (nm) to energies (eV)
        energies = 1239.84193 / self.wavelengths
        e_min, e_max = min(energies), max(energies)

        x_param = self.x_param_cb.currentText()
        y_param = self.y_param_cb.currentText()

        # Update labels dynamically based on sweep variables
        self.map_plot.setLabel('bottom', x_param)
        self.map_plot.setLabel('left', y_param)
        self.wl_vs_y_plot.setLabel('left', y_param)
        self.wl_vs_y_plot.setLabel('bottom', 'Energy', units='eV')  # Ensure bottom is labeled Energy (eV)
        self.wl_vs_x_plot.setLabel('left', x_param)
        self.wl_vs_x_plot.setLabel('bottom', 'Energy', units='eV')    # Ensure bottom is labeled Energy (eV)

        # Panel A Map rect
        width_map = x_max - x_min if x_max != x_min else 1.0
        height_map = y_max - y_min if y_max != y_min else 1.0
        rect_map = QtCore.QRectF(x_min, y_min, width_map, height_map)
        if self.map_img.image is not None:
            self.map_img.setRect(rect_map)
        self.map_plot.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max)
        self.map_plot.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0)

        # Panel B Vertical Slice rect (Energy vs Y)
        width_vslice = e_max - e_min if e_max != e_min else 1.0
        height_vslice = y_max - y_min if y_max != y_min else 1.0
        rect_vslice = QtCore.QRectF(e_min, y_min, width_vslice, height_vslice)
        if self.wl_vs_y_img.image is not None:
            self.wl_vs_y_img.setRect(rect_vslice)
        self.wl_vs_y_plot.setLimits(xMin=e_min, xMax=e_max, yMin=y_min, yMax=y_max)
        self.wl_vs_y_plot.setRange(xRange=(e_min, e_max), yRange=(y_min, y_max), padding=0)

        # Panel C Horizontal Slice rect (Energy vs X)
        width_hslice = x_max - x_min if x_max != x_min else 1.0
        height_hslice = e_max - e_min if e_max != e_min else 1.0
        rect_hslice = QtCore.QRectF(e_min, x_min, height_hslice, width_hslice)
        if self.wl_vs_x_img.image is not None:
            self.wl_vs_x_img.setRect(rect_hslice)
        self.wl_vs_x_plot.setLimits(xMin=e_min, xMax=e_max, yMin=x_min, yMax=x_max)
        self.wl_vs_x_plot.setRange(xRange=(e_min, e_max), yRange=(x_min, x_max), padding=0)

        # Panel D limits
        self.spectrum_plot.setLimits(xMin=e_min, xMax=e_max)
        self.spectrum_plot.setRange(xRange=(e_min, e_max), padding=0)

        # Auto-center integration regions on the spectrum plot if bounds are reset
        if hasattr(self, 'integration_region'):
            reg_min, reg_max = self.integration_region.getRegion()
            if reg_min < e_min or reg_max > e_max or reg_min >= reg_max or (reg_min == 0.0 and reg_max == 1.0):
                e_center = (e_min + e_max) / 2.0
                self.integration_region.blockSignals(True)
                self.integration_region.setRegion((e_center - 0.02, e_center + 0.02))
                self.integration_region.blockSignals(False)

        if hasattr(self, 'integration_h_region'):
            h_reg_min, h_reg_max = self.integration_h_region.getRegion()
            if h_reg_min < 0 or h_reg_max <= h_reg_min or (h_reg_min == 0.0 and h_reg_max == 1.0):
                self.integration_h_region.blockSignals(True)
                self.integration_h_region.setRegion((200.0, 1500.0))
                self.integration_h_region.blockSignals(False)

    def pause_sweep(self):
        if not self.is_running:
            return
        if self.is_paused:
            # Already paused, resume it
            self.is_paused = False
            self.btn_pause.setText("⏸ Interrupt")
            self.btn_start.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.lbl_status.setText("Status: Sweeping...")
            if getattr(self, "pause_start_time", None) is not None:
                self.total_paused_time += time.time() - self.pause_start_time
                self.pause_start_time = None
            if self.sweep_worker:
                self.sweep_worker.resume()
        else:
            self.is_paused = True
            self.btn_pause.setText("▶ Resume")
            self.btn_start.setEnabled(True)
            self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
            self.lbl_status.setText("Status: Paused (Interrupt)")
            self.pause_start_time = time.time()
            if self.sweep_worker:
                self.sweep_worker.pause()

    def begin_sweep(self):
        """Alias for start_sweep to maintain compatibility with Dual_gate GUI naming."""
        if self.is_running and self.is_paused:
            self.is_paused = False
            self.btn_pause.setText("⏸ Interrupt")
            self.btn_pause.setEnabled(True)
            self.btn_start.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.lbl_status.setText("Status: Sweeping...")
            if getattr(self, "pause_start_time", None) is not None:
                self.total_paused_time += time.time() - self.pause_start_time
                self.pause_start_time = None
            if self.sweep_worker:
                self.sweep_worker.resume()
            return
        self.start_sweep()

    def on_interrupt_clicked(self):
        """Alias for pause_sweep to maintain compatibility with Dual_gate GUI naming."""
        self.pause_sweep()

    def resume_status_worker(self):
        if hasattr(self, 'live_status_worker') and self.live_status_worker:
            self.live_status_worker.resume()
        self.station.live_status_paused = False

    def reset_sweep(self):
        was_running = self.is_running
        if self.is_running:
            self.was_aborted = True
            self.trigger_autosave(was_interrupted=True)
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.lbl_status.setText("Status: Aborting...")
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(False)
            self.btn_reset.setEnabled(False)
            self.sweep_worker.stop()
            
            # Abort camera acquisition to wake up get_spectrum()
            try:
                if hasattr(self.station, 'camera') and hasattr(self.station.camera, 'remote_lf'):
                    self.station.camera.remote_lf.abort_acquisition()
            except Exception as e:
                print(f"PLMapWindow: Error calling abort_acquisition: {e}")
                
            return  # We wait for the finished/error signal to clean up
        
        self.is_running = False
        self.is_paused = False
        self.sweep_start_time = None
        self.sweep_total_duration = 0.0
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Interrupt")
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.set_inputs_enabled(True)
        
        self.progress_bar.setValue(0)
        self.lbl_expected_end.setText("Expected End: --:--:--")
        self.lbl_status.setText("Status: Idle")
        QtCore.QTimer.singleShot(500, self.resume_status_worker)

    def _on_point_measured(self, ix, iy, xv, yv, spectrum, meas_v1, meas_v2, curr_ch1, curr_ch2, meas_b, raw_frames):
        self.data_spectra[iy, ix, :] = spectrum
        if getattr(self, "data_raw_spectra", None) is not None:
            self.data_raw_spectra[iy, ix, :, :] = raw_frames
        self.data_measured_v1[iy, ix] = meas_v1
        self.data_measured_v2[iy, ix] = meas_v2
        self.data_measured_b[iy, ix] = meas_b
        self.data_current_ch1[iy, ix] = curr_ch1
        self.data_current_ch2[iy, ix] = curr_ch2
        self.data_measured_mask[iy, ix] = True
        
        # Update B, V1, V2 labels
        self.lbl_current_b.setText(f"B = {meas_b:.4f} T")
        self.lbl_target_vc1.setText(f"V1 = {meas_v1:.4f} V")
        self.lbl_target_vc2.setText(f"V2 = {meas_v2:.4f} V")
        
        # Calculate heatmap and update plots
        self.update_map_plot()
        
        # If this is the first point or matches crosshair, refresh slices
        if np.sum(self.data_measured_mask) == 1:
            self.update_crosshair_values(xv, yv)
            self.refresh_slice_plots(xv, yv)
        else:
            # Sync slices if the active crosshair point was just updated
            x_cross = self.cross_v.value()
            y_cross = self.cross_h.value()
            idx_x = np.argmin(np.abs(self.x_coords - x_cross))
            idx_y = np.argmin(np.abs(self.y_coords - y_cross))
            if idx_x == ix or idx_y == iy:
                self.refresh_slice_plots(x_cross, y_cross)
                
        # Do not perform continuous autoscale as the toggle button is removed

    def _on_progress_updated(self, current, total):
        self.progress_bar.setValue(int(100 * current / total))
        
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
                exposure_time=self.sweep_exposure_time,
                settling_time=self.sweep_settling_time,
                num_spectra=self.sweep_num_spectra,
                sweep_rate_b=self.sweep_rate_b,
                initial_b=last_b,
                initial_v1=last_v1,
                initial_v2=last_v2,
                ramp_rate_v1=self.sweep_ramp_rate_v1,
                ramp_rate_v2=self.sweep_ramp_rate_v2
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

    def _on_sweep_finished(self):
        if getattr(self, "was_aborted", False):
            self.is_running = False
            self.is_paused = False
            self.sweep_start_time = None
            self.sweep_total_duration = 0.0
            self.total_paused_time = 0.0
            self.pause_start_time = None
            self.btn_start.setEnabled(True)
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
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.lbl_expected_end.setText("Sweep finished")
        self.lbl_status.setText("Status: Finished")
        self.update_map_plot()
        self.set_inputs_enabled(True)
        QtCore.QTimer.singleShot(500, self.resume_status_worker)
        QtWidgets.QMessageBox.information(self, "Sweep Complete", "Vector map sweep finished successfully!")

    def _on_sweep_error(self, msg):
        if getattr(self, "was_aborted", False):
            self.is_running = False
            self.is_paused = False
            self.sweep_start_time = None
            self.sweep_total_duration = 0.0
            self.total_paused_time = 0.0
            self.pause_start_time = None
            self.btn_start.setEnabled(True)
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
            
        self.trigger_autosave(was_interrupted=True)
        self.is_running = False
        QtWidgets.QMessageBox.critical(self, "Sweep Error", msg)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.lbl_status.setText("Status: Error")
        self.set_inputs_enabled(True)
        QtCore.QTimer.singleShot(500, self.resume_status_worker)

    def prompt_save_on_interrupt(self):
        if not np.any(self.data_measured_mask):
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Save Interrupted Map",
            "The sweep was interrupted. Would you like to save the swept data?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.save_data()

    def update_map_plot(self):
        if self.data_spectra is None:
            return
            
        wl_min, wl_max = self.integration_region.getRegion()
        h_min, h_max = self.integration_h_region.getRegion()
        
        # Sort values in case linear regions are dragged in reverse order
        wl_min, wl_max = sorted([wl_min, wl_max])
        h_min, h_max = sorted([h_min, h_max])
        
        energies = 1239.84193 / self.wavelengths
        idx_range = np.where((energies >= wl_min) & (energies <= wl_max))[0]
        
        if len(idx_range) > 0:
            # Clamp/clip values within horizontal limits (h_min, h_max) and integrate
            sub_spectra = self.get_processed_spectra()[:, :, idx_range]
            clamped_spectra = np.clip(sub_spectra, h_min, h_max) - h_min
            map_data = np.sum(clamped_spectra, axis=2)
        else:
            map_data = np.zeros((len(self.y_coords), len(self.x_coords)))
            
        self.map_img.setImage(map_data, autoLevels=True)
        
        # Update LUT color scale limits
        if map_data.size > 0:
            min_v, max_v = np.min(map_data), np.max(map_data)
            if min_v == max_v:
                max_v += 1.0
            self.map_colorbar.setLevels(min_v, max_v)
            
        # Do not perform continuous autoscale as the toggle button is removed

    def update_crosshair_values(self, x_val, y_val):
        self.cross_v.blockSignals(True)
        self.cross_h.blockSignals(True)
        self.cross_v.setValue(x_val)
        self.cross_h.setValue(y_val)
        self.cross_v.blockSignals(False)
        self.cross_h.blockSignals(False)

    def on_crosshair_moved(self):
        self.refresh_slice_plots(self.cross_v.value(), self.cross_h.value())

    def refresh_slice_plots(self, x_val, y_val):
        if self.data_spectra is None or not np.any(self.data_measured_mask):
            return
            
        idx_x = np.argmin(np.abs(self.x_coords - x_val))
        idx_y = np.argmin(np.abs(self.y_coords - y_val))
        
        processed_spectra = self.get_processed_spectra()

        # 1. Update Panel D: Spectrum plot
        spectrum = processed_spectra[idx_y, idx_x, :]
        energies = 1239.84193 / self.wavelengths
        idx_sort = np.argsort(energies)
        self.spectrum_curve.setData(energies[idx_sort], spectrum[idx_sort])
        
        vb_spec = self.spectrum_plot.getViewBox()
        if vb_spec.state['autoRange'][1]:
            y_min, y_max = np.min(spectrum), np.max(spectrum)
            y_range = y_max - y_min if y_max != y_min else 1.0
            vb_spec.setRange(yRange=(y_min - 0.1 * y_range, y_max + 0.1 * y_range), padding=0)

        # 2. Update Panel B: Vertical Slice (Energy vs Y)
        vslice_data = processed_spectra[:, idx_x, :]  # shape: (y_pts, wl_pts)
        auto_levels = getattr(self, "reset_slice_levels", True)
        self.wl_vs_y_img.setImage(vslice_data[:, ::-1], autoLevels=auto_levels)
        if auto_levels and vslice_data.size > 0:
            min_v, max_v = np.min(vslice_data), np.max(vslice_data)
            if min_v == max_v:
                max_v += 1.0
            self.wl_vs_y_colorbar.setLevels(min_v, max_v)

        # 3. Update Panel C: Horizontal Slice (X vs Energy)
        hslice_data = processed_spectra[idx_y, :, ::-1]  # shape: (x_pts, wl_pts)
        self.wl_vs_x_img.setImage(hslice_data, autoLevels=auto_levels)
        if auto_levels and hslice_data.size > 0:
            min_h, max_h = np.min(hslice_data), np.max(hslice_data)
            if min_h == max_h:
                max_h += 1.0
            self.wl_vs_x_colorbar.setLevels(min_h, max_h)
            
        self.reset_slice_levels = False
            
        # Do not perform continuous autoscale as the toggle button is removed

    def save_data(self, file_path=None):
        if isinstance(file_path, bool):
            file_path = None
        if not np.any(self.data_measured_mask):
            if file_path is None:
                QtWidgets.QMessageBox.warning(self, "No Data", "There is no sweep data to save.")
            return

        try:
            # Ensure the full value_storage (including shared and all sections) is up‑to‑date
            self.save_storage()
            data_dir = measurement_data_dir(
                parent_dir, getattr(self, "measurement_started_at", None)
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"vector_map_{timestamp[-6:]}.h5"
            default_name = dated_measurement_filename(default_name, getattr(self, "measurement_started_at", None))
            if file_path is None:
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save PL Map",
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
                f.create_dataset("wavelengths", data=self.wavelengths)
                f.create_dataset("wavelength_axis", data=self.wavelengths)
                f.create_dataset("spectra", data=self.data_spectra)
                if getattr(self, "data_raw_spectra", None) is not None and self.data_raw_spectra.size > 0:
                    f.create_dataset("raw_intensities", data=self.data_raw_spectra)
                f.create_dataset("measured_v1", data=self.data_measured_v1)
                f.create_dataset("measured_v2", data=self.data_measured_v2)
                if self.data_measured_b is not None and self.data_measured_b.size > 0:
                    f.create_dataset("measured_b", data=self.data_measured_b)
                f.create_dataset("current_ch1", data=self.data_current_ch1)
                f.create_dataset("current_ch2", data=self.data_current_ch2)
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
                # Store the entire value_storage JSON as a single dataset
                dtype = h5py.string_dtype(encoding="utf-8")
                
                # Create metadata group/folder
                meta_grp = f.create_group("metadata")
                meta_grp.create_dataset("value_storage", data=json.dumps(self.value_storage), dtype=dtype)
                
                # Add exposure time, grating, and central wavelength as separate datasets
                vs_data = self.value_storage
                shared = vs_data.get("shared", {})
                single_spec = vs_data.get("single_spectra", {})
                exposure_time = shared.get("exposure_time", 0.0)
                grating = single_spec.get("grating", "")
                central_wavelength = single_spec.get("wavelength", 0.0)
                
                meta_grp.create_dataset("exposure_time", data=exposure_time)
                meta_grp.create_dataset("grating", data=grating, dtype=dtype)
                meta_grp.create_dataset("central_wavelength", data=central_wavelength)

            if file_path is None:
                QtWidgets.QMessageBox.information(self, "Saved", f"Data saved to {filename}")
            else:
                print(f"Auto-saved data to {filename}")
        except Exception as e:
            if file_path is None:
                QtWidgets.QMessageBox.critical(self, "Save error", str(e))
            else:
                print(f"Auto-save failed: {e}")

    def trigger_autosave(self, was_interrupted=False):
        if getattr(self, "_autosaved_this_run", False):
            return
        if not np.any(self.data_measured_mask):
            return
        autosave_dir = measurement_data_dir(
            parent_dir, getattr(self, "measurement_started_at", None), autosave=True
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_interrupted" if was_interrupted else ""
        filename = f"autosave_vector_map_{timestamp[-6:]}{suffix}.h5"
        filename = dated_measurement_filename(filename, getattr(self, "measurement_started_at", None))
        file_path = os.path.join(autosave_dir, filename)
        try:
            self.save_data(file_path=file_path)
            self._autosaved_this_run = True
        except Exception as e:
            print(f"Auto-save trigger failed: {e}")

    def apply_theme(self):
        from theme import SCIENTIFIC_THEME_QSS
        self.setStyleSheet(SCIENTIFIC_THEME_QSS)

        # The theme hides spin-box arrows (zero-width buttons). Also setting NoButtons
        # leaves a 1 px number field with PyQt6 >= 6.12, so it is not used.

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
            "x_start": -10.0,
            "x_end": 10.0,
            "x_pts": 21,
            "y_start": -10.0,
            "y_end": 10.0,
            "y_pts": 21,
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
                pl_data = raw_data.get("photoluminescence_map", {})
                shared_data = raw_data.get("shared", {})
                c1_name = shared_data.get("c1_name", "Vc1")
                c2_name = shared_data.get("c2_name", "Vc2")
                for key in defaults:
                    self.stored_settings[key] = pl_data.get(key, raw_data.get(key, defaults[key]))
            except Exception as e:
                print(f"Error loading value_storage: {e}")
        
        self.c1_name = c1_name
        self.c2_name = c2_name

        # Update items in parameter combos dynamically based on shared variables
        self.x_param_cb.blockSignals(True)
        self.y_param_cb.blockSignals(True)
        self.x_param_cb.clear()
        self.y_param_cb.clear()
        param_list = ["position X", "position Y", "V1", "V2", c1_name, c2_name, "B"]
        self.x_param_cb.addItems(param_list)
        self.y_param_cb.addItems(param_list)

        # Handle backward compatibility / rename fallback mapping
        x_val = self.stored_settings["x_param"]
        if x_val == "Vc1":
            x_val = c1_name
        elif x_val == "Vc2":
            x_val = c2_name

        y_val = self.stored_settings["y_param"]
        if y_val == "Vc1":
            y_val = c1_name
        elif y_val == "Vc2":
            y_val = c2_name

        self.x_param_cb.setCurrentText(x_val)
        self.y_param_cb.setCurrentText(y_val)
        self.x_param_cb.blockSignals(False)
        self.y_param_cb.blockSignals(False)

        # Populate UI from stored_settings
        self.x_start_spin.setValue(self.stored_settings["x_start"])
        self.x_end_spin.setValue(self.stored_settings["x_end"])
        self.x_pts_spin.setValue(self.stored_settings["x_pts"])
        self.y_start_spin.setValue(self.stored_settings["y_start"])
        self.y_end_spin.setValue(self.stored_settings["y_end"])
        self.y_pts_spin.setValue(self.stored_settings["y_pts"])
        self.pixel_x_spin.setValue(self.stored_settings["pixel_x"])
        self.pixel_y_spin.setValue(self.stored_settings["pixel_y"])
        
        # Keep value_storage dict populated, preserving existing keys like 'shared'
        if isinstance(self.value_storage, dict):
            if "photoluminescence_map" not in self.value_storage:
                self.value_storage["photoluminescence_map"] = {}
            for key, val in self.stored_settings.items():
                self.value_storage["photoluminescence_map"][key] = val
        else:
            self.value_storage = self.stored_settings.copy()

    def save_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        try:
            current_data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    current_data = json.load(f)
            
            if "photoluminescence_map" not in current_data:
                current_data["photoluminescence_map"] = {}
                
            current_data["photoluminescence_map"]["x_param"] = self.x_param_cb.currentText()
            current_data["photoluminescence_map"]["y_param"] = self.y_param_cb.currentText()
            current_data["photoluminescence_map"]["x_start"] = self.x_start_spin.value()
            current_data["photoluminescence_map"]["x_end"] = self.x_end_spin.value()
            current_data["photoluminescence_map"]["x_pts"] = self.x_pts_spin.value()
            current_data["photoluminescence_map"]["y_start"] = self.y_start_spin.value()
            current_data["photoluminescence_map"]["y_end"] = self.y_end_spin.value()
            current_data["photoluminescence_map"]["y_pts"] = self.y_pts_spin.value()
            current_data["photoluminescence_map"]["pixel_x"] = self.pixel_x_spin.value()
            current_data["photoluminescence_map"]["pixel_y"] = self.pixel_y_spin.value()
            
            self.value_storage = current_data
            with open(storage_path, "w") as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Error saving value_storage: {e}")

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

    def reload_gui_from_storage(self):
        self.load_storage()

    def set_inputs_enabled(self, enabled: bool):
        """Lock or unlock parameter and scan area configuration inputs."""
        self.x_param_cb.setEnabled(enabled)
        self.y_param_cb.setEnabled(enabled)
        self.scan_group.setEnabled(enabled)

    def centralWidget(self):
        return self

if __name__ == '__main__':
    from station_setup import initialize_station
    station = initialize_station(simulated=False)
    app = QtWidgets.QApplication(sys.argv)
    w = PhotoluminescenceMapWindow(station)
    w.show()
    sys.exit(app.exec())
