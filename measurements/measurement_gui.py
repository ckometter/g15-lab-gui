import os
import sys
import json
import time
import h5py
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Ensure parent directory is in path for importing station_setup
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from station_setup import initialize_station

# Setup pyqtgraph themes (Light Theme)
pg.setConfigOption('background', '#ffffff')
pg.setConfigOption('foreground', '#1e293b')

def transform_c_to_v(c1, c2, a, b, c, d):
    """
    Convert virtual gate coordinates (c1, c2) to physical voltages (v1, v2)
    using the linear mapping:
      c1 = a*v1 + b*v2
      c2 = c*v1 + d*v2
    """
    det = a * d - b * c
    if abs(det) < 1e-9:
        raise ValueError("Determinant of transformation matrix is zero. Coefficients are linearly dependent!")
    v1 = (d * c1 - b * c2) / det
    v2 = (-c * c1 + a * c2) / det
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

    def run(self):
        try:
            if self.param_name == "curr":
                self.channel.set_r.curr(self.target_val)
            else:
                self.channel.set_r.volt(self.target_val)
            try:
                # Release front-panel remote lock
                self.channel.parent.return_to_local()
            except Exception:
                pass
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class SweepWorker(QtCore.QThread):
    """
    Background worker thread that runs the gate sweep measurement.
    Communicates back to the main thread via Qt signals.
    """
    point_measured = QtCore.pyqtSignal(int, int, float, float, float, float, np.ndarray, float, float)
    progress_updated = QtCore.pyqtSignal(int, int)
    status_updated = QtCore.pyqtSignal(str)
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
        self.pause_cond.wakeAll()
        self.mutex.unlock()

    def run(self):
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

            # Set CCD exposure
            camera.exposure_time(exposure_time)

            # Generate sweep coordinate arrays
            c1_arr = np.linspace(c1_start, c1_end, c1_pts) if c1_pts > 1 else np.array([c1_start])
            c2_arr = np.linspace(c2_start, c2_end, c2_pts) if c2_pts > 1 else np.array([c2_start])

            if sweep_mode == "1D Sweep C1 (Hold C2)":
                c2_arr = np.array([c2_start])
            elif sweep_mode == "1D Sweep C2 (Hold C1)":
                c1_arr = np.array([c1_start])

            total_points = len(c1_arr) * len(c2_arr)
            point_count = 0

            # Determine outer and inner axes
            # By default: outer is C1, inner is C2 (or vice versa if 1D)
            if sweep_mode == "1D Sweep C2 (Hold C1)":
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
                    self.mutex.lock()
                    while self._is_paused:
                        self.status_updated.emit("Status: Paused")
                        self.pause_cond.wait(self.mutex)
                    if not self._is_running:
                        self.mutex.unlock()
                        return
                    self.mutex.unlock()

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

                    # 2. Ramp physical voltages (blocking hardware set call)
                    self.status_updated.emit(f"Status: GOTO V1={v1:.4f}V, V2={v2:.4f}V")
                    smu.ch1.set_r.volt(v1)
                    smu.ch2.set_r.volt(v2)

                    # 3. Wait settling time
                    if settling_time > 0:
                        time.sleep(settling_time)

                    # 4. Measure
                    self.status_updated.emit("Status: Acquiring CCD spectrum...")
                    img = camera.image()
                    spectrum = np.sum(img, axis=0) # Sum vertically to get 1D spectrum counts

                    # Query hardware output states before measuring current
                    # to prevent automated query-based power-on
                    try:
                        ch1_on = bool(smu.ch1.output())
                        curr_ch1 = smu.ch1.measure_curr() if ch1_on else 0.0
                    except Exception:
                        curr_ch1 = 0.0

                    try:
                        ch2_on = bool(smu.ch2.output())
                        curr_ch2 = smu.ch2.measure_curr() if ch2_on else 0.0
                    except Exception:
                        curr_ch2 = 0.0

                    point_count += 1
                    self.point_measured.emit(i, j, c1, c2, v1, v2, spectrum, curr_ch1, curr_ch2)
                    self.progress_updated.emit(point_count, total_points)

            try:
                smu.return_to_local()
            except Exception:
                pass
            self.finished.emit()

        except Exception as e:
            try:
                smu.return_to_local()
            except Exception:
                pass
            self.error_occurred.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, station):
        super().__init__()
        self.station = station
        self.setWindowTitle("Electrostatic Gate Control & 2D Sweep Spectrometer")
        self.setMinimumSize(1280, 850)

        # Application State
        self.is_running = False
        self.is_paused = False
        self.sweep_worker = None
        self.ramp_worker = None

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

        # Energy calibration axis
        self.energy_axis = np.linspace(1.5, 3.0, 512) # Dummy range initialized at startup
        self.energy_sort_idx = np.arange(512)

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

    # --- Value Storage Persistence ---
    def load_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        defaults = {
            "exposure_time": 0.1,
            "settling_time": 0.05,
            "ramprate1": 0.1,
            "ramprate2": 0.1,
            "ramp_step": 0.0005
        }
        self.stored_settings = defaults
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    self.stored_settings = json.load(f)
                    # Migrate old ramp_rate keys if present
                    if "ramp_rate" in self.stored_settings:
                        old_rate = self.stored_settings["ramp_rate"]
                        self.stored_settings.setdefault("ramprate1", old_rate)
                        self.stored_settings.setdefault("ramprate2", old_rate)
                    if "ramp_rate_1" in self.stored_settings:
                        self.stored_settings.setdefault("ramprate1", self.stored_settings["ramp_rate_1"])
                    if "ramp_rate_2" in self.stored_settings:
                        self.stored_settings.setdefault("ramprate2", self.stored_settings["ramp_rate_2"])
            except Exception as e:
                print(f"Error loading value_storage: {e}")

    def save_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        try:
            with open(storage_path, "w") as f:
                json.dump(self.stored_settings, f, indent=4)
        except Exception as e:
            print(f"Error saving to value_storage: {e}")

    # --- UI Building ---
    def setup_ui(self):
        # Initialize Status Bar and Connection Status Indicator
        status_bar = self.statusBar()
        self.conn_status_lbl = QtWidgets.QLabel()
        if self.station.smu.simulated:
            self.conn_status_lbl.setText("Keysight SMU: SIMULATED (Connection Locked/Failed)  ")
            self.conn_status_lbl.setStyleSheet("color: #f9e2af; font-weight: bold; font-family: Consolas, monospace;")
        else:
            self.conn_status_lbl.setText("Keysight SMU: CONNECTED (Physical)  ")
            self.conn_status_lbl.setStyleSheet("color: #a6e3a1; font-weight: bold; font-family: Consolas, monospace;")
        status_bar.addPermanentWidget(self.conn_status_lbl)

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
        right_panel = QtWidgets.QTabWidget()
        main_layout.addWidget(right_panel, 1)

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
        self.ch1_setpoint.valueChanged.connect(self.on_ch1_setpoint_enter)
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
        self.ch1_ramp_rate.setRange(0.001, 100.0)
        self.ch1_ramp_rate.setValue(self.stored_settings.get("ramprate1", 0.1))
        self.ch1_ramp_rate.valueChanged.connect(self.on_ch1_ramping_params_changed)
        grid1.addWidget(self.ch1_ramp_rate, 2, 1)

        # 5. Compliance
        self.ch1_compliance_lbl = QtWidgets.QLabel("Compliance (A):")
        grid1.addWidget(self.ch1_compliance_lbl, 3, 0)
        self.ch1_compliance = QtWidgets.QDoubleSpinBox()
        self.ch1_compliance.setRange(1e-9, 3.0)
        self.ch1_compliance.setDecimals(6)
        self.ch1_compliance.setValue(0.01)
        self.ch1_compliance.setSingleStep(0.001)
        self.ch1_compliance.valueChanged.connect(self.on_compliance_changed)
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
        self.ch2_setpoint.valueChanged.connect(self.on_ch2_setpoint_enter)
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
        self.ch2_ramp_rate.setRange(0.001, 100.0)
        self.ch2_ramp_rate.setValue(self.stored_settings.get("ramprate2", 0.1))
        self.ch2_ramp_rate.valueChanged.connect(self.on_ch2_ramping_params_changed)
        grid2.addWidget(self.ch2_ramp_rate, 2, 1)

        # 5. Compliance
        self.ch2_compliance_lbl = QtWidgets.QLabel("Compliance (A):")
        grid2.addWidget(self.ch2_compliance_lbl, 3, 0)
        self.ch2_compliance = QtWidgets.QDoubleSpinBox()
        self.ch2_compliance.setRange(1e-9, 3.0)
        self.ch2_compliance.setDecimals(6)
        self.ch2_compliance.setValue(0.01)
        self.ch2_compliance.setSingleStep(0.001)
        self.ch2_compliance.valueChanged.connect(self.on_compliance_changed)
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
        ctrl_c1_box = QtWidgets.QGroupBox("[ Control Variable 1 ]")
        ctrl_c1_layout = QtWidgets.QVBoxLayout(ctrl_c1_box)
        ctrl_c1_layout.addWidget(QtWidgets.QLabel("Virtual Gate C1 (V):"))
        
        c1_grid = QtWidgets.QGridLayout()
        c1_grid.setSpacing(6)
        c1_grid.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
        self.c1_start = QtWidgets.QDoubleSpinBox()
        self.c1_start.setRange(-210.0, 210.0)
        self.c1_start.setValue(-0.005)
        self.c1_start.setDecimals(4)
        c1_grid.addWidget(self.c1_start, 0, 1)

        c1_grid.addWidget(QtWidgets.QLabel("End:"), 0, 2)
        self.c1_end = QtWidgets.QDoubleSpinBox()
        self.c1_end.setRange(-210.0, 210.0)
        self.c1_end.setValue(0.005)
        self.c1_end.setDecimals(4)
        c1_grid.addWidget(self.c1_end, 0, 3)

        c1_grid.addWidget(QtWidgets.QLabel("Points:"), 1, 0)
        self.c1_pts = QtWidgets.QSpinBox()
        self.c1_pts.setRange(1, 1000)
        self.c1_pts.setValue(21)
        c1_grid.addWidget(self.c1_pts, 1, 1)

        c1_grid.addWidget(QtWidgets.QLabel("Step:"), 1, 2)
        self.c1_step_lbl = QtWidgets.QLabel("0.0 V")
        c1_grid.addWidget(self.c1_step_lbl, 1, 3)
        
        ctrl_c1_layout.addLayout(c1_grid)
        ctrl_row.addWidget(ctrl_c1_box)

        # Control 2
        ctrl_c2_box = QtWidgets.QGroupBox("[ Control Variable 2 ]")
        ctrl_c2_layout = QtWidgets.QVBoxLayout(ctrl_c2_box)
        ctrl_c2_layout.addWidget(QtWidgets.QLabel("Virtual Gate C2 (V):"))
        
        c2_grid = QtWidgets.QGridLayout()
        c2_grid.setSpacing(6)
        c2_grid.addWidget(QtWidgets.QLabel("Start:"), 0, 0)
        self.c2_start = QtWidgets.QDoubleSpinBox()
        self.c2_start.setRange(-210.0, 210.0)
        self.c2_start.setValue(-0.005)
        self.c2_start.setDecimals(4)
        c2_grid.addWidget(self.c2_start, 0, 1)

        c2_grid.addWidget(QtWidgets.QLabel("End:"), 0, 2)
        self.c2_end = QtWidgets.QDoubleSpinBox()
        self.c2_end.setRange(-210.0, 210.0)
        self.c2_end.setValue(0.005)
        self.c2_end.setDecimals(4)
        c2_grid.addWidget(self.c2_end, 0, 3)

        c2_grid.addWidget(QtWidgets.QLabel("Points:"), 1, 0)
        self.c2_pts = QtWidgets.QSpinBox()
        self.c2_pts.setRange(1, 1000)
        self.c2_pts.setValue(21)
        c2_grid.addWidget(self.c2_pts, 1, 1)

        c2_grid.addWidget(QtWidgets.QLabel("Step:"), 1, 2)
        self.c2_step_lbl = QtWidgets.QLabel("0.0 V")
        c2_grid.addWidget(self.c2_step_lbl, 1, 3)

        ctrl_c2_layout.addLayout(c2_grid)
        ctrl_row.addWidget(ctrl_c2_box)
        
        left_layout.addLayout(ctrl_row)

        # Bind events to update step sizes dynamically
        for box in (self.c1_start, self.c1_end, self.c1_pts):
            box.valueChanged.connect(self.update_step_calculations)
        for box in (self.c2_start, self.c2_end, self.c2_pts):
            box.valueChanged.connect(self.update_step_calculations)
        self.update_step_calculations()

        # --- ROW 3: Gate Variables coefficients (full width) ---
        gate_box = QtWidgets.QGroupBox("[ Gate Variables ]")
        gate_layout = QtWidgets.QGridLayout(gate_box)
        gate_layout.setSpacing(10)

        gate_layout.addWidget(QtWidgets.QLabel("c1 = v1 * a + v2 * b"), 0, 0, 1, 2)
        gate_layout.addWidget(QtWidgets.QLabel("a:"), 1, 0)
        self.coeff_a = QtWidgets.QDoubleSpinBox()
        self.coeff_a.setValue(1.0)
        self.coeff_a.setRange(-100, 100)
        gate_layout.addWidget(self.coeff_a, 1, 1)
        
        gate_layout.addWidget(QtWidgets.QLabel("b:"), 1, 2)
        self.coeff_b = QtWidgets.QDoubleSpinBox()
        self.coeff_b.setValue(0.0)
        self.coeff_b.setRange(-100, 100)
        gate_layout.addWidget(self.coeff_b, 1, 3)

        gate_layout.addWidget(QtWidgets.QLabel("c2 = v1 * c + v2 * d"), 2, 0, 1, 2)
        gate_layout.addWidget(QtWidgets.QLabel("c:"), 3, 0)
        self.coeff_c = QtWidgets.QDoubleSpinBox()
        self.coeff_c.setValue(0.0)
        self.coeff_c.setRange(-100, 100)
        gate_layout.addWidget(self.coeff_c, 3, 1)

        gate_layout.addWidget(QtWidgets.QLabel("d:"), 3, 2)
        self.coeff_d = QtWidgets.QDoubleSpinBox()
        self.coeff_d.setValue(1.0)
        self.coeff_d.setRange(-100, 100)
        gate_layout.addWidget(self.coeff_d, 3, 3)

        left_layout.addWidget(gate_box)

        # --- ROW 4: Other Settings & Measurement Control next to each other ---
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(10)

        # Other Settings Box
        other_box = QtWidgets.QGroupBox("[ Other ]")
        other_layout = QtWidgets.QVBoxLayout(other_box)
        other_layout.setSpacing(6)
        
        other_layout.addWidget(QtWidgets.QLabel("Sweep Configuration:"))
        self.sweep_mode_combo = QtWidgets.QComboBox()
        self.sweep_mode_combo.addItems(["2D Sweep (C1 & C2)", "1D Sweep C1 (Hold C2)", "1D Sweep C2 (Hold C1)"])
        other_layout.addWidget(self.sweep_mode_combo)

        self.serpentine_box = QtWidgets.QCheckBox("Serpentine (Snake) Sweep")
        self.serpentine_box.setChecked(True)
        other_layout.addWidget(self.serpentine_box)

        # Timing and Ramping Grid
        timing_ramp_grid = QtWidgets.QGridLayout()
        timing_ramp_grid.setSpacing(6)
        timing_ramp_grid.addWidget(QtWidgets.QLabel("Exposure (s):"), 0, 0)
        self.exposure_box = QtWidgets.QDoubleSpinBox()
        self.exposure_box.setRange(0.001, 10.0)
        self.exposure_box.setDecimals(3)
        self.exposure_box.setValue(self.stored_settings.get("exposure_time", 0.1))
        self.exposure_box.valueChanged.connect(self.on_timing_params_changed)
        timing_ramp_grid.addWidget(self.exposure_box, 0, 1)

        timing_ramp_grid.addWidget(QtWidgets.QLabel("Settling (s):"), 1, 0)
        self.settling_box = QtWidgets.QDoubleSpinBox()
        self.settling_box.setRange(0.0, 10.0)
        self.settling_box.setDecimals(3)
        self.settling_box.setValue(self.stored_settings.get("settling_time", 0.05))
        self.settling_box.valueChanged.connect(self.on_timing_params_changed)
        timing_ramp_grid.addWidget(self.settling_box, 1, 1)

        timing_ramp_grid.addWidget(QtWidgets.QLabel("Ramp Step (V):"), 2, 0)
        self.ramp_step_box = QtWidgets.QDoubleSpinBox()
        self.ramp_step_box.setRange(0.0001, 1.0)
        self.ramp_step_box.setDecimals(5)
        self.ramp_step_box.setValue(self.stored_settings.get("ramp_step", 0.0005))
        self.ramp_step_box.valueChanged.connect(self.on_ramp_step_changed)
        timing_ramp_grid.addWidget(self.ramp_step_box, 2, 1)
        
        other_layout.addLayout(timing_ramp_grid)
        bottom_row.addWidget(other_box)

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
        self.pause_btn.clicked.connect(self.pause_sweep)
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

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        meas_layout.addWidget(self.progress_bar)

        bottom_row.addWidget(meas_box)
        left_layout.addLayout(bottom_row)

        # --- RIGHT TAB PANEL SECTIONS ---

        # Tab 1: Gate Sweep (CCD Spectra)
        self.tab1 = QtWidgets.QWidget()
        right_panel.addTab(self.tab1, "Gate Sweep (Spectra)")
        self.setup_tab1()

        # Tab 2: Top Gate current
        self.tab2 = QtWidgets.QWidget()
        right_panel.addTab(self.tab2, "Top Gate (Ch1 Current)")
        self.setup_tab2()

        # Tab 3: Bottom Gate current
        self.tab3 = QtWidgets.QWidget()
        right_panel.addTab(self.tab3, "Bottom Gate (Ch2 Current)")
        self.setup_tab3()

    def setup_tab1(self):
        """Setup 2x2 grid of plots for Gating vs Energy Spectra in Tab 1"""
        layout = QtWidgets.QGridLayout(self.tab1)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # 1. Gate Map Plot (Top Left)
        self.gatemap_win = pg.PlotWidget(title="2D Gate Map (Integrated Counts)")
        self.gatemap_img = pg.ImageItem()
        self.gatemap_img.setOpts(axisOrder='row-major')
        self.gatemap_win.addItem(self.gatemap_img)
        self.gatemap_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.gatemap_win.setLabel('bottom', 'Virtual Gate C1', units='V')
        
        # Color bar
        self.gatemap_colorbar = pg.ColorBarItem(values=(0, 65535), colorMap='viridis')
        self.gatemap_colorbar.setImageItem(self.gatemap_img)
        
        # Crosshair lines
        self.cross1_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.cross1_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#4f46e5', width=2))
        self.gatemap_win.addItem(self.cross1_v)
        self.gatemap_win.addItem(self.cross1_h)
        self.cross1_v.sigPositionChanged.connect(self.on_crosshair_moved)
        self.cross1_h.sigPositionChanged.connect(self.on_crosshair_moved)

        layout.addWidget(self.gatemap_win, 0, 0)

        # 2. Vertical Slice (Top Right)
        self.vslice_win = pg.PlotWidget(title="Vertical Slice spectra")
        self.vslice_img = pg.ImageItem()
        self.vslice_img.setOpts(axisOrder='row-major')
        self.vslice_win.addItem(self.vslice_img)
        self.vslice_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.vslice_win.setLabel('bottom', 'Energy', units='eV')
        
        self.vslice_colorbar = pg.ColorBarItem(values=(0, 65535), colorMap='viridis')
        self.vslice_colorbar.setImageItem(self.vslice_img)

        # Draggable vertical line showing slice marker
        self.vslice_marker = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#dc2626', width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.vslice_win.addItem(self.vslice_marker)

        layout.addWidget(self.vslice_win, 0, 1)

        # 3. Horizontal Slice (Bottom Left)
        self.hslice_win = pg.PlotWidget(title="Horizontal Slice spectra")
        self.hslice_img = pg.ImageItem()
        self.hslice_img.setOpts(axisOrder='row-major')
        self.hslice_win.addItem(self.hslice_img)
        self.hslice_win.setLabel('left', 'Energy', units='eV')
        self.hslice_win.setLabel('bottom', 'Virtual Gate C1', units='V')

        self.hslice_colorbar = pg.ColorBarItem(values=(0, 65535), colorMap='viridis')
        self.hslice_colorbar.setImageItem(self.hslice_img)

        self.hslice_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#dc2626', width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.hslice_win.addItem(self.hslice_marker)

        layout.addWidget(self.hslice_win, 1, 0)

        # 4. Spectrum at Cross (Bottom Right)
        self.spectrum_win = pg.PlotWidget(title="CCD Spectrum at Gating Crosshair")
        self.spectrum_curve = self.spectrum_win.plot(pen=pg.mkPen('#7c3aed', width=2))
        self.spectrum_win.setLabel('left', 'Counts')
        self.spectrum_win.setLabel('bottom', 'Energy', units='eV')
        self.spectrum_win.showGrid(x=True, y=True, alpha=0.3)

        # Draggable region to define integration window for Gate Map
        self.int_region = pg.LinearRegionItem(values=(2.4, 2.6), brush=pg.mkBrush(16, 185, 129, 40), pen=pg.mkPen('#10b981', width=1.5))
        self.spectrum_win.addItem(self.int_region)
        self.int_region.sigRegionChanged.connect(self.on_integration_region_changed)

        layout.addWidget(self.spectrum_win, 1, 1)

    def setup_tab2(self):
        """Setup layout for Ch1 Current vs physical Top Gate and virtual C2"""
        layout = QtWidgets.QHBoxLayout(self.tab2)
        layout.setContentsMargins(5, 5, 5, 5)

        # Left: Heatmap
        self.t2_heatmap_win = pg.PlotWidget(title="Top Gate Current Heatmap")
        self.t2_heatmap_img = pg.ImageItem()
        self.t2_heatmap_img.setOpts(axisOrder='row-major')
        self.t2_heatmap_win.addItem(self.t2_heatmap_img)
        self.t2_heatmap_win.setLabel('left', 'Virtual Gate C2', units='V')
        self.t2_heatmap_win.setLabel('bottom', 'Physical Vtop (Ch1)', units='V')

        self.t2_colorbar = pg.ColorBarItem(colorMap='viridis')
        self.t2_colorbar.setImageItem(self.t2_heatmap_img)

        # Draggable crosshair
        self.cross2_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#ea580c', width=2))
        self.cross2_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#ea580c', width=2))
        self.t2_heatmap_win.addItem(self.cross2_v)
        self.t2_heatmap_win.addItem(self.cross2_h)
        self.cross2_v.sigPositionChanged.connect(self.on_tab2_crosshair_moved)
        self.cross2_h.sigPositionChanged.connect(self.on_tab2_crosshair_moved)

        layout.addWidget(self.t2_heatmap_win, 3)

        # Right: Slice plots (vertical layout)
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(6)

        self.t2_hslice_win = pg.PlotWidget(title="Slice at fixed C2")
        self.t2_hslice_curve = self.t2_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t2_hslice_win.setLabel('left', 'Current', units='A')
        self.t2_hslice_win.setLabel('bottom', 'Vtop (Ch1)', units='V')
        self.t2_hslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t2_hslice_win)

        self.t2_vslice_win = pg.PlotWidget(title="Slice at fixed Vtop")
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
        self.t3_heatmap_win = pg.PlotWidget(title="Bottom Gate Current Heatmap")
        self.t3_heatmap_img = pg.ImageItem()
        self.t3_heatmap_img.setOpts(axisOrder='row-major')
        self.t3_heatmap_win.addItem(self.t3_heatmap_img)
        self.t3_heatmap_win.setLabel('left', 'Virtual Gate C1', units='V')
        self.t3_heatmap_win.setLabel('bottom', 'Physical Vbottom (Ch2)', units='V')

        self.t3_colorbar = pg.ColorBarItem(colorMap='viridis')
        self.t3_colorbar.setImageItem(self.t3_heatmap_img)

        # Draggable crosshair
        self.cross3_v = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#16a34a', width=2))
        self.cross3_h = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#16a34a', width=2))
        self.t3_heatmap_win.addItem(self.cross3_v)
        self.t3_heatmap_win.addItem(self.cross3_h)
        self.cross3_v.sigPositionChanged.connect(self.on_tab3_crosshair_moved)
        self.cross3_h.sigPositionChanged.connect(self.on_tab3_crosshair_moved)

        layout.addWidget(self.t3_heatmap_win, 3)

        # Right: Slice plots (vertical layout)
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(6)

        self.t3_hslice_win = pg.PlotWidget(title="Slice at fixed C1")
        self.t3_hslice_curve = self.t3_hslice_win.plot(pen=pg.mkPen('#2563eb', width=2))
        self.t3_hslice_win.setLabel('left', 'Current', units='A')
        self.t3_hslice_win.setLabel('bottom', 'Vbottom (Ch2)', units='V')
        self.t3_hslice_win.showGrid(x=True, y=True, alpha=0.3)
        right_layout.addWidget(self.t3_hslice_win)

        self.t3_vslice_win = pg.PlotWidget(title="Slice at fixed Vbottom")
        self.t3_vslice_curve = self.t3_vslice_win.plot(pen=pg.mkPen('#d97706', width=2))
        self.t3_vslice_win.setLabel('left', 'Current', units='A')
        self.t3_vslice_win.setLabel('bottom', 'Virtual Gate C1', units='V')
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
        unit = "A" if param_name == "curr" else "V"
        self.status_lbl.setText(f"Status: Ramping {ch_name} to {target} {unit}...")
        
        self.ramp_worker = SingleRampWorker(channel, target, param_name)
        self.ramp_worker.finished.connect(self.on_ramp_finished)
        self.ramp_worker.error.connect(self.on_ramp_error)
        self.ramp_worker.start()

    def on_ramp_finished(self):
        self.status_lbl.setText("Status: Ramping complete.")

    def on_ramp_error(self, err_msg):
        self.status_lbl.setText(f"Status: Ramping error")
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

    def on_ramp_step_changed(self):
        """Save shared ramp step to value_storage and sync with hardware driver channels."""
        step = self.ramp_step_box.value()
        self.stored_settings["ramp_step"] = step
        self.save_storage()
        try:
            self.station.smu.ch1.ramp_step(step)
            self.station.smu.ch2.ramp_step(step)
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
        self.ramp_step_box.setEnabled(enabled)
        self.sweep_mode_combo.setEnabled(enabled)
        self.serpentine_box.setEnabled(enabled)

    # --- Sweep Measurement Control ---
    def begin_sweep(self):
        if self.is_running:
            return

        if self.is_paused:
            # Resume sweep
            self.is_paused = False
            self.pause_btn.setText("⏸ Interrupt")
            self.pause_btn.setEnabled(True)
            self.begin_btn.setEnabled(False)
            self.status_lbl.setText("Status: Sweeping...")
            self.sweep_worker.resume()
            return

        # Prepare parameters and validate
        a = self.coeff_a.value()
        b = self.coeff_b.value()
        c = self.coeff_c.value()
        d = self.coeff_d.value()
        
        det = a * d - b * c
        if abs(det) < 1e-9:
            QtWidgets.QMessageBox.critical(self, "Invalid Coefficients", "The transformation matrix is singular (determinant = 0). Choose linearly independent coefficients.")
            return

        # Build coordinate arrays
        c1_start = self.c1_start.value()
        c1_end = self.c1_end.value()
        c1_pts = self.c1_pts.value()

        c2_start = self.c2_start.value()
        c2_end = self.c2_end.value()
        c2_pts = self.c2_pts.value()

        sweep_mode = self.sweep_mode_combo.currentText()

        # Build ranges for validation and storage allocation
        if sweep_mode == "1D Sweep C1 (Hold C2)":
            c2_pts_actual = 1
            c1_pts_actual = c1_pts
        elif sweep_mode == "1D Sweep C2 (Hold C1)":
            c1_pts_actual = 1
            c2_pts_actual = c2_pts
        else:
            c1_pts_actual = c1_pts
            c2_pts_actual = c2_pts

        self.c1_coords = np.linspace(c1_start, c1_end, c1_pts_actual) if c1_pts_actual > 1 else np.array([c1_start])
        self.c2_coords = np.linspace(c2_start, c2_end, c2_pts_actual) if c2_pts_actual > 1 else np.array([c2_start])

        # Allocate data containers
        self.data_spectra = np.zeros((c2_pts_actual, c1_pts_actual, 512))
        self.data_current_ch1 = np.zeros((c2_pts_actual, c1_pts_actual))
        self.data_current_ch2 = np.zeros((c2_pts_actual, c1_pts_actual))
        self.data_measured_mask = np.zeros((c2_pts_actual, c1_pts_actual), dtype=bool)

        # Precalculate grid coordinates for Tab 2/3 Heatmaps
        self.v1_grid = np.zeros((c2_pts_actual, c1_pts_actual))
        self.v2_grid = np.zeros((c2_pts_actual, c1_pts_actual))
        for j, c2 in enumerate(self.c2_coords):
            for i, c1 in enumerate(self.c1_coords):
                v1, v2 = transform_c_to_v(c1, c2, a, b, c, d)
                self.v1_grid[j, i] = v1
                self.v2_grid[j, i] = v2

        # Initialize heatmap images with zero arrays of the correct size
        # so pyqtgraph can resolve their size when setRect is called
        self.gatemap_img.setImage(np.zeros((c2_pts_actual, c1_pts_actual)))
        self.vslice_img.setImage(np.zeros((c2_pts_actual, 512)))
        self.hslice_img.setImage(np.zeros((512, c1_pts_actual)))
        self.t2_heatmap_img.setImage(np.zeros((c2_pts_actual, c1_pts_actual)))
        self.t3_heatmap_img.setImage(np.zeros((c1_pts_actual, c2_pts_actual)))

        # Dynamically retrieve monochromator wavelength to calibrate Energy axis
        try:
            mono_wl = self.station.monochromator.wavelength()
        except Exception:
            mono_wl = 500.0 # fallback

        # Calculate energy axis and sorting index
        pixels = np.arange(512)
        wavelengths = mono_wl + (pixels - 256) * 0.05
        # Convert nm to eV
        energy_raw = 1239.84193 / wavelengths
        self.energy_sort_idx = np.argsort(energy_raw)
        self.energy_axis = energy_raw[self.energy_sort_idx]

        # Reset plots and set axes ranges
        self.reset_plot_scaling()

        # Start sweep thread
        params = {
            'a': a, 'b': b, 'c': c, 'd': d,
            'c1_start': c1_start, 'c1_end': c1_end, 'c1_pts': c1_pts,
            'c2_start': c2_start, 'c2_end': c2_end, 'c2_pts': c2_pts,
            'sweep_mode': sweep_mode,
            'serpentine': self.serpentine_box.isChecked(),
            'exposure_time': self.exposure_box.value(),
            'settling_time': self.settling_box.value()
        }

        self.is_running = True
        self.is_paused = False
        
        self.begin_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.set_inputs_enabled(False)

        self.sweep_worker = SweepWorker(self.station, params)
        self.sweep_worker.point_measured.connect(self.on_point_measured)
        self.sweep_worker.progress_updated.connect(self.on_progress_updated)
        self.sweep_worker.status_updated.connect(self.status_lbl.setText)
        self.sweep_worker.finished.connect(self.on_sweep_finished)
        self.sweep_worker.error_occurred.connect(self.on_sweep_error)
        self.sweep_worker.start()

    def pause_sweep(self):
        if not self.is_running:
            return
        if self.is_paused:
            # Already paused, resume it
            self.is_paused = False
            self.pause_btn.setText("⏸ Interrupt")
            self.begin_btn.setEnabled(False)
            self.status_lbl.setText("Status: Sweeping...")
            self.sweep_worker.resume()
        else:
            self.is_paused = True
            self.pause_btn.setText("▶ Resume")
            self.begin_btn.setEnabled(True)
            self.status_lbl.setText("Status: Paused (Interrupt)")
            self.sweep_worker.pause()

    def reset_sweep(self):
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        
        self.is_running = False
        self.is_paused = False
        self.begin_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.save_btn.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.set_inputs_enabled(True)
        
        self.progress_bar.setValue(0)
        self.status_lbl.setText("Status: Reset complete")

    def on_point_measured(self, i, j, c1, c2, v1, v2, spectrum, curr_ch1, curr_ch2):
        # Sort spectrum to align with increasing energy values
        sorted_spectrum = spectrum[self.energy_sort_idx]
        
        # Save measurement point
        self.data_spectra[j, i, :] = sorted_spectrum
        self.data_current_ch1[j, i] = curr_ch1
        self.data_current_ch2[j, i] = curr_ch2
        self.data_measured_mask[j, i] = True

        # Redraw plots
        self.update_heatmaps_and_slices()

    def on_progress_updated(self, current, total):
        pct = int(100 * current / total)
        self.progress_bar.setValue(pct)

    def on_sweep_finished(self):
        self.is_running = False
        self.begin_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.set_inputs_enabled(True)
        self.status_lbl.setText("Status: Sweep Finished (Data ready to save)")
        QtWidgets.QMessageBox.information(self, "Sweep Complete", "Electrostatic gate sweep finished successfully!")

    def on_sweep_error(self, err_msg):
        self.is_running = False
        self.begin_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.save_btn.setEnabled(True if np.any(self.data_measured_mask) else False)
        self.set_inputs_enabled(True)
        self.status_lbl.setText("Status: Error occurred")
        QtWidgets.QMessageBox.critical(self, "Sweep Error", f"Measurement aborted due to error:\n{err_msg}")

    # --- Plot Sizing & Coordinates Setup ---
    def reset_plot_scaling(self):
        """Set initial geometric sizes and labels for heatmap plots."""
        # Tab 1: Gate Map (integrated spectra)
        dx = (self.c1_coords[-1] - self.c1_coords[0]) / (len(self.c1_coords) - 1) if len(self.c1_coords) > 1 else 1.0
        dy = (self.c2_coords[-1] - self.c2_coords[0]) / (len(self.c2_coords) - 1) if len(self.c2_coords) > 1 else 1.0
        
        # Set bounds for Gate Map
        rect1 = QtCore.QRectF(self.c1_coords[0] - dx/2, self.c2_coords[0] - dy/2, 
                              self.c1_coords[-1] - self.c1_coords[0] + dx, 
                              self.c2_coords[-1] - self.c2_coords[0] + dy)
        self.gatemap_img.setRect(rect1)

        # Tab 1: Vertical Slice (Y axis is C2, X axis is Energy)
        dE = self.energy_axis[1] - self.energy_axis[0]
        rect_vslice = QtCore.QRectF(self.energy_axis[0] - dE/2, self.c2_coords[0] - dy/2,
                                    self.energy_axis[-1] - self.energy_axis[0] + dE,
                                    self.c2_coords[-1] - self.c2_coords[0] + dy)
        self.vslice_img.setRect(rect_vslice)

        # Tab 1: Horizontal Slice (X axis is C1, Y axis is Energy)
        rect_hslice = QtCore.QRectF(self.c1_coords[0] - dx/2, self.energy_axis[0] - dE/2,
                                    self.c1_coords[-1] - self.c1_coords[0] + dx,
                                    self.energy_axis[-1] - self.energy_axis[0] + dE)
        self.hslice_img.setRect(rect_hslice)

        # Tab 2: Top Gate Current Heatmap (X axis is Vtop physical, Y axis is C2)
        # Physical grids may not be perfectly uniform, but we map using minimum/maximum coordinates
        v1_min, v1_max = np.min(self.v1_grid), np.max(self.v1_grid)
        dv1 = (v1_max - v1_min) / (self.v1_grid.shape[1] - 1) if self.v1_grid.shape[1] > 1 else 1.0
        rect2 = QtCore.QRectF(v1_min - dv1/2, self.c2_coords[0] - dy/2,
                              v1_max - v1_min + dv1,
                              self.c2_coords[-1] - self.c2_coords[0] + dy)
        self.t2_heatmap_img.setRect(rect2)

        # Tab 3: Bottom Gate Current Heatmap (X axis is Vbottom physical, Y axis is C1)
        v2_min, v2_max = np.min(self.v2_grid), np.max(self.v2_grid)
        dv2 = (v2_max - v2_min) / (self.v2_grid.shape[0] - 1) if self.v2_grid.shape[0] > 1 else 1.0
        rect3 = QtCore.QRectF(v2_min - dv2/2, self.c1_coords[0] - dx/2,
                              v2_max - v2_min + dv2,
                              self.c1_coords[-1] - self.c1_coords[0] + dx)
        self.t3_heatmap_img.setRect(rect3)

        # Initialize crosshairs to center of ranges
        self.cross1_v.setValue((self.c1_coords[0] + self.c1_coords[-1]) / 2)
        self.cross1_h.setValue((self.c2_coords[0] + self.c2_coords[-1]) / 2)
        self.cross2_v.setValue((v1_min + v1_max) / 2)
        self.cross2_h.setValue((self.c2_coords[0] + self.c2_coords[-1]) / 2)
        self.cross3_v.setValue((v2_min + v2_max) / 2)
        self.cross3_h.setValue((self.c1_coords[0] + self.c1_coords[-1]) / 2)

    # --- Live Rendering updates ---
    def update_heatmaps_and_slices(self):
        """Calculates integrated counts and currents to update heatmaps and slices."""
        if not np.any(self.data_measured_mask):
            return

        # 1. Gate Map Integration
        e_min, e_max = self.int_region.getRegion()
        # Find indices corresponding to this energy window
        idx_range = np.where((self.energy_axis >= e_min) & (self.energy_axis <= e_max))[0]
        
        if len(idx_range) > 0:
            # Sum counts over the chosen spectral region
            integrated_map = np.sum(self.data_spectra[:, :, idx_range], axis=2)
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
            self.gatemap_colorbar.setLevels((min_val, max_val))

        # Update Tab 2 and Tab 3 current heatmaps
        self.t2_heatmap_img.setImage(self.data_current_ch1)
        if np.any(self.data_measured_mask):
            c1_meas = self.data_current_ch1[self.data_measured_mask]
            self.t2_colorbar.setLevels((np.min(c1_meas), np.max(c1_meas)))

        self.t3_heatmap_img.setImage(self.data_current_ch2.T)
        if np.any(self.data_measured_mask):
            c2_meas = self.data_current_ch2[self.data_measured_mask]
            self.t3_colorbar.setLevels((np.min(c2_meas), np.max(c2_meas)))

        # Force refreshing current slices based on active crosshairs
        self.on_crosshair_moved()
        self.on_tab2_crosshair_moved()
        self.on_tab3_crosshair_moved()

    def on_integration_region_changed(self):
        """Called dynamically as the shaded spectral region is dragged."""
        self.update_heatmaps_and_slices()

    def on_crosshair_moved(self):
        """Update Tab 1 slice cuts and spectrum plot when crosshairs are dragged."""
        if not np.any(self.data_measured_mask):
            return

        c1_val = self.cross1_v.value()
        c2_val = self.cross1_h.value()

        # Find closest grid indexes
        idx_c1 = np.argmin(np.abs(self.c1_coords - c1_val))
        idx_c2 = np.argmin(np.abs(self.c2_coords - c2_val))

        # 1. Update 1D Spectrum at crosshair intersection
        spectrum = self.data_spectra[idx_c2, idx_c1, :]
        self.spectrum_curve.setData(self.energy_axis, spectrum)

        # 2. Update Vertical Slice (all C2 coordinates at fixed C1)
        # Array shape: (c2_len, 512)
        vslice_data = self.data_spectra[:, idx_c1, :]
        self.vslice_img.setImage(vslice_data)
        
        # Position dashed line indicator on vertical slice
        self.vslice_marker.setValue(self.c2_coords[idx_c2])

        # 3. Update Horizontal Slice (all C1 coordinates at fixed C2)
        # Array shape: (512, c1_len) - transposed for row-major format
        hslice_data = self.data_spectra[idx_c2, :, :].T
        self.hslice_img.setImage(hslice_data)
        
        # Position dashed line indicator on horizontal slice
        self.hslice_marker.setValue(self.c1_coords[idx_c1])

    def on_tab2_crosshair_moved(self):
        """Update Tab 2 slices when its crosshair is dragged."""
        if not np.any(self.data_measured_mask):
            return

        v1_val = self.cross2_v.value()
        c2_val = self.cross2_h.value()

        # Find closest indices in grid
        # Average v1 value along columns
        v1_avg = np.mean(self.v1_grid, axis=0)
        idx_v1 = np.argmin(np.abs(v1_avg - v1_val))
        idx_c2 = np.argmin(np.abs(self.c2_coords - c2_val))

        # Horizontal slice: Current vs Vtop (at fixed C2)
        # We grab the row along the C2 coordinate
        curr_vs_v1 = self.data_current_ch1[idx_c2, :]
        v1_coords_row = self.v1_grid[idx_c2, :]
        self.t2_hslice_curve.setData(v1_coords_row, curr_vs_v1)

        # Vertical slice: Current vs C2 (at fixed Vtop)
        curr_vs_c2 = self.data_current_ch1[:, idx_v1]
        self.t2_vslice_curve.setData(self.c2_coords, curr_vs_c2)

    def on_tab3_crosshair_moved(self):
        """Update Tab 3 slices when its crosshair is dragged."""
        if not np.any(self.data_measured_mask):
            return

        v2_val = self.cross3_v.value()
        c1_val = self.cross3_h.value()

        # Find closest indices in grid
        v2_avg = np.mean(self.v2_grid, axis=1) # average along rows
        idx_v2 = np.argmin(np.abs(v2_avg - v2_val))
        idx_c1 = np.argmin(np.abs(self.c1_coords - c1_val))

        # Horizontal slice: Current vs Vbottom (at fixed C1)
        curr_vs_v2 = self.data_current_ch2[:, idx_c1]
        v2_coords_col = self.v2_grid[:, idx_c1]
        self.t3_hslice_curve.setData(v2_coords_col, curr_vs_v2)

        # Vertical slice: Current vs C1 (at fixed Vbottom)
        curr_vs_c1 = self.data_current_ch2[idx_v2, :]
        self.t3_vslice_curve.setData(self.c1_coords, curr_vs_c1)

    # --- HDF5 Data Storage Saving ---
    def save_sweep_data(self):
        """Save the measured dataset (spectra and currents) to a structured HDF5 file."""
        if not np.any(self.data_measured_mask):
            QtWidgets.QMessageBox.warning(self, "No Data", "There is no sweep data to save.")
            return

        try:
            data_dir = os.path.join(parent_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_filename = f"gate_sweep_{timestamp}.h5"

            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Measurement Data",
                os.path.join(data_dir, default_filename),
                "HDF5 files (*.h5);;All files (*.*)"
            )

            if not filename:
                return

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
                f.create_dataset("measured_mask", data=self.data_measured_mask)

                # Save coefficients and metadata
                f.attrs["coeff_a"] = self.coeff_a.value()
                f.attrs["coeff_b"] = self.coeff_b.value()
                f.attrs["coeff_c"] = self.coeff_c.value()
                f.attrs["coeff_d"] = self.coeff_d.value()
                f.attrs["exposure_time"] = self.exposure_box.value()
                f.attrs["settling_time"] = self.settling_box.value()
                f.attrs["timestamp"] = timestamp

            self.status_lbl.setText(f"Status: Saved {os.path.basename(filename)}")
            QtWidgets.QMessageBox.information(self, "Success", f"Data saved successfully to:\n{filename}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save dataset:\n{e}")

    def closeEvent(self, event):
        """Cleanup thread objects on exit to avoid dangling process exceptions."""
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        event.accept()


if __name__ == '__main__':
    print("Initializing physical station for sweep GUI...")
    simulated_devices = {
        'smu': False,          # Connect to the physical Keysight SMU
        'monochromator': True, 
        'rotator1': True,      
        'rotator2': True,      
        'camera': True,        
        'piezo150': True,      
        'piezo300': True       
    }
    station = initialize_station(simulated=False, simulated_devices=simulated_devices)
    
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(station)
    win.show()
    sys.exit(app.exec())
