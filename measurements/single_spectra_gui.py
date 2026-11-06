import os
import sys
import json
import time
import threading
import numpy as np
import Pyro4
import h5py
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Ensure parent directory is in path for importing station_setup and drivers
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from measurements.utils import configure_plot_axes
from drivers.lightfield_qcodes import lf_signal_bridge
from station_setup import initialize_station, simulated_devices
from measurement_data_paths import dated_measurement_filename, dated_measurement_path
from measurement_data_paths import measurement_data_dir, measurement_start_text, new_measurement_start, measurement_device_modes


class AcquisitionWorker(QtCore.QThread):
    """
    Background worker thread to handle camera spectrum acquisitions.
    Supports single runs (Acquire, Background, Reference) and Continuous live plotting.
    """
    point_measured = QtCore.pyqtSignal(np.ndarray, np.ndarray, object)  # wavelengths, intensities, raw_intensities
    progress_updated = QtCore.pyqtSignal(int, int)  # current, total
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, station, mode, num_spectra, average, accum_method, center_wl=500.0, parent=None):
        super().__init__(parent)
        self.station = station
        self.mode = mode  # "single", "continuous", "background", "reference"
        self.num_spectra = num_spectra
        self.average = average
        self.accum_method = accum_method
        self.center_wl = center_wl
        self._stop_requested = False
        self._paused = False
        self._preview_lock = threading.Lock()
        self._latest_preview = None

    def take_latest_preview(self):
        """One-slot mailbox: slow rendering cannot queue outdated live frames."""
        with self._preview_lock:
            frame, self._latest_preview = self._latest_preview, None
        return frame

    def stop(self):
        self._stop_requested = True

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def run(self):
        try:
            camera = self.station.camera
            
            if self.mode == "continuous":
                previous_frames = camera.num_frames()
                camera.num_frames(1)
                try:
                    camera.remote_lf.start_preview()
                    while not self._stop_requested:
                        poll_started = time.monotonic()
                        if self._paused:
                            self.msleep(20)
                            continue
                        data = camera.remote_lf.get_latest_preview_data()
                        if data is not None and not isinstance(data, str):
                            intensities = np.asarray(data, dtype=float)
                            wavelengths = np.asarray(camera.remote_lf.get_wavelengths_cache(), dtype=float)
                            if (intensities.ndim != 1 or wavelengths.ndim != 1
                                    or intensities.shape != wavelengths.shape
                                    or not np.all(np.isfinite(intensities))
                                    or not np.all(np.isfinite(wavelengths))):
                                raise ValueError("Live spectrum has no valid wavelength calibration or intensity array.")
                            # Keep the completed frame even if Stop was clicked
                            # during a legacy NIRvana acquisition.
                            with self._preview_lock:
                                self._latest_preview = (wavelengths, intensities, None)
                        # Native streams are polled at up to 30 Hz. Blocking
                        # legacy acquisitions need no additional 100 ms delay.
                        remaining_ms = int(33 - 1000 * (time.monotonic() - poll_started))
                        if remaining_ms > 0 and not self._stop_requested:
                            self.msleep(remaining_ms)
                finally:
                    try:
                        camera.remote_lf.stop_preview()
                    finally:
                        camera.num_frames(previous_frames)
                return

            # --- Hardware native multi-exposure acquisition ---
            try:
                camera.num_frames(self.num_spectra)
            except Exception as e:
                print(f"Error setting remote num_frames to {self.num_spectra}: {e}")

            wl_arr, intensities_arr = self.capture_one()
            if intensities_arr is None or wl_arr is None:
                return

            # Convert to float for mathematical operations
            intensities_arr = np.asarray(intensities_arr, dtype=float)

            if intensities_arr.ndim == 2:
                if self.average or self.accum_method == "Average":
                    processed_arr = np.mean(intensities_arr, axis=0)
                else:
                    processed_arr = np.sum(intensities_arr, axis=0)
            else:
                processed_arr = intensities_arr

            self.progress_updated.emit(self.num_spectra, self.num_spectra)
            self.point_measured.emit(wl_arr, processed_arr, intensities_arr)

        except Exception as e:
            if not self._stop_requested:
                self.error_occurred.emit(str(e))

    def capture_one(self):
        camera = self.station.camera
        # --- Physical Remote Acquisition via Pyro4 ---
        # Note: get_spectrum returns (intensities, wavelengths)
        res = camera.remote_lf.get_spectrum()
        if res is None:
            return None, None
        intensities, wl = res
        
        wl_arr = np.array(wl)
        intensities_arr = np.array(intensities)
        
        # If the remote server returns 2D multi-frame arrays, extract the first row for wavelengths
        if wl_arr.ndim == 2:
            wl_arr = wl_arr[0]
            
        # If the intensity array is 2D and has only 1 frame, flatten it to 1D
        if intensities_arr.ndim == 2 and intensities_arr.shape[0] == 1:
            intensities_arr = intensities_arr[0]
            
        return wl_arr, intensities_arr


class IntermediatePreviewWorker(QtCore.QThread):
    """Query intermediate frames outside the GUI thread using an owned socket."""
    def __init__(self, uri, spectrometer_type, generation, parent=None):
        super().__init__(parent)
        self.uri, self.spectrometer_type = uri, spectrometer_type
        self.generation = generation
        self.frame = None

    def run(self):
        proxy = None
        try:
            proxy = Pyro4.Proxy(self.uri)
            proxy._pyroTimeout = 2.0
            if self.spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                from drivers.remote_client import PyroProxyWrapper
                proxy = PyroProxyWrapper(proxy)
            data = proxy.get_latest_preview_data()
            if data is None or isinstance(data, str):
                return
            wavelengths = np.asarray(proxy.get_wavelengths_cache(), dtype=float)
            intensities = np.asarray(data, dtype=float)
            if (wavelengths.ndim == 1 and intensities.shape == wavelengths.shape
                    and wavelengths.size and np.all(np.isfinite(wavelengths))
                    and np.all(np.isfinite(intensities))):
                self.frame = (wavelengths, intensities)
        except Exception:
            # An optional intermediate frame must not fail a recorded exposure.
            pass
        finally:
            if proxy is not None:
                try:
                    proxy._pyroRelease()
                except Exception:
                    pass


class SpectrometerControlWorker(QtCore.QThread):
    """Keep remote motor operations off the GUI thread; never retry movement."""
    def __init__(self, remote, command, value, parent=None):
        super().__init__(parent)
        self.remote, self.command, self.value = remote, command, value
        self.result, self.error = None, None

    def run(self):
        try:
            self.result = getattr(self.remote, self.command)(self.value)
        except Exception as exc:
            self.error = str(exc)


class StatusWorker(QtCore.QThread):
    """
    Background worker thread to poll camera/spectrometer status (temperature, gain, grating)
    using its own dedicated Pyro4 Proxy socket connection.
    This prevents the main PyQt GUI thread from freezing.
    """
    status_polled = QtCore.pyqtSignal(dict)

    def __init__(self, uri, spectrometer_type="lightfieldG15", camera=None, parent=None):
        super().__init__(parent)
        self.uri = uri
        self.spectrometer_type = spectrometer_type
        self.camera = camera
        self._stop_requested = False
        self._is_paused = False

    def stop(self):
        self._stop_requested = True

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def run(self):
        if self.spectrometer_type == 'lightfieldG11' or getattr(self.camera, 'simulated', False):
            while not self._stop_requested:
                if self._is_paused:
                    self.msleep(100)
                    continue
                try:
                    if self.camera is not None and getattr(self.camera, 'remote_lf', None) is not None:
                        data = self.camera.remote_lf.get_status()
                        if not self._stop_requested:
                            self.status_polled.emit(data)
                except Exception as e:
                    if not self._stop_requested:
                        self.status_polled.emit({
                            'success': False,
                            'error': str(e)
                        })
                for _ in range(30):
                    if self._stop_requested:
                        break
                    self.msleep(100)
            return

        # Establish its own dedicated Pyro4 Proxy for thread safety and independence
        Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
        Pyro4.config.SERIALIZER = 'pickle'
        Pyro4.config.PICKLE_PROTOCOL_VERSION = 2
        proxy = None
        try:
            raw_proxy = Pyro4.Proxy(self.uri)
            if self.spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                from drivers.remote_client import PyroProxyWrapper
                proxy = PyroProxyWrapper(raw_proxy)
            else:
                proxy = raw_proxy
            proxy._pyroTimeout = 5.0
        except Exception as e:
            print(f"StatusWorker: Failed to create proxy to {self.uri}: {e}")
            return

        while not self._stop_requested:
            if self._is_paused:
                self.msleep(100)
                continue
            try:
                # Rebind if necessary
                try:
                    proxy._pyroBind()
                except Exception:
                    pass

                # Fetch all status parameters in a single sleep-free network call
                data = proxy.get_status()
                if not self._stop_requested:
                    self.status_polled.emit(data)

            except Exception as e:
                err_str = str(e).lower()
                # If we lost connection, try to reconnect the proxy
                if any(term in err_str for term in ["connection", "closed", "10054", "10060", "lost", "refused"]):
                    try:
                        proxy._pyroReconnect(tries=1)
                    except Exception:
                        pass
                
                # Emit failure state
                if not self._stop_requested:
                    self.status_polled.emit({
                        'success': False,
                        'error': str(e)
                    })

            # Wait 3 seconds, checking self._stop_requested every 100ms
            for _ in range(30):
                if self._stop_requested:
                    break
                self.msleep(100)

        # Cleanup proxy connection
        if proxy is not None:
            try:
                proxy._pyroRelease()
            except Exception:
                pass


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
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(4)
            
            self.btn_grid_A = QtWidgets.QPushButton("")
            self.btn_grid_A.setToolTip("Select Plot")
            self.btn_grid_A.setCheckable(True)
            self.btn_grid_A.setChecked(True)
            
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
            
            grid.addWidget(self.btn_grid_A, 0, 0)
            layout.addLayout(grid)
        
        # 2. Axis Selection Buttons
        layout.addWidget(QtWidgets.QLabel("Autoscale Axis:"))
        axis_layout = QtWidgets.QHBoxLayout()
        axis_layout.setSpacing(4)

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

        axis_layout.addWidget(self.btn_axis_both)
        axis_layout.addWidget(self.btn_axis_x)
        axis_layout.addWidget(self.btn_axis_y)

        self.btn_axis_both.setChecked(True)
        layout.addLayout(axis_layout)
        
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

    def get_axis_mode(self):
        if self.btn_axis_both.isChecked():
            return "Both"
        elif self.btn_axis_x.isChecked():
            return "X"
        elif self.btn_axis_y.isChecked():
            return "Y"
        else:
            return "None"


class SingleSpectraWindow(QtWidgets.QMainWindow):
    """
    Main Window for the "Single Spectra" GUI.
    """
    def __init__(self, station, embed=False):
        super().__init__()
        self.station = station
        self._embed = embed
        self.parent_dir = parent_dir
        self.setWindowTitle("Single Spectra Acquisition Dashboard")
        self.setMinimumSize(1150, 750)

        # Active Data Store
        self.signal_x = None
        self.signal_y = None
        self.signal_y_raw = None
        
        self.bg_x = None
        self.bg_y = None
        self.bg_y_raw = None
        
        self.ref_x = None
        self.ref_y = None
        self.ref_y_raw = None

        # Threading Worker
        self.worker = None
        self.is_running = False
        self.is_paused = False
        self.plotter_mode = False
        self._live_autoscale = True
        self._preview_generation = 0
        self._intermediate_worker = None

        # Last known states for removed UI widgets
        self.last_known_shutter = ""
        self.last_known_ccd_status = ""

        # Load initial value_storage settings
        self.load_value_storage()
        
        # Apply initial exposure time to QCoDeS camera
        try:
            self.station.camera.exposure_time(self.stored_exposure_time)
        except Exception as e:
            print(f"Startup warning: could not set camera exposure time ({e})")

        self.setup_ui()
        self.apply_qss_theme()
        
        self.reconnect_status_worker()
        lf_signal_bridge.exposure_time_changed.connect(self.on_exposure_signal)
        lf_signal_bridge.num_frames_changed.connect(self.on_num_frames_signal)
        lf_signal_bridge.wavelength_changed.connect(self.on_wavelength_signal)
        lf_signal_bridge.grating_changed.connect(self.on_grating_signal)
        lf_signal_bridge.target_temperature_changed.connect(self.on_target_temperature_signal)
        lf_signal_bridge.shutter_control_changed.connect(self.on_shutter_signal)

        self.update_est_time()
        self.view_control_popup = ViewControlPopup(self, has_grid=True)
        
        # Timers only hand off work or draw the most recent completed frame;
        # no camera/network call is made from a GUI timer.
        self.live_poll_timer = QtCore.QTimer(self)
        self.live_poll_timer.setInterval(200)
        self.live_poll_timer.timeout.connect(self.poll_intermediate_frame)
        self.live_render_timer = QtCore.QTimer(self)
        self.live_render_timer.setInterval(33)
        self.live_render_timer.timeout.connect(self.render_latest_live_frame)

    def reconnect_status_worker(self):
        old_worker = getattr(self, "status_worker", None)
        if old_worker is not None:
            old_worker.stop()
            old_worker.wait()
            old_worker.deleteLater()
        self.spectrometer_status = {}
        camera = self.station.camera
        uri = getattr(camera.remote_lf, "_pyroUri", None)
        self.status_worker = StatusWorker(uri, spectrometer_type=camera.spectrometer_type,
                                          camera=camera, parent=self)
        self.status_worker.status_polled.connect(self.on_status_polled)
        self.status_worker.start()
        self.set_ui_enabled(True)

    # --- value_storage Sync ---
    def load_value_storage(self):
        storage_path = os.path.join(self.parent_dir, "value_storage")
        # Defaults
        self.stored_exposure_time = 0.1
        self.stored_grating        = "1500 g/mm"
        self.stored_wavelength     = 760.0
        self.stored_target_temp    = -70.0
        self.stored_num_spectra    = 1
        self.stored_average        = "Yes"
        self.stored_accum_method   = "Average"
        # plot_modes stored as dict {mode_name: bool}
        self.stored_plot_modes     = {"Signal": True, "Background": False,
                                      "Reference": False, "Signal - Background": False,
                                      "(Signal - Reference) / (Reference - Background)": False}
        self.stored_x_axis_scale   = "Wavelength (nm)"
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    data = json.load(f)
                shared_data = data.get("shared", {})
                single_data = data.get("single_spectra", {})
                self.stored_exposure_time = shared_data.get("exposure_time",  data.get("exposure_time", self.stored_exposure_time))
                self.stored_grating       = single_data.get("grating",        data.get("grating", self.stored_grating))
                self.stored_wavelength    = single_data.get("wavelength",     data.get("wavelength", self.stored_wavelength))
                self.stored_target_temp   = single_data.get("target_temp",    data.get("target_temp", self.stored_target_temp))
                self.stored_num_spectra   = single_data.get("num_spectra",    data.get("num_spectra", self.stored_num_spectra))
                # Force average and accumulation to always be Yes/Average
                self.stored_average       = "Yes"
                self.stored_accum_method  = "Average"
                stored = single_data.get("plot_modes", data.get("plot_modes"))
                if isinstance(stored, dict):
                    self.stored_plot_modes.update(stored)
                self.stored_x_axis_scale  = single_data.get("x_axis_scale",   data.get("x_axis_scale", self.stored_x_axis_scale))
            except Exception as e:
                print(f"Error loading value_storage: {e}")

    def reload_gui_from_storage(self):
        if self.plotter_mode:
            return
        self.load_value_storage()
        
        self.exposure_box.blockSignals(True)
        self.exposure_box.setValue(self.stored_exposure_time)
        self.exposure_box.blockSignals(False)

        self.grating_combo.blockSignals(True)
        self.grating_combo.setCurrentText(self.stored_grating)
        self.grating_combo.blockSignals(False)

        self.wl_box.blockSignals(True)
        self.wl_box.setValue(self.stored_wavelength)
        self.wl_box.blockSignals(False)

        self.energy_box.blockSignals(True)
        self.energy_box.setValue(1239.84193 / self.stored_wavelength)
        self.energy_box.blockSignals(False)

        self.target_temp_box.blockSignals(True)
        self.target_temp_box.setValue(self.stored_target_temp)
        if not self.target_temp_title_lbl.isEnabled():
            self.target_temp_box.lineEdit().setText("")
        self.target_temp_box.blockSignals(False)

        # Restore stored values for UI elements (num spectra, average, accumulation)
        self.num_spectra_box.blockSignals(True)
        self.num_spectra_box.setValue(self.stored_num_spectra)
        self.num_spectra_box.blockSignals(False)

        # Force hidden combos to Yes/Average regardless of stored values
        self.average_combo.blockSignals(True)
        self.average_combo.setCurrentText("Yes")
        self.average_combo.blockSignals(False)

        self.accum_combo.blockSignals(True)
        self.accum_combo.setCurrentText("Average")
        self.accum_combo.blockSignals(False)

        self.scale_combo.blockSignals(True)
        self.scale_combo.setCurrentText(self.stored_x_axis_scale)
        self.scale_combo.blockSignals(False)

        for mode, cb in self.plot_mode_checks.items():
            cb.blockSignals(True)
            cb.setChecked(self.stored_plot_modes.get(mode, mode == "Signal"))
            cb.blockSignals(False)

        self.update_est_time()
        try:
            self.station.camera.exposure_time(self.stored_exposure_time)
        except Exception:
            pass

    def save_value_storage(self):
        if self.plotter_mode:
            return
        storage_path = os.path.join(self.parent_dir, "value_storage")
        data = {}
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    data = json.load(f)
            except Exception:
                pass
        if "shared" not in data:
            data["shared"] = {}
        if "single_spectra" not in data:
            data["single_spectra"] = {}
        data["shared"]["exposure_time"] = self.exposure_box.value()
        camera = getattr(self.station, "camera", None)
        data["shared"]["spectrometer"] = getattr(camera, "spectrometer_type", "lightfieldG15")
        data["single_spectra"]["grating"]       = self.grating_combo.currentText()
        data["single_spectra"]["wavelength"]    = self.wl_box.value()
        data["single_spectra"]["target_temp"]   = self.target_temp_box.value()
        # Use stored values for removed UI elements
        data["single_spectra"]["num_spectra"]   = self.num_spectra_box.value()
        data["single_spectra"]["average"]       = self.average_combo.currentText()
        data["single_spectra"]["accum_method"]  = self.accum_combo.currentText()
        data["single_spectra"]["plot_modes"]    = {m: cb.isChecked() for m, cb in self.plot_mode_checks.items()}
        data["single_spectra"]["x_axis_scale"]  = self.scale_combo.currentText()
        
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
            data.pop(key, None)
            
        try:
            with open(storage_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving value_storage: {e}")

    # --- UI Setup ---
    def setup_ui(self):
        # Main layout holds top control panel and bottom plotting/actions split
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ----------------------------------------------------
        # TOP ROW: 4 Configuration Cards Side-by-Side
        # ----------------------------------------------------
        top_row_layout = QtWidgets.QHBoxLayout()
        top_row_layout.setSpacing(10)

        # Box 1: Spectrograph Settings
        spec_box = QtWidgets.QGroupBox("Spectrograph Settings")
        spec_layout = QtWidgets.QFormLayout(spec_box)
        spec_layout.setSpacing(8)
        
        self.grating_combo = QtWidgets.QComboBox()
        # Human‑readable labels mapped to exact LightField strings
        self.grating_map = {
            "1500 g/mm": "[hol,1500][0][0]",
            "1200 g/mm": "[750  nm,1200][1][0]",
            "600 g/mm": "[600][0][0]",
            "300 g/mm": "[1.0  \u00e6m,300][2][0]",
            "150 g/mm": "[150][0][0]"
        }
        # Reverse lookup for syncing UI with remote state
        self.grating_reverse = {v: k for k, v in self.grating_map.items()}
        self.grating_combo.addItems(["1500 g/mm", "1200 g/mm", "600 g/mm", "300 g/mm", "150 g/mm"])
        self.grating_combo.blockSignals(True)
        self.grating_combo.setCurrentText(self.stored_grating)
        self.grating_combo.blockSignals(False)
        self.grating_combo.currentTextChanged.connect(self.on_grating_changed)
        self.grating_setting = False
        self.last_grating_label = ""  # remembers last successful label
        spec_layout.addRow(QtWidgets.QLabel("Grating:"), self.grating_combo)
        
        self.wl_box = QtWidgets.QDoubleSpinBox()
        self.wl_box.setRange(200.0, 2500.0)
        self.wl_box.setDecimals(2)
        self.wl_box.setSingleStep(1.0)
        # Fetch current wavelength (fall back to stored value)
        try:
            self.wl_box.setValue(self.station.monochromator.wavelength())
        except Exception:
            self.wl_box.setValue(self.stored_wavelength)
        self.wl_box.valueChanged.connect(self.sync_wl_to_energy)
        self.wl_box.editingFinished.connect(self.on_wavelength_changed)
        spec_layout.addRow(QtWidgets.QLabel("Center Wavelength (nm):"), self.wl_box)
        
        self.energy_box = QtWidgets.QDoubleSpinBox()
        self.energy_box.setRange(0.49, 6.2)
        self.energy_box.setDecimals(4)
        self.energy_box.setSingleStep(0.01)
        self.energy_box.setValue(1239.84193 / self.wl_box.value())
        self.energy_box.valueChanged.connect(self.sync_energy_to_wl)
        self.energy_box.editingFinished.connect(self.on_energy_changed)
        spec_layout.addRow(QtWidgets.QLabel("Center Energy (eV):"), self.energy_box)
        
        self.spectrometer_readback = QtWidgets.QLabel("Waiting for spectrometer readback...")
        self.spectrometer_readback.setWordWrap(True)
        spec_layout.addRow(self.spectrometer_readback)
        top_row_layout.addWidget(spec_box)

        # Box 2: Camera Settings
        self.cam_box = QtWidgets.QGroupBox("Camera Settings")
        self.cam_box.setEnabled(False)  # Disabled initially until status is polled
        cam_layout = QtWidgets.QFormLayout(self.cam_box)
        cam_layout.setSpacing(8)
        
        # Target Temperature input field
        self.target_temp_title_lbl = QtWidgets.QLabel("CCD Temperature (°C):")
        self.target_temp_title_lbl.setEnabled(False)
        self.target_temp_box = QtWidgets.QDoubleSpinBox()
        self.target_temp_box.setRange(-140.0, 20.0)
        self.target_temp_box.setDecimals(1)
        self.target_temp_box.setSingleStep(5.0)
        self.target_temp_box.setValue(self.stored_target_temp)
        self.target_temp_box.setEnabled(False)
        self.target_temp_box.lineEdit().setText("")
        self.target_temp_box.editingFinished.connect(self.on_target_temperature_changed)
        cam_layout.addRow(self.target_temp_title_lbl, self.target_temp_box)
        
        # Current Temperature label
        self.ccd_temp_title_lbl = QtWidgets.QLabel("Current Temperature:")
        self.ccd_temp_title_lbl.setEnabled(False)
        self.ccd_temp_lbl = QtWidgets.QLabel("")
        self.ccd_temp_lbl.setEnabled(False)
        cam_layout.addRow(self.ccd_temp_title_lbl, self.ccd_temp_lbl)
        
        top_row_layout.addWidget(self.cam_box)

        # Check if spectrometer is G11/G15 to determine visibility of temperature controls
        spec_type = "lightfieldG15"
        if hasattr(self, "station") and hasattr(self.station, "camera") and self.station.camera is not None:
            spec_type = getattr(self.station.camera, "spectrometer_type", "lightfieldG15")
        if "g11" in spec_type.lower():
            self.cam_box.setVisible(False)
        else:
            self.cam_box.setVisible(True)

        # Box 3: Exposure Settings
        exp_box = QtWidgets.QGroupBox("Exposure Settings")
        exp_layout = QtWidgets.QFormLayout(exp_box)
        exp_layout.setSpacing(8)
        
        self.exposure_box = QtWidgets.QDoubleSpinBox()
        self.exposure_box.setRange(0.001, 1000.0)
        self.exposure_box.setDecimals(3)
        self.exposure_box.setSingleStep(0.01)
        self.exposure_box.setValue(self.stored_exposure_time)
        self.exposure_box.valueChanged.connect(self.on_exposure_changed)
        exp_layout.addRow(QtWidgets.QLabel("Exposure Time (s):"), self.exposure_box)
        
        self.gain_lbl = QtWidgets.QLabel("*")
        
        top_row_layout.addWidget(exp_box)

        # Box 4: Multiexposure
        multi_box = QtWidgets.QGroupBox("Multiexposure Settings")
        multi_layout = QtWidgets.QFormLayout(multi_box)
        multi_layout.setSpacing(8)
        
        # Number of spectra (frames) control
        self.num_spectra_box = QtWidgets.QSpinBox()
        self.num_spectra_box.setRange(1, 10000)
        self.num_spectra_box.setValue(self.stored_num_spectra)
        self.num_spectra_box.valueChanged.connect(self.on_num_spectra_changed)
        multi_layout.addRow(QtWidgets.QLabel("No. Spectra:"), self.num_spectra_box)

        # Hidden Average combo (forced to "Yes")
        self.average_combo = QtWidgets.QComboBox()
        self.average_combo.addItems(["Yes"])
        self.average_combo.setCurrentText("Yes")
        self.average_combo.setEnabled(False)
        self.average_combo.hide()

        # Hidden Accumulation combo (forced to "Average")
        self.accum_combo = QtWidgets.QComboBox()
        self.accum_combo.addItems(["Average"])
        self.accum_combo.setCurrentText("Average")
        self.accum_combo.setEnabled(False)
        self.accum_combo.hide()
        
        # Average UI removed
        
        # Accumulation method UI removed
        
        top_row_layout.addWidget(multi_box)

        main_layout.addLayout(top_row_layout)

        # ----------------------------------------------------
        # MIDDLE AREA: Left Plot + Right Buttons
        # ----------------------------------------------------
        middle_layout = QtWidgets.QHBoxLayout()
        middle_layout.setSpacing(15)

        # Left Container: Plot + Plot Configuration Layout
        plot_container = QtWidgets.QWidget()
        plot_vbox = QtWidgets.QVBoxLayout(plot_container)
        plot_vbox.setContentsMargins(0, 0, 0, 0)
        plot_vbox.setSpacing(8)

        # Header for the View Control button (placed on the top right)
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
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
        self.btn_view_control.setObjectName("viewControlBtn")
        self.btn_view_control.setToolTip("Show view and autoscale configuration options")
        self.btn_view_control.clicked.connect(self.toggle_view_control_popup)
        header_layout.addWidget(self.btn_view_control)
        plot_vbox.addLayout(header_layout)

        # Plot Config controls – checkboxes so multiple modes can be shown at once
        plot_ctrls = QtWidgets.QHBoxLayout()

        # Mode colours: same order as PLOT_MODES
        PLOT_MODES = [
            ("Signal",                                          "#3b82f6"),
            ("Background",                                      "#f59e0b"),
            ("Reference",                                       "#10b981"),
            ("Signal - Background",                             "#a855f7"),
            ("(Signal - Reference) / (Reference - Background)", "#ef4444"),
        ]
        self.plot_modes_cfg  = PLOT_MODES          # keep for replot colour lookup
        self.plot_mode_checks = {}                 # mode -> QCheckBox
        self.plot_curves      = {}                 # mode -> PlotDataItem (created after plot_widget)

        for mode, colour in PLOT_MODES:
            cb = QtWidgets.QCheckBox(mode)
            cb.setChecked(self.stored_plot_modes.get(mode, mode == "Signal"))
            cb.setStyleSheet(f"""
                QCheckBox {{
                    color: {colour};
                    font-size: 11px;
                }}
                QCheckBox::indicator {{
                    border: 1px solid #94A3B8;
                    border-radius: 0px;
                    background-color: #FFFFFF;
                    width: 14px;
                    height: 14px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: #475569;
                    border-color: #334155;
                }}
            """)
            cb.stateChanged.connect(self.replot)
            cb.stateChanged.connect(lambda _: self.save_value_storage())
            self.plot_mode_checks[mode] = cb
            plot_ctrls.addWidget(cb)
            plot_ctrls.addSpacing(6)

        plot_ctrls.addStretch(1)
        plot_ctrls.addWidget(QtWidgets.QLabel("X-Axis:"))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["Wavelength (nm)", "Energy (eV)"])
        self.scale_combo.setCurrentText(self.stored_x_axis_scale)
        self.scale_combo.currentTextChanged.connect(self.replot)
        self.scale_combo.currentTextChanged.connect(lambda: self.save_value_storage())
        plot_ctrls.addWidget(self.scale_combo)

        plot_vbox.addLayout(plot_ctrls)

        # pyqtgraph PlotWidget
        self.plot_widget = pg.PlotWidget(title="Spectrograph Analysis View")
        self.plot_widget.setLabel('left', 'Intensity', 'counts')
        self.plot_widget.setLabel('bottom', 'Wavelength', 'nm')
        configure_plot_axes(self.plot_widget)
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.getViewBox().sigRangeChangedManually.connect(self._on_manual_plot_range)
        self.plot_widget.getViewBox().sigStateChanged.connect(self._on_plot_view_state)
        # Create one PlotDataItem per mode
        for mode, colour in self.plot_modes_cfg:
            curve = self.plot_widget.plot([], [], name=mode, pen=pg.mkPen(colour, width=2))
            self.plot_curves[mode] = curve
        
        plot_vbox.addWidget(self.plot_widget, 1)

        # Right Container: Control Buttons Box
        btn_box = QtWidgets.QGroupBox("Execution Controls")
        btn_layout = QtWidgets.QVBoxLayout(btn_box)
        btn_layout.setSpacing(12)
        btn_layout.setContentsMargins(15, 20, 15, 15)
        btn_box.setFixedWidth(240)

        self.run_btn = QtWidgets.QPushButton("Live on")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setCheckable(True)
        self.run_btn.clicked.connect(self.toggle_run)
        btn_layout.addWidget(self.run_btn)

        self.acquire_btn = QtWidgets.QPushButton("Acquire")
        self.acquire_btn.setObjectName("acquireBtn")
        self.acquire_btn.clicked.connect(lambda: self.start_acquisition("single"))
        btn_layout.addWidget(self.acquire_btn)

        self.bg_btn = QtWidgets.QPushButton("Background")
        self.bg_btn.setObjectName("bgBtn")
        self.bg_btn.clicked.connect(lambda: self.start_acquisition("background"))
        btn_layout.addWidget(self.bg_btn)

        self.ref_btn = QtWidgets.QPushButton("Reference")
        self.ref_btn.setObjectName("refBtn")
        self.ref_btn.clicked.connect(lambda: self.start_acquisition("reference"))
        btn_layout.addWidget(self.ref_btn)

        self.reset_btn = QtWidgets.QPushButton("↺ Reset")
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self.reset_acquisition)
        btn_layout.addWidget(self.reset_btn)

        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.setEnabled(False)  # Disabled until data is acquired
        self.save_btn.clicked.connect(self.save_data_dialog)
        btn_layout.addWidget(self.save_btn)

        btn_layout.addStretch(1)

        # Estimation & Progress Bar
        self.est_time_lbl = QtWidgets.QLabel("Est. total time: 0.0 s")
        self.est_time_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; font-weight: bold; color: #f8fafc;")
        btn_layout.addWidget(self.est_time_lbl)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)

        self.status_lbl = QtWidgets.QLabel("Status: Idle")
        self.status_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        btn_layout.addWidget(self.status_lbl)

        middle_layout.addWidget(btn_box)
        middle_layout.addWidget(plot_container, 1)

        main_layout.addLayout(middle_layout)

        # Add Permanent Device Connectivity Indicator to Status Bar (only when not embedded)
        if not getattr(self, "_embed", False):
            status_bar = self.statusBar()
            spec_name = "LightField"
            location = "G15-Pylon"
            if hasattr(self, "station") and hasattr(self.station, "camera") and self.station.camera is not None:
                spec_type = getattr(self.station.camera, "spectrometer_type", "lightfieldG15")
                if spec_type in ["wingspecG15", "winspecG15"]:
                    spec_name = "WinSpec"
                    location = "phd-exile-phys"
                elif spec_type in ["winspecG11", "wingspecG11"]:
                    spec_name = "WinSpec (G11)"
                    location = "bluefors-pc"
                elif spec_type == "lightfieldG11":
                    spec_name = "NIRvana (G11)"
                    location = "DESKTOP-LSQKIT5.dhcp-int-phys.ethz.ch"
            self.conn_lbl = QtWidgets.QLabel(f"{spec_name}: CONNECTED ({location})  ")
            self.conn_lbl.setStyleSheet("color: #10b981; font-weight: bold; font-family: Consolas, monospace;")
            status_bar.addPermanentWidget(self.conn_lbl)

        # Native preview displays spectra without creating acquisition files.
        spec_type = "lightfieldG15"
        if hasattr(self, "station") and hasattr(self.station, "camera") and self.station.camera is not None:
            spec_type = getattr(self.station.camera, "spectrometer_type", "lightfieldG15")
        
        self.run_btn.setToolTip("Continuously display spectra for alignment without saving them in the dashboard.")
            
    # --- Device Event Callbacks ---
    def sync_wl_to_energy(self, wl):
        self.energy_box.blockSignals(True)
        if wl > 0:
            self.energy_box.setValue(1239.84193 / wl)
        else:
            self.energy_box.setValue(0.0)
        self.energy_box.blockSignals(False)

    def sync_energy_to_wl(self, energy):
        self.wl_box.blockSignals(True)
        if energy > 0:
            self.wl_box.setValue(1239.84193 / energy)
        else:
            self.wl_box.setValue(200.0)
        self.wl_box.blockSignals(False)

    def _uses_nirvana_controls(self):
        camera = self.station.camera
        return not camera.simulated and camera.spectrometer_type == "lightfieldG11"

    def _start_spectrometer_control(self, command, value):
        if self.plotter_mode:
            return
        if self.is_running or getattr(self, "grating_setting", False):
            return
        status = getattr(self, "spectrometer_status", {})
        capability = "wavelength" if command == "set_wavelength" else "grating"
        if not status.get("capabilities", {}).get(capability, False):
            return
        if command == "set_wavelength" and status.get("wl") is not None and abs(status["wl"] - value) < 0.01:
            return
        self.grating_setting = True
        self.set_ui_enabled(False)
        self.status_worker.pause()
        self.status_lbl.setText("Waiting for LightField spectrometer readback...")
        self.control_worker = SpectrometerControlWorker(self.station.camera.remote_lf,
                                                        command, value, self)
        self.control_worker.finished.connect(self._finish_spectrometer_control)
        self.control_worker.start()
        parent = self.grating_combo.window()
        if parent is not self and hasattr(parent, "check_measurement_status"):
            parent.check_measurement_status()

    def _finish_spectrometer_control(self):
        worker = self.control_worker
        self.grating_setting = False
        if worker.result is not None:
            self.on_status_polled(worker.result)
            self.status_lbl.setText("Status: LightField setting confirmed by readback")
            self.save_value_storage()
        else:
            self.spectrometer_status = {}
            self.spectrometer_readback.setText("Setting could not be confirmed. Waiting for fresh readback.")
            self.status_lbl.setText("Spectrometer error: " + str(worker.error))
        self.set_ui_enabled(True)
        self.status_worker.resume()
        self.control_worker = None
        worker.deleteLater()
        parent = self.grating_combo.window()
        if parent is not self and hasattr(parent, "check_measurement_status"):
            parent.check_measurement_status()

    def on_wavelength_changed(self):
        if self.plotter_mode:
            return
        wl = self.wl_box.value()
        if self._uses_nirvana_controls():
            self._start_spectrometer_control("set_wavelength", wl)
            return
        try:
            self.station.monochromator.wavelength(wl)
            if self.station.camera.simulated:
                self.station.camera.remote_lf.wavelength = wl
                self.station.camera.remote_lf._cached_wavelengths = None
            self.status_lbl.setText(f"Status: Wavelength set to {wl:.2f} nm")
            self.save_value_storage()
        except Exception as e:
            self.status_lbl.setText(f"Error: Failed to set wavelength ({e})")

    def on_energy_changed(self):
        if self.plotter_mode:
            return
        if self._uses_nirvana_controls():
            self.on_wavelength_changed()
            return
        wl = self.wl_box.value()
        energy = self.energy_box.value()
        try:
            self.station.monochromator.wavelength(wl)
            if self.station.camera.simulated:
                self.station.camera.remote_lf.wavelength = wl
                self.station.camera.remote_lf._cached_wavelengths = None
            self.status_lbl.setText(f"Status: Wavelength set to {wl:.2f} nm ({energy:.4f} eV)")
            self.save_value_storage()
        except Exception as e:
            self.status_lbl.setText(f"Error: Failed to set wavelength ({e})")

    def on_grating_changed(self):
        if self.plotter_mode:
            return
        label = self.grating_combo.currentText()
        if self._uses_nirvana_controls():
            value = self.grating_map.get(label)
            if value is not None:
                self._start_spectrometer_control("set_grating", value)
            return
        self.grating_setting = True
        self.status_lbl.setText(f"Setting grating to {label}")
        QtWidgets.QApplication.processEvents()
        try:
            grating_str = self.grating_map.get(label)
            if grating_str is None:
                raise ValueError(f"Unknown grating label {label!r}")
            self.station.camera.remote_lf.set_grating(grating_str)
            self.last_grating_label = label
            self.save_value_storage()
            self.status_lbl.setText(label)
            # Revert to Idle after 2 s; keep grating_setting True for 5 s so
            # the poll cannot override the combo while the grating physically moves.
            QtCore.QTimer.singleShot(2000, lambda: self.status_lbl.setText("Status: Idle"))
            QtCore.QTimer.singleShot(5000, self._clear_grating_setting)
        except Exception as e:
            self.status_lbl.setText(f"Error setting grating: {e}")
            self._clear_grating_setting()

    def _clear_grating_setting(self):
        self.grating_setting = False




    def on_exposure_changed(self):
        if self.plotter_mode:
            return
        val = self.exposure_box.value()
        try:
            self.station.camera.exposure_time(val)
        except Exception as e:
            print(f"Error setting camera exposure time: {e}")
        self.save_value_storage()
        self.update_est_time()

    def on_num_spectra_changed(self):
        if self.plotter_mode:
            return
        val = self.num_spectra_box.value()
        try:
            self.station.camera.num_frames(val)
        except Exception as e:
            print(f"Error setting camera num_frames: {e}")
        self.save_value_storage()
        self.update_est_time()

    def update_est_time(self):
        """Calculate and display the estimated total acquisition time.
        Uses the current value from the Number of Spectra (frames) control.
        """
        t = self.exposure_box.value()
        n = self.num_spectra_box.value()
        total = t * n
        self.est_time_lbl.setText(f"Est. total time: {total:.1f} s")
        self.save_value_storage()

    def on_target_temperature_changed(self):
        if self.plotter_mode:
            return
        val = self.target_temp_box.value()
        try:
            self.station.camera.remote_lf.set_target_temperature(val)
            self.status_lbl.setText(f"Status: Target temperature set to {val:.1f} °C")
            self.save_value_storage()
        except Exception as e:
            self.status_lbl.setText(f"Error: Failed to set target temperature ({e})")



    # --- Periodic Device Polling ---
    # --- Slots for instrument signal bridge ---
    def on_exposure_signal(self, val):
        if self.plotter_mode:
            return
        # Update exposure UI without triggering recursive set
        self.exposure_box.blockSignals(True)
        self.exposure_box.setValue(val)
        self.exposure_box.blockSignals(False)
        self.save_value_storage()
        self.update_est_time()
        self.status_lbl.setText(f"Status: Exposure updated to {val:.3f}s (remote)")

    def on_num_frames_signal(self, val):
        if self.plotter_mode:
            return
        if self.is_running:
            return
        self.num_spectra_box.blockSignals(True)
        self.num_spectra_box.setValue(val)
        self.num_spectra_box.blockSignals(False)
        self.save_value_storage()
        self.update_est_time()
        self.status_lbl.setText(f"Status: Num spectra set to {val} (remote)")

    def on_wavelength_signal(self, wl):
        if self.plotter_mode:
            return
        # Update wavelength and synced energy
        if abs(self.wl_box.value() - wl) > 0.01:
            self.wl_box.blockSignals(True)
            self.wl_box.setValue(wl)
            self.wl_box.blockSignals(False)
            # Sync energy
            self.sync_wl_to_energy(wl)
            self.save_value_storage()
            self.status_lbl.setText(f"Status: Wavelength updated to {wl:.2f} nm (remote)")

    def on_grating_signal(self, grating):
        if self.plotter_mode:
            return
        value = str(grating)
        label = self.grating_reverse.get(value, value)
        self.grating_map[label] = value
        self.grating_reverse[value] = label
        self.grating_combo.blockSignals(True)
        if self.grating_combo.findText(label) < 0:
            self.grating_combo.addItem(label)
        self.grating_combo.setCurrentText(label)
        self.grating_combo.blockSignals(False)

    def on_target_temperature_signal(self, val):
        if self.plotter_mode:
            return
        self.target_temp_box.blockSignals(True)
        self.target_temp_box.setValue(val)
        self.target_temp_box.blockSignals(False)
        self.save_value_storage()
        self.status_lbl.setText(f"Status: Target temperature updated to {val:.1f} °C (remote)")

    def on_shutter_signal(self, val):
        if self.plotter_mode:
            return
        try:
            idx = int(val)
            shutter_map = {1: "Normal", 2: "Always Closed", 3: "Always Open"}
            self.last_known_shutter = shutter_map.get(idx, f"UNKNOWN ({val})")
            self.status_lbl.setText(f"Status: Shutter updated to {self.last_known_shutter} (remote)")
        except Exception:
            pass

    def on_status_polled(self, data):
        if self.plotter_mode:
            return
        if self._uses_nirvana_controls() and self.grating_setting:
            return  # Discard a read started before the current movement.
        self.spectrometer_status = dict(data)
        if self._uses_nirvana_controls():
            if "gratings" in data:
                choices = [str(value) for value in data["gratings"]]
                if choices != list(self.grating_map):
                    self.grating_map = {value: value for value in choices}
                    self.grating_reverse = dict(self.grating_map)
                    self.grating_combo.blockSignals(True)
                    self.grating_combo.clear()
                    self.grating_combo.addItems(choices)
                    self.grating_combo.blockSignals(False)
            self._update_spectrometer_controls()
        if not data.get('success', False):
            # Check connection issue
            err = data.get('error', '')
            print(f"Background status poll error: {err}")
            
            self.cam_box.setEnabled(False)
            
            self.target_temp_title_lbl.setEnabled(False)
            self.target_temp_box.setEnabled(False)
            self.target_temp_box.lineEdit().setText("")
            
            self.ccd_temp_title_lbl.setEnabled(False)
            self.ccd_temp_lbl.setEnabled(False)
            self.ccd_temp_lbl.setText("")
            
            self.last_known_ccd_status = ""
            self.last_known_shutter = ""
            
            self.gain_lbl.setText("*")
            return

        # 1. Update Actual Temperature
        temp = data.get('temp')
        try:
            if temp is not None and temp != "*":
                self.cam_box.setEnabled(True)
                if isinstance(temp, (int, float)):
                    temp_text = f"{temp:.1f} °C"
                else:
                    temp_text = str(temp)
                self.ccd_temp_title_lbl.setEnabled(True)
                self.ccd_temp_lbl.setEnabled(True)
                if self.ccd_temp_lbl.text() != temp_text:
                    self.ccd_temp_lbl.setText(temp_text)
            else:
                self.cam_box.setEnabled(False)
                self.ccd_temp_title_lbl.setEnabled(False)
                self.ccd_temp_lbl.setEnabled(False)
                self.ccd_temp_lbl.setText("")
        except Exception:
            pass
            
        # 2. Update Target Temperature
        target_temp = data.get('target_temp')
        try:
            if target_temp is not None and target_temp != "*":
                spec_type = getattr(self.station.camera, "spectrometer_type", "lightfieldG15")
                supports_hardware_cfg = self.station.camera.simulated or spec_type != "lightfieldG11"
                ui_enabled = self.acquire_btn.isEnabled() and supports_hardware_cfg
                
                self.target_temp_title_lbl.setEnabled(True)
                self.target_temp_box.setEnabled(ui_enabled)
                if isinstance(target_temp, (int, float)) and not self.target_temp_box.hasFocus():
                    if abs(self.target_temp_box.value() - target_temp) > 0.1:
                        self.target_temp_box.blockSignals(True)
                        self.target_temp_box.setValue(target_temp)
                        self.target_temp_box.blockSignals(False)
            else:
                self.target_temp_title_lbl.setEnabled(False)
                self.target_temp_box.setEnabled(False)
                self.target_temp_box.lineEdit().setText("")
        except Exception:
            pass

        # 4. Update Temperature Status
        status = data.get('status')
        try:
            if status is not None and status != "*":
                status_map = {
                    1: "LOCKED",
                    2: "COOLING",
                    3: "HEATING",
                    4: "OFF"
                }
                if isinstance(status, int):
                    self.last_known_ccd_status = status_map.get(status, f"UNKNOWN ({status})")
                else:
                    self.last_known_ccd_status = str(status)
            else:
                self.last_known_ccd_status = ""
        except Exception:
            pass

        # 5. Update ADC Analog Gain
        gain = data.get('gain')
        try:
            if gain is not None:
                gain_str = str(gain)
                if self.gain_lbl.text() != gain_str:
                    self.gain_lbl.setText(gain_str)
            else:
                if self.gain_lbl.text() != "*":
                    self.gain_lbl.setText("*")
        except Exception:
            pass

        # Display the actual returned grating, including values absent from the
        # old hard-coded list. Never replace unknown hardware with a cached label.
        grating = data.get("grating")
        if grating is not None and not self.grating_setting:
            self.on_grating_signal(grating)
        wl = data.get("wl")
        if data.get("status_source") == "local_settings":
            self.spectrometer_readback.setText(
                "Live uses repeated acquisitions; the dashboard does not save them. "
                "LightField's existing export settings still apply. Set grating/wavelength in LightField."
                if self._uses_legacy_nirvana_live() else
                "Remote grating/wavelength controls unavailable. Normal Acquire remains available.")
            self.grating_combo.blockSignals(True)
            self.grating_combo.setCurrentIndex(-1)
            self.grating_combo.setPlaceholderText("Unavailable from server")
            self.grating_combo.blockSignals(False)
            self.wl_box.lineEdit().setText("Unavailable")
            self.energy_box.lineEdit().setText("Unavailable")
        else:
            self.spectrometer_readback.setText(
                f"{'SIMULATED' if self.station.camera.simulated else 'Hardware readback'}: "
                f"grating {grating if grating is not None else 'unavailable'}; "
                f"centre {str(wl) + ' nm' if wl is not None else 'unavailable'}")

        # 7. Update Exposure Time (only if user is not focusing the box)
        exposure = data.get('exposure')
        try:
            if exposure is not None and not self.exposure_box.hasFocus():
                if abs(self.exposure_box.value() - exposure) > 1e-4:
                    self.exposure_box.blockSignals(True)
                    self.exposure_box.setValue(exposure)
                    self.exposure_box.blockSignals(False)
                    # Keep local caching variable in sync and update estimate
                    self.stored_exposure_time = exposure
                    self.update_est_time()
        except Exception:
            pass

        # 8. Update Center Wavelength (only if user is not focusing the box)
        wl = data.get('wl')
        try:
            if wl is not None and not self.wl_box.hasFocus():
                if abs(self.wl_box.value() - wl) > 0.01 or self.wl_box.lineEdit().text() == "Unavailable":
                    self.wl_box.blockSignals(True)
                    self.wl_box.setValue(wl)
                    self.wl_box.blockSignals(False)
                    self.sync_wl_to_energy(wl)
        except Exception:
            pass

        # 9. Update Shutter Mode
        shutter = data.get('shutter')
        try:
            if shutter is not None and shutter != "*":
                idx = int(shutter)
                shutter_map = {1: "Normal", 2: "Always Closed", 3: "Always Open"}
                self.last_known_shutter = shutter_map.get(idx, f"UNKNOWN ({shutter})")
            else:
                self.last_known_shutter = ""
        except Exception:
            pass

        # 10. Update Number of Spectra/Frames (only if user is not focusing the box and not running)
        num_frames = data.get('num_frames')
        try:
            if num_frames is not None and not self.is_running and not self.num_spectra_box.hasFocus():
                if int(self.num_spectra_box.value()) != int(num_frames):
                    self.num_spectra_box.blockSignals(True)
                    self.num_spectra_box.setValue(num_frames)
                    self.num_spectra_box.blockSignals(False)
                    self.update_est_time()
        except Exception:
            pass



    # --- Core Acquisition Control ---
    # --- Core Acquisition Control ---
    def start_acquisition(self, mode):
        if self.plotter_mode or self.is_running:
            return
        if getattr(self, "grating_setting", False):
            return
        self.measurement_device_modes = measurement_device_modes(self.station)
        self.measurement_started_at = new_measurement_start()
        self._interrupt_requested = False
        self._autosaved_this_run = False
        self.status_bar_update("Acquiring...")
        self.set_ui_enabled(False)
        self.progress_bar.setMaximum(0)  # Infinite loader pattern during active camera acquisition
        self.progress_bar.setValue(0)
        self.reset_btn.setEnabled(True)
        self.is_running = True
        self.station.live_status_paused = True
        self.is_paused = False
        self.active_acquisition_mode = mode
        self._acquisition_failed = False
        self._acquisition_error = ""
        self._received_data_this_run = False
        self._preview_generation += 1
        
        camera = self.station.camera
        spec_type = getattr(camera, "spectrometer_type", "lightfieldG15")
        if spec_type not in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
            self.live_poll_timer.start()

        num_spectra = self.num_spectra_box.value()
        average = self.average_combo.currentText() == "Yes"
        accum_method = self.accum_combo.currentText()
        
        self.worker = AcquisitionWorker(
            self.station, mode, num_spectra, average, accum_method, center_wl=self.wl_box.value(), parent=self
        )
        
        self.worker.point_measured.connect(lambda xl, yl, y_raw=None: self.on_data_acquired(mode, xl, yl, y_raw))
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.finished.connect(self.on_acquisition_finished)
        self.worker.error_occurred.connect(self.on_acquisition_error)
        self.worker.start()

    def toggle_run(self):
        if self.plotter_mode:
            self.run_btn.setChecked(False)
            return
        if getattr(self, "grating_setting", False):
            self.run_btn.setChecked(False)
            return
        if self.run_btn.isChecked():
            if not self._live_preview_available():
                self.run_btn.blockSignals(True)
                self.run_btn.setChecked(False)
                self.run_btn.blockSignals(False)
                self._update_spectrometer_controls()
                self.status_bar_update("Live unavailable: the connected NIRvana server does not provide a no-save stream. Normal Acquire is available.")
                return
            self.active_acquisition_mode = "continuous"
            self._first_live_frame = True
            self._interrupt_requested = False
            self._autosaved_this_run = False
            self._acquisition_failed = False
            self._acquisition_error = ""
            self._received_data_this_run = False
            self.set_ui_enabled(False)
            self.run_btn.setEnabled(True)
            self.station.live_status_paused = True
            # Disable other triggers
            self.acquire_btn.setEnabled(False)
            self.bg_btn.setEnabled(False)
            self.ref_btn.setEnabled(False)
            self.reset_btn.setEnabled(True)
            self.is_running = True
            self.run_btn.setText("Live off")
            self.is_paused = False
            self.status_bar_update("Waiting for the first live spectrum; dashboard saving is off.")
            self.progress_bar.setMaximum(0)  # Infinite loader pattern
            
            # Automatically check the "Signal" box so the user immediately sees it
            if "Signal" in self.plot_mode_checks:
                self.plot_mode_checks["Signal"].setChecked(True)
                
            if self._live_autoscale:
                self.plot_widget.enableAutoRange()
            
            self.worker = AcquisitionWorker(self.station, "continuous", 1, False, "Sum", center_wl=self.wl_box.value(), parent=self)
            self.worker.error_occurred.connect(self.on_acquisition_error)
            self.worker.finished.connect(self.on_acquisition_finished)
            self.live_render_timer.start()
            self.worker.start()
        else:
            self.reset_acquisition()

    def reset_acquisition(self):
        if not self.is_running:
            return
        self._interrupt_requested = True
        self.live_poll_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            # Set the stop flag before cancelling the socket so cancellation
            # cannot be mistaken for a failed measurement.
            self.worker.stop()
            finish_current = (self._uses_legacy_nirvana_live()
                              and self.active_acquisition_mode == "continuous")
            if not finish_current:
                try:
                    self.station.camera.abort_acquisition()
                except Exception as exc:
                    print(f"Could not cancel acquisition: {exc}")
            self.run_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.status_bar_update("Stopping after the current spectrum..." if finish_current else "Stopping acquisition...")
        else:
            self.on_acquisition_finished()

    def on_progress_updated(self, current, total):
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setMaximum(100)
        pct = int(100 * current / total)
        self.progress_bar.setValue(pct)

    def render_latest_live_frame(self):
        worker = self.worker
        if worker is None or getattr(self, "active_acquisition_mode", None) != "continuous":
            return
        frame = worker.take_latest_preview()
        if frame is not None:
            self.on_data_acquired("continuous", *frame)

    def poll_intermediate_frame(self):
        if (self.plotter_mode or not self.is_running or self._intermediate_worker is not None
                or getattr(self, "active_acquisition_mode", None) == "continuous"):
            return
        camera = self.station.camera
        uri = getattr(getattr(camera, "remote_lf", None), "_pyroUri", None)
        if uri is None:
            return
        worker = IntermediatePreviewWorker(uri, camera.spectrometer_type, self._preview_generation, self)
        self._intermediate_worker = worker
        worker.finished.connect(self._intermediate_frame_finished)
        worker.start()

    def _intermediate_frame_finished(self):
        worker = self._intermediate_worker
        if worker is None:
            return
        self._intermediate_worker = None
        if (worker.generation == self._preview_generation and self.is_running
                and not self.plotter_mode and worker.frame is not None
                and not getattr(self, "_received_data_this_run", False)):
            mode = self.active_acquisition_mode
            prefix = {"single": "signal", "background": "bg", "reference": "ref"}.get(mode)
            if prefix is not None:
                x, y = worker.frame
                setattr(self, prefix + "_x", x)
                setattr(self, prefix + "_y", y)
                self.replot()
        worker.deleteLater()

    def on_data_acquired(self, mode, x, y, y_raw=None):
        self._received_data_this_run = True
        # Save variables depending on the trigger button
        if mode == "continuous" or mode == "single":
            self.signal_x = x
            self.signal_y = y
            self.signal_y_raw = y_raw
            if "Signal" in self.plot_mode_checks and not self.plot_mode_checks["Signal"].isChecked():
                self.plot_mode_checks["Signal"].setChecked(True)
        elif mode == "background":
            self.bg_x = x
            self.bg_y = y
            self.bg_y_raw = y_raw
            if "Background" in self.plot_mode_checks and not self.plot_mode_checks["Background"].isChecked():
                self.plot_mode_checks["Background"].setChecked(True)
        elif mode == "reference":
            self.ref_x = x
            self.ref_y = y
            self.ref_y_raw = y_raw
            if "Reference" in self.plot_mode_checks and not self.plot_mode_checks["Reference"].isChecked():
                self.plot_mode_checks["Reference"].setChecked(True)
            
        self.replot()
        if mode == "continuous" and getattr(self, "_first_live_frame", False):
            self._first_live_frame = False
            self.status_bar_update("Live alignment; dashboard saving is off.")
        if mode != "continuous":
            self.update_save_button_state()

    def on_acquisition_finished(self):
        self.live_poll_timer.stop()
        continuous = getattr(self, "active_acquisition_mode", None) == "continuous"
        if continuous:
            self.render_latest_live_frame()
        self.live_render_timer.stop()
        self._preview_generation += 1
        self.is_running = False
        self.station.live_status_paused = False
        self.is_paused = False
        self.run_btn.blockSignals(True)
        self.run_btn.setChecked(False)
        self.run_btn.setText("Live on")
        self.run_btn.blockSignals(False)
        self.set_ui_enabled(True)
        self.reset_btn.setEnabled(False)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0 if continuous else 100)
        if not continuous and getattr(self, "_received_data_this_run", False) and not getattr(self, "_acquisition_failed", False):
            self.trigger_autosave(was_interrupted=getattr(self, "_interrupt_requested", False))
        if getattr(self, "_acquisition_failed", False):
            self.status_bar_update("Acquisition failed: " + getattr(self, "_acquisition_error", "Unknown error"))
        else:
            self.status_bar_update("Live stopped; last spectrum retained. No spectra saved by the dashboard." if continuous else "Acquisition finished.")

    def on_acquisition_error(self, err_msg):
        self._acquisition_failed = True
        self._acquisition_error = str(err_msg)
        self.status_bar_update(f"Acquisition failed: {err_msg}")
        QtWidgets.QMessageBox.critical(self, "Acquisition Error", f"An error occurred during acquisition:\n{err_msg}")

    def trigger_autosave(self, was_interrupted=False):
        if getattr(self, "active_acquisition_mode", None) == "continuous":
            return
        if getattr(self, "_autosaved_this_run", False):
            return
        has_data = (self.signal_y is not None) or (self.bg_y is not None) or (self.ref_y is not None)
        if not has_data:
            return
        autosave_dir = measurement_data_dir(
            parent_dir, getattr(self, "measurement_started_at", None), autosave=True
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_interrupted" if was_interrupted else ""
        filename = f"autosave_spectra_{timestamp[-6:]}{suffix}.h5"
        filename = dated_measurement_filename(filename, getattr(self, "measurement_started_at", None))
        file_path = os.path.join(autosave_dir, filename)
        try:
            self.save_data_to_file(file_path)
            self._autosaved_this_run = True
            print(f"Auto-saved data to {file_path}")
        except Exception as e:
            print(f"Auto-save failed: {e}")

    def prompt_save_on_interrupt(self):
        has_data = (self.signal_y is not None) or (self.bg_y is not None) or (self.ref_y is not None)
        if not has_data:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Save Interrupted Measurement",
            "The acquisition was interrupted. Would you like to save the acquired data?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.save_data_dialog()

    def status_bar_update(self, text):
        self.status_lbl.setText(f"Status: {text}")

    def update_save_button_state(self):
        # Enable save button if we have any of signal, background, or reference data
        has_data = (self.signal_y is not None) or (self.bg_y is not None) or (self.ref_y is not None)
        self.save_btn.setEnabled(has_data and not self.is_running)



    def save_data_to_file(self, file_path, selections=None):
        if self.plotter_mode:
            return
        """Save selected datasets to file.
        :param file_path: Destination file path.
        :param selections: List of dataset names to include. Possible values: 'Signal', 'Background', 'Reference'.
        If None, all available datasets are saved.
        """
        if self.is_running and getattr(self, "active_acquisition_mode", None) == "continuous":
            raise RuntimeError("Live alignment spectra are not saved. Stop Live and use Acquire to record a spectrum.")
        if selections is None:
            selections = []
        # If no selections specified, default to all available datasets
        if not selections:
            if self.signal_x is not None:
                selections.append('Signal')
            if self.bg_x is not None:
                selections.append('Background')
            if self.ref_x is not None:
                selections.append('Reference')
        # Determine the primary wavelength grid based on selected datasets
        x_grid = None
        # Prioritize signal, then background, then reference
        candidates = []
        if 'Signal' in selections and self.signal_x is not None:
            candidates.append(self.signal_x)
        if 'Background' in selections and self.bg_x is not None:
            candidates.append(self.bg_x)
        if 'Reference' in selections and self.ref_x is not None:
            candidates.append(self.ref_x)
        for x_arr in candidates:
            if x_arr is not None:
                x_grid = x_arr
                break
        if x_grid is None:
            # Fallback to any available x grid
            for x_arr in (self.signal_x, self.bg_x, self.ref_x):
                if x_arr is not None:
                    x_grid = x_arr
                    break
        if x_grid is None:
            raise ValueError("No data available to save.")

        # JSON formatting option
        if file_path.endswith('.json'):
            data = {}
            if 'Signal' in selections and self.signal_x is not None:
                data["signal"] = {"wavelength": self.signal_x.tolist(), "intensity": self.signal_y.tolist()}
                sig_raw = getattr(self, 'signal_y_raw', None)
                if sig_raw is None:
                    sig_raw = np.array([self.signal_y])
                data["signal"]["raw_intensities"] = sig_raw.tolist()
            if 'Background' in selections and self.bg_x is not None:
                data["background"] = {"wavelength": self.bg_x.tolist(), "intensity": self.bg_y.tolist()}
                bg_raw = getattr(self, 'bg_y_raw', None)
                if bg_raw is None:
                    bg_raw = np.array([self.bg_y])
                data["background"]["raw_intensities"] = bg_raw.tolist()
            if 'Reference' in selections and self.ref_x is not None:
                data["reference"] = {"wavelength": self.ref_x.tolist(), "intensity": self.ref_y.tolist()}
                ref_raw = getattr(self, 'ref_y_raw', None)
                if ref_raw is None:
                    ref_raw = np.array([self.ref_y])
                data["reference"]["raw_intensities"] = ref_raw.tolist()


            # Add metadata
            metadata = {}
            try:
                metadata["center_wavelength_nm"] = self.wl_box.value()
                metadata["grating"] = self.grating_combo.currentText()
                metadata["exposure_time_s"] = self.exposure_box.value()
                metadata["shutter"] = getattr(self, 'last_known_shutter', '')
                metadata["ccd_temperature_C"] = self.ccd_temp_lbl.text()
                metadata["ccd_status"] = getattr(self, 'last_known_ccd_status', '')
                metadata["gain"] = self.gain_lbl.text()
                metadata["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
                metadata["measurement_started_at"] = measurement_start_text(
                    getattr(self, "measurement_started_at", None)
                )
            except Exception:
                pass
            metadata["device_modes"] = getattr(self, "measurement_device_modes", {})
            data["metadata"] = metadata
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            return

        # HDF5 formatting option
        if file_path.endswith('.h5'):
            with h5py.File(file_path, 'w') as h5f:
                h5f.attrs["device_modes"] = json.dumps(getattr(self, "measurement_device_modes", {}))
                # Query current B field
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
                h5f.create_dataset("measured_b", data=meas_b)

                # Query measured V1 and V2
                meas_v1 = 0.0
                meas_v2 = 0.0
                try:
                    smu = getattr(self.station, "smu", None)
                    if smu is not None and not getattr(smu, "simulated", False):
                        try:
                            ch1_on = bool(smu.ch1.output())
                            meas_v1 = smu.ch1.measure_volt() if ch1_on else 0.0
                        except Exception:
                            meas_v1 = smu.ch1.volt()
                        try:
                            ch2_on = bool(smu.ch2.output())
                            meas_v2 = smu.ch2.measure_volt() if ch2_on else 0.0
                        except Exception:
                            meas_v2 = smu.ch2.volt()
                except Exception:
                    pass
                h5f.create_dataset("measured_v1", data=meas_v1)
                h5f.create_dataset("measured_v2", data=meas_v2)

                # Store each selected dataset in its own group
                if 'Signal' in selections and self.signal_x is not None:
                    grp = h5f.create_group('signal')
                    grp.create_dataset('wavelength', data=self.signal_x)
                    grp.create_dataset('intensity', data=self.signal_y)
                    sig_raw = getattr(self, 'signal_y_raw', None)
                    if sig_raw is None:
                        sig_raw = np.array([self.signal_y])
                    grp.create_dataset('raw_intensities', data=sig_raw)
                if 'Background' in selections and self.bg_x is not None:
                    grp = h5f.create_group('background')
                    grp.create_dataset('wavelength', data=self.bg_x)
                    grp.create_dataset('intensity', data=self.bg_y)
                    bg_raw = getattr(self, 'bg_y_raw', None)
                    if bg_raw is None:
                        bg_raw = np.array([self.bg_y])
                    grp.create_dataset('raw_intensities', data=bg_raw)
                if 'Reference' in selections and self.ref_x is not None:
                    grp = h5f.create_group('reference')
                    grp.create_dataset('wavelength', data=self.ref_x)
                    grp.create_dataset('intensity', data=self.ref_y)
                    ref_raw = getattr(self, 'ref_y_raw', None)
                    if ref_raw is None:
                        ref_raw = np.array([self.ref_y])
                    grp.create_dataset('raw_intensities', data=ref_raw)

                # Metadata as attributes on root
                try:
                    h5f.attrs['center_wavelength_nm'] = self.wl_box.value()
                    h5f.attrs['grating'] = self.grating_combo.currentText()
                    h5f.attrs['exposure_time_s'] = self.exposure_box.value()
                    h5f.attrs['shutter'] = getattr(self, 'last_known_shutter', '')
                    h5f.attrs['ccd_temperature_C'] = self.ccd_temp_lbl.text()
                    h5f.attrs['ccd_status'] = getattr(self, 'last_known_ccd_status', '')
                    h5f.attrs['gain'] = self.gain_lbl.text()
                    h5f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    h5f.attrs['measurement_started_at'] = measurement_start_text(
                        getattr(self, "measurement_started_at", None)
                    )
                    # Store full value_storage JSON as a dataset for full state recovery
                    try:
                        storage_path = os.path.join(parent_dir, "value_storage")
                        if os.path.exists(storage_path):
                            with open(storage_path, "r") as f_vs:
                                vs_data = json.load(f_vs)
                            
                            # Create metadata group/folder
                            meta_grp = h5f.create_group("metadata")
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
                except Exception:
                    pass
                return

        # Text/CSV column-based formatting option
        metadata_lines = []
        metadata_lines.append("# --- Spectroscopy Data Save ---")
        metadata_lines.append("# Device modes: " + json.dumps(getattr(self, "measurement_device_modes", {})))
        metadata_lines.append(f"# Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        metadata_lines.append(
            f"# Measurement started: {measurement_start_text(getattr(self, 'measurement_started_at', None))}"
        )
        
        try:
            metadata_lines.append(f"# Monochromator Center Wavelength (nm): {self.wl_box.value()}")
            metadata_lines.append(f"# Grating: {self.grating_combo.currentText()}")
            metadata_lines.append(f"# Exposure Time (s): {self.exposure_box.value()}")
            metadata_lines.append(f"# Shutter: {getattr(self, 'last_known_shutter', '')}")
            metadata_lines.append(f"# CCD Temperature: {self.ccd_temp_lbl.text()}")
            metadata_lines.append(f"# CCD Status: {getattr(self, 'last_known_ccd_status', '')}")
            metadata_lines.append(f"# Preamplifier Gain: {self.gain_lbl.text()}")
        except Exception:
            pass
            
        header_cols = ["Wavelength (nm)"]
        data_cols = [x_grid]
        
        # Helper to safely interpolate if lengths differ (though they should match)
        def get_column_data(ref_x, ref_y):
            if ref_y is None:
                return None
            if len(ref_x) == len(x_grid) and np.allclose(ref_x, x_grid):
                return ref_y
            return np.interp(x_grid, ref_x, ref_y)
            
        if 'Signal' in selections and self.signal_y is not None:
            header_cols.append("Signal (counts)")
            data_cols.append(self.signal_y)
            
        if 'Background' in selections and self.bg_y is not None:
            header_cols.append("Background (counts)")
            data_cols.append(get_column_data(self.bg_x, self.bg_y))
            
        if 'Reference' in selections and self.ref_y is not None:
            header_cols.append("Reference (counts)")
            data_cols.append(get_column_data(self.ref_x, self.ref_y))
            

            
        metadata_lines.append("# Columns: " + ", ".join(header_cols))
        
        delimiter = "," if file_path.endswith('.csv') else "\t"
        with open(file_path, 'w') as f:
            for line in metadata_lines:
                f.write(line + "\n")
                
            num_rows = len(x_grid)
            for i in range(num_rows):
                row_vals = []
                for col in data_cols:
                    val = col[i]
                    if isinstance(val, float):
                        row_vals.append(f"{val:.6f}")
                    else:
                        row_vals.append(str(val))
                f.write(delimiter.join(row_vals) + "\n")

    def save_data_dialog(self):
        """Prompt for file location, suggest an automated filename, and save selected data.
        After a successful save, the current value_storage is persisted.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"spectra_{timestamp[-6:]}.h5"
        default_name = dated_measurement_filename(default_name, getattr(self, "measurement_started_at", None))
        default_dir = measurement_data_dir(
            parent_dir, getattr(self, "measurement_started_at", None)
        )
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Spectra Data",
            os.path.join(default_dir, default_name),
            "HDF5 Files (*.h5);;JSON Files (*.json);;CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return
        file_path = dated_measurement_path(file_path, getattr(self, "measurement_started_at", None))
        try:
            self.save_data_to_file(file_path, selections=None)
            self.status_bar_update(f"Saved to {os.path.basename(file_path)}")
            self.save_value_storage()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save data:\n{e}")

    def set_plotter_mode(self, enabled):
        if enabled and (self.is_running or getattr(self, "grating_setting", False)):
            raise RuntimeError("Finish acquisition or spectrometer movement before entering Plotter mode.")
        self.plotter_mode = bool(enabled)
        if self.plotter_mode:
            self.status_worker.pause()
            self.live_poll_timer.stop()
            self.live_render_timer.stop()
        else:
            self.status_worker.resume()
        self.set_ui_enabled(not self.is_running)

    def set_ui_enabled(self, enabled):
        enabled = enabled and not self.plotter_mode
        self._ui_enabled = enabled
        self.acquire_btn.setEnabled(enabled)
        self.bg_btn.setEnabled(enabled)
        self.ref_btn.setEnabled(enabled)
        self.run_btn.setEnabled(enabled)
        
        spec_type = "lightfieldG15"
        if hasattr(self, "station") and hasattr(self.station, "camera") and self.station.camera is not None:
            spec_type = getattr(self.station.camera, "spectrometer_type", "lightfieldG15")
        supports_hardware_cfg = self.station.camera.simulated or spec_type != "lightfieldG11"
        self._update_spectrometer_controls()
        self.exposure_box.setEnabled(enabled)
        self.num_spectra_box.setEnabled(enabled)
        self.average_combo.setEnabled(enabled)
        self.accum_combo.setEnabled(enabled)
        has_temp = self.target_temp_title_lbl.isEnabled()
        self.target_temp_box.setEnabled(enabled and supports_hardware_cfg and has_temp)

        # Update save button state based on data availability
        if enabled:
            self.update_save_button_state()
        else:
            self.save_btn.setEnabled(False)



    def _uses_legacy_nirvana_live(self):
        return (self._uses_nirvana_controls()
                and not getattr(self.station.camera.remote_lf, "controls_enabled", False))

    def _live_preview_available(self):
        if self._uses_legacy_nirvana_live():
            return True
        if not self._uses_nirvana_controls():
            return True
        status = getattr(self, "spectrometer_status", {})
        return bool(status.get("success") and status.get("capabilities", {}).get("preview"))

    def _update_spectrometer_controls(self):
        enabled = getattr(self, "_ui_enabled", True) and not self.plotter_mode
        nirvana = self._uses_nirvana_controls()
        status = getattr(self, "spectrometer_status", {})
        capabilities = status.get("capabilities", {})
        idle = not status.get("busy", False)
        for control, capability in ((self.grating_combo, "grating"),
                                    (self.wl_box, "wavelength"), (self.energy_box, "wavelength")):
            supported = not nirvana or (status.get("success", False) and capabilities.get(capability, False))
            control.setEnabled(enabled and supported and (not nirvana or idle))
            control.setToolTip("" if supported else "Requires LightField readback from the updated NIRvana server (port 65433).")
        preview = self._live_preview_available()
        # Keep the Stop button active while a live acquisition is running.
        if not (self.is_running and getattr(self, "active_acquisition_mode", None) == "continuous"):
            self.run_btn.setEnabled(enabled and preview)
            self.run_btn.setText("Live on")
        self.run_btn.setToolTip("Repeated acquisitions using the existing NIRvana server. No dashboard files are saved; LightField's existing export settings still apply. Stop finishes the current spectrum."
                               if self._uses_legacy_nirvana_live() else
                               "Live alignment: no spectra saved. Camera rows are summed for NIRvana preview."
                               if preview else "The connected NIRvana server only supports file-export acquisitions. A no-save live stream is unavailable; normal Acquire still works.")

    def _get_data_for_mode(self, mode):
        """Return (x, y) arrays for a single plot mode, or (None, None) if unavailable."""

        if mode == "Background" and self.bg_y is not None:
            return self.bg_x, self.bg_y

        if mode == "Reference" and self.ref_y is not None:
            return self.ref_x, self.ref_y

        if self.signal_y is None:
            return None, None

        x = self.signal_x
        y = self.signal_y
        if mode == "Signal":
            return x, y

        def get_interp(ref_x, ref_y):
            if ref_y is None or ref_x is None:
                return np.zeros_like(x)
            if len(ref_x) == len(x) and np.allclose(ref_x, x):
                return ref_y
            return np.interp(x, ref_x, ref_y)

        bg_y_interp = get_interp(self.bg_x, self.bg_y)

        if mode == "Background":
            return self.bg_x, self.bg_y
        elif mode == "Reference":
            return self.ref_x, self.ref_y
        elif mode == "Signal - Background":
            return x, y - bg_y_interp
        elif mode == "(Signal - Reference) / (Reference - Background)":
            ref_y_interp = get_interp(self.ref_x, self.ref_y)
            denom = ref_y_interp - bg_y_interp
            denom[denom == 0.0] = 1e-9
            return x, (y - ref_y_interp) / denom
    def closeEvent(self, event):
        if getattr(self, "control_worker", None) is not None and self.control_worker.isRunning():
            event.ignore()
            return
        if hasattr(self, "status_worker") and self.status_worker.isRunning():
            self.status_worker.stop()
            self.status_worker.wait()
        if getattr(self, "worker", None) and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        if self._intermediate_worker is not None:
            self._intermediate_worker.wait()
        self.save_value_storage()
        super().closeEvent(event)

    def _on_manual_plot_range(self, _axes):
        self._live_autoscale = False
        self.plot_widget.disableAutoRange()

    def _on_plot_view_state(self, view):
        # PyQtGraph's "A" (auto-range) button also resumes live autoscaling.
        if all(view.autoRangeEnabled()):
            self._live_autoscale = True

    def replot(self):
        scale = self.scale_combo.currentText()
        use_energy = scale == "Energy (eV)"
        if scale != getattr(self, "_plotted_axis_scale", None):
            self.plot_widget.setLabel('bottom', 'Energy' if use_energy else 'Wavelength',
                                      'eV' if use_energy else 'nm')
            configure_plot_axes(self.plot_widget)
            self._plotted_axis_scale = scale

        for mode, curve in self.plot_curves.items():
            cb = self.plot_mode_checks[mode]
            if cb.isChecked():
                x, y = self._get_data_for_mode(mode)
                if x is not None and y is not None:
                    if use_energy:
                        x_arr = 1239.84193 / np.asarray(x)
                        idx   = np.argsort(x_arr)
                        curve.setData(x_arr[idx], np.asarray(y)[idx])
                    else:
                        curve.setData(np.asarray(x), np.asarray(y))
                else:
                    curve.setData([], [])
            elif curve.xData is not None and len(curve.xData):
                curve.setData([], [])

    def scale_plots(self):
        """Scale the plot to fit all active/visible curves (those toggled on and with data)."""
        x_min, x_max = None, None
        y_min, y_max = None, None
        
        for mode, curve in self.plot_curves.items():
            cb = self.plot_mode_checks[mode]
            if cb.isChecked():
                x_data = curve.xData
                y_data = curve.yData
                if x_data is not None and len(x_data) > 0 and y_data is not None and len(y_data) > 0:
                    x_local_min = np.min(x_data)
                    x_local_max = np.max(x_data)
                    y_local_min = np.min(y_data)
                    y_local_max = np.max(y_data)
                    
                    if x_min is None or x_local_min < x_min:
                        x_min = x_local_min
                    if x_max is None or x_local_max > x_max:
                        x_max = x_local_max
                    if y_min is None or y_local_min < y_min:
                        y_min = y_local_min
                    if y_max is None or y_local_max > y_max:
                        y_max = y_local_max
                        
        if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            x_range = x_max - x_min
            if x_range <= 0:
                x_range = 1.0
            y_range = y_max - y_min
            if y_range <= 0:
                y_range = 1.0
            
            # Apply 2% padding to X and 5% padding to Y
            self.plot_widget.setRange(
                xRange=(x_min - 0.02 * x_range, x_max + 0.02 * x_range),
                yRange=(y_min - 0.05 * y_range, y_max + 0.05 * y_range),
                padding=0
            )
        else:
            # Fallback to standard autoRange if no active curve has data
            self.plot_widget.autoRange()
        self._live_autoscale = True
        if self.is_running and getattr(self, "active_acquisition_mode", None) == "continuous":
            self.plot_widget.enableAutoRange()

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
        if mode == "Both":
            self.scale_plots()
            return
            
        x_min, x_max = None, None
        y_min, y_max = None, None
        
        for mode_name, curve in self.plot_curves.items():
            cb = self.plot_mode_checks[mode_name]
            if cb.isChecked():
                x_data = curve.xData
                y_data = curve.yData
                if x_data is not None and len(x_data) > 0 and y_data is not None and len(y_data) > 0:
                    x_local_min = np.min(x_data)
                    x_local_max = np.max(x_data)
                    y_local_min = np.min(y_data)
                    y_local_max = np.max(y_data)
                    
                    if x_min is None or x_local_min < x_min:
                        x_min = x_local_min
                    if x_max is None or x_local_max > x_max:
                        x_max = x_local_max
                    if y_min is None or y_local_min < y_min:
                        y_min = y_local_min
                    if y_max is None or y_local_max > y_max:
                        y_max = y_local_max
                        
        if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            x_range = x_max - x_min
            if x_range <= 0:
                x_range = 1.0
            y_range = y_max - y_min
            if y_range <= 0:
                y_range = 1.0
            
            if mode == "Both":
                self.plot_widget.setRange(
                    xRange=(x_min - 0.02 * x_range, x_max + 0.02 * x_range),
                    yRange=(y_min - 0.05 * y_range, y_max + 0.05 * y_range),
                    padding=0
                )
            elif mode == "X":
                self.plot_widget.setRange(
                    xRange=(x_min - 0.02 * x_range, x_max + 0.02 * x_range),
                    padding=0
                )
            elif mode == "Y":
                self.plot_widget.setRange(
                    yRange=(y_min - 0.05 * y_range, y_max + 0.05 * y_range),
                    padding=0
                )
        else:
            if mode == "Both":
                self.plot_widget.autoRange()

    def apply_qss_theme(self):
        from theme import SCIENTIFIC_THEME_QSS
        self.setStyleSheet(SCIENTIFIC_THEME_QSS)
        # Also set the pyqtgraph plot background to white
        self.plot_widget.setBackground('w')
        self.plot_widget.getAxis('left').setPen(pg.mkPen('#334155'))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen('#334155'))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen('#334155'))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen('#334155'))


# --- Application Entry Point ---
if __name__ == '__main__':
    # Initialize QApplication
    app = QtWidgets.QApplication(sys.argv)
    
    # Initialize station (will automatically try G15-Pylon or fall back to simulation)
    print("Initializing QCoDeS station...")
    station = initialize_station(simulated=False, simulated_devices=simulated_devices)
    
    # Launch Single Spectra Window
    win = SingleSpectraWindow(station)
    win.show()
    sys.exit(app.exec())
