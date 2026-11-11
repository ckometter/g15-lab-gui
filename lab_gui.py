import sys
import os

# Ensure the root directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt6 import QtWidgets, QtGui, QtCore

import station_setup
from measurements.single_spectra_gui import SingleSpectraWindow
from measurements.Dual_gate_sweep_gui import MainWindow as DualGateMainWindow
from measurements.B_V_sweep_gui import BVSweepWindow
from measurements.polarization_control_gui import PolarizationControlWindow
from measurements.photoluminescence_map_gui import PhotoluminescenceMapWindow
from measurements.scalar_map_gui import ScalarMapWindow
from measurements.position_tab import PositionTab
from measurements.position_dgs_gui import PositionDGSWindow
from measurements.capacitance_gui import CapacitanceWindow
from measurements.transport_gui import TransportWindow
initialize_station = station_setup.initialize_station

# Tab indices are positional; new tabs are appended so earlier indices are unchanged.
CAPACITANCE_TAB = 8
TRANSPORT_TAB = 9


class LabMainWindow(QtWidgets.QMainWindow):
    """Top‑level window that hosts the two measurement GUIs as left‑side tabs.

    * The tab bar is placed on the **West** side (vertical tabs on the left).
    * Each sub‑GUI is instantiated with ``embed=True`` so that its own status
      bar is suppressed – the outer ``LabMainWindow`` provides a shared status bar.
    """

    def __init__(self, station, device_manager=None):
        super().__init__()
        self.station = station
        self.device_manager = device_manager
        self.devices_changing = False
        self.plotter_mode = False
        self._plotter_transition = False
        from station_devices import ensure_simulated_devices
        ensure_simulated_devices(station)
        self.setWindowTitle("G15‑Qcodes Lab Dashboard")
        self.setMinimumSize(1400, 850)

        # ---- Central Tab Widget -------------------------------------------------
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.TabPosition.West)
        self.tab_widget.setMovable(False)
        self.setCentralWidget(self.tab_widget)

        # ---- Single Spectra Tab -------------------------------------------------
        self.single_window = SingleSpectraWindow(station, embed=True)
        single_widget = self.single_window.centralWidget()
        self.tab_widget.addTab(single_widget, "Single Spectra")

        # ---- Dual‑Gate Sweep Tab ----------------------------------------------
        self.dual_window = DualGateMainWindow(station, embed=True)
        self.dual_window.single_spectra_window = self.single_window
        dual_widget = self.dual_window.centralWidget()
        self.tab_widget.addTab(dual_widget, "Dual‑Gate Sweep")

        # ---- B/V Sweep Tab ------------------------------------------------------
        self.bv_window = BVSweepWindow(station, embed=True)
        self.bv_window.single_spectra_window = self.single_window
        bv_widget = self.bv_window.centralWidget()
        self.tab_widget.addTab(bv_widget, "B/V Sweep")

        # ---- Vector Map Tab ----------------------------------------
        self.pl_map_window = PhotoluminescenceMapWindow(station, embed=True)
        self.pl_map_window.single_spectra_window = self.single_window
        pl_map_widget = self.pl_map_window.centralWidget()
        self.tab_widget.addTab(pl_map_widget, "Vector Map")

        # ---- Scalar Map Tab ----------------------------------------
        self.scalar_map_window = ScalarMapWindow(station, embed=True)
        scalar_map_widget = self.scalar_map_window.centralWidget()
        self.tab_widget.addTab(scalar_map_widget, "Scalar Map")

        # ---- Dashboard Tab -----------------------------------------------------------
        self.position_tab = PositionTab(self.station)
        self.tab_widget.addTab(self.position_tab, "Dashboard")

        # ---- Polarization Control Tab -------------------------------------------
        self.pol_window = PolarizationControlWindow(station, embed=True)
        pol_widget = self.pol_window.centralWidget()
        self.tab_widget.addTab(pol_widget, "Polarization control")

        self.position_dgs_window = PositionDGSWindow(station, embed=True)
        self.position_dgs_window.dashboard = self
        self.position_dgs_window.single_spectra_window = self.single_window
        self.tab_widget.addTab(self.position_dgs_window.centralWidget(), "Position DGS")

        # ---- Capacitance Bridge Tab --------------------------------------------
        self.capacitance_window = CapacitanceWindow(station, embed=True)
        self.capacitance_window.dashboard = self
        self.tab_widget.addTab(self.capacitance_window.centralWidget(), "Capacitance")

        # ---- Four-probe Transport Tab ------------------------------------------
        self.transport_window = TransportWindow(station, embed=True)
        self.transport_window.dashboard = self
        self.tab_widget.addTab(self.transport_window.centralWidget(), "Transport")
        # Gate-map tabs (measurements/gate_map_gui.py) share one interface: busy(),
        # save_storage(), reload_gui_from_storage(), live_status_worker, shutdown().
        self.gate_map_tabs = {CAPACITANCE_TAB: self.capacitance_window, TRANSPORT_TAB: self.transport_window}

        for index, window in ((0, self.single_window), (1, self.dual_window), (7, self.position_dgs_window)):
            window.dashboard = self
            window.plotter_mode = False
            # QTabWidget reparents the embedded central widgets away from their
            # QMainWindow wrappers; retain the actual panel for display state.
            window._panel_widget = self.tab_widget.widget(index)
        from data_plotter import DataPlotter
        self.data_plotter = DataPlotter(self)
        self.mode_toolbar = QtWidgets.QToolBar('App mode', self)
        self.mode_toolbar.setMovable(False)
        self.mode_toolbar.addWidget(QtWidgets.QLabel('  Measurement  '))
        self.mode_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mode_slider.setRange(0, 1)
        self.mode_slider.setFixedWidth(58)
        self.mode_slider.setAccessibleName('Measurement or Data plotter mode')
        self.mode_slider.setToolTip('Switch while idle. Data plotter pauses device polling and preserves measurement setup.')
        self.mode_slider.valueChanged.connect(lambda value: self.set_plotter_mode(bool(value)))
        self.mode_toolbar.addWidget(self.mode_slider)
        self.mode_toolbar.addWidget(QtWidgets.QLabel('  Data plotter  '))
        self.mode_status = QtWidgets.QLabel('Measurement controls active')
        self.mode_toolbar.addWidget(self.mode_status)
        self.addToolBar(self.mode_toolbar)
        self._plotter_timer = QtCore.QTimer(self)
        self._plotter_timer.setInterval(25)
        self._plotter_timer.timeout.connect(self._finish_plotter_entry)

        # Connect tab change signal to reload values from value_storage
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Setup polling timer to detect active measurements and prevent tab collisions
        self.lockout_timer = QtCore.QTimer(self)
        self.lockout_timer.setInterval(200) # Check every 200 ms
        self.lockout_timer.timeout.connect(self.check_measurement_status)
        self.lockout_timer.start()

        # ---- Shared Status Bar --------------------------------------------------
        status_bar = self.statusBar()

        # # SMU connection label (mirrors DualGateMainWindow's original label)
        # smu_label = QtWidgets.QLabel()
        # if self.station.smu.simulated:
        #     smu_label.setText("Keysight SMU: SIMULATED (Connection Locked/Failed)  ")
        #     smu_label.setStyleSheet(
        #         "color: #f9e2af; font-weight: bold; font-family: Consolas, monospace;"
        #     )
        # else:
        #     smu_label.setText("Keysight SMU: CONNECTED (Physical)  ")
        #     smu_label.setStyleSheet(
        #         "color: #a6e3a1; font-weight: bold; font-family: Consolas, monospace;"
        #     )
        # status_bar.addPermanentWidget(smu_label)

        # Spectrometer connection status label
        spec_name = "LightField G15"
        location = "G15-Pylon"
        if hasattr(self.station, "camera") and self.station.camera is not None:
            spec_type = getattr(self.station.camera, "spectrometer_type", "lightfieldG15")
            if spec_type in ["wingspecG15", "winspecG15"]:
                spec_name = "WinSpec G15"
                location = "phd-exile-phys"
            elif spec_type in ["winspecG11", "wingspecG11"]:
                spec_name = "WinSpec G11"
                location = "bluefors-pc"
            elif spec_type == "lightfieldG11":
                spec_name = "NIRvana G9"
                remote = self.station.camera.remote_lf
                location = f"{getattr(remote, 'server_ip', '?')}:{getattr(remote, 'port', '?')}"
        connection_label = "CONNECTED"
        if getattr(getattr(self.station, 'camera', None), 'simulated', False):
            connection_label = "SIMULATED"
        elif spec_name == "NIRvana G9":
            connection_label = "CONFIGURED (connection unverified)"
        self.lf_label = lf_label = QtWidgets.QLabel(f"{spec_name}: {connection_label} ({location})  ")
        lf_label.setStyleSheet(
            "color: #059669; font-weight: bold; font-family: Consolas, monospace; font-size: 11px;"
        )
        status_bar.addPermanentWidget(lf_label)
        self.simulation_label = QtWidgets.QLabel()
        self.simulation_label.setStyleSheet("color: #7c3aed; font-weight: bold;")
        status_bar.addWidget(self.simulation_label, 1)
        self.update_device_indicators()
        if self.device_manager is not None:
            devices_button = QtWidgets.QPushButton("Devices / Errors")
            devices_button.clicked.connect(self.show_device_manager)
            status_bar.addPermanentWidget(devices_button)

        # Apply global stylesheet matching Single Spectra theme
        self.apply_theme()
        from gui_safety import install_parameter_wheel_guard
        install_parameter_wheel_guard(self)

    def set_plotter_mode(self, enabled):
        """Switch only when idle; do not issue source or movement commands."""
        if self._plotter_transition or self.data_plotter.busy():
            with QtCore.QSignalBlocker(self.mode_slider):
                self.mode_slider.setValue(int(self.plotter_mode))
            return False
        if enabled == self.plotter_mode:
            return True
        if self.devices_changing or self.measurement_active() or (self.device_manager and self.device_manager.is_busy()):
            with QtCore.QSignalBlocker(self.mode_slider):
                self.mode_slider.setValue(int(self.plotter_mode))
            self.statusBar().showMessage('Finish the active measurement, motion, or device connection before switching modes.', 6000)
            return False
        with QtCore.QSignalBlocker(self.mode_slider):
            self.mode_slider.setValue(int(enabled))
        if enabled:
            self._measurement_tab = self.tab_widget.currentIndex()
            self._previous_polling_paused = getattr(self.station, 'live_status_paused', False)
            self.plotter_mode = True
            self._plotter_transition = True
            for window in self.data_plotter.windows.values():
                window.plotter_mode = True
            self.station.live_status_paused = True
            self.mode_slider.setEnabled(False)
            self.tab_widget.setEnabled(False)
            self.mode_status.setText('Waiting for the current device read to finish…')
            self.position_tab.polling_timer.stop()
            self._plotter_pollers = []
            for window, attribute in ((self.single_window, 'status_worker'),
                                      (self.dual_window, 'live_status_worker'),
                                      (self.position_dgs_window, 'live_status_worker'),
                                      (self.pl_map_window, 'live_status_worker'),
                                      (self.scalar_map_window, 'live_status_worker'),
                                      *[(window, 'live_status_worker') for window in self.gate_map_tabs.values()],
                                      (self.pol_window, 'worker')):
                worker = getattr(window, attribute, None)
                if worker is not None and worker.isRunning():
                    self._plotter_pollers.append(worker)
                    worker.stop()
            self._plotter_timer.start()
            self._finish_plotter_entry()
        else:
            self.data_plotter.leave()
            self.plotter_mode = False
            for window in self.data_plotter.windows.values():
                window.plotter_mode = False
            self.station.live_status_paused = self._previous_polling_paused
            with QtCore.QSignalBlocker(self.tab_widget):
                for index in range(self.tab_widget.count()):
                    self.tab_widget.setTabVisible(index, True)
                    self.tab_widget.setTabEnabled(index, True)
                self.tab_widget.setCurrentIndex(self._measurement_tab)
            self.current_tab_idx = self._measurement_tab
            for worker in self._plotter_pollers:
                if hasattr(worker, '_stop_requested'):
                    worker._stop_requested = False
                if hasattr(worker, '_is_running'):
                    worker._is_running = True
                if hasattr(worker, '_is_paused'):
                    worker._is_paused = True
                worker.start()
            self.mode_status.setText('Measurement controls active')
            self.check_measurement_status()
        return True

    def _finish_plotter_entry(self):
        if not self._plotter_transition:
            return
        if any(worker.isRunning() for worker in self._plotter_pollers):
            return
        self._plotter_timer.stop()
        try:
            self.data_plotter.enter()
        except Exception as exc:
            self._plotter_transition = False
            self.tab_widget.setEnabled(True)
            self.set_plotter_mode(False)
            self.statusBar().showMessage(f'Could not enter Data plotter: {exc}', 10000)
            return
        with QtCore.QSignalBlocker(self.tab_widget):
            for index in range(self.tab_widget.count()):
                self.tab_widget.setTabVisible(index, index in (0, 1, 7))
                self.tab_widget.setTabEnabled(index, index in (0, 1, 7))
            if self.tab_widget.currentIndex() not in (0, 1, 7):
                self.tab_widget.setCurrentIndex(0)
        self._plotter_transition = False
        self.tab_widget.setEnabled(True)
        self.mode_slider.setEnabled(True)
        self.mode_status.setText('Read-only saved data · polling and acquisition off')

    def apply_theme(self):
        """Premium light theme styling matching Single Spectra applied globally to the main dashboard window, tab bar, and unstyled tabs."""
        from theme import SCIENTIFIC_THEME_QSS
        self.setStyleSheet(SCIENTIFIC_THEME_QSS)

    def show_device_manager(self):
        self.device_manager.show()
        self.device_manager.raise_()
        self.device_manager.activateWindow()

    def measurement_active(self):
        # The polarization worker runs continuously for polling, even when idle.
        for name in ("single_window", "dual_window", "bv_window", "pl_map_window",
                     "scalar_map_window", "position_dgs_window", "capacitance_window", "transport_window"):
            window = getattr(self, name, None)
            if window is None:
                continue
            if getattr(window, "is_running", False):
                return True
            for worker_name in ("worker", "sweep_worker", "ramp_worker", "control_worker", "_intermediate_worker"):
                worker = getattr(window, worker_name, None)
                if worker is not None and worker.isRunning():
                    return True
        position = getattr(self, "position_tab", None)
        if position is not None and position.motion_active():
            return True
        if getattr(position, "is_continuous_moving", False):
            return True
        magnet_worker = getattr(position, "magnet_worker", None)
        if magnet_worker is not None and magnet_worker.isRunning():
            return True
        pol_worker = getattr(getattr(self, "pol_window", None), "worker", None)
        if pol_worker is not None and (pol_worker.cmd_queue or getattr(pol_worker, "moving", False)):
            return True
        return False

    def manual_dc_ramp_active(self):
        return any(getattr(window, 'ramp_worker', None) is not None
                   and window.ramp_worker.isRunning()
                   for window in (self.dual_window, self.position_dgs_window))

    def spectrum_navigation_allowed(self, index):
        """A manual DC ramp may continue while the spectrum is viewed."""
        if index not in (0, 1, 7) or not self.manual_dc_ramp_active():
            return False
        owner_index = 1 if (self.dual_window.ramp_worker is not None
                            and self.dual_window.ramp_worker.isRunning()) else 7
        if index not in (0, owner_index):
            return False
        if self.position_tab.motion_active():
            return False
        for window in (self.dual_window, self.bv_window, self.pl_map_window,
                       self.scalar_map_window, self.position_dgs_window, *self.gate_map_tabs.values()):
            if getattr(window, 'is_running', False):
                return False
            for name in ('worker', 'sweep_worker', 'control_worker', '_intermediate_worker'):
                worker = getattr(window, name, None)
                if worker is not None and worker.isRunning():
                    return False
        pol_worker = getattr(self.pol_window, 'worker', None)
        return not (pol_worker and (pol_worker.cmd_queue or getattr(pol_worker, 'moving', False)))

    def update_device_indicators(self):
        from station_devices import DEVICE_LABELS
        simulated = [name for name, device in self.station.components.items()
                     if getattr(device, "simulated", False)]
        self.simulation_label.setText("SIMULATION: " + ", ".join(simulated) if simulated else "Physical devices selected")
        self.simulation_label.setToolTip("\n".join(DEVICE_LABELS.get(name, name) for name in simulated))
        self.tab_widget.setTabToolTip(7, self.simulation_label.text())
        for name, tabs in {"smu": (1, 2, CAPACITANCE_TAB, TRANSPORT_TAB), "camera": (0, 1, 2, 3), "magnet": (2,),
                           "piezo300": (3, 4, 5), "piezo150": (5,), "rotator1": (6,),
                           "rotator2": (6,), "rotator3": (6,), "lockin": (CAPACITANCE_TAB, TRANSPORT_TAB),
                           "lockin2": (TRANSPORT_TAB,), "awg": (CAPACITANCE_TAB,),
                           "dac": (1, CAPACITANCE_TAB, TRANSPORT_TAB)}.items():
            # Full list is always visible, including devices used indirectly.
            for index in tabs:
                self.tab_widget.setTabToolTip(index, self.simulation_label.text())
        camera = self.station.camera
        if camera.simulated:
            self.lf_label.setText("Spectrometer: SIMULATED")
            self.lf_label.setStyleSheet("color: #7c3aed; font-weight: bold;")
        else:
            self.lf_label.setText(f"Spectrometer: {camera.spectrometer_type} (physical)")
            self.lf_label.setStyleSheet("color: #059669; font-weight: bold;")

    def prepare_device_change(self):
        if self.plotter_mode or self.devices_changing or self.measurement_active():
            return False
        self.devices_changing = True
        self.centralWidget().setEnabled(False)
        self.lockout_timer.stop()
        self.position_tab.polling_timer.stop()
        self._device_pollers = []
        for window, attr in ((self.single_window, "status_worker"),
                             (self.dual_window, "live_status_worker"),
                             (self.position_dgs_window, "live_status_worker"),
                             (self.pl_map_window, "live_status_worker"),
                             (self.scalar_map_window, "live_status_worker"),
                             *[(window, "live_status_worker") for window in self.gate_map_tabs.values()],
                             (self.pol_window, "worker")):
            worker = getattr(window, attr, None)
            if worker is not None and worker.isRunning():
                self._device_pollers.append(worker)
                worker.stop()
        if any(not worker.wait(6500) for worker in self._device_pollers):
            # No resources may be replaced while a read is still in progress.
            self._resume_timer = QtCore.QTimer(self)
            self._resume_timer.timeout.connect(self._resume_after_polling)
            self._resume_timer.start(100)
            return False
        return True

    def _resume_after_polling(self):
        if any(worker.isRunning() for worker in self._device_pollers):
            return
        self._resume_timer.stop()
        self.finish_device_change()

    def finish_device_change(self):
        for window in (self.dual_window, self.position_dgs_window):
            window._compliance_readbacks = {}
            window._update_compliance_controls()
        self.position_tab.stepper_controller = self.station.piezo150
        self.position_tab.scanner_controller = self.station.piezo300
        self.position_tab._scanner_command_error = ""
        self.position_tab.sync_scanner_settings()
        self.position_tab.refresh_stepper_coordinates()
        self.position_tab.refresh_scanner_readbacks()
        self.position_dgs_window._scan_origin_loaded = False
        if self.tab_widget.currentIndex() == 7:
            self.position_dgs_window.reload_gui_from_storage()
        self.position_tab.magnet_controller = self.station.magnet
        old_status = getattr(self.single_window, "status_worker", None)
        for worker in getattr(self, "_device_pollers", []):
            if worker is old_status:
                continue
            if hasattr(worker, "_stop_requested"):
                worker._stop_requested = False
            if hasattr(worker, "_is_running"):
                worker._is_running = True
            worker.start()
        self.single_window.reconnect_status_worker()
        self.single_window.live_poll_proxy = None
        # Instruments are looked up at use time; refresh labels and the pollers' outputs.
        for window in self.gate_map_tabs.values():
            window.on_devices_changed()
        self.devices_changing = False
        self.centralWidget().setEnabled(True)
        self.update_device_indicators()
        self.lockout_timer.start()
        self.check_measurement_status()

    def on_tab_changed(self, index):
        if self.plotter_mode:
            self.data_plotter.show_current_file()
            return
        previous = getattr(self, 'current_tab_idx', 0)
        if index != previous and self.measurement_active():
            live_alignment = (getattr(self.single_window, 'is_running', False)
                              and getattr(self.single_window, 'active_acquisition_mode', None) == 'continuous')
            if not ((live_alignment and index in (0, 5, 6)) or self.spectrum_navigation_allowed(index)):
                # Reject competing changes before saving/reloading controls;
                # this also covers the interval before the polling timer fires.
                with QtCore.QSignalBlocker(self.tab_widget):
                    self.tab_widget.setCurrentIndex(previous)
                return
        # Automatically close any open View Control popups
        for win_attr in ("single_window", "dual_window", "bv_window", "pol_window", "pl_map_window", "scalar_map_window", "position_dgs_window"):
            win = getattr(self, win_attr, None)
            if win and hasattr(win, "view_control_popup"):
                try:
                    if win.view_control_popup.isVisible():
                        win.view_control_popup.hide()
                except Exception:
                    pass

        # Save settings of the previous tab before loading the new one
        prev_idx = getattr(self, "current_tab_idx", 0)
        if prev_idx == 0:
            if hasattr(self.single_window, "save_value_storage"):
                try:
                    self.single_window.save_value_storage()
                except Exception as e:
                    print(f"Error saving single spectra settings: {e}")
        elif prev_idx == 1:
            if hasattr(self.dual_window, "save_storage"):
                try:
                    self.dual_window.save_storage()
                except Exception as e:
                    print(f"Error saving dual gate settings: {e}")
        elif prev_idx == 2:
            if hasattr(self.bv_window, "save_storage"):
                try:
                    self.bv_window.save_storage()
                except Exception as e:
                    print(f"Error saving bv sweep settings: {e}")
        elif prev_idx == 3:
            if hasattr(self.pl_map_window, "save_storage"):
                try:
                    self.pl_map_window.save_storage()
                except Exception as e:
                    print(f"Error saving pl map settings: {e}")
        elif prev_idx == 4:
            if hasattr(self.scalar_map_window, "save_storage"):
                try:
                    self.scalar_map_window.save_storage()
                except Exception as e:
                    print(f"Error saving scalar map settings: {e}")
        elif prev_idx == 6:
            if hasattr(self.pol_window, "save_storage"):
                try:
                    self.pol_window.save_storage()
                except Exception as e:
                    print(f"Error saving polarization settings: {e}")
        elif prev_idx in self.gate_map_tabs:
            window = self.gate_map_tabs[prev_idx]
            try:
                window.save_storage()
            except Exception as e:
                print(f"Error saving {window.TAB_NAME.lower()} settings: {e}")

        # Update current tab index
        self.current_tab_idx = index

        # Dynamically reload values from value_storage when a tab is selected
        if index == 0:
            if not self.single_window.is_running and hasattr(self.single_window, "reload_gui_from_storage"):
                self.single_window.reload_gui_from_storage()
        elif index == 1:
            if not self.manual_dc_ramp_active() and hasattr(self.dual_window, "reload_gui_from_storage"):
                self.dual_window.reload_gui_from_storage()
        elif index == 2:
            if hasattr(self.bv_window, "reload_gui_from_storage"):
                self.bv_window.reload_gui_from_storage()
        elif index == 3:
            if hasattr(self.pl_map_window, "reload_gui_from_storage"):
                self.pl_map_window.reload_gui_from_storage()
        elif index == 4:
            if hasattr(self.scalar_map_window, "reload_gui_from_storage"):
                self.scalar_map_window.reload_gui_from_storage()
        elif index == 5:
            if hasattr(self.position_tab, "reload_gui_from_storage"):
                self.position_tab.reload_gui_from_storage()
        elif index == 6:
            if hasattr(self.pol_window, "reload_gui_from_storage"):
                self.pol_window.reload_gui_from_storage()
        elif index == 7:
            if not self.manual_dc_ramp_active():
                self.position_dgs_window.reload_gui_from_storage()
        elif index in self.gate_map_tabs:
            self.gate_map_tabs[index].reload_gui_from_storage()

        # Trigger immediate status worker updates based on the new active tab
        self.check_measurement_status()

    def check_measurement_status(self):
        if self.plotter_mode:
            return
        self.mode_slider.setEnabled(not self.devices_changing and not self.measurement_active())
        camera = getattr(self.station, "camera", None)
        if camera is not None and not camera.simulated and camera.spectrometer_type == "lightfieldG11":
            health = camera.remote_lf.acquisition_health
            state = health["state"]
            colors = {"failed": "#b91c1c", "verified": "#047857",
                      "checking": "#1d4ed8", "acquiring": "#1d4ed8", "unverified": "#92400e"}
            if hasattr(self, "lf_label"):
                self.lf_label.setText(f"NIRvana G9: {state.upper()} | Last valid spectrum: {health['last_success'] or 'none'}  ")
                self.lf_label.setToolTip(health["details"])
                self.lf_label.setStyleSheet(f"color: {colors[state]}; font-weight: bold;")
        # 1. Determine if any measurement is active
        meas_running = bool(getattr(self.single_window, "grating_setting", False)) or self.measurement_active()
        meas_running = meas_running or self.position_dgs_window.is_running
        
        # Single Spectra
        if getattr(self.single_window, "is_running", False):
            meas_running = True
            
        # Dual-Gate Sweep
        if getattr(self.dual_window, "is_running", False):
            meas_running = True
            
        # B/V Sweep
        if getattr(self.bv_window, "is_running", False):
            meas_running = True
            
        # Vector Map
        if getattr(self.pl_map_window, "is_running", False):
            meas_running = True

        # Scalar Map
        if getattr(self.scalar_map_window, "is_running", False):
            meas_running = True

        # Gate-map tabs: capacitance bridge and transport (sweeps, balances, AWG, manual electrode ramps)
        if any(window.busy() for window in self.gate_map_tabs.values()):
            meas_running = True

        # 2. Control tab bar interaction
        current_idx = self.tab_widget.currentIndex()
        live_alignment = (getattr(self.single_window, "is_running", False)
                          and getattr(self.single_window, "active_acquisition_mode", None) == "continuous")
        for i in range(self.tab_widget.count()):
            if self.spectrum_navigation_allowed(i):
                self.tab_widget.setTabEnabled(i, True)
            elif live_alignment:
                # Keep the focus/polarization controls and the Live stop button
                # accessible, while preventing competing measurement sweeps.
                self.tab_widget.setTabEnabled(i, i in (0, 5, 6))
            elif meas_running:
                # Disable all tabs except the currently active one
                if i != current_idx:
                    self.tab_widget.setTabEnabled(i, False)
            else:
                self.tab_widget.setTabEnabled(i, True)

        # 3. Control background status polling
        # Single Spectra Status Worker (active when index == 0 and no measurement is running)
        single_worker = getattr(self.single_window, "status_worker", None)
        if single_worker:
            should_run = (not meas_running) and (current_idx == 0)
            if should_run:
                if getattr(single_worker, "_is_paused", False):
                    single_worker.resume()
            else:
                if not getattr(single_worker, "_is_paused", False):
                    single_worker.pause()

        # Dual-Gate Sweep Live Status Worker (active when index == 1 and no measurement is running)
        dual_worker = getattr(self.dual_window, "live_status_worker", None)
        if dual_worker:
            should_run = (not meas_running) and (current_idx == 1)
            if should_run:
                if getattr(dual_worker, "_is_paused", False):
                    dual_worker.resume()
            else:
                if not getattr(dual_worker, "_is_paused", False):
                    dual_worker.pause()

        # Vector Map Live Status Worker (active when index == 3 and no measurement is running)
        position_dgs_worker = self.position_dgs_window.live_status_worker
        if not meas_running and current_idx == 7:
            position_dgs_worker.resume()
        else:
            position_dgs_worker.pause()

        pl_worker = getattr(self.pl_map_window, "live_status_worker", None)
        if pl_worker:
            should_run = (not meas_running) and (current_idx == 3)
            if should_run:
                if getattr(pl_worker, "_is_paused", False):
                    pl_worker.resume()
            else:
                if not getattr(pl_worker, "_is_paused", False):
                    pl_worker.pause()

        # Scalar Map Live Status Worker (active when index == 4 and no measurement is running)
        scalar_worker = getattr(self.scalar_map_window, "live_status_worker", None)
        if scalar_worker:
            should_run = (not meas_running) and (current_idx == 4)
            if should_run:
                if getattr(scalar_worker, "_is_paused", False):
                    scalar_worker.resume()
            else:
                if not getattr(scalar_worker, "_is_paused", False):
                    scalar_worker.pause()

        # Gate-map Live Status Workers (lock-in and electrode readout while idle on their tab)
        for tab_index, window in self.gate_map_tabs.items():
            gate_map_worker = getattr(window, "live_status_worker", None)
            if gate_map_worker:
                should_run = (not meas_running) and (current_idx == tab_index)
                if should_run:
                    if getattr(gate_map_worker, "_is_paused", False):
                        gate_map_worker.resume()
                else:
                    if not getattr(gate_map_worker, "_is_paused", False):
                        gate_map_worker.pause()

        # Dashboard Polling Timer (active when index == 5 and no measurement is running)
        if hasattr(self, 'position_tab') and self.position_tab:
            should_run = (not meas_running or live_alignment) and (current_idx == 5)
            if should_run:
                if not self.position_tab.polling_timer.isActive():
                    self.position_tab.polling_timer.start(1000)
            else:
                if self.position_tab.polling_timer.isActive():
                    self.position_tab.polling_timer.stop()

    # Ensure clean shutdown when the outer window is closed
    def closeEvent(self, event):
        if self.position_tab.motion_active():
            if self.position_tab.scanner_worker:
                self.position_tab.scanner_worker.stop()
            self.position_tab.stop_stepper_motion()
            event.ignore()
            self.statusBar().showMessage('Stopping scanner / stepper motion before closing...')
            QtCore.QTimer.singleShot(100, self.close)
            return
        if self.data_plotter.busy() or self._plotter_transition:
            event.ignore()
            self.statusBar().showMessage('Wait for file loading or the mode transition to finish before closing.', 5000)
            return
        control_worker = getattr(self.single_window, "control_worker", None)
        if control_worker is not None and control_worker.isRunning():
            event.ignore()
            return
        if self.device_manager is not None and (self.device_manager.is_busy() or self.devices_changing):
            event.ignore()
            self.show_device_manager()
            return
        self.single_window.live_poll_timer.stop()
        manual_workers = [getattr(window, 'ramp_worker', None)
                          for window in (self.dual_window, self.position_dgs_window, self.bv_window,
                                         self.pl_map_window, self.scalar_map_window)]
        running_ramps = [worker for worker in manual_workers if worker is not None and worker.isRunning()]
        intermediate = getattr(self.single_window, '_intermediate_worker', None)
        if running_ramps or (intermediate is not None and intermediate.isRunning()):
            for worker in running_ramps:
                worker.stop()
            # Keep owners alive until the current instrument call unwinds;
            # a GUI-thread wait here would freeze Stop and status updates.
            event.ignore()
            self.statusBar().showMessage('Stopping the active ramp or finishing the current preview read before closing…')
            QtCore.QTimer.singleShot(100, self.close)
            return
        gate_map_workers = [(window, worker) for window in self.gate_map_tabs.values()
                            for worker in (window.sweep_worker, window.ramp_worker, window.control_worker)
                            if worker is not None and worker.isRunning()]
        if gate_map_workers:
            for _window, worker in gate_map_workers:
                worker.stop()
            # The worker ends after its current instrument call, autosaves partial
            # data and exits; waiting here would freeze the GUI thread.
            event.ignore()
            names = sorted({window.TAB_NAME.lower() for window, _worker in gate_map_workers})
            self.statusBar().showMessage(f"Stopping the {' and '.join(names)} measurement before closing…")
            QtCore.QTimer.singleShot(100, self.close)
            return
        self.lockout_timer.stop()
        if hasattr(self.position_tab, "polling_timer"):
            self.position_tab.polling_timer.stop()
        position_worker = self.position_dgs_window.sweep_worker
        if position_worker is not None and position_worker.isRunning():
            position_worker.stop()
        acquisition_workers = (
            getattr(self.position_dgs_window, "sweep_worker", None),
            getattr(self.single_window, "worker", None),
            getattr(self.dual_window, "sweep_worker", None),
            getattr(self.bv_window, "sweep_worker", None),
            getattr(self.pl_map_window, "sweep_worker", None),
        )
        if any(worker is not None and worker.isRunning() for worker in acquisition_workers):
            try:
                self.station.camera.abort_acquisition()
            except Exception:
                pass
        for window in (self.dual_window, self.pl_map_window, self.scalar_map_window, self.position_dgs_window):
            status_worker = getattr(window, "live_status_worker", None)
            if status_worker is not None and status_worker.isRunning():
                status_worker.stop()
                status_worker.wait()
        # Let the embedded windows clean up their workers if needed
        position_worker = self.position_dgs_window.sweep_worker
        if position_worker is not None and position_worker.isRunning():
            position_worker.stop()
            position_worker.wait()
        try:
            if hasattr(self.single_window, "status_worker") and self.single_window.status_worker.isRunning():
                self.single_window.status_worker.stop()
                self.single_window.status_worker.wait()
        except Exception:
            pass
        try:
            if getattr(self.single_window, "worker", None) and self.single_window.worker.isRunning():
                self.single_window.worker.stop()
                self.single_window.worker.wait()
        except Exception:
            pass
        try:
            if getattr(self.dual_window, "sweep_worker", None) and self.dual_window.sweep_worker.isRunning():
                self.dual_window.sweep_worker.stop()
                self.dual_window.sweep_worker.wait()
        except Exception:
            pass
        try:
            if getattr(self.bv_window, "sweep_worker", None) and self.bv_window.sweep_worker.isRunning():
                self.bv_window.sweep_worker.stop()
                self.bv_window.sweep_worker.wait()
        except Exception:
            pass
        try:
            if getattr(self.pol_window, "worker", None) and self.pol_window.worker.isRunning():
                self.pol_window.worker.stop()
                self.pol_window.worker.wait()
        except Exception:
            pass
        try:
            if getattr(self.pl_map_window, "sweep_worker", None) and self.pl_map_window.sweep_worker.isRunning():
                self.pl_map_window.sweep_worker.stop()
                self.pl_map_window.sweep_worker.wait()
        except Exception:
            pass
        try:
            if getattr(self.scalar_map_window, "sweep_worker", None) and self.scalar_map_window.sweep_worker.isRunning():
                self.scalar_map_window.sweep_worker.stop()
                self.scalar_map_window.sweep_worker.wait()
        except Exception:
            pass
        magnet_worker = getattr(self.position_tab, "magnet_worker", None)
        if magnet_worker is not None and magnet_worker.isRunning():
            magnet_worker.stop()
            magnet_worker.wait()
        for window in self.gate_map_tabs.values():
            try:
                # Stops its live poller and saves its settings; its workers already ended above.
                window.shutdown()
            except Exception:
                pass
        if self.plotter_mode:
            # Restore the measurement forms before the usual close-time saves.
            # Keep plotter guards active; shutdown must not restart polling.
            self.data_plotter.leave()
            for window in self.data_plotter.windows.values():
                window.plotter_mode = True
        self.data_plotter.shutdown()
        from gui_safety import uninstall_parameter_wheel_guard
        # Embedded DGS wrappers retain their own filters even after their
        # central widgets are reparented into the dashboard's tabs.
        for window in (self.dual_window, self.position_dgs_window, self):
            uninstall_parameter_wheel_guard(window)
        # Save any persisted settings
        try:
            self.single_window.save_value_storage()
        except Exception:
            pass
        try:
            self.dual_window.save_storage()
        except Exception:
            pass
        try:
            self.bv_window.save_storage()
        except Exception:
            pass
        try:
            self.pol_window.save_storage()
        except Exception:
            pass
        try:
            self.pl_map_window.save_storage()
        except Exception:
            pass
        try:
            self.scalar_map_window.save_storage()
        except Exception:
            pass
        try:
            self.position_tab.save_storage()
        except Exception:
            pass
        if self.device_manager is not None:
            self.device_manager.close()
        super().closeEvent(event)


def main():
    """Run one device manager and dashboard process without automatic restart."""
    import traceback

    from device_manager import DeviceManagerWindow, GuiLogStream
    from single_instance import acquire_lock
    from station_devices import create_station

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("G15 QCoDeS Lab Dashboard")
    app.setQuitOnLastWindowClosed(True)
    lock, acquired = acquire_lock(current_dir)
    if not acquired:
        if lock.error() == QtCore.QLockFile.LockError.LockFailedError:
            message = "The G15 lab dashboard is already open. Its devices were not initialized again."
        else:
            message = f"Could not secure the lab dashboard's single-instance lock: {lock.error()}"
        QtWidgets.QMessageBox.information(None, "G15 Lab Dashboard", message)
        return 0 if lock.error() == QtCore.QLockFile.LockError.LockFailedError else 1

    station = None
    try:
        station = create_station(os.path.join(current_dir, "my_measurement_data.db"))
        manager = None

        def open_dashboard():
            try:
                dashboard = manager.dashboard
                if dashboard is None:
                    dashboard = LabMainWindow(station, device_manager=manager)
                    manager.dashboard = dashboard
                    manager.dashboard_button.setText("Return to Measurement Dashboard")
                dashboard.show()
                dashboard.raise_()
                dashboard.activateWindow()
            except Exception as exc:
                manager.details.setPlainText(traceback.format_exc())
                QtWidgets.QMessageBox.critical(manager, "Dashboard startup failed", str(exc))

        manager = DeviceManagerWindow(station, open_dashboard)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        stdout_stream = GuiLogStream(original_stdout, manager)
        stderr_stream = GuiLogStream(original_stderr, manager)
        stdout_stream.text_available.connect(manager.append_log)
        stderr_stream.text_available.connect(manager.append_log)
        sys.stdout, sys.stderr = stdout_stream, stderr_stream
        manager.fill_missing_devices()
        manager.dashboard = LabMainWindow(station, device_manager=manager)
        manager.show()
        try:
            return app.exec()
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "Station startup failed", f"{exc}\n\n{traceback.format_exc()}")
        return 1
    finally:
        if station is not None:
            station_setup.Instrument.close_all()
        lock.unlock()


if __name__ == "__main__":
    sys.exit(main())
