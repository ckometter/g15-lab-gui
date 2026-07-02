import sys
import os

# Ensure parent directory and measurement sub-package directory are in path
# The structure assumes this script is inside a package directory at the same level as 'measurements'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PyQt6 import QtWidgets, QtGui, QtCore

# Ensure parent directory is in path for imports
import sys, os
parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_root = os.path.dirname(parent_dir)
if parent_root not in sys.path:
    sys.path.insert(0, parent_root)

import station_setup
from measurements.single_spectra_gui import SingleSpectraWindow
from measurements.Dual_gate_sweep_gui import MainWindow as DualGateMainWindow
from measurements.B_V_sweep_gui import BVSweepWindow
from measurements.polarization_control_gui import PolarizationControlWindow
from measurements.photoluminescence_map_gui import PhotoluminescenceMapWindow
from measurements.scalar_map_gui import ScalarMapWindow
from measurements.position_tab import PositionTab
initialize_station = station_setup.initialize_station


class LabMainWindow(QtWidgets.QMainWindow):
    """Top‑level window that hosts the two measurement GUIs as left‑side tabs.

    * The tab bar is placed on the **West** side (vertical tabs on the left).
    * Each sub‑GUI is instantiated with ``embed=True`` so that its own status
      bar is suppressed – the outer ``LabMainWindow`` provides a shared status bar.
    """

    def __init__(self, station):
        super().__init__()
        self.station = station
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
                spec_name = "Lightfield G11"
                location = "10.236.10.143"
        lf_label = QtWidgets.QLabel(f"{spec_name}: CONNECTED ({location})  ")
        lf_label.setStyleSheet(
            "color: #10b981; font-weight: bold; font-family: Consolas, monospace;"
        )
        status_bar.addPermanentWidget(lf_label)

        # Apply global stylesheet matching Single Spectra theme
        self.apply_theme()

    def apply_theme(self):
        """Premium light theme styling matching Single Spectra applied globally to the main dashboard window, tab bar, and unstyled tabs."""
        global_qss = """
        QMainWindow, QWidget {
            background-color: #ffffff;
            color: #000000;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            border-radius: 0px;
            background-color: #ffffff;
            padding: 5px;
        }
        QTabBar::tab {
            background-color: #f5f5f5;
            color: #555555;
            border: 1px solid #cccccc;
            border-right-color: none;
            border-top-left-radius: 0px;
            border-bottom-left-radius: 0px;
            padding: 12px 16px;
            margin-bottom: 2px;
            font-weight: bold;
            font-size: 12px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            color: #000000;
            border-right-color: transparent;
            font-weight: bold;
        }
        QTabBar::tab:hover {
            background-color: #eeeeee;
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
        QScrollBar:vertical {
            border: none;
            background: #f5f5f5;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #cccccc;
            min-height: 20px;
            border-radius: 0px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """
        self.setStyleSheet(global_qss)

    def on_tab_changed(self, index):
        # Automatically close any open View Control popups
        for win_attr in ("single_window", "dual_window", "bv_window", "pol_window", "pl_map_window", "scalar_map_window"):
            win = getattr(self, win_attr, None)
            if win and hasattr(win, "view_control_popup"):
                try:
                    if win.view_control_popup.isVisible():
                        win.view_control_popup.hide()
                except Exception:
                    pass

        # Dynamically reload values from value_storage when a tab is selected
        if index == 0:
            if hasattr(self.single_window, "reload_gui_from_storage"):
                self.single_window.reload_gui_from_storage()
        elif index == 1:
            if hasattr(self.dual_window, "reload_gui_from_storage"):
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

        # Trigger immediate status worker updates based on the new active tab
        self.check_measurement_status()

    def check_measurement_status(self):
        # 1. Determine if any measurement is active
        meas_running = False
        
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

        # 2. Control tab bar interaction
        current_idx = self.tab_widget.currentIndex()
        for i in range(self.tab_widget.count()):
            if meas_running:
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

        # Dashboard Polling Timer (active when index == 5 and no measurement is running)
        if hasattr(self, 'position_tab') and self.position_tab:
            should_run = (not meas_running) and (current_idx == 5)
            if should_run:
                if not self.position_tab.polling_timer.isActive():
                    self.position_tab.polling_timer.start(1000)
            else:
                if self.position_tab.polling_timer.isActive():
                    self.position_tab.polling_timer.stop()

    # Ensure clean shutdown when the outer window is closed
    def closeEvent(self, event):
        # Let the embedded windows clean up their workers if needed
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
        super().closeEvent(event)


if __name__ == "__main__":
    # ----- Station configuration -----
    print("Initializing QCoDeS station for unified Lab GUI...")
    station = initialize_station(simulated=False, simulated_devices=station_setup.simulated_devices)

    app = QtWidgets.QApplication(sys.argv)
    win = LabMainWindow(station)
    win.show()
    sys.exit(app.exec())
