"""Device checks and reconnection for a persistent measurement dashboard."""

import contextlib
import io
import traceback
import warnings
from datetime import datetime

from PyQt6 import QtCore, QtGui, QtWidgets

import station_setup
from station_devices import DEVICE_LABELS, DEVICE_NAMES, configure_device, ensure_simulated_devices
from drivers.remote_client import NIRvanaSocketWrapper
from device_manager_theme import COLORS, DEVICE_MANAGER_QSS


class StatusDelegate(QtWidgets.QStyledItemDelegate):
    """Keep the status color legible even when its row is selected."""

    def paint(self, painter, option, index):
        foreground = index.data(QtCore.Qt.ItemDataRole.ForegroundRole)
        if foreground is not None:
            option.palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, foreground.color())
        super().paint(painter, option, index)


class GuiLogStream(QtCore.QObject):
    """Forward ordinary Python output to the application's log pane."""

    text_available = QtCore.pyqtSignal(str)

    def __init__(self, original, parent=None):
        super().__init__(parent)
        self.original = original

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def write(self, text):
        if text:
            self.text_available.emit(str(text))
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


class InitializationWorker(QtCore.QThread):
    device_started = QtCore.pyqtSignal(str)
    device_finished = QtCore.pyqtSignal(str, bool, str, str)

    def __init__(self, station, requests, parent=None):
        super().__init__(parent)
        self.station = station
        self.requests = requests

    def run(self):
        for name, mode in self.requests:
            self.device_started.emit(name)
            output = io.StringIO()
            status = "FAILED"
            success = False
            error = ""
            with warnings.catch_warnings(record=True) as captured_warnings:
                try:
                    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                        status = configure_device(self.station, name, mode)
                    success = True
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            warning_text = "\n".join(str(item.message) for item in captured_warnings)
            details = "\n".join(
                part for part in (output.getvalue().strip(), warning_text, error) if part
            )
            self.device_finished.emit(name, success, status, details)


class ConnectionScanWorker(QtCore.QThread):
    scanned = QtCore.pyqtSignal(dict)

    def __init__(self, station, active, parent=None):
        super().__init__(parent)
        self.station, self.active = station, active

    def run(self):
        from device_discovery import scan_attocubes, occupied_serial_ports
        try:
            report = scan_attocubes(skip_ports=occupied_serial_ports(self.station),
                                    identify=not self.active)
            report['readbacks'] = {}
            for name in ('piezo150', 'piezo300'):
                controller = self.station.components.get(name)
                if controller is None or controller.simulated:
                    continue
                lock = getattr(controller, 'comm_lock', None)
                if lock is None or not lock.acquire(blocking=False):
                    continue
                try:
                    if name == 'piezo300':
                        values = [controller.get_axis_status(axis) for axis in (1, 2)]
                        detail = '\n'.join(f"Axis {axis}: offset {value['offset']:.3f} V, output {value['output']:.3f} V, mode {value['mode']}"
                                           for axis, value in enumerate(values, 1))
                    elif not self.active:
                        detail = controller.ask('getv 1')
                        if 'not in computer control mode' in detail.lower():
                            report['readbacks'][name] = (True, 'MANUAL (enable computer control)',
                                                        'ANC150 is connected. Enable computer control on its front panel to use it remotely. No modes or outputs were changed.')
                            continue
                        if 'voltage' not in detail.lower() or 'error' in detail.lower():
                            raise RuntimeError(detail)
                    else:
                        continue
                    report['readbacks'][name] = (True, 'VERIFIED (readback)', detail)
                except Exception as exc:
                    report['readbacks'][name] = (False, 'READBACK FAILED', str(exc))
                finally:
                    lock.release()
            self.scanned.emit(report)
        except Exception as exc:
            self.scanned.emit({'error': str(exc)})


class DeviceManagerWindow(QtWidgets.QMainWindow):
    """Inspect and reconnect individual devices while the dashboard is idle."""

    def __init__(self, station, open_dashboard, parent=None):
        super().__init__(parent)
        self.station = station
        self._open_dashboard_callback = open_dashboard
        self.worker = None
        self.scan_worker = None
        self.results = {}
        self.rows = {}
        self.dashboard = None
        self._last_health = None
        self.setWindowTitle("G15 Lab Device Manager")
        self.resize(1120, 940)
        self.setMinimumSize(1000, 780)
        self.setStyleSheet(DEVICE_MANAGER_QSS)

        body = QtWidgets.QWidget(self)
        body.setObjectName("managerBody")
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        title = QtWidgets.QLabel("Lab device manager")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QtWidgets.QLabel(
            "Scan connections at any time. Connect or retry repaired devices while idle; the same measurement dashboard is refreshed automatically."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("summary")
        layout.addWidget(self.summary)
        self.spectrometer_info = QtWidgets.QLabel(
            "Open the dashboard at any time. Unchecked devices use simulation until you connect them. NIRvana checks acquire two 50 ms test spectra."
        )
        self.spectrometer_info.setObjectName("spectrometer")
        self.spectrometer_info.setWordWrap(True)
        layout.addWidget(self.spectrometer_info)
        self.table = QtWidgets.QTableWidget(len(DEVICE_NAMES), 4)
        self.table.setHorizontalHeaderLabels(("Device", "Mode", "Status", "Action"))
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(43)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(340)
        self.table.setItemDelegateForColumn(2, StatusDelegate(self.table))
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(3, 160)
        layout.addWidget(self.table, 3)

        for row, name in enumerate(DEVICE_NAMES):
            self.rows[name] = row
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(DEVICE_LABELS[name]))
            mode = QtWidgets.QComboBox()
            mode.addItems(("Auto", "Physical", "Simulation"))
            mode.setToolTip("Auto: physical first, simulation on failure. Physical: report failure without fallback.")
            self.table.setCellWidget(row, 1, mode)
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem("Not initialized"))
            self.set_device_status(name, "Not initialized", "idle")
            button = QtWidgets.QPushButton("Connect / Check")
            button.clicked.connect(
                lambda checked=False, device=name: self.initialize_devices((device,), recheck=True)
            )
            self.table.setCellWidget(row, 3, button)

        self.log_tabs = QtWidgets.QTabWidget()
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select a device to see its initialization output or error.")
        self.log_tabs.addTab(self.details, "Device details")
        self.table.itemSelectionChanged.connect(self.show_selected_details)

        self.application_log = QtWidgets.QPlainTextEdit()
        self.application_log.setReadOnly(True)
        self.application_log.document().setMaximumBlockCount(1000)
        self.log_tabs.addTab(self.application_log, "Application log")
        layout.addWidget(self.log_tabs, 1)

        self.readiness = QtWidgets.QLabel()
        self.readiness.setObjectName("footnote")
        self.readiness.setWordWrap(True)
        layout.addWidget(self.readiness)

        buttons = QtWidgets.QHBoxLayout()
        self.scan_button = QtWidgets.QPushButton("Scan connections")
        self.scan_button.setToolTip("Read-only port discovery and available ANC readbacks. No acquisition, voltage change or reconnection.")
        self.scan_button.clicked.connect(self.scan_connections)
        buttons.addWidget(self.scan_button)
        self.all_button = QtWidgets.QPushButton("Check All Devices")
        self.all_button.clicked.connect(
            lambda: self.initialize_devices(DEVICE_NAMES)
        )
        buttons.addWidget(self.all_button)
        buttons.addStretch()
        self.dashboard_button = QtWidgets.QPushButton("Open Measurement Dashboard")
        self.dashboard_button.setObjectName("primary")
        self.dashboard_button.setEnabled(True)
        self.dashboard_button.clicked.connect(self.open_dashboard)
        buttons.addWidget(self.dashboard_button)
        layout.addLayout(buttons)
        self.setCentralWidget(body)
        self.health_timer = QtCore.QTimer(self)
        self.health_timer.setInterval(300)
        self.health_timer.timeout.connect(self.refresh_health)
        self.health_timer.start()
        self.update_summary()

    def is_busy(self):
        return ((self.worker is not None and self.worker.isRunning())
                or (self.scan_worker is not None and self.scan_worker.isRunning()))

    def scan_connections(self):
        if self.is_busy():
            return
        active = self.dashboard is not None and self.dashboard.measurement_active()
        self.scan_button.setEnabled(False)
        self.scan_worker = ConnectionScanWorker(self.station, active, self)
        self.scan_worker.scanned.connect(self.on_connections_scanned)
        self.scan_worker.finished.connect(lambda: self.scan_button.setEnabled(True))
        self.scan_worker.start()
        self.readiness.setText("Reading connections; measurements continue. No devices are being replaced.")

    def on_connections_scanned(self, report):
        if 'error' in report:
            text = 'Connection scan failed: ' + report['error']
        else:
            lines = ['Connection scan ' + datetime.now().astimezone().isoformat(timespec='seconds')]
            lines.extend(f"{item['port']}: {item['description']}" for item in report['ports'])
            lines.extend(f"Identified {item['model']} on {item['port']}" for item in report['controllers'])
            lines.extend(f"{port}: unavailable for probing ({error})" for port, error in report['errors'].items())
            if not report['identity_probed']:
                lines.append('Measurement active: new-port identity probes deferred. Existing ANC300 readback and port inventory only.')
            for name, result in report['readbacks'].items():
                self.on_device_finished(name, *result)
            lines.append('Use Connect / Retry while idle to switch a repaired device from simulation to hardware. The dashboard and its data are retained.')
            text = '\n'.join(lines)
        self.append_log('\n' + text + '\n')
        self.log_tabs.setCurrentIndex(1)
        self.readiness.setText('Connection scan complete. Reconnect individual devices while idle; no app restart is needed.')
        self._last_health = None

    def is_nirvana(self):
        camera = getattr(self.station, "components", {}).get("camera")
        return isinstance(getattr(camera, "remote_lf", None), NIRvanaSocketWrapper)

    def initialize_devices(self, names, *, recheck=False):
        if self.is_busy():
            return
        if self.dashboard is not None and self.dashboard.measurement_active():
            self.readiness.setText("Finish or stop the active measurement before rechecking devices.")
            return
        requests = []
        for name in names:
            row = self.rows[name]
            requests.append((name, self.table.cellWidget(row, 1).currentText()))
        if not requests:
            return
        if self.dashboard is not None and not self.dashboard.prepare_device_change():
            self.readiness.setText("A device is still busy. Wait for polling or movement to finish, then retry.")
            return
        self.all_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.dashboard_button.setEnabled(self.dashboard is not None)
        for row in range(self.table.rowCount()):
            self.table.cellWidget(row, 3).setEnabled(False)
            self.table.cellWidget(row, 1).setEnabled(False)
        self.worker = InitializationWorker(self.station, requests, self)
        self.worker.device_started.connect(self.on_device_started)
        self.worker.device_finished.connect(self.on_device_finished)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.readiness.setText(
            "Checking devices. NIRvana needs two complete spectra; CSV export can take up to 180 s per check."
        )

    def on_device_started(self, name):
        status = "Checking two spectra..." if name == "camera" and self.table.cellWidget(self.rows[name], 1).currentText() != "Simulation" and station_setup.SPECTROMETER_TYPE == "lightfieldG11" else "Initializing..."
        self.set_device_status(name, status, "busy")
        self.table.selectRow(self.rows[name])

    def on_device_finished(self, name, success, status, details):
        self.results[name] = (success, status, details)
        row = self.rows[name]
        self.set_device_status(name, status, self.status_kind(success, status))
        self.table.item(row, 2).setToolTip(details)
        self.table.selectRow(row)
        self.show_selected_details()
        if name == "camera":
            self.spectrometer_info.setText(f"Spectrometer: {status}\n{details.splitlines()[0] if details else 'See device details below.'}")
        self.append_log(f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] {DEVICE_LABELS[name]}: {status}\n{details}\n")
        self.update_summary()

    def on_worker_finished(self):
        if self.dashboard is not None and getattr(self.dashboard, "devices_changing", False):
            # Keep a complete station even after a strict Physical-mode failure.
            self.fill_missing_devices()
            self.dashboard.finish_device_change()
        self.all_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        for name, row in self.rows.items():
            self.table.cellWidget(row, 1).setEnabled(True)
            self.table.cellWidget(row, 3).setEnabled(True)
            self.table.cellWidget(row, 3).setText("Connect / Retry")
        self.dashboard_button.setEnabled(True)
        self.update_summary()

    def fill_missing_devices(self):
        missing = [name for name in DEVICE_NAMES if name not in self.station.components]
        ensure_simulated_devices(self.station)
        for name in missing:
            previous = self.results.get(name)
            details = previous[2] if previous else "Physical connection has not been checked."
            self.on_device_finished(name, True, "SIMULATED (not connected)", details)

    def open_dashboard(self):
        if self.is_busy() and self.dashboard is None:
            return
        if self.dashboard is None:
            self.fill_missing_devices()
        self._open_dashboard_callback()

    @staticmethod
    def status_kind(success, status):
        if not success:
            return "failed"
        if status.startswith("SIMULATED"):
            return "simulated"
        if "CONFIGURED" in status or "LOCAL" in status or status.startswith("MANUAL"):
            return "limited"
        return "ready"

    def set_device_status(self, name, text, kind):
        item = self.table.item(self.rows[name], 2)
        item.setText(text)
        item.setForeground(QtGui.QColor(COLORS[kind]))
        font = item.font()
        font.setBold(True)
        item.setFont(font)

    def update_summary(self):
        counts = {key: 0 for key in COLORS}
        for name in DEVICE_NAMES:
            result = self.results.get(name)
            counts[self.status_kind(*result[:2]) if result else "idle"] += 1
        labels = {"ready": "Initialized / verified", "failed": "Failed", "simulated": "Simulated",
                  "limited": "Limited / unverified", "idle": "Pending"}
        self.summary.setText(" &nbsp; &nbsp; ".join(
            f'<span style="color:{COLORS[kind]}"><b>{counts[kind]}</b> {label}</span>'
            for kind, label in labels.items()
        ))
        if self.is_busy():
            message = "Checking devices. NIRvana export may take up to 180 s per check."
        else:
            message = "Dashboard available. Purple = simulation; green = verified readback. Retry devices while measurements are idle."
        self.readiness.setText(message)

    def refresh_health(self):
        # Read local evidence only. The server has no ping/status command.
        if not self.is_nirvana():
            return
        health = self.station.components["camera"].remote_lf.acquisition_health
        if health is self._last_health:
            return
        self._last_health = health
        remote = self.station.components["camera"].remote_lf
        state = health["state"]
        last = health["last_success"] or "none this session"
        self.spectrometer_info.setText(
            f"NIRvana G9  |  {remote.server_ip}:{remote.port}  |  {state.upper()}\n"
            f"Last valid spectrum: {last}. Status is based on acquisitions, not a live heartbeat."
        )
        if self.is_busy():
            return  # The initialization worker reports the combined two-test result.
        if state in ("verified", "failed"):
            if state == "verified":
                status = "VERIFIED (2/2 spectra)" if health["details"].startswith("Two consecutive") else "SPECTRUM RECEIVED"
            else:
                status = "ACQUISITION FAILED"
            self.results["camera"] = (state == "verified", status, health["details"])
            self.set_device_status("camera", status, "ready" if state == "verified" else "failed")
            self.table.item(self.rows["camera"], 2).setToolTip(health["details"])
            if state == "failed":
                self.append_log(f"\nNIRvana acquisition failed: {health['details']}\n")
                self.table.selectRow(self.rows["camera"])
                self.log_tabs.setCurrentIndex(0)
            self.show_selected_details()
            self.on_worker_finished()
        elif state in ("acquiring", "checking"):
            self.set_device_status("camera", state.upper(), "busy")

    def show_selected_details(self):
        row = self.table.currentRow()
        if row < 0:
            return
        name = DEVICE_NAMES[row]
        self.details.setPlainText(self.results.get(name, (False, "", ""))[2])

    def append_log(self, text):
        cursor = self.application_log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.application_log.setTextCursor(cursor)

    def closeEvent(self, event):
        if self.is_busy():
            event.ignore()
            QtWidgets.QMessageBox.information(
                self, "Device check in progress",
                "Wait for the current device check to finish before closing."
            )
            return
        if self.dashboard is not None and not self.dashboard.isVisible() and not getattr(self, "_closing", False):
            self._closing = True
            self.dashboard.close()
            self._closing = False
        super().closeEvent(event)


def main():
    """Use the shared station launcher for both manager and dashboard."""
    from lab_gui import main as launch
    return launch()


if __name__ == '__main__':
    import sys
    sys.exit(main())
