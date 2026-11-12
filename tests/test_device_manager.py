"""Launcher checks run without opening any laboratory hardware."""

import os
import tempfile
import unittest
import threading
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from qcodes.station import Station

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from device_manager import DeviceManagerWindow
from single_instance import acquire_lock
from station_devices import DEVICE_NAMES
from drivers.remote_client import NIRvanaSocketWrapper, ClientLightFieldCamera


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_health_timer_tolerates_a_closed_camera_during_replacement(self):
        station = SimpleNamespace(components={'camera': SimpleNamespace()})
        self.assertFalse(DeviceManagerWindow.is_nirvana(SimpleNamespace(station=station)))

    def test_second_launcher_cannot_take_the_same_lock(self):
        with tempfile.TemporaryDirectory() as project_dir:
            first, first_ok = acquire_lock(project_dir)
            second, second_ok = acquire_lock(project_dir)
            try:
                self.assertTrue(first_ok)
                self.assertFalse(second_ok)
            finally:
                first.unlock()
            third, third_ok = acquire_lock(project_dir)
            try:
                self.assertTrue(third_ok)
            finally:
                third.unlock()

    def test_dashboard_can_open_before_any_physical_initialization(self):
        station = Station()
        self.addCleanup(lambda: [c.close() for c in list(station.components.values())])
        opened = MagicMock()
        manager = DeviceManagerWindow(station, opened)
        self.addCleanup(manager.close)
        self.assertTrue(manager.dashboard_button.isEnabled())
        manager.open_dashboard()
        opened.assert_called_once()
        self.assertEqual(set(station.components), set(DEVICE_NAMES))
        self.assertTrue(all(c.simulated for c in station.components.values()))
        self.assertTrue(all(v[1].startswith("SIMULATED") for v in manager.results.values()))

    def test_auto_failure_creates_simulation_and_keeps_error_visible(self):
        import station_devices
        station = Station()
        self.addCleanup(lambda: [c.close() for c in list(station.components.values())])
        manager = DeviceManagerWindow(station, lambda: None)
        self.addCleanup(manager.close)
        real_initialize = station_devices.initialize_device
        def offline_initialize(station, name, *, simulated):
            if not simulated:
                raise RuntimeError("USB device absent")
            return real_initialize(station, name, simulated=True)
        with patch("station_devices.initialize_device", side_effect=offline_initialize):
            manager.initialize_devices(("smu",))
            self.assertTrue(manager.worker.wait(3000))
            self.app.processEvents()
        self.assertTrue(station.smu.simulated)
        self.assertIn("USB device absent", manager.results["smu"][2])
        self.assertEqual(manager.results["smu"][1], "SIMULATED (physical failed)")
        self.assertTrue(manager.dashboard_button.isEnabled())
        self.assertTrue(manager.table.cellWidget(manager.rows["smu"], 1).isEnabled())

    def test_strict_physical_failure_does_not_implicitly_connect_simulation(self):
        station = Station()
        manager = DeviceManagerWindow(station, lambda: None)
        self.addCleanup(manager.close)
        manager.table.cellWidget(manager.rows["camera"], 1).setCurrentText("Physical")
        with patch("station_devices.initialize_device", side_effect=RuntimeError("No CSV")) as initialize:
            manager.initialize_devices(("camera",))
            self.assertTrue(manager.worker.wait(3000))
            self.app.processEvents()
        self.assertEqual(initialize.call_count, 1)
        self.assertFalse(manager.results["camera"][0])
        self.assertNotIn("camera", station.components)
        self.assertTrue(manager.dashboard_button.isEnabled())

    def test_health_failure_is_red_and_recheck_blocked_during_measurement(self):
        remote = NIRvanaSocketWrapper("offline.test", 65432)
        camera = SimpleNamespace(remote_lf=remote, simulated=False)
        station = SimpleNamespace(components={"camera": camera})
        manager = DeviceManagerWindow(station, lambda: None)
        self.addCleanup(manager.close)
        manager.results = {name: (True, "INITIALIZED", "") for name in DEVICE_NAMES}
        remote._set_health("failed", "CSV transfer failed")
        manager.refresh_health()
        self.assertTrue(manager.dashboard_button.isEnabled())
        self.assertFalse(manager.results["camera"][0])
        manager.dashboard = MagicMock()
        manager.dashboard.measurement_active.return_value = True
        manager.initialize_devices(("camera",), recheck=True)
        self.assertIsNone(manager.worker)

    def test_recheck_reuses_camera_and_suspends_dashboard_polling(self):
        camera = SimpleNamespace(remote_lf=NIRvanaSocketWrapper("offline.test", 65432), simulated=False)
        station = SimpleNamespace(components={"camera": camera})
        manager = DeviceManagerWindow(station, lambda: None)
        self.addCleanup(manager.close)
        manager.dashboard = MagicMock()
        manager.dashboard.measurement_active.return_value = False
        manager.dashboard.prepare_device_change.return_value = True
        manager.dashboard.devices_changing = True
        with patch.object(camera.remote_lf, "verify_acquisition", return_value="Checked twice") as verify, patch.object(manager, "fill_missing_devices"):
            manager.initialize_devices(("camera",), recheck=True)
            manager.dashboard.prepare_device_change.assert_called_once()
            self.assertTrue(manager.worker.wait(3000))
            self.app.processEvents()
            manager.dashboard.finish_device_change.assert_called_once()
            verify.assert_called_once()
        self.assertIs(station.components["camera"], camera)

    def test_device_modes_are_not_locked_after_initialization(self):
        station = Station()
        self.addCleanup(lambda: [c.close() for c in list(station.components.values())])
        manager = DeviceManagerWindow(station, lambda: None)
        self.addCleanup(manager.close)
        combo = manager.table.cellWidget(manager.rows["piezo150"], 1)
        combo.setCurrentText("Simulation")
        manager.initialize_devices(("piezo150",))
        self.assertTrue(manager.worker.wait(3000))
        self.app.processEvents()
        component = station.piezo150
        manager.initialize_devices(("piezo150",), recheck=True)
        self.assertTrue(manager.worker.wait(3000))
        self.app.processEvents()
        self.assertIs(station.piezo150, component)
        self.assertTrue(combo.isEnabled())

    def test_connection_scan_remains_available_during_measurement_without_reconnect(self):
        controller = SimpleNamespace(simulated=False, comm_lock=threading.RLock(),
                                     get_axis_status=MagicMock(return_value={'offset': 9., 'output': 9., 'mode': 'off'}))
        station = SimpleNamespace(components={'piezo300': controller})
        manager = DeviceManagerWindow(station, lambda: None)
        self.addCleanup(manager.close)
        manager.dashboard = MagicMock()
        manager.dashboard.measurement_active.return_value = True
        report = {'ports': [], 'controllers': [], 'errors': {}, 'identity_probed': False}
        with patch('device_discovery.scan_attocubes', return_value=report) as scan, patch('device_manager.configure_device') as configure:
            manager.scan_connections()
            self.assertTrue(manager.scan_worker.wait(3000))
            self.app.processEvents()
            self.assertFalse(scan.call_args.kwargs['identify'])
            configure.assert_not_called()
        manager.dashboard.prepare_device_change.assert_not_called()
        self.assertTrue(manager.results['piezo300'][0])
        self.assertIn('9.000 V', manager.results['piezo300'][2])
        self.assertIs(station.components['piezo300'], controller)
        self.assertTrue(manager.scan_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
