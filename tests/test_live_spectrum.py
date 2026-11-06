"""Live display, responsiveness and plotter isolation with simulated cameras only."""
import contextlib
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PyQt6 import QtCore, QtTest, QtWidgets
from qcodes.station import Station
from measurements import single_spectra_gui as spectra
from station_devices import configure_device


class LiveSpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        self.directory = Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        self.context.enter_context(patch.object(spectra, "parent_dir", str(self.directory)))
        self.context.enter_context(patch("socket.create_connection", side_effect=AssertionError("Physical network forbidden")))
        self.context.enter_context(patch("pyvisa.ResourceManager", side_effect=AssertionError("Physical VISA forbidden")))
        self.context.enter_context(patch("Pyro4.Proxy", side_effect=AssertionError("Physical proxy forbidden")))
        self.errors = self.context.enter_context(patch.object(QtWidgets.QMessageBox, "critical"))
        self.station = Station()
        configure_device(self.station, "camera", "Simulation")
        configure_device(self.station, "monochromator", "Simulation")
        self.window = spectra.SingleSpectraWindow(self.station)
        self.addCleanup(self.close_window)
        self.window.status_worker.pause()
        self.window.show()
        self.app.processEvents()

    def close_window(self):
        self.window.is_running = False
        self.window.close()
        self.app.processEvents()
        for device in list(self.station.components.values()):
            device.close()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        self.assertTrue(predicate())

    def test_last_completed_live_frame_remains_and_does_not_autosave(self):
        window = self.window
        window.signal_x, window.signal_y = np.array([700., 701.]), np.array([-12., -34.])
        self.station.camera.num_frames(7)
        with patch.object(window, "trigger_autosave") as autosave:
            window.run_btn.click()
            self.wait_for(lambda: getattr(window, "_received_data_this_run", False))
            window.run_btn.click()
            self.wait_for(lambda: not window.is_running)
            self.assertEqual(window.signal_y.size, 512)
            np.testing.assert_array_equal(window.plot_curves["Signal"].yData, window.signal_y)
            self.assertEqual(window.run_btn.text(), "Live on")
            self.assertEqual(self.station.camera.num_frames(), 7)
            autosave.assert_not_called()
            self.assertEqual(list(self.directory.rglob("*.h5")), [])
        self.errors.assert_not_called()

    def test_autoscale_tracks_new_frames_until_manual_zoom_and_resumes_on_auto(self):
        window = self.window
        window.is_running, window.active_acquisition_mode = True, "continuous"
        x = np.array([700., 701., 702.])
        window.on_data_acquired("continuous", x, np.array([1., 2., 3.]))
        self.app.processEvents()
        window.on_data_acquired("continuous", x, np.array([100., 200., 300.]))
        self.app.processEvents()
        view = window.plot_widget.getViewBox()
        self.assertLess(view.viewRange()[1][0], 100)
        self.assertGreater(view.viewRange()[1][1], 300)
        view.setRange(xRange=(700.5, 701.5), yRange=(150., 250.), padding=0)
        view.sigRangeChangedManually.emit([True, True])
        zoom = view.viewRange()
        window.on_data_acquired("continuous", x, np.array([1000., 2000., 3000.]))
        self.app.processEvents()
        np.testing.assert_allclose(view.viewRange(), zoom)
        self.assertFalse(window._live_autoscale)
        window.scale_plots()
        window.on_data_acquired("continuous", x, np.array([10000., 20000., 30000.]))
        self.app.processEvents()
        self.assertTrue(window._live_autoscale)
        self.assertGreater(view.viewRange()[1][1], 30000)

    def test_worker_coalesces_frames_and_preserves_frame_completed_during_stop(self):
        remote, camera = Mock(), Mock()
        camera.remote_lf, camera.num_frames.return_value = remote, 7
        worker = spectra.AcquisitionWorker(SimpleNamespace(camera=camera), "continuous", 1, False, "Sum")
        remote.get_wavelengths_cache.return_value = [700., 701.]
        count = 0
        def preview():
            nonlocal count
            count += 1
            if count == 200:
                worker.stop()
            return [float(count), float(count + 1)]
        remote.get_latest_preview_data.side_effect = preview
        delivered = []
        worker.point_measured.connect(lambda *frame: delivered.append(frame))
        with patch.object(worker, "msleep"):
            worker.run()
        self.assertEqual(count, 200)
        self.assertEqual(delivered, [])
        np.testing.assert_array_equal(worker.take_latest_preview()[1], [200., 201.])
        self.assertIsNone(worker.take_latest_preview())
        self.assertEqual(camera.num_frames.call_args.args, (7,))
        remote.stop_preview.assert_called_once()

    def test_slow_intermediate_query_does_not_block_gui_and_stale_result_is_dropped(self):
        window = self.window
        window.status_worker.stop()
        window.status_worker.wait()
        original_remote = self.station.camera.remote_lf
        self.station.camera.remote_lf = SimpleNamespace(_pyroUri="PYRO:offline@test:1")
        self.addCleanup(setattr, self.station.camera, "remote_lf", original_remote)
        window.is_running, window.active_acquisition_mode = True, "single"
        entered, release = threading.Event(), threading.Event()
        proxy = Mock()
        def preview():
            entered.set()
            release.wait(2)
            return [1., 2.]
        proxy.get_latest_preview_data.side_effect = preview
        proxy.get_wavelengths_cache.return_value = [700., 701.]
        ticks = []
        heartbeat = QtCore.QTimer()
        heartbeat.setInterval(5)
        heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
        heartbeat.start()
        try:
            with patch.object(spectra.Pyro4, "Proxy", return_value=proxy) as factory:
                window.poll_intermediate_frame()
                self.wait_for(entered.is_set)
                QtTest.QTest.qWait(60)
                self.assertGreaterEqual(len(ticks), 3)
                window.poll_intermediate_frame()
                factory.assert_called_once()
                window._preview_generation += 1
                release.set()
                self.wait_for(lambda: window._intermediate_worker is None)
                self.assertIsNone(window.signal_y)
        finally:
            release.set()
            heartbeat.stop()
            if window._intermediate_worker is not None:
                window._intermediate_worker.wait()
                self.app.processEvents()

    def test_plotter_mode_blocks_hardware_handlers_and_persistent_settings(self):
        window = self.window
        window.set_plotter_mode(True)
        settings = (self.directory / "value_storage").read_bytes()
        with patch.object(self.station.camera, "exposure_time") as exposure, \
                patch.object(self.station.camera, "num_frames") as frames, \
                patch.object(self.station.monochromator, "wavelength") as wavelength, \
                patch.object(self.station.camera.remote_lf, "set_grating") as grating, \
                patch.object(self.station.camera.remote_lf, "set_target_temperature") as temperature, \
                patch.object(spectra, "AcquisitionWorker") as worker:
            window.start_acquisition("single")
            window.run_btn.setChecked(True)
            window.toggle_run()
            window.on_exposure_changed()
            window.on_num_spectra_changed()
            window.on_wavelength_changed()
            window.on_energy_changed()
            window.on_grating_changed()
            window.on_target_temperature_changed()
            window.reload_gui_from_storage()
            window.save_value_storage()
            window.save_data_to_file(str(self.directory / "blocked-plotter-save.h5"))
            self.assertFalse((self.directory / "blocked-plotter-save.h5").exists())
            for action in (exposure, frames, wavelength, grating, temperature, worker):
                action.assert_not_called()
        self.assertFalse(window.run_btn.isEnabled())
        self.assertFalse(window.acquire_btn.isEnabled())
        self.assertEqual((self.directory / "value_storage").read_bytes(), settings)
        window.set_plotter_mode(False)
        self.assertTrue(window.acquire_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
