"""Exercise the actual dashboard using simulation and temporary data only."""
import contextlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import time
import threading
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import h5py
import numpy as np
from PyQt6 import QtWidgets, QtTest
from qcodes.station import Station
from lab_gui import LabMainWindow
from measurements.position_tab import PositionTab
from station_devices import configure_device, remove_device


class DashboardWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        directory = self.context.enter_context(tempfile.TemporaryDirectory())
        self.directory = Path(directory)
        (self.directory / "value_storage").write_text(json.dumps({
            "shared": {"exposure_time": 0.01, "settling_time": 0.0},
            "single_spectra": {"num_spectra": 2},
        }))
        for name in ("single_spectra_gui", "Dual_gate_sweep_gui", "B_V_sweep_gui",
                     "photoluminescence_map_gui", "scalar_map_gui", "polarization_control_gui", "position_dgs_gui",
                     "gate_map_gui", "capacitance_gui", "transport_gui"):
            module = importlib.import_module("measurements." + name)
            self.context.enter_context(patch.object(module, "parent_dir", directory))
        self.context.enter_context(patch.object(PositionTab, "save_storage"))
        self.context.enter_context(patch("scanner_safety.wait_for_motion", side_effect=lambda seconds, keep_running: keep_running()))
        # Any unexpected physical network/VISA access makes the test fail.
        self.context.enter_context(patch("pyvisa.ResourceManager", side_effect=AssertionError("Hardware access in simulation test")))
        self.context.enter_context(patch("socket.create_connection", side_effect=AssertionError("Network access in dashboard test")))
        self.context.enter_context(patch("Pyro4.Proxy", side_effect=AssertionError("Remote access in simulation test")))
        self.errors = self.context.enter_context(patch.object(QtWidgets.QMessageBox, "critical"))
        self.context.enter_context(patch.object(QtWidgets.QMessageBox, "information"))
        self.station = Station()
        self.dashboard = LabMainWindow(self.station)
        self.addCleanup(self.close_dashboard)
        self.app.processEvents()

    def close_dashboard(self):
        self.dashboard.close()
        self.app.processEvents()
        for component in list(self.station.components.values()):
            component.close()

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(20)
        self.assertTrue(predicate())

    def test_live_never_autosaves_and_recorded_acquisition_still_saves(self):
        window = self.dashboard.single_window
        saved_x = np.array([700., 701.])
        saved_y = np.array([12., 34.])
        window.signal_x, window.signal_y = saved_x, saved_y
        self.station.camera.num_frames(7)
        self.assertEqual(window.run_btn.text(), "Live on")
        window.run_btn.click()
        self.wait_for(lambda: window.signal_y is not saved_y)
        first_curve = window.plot_curves["Signal"].yData.copy()
        self.wait_for(lambda: not np.array_equal(window.plot_curves["Signal"].yData, first_curve))
        np.testing.assert_array_equal(window.plot_curves["Signal"].yData, window.signal_y)
        self.assertTrue(window.is_running)
        self.assertEqual(window.run_btn.text(), "Live off")
        self.dashboard.check_measurement_status()
        for index in (0, 5, 6):
            self.assertTrue(self.dashboard.tab_widget.isTabEnabled(index))
        self.assertFalse(self.dashboard.tab_widget.isTabEnabled(1))
        self.assertFalse(window.save_btn.isEnabled())
        self.assertFalse(self.dashboard.prepare_device_change())
        window.trigger_autosave(was_interrupted=True)
        self.assertEqual(list(self.directory.rglob("*.h5")), [])
        window.run_btn.click()
        self.wait_for(lambda: not window.is_running)
        self.assertIsNot(window.signal_y, saved_y)
        np.testing.assert_array_equal(window.plot_curves["Signal"].yData, window.signal_y)
        self.assertEqual(window.run_btn.text(), "Live on")
        self.assertEqual(self.station.camera.num_frames(), 7)
        self.assertEqual(list(self.directory.rglob("*.h5")), [])
        window.num_spectra_box.setValue(2)
        window.start_acquisition("single")
        self.wait_for(lambda: not window.is_running)
        files = list(self.directory.rglob("*.h5"))
        self.assertEqual(len(files), 1)
        with h5py.File(files[0]) as data:
            self.assertEqual(data["signal/raw_intensities"].shape, (2, 512))
            self.assertEqual(data["signal/wavelength"].shape, (512,))
            self.assertEqual(json.loads(data.attrs["device_modes"])["camera"], "simulated")
        self.errors.assert_not_called()

    def test_device_replacement_keeps_dashboard_data_and_rebinds_controllers(self):
        old_camera = self.station.camera
        old_piezo = self.station.piezo300
        saved = np.ones((1, 1, 512))
        self.dashboard.dual_window.data_spectra = saved
        self.assertTrue(self.dashboard.prepare_device_change())
        remove_device(self.station, "camera")
        remove_device(self.station, "piezo300")
        configure_device(self.station, "camera", "Simulation")
        configure_device(self.station, "piezo300", "Simulation")
        self.dashboard.finish_device_change()
        self.assertIsNot(self.station.camera, old_camera)
        self.assertIsNot(self.station.piezo300, old_piezo)
        self.assertIs(self.dashboard.position_tab.scanner_controller, self.station.piezo300)
        self.assertIs(self.dashboard.single_window.status_worker.camera, self.station.camera)
        self.assertIs(self.dashboard.dual_window.data_spectra, saved)
        self.assertTrue(self.dashboard.centralWidget().isEnabled())
        self.assertIn("camera", self.dashboard.simulation_label.text())

    def test_grating_readback_accepts_actual_unknown_value_without_writes(self):
        window = self.dashboard.single_window
        status = self.station.camera.remote_lf.get_status()
        status.update(grating="[custom 900 g/mm][4][0]", wl=1300.)
        with patch.object(self.station.monochromator, "_set_wavelength") as setter:
            window.on_status_polled(status)
            setter.assert_not_called()
        self.assertEqual(window.grating_combo.currentText(), status["grating"])
        self.assertEqual(window.wl_box.value(), 1300.)
        self.errors.assert_not_called()

    def setup_fake_nirvana(self):
        from drivers.remote_client import NIRvanaSocketWrapper
        window = self.dashboard.single_window
        window.status_worker.stop()
        window.status_worker.wait()
        camera = self.station.camera
        camera.simulated = False
        camera.spectrometer_type = "lightfieldG11"
        camera.remote_lf = NIRvanaSocketWrapper("offline.test", 65432, enable_controls=True)
        status = {
            "success": True, "status_source": "lightfield", "wl": 1100.,
            "grating": "[150 g/mm][0]", "gratings": ["[150 g/mm][0]", "[600 g/mm][1]"],
            "capabilities": {"grating": True, "wavelength": True, "preview": True},
        }
        camera.remote_lf.controls = Mock()
        camera.remote_lf.controls.get_status.return_value = status
        return window, camera.remote_lf, status

    def test_nirvana_discovery_populates_only_real_gratings_without_movement(self):
        window, remote, status = self.setup_fake_nirvana()
        window.on_status_polled({"success": True, "status_source": "local_settings"})
        self.assertFalse(window.grating_combo.isEnabled())
        self.assertFalse(window.run_btn.isEnabled())
        self.assertTrue(window.acquire_btn.isEnabled())
        window.on_status_polled(status)
        self.assertEqual([window.grating_combo.itemText(i) for i in range(window.grating_combo.count())], status["gratings"])
        self.assertEqual(window.grating_combo.currentText(), status["grating"])
        self.assertEqual(window.wl_box.value(), status["wl"])
        self.assertNotEqual(window.wl_box.lineEdit().text(), "Unavailable")
        self.assertTrue(window.grating_combo.isEnabled())
        self.assertTrue(window.wl_box.isEnabled())
        self.assertTrue(window.run_btn.isEnabled())
        remote.controls.set_grating.assert_not_called()
        remote.controls.set_wavelength.assert_not_called()
        window.on_wavelength_changed()  # Leaving an unchanged field cannot move hardware.
        remote.controls.set_wavelength.assert_not_called()

    def test_nirvana_grating_command_runs_in_worker_and_blocks_competing_actions(self):
        window, remote, status = self.setup_fake_nirvana()
        window.on_status_polled(status)
        finish = threading.Event()
        def set_grating(value):
            finish.wait(timeout=3)
            return dict(status, grating=value)
        remote.controls.set_grating.side_effect = set_grating
        try:
            window.grating_combo.setCurrentText(status["gratings"][1])
            self.assertTrue(window.grating_setting)
            self.assertTrue(self.dashboard.measurement_active())
            self.assertFalse(window.acquire_btn.isEnabled())
            self.assertFalse(window.wl_box.isEnabled())
            self.assertFalse(self.dashboard.tab_widget.isTabEnabled(1))
            self.assertFalse(self.dashboard.prepare_device_change())
            window.start_acquisition("single")
            self.assertFalse(window.is_running)
        finally:
            finish.set()
            self.wait_for(lambda: not window.grating_setting)
        remote.controls.set_grating.assert_called_once_with(status["gratings"][1])
        self.assertEqual(window.grating_combo.currentText(), status["gratings"][1])
        self.assertTrue(window.acquire_btn.isEnabled())
        self.assertIn("confirmed", window.status_lbl.text())
        self.errors.assert_not_called()

    def test_native_nirvana_unavailable_stream_stays_disabled_after_acquire(self):
        window, remote, _ = self.setup_fake_nirvana()
        window.on_status_polled({"success": True, "status_source": "local_settings"})
        remote.get_spectrum = Mock(return_value=([[10., 20.]], [1100., 1101.]))
        window.num_spectra_box.setValue(1)
        window.acquire_btn.click()
        self.wait_for(lambda: not window.is_running)
        remote.get_spectrum.assert_called_once()
        self.assertTrue(window.acquire_btn.isEnabled())
        self.assertFalse(window.run_btn.isEnabled())
        self.assertEqual(window.run_btn.text(), "Live on")
        self.assertEqual(len(list(self.directory.rglob("*.h5"))), 1)
        remote.controls.request.assert_not_called()
        self.errors.assert_not_called()

    def test_unchanged_nirvana_live_updates_stops_cleanly_and_does_not_save(self):
        window, remote, _ = self.setup_fake_nirvana()
        remote.controls_enabled = False
        window.on_status_polled(remote.get_status())
        self.assertTrue(window.run_btn.isEnabled())
        self.assertEqual(window.run_btn.text(), "Live on")
        self.station.camera.num_frames(7)
        saved = np.array([81., 82.])
        window.signal_x, window.signal_y = np.array([1100., 1101.]), saved
        calls = []
        waiting, release = threading.Event(), threading.Event()
        def trigger(params, **kwargs):
            calls.append(dict(params))
            if len(calls) == 3:
                waiting.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Test did not release the current spectrum")
            return np.array([1100., 1101.]), np.tile([len(calls), len(calls) + 1.], (params['num_frames'], 1))
        with patch("NIRvana_send_trigger.trigger", side_effect=trigger), patch.object(self.station.camera, "abort_acquisition") as abort:
            try:
                window.run_btn.click()
                self.wait_for(waiting.is_set)
                self.wait_for(lambda: np.array_equal(window.plot_curves['Signal'].yData, [2., 3.]))
                np.testing.assert_array_equal(window.plot_curves['Signal'].yData, [2., 3.])
                self.assertFalse(window.save_btn.isEnabled())
                self.assertEqual(list(self.directory.rglob('*.h5')), [])
                window.run_btn.click()
                self.assertTrue(window.is_running)
                self.assertIn('current spectrum', window.status_lbl.text())
                abort.assert_not_called()
            finally:
                release.set()
                self.wait_for(lambda: not window.is_running)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(command['command'] == 'start_acquisition' and command['num_frames'] == 1 for command in calls))
            np.testing.assert_array_equal(window.signal_y, [3., 4.])
            self.assertEqual(self.station.camera.num_frames(), 7)
            self.assertFalse(remote._acquisition_lock.locked())
            self.assertEqual(list(self.directory.rglob('*.h5')), [])
            self.assertTrue(window.run_btn.isEnabled())
            window.num_spectra_box.setValue(2)
            window.acquire_btn.click()
            self.wait_for(lambda: not window.is_running)
            self.assertEqual(len(list(self.directory.rglob('*.h5'))), 1)
            self.assertEqual(calls[-1]['num_frames'], 2)
        remote.controls.request.assert_not_called()
        remote.controls.get_status.assert_not_called()
        self.errors.assert_not_called()

    def test_native_nirvana_unavailable_stream_cannot_start_a_request(self):
        window, remote, _ = self.setup_fake_nirvana()
        window.on_status_polled({"success": True, "status_source": "local_settings"})
        saved = window.signal_y
        window.run_btn.setEnabled(True)  # Reproduce stale/incorrect button state.
        with patch.object(remote, "start_preview") as preview, patch.object(remote, "get_spectrum") as acquire:
            window.run_btn.click()
            preview.assert_not_called()
            acquire.assert_not_called()
        self.assertFalse(window.is_running)
        self.assertFalse(window.run_btn.isChecked())
        self.assertFalse(window.run_btn.isEnabled())
        self.assertIn("Live unavailable", window.status_lbl.text())
        self.assertIs(window.signal_y, saved)
        self.assertEqual(list(self.directory.rglob("*.h5")), [])
        remote.controls.request.assert_not_called()

    def test_nirvana_wavelength_uses_real_camera_connection_and_reports_failure(self):
        window, remote, status = self.setup_fake_nirvana()
        window.on_status_polled(status)
        remote.controls.set_wavelength.return_value = dict(status, wl=1250.)
        window.wl_box.setValue(1250.)
        window.on_wavelength_changed()
        self.wait_for(lambda: not window.grating_setting)
        remote.controls.set_wavelength.assert_called_once_with(1250.)
        self.assertEqual(window.wl_box.value(), 1250.)
        remote.controls.set_grating.side_effect = RuntimeError("motor unavailable")
        window.grating_combo.setCurrentText(status["gratings"][1])
        self.wait_for(lambda: not window.grating_setting)
        self.assertIn("motor unavailable", window.status_lbl.text())
        self.assertFalse(window.grating_combo.isEnabled())
        self.assertTrue(window.acquire_btn.isEnabled())

    def test_scanner_loads_real_readbacks_and_steps_both_directions_from_them(self):
        tab = self.dashboard.position_tab
        scanner = self.station.piezo300
        scanner._write_voltage_step(1, 9.)
        scanner._write_voltage_step(2, 32.099)
        with patch.object(scanner, "_write_voltage_step", wraps=scanner._write_voltage_step) as setter:
            tab.load_storage()
            self.assertEqual(tab.vx_set_spin.value(), 9.)
            self.assertEqual(tab.vy_set_spin.value(), 32.099)
            setter.assert_not_called()
            for button in (tab.s_btn_left, tab.s_btn_right, tab.s_btn_up, tab.s_btn_down):
                self.assertTrue(button.isEnabled())
            tab.s_step_size_edit.setText('0.1')
            tab.s_btn_left.click()
            self.wait_for(lambda: not tab.scanner_moving())
            self.app.processEvents()
            self.assertAlmostEqual(scanner.get_voltage(1), 8.9)
            tab.s_btn_right.click()
            self.wait_for(lambda: not tab.scanner_moving())
            self.app.processEvents()
            self.assertAlmostEqual(scanner.get_voltage(1), 9.)
            tab.s_btn_down.click()
            self.wait_for(lambda: not tab.scanner_moving())
            self.app.processEvents()
            self.assertAlmostEqual(scanner.get_voltage(2), 31.999)
            tab.s_btn_up.click()
            self.wait_for(lambda: not tab.scanner_moving())
            self.app.processEvents()
            self.assertAlmostEqual(scanner.get_voltage(2), 32.099)
        self.assertIn('SIMULATED ANC300', tab.scanner_status.text())

    def test_scanner_failed_read_cannot_move_from_stale_label_or_zero(self):
        tab = self.dashboard.position_tab
        scanner = self.station.piezo300
        scanner._write_voltage_step(1, 9.)
        tab.refresh_scanner_readbacks()
        with patch.object(scanner, 'get_voltage', side_effect=RuntimeError('Disconnected')), patch.object(scanner, '_write_voltage_step') as setter:
            tab.move_scanner('x', -1)
            self.wait_for(lambda: not tab.scanner_moving())
            self.app.processEvents()
            setter.assert_not_called()
            self.assertIn('unavailable', tab.vx_label.text())
            self.assertIn('Disconnected', tab.scanner_status.text())
            self.assertFalse(tab.s_btn_left.isEnabled())
            self.assertFalse(tab.vx_set_spin.isEnabled())
        tab.refresh_scanner_readbacks()
        self.assertEqual(tab.vx_set_spin.value(), 9.)
        self.assertTrue(tab.s_btn_left.isEnabled())

    def test_scanner_failed_write_keeps_readback_and_shows_error(self):
        tab = self.dashboard.position_tab
        scanner = self.station.piezo300
        scanner._write_voltage_step(2, 32.099)
        tab.refresh_scanner_readbacks()
        with patch.object(scanner, '_write_voltage_step', side_effect=RuntimeError('Write failed')):
            tab.move_scanner('y', 1)
            self.wait_for(lambda: not tab.scanner_moving())
            self.app.processEvents()
        self.assertEqual(tab.vy_set_spin.value(), 32.099)
        self.assertIn('Write failed', tab.scanner_status.text())

    def test_viridis_default_and_reset(self):
        from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
        for window, attr in ((self.dashboard.dual_window, "gatemap_colorbar"),
                             (self.dashboard.bv_window, "gatemap_colorbar")):
            state = getattr(window, attr).gradient.saveState()
            self.assertEqual(state["ticks"], Gradients["viridis"]["ticks"])
            window.vslice_colorbar.gradient.loadPreset("thermal")
            window.scale_plots()
            self.assertEqual(window.vslice_colorbar.gradient.saveState()["ticks"], Gradients["viridis"]["ticks"])

        window = self.dashboard.dual_window
        expected_lut = window.gatemap_colorbar.gradient.getLookupTable(256)
        window.gatemap_colorbar.gradient.loadPreset("thermal")
        window.scale_plots()
        window.gatemap_img.setImage(np.arange(4, dtype=float).reshape(2, 2))
        self.assertEqual(window.gatemap_colorbar.gradient.saveState()["ticks"], Gradients["viridis"]["ticks"])
        np.testing.assert_array_equal(window.gatemap_img.lut(window.gatemap_img.image, 256), expected_lut)

    def test_all_measurement_plot_axes_and_ticks_adapt_to_zoom(self):
        import pyqtgraph as pg
        windows = (self.dashboard.single_window, self.dashboard.dual_window,
                   self.dashboard.bv_window, self.dashboard.pl_map_window,
                   self.dashboard.scalar_map_window, self.dashboard.position_dgs_window)
        # The reset action used to restore five fixed labels in the DGS slices.
        self.dashboard.dual_window.reset_plot_scaling()
        self.dashboard.position_dgs_window.reset_plot_scaling()
        checked = 0
        for window in windows:
            plots = [value for value in vars(window).values() if isinstance(value, pg.PlotWidget)]
            self.assertTrue(plots)
            for plot in plots:
                plot.setRange(xRange=(-10, 10), yRange=(-10, 10), padding=0)
                wide = {side: plot.getAxis(side).tickValues(*plot.getAxis(side).range, 400)[0][0]
                        for side in ('bottom', 'left')}
                plot.setRange(xRange=(.0011, .0019), yRange=(.0011, .0019), padding=0)
                for side in ('bottom', 'left'):
                    axis = plot.getAxis(side)
                    with self.subTest(window=type(window).__name__, plot=plot.plotItem.titleLabel.text, axis=side):
                        self.assertIsNone(axis._tickLevels)
                        spacing, values = axis.tickValues(*axis.range, 400)[0]
                        self.assertLess(spacing, wide[side])
                        self.assertGreaterEqual(len(values), 3)
                        labels = axis.tickStrings(values, axis.autoSIPrefixScale, spacing)
                        self.assertEqual(len(labels), len(set(labels)))
                    checked += 1
        self.assertGreater(checked, 60)

    def test_voltage_axes_keep_volts_and_correct_numbers_after_zoom_and_reset(self):
        import pyqtgraph as pg
        for window in (self.dashboard.dual_window, self.dashboard.bv_window,
                       self.dashboard.position_dgs_window):
            window.update_variable_names()
            window.reset_plot_scaling()
            for plot in (value for value in vars(window).values() if isinstance(value, pg.PlotWidget)):
                for side in ('bottom', 'left'):
                    axis = plot.getAxis(side)
                    if axis.labelUnits != 'V': continue
                    with self.subTest(window=type(window).__name__, plot=plot.plotItem.titleLabel.text, axis=side):
                        for low, high in ((-.01, .01), (9.0001, 9.0009), (32, 32.1)):
                            axis.setRange(low, high)
                            self.assertEqual(axis.autoSIPrefixScale, 1)
                            self.assertEqual(axis.labelUnitPrefix, '')
                            self.assertIn('(V)', axis.label.toPlainText())
                        ticks = [9.0002, 9.0004, 9.0006]
                        labels = axis.tickStrings(ticks, axis.autoSIPrefixScale, .0002)
                        np.testing.assert_allclose([float(label) for label in labels], ticks, rtol=0, atol=1e-10)
        # Keep readable prefixes for tiny currents and optical powers.
        current = self.dashboard.dual_window.t2_hslice_win.getAxis('left')
        current.setRange(-1e-9, 1e-9)
        self.assertTrue(current.autoSIPrefix)
        power = self.dashboard.scalar_map_window.slice_x_plot.getAxis('left')
        power.setRange(0, 1e-6)
        self.assertTrue(power.autoSIPrefix)

    def test_compliance_is_editable_only_in_dgs_and_never_applied_from_saved_settings(self):
        from drivers.keysight_b2902b import GATE_CURRENT_COMPLIANCE_A
        settings = self.directory / 'value_storage'
        data = json.loads(settings.read_text())
        data.setdefault('dual_gate', {}).update(v1_compliance=.1, v2_compliance=.01)
        settings.write_text(json.dumps(data))
        self.dashboard.dual_window.reload_gui_from_storage()
        for window in (self.dashboard.dual_window, self.dashboard.position_dgs_window):
            window.set_inputs_enabled(True)
            window.on_mode_changed(write_hardware=False)
            for axis in (1, 2):
                box = getattr(window, f'ch{axis}_compliance')
                self.assertEqual(box.isReadOnly(), window is self.dashboard.position_dgs_window)
                self.assertEqual(box.value(), GATE_CURRENT_COMPLIANCE_A)
                box.setValue(.1)
                window.on_compliance_changed()
                self.assertEqual(box.value(), GATE_CURRENT_COMPLIANCE_A)
        self.dashboard.dual_window.save_storage()
        saved = json.loads(settings.read_text())['dual_gate']
        for axis in (1, 2):
            self.assertEqual(saved[f'v{axis}_compliance'], GATE_CURRENT_COMPLIANCE_A)
            self.assertEqual(getattr(self.station.smu, f'ch{axis}').curr_limit(), GATE_CURRENT_COMPLIANCE_A)
        self.errors.assert_not_called()

    def test_dgs_applies_independent_current_limits_and_preserves_them_in_sweeps(self):
        window = self.dashboard.dual_window
        self.dashboard.tab_widget.setCurrentIndex(1)
        limits = {1: 2e-6, 2: 5e-5}
        with patch.object(self.station.smu.ch1.set_r, 'volt') as first_ramp, \
             patch.object(self.station.smu.ch2.set_r, 'volt') as second_ramp:
            for axis, value in limits.items():
                box = getattr(window, f'ch{axis}_compliance')
                self.assertFalse(box.isReadOnly())
                box.setValue(value)
                box.editingFinished.emit()
                self.assertTrue(window._applying_compliance)
                self.assertFalse(window.begin_btn.isEnabled())
                self.wait_for(lambda: not window._applying_compliance and not window.ramp_worker.isRunning())
                self.assertEqual(getattr(self.station.smu, f'ch{axis}').curr_limit(), value)
                if axis == 1:
                    self.assertEqual(self.station.smu.ch2.curr_limit(), 1e-7)
            first_ramp.assert_not_called()
            second_ramp.assert_not_called()
        window.c1_start.setValue(0)
        window.c2_start.setValue(0)
        window.c1_pts.setValue(1)
        window.c2_pts.setValue(1)
        window.begin_sweep()
        self.wait_for(lambda: not window.is_running)
        for axis, value in limits.items():
            self.assertEqual(getattr(self.station.smu, f'ch{axis}').curr_limit(), value)
        saved = list(self.directory.rglob('*.h5'))
        self.assertEqual(len(saved), 1)
        with h5py.File(saved[0]) as data:
            for axis, value in limits.items():
                self.assertEqual(data.attrs[f'current_compliance_ch{axis}_A'], value)
        self.errors.assert_not_called()

    def test_polling_cannot_overwrite_a_current_limit_being_edited(self):
        window = self.dashboard.dual_window
        window.ch1_compliance.setValue(2e-6)
        window.on_compliance_readback({1: ('VOLT', 1e-7), 2: ('VOLT', 1e-7)})
        self.assertEqual(window.ch1_compliance.value(), 2e-6)
        self.assertEqual(self.station.smu.ch1.curr_limit(), 1e-7)
        window.ch1_compliance.editingFinished.emit()
        self.wait_for(lambda: not window._applying_compliance and not window.ramp_worker.isRunning())
        self.assertEqual(self.station.smu.ch1.curr_limit(), 2e-6)
        self.errors.assert_not_called()

    def test_failed_compliance_update_reports_error_and_keeps_previous_target(self):
        window = self.dashboard.dual_window
        with patch.object(self.station.smu.ch1.curr_limit, 'set', side_effect=RuntimeError('Write failed')):
            window.ch1_compliance.setValue(2e-6)
            window.ch1_compliance.editingFinished.emit()
            self.wait_for(lambda: not window._applying_compliance and not window.ramp_worker.isRunning())
        self.assertEqual(self.station.smu.ch1.current_compliance_target, 1e-7)
        self.assertEqual(self.station.smu.ch1.curr_limit(), 1e-7)
        self.assertTrue(window.ch1_compliance.isEnabled())
        self.assertIn('failed', window.status_lbl.text())
        self.errors.assert_called_once()

    def test_compliance_cannot_change_during_another_acquisition_or_in_plotter_mode(self):
        window = self.dashboard.dual_window
        with patch.object(self.station.smu.ch1.curr_limit, 'set') as setter:
            self.dashboard.single_window.is_running = True
            try:
                window.ch1_compliance.setValue(2e-6)
                window.ch1_compliance.editingFinished.emit()
                setter.assert_not_called()
                self.assertFalse(window._applying_compliance)
            finally:
                self.dashboard.single_window.is_running = False
            window.plotter_mode = True
            try:
                window.ch1_compliance.setValue(5e-6)
                window.ch1_compliance.editingFinished.emit()
                setter.assert_not_called()
                self.assertFalse(window._applying_compliance)
            finally:
                window.plotter_mode = False

    def test_current_mode_compliance_is_readback_only_and_never_written_by_gui(self):
        from measurements.Dual_gate_sweep_gui import read_compliance_values
        window = self.dashboard.dual_window
        window.ch1_mode.setCurrentText('CURR')
        self.assertEqual(window.ch1_setpoint.maximum(), 1e-7)
        window.on_compliance_readback(read_compliance_values(self.station.smu))
        self.assertTrue(window.ch1_compliance.isReadOnly())
        self.assertEqual(window.ch1_compliance.value(), 10.)
        with patch.object(self.station.smu.ch1.volt_limit, 'set') as setter:
            window.ch1_compliance.setValue(5.)
            window.on_compliance_changed()
            window.save_storage()
            setter.assert_not_called()
        self.assertEqual(window.ch1_compliance.value(), 10.)
        self.assertEqual(self.station.smu.ch1.volt_limit(), 10.)
        self.errors.assert_not_called()

    def test_compliance_readback_does_not_hide_a_wrong_or_missing_instrument_value(self):
        window = self.dashboard.dual_window
        window.on_compliance_readback({1: ('VOLT', .1), 2: None})
        self.assertEqual(window.ch1_compliance.value(), .1)
        self.assertIn('#b91c1c', window.ch1_compliance.styleSheet())
        self.assertIn('unavailable', window.ch2_compliance.text())
        with patch.object(self.station.smu.ch1.curr_limit, 'set') as setter:
            window.save_storage()
            window.reload_gui_from_storage()
            setter.assert_not_called()
        self.assertEqual(window.ch1_compliance.value(), .1)

    def test_failed_compliance_check_blocks_dgs_and_position_scan_before_any_movement(self):
        for index, window in ((1, self.dashboard.dual_window), (7, self.dashboard.position_dgs_window)):
            self.dashboard.tab_widget.setCurrentIndex(index)
            with patch.object(self.station.smu, 'enforce_current_compliance', side_effect=RuntimeError('100 nA readback failed')), \
                 patch.object(self.station.piezo300, '_write_voltage_step') as move, \
                 patch.object(self.station.smu.ch1.set_r, 'volt') as gate, \
                 patch.object(self.station.camera.remote_lf, 'get_spectrum') as acquire:
                window.begin_sweep()
                self.assertFalse(window.is_running)
                self.assertIn('blocked', window.status_lbl.text())
                move.assert_not_called()
                gate.assert_not_called()
                acquire.assert_not_called()
        self.assertEqual(list(self.directory.rglob('*.h5')), [])
        self.assertEqual(self.errors.call_count, 2)

    def test_busy_tab_change_is_rejected_before_any_settings_reload(self):
        dashboard = self.dashboard
        dashboard.tab_widget.setCurrentIndex(1)
        dashboard.dual_window.is_running = True
        try:
            with patch.object(dashboard.dual_window, 'save_storage') as save, \
                 patch.object(dashboard.single_window, 'reload_gui_from_storage') as reload:
                dashboard.tab_widget.setCurrentIndex(0)
                self.assertEqual(dashboard.tab_widget.currentIndex(), 1)
                save.assert_not_called()
                reload.assert_not_called()
        finally:
            dashboard.dual_window.is_running = False
        dashboard.tab_widget.setCurrentIndex(0)
        dashboard.single_window.is_running = True
        dashboard.single_window.active_acquisition_mode = 'continuous'
        try:
            dashboard.tab_widget.setCurrentIndex(5)
            self.assertEqual(dashboard.tab_widget.currentIndex(), 5)
            dashboard.tab_widget.setCurrentIndex(0)
            self.assertEqual(dashboard.tab_widget.currentIndex(), 0)
        finally:
            dashboard.single_window.is_running = False

    def test_output_enable_ramp_does_not_block_gui_and_is_stoppable(self):
        self.dashboard.tab_widget.setCurrentIndex(1)
        window = self.dashboard.dual_window
        entered, release = threading.Event(), threading.Event()
        def slow_enable(enabled):
            entered.set()
            release.wait(3)
        with patch.object(self.station.smu.ch1.output, 'set', side_effect=slow_enable):
            try:
                window.ch1_out_btn.click()
                self.wait_for(entered.is_set)
                from PyQt6 import QtCore
                ticks = []
                timer = QtCore.QTimer()
                timer.timeout.connect(lambda: ticks.append(True))
                timer.start(5)
                QtTest.QTest.qWait(70)
                timer.stop()
                self.assertGreaterEqual(len(ticks), 3)
                self.assertTrue(window.ch1_stop_btn.isEnabled())
                self.assertFalse(window.begin_btn.isEnabled())
                self.assertTrue(self.dashboard.tab_widget.isTabEnabled(0))
                self.assertFalse(self.dashboard.tab_widget.isTabEnabled(2))
                self.assertFalse(self.dashboard.tab_widget.isTabEnabled(7))
                self.dashboard.tab_widget.setCurrentIndex(0)
                self.assertEqual(self.dashboard.tab_widget.currentIndex(), 0)
                self.assertTrue(window.ramp_worker.isRunning())
                self.dashboard.single_window.is_running = True
                self.dashboard.single_window.active_acquisition_mode = 'continuous'
                self.dashboard.check_measurement_status()
                self.dashboard.tab_widget.setCurrentIndex(1)
                self.assertEqual(self.dashboard.tab_widget.currentIndex(), 1)
                self.dashboard.single_window.is_running = False
                with patch.object(self.station.camera.remote_lf, 'get_spectrum') as capture:
                    window.begin_sweep()
                    capture.assert_not_called()
                    self.assertIsNone(window.sweep_worker)
                window.ch1_stop_btn.click()
                self.assertTrue(window.ramp_worker._stop_requested)
            finally:
                self.dashboard.single_window.is_running = False
                release.set()
                self.wait_for(lambda: not window.ramp_worker.isRunning())
                self.app.processEvents()
        self.errors.assert_not_called()

    def test_keysight_buttons_show_current_state_and_toggle_only_selected_channel(self):
        self.dashboard.tab_widget.setCurrentIndex(1)
        window = self.dashboard.dual_window
        for axis in (1, 2):
            channel = getattr(self.station.smu, f"ch{axis}")
            other = getattr(self.station.smu, f"ch{3-axis}")
            button = getattr(window, f"ch{axis}_out_btn")
            self.assertFalse(channel.output())
            self.assertEqual(button.text(), "OFF")
            button.click()
            self.wait_for(lambda: window.ramp_worker is not None and not window.ramp_worker.isRunning() and button.text() == "ON" and button.isEnabled())
            self.assertTrue(channel.output())
            self.assertTrue(button.isChecked())
            self.assertEqual(button.text(), "ON")
            self.assertFalse(other.output())
            button.click()
            self.assertFalse(channel.output())
            self.assertFalse(button.isChecked())
            self.assertEqual(button.text(), "OFF")
        self.errors.assert_not_called()

    def test_keysight_startup_preserves_enabled_outputs_and_shows_on_state(self):
        from measurements.Dual_gate_sweep_gui import MainWindow, LiveStatusWorker
        ch1, ch2 = self.station.smu.ch1, self.station.smu.ch2
        ch1.output(True)
        ch2.output(True)
        with patch.object(ch1.output, "set", wraps=ch1.output.set) as set1, \
             patch.object(ch2.output, "set", wraps=ch2.output.set) as set2, \
             patch.object(LiveStatusWorker, "start"):
            window = MainWindow(self.station)
            try:
                for button in (window.ch1_out_btn, window.ch2_out_btn):
                    self.assertTrue(button.isChecked())
                    self.assertEqual(button.text(), "ON")
                self.assertTrue(ch1.output())
                self.assertTrue(ch2.output())
                set1.assert_not_called()
                set2.assert_not_called()
            finally:
                window.close()

    def test_simulated_gate_sweep_saves_coordinates_voltages_and_raw_frames(self):
        window = self.dashboard.dual_window
        self.dashboard.tab_widget.setCurrentIndex(1)
        for axis in (1, 2):
            getattr(window, f"c{axis}_start").setValue(0)
            getattr(window, f"c{axis}_end").setValue(0.01)
            getattr(window, f"c{axis}_pts").setValue(2)
        window.exposure_box.setValue(0.01)
        window.settling_box.setValue(0)
        window.begin_sweep()
        self.wait_for(lambda: not window.is_running, timeout=10)
        self.assertTrue(np.all(window.data_measured_mask))
        files = list(self.directory.rglob("*gate_sweep*.h5"))
        self.assertEqual(len(files), 1)
        with h5py.File(files[0]) as data:
            for name in ("c1_coords", "c2_coords", "v1_grid", "v2_grid",
                         "measured_volt_ch1", "measured_volt_ch2", "wavelength_axis", "raw_intensities"):
                self.assertIn(name, data)
            self.assertEqual(data["spectra"].shape, (2, 2, 512))
            self.assertEqual(json.loads(data.attrs["device_modes"])["smu"], "simulated")
        self.errors.assert_not_called()

    def test_dgs_optional_end_ramp_returns_gates_to_zero_and_is_persisted(self):
        window = self.dashboard.dual_window
        self.dashboard.tab_widget.setCurrentIndex(1)
        for axis, end in ((1, 0.3), (2, -0.2)):
            getattr(window, f"c{axis}_start").setValue(end / 2)
            getattr(window, f"c{axis}_end").setValue(end)
            getattr(window, f"c{axis}_pts").setValue(2)
        window.exposure_box.setValue(0.01)
        window.settling_box.setValue(0)
        # Raster order, so the last point is (C1 end, C2 end) = (0.3, -0.2).
        window.sweep_mode_combo.setCurrentText("Hold C2, sweep C1")
        window.serpentine_box.setChecked(False)
        # Default off: the gates hold the last sweep point, as before.
        self.assertFalse(window.ramp_zero_box.isChecked())
        window.begin_sweep()
        self.wait_for(lambda: not window.is_running, timeout=10)
        self.assertAlmostEqual(self.station.smu.ch1.volt(), 0.3)
        self.assertAlmostEqual(self.station.smu.ch2.volt(), -0.2)
        window.ramp_zero_box.setChecked(True)
        self.assertTrue(json.loads((self.directory / "value_storage").read_text())["dual_gate"]["ramp_to_zero"])
        window.begin_sweep()
        self.assertFalse(window.ramp_zero_box.isEnabled())
        self.wait_for(lambda: not window.is_running, timeout=10)
        self.assertTrue(np.all(window.data_measured_mask))
        self.assertEqual(self.station.smu.ch1.volt(), 0.0)
        self.assertEqual(self.station.smu.ch2.volt(), 0.0)
        self.assertTrue(window.ramp_zero_box.isEnabled())
        # Position DGS reuses this panel but never offers the end ramp.
        self.assertTrue(self.dashboard.position_dgs_window.ramp_zero_box.isHidden())
        self.assertFalse(self.dashboard.position_dgs_window.ramp_zero_box.isChecked())
        self.errors.assert_not_called()

    def test_position_dgs_records_all_positions_and_reopens_without_motion(self):
        window = self.dashboard.position_dgs_window
        calibration_x = np.asarray(self.station.camera.remote_lf.get_wavelengths_cache())
        self.dashboard.single_window.bg_x = calibration_x
        self.dashboard.single_window.bg_y = np.full(calibration_x.size, 2.)
        self.dashboard.single_window.ref_x = calibration_x
        self.dashboard.single_window.ref_y = np.full(calibration_x.size, 10.)
        self.dashboard.single_window.num_spectra_box.setValue(2)
        scanner = self.station.piezo300
        self.assertTrue(scanner.simulated)
        scanner._write_voltage_step(1, 9.)
        scanner._write_voltage_step(2, 32.)
        self.dashboard.tab_widget.setCurrentIndex(7)
        for axis, start, end, points in ((1, -.01, .01, 3), (2, .02, .03, 2)):
            getattr(window, f"c{axis}_start").setValue(start)
            getattr(window, f"c{axis}_end").setValue(end)
            getattr(window, f"c{axis}_pts").setValue(points)
        for axis in ('x', 'y'):
            window.scan_inputs[axis + '_size'].setValue(0.01)
            window.scan_inputs[axis + '_step'].setValue(0.01)
        window.scan_inputs['settle'].setValue(0)
        window.exposure_box.setValue(0.001)
        window.settling_box.setValue(0)
        motion, acquired_at = [], []
        raw_by_point = {}
        setter = scanner._write_voltage_step
        def move(axis, target):
            motion.append((axis, target, scanner.get_voltage(1), scanner.get_voltage(2)))
            return setter(axis, target)
        def spectrum():
            x, y = scanner.get_voltage(1), scanner.get_voltage(2)
            acquired_at.append((x, y))
            ix, iy = round((x - 9) / .01), round((y - 32) / .01)
            i = round((self.station.smu.ch1.volt() + .01) / .01)
            j = round((self.station.smu.ch2.volt() - .02) / .01)
            # Distinct data for every scanner pixel, gate pair, frame and wavelength.
            signal = 1 + 1000 * iy + 100 * ix + 10 * j + i + np.arange(512) / 1000
            raw = np.stack((signal, signal + 1))
            raw_by_point[iy, ix, j, i] = raw
            return raw, calibration_x
        with patch.object(scanner, '_write_voltage_step', side_effect=move), \
             patch.object(self.station.camera.remote_lf, 'get_spectrum', side_effect=spectrum):
            window.begin_sweep()
            self.assertTrue(window.is_running)
            self.assertTrue(self.dashboard.measurement_active())
            self.assertFalse(self.dashboard.prepare_device_change())
            self.dashboard.check_measurement_status()
            self.assertFalse(self.dashboard.tab_widget.isTabEnabled(5))
            self.wait_for(lambda: not window.is_running, timeout=20)
        np.testing.assert_allclose(acquired_at, np.repeat([[9, 32], [9.01, 32], [9, 32.01], [9.01, 32.01]], 6, axis=0))
        self.assertTrue(any(axis == 2 for axis, *_ in motion))
        for axis, target, x_before, y_before in motion:
            if axis == 2: self.assertAlmostEqual(x_before, 9)  # X returned before EVERY Y step.
        self.errors.assert_not_called()
        self.assertTrue(window.sweep_worker.completed)
        with h5py.File(window.store.path) as data:
            self.assertEqual(data['spectra'].shape, (2, 2, 2, 3, 512))
            self.assertEqual(data['raw_intensities'].shape, (2, 2, 2, 3, 2, 512))
            self.assertTrue(data['measured_mask'][:].all())
            self.assertTrue(data['position_complete'][:].all())
            self.assertEqual(data['spectra'].dims[0].label, 'y_voltages')
            self.assertEqual(data['spectra'].dims[1].label, 'x_voltages')
            self.assertEqual(len(raw_by_point), 24)
            for key, raw in raw_by_point.items():
                np.testing.assert_allclose(data['raw_intensities'][key], raw[:, window.energy_sort_idx])
            per_pixel = data['spectra'][:]
            expected = data['spectra'][0, 1]
            self.assertEqual(data.attrs['status'], 'complete')
        # Browsing pixels must replace all DGS data, without commanding any motion.
        with patch.object(scanner, '_write_voltage_step') as move:
            for iy, ix in ((1, 1), (0, 0), (1, 0), (0, 1)):
                window.select_position(iy, ix)
                np.testing.assert_array_equal(window.data_spectra, per_pixel[iy, ix])
                np.testing.assert_array_equal(window.spectrum_curve.yData,
                                             per_pixel[iy, ix, np.argmin(abs(window.c2_coords - window.cross1_h.value())),
                                                       np.argmin(abs(window.c1_coords - window.cross1_v.value()))])
                self.assertIn('6/6 gate points', window.selected_label.text())
            move.assert_not_called()
        filename = self.directory / window.default_sweep_filename()
        self.assertTrue(filename.name.startswith('PSDGS_RC_'))
        window.save_sweep_data(str(filename))
        self.assertTrue(filename.exists())
        window.c1_pts.setValue(4)
        self.dashboard.single_window.ref_y = np.full(calibration_x.size, 99.)
        with patch.object(self.station.piezo300, '_write_voltage_step') as move:
            with patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(filename), '')):
                window.open_scan()
            window.select_position(0, 1)
            move.assert_not_called()
        np.testing.assert_array_equal(window.data_spectra, expected)
        with patch.object(window.view_control_popup, 'get_source_option', return_value='(S-R)/(R-B)'):
            np.testing.assert_allclose(window.get_processed_spectra(), (expected - 10) / 8)
        self.assertAlmostEqual(self.station.piezo300.get_voltage(1), 9)
        self.assertAlmostEqual(self.station.piezo300.get_voltage(2), 32.01)
        self.errors.assert_not_called()

    def test_position_dgs_offset_button_reads_fresh_values_without_movement_or_mode_query(self):
        window = self.dashboard.position_dgs_window
        scanner = self.station.piezo300
        scanner._write_voltage_step(1, 9)
        scanner._write_voltage_step(2, 32.099)
        with patch.object(scanner, 'get_axis_status', side_effect=RuntimeError('Mode query unavailable')) as status, \
             patch.object(scanner, '_write_voltage_step') as setter:
            window.read_origin_btn.click()
            self.assertEqual(window.scan_inputs['x_start'].value(), 9)
            self.assertEqual(window.scan_inputs['y_start'].value(), 32.099)
            self.assertTrue(window._scan_origin_loaded)
            self.assertIn('copied to Start', window.scan_info.text())
            self.assertIn('SIMULATED', window.scan_info.text())
            self.assertAlmostEqual(window.scan_inputs['x_size'].value(), .1)
            self.assertAlmostEqual(window.scan_inputs['x_step'].value(), .1)
            status.assert_not_called()
            setter.assert_not_called()
            window.begin_sweep()  # Reading does not bypass movement validation.
            self.assertFalse(window.is_running)
            setter.assert_not_called()
        self.errors.assert_called_once()
        scanner._write_voltage_step(1, 9.25)
        window.read_origin_btn.click()
        self.assertEqual(window.scan_inputs['x_start'].value(), 9.25)

    def test_position_dgs_failed_offset_read_preserves_start_values_and_blocks_scan(self):
        window = self.dashboard.position_dgs_window
        window.scan_inputs['x_start'].setValue(9)
        window.scan_inputs['y_start'].setValue(32)
        with patch.object(self.station.piezo300, 'get_voltage', side_effect=[10, RuntimeError('Disconnected')]), \
             patch.object(self.station.piezo300, '_write_voltage_step') as setter:
            window.read_origin_btn.click()
            self.assertEqual(window.scan_inputs['x_start'].value(), 9)
            self.assertEqual(window.scan_inputs['y_start'].value(), 32)
            self.assertFalse(window._scan_origin_loaded)
            self.assertIn('Disconnected', window.scan_info.text())
            window.begin_sweep()
            setter.assert_not_called()
        self.assertFalse(window.is_running)

    def test_position_dgs_invalid_session_then_reconnect_refreshes_open_gui(self):
        from pyvisa.errors import InvalidSession
        self.dashboard.tab_widget.setCurrentIndex(7)
        window = self.dashboard.position_dgs_window
        old = self.station.piezo300
        other = self.station.smu
        saved = np.ones((2, 2, 512))
        window.data_spectra = saved
        with patch.object(old, 'get_voltage', side_effect=InvalidSession()):
            window.read_origin_btn.click()
        self.assertFalse(window._scan_origin_loaded)
        self.assertIn('connection is closed', window.scan_info.text())
        self.assertIn('Connect / Retry', window.scan_info.text())
        self.assertTrue(self.dashboard.prepare_device_change())
        remove_device(self.station, 'piezo300')
        configure_device(self.station, 'piezo300', 'Simulation')
        new = self.station.piezo300
        new._write_voltage_step(1, 9.25)
        new._write_voltage_step(2, 32.099)
        with patch.object(new, '_write_voltage_step') as move:
            self.dashboard.finish_device_change()
            move.assert_not_called()
        self.assertTrue(window._scan_origin_loaded)
        self.assertEqual(window.scan_inputs['x_start'].value(), 9.25)
        self.assertEqual(window.scan_inputs['y_start'].value(), 32.099)
        self.assertIn('copied to Start', window.scan_info.text())
        self.assertIs(window.data_spectra, saved)
        self.assertIs(self.station.smu, other)
        self.assertIs(self.dashboard.position_tab.scanner_controller, new)
        self.errors.assert_not_called()

    def test_position_dgs_preview_explains_span_steps_and_limit_after_read(self):
        window = self.dashboard.position_dgs_window
        self.station.piezo300.set_voltage(1, 9)
        window.read_origin_btn.click()
        window.scan_inputs['x_size'].setValue(.2)
        window.scan_inputs['x_step'].setValue(.1)
        window.scan_inputs['y_size'].setValue(0)
        self.assertIn('X: 9 → 9.2 V (3 positions)', window.scan_preview.text())
        self.assertIn('3 positions total', window.scan_preview.text())
        self.station.piezo300._voltages[1] = 61  # External controller change, outside software limits.
        window.read_origin_btn.click()
        self.assertEqual(window.scan_inputs['x_start'].value(), 61)
        self.assertIn('Scan settings invalid', window.scan_preview.text())
        with patch.object(self.station.piezo300, '_write_voltage_step') as setter:
            window.begin_sweep()
            setter.assert_not_called()
        self.errors.assert_called_once()

    def test_position_dgs_over_limit_fails_before_scanner_or_gate_writes(self):
        window = self.dashboard.position_dgs_window
        window.read_scan_origin()
        window.scan_inputs['x_start'].setValue(59.9)
        window.scan_inputs['x_size'].setValue(0.2)
        with patch.object(self.station.piezo300, '_write_voltage_step') as move, \
             patch.object(self.station.smu.ch1.volt, 'set') as gate:
            window.begin_sweep()
            move.assert_not_called()
            gate.assert_not_called()
        self.assertFalse(window.is_running)
        self.assertIsNone(window.store)
        self.assertIn('60', window.status_lbl.text())
        self.errors.assert_called_once()

    def test_position_dgs_stop_keeps_partial_data_without_return_motion(self):
        window = self.dashboard.position_dgs_window
        self.dashboard.tab_widget.setCurrentIndex(7)
        for axis in (1, 2):
            getattr(window, f'c{axis}_pts').setValue(1)
            getattr(window, f'c{axis}_start').setValue(0)
        window.scan_inputs['x_size'].setValue(.01)
        window.scan_inputs['x_step'].setValue(.01)
        window.scan_inputs['y_size'].setValue(0)
        window.scan_inputs['settle'].setValue(0)
        window.settling_box.setValue(0)
        window.exposure_box.setValue(.001)
        waiting, release = threading.Event(), threading.Event()
        acquire = self.station.camera.remote_lf.get_spectrum
        calls = []
        def spectrum():
            calls.append(1)
            if len(calls) == 2:
                waiting.set()
                release.wait(timeout=5)
            return acquire()
        with patch.object(self.station.camera.remote_lf, 'get_spectrum', side_effect=spectrum):
            try:
                window.begin_sweep()
                self.wait_for(waiting.is_set)
                window.pause_sweep()
                self.assertTrue(window.is_paused)
                self.assertFalse(window.read_origin_btn.isEnabled())
                self.assertFalse(self.dashboard.prepare_device_change())
                window.reset_sweep()
            finally:
                release.set()
                self.wait_for(lambda: not window.is_running)
        with h5py.File(window.store.path) as data:
            self.assertEqual(data.attrs['status'], 'interrupted')
            self.assertEqual(np.count_nonzero(data['measured_mask']), 1)
        self.assertAlmostEqual(self.station.piezo300.get_voltage(1), .01)
        self.assertTrue(window.save_btn.isEnabled())
        self.errors.assert_not_called()

    def test_position_dgs_acquisition_failure_retains_first_point(self):
        window = self.dashboard.position_dgs_window
        self.dashboard.tab_widget.setCurrentIndex(7)
        for axis in (1, 2):
            getattr(window, f'c{axis}_pts').setValue(1)
            getattr(window, f'c{axis}_start').setValue(0)
        window.scan_inputs['x_size'].setValue(.01)
        window.scan_inputs['x_step'].setValue(.01)
        window.scan_inputs['y_size'].setValue(0)
        window.scan_inputs['settle'].setValue(0)
        window.exposure_box.setValue(.001)
        window.settling_box.setValue(0)
        acquire = self.station.camera.remote_lf.get_spectrum
        calls = []
        def spectrum():
            calls.append(1)
            if len(calls) == 2: raise RuntimeError('Simulated camera disconnect')
            return acquire()
        with patch.object(self.station.camera.remote_lf, 'get_spectrum', side_effect=spectrum):
            window.begin_sweep()
            self.wait_for(lambda: not window.is_running)
        with h5py.File(window.store.path) as data:
            self.assertEqual(data.attrs['status'], 'error')
            self.assertIn('disconnect', data.attrs['error'])
            self.assertEqual(np.count_nonzero(data['measured_mask']), 1)
        self.assertAlmostEqual(self.station.piezo300.get_voltage(1), .01)
        self.errors.assert_called_once()

    def test_live_error_stops_cleanly_without_saving_old_data(self):
        window = self.dashboard.single_window
        saved = np.array([11., 12.])
        window.signal_x = np.array([700., 701.])
        window.signal_y = saved
        with patch.object(self.station.camera.remote_lf, "get_latest_preview_data", side_effect=RuntimeError("Camera disconnected")):
            window.run_btn.click()
            self.wait_for(lambda: not window.is_running)
        self.errors.assert_called_once()
        self.assertIn("Camera disconnected", window.status_lbl.text())
        self.assertIs(window.signal_y, saved)
        self.assertEqual(list(self.directory.rglob("*.h5")), [])
        self.assertTrue(window.run_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
