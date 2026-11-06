"""Read-only plotting must never mutate a measurement setup or contact devices."""
import json
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from PyQt6 import QtCore, QtTest, QtWidgets

import test_dashboard_workflow as dashboard_checks
from data_plotter import apply_dataset, read_dataset
from position_dgs import PositionScanStore


class DataPlotterTests(unittest.TestCase):
    setUpClass = classmethod(dashboard_checks.DashboardWorkflowTests.setUpClass.__func__)
    setUp = dashboard_checks.DashboardWorkflowTests.setUp
    close_dashboard = dashboard_checks.DashboardWorkflowTests.close_dashboard
    wait_for = dashboard_checks.DashboardWorkflowTests.wait_for

    def enter_plotter(self):
        self.assertIsNone(self.dashboard.data_plotter.model)
        self.assertTrue(self.dashboard.set_plotter_mode(True))
        self.wait_for(lambda: not self.dashboard._plotter_transition)
        self.assertIsNotNone(self.dashboard.data_plotter.model)

    def make_dgs(self):
        path = self.directory / 'example-DGS.h5'
        with h5py.File(path, 'w') as f:
            for name, values in {'c1_coords': [0, 1], 'c2_coords': [2, 3],
                                 'energy_axis': [1.0, 1.1, 1.2],
                                 'wavelength_axis': 1239.84193 / np.array([1.0, 1.1, 1.2])}.items():
                f[name] = values
            f['v1_grid'] = [[0, 2], [0, 2]]
            f['v2_grid'] = [[4, 4], [6, 6]]
            f['spectra'] = np.arange(12).reshape(2, 2, 3) + 20
            f['raw_intensities'] = f['spectra'][:][:, :, None, :]
            f['current_ch1'] = np.ones((2, 2)) * 1e-10
            f['current_ch2'] = np.ones((2, 2)) * 2e-10
            f['measured_mask'] = [[True, True], [True, False]]
            f.attrs['c1_name'], f.attrs['c2_name'] = 'Vmu', 'VE'
            f.attrs['coeff_a'], f.attrs['coeff_d'] = 0.5, 0.5
            f.attrs['exposure_time'] = 0.2
            f.attrs['device_modes'] = json.dumps({'camera': 'simulated', 'smu': 'physical'})
            f['background/wavelength'] = [1000, 1100, 1300]
            f['background/intensity'] = [2, 2, 2]
        return path

    def test_dgs_load_preserves_recorded_grids_calibration_and_restores_measurement(self):
        window = self.dashboard.dual_window
        initial_spectra = window.data_spectra
        initial_c1 = window.c1_start.value()
        initial_coeff = window.coeff_a.value()
        initial_storage = (self.directory / 'value_storage').read_bytes()
        self.enter_plotter()
        dataset = read_dataset(self.make_dgs())
        self.dashboard.data_plotter.display(dataset)
        np.testing.assert_array_equal(window.v1_grid, [[0, 2], [0, 2]])
        self.assertEqual(window.c1_name_edit.text(), 'Vmu')
        self.assertEqual(window.coeff_a.value(), 0.5)
        self.assertGreater(np.nanmax(window.gatemap_img.image), 0)
        self.assertTrue(np.isnan(window.gatemap_img.image[1, 1]))
        self.assertAlmostEqual(window.vslice_img.mapRectToParent(window.vslice_img.boundingRect()).top(), 2.0)
        np.testing.assert_array_equal(window.data_measured_mask, [[True, True], [True, False]])
        window.view_control_popup.combo_source.setCurrentText('Signal - Background')
        expected = np.where(dataset['data']['measured_mask'][:, :, None], dataset['data']['spectra'] - 2, np.nan)
        np.testing.assert_allclose(window.get_processed_spectra(), expected)
        self.assertTrue(np.isnan(window.data_current_ch1[1, 1]))
        self.assertTrue(np.isnan(window.data_raw_spectra[1, 1]).all())
        window.update_crosshair_values(1, 3, 1.1)
        window.on_crosshair_moved()
        window.scale_plots()
        shown = window.spectrum_curve.yData
        self.assertTrue(shown is None or not np.isfinite(shown).any())
        for plot in (window.spectrum_win, window.vslice_win, window.hslice_win):
            self.assertTrue(np.isfinite(plot.getViewBox().viewRange()).all())
        self.assertTrue(np.isfinite(window.vslice_colorbar.getLevels()).all())
        self.assertTrue(np.isfinite(window.hslice_colorbar.getLevels()).all())
        with h5py.File(dataset['path'], 'r') as f:
            self.assertTrue(np.isfinite(f['spectra'][1, 1]).all())
        self.assertIsNot(window.single_spectra_window, self.dashboard.single_window)
        self.assertEqual((self.directory / 'value_storage').read_bytes(), initial_storage)
        self.assertFalse(window.begin_btn.isEnabled())
        self.assertTrue(window.btn_view_control.isEnabled())
        self.assertTrue(self.dashboard.set_plotter_mode(False))
        self.assertIsNone(self.dashboard.data_plotter.model)
        self.assertIs(window.data_spectra, initial_spectra)
        self.assertEqual(window.c1_start.value(), initial_c1)
        self.assertEqual(window.coeff_a.value(), initial_coeff)
        self.assertIs(window.single_spectra_window, self.dashboard.single_window)
        self.assertEqual((self.directory / 'value_storage').read_bytes(), initial_storage)

    def test_only_supported_tabs_and_no_hardware_calls_during_analysis(self):
        self.enter_plotter()
        for index in range(8):
            self.assertEqual(self.dashboard.tab_widget.isTabVisible(index), index in (0, 1, 7))
            self.assertEqual(self.dashboard.tab_widget.isTabEnabled(index), index in (0, 1, 7))
        self.assertTrue(all(not worker.isRunning() for worker in self.dashboard._plotter_pollers))
        fail = Mock(side_effect=AssertionError('Analysis must not call hardware'))
        with patch.object(self.station.smu.ch1, 'volt', fail), \
             patch.object(self.station.smu.ch2, 'volt', fail), \
             patch.object(self.station.piezo300, 'get_voltage', fail), \
             patch.object(self.station.camera, 'acquire_spectrum', fail, create=True), \
             patch.object(self.station.camera, 'exposure_time', fail):
            for index in (7, 1, 0):
                self.dashboard.tab_widget.setCurrentIndex(index)
                self.dashboard.check_measurement_status()
            self.dashboard.position_dgs_window.read_scan_origin()
            self.dashboard.dual_window.begin_sweep()
            self.dashboard.single_window.start_acquisition('single')
            self.dashboard.data_plotter.display(read_dataset(self.make_dgs()))
            exported = self.directory / 'must-not-export.h5'
            self.dashboard.dual_window.save_sweep_data(str(exported))
            self.dashboard.dual_window.trigger_autosave()
            self.dashboard.position_dgs_window.save_sweep_data(str(exported))
            self.assertFalse(exported.exists())
            QtTest.QTest.qWait(250)
            fail.assert_not_called()
        self.assertFalse(self.dashboard.prepare_device_change())

    def test_browser_filter_fit_and_dataset_details(self):
        path = self.make_dgs()
        self.enter_plotter()
        plotter = self.dashboard.data_plotter
        plotter.set_folder(self.directory)
        plotter.search.setText('example')
        self.assertEqual(plotter.model.nameFilters(), ['*example*.h5', '*example*.hdf5'])
        plotter.open_index(QtCore.QModelIndex())
        plotter.display(read_dataset(path))
        self.assertIn('Measured gates: 3 / 4 (75%)', plotter.info.text())
        self.assertIn('Spectral samples: 3', plotter.info.text())
        self.assertIn('Recorded simulated devices: camera', plotter.info.text())
        window = self.dashboard.dual_window
        region = window.spectrum_v_region.getRegion()
        with patch.object(window, 'scale_plots', wraps=window.scale_plots) as scale:
            plotter.fit_button.click()
            scale.assert_called_once()
        self.assertEqual(window.spectrum_v_region.getRegion(), region)
        self.assertTrue(np.isfinite(window.spectrum_win.getViewBox().viewRange()).all())
        plotter.search.clear()
        self.assertEqual(plotter.model.nameFilters(), ['**.h5', '**.hdf5'])

    def test_spectrum_async_browser_loading_and_live_data_restored(self):
        window = self.dashboard.single_window
        original_x, original_y = np.array([750., 760.]), np.array([9., 10.])
        window.signal_x, window.signal_y = original_x, original_y
        window._live_autoscale = False
        path = self.directory / 'spectrum.h5'
        with h5py.File(path, 'w') as f:
            f['signal/wavelength'] = [800, 810, 820]
            f['signal/intensity'] = [100, 200, 150]
        self.enter_plotter()
        self.dashboard.data_plotter.load_file(path)
        self.assertFalse(self.dashboard.data_plotter.progress.isHidden())
        self.assertFalse(self.dashboard.data_plotter.tree.isEnabled())
        self.wait_for(lambda: not self.dashboard.data_plotter.busy())
        self.wait_for(lambda: getattr(window, 'plotter_file', None) is not None)
        self.wait_for(lambda: self.dashboard.data_plotter.progress.isHidden())
        self.assertTrue(self.dashboard.data_plotter.tree.isEnabled())
        self.assertIn('Signal: 3 samples, 800', self.dashboard.data_plotter.info.text())
        np.testing.assert_array_equal(window.plot_curves['Signal'].yData, [100, 200, 150])
        self.assertTrue(self.dashboard.set_plotter_mode(False))
        self.assertIs(window.signal_x, original_x)
        self.assertIs(window.signal_y, original_y)
        self.assertFalse(window._live_autoscale)

    def test_descending_calibration_keeps_frames_aligned_and_interpolates_correctly(self):
        path = self.directory / 'descending-spectrum.h5'
        with h5py.File(path, 'w') as f:
            f['signal/wavelength'] = [800, 810, 820]
            f['signal/intensity'] = [100, 200, 150]
            f['background/wavelength'] = [820, 810, 800]
            f['background/intensity'] = [30, 20, 10]
            f['background/raw_intensities'] = [[31, 21, 11], [29, 19, 9]]
        original = path.read_bytes()
        self.enter_plotter()
        self.dashboard.data_plotter.display(read_dataset(path))
        window = self.dashboard.single_window
        _, subtracted = window._get_data_for_mode('Signal - Background')
        np.testing.assert_array_equal(subtracted, [90, 180, 120])
        np.testing.assert_array_equal(window.bg_y_raw, [[11, 21, 31], [9, 19, 29]])
        self.assertEqual(path.read_bytes(), original)

    def test_partial_position_scan_opens_first_position_with_data(self):
        path = self.directory / 'partial-position.h5'
        store = PositionScanStore(path)
        store.create(x=[9, 9.1], y=[32], c1=[0], c2=[0], wavelengths=[800, 810, 820], frames=1,
                     v1=np.array([[0]]), v2=np.array([[0]]),
                     metadata={'gate_coefficients': [1, 0, 0, 1]})
        with h5py.File(path, 'r+') as f:
            f['spectra'][0, 1] = [[[7, 8, 9]]]
            f['raw_intensities'][0, 1] = [[[[7, 8, 9]]]]
            f['current_ch1'][0, 1] = f['current_ch2'][0, 1] = [[0]]
            f['measured_mask'][0, 1] = True
        self.enter_plotter()
        self.dashboard.data_plotter.display(read_dataset(path))
        window = self.dashboard.position_dgs_window
        self.assertEqual(window.selected_position, (0, 1))
        self.assertTrue(np.isfinite(window.data_spectra).all())
        self.assertIn('Measured gates across scan: 1 / 2 (50%)', self.dashboard.data_plotter.info.text())

    def test_invalid_calibration_is_rejected_without_replacing_display(self):
        self.enter_plotter()
        plotter = self.dashboard.data_plotter
        plotter.display(read_dataset(self.make_dgs()))
        original = self.dashboard.dual_window.data_spectra
        path = self.directory / 'bad-calibration.h5'
        with h5py.File(path, 'w') as f:
            f['signal/wavelength'] = [800, 800]
            f['signal/intensity'] = [1, 2]
        with patch.object(QtWidgets.QMessageBox, 'warning') as warning:
            plotter.load_file(path)
            self.wait_for(lambda: warning.called)
            self.wait_for(lambda: plotter.progress.isHidden())
        self.assertIn('wavelengths must be unique', plotter.info.text())
        self.assertIs(self.dashboard.dual_window.data_spectra, original)
        self.assertTrue(plotter.tree.isEnabled())
        self.assertTrue(self.dashboard.mode_slider.isEnabled())

    def test_legacy_dgs_uses_recorded_energy_when_wavelength_axis_is_absent(self):
        path = self.make_dgs()
        with h5py.File(path, 'r+') as f:
            del f['wavelength_axis']
            del f['raw_intensities']
            f['background/raw_intensities'] = [2, 2, 2]
        original = path.read_bytes()
        dataset = read_dataset(path)
        np.testing.assert_allclose(dataset['axes']['wavelength_axis'],
                                   1239.84193 / dataset['axes']['energy_axis'])
        np.testing.assert_array_equal(dataset['calibration'].bg_y_raw, [[2, 2, 2]])
        self.enter_plotter()
        self.dashboard.data_plotter.display(dataset)
        window = self.dashboard.dual_window
        self.assertGreater(np.nanmax(window.gatemap_img.image), 0)
        self.assertTrue(np.isnan(window.gatemap_img.image[1, 1]))
        self.assertEqual(path.read_bytes(), original)

    def test_finished_loader_keeps_measurement_locked_until_queued_result_is_handled(self):
        path = self.make_dgs()
        window = self.dashboard.dual_window
        original = window.data_spectra
        self.enter_plotter()
        plotter = self.dashboard.data_plotter
        plotter.load_file(path)
        # Finish the native worker without delivering its GUI signals yet.
        self.assertTrue(plotter.worker.wait(5000))
        self.assertTrue(plotter.busy())
        self.assertFalse(self.dashboard.set_plotter_mode(False))
        self.wait_for(lambda: not plotter.busy())
        self.assertEqual(window.plotter_file, str(path.resolve()))
        self.assertTrue(self.dashboard.set_plotter_mode(False))
        self.assertIs(window.data_spectra, original)
        # Even a delayed callback must be harmless once measurement is active.
        plotter.display(read_dataset(path))
        self.assertIs(window.data_spectra, original)

    def test_position_scan_pixels_keep_independent_dgs_and_load_no_scanner_offsets(self):
        path = self.directory / 'position.h5'
        store = PositionScanStore(path)
        store.create(x=[9, 9.1], y=[32], c1=[0, 1], c2=[0], wavelengths=[800, 810, 820], frames=1,
                     v1=np.array([[0, 1]]), v2=np.array([[0, 0]]),
                     metadata={'gate_coefficients': [1, 0, 0, 1], 'c1_name': 'Top', 'c2_name': 'Bottom'})
        with h5py.File(path, 'r+') as f:
            f['spectra'][0, 0] = [[[1, 2, 3], [4, 5, 6]]]
            f['spectra'][0, 1] = [[[7, 8, 9], [10, 11, 12]]]
            f['measured_mask'][:] = True
            f['measured_mask'][0, 1, 0, 1] = False
        self.enter_plotter()
        window = self.dashboard.position_dgs_window
        start = window.scan_inputs['x_start'].value()
        self.dashboard.data_plotter.display(read_dataset(path))
        self.assertEqual(self.dashboard.tab_widget.currentIndex(), 7)
        self.assertEqual(window.scan_inputs['x_start'].value(), start)
        np.testing.assert_array_equal(window.data_spectra[0, 0], [1, 2, 3])
        window.select_position(0, 1)
        np.testing.assert_array_equal(window.data_spectra[0, 0], [7, 8, 9])
        self.assertTrue(np.isnan(window.data_spectra[0, 1]).all())
        self.assertFalse(window.data_measured_mask[0, 1])
        self.assertTrue(self.dashboard.set_plotter_mode(False))
        self.assertIsNone(window.store)
        self.assertIsNone(window.selected_position)

    def test_active_measurement_blocks_mode_switch_and_invalid_file_preserves_data(self):
        self.dashboard.dual_window.is_running = True
        try:
            self.assertFalse(self.dashboard.set_plotter_mode(True))
            self.assertFalse(self.dashboard.plotter_mode)
        finally:
            self.dashboard.dual_window.is_running = False
        self.enter_plotter()
        window = self.dashboard.dual_window
        original = window.data_spectra
        path = self.make_dgs()
        with h5py.File(path, 'r+') as f:
            del f['measured_mask']
        with self.assertRaisesRegex(ValueError, 'missing measured_mask'):
            read_dataset(path)
        self.assertIs(window.data_spectra, original)

    def test_manual_ramp_locks_mode_tabs_and_requests_stop_before_close(self):
        from PyQt6 import QtGui
        worker = Mock()
        worker.isRunning.return_value = True
        window = self.dashboard.dual_window
        window.ramp_worker = worker
        try:
            self.dashboard.check_measurement_status()
            self.assertFalse(self.dashboard.mode_slider.isEnabled())
            self.assertTrue(self.dashboard.tab_widget.isTabEnabled(0))
            self.assertTrue(self.dashboard.tab_widget.isTabEnabled(1))
            self.assertFalse(self.dashboard.tab_widget.isTabEnabled(2))
            self.assertFalse(self.dashboard.set_plotter_mode(True))
            self.assertFalse(self.dashboard.prepare_device_change())
            event = QtGui.QCloseEvent()
            with patch.object(QtCore.QTimer, 'singleShot') as retry:
                self.dashboard.closeEvent(event)
                worker.stop.assert_called_once()
                self.assertFalse(event.isAccepted())
                self.assertTrue(self.dashboard._parameter_wheel_guard._installed)
                self.assertTrue(window._parameter_wheel_guard._installed)
                retry.assert_called_once()
        finally:
            window.ramp_worker = None
            self.dashboard.check_measurement_status()


if __name__ == '__main__':
    unittest.main()
