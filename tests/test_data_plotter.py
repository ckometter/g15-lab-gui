"""The Data plotter is a separate, read-only program.

It must never mutate a measurement setup, contact devices, lock or write files, or
disturb a measurement program that is writing the file being viewed.
"""
import contextlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import h5py
import numpy as np
from PyQt6 import QtCore, QtTest, QtWidgets

import data_plotter
from data_plotter import DataPlotterWindow, ReadOnlyScanStore, describe, read_dataset
from measurements.capacitance_gui import write_capacitance_file
from measurements.transport_gui import write_transport_file
from position_dgs import PositionScanStore
import test_dashboard_workflow as dashboard_checks

REPO = Path(data_plotter.__file__).resolve().parent
MEASUREMENT_MODULES = ('single_spectra_gui', 'Dual_gate_sweep_gui', 'B_V_sweep_gui', 'photoluminescence_map_gui',
                       'scalar_map_gui', 'polarization_control_gui', 'position_dgs_gui', 'gate_map_gui',
                       'capacitance_gui', 'transport_gui')


def make_dgs_file(path, c1_name='Vmu'):
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
        f.attrs['c1_name'], f.attrs['c2_name'] = c1_name, 'VE'
        f.attrs['coeff_a'], f.attrs['coeff_d'] = 0.5, 0.5
        f.attrs['exposure_time'] = 0.2
        f.attrs['device_modes'] = json.dumps({'camera': 'simulated', 'smu': 'physical'})
        f['background/wavelength'] = [1000, 1100, 1300]
        f['background/intensity'] = [2, 2, 2]
    return path


def make_spectrum_file(path, wavelengths=(800, 810, 820), intensities=(100, 200, 150)):
    with h5py.File(path, 'w') as f:
        f['signal/wavelength'] = list(wavelengths)
        f['signal/intensity'] = list(intensities)
    return path


def make_position_scan_file(path):
    """Two scanner positions with two gates each; position (0, 1) lacks its second gate."""
    store = PositionScanStore(path)
    store.create(x=[9, 9.1], y=[32], c1=[0, 1], c2=[0], wavelengths=[800, 810, 820], frames=1,
                 v1=np.array([[0, 1]]), v2=np.array([[0, 0]]),
                 metadata={'gate_coefficients': [1, 0, 0, 1], 'c1_name': 'Top', 'c2_name': 'Bottom'})
    with h5py.File(path, 'r+') as f:
        f['spectra'][0, 0] = [[[1, 2, 3], [4, 5, 6]]]
        f['spectra'][0, 1] = [[[7, 8, 9], [10, 11, 12]]]
        f['measured_mask'][:] = True
        f['measured_mask'][0, 1, 0, 1] = False
    return path


def _datasets(f, values):
    for name, data in values.items():
        f.create_dataset(name, data=data)


def make_bv_file(path):
    """The datasets of BVSweepWindow.save_sweep_data: control x field, spectra in energy order."""
    v, b, energy = np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 1.1, 1.2])
    shape = (b.size, v.size)
    spectra = np.arange(18, dtype=float).reshape(shape + (3,)) + 10
    mask = np.ones(shape, dtype=bool)
    mask[1, 2] = False
    spectra[1, 2] = 0  # The tab allocates zeros; an unmeasured point stays 0.
    with h5py.File(path, 'w') as f:
        _datasets(f, {'v_coords': v, 'b_coords': b, 'v1_grid': np.tile(v, (2, 1)), 'v2_grid': np.zeros(shape),
                      'energy_axis': energy, 'wavelength_axis': 1239.84193 / energy, 'spectra': spectra,
                      'raw_intensities': spectra[:, :, None, :], 'current_ch1': np.full(shape, 1e-12),
                      'current_ch2': np.full(shape, 2e-12), 'measured_v1': np.tile(v, (2, 1)),
                      'measured_v2': np.zeros(shape), 'measured_b': np.tile(b[:, None], (1, 3)),
                      'measured_mask': mask})
        f['background/wavelength'] = [1000, 1100, 1300]
        f['background/intensity'] = [2, 2, 2]
        f.attrs['device_modes'] = json.dumps({'magnet': 'simulated'})
        f.attrs['control_coeff_ch1'], f.attrs['control_coeff_ch2'] = 1.0, 0.5
    return path


def make_map_file(path, vector=True):
    """The datasets of the Vector Map (spectra) or Scalar Map (power) save_data."""
    x, y = np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0])
    shape = (y.size, x.size)
    mask = np.ones(shape, dtype=bool)
    mask[1, 2] = False
    with h5py.File(path, 'w') as f:
        _datasets(f, {'x_coords': x, 'y_coords': y, 'measured_v1': np.zeros(shape), 'measured_v2': np.zeros(shape),
                      'measured_b': np.zeros(shape), 'current_ch1': np.zeros(shape), 'current_ch2': np.zeros(shape),
                      'measured_mask': mask})
        if vector:
            wavelengths = np.array([1100.0, 1050.0, 1000.0])
            spectra = np.arange(18, dtype=float).reshape(shape + (3,)) + 100
            spectra[~mask] = 0
            _datasets(f, {'wavelengths': wavelengths, 'wavelength_axis': wavelengths, 'spectra': spectra,
                          'raw_intensities': spectra[:, :, None, :]})
            params = {'x_param': 'V1', 'y_param': 'B'}
        else:
            power = np.arange(6, dtype=float).reshape(shape) * 1e-6 + 1e-6
            power[~mask] = 0
            f.create_dataset('power', data=power)
            params = {'x_param': 'position X', 'y_param': 'position Y'}
        f.create_dataset('metadata/value_storage', data=json.dumps({'params': params}),
                         dtype=h5py.string_dtype(encoding='utf-8'))
        f.attrs['device_modes'] = json.dumps({'piezo300': 'simulated'})
    return path


def make_gate_map_file(window, path):
    """A finished 3 x 2 run, written by the Capacitance or Transport tab's own file writer."""
    window.x_points_spin.setValue(3)
    window.y_points_spin.setValue(2)
    settings = window.collect_settings()
    plan = window.build_plan(settings)
    mask = np.ones(plan.shape, dtype=bool)
    mask.flat[-1] = False
    data = {key: np.where(mask, np.arange(mask.size, dtype=float).reshape(plan.shape) + 1, np.nan)
            for key in window.POINT_FIELDS}
    data.update({key: np.full(plan.shape, fill, dtype=int) for key, fill in window.INT_FIELDS.items()})
    common = dict(plan=plan, data=data, mask=mask, settings=settings, value_storage={},
                  attrs={'status': 'complete', 'device_modes': {'lockin': 'simulated'}})
    if window.DATA_NOUN == 'capacitance':
        write_capacitance_file(path, balances=[], **common)
    else:
        write_transport_file(path, **common)
    return path


def child_environment():
    """A child process with HDF5's default file locking, as the dashboard runs."""
    environment = dict(os.environ)
    environment.pop('HDF5_USE_FILE_LOCKING', None)
    return environment


def start_python(code, *args):
    return subprocess.Popen([sys.executable, '-c', textwrap.dedent(code), *map(str, args)],
                            cwd=REPO, env=child_environment(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def run_program(*args):
    return subprocess.run([sys.executable, str(REPO / 'data_plotter.py'), *map(str, args)],
                          cwd=REPO, env=child_environment(), capture_output=True, text=True, timeout=120)


def wait_until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out waiting for the condition.')
        time.sleep(0.02)


class DataPlotterWindowTests(unittest.TestCase):
    """The viewer window, over simulated devices, temporary settings and temporary data."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        directory = self.context.enter_context(tempfile.TemporaryDirectory())
        self.directory = Path(directory)
        (self.directory / 'value_storage').write_text(json.dumps({
            'shared': {'exposure_time': 0.01, 'settling_time': 0.0},
            'single_spectra': {'num_spectra': 2},
        }))
        for name in MEASUREMENT_MODULES:
            module = importlib.import_module('measurements.' + name)
            self.context.enter_context(patch.object(module, 'parent_dir', directory))
        # Any instrument, spectrometer-server or network access makes the test fail.
        for target in ('pyvisa.ResourceManager', 'socket.create_connection', 'Pyro4.Proxy'):
            self.context.enter_context(patch(target, side_effect=AssertionError('Device access in a Data plotter test')))
        self.warnings = self.context.enter_context(patch.object(QtWidgets.QMessageBox, 'warning'))
        self.context.enter_context(patch.object(QtWidgets.QMessageBox, 'critical'))
        self.context.enter_context(patch.object(QtWidgets.QMessageBox, 'information'))
        self.station = data_plotter.simulated_station()
        self.viewer = DataPlotterWindow(self.station, start=self.directory)
        self.addCleanup(self.close_viewer)
        self.app.processEvents()
        # Building the analysis windows writes their settings; nothing afterwards may.
        self.storage = self.storage_bytes()

    def close_viewer(self):
        self.wait_for(lambda: not self.viewer.busy() and not self.viewer.position_dgs_window.position_reader_busy())
        self.viewer.close()
        self.app.processEvents()
        for component in list(self.station.components.values()):
            component.close()

    def wait_for(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(20)
        self.assertTrue(predicate())

    def storage_bytes(self):
        return (self.directory / 'value_storage').read_bytes()

    def make_dgs(self, name='example-DGS.h5', c1_name='Vmu'):
        return make_dgs_file(self.directory / name, c1_name)

    def load(self, path, position=None):
        self.viewer.load_file(path, position)
        self.wait_for(lambda: not self.viewer.busy())

    def test_dgs_file_keeps_recorded_grids_calibration_and_missing_points(self):
        window = self.viewer.dual_window
        dataset = read_dataset(self.make_dgs())
        self.viewer.display(dataset)
        self.assertIs(self.viewer.current_window(), window)
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
        self.assertIsNot(window.single_spectra_window, self.viewer.single_window)
        self.assertFalse(window.begin_btn.isEnabled())
        self.assertTrue(window.btn_view_control.isEnabled())
        self.assertIn('Measured gates: 3 / 4 (75%)', self.viewer.info.text())
        self.assertEqual(self.storage_bytes(), self.storage)

    def test_viewer_stops_polling_and_never_calls_hardware_or_saves(self):
        running = [(kind, name) for kind, window in self.viewer.windows.items()
                   for name, value in vars(window).items()
                   if isinstance(value, QtCore.QThread) and value.isRunning()]
        self.assertEqual(running, [])
        self.assertFalse(self.viewer.single_window.status_worker.isRunning())
        self.assertFalse(self.viewer.transport_window.live_status_worker.isRunning())
        tabs = [self.viewer.tab_widget.tabText(index) for index in range(self.viewer.tab_widget.count())]
        self.assertEqual(tabs, ['Spectrum', 'Dual-Gate Sweep', 'B/V Sweep', 'Vector Map', 'Scalar Map',
                                'Position DGS', 'Capacitance', 'Transport'])
        for window in self.viewer.windows.values():
            # Acquisition controls are hidden; every tab keeps its plots.
            self.assertTrue(all(control.isHidden() for control in data_plotter._measurement_controls(window)))
        fail = Mock(side_effect=AssertionError('Analysis must not call hardware'))
        with patch.object(self.station.smu.ch1, 'volt', fail), \
             patch.object(self.station.smu.ch2, 'volt', fail), \
             patch.object(self.station.piezo300, 'get_voltage', fail), \
             patch.object(self.station.camera, 'acquire_spectrum', fail, create=True), \
             patch.object(self.station.camera, 'exposure_time', fail):
            for index in (2, 1, 0):
                self.viewer.tab_widget.setCurrentIndex(index)
            self.viewer.position_dgs_window.read_scan_origin()
            self.viewer.dual_window.begin_sweep()
            self.viewer.single_window.start_acquisition('single')
            self.viewer.display(read_dataset(self.make_dgs()))
            exported = self.directory / 'must-not-export.h5'
            self.viewer.dual_window.save_sweep_data(str(exported))
            self.viewer.dual_window.trigger_autosave()
            self.viewer.position_dgs_window.save_sweep_data(str(exported))
            QtTest.QTest.qWait(250)
            fail.assert_not_called()
        self.assertEqual([path.name for path in self.directory.rglob('*.h5')], ['example-DGS.h5'])
        self.assertEqual(self.storage_bytes(), self.storage)

    def test_browser_filter_fit_open_and_dataset_details(self):
        path = self.make_dgs()
        viewer = self.viewer
        self.assertEqual(viewer.folder.text(), str(self.directory))
        viewer.search.setText('example')
        self.assertEqual(viewer.model.nameFilters(), ['*example*.h5', '*example*.hdf5'])
        viewer.open_index(QtCore.QModelIndex())
        self.assertFalse(viewer.busy())
        self.wait_for(lambda: viewer.model.index(str(path)).isValid())  # The listing loads in Qt's thread.
        viewer.open_index(viewer.model.index(str(path)))
        self.wait_for(lambda: not viewer.busy())
        self.assertIn('Measured gates: 3 / 4 (75%)', viewer.info.text())
        self.assertIn('Spectral samples: 3', viewer.info.text())
        self.assertIn('Recorded simulated devices: camera', viewer.info.text())
        window = viewer.dual_window
        self.assertEqual(window.plotter_file, str(path.resolve()))
        region = window.spectrum_v_region.getRegion()
        with patch.object(window, 'scale_plots', wraps=window.scale_plots) as scale:
            viewer.fit_button.click()
            scale.assert_called_once()
        self.assertEqual(window.spectrum_v_region.getRegion(), region)
        self.assertTrue(np.isfinite(window.spectrum_win.getViewBox().viewRange()).all())
        viewer.search.clear()
        self.assertEqual(viewer.model.nameFilters(), ['**.h5', '**.hdf5'])

    def test_spectrum_loads_in_a_worker_thread_while_browsing_continues(self):
        window = self.viewer.single_window
        path = make_spectrum_file(self.directory / 'spectrum.h5')
        self.viewer.load_file(path)
        self.assertTrue(self.viewer.busy())
        self.assertFalse(self.viewer.progress.isHidden())
        self.assertTrue(self.viewer.tree.isEnabled())
        self.wait_for(lambda: not self.viewer.busy())
        self.assertEqual(window.plotter_file, str(path.resolve()))
        self.assertTrue(self.viewer.progress.isHidden())
        self.assertTrue(self.viewer.reload_button.isEnabled())
        self.assertIn('Signal: 3 samples, 800', self.viewer.info.text())
        np.testing.assert_array_equal(window.plot_curves['Signal'].yData, [100, 200, 150])

    def test_file_chosen_during_a_load_is_opened_next_and_skipped_ones_are_not(self):
        dgs = self.make_dgs()
        spectrum = make_spectrum_file(self.directory / 'spectrum.h5')
        position = make_position_scan_file(self.directory / 'position.h5')
        self.viewer.load_file(dgs)
        self.viewer.load_file(spectrum)
        self.viewer.load_file(position)
        self.wait_for(lambda: getattr(self.viewer.position_dgs_window, 'plotter_file', None) is not None)
        self.wait_for(lambda: not self.viewer.busy())
        self.assertEqual(self.viewer.dual_window.plotter_file, str(dgs.resolve()))
        self.assertIsNone(getattr(self.viewer.single_window, 'plotter_file', None))
        self.assertIs(self.viewer.current_window(), self.viewer.position_dgs_window)

    def test_descending_calibration_keeps_frames_aligned_and_interpolates_correctly(self):
        path = self.directory / 'descending-spectrum.h5'
        with h5py.File(path, 'w') as f:
            f['signal/wavelength'] = [800, 810, 820]
            f['signal/intensity'] = [100, 200, 150]
            f['background/wavelength'] = [820, 810, 800]
            f['background/intensity'] = [30, 20, 10]
            f['background/raw_intensities'] = [[31, 21, 11], [29, 19, 9]]
        original = path.read_bytes()
        self.viewer.display(read_dataset(path))
        window = self.viewer.single_window
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
        self.viewer.display(read_dataset(path))
        window = self.viewer.position_dgs_window
        self.assertEqual(window.selected_position, (0, 1))
        self.assertTrue(np.isfinite(window.data_spectra).all())
        self.assertIn('Measured gates across scan: 1 / 2 (50%)', self.viewer.info.text())
        self.assertIn('Scanner positions with data: 1 / 2', self.viewer.info.text())

    def test_invalid_calibration_is_rejected_without_replacing_display(self):
        self.viewer.display(read_dataset(self.make_dgs()))
        original = self.viewer.dual_window.data_spectra
        path = self.directory / 'bad-calibration.h5'
        with h5py.File(path, 'w') as f:
            f['signal/wavelength'] = [800, 800]
            f['signal/intensity'] = [1, 2]
        started = time.monotonic()
        self.load(path)
        self.assertLess(time.monotonic() - started, 3)  # A stable invalid file is not retried.
        self.warnings.assert_called_once()
        self.assertIn('wavelengths must be unique', self.viewer.info.text())
        self.assertIs(self.viewer.dual_window.data_spectra, original)
        self.assertTrue(self.viewer.tree.isEnabled())

    def test_display_error_restores_the_previous_file(self):
        window = self.viewer.dual_window
        first = read_dataset(self.make_dgs())
        self.viewer.display(first)
        shown = window.data_spectra
        second = read_dataset(self.make_dgs('other-DGS.h5', c1_name='Other'))
        draw = window.update_heatmaps_and_slices
        calls = []

        def fail_once(*args, **kwargs):
            calls.append(None)
            if len(calls) == 1:
                raise RuntimeError('drawing failed')
            return draw(*args, **kwargs)

        with patch.object(window, 'update_heatmaps_and_slices', side_effect=fail_once):
            self.viewer.display(second)
        self.warnings.assert_called_once()
        self.assertIn('drawing failed', self.viewer.info.text())
        self.assertIs(window.data_spectra, shown)
        self.assertEqual(window.plotter_file, first['path'])
        self.assertEqual(window.c1_name_edit.text(), 'Vmu')

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
        self.viewer.display(dataset)
        window = self.viewer.dual_window
        self.assertGreater(np.nanmax(window.gatemap_img.image), 0)
        self.assertTrue(np.isnan(window.gatemap_img.image[1, 1]))
        self.assertEqual(path.read_bytes(), original)

    def test_position_pixels_are_read_off_the_gui_thread_and_only_the_last_click_is_shown(self):
        path = make_position_scan_file(self.directory / 'position.h5')
        window = self.viewer.position_dgs_window
        start = window.scan_inputs['x_start'].value()
        self.viewer.display(read_dataset(path))
        self.assertIs(self.viewer.current_window(), window)
        self.assertEqual(window.scan_inputs['x_start'].value(), start)
        np.testing.assert_array_equal(window.data_spectra[0, 0], [1, 2, 3])
        threads = []
        real_file = h5py.File

        def record(*args, **kwargs):
            threads.append(threading.current_thread())
            return real_file(*args, **kwargs)

        with patch.object(h5py, 'File', side_effect=record):
            window.select_position(0, 1)
            self.assertTrue(window.position_reader_busy())  # The GUI thread returned before reading.
            self.assertIn('Reading the DGS', window.selected_label.text())
            self.wait_for(lambda: window.selected_position == (0, 1) and not window.position_reader_busy())
            np.testing.assert_array_equal(window.data_spectra[0, 0], [7, 8, 9])
            self.assertTrue(np.isnan(window.data_spectra[0, 1]).all())
            self.assertFalse(window.data_measured_mask[0, 1])
            window.select_position(0, 0)
            window.select_position(0, 1)
            self.wait_for(lambda: not window.position_reader_busy())
        self.assertEqual(window.selected_position, (0, 1))
        np.testing.assert_array_equal(window.data_spectra[0, 0], [7, 8, 9])
        self.assertEqual(len(threads), 3)
        self.assertNotIn(threading.main_thread(), threads)

    def test_reload_shows_points_added_since_and_keeps_the_selected_pixel(self):
        path = make_position_scan_file(self.directory / 'position.h5')
        window = self.viewer.position_dgs_window
        self.load(path, position=(0, 1))
        self.assertEqual(window.selected_position, (0, 1))
        self.assertTrue(np.isnan(window.data_spectra[0, 1]).all())
        with h5py.File(path, 'r+') as f:  # Meanwhile the scan measures the missing gate.
            f['measured_mask'][0, 1, 0, 1] = True
            f.attrs['status'] = 'running'
        self.viewer.reload()
        self.wait_for(lambda: not self.viewer.busy())
        self.assertEqual(window.selected_position, (0, 1))
        np.testing.assert_array_equal(window.data_spectra[0, 1], [10, 11, 12])
        self.assertIn('Reload shows newer positions', self.viewer.info.text())

    def test_bv_sweep_file_opens_in_its_tab_and_is_not_taken_for_a_spectrum(self):
        path = make_bv_file(self.directory / 'bv_sweep.h5')
        self.load(path)
        window = self.viewer.bv_window
        self.assertIs(self.viewer.current_window(), window)
        self.assertEqual(window.plotter_file, str(path.resolve()))
        np.testing.assert_array_equal(window.v_coords, [0, 0.5, 1])
        np.testing.assert_array_equal(window.b_coords, [0, 1])
        self.assertEqual((window.v_pts.value(), window.b_pts.value()), (3, 2))
        self.assertEqual(window.control_coeff_ch2.value(), 0.5)
        self.assertEqual(window.gatemap_img.image.shape, (2, 3))
        self.assertGreater(window.gatemap_img.image[0, 0], 0)
        # The file's own background, not whatever the Spectrum tab shows.
        self.assertIsNot(window.single_spectra_window, self.viewer.single_window)
        window.view_control_popup.combo_source.setCurrentText('Signal - BG')  # The B/V tab's name.
        self.assertEqual(window.view_control_popup.get_source_option(), 'Signal - BG')
        with h5py.File(path, 'r') as f:
            np.testing.assert_allclose(window.get_processed_spectra(), f['spectra'][:] - 2)
        self.assertIn('Control: 0 to 1 V (3 points)', self.viewer.info.text())
        self.assertIn('Field: 0 to 1 T (2 points)', self.viewer.info.text())
        self.assertIn('Measured points: 5 / 6 (83%)', self.viewer.info.text())
        self.assertIn('Unmeasured points show 0', self.viewer.info.text())
        self.assertEqual(describe(read_dataset(path)),
                         'B/V Sweep: 3 control x 2 field points, 3 spectral samples; 5 / 6 points measured')
        self.assertEqual(self.storage_bytes(), self.storage)

    def test_vector_and_scalar_maps_open_with_their_recorded_axes(self):
        vector = make_map_file(self.directory / 'vector_map.h5')
        scalar = make_map_file(self.directory / 'scalar_map.h5', vector=False)
        self.load(vector)
        window = self.viewer.pl_map_window
        self.assertIs(self.viewer.current_window(), window)
        self.assertEqual((window.x_param_cb.currentText(), window.y_param_cb.currentText()), ('V1', 'B'))
        self.assertEqual(window.map_img.image.shape, (2, 3))
        self.assertGreater(window.map_img.image[0, 0], 0)
        self.assertEqual((window.cross_v.value(), window.cross_h.value()), (0.0, 5.0))
        self.assertEqual(len(window.spectrum_curve.xData), 3)
        self.assertIn('V1: 0 to 2 (3 points)', self.viewer.info.text())
        self.assertIn('Spectral samples: 3', self.viewer.info.text())
        self.load(scalar)
        window = self.viewer.scalar_map_window
        self.assertIs(self.viewer.current_window(), window)
        self.assertEqual(window.x_param_cb.currentText(), 'position X')
        np.testing.assert_allclose(window.map_img.image, [[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 0]])
        np.testing.assert_allclose(window.slice_x_curve.yData, [1e-6, 2e-6, 3e-6])
        self.assertIn('Measured points: 5 / 6 (83%)', self.viewer.info.text())
        self.assertEqual(describe(read_dataset(vector)),
                         'Vector Map: 3 x 2 points, 3 spectral samples; 5 / 6 points measured')
        self.assertEqual(describe(read_dataset(scalar)), 'Scalar Map: 3 x 2 points; 5 / 6 points measured')
        with patch.object(window.map_plot, 'enableAutoRange', wraps=window.map_plot.enableAutoRange) as fit:
            self.viewer.fit_button.click()
        fit.assert_called()
        self.assertEqual(self.storage_bytes(), self.storage)

    def test_capacitance_and_transport_files_open_as_their_tabs_own_open_shows_them(self):
        for kind, window in (('capacitance', self.viewer.capacitance_window),
                             ('transport', self.viewer.transport_window)):
            with self.subTest(kind):
                path = make_gate_map_file(window, self.directory / f'{kind}.h5')
                self.load(path)
                self.assertIs(self.viewer.current_window(), window)
                self.assertEqual(window.loaded_path, str(path.resolve()))
                self.assertEqual(window.plan.shape, (2, 3))
                np.testing.assert_array_equal(window.measured_mask, [[True, True, True], [True, True, False]])
                self.assertEqual(window.run_outcome, 'complete')
                self.assertFalse(window.is_running)
                self.assertIn('Measured points: 5 / 6 (83%)', self.viewer.info.text())
                self.assertIn(f'{window.plan.names[0]}: ', self.viewer.info.text())  # The file's axis name.
                self.assertIn('Recorded simulated devices: lockin', self.viewer.info.text())
                self.assertEqual(describe(read_dataset(path)),
                                 f'{kind.capitalize()}: 3 x 2 points; 5 / 6 points measured; recorded status complete')
        self.assertFalse(self.viewer.transport_window.live_status_worker.isRunning())

    def test_measurement_files_of_other_programs_are_refused(self):
        path = self.directory / 'levelmeter.h5'
        with h5py.File(path, 'w') as f:
            f['main_data/level'] = [1.0, 2.0]
        with self.assertRaisesRegex(ValueError, 'Choose a saved measurement'):
            read_dataset(path)

    def test_file_without_measured_mask_is_rejected_and_display_kept(self):
        window = self.viewer.dual_window
        original = window.data_spectra
        path = self.make_dgs()
        with h5py.File(path, 'r+') as f:
            del f['measured_mask']
        with self.assertRaisesRegex(ValueError, 'missing measured_mask'):
            read_dataset(path)
        self.assertIs(window.data_spectra, original)

    def test_every_file_open_is_read_only_and_lock_free(self):
        paths = (self.make_dgs(), make_spectrum_file(self.directory / 'spectrum.h5'),
                 make_position_scan_file(self.directory / 'position.h5'))
        opens = []
        real_file = h5py.File

        def record(name, mode='r', *args, **kwargs):
            opens.append((mode, kwargs.get('locking')))
            return real_file(name, mode, *args, **kwargs)

        with patch.object(h5py, 'File', side_effect=record):
            for path in paths:
                self.load(path)
            window = self.viewer.position_dgs_window
            window.select_position(0, 1)
            self.wait_for(lambda: window.selected_position == (0, 1) and not window.position_reader_busy())
            self.viewer.reload()
            self.wait_for(lambda: not self.viewer.busy())
        self.assertGreaterEqual(len(opens), 5)
        self.assertEqual(set(opens), {('r', False)})

    def test_settings_stay_private_and_smu_ramp_settings_are_not_stored(self):
        channel = self.station.smu.ch1
        with patch.object(type(channel), '_save_param_to_storage') as stored:
            channel.ramp_rate(0.123)
        stored.assert_not_called()
        source, target = self.directory / 'source', self.directory / 'target'
        source.mkdir()
        target.mkdir()
        (source / 'value_storage').write_text('{"shared": {}}')
        modules = [importlib.import_module(name) for name in data_plotter.SETTINGS_MODULES]
        try:
            data_plotter.use_private_settings(target, source=source)
            self.assertEqual((target / 'value_storage').read_text(), '{"shared": {}}')
            self.assertEqual({module.parent_dir for module in modules}, {str(target)})
        finally:
            for module in modules:
                module.parent_dir = str(self.directory)

    def test_closing_during_a_read_is_deferred_without_blocking_the_window(self):
        path = self.make_dgs()
        release = threading.Event()
        read_file = data_plotter._read_file

        def slow_read(*args, **kwargs):
            release.wait(10)
            return read_file(*args, **kwargs)

        with patch.object(data_plotter, '_read_file', side_effect=slow_read):
            self.viewer.load_file(path)
            self.assertFalse(self.viewer.close())
            self.assertIn('Closing when the current read ends', self.viewer.info.text())
            self.viewer.load_file(make_spectrum_file(self.directory / 'spectrum.h5'))
            self.assertIsNotNone(self.viewer.model)
            release.set()
            self.wait_for(lambda: self.viewer.model is None)
        self.assertIsNone(getattr(self.viewer.dual_window, 'plotter_file', None))
        self.assertIsNone(getattr(self.viewer.single_window, 'plotter_file', None))


class FileSafetyTests(unittest.TestCase):
    """Reading files that another process may be writing, as the viewer process does."""

    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    @unittest.skipUnless(sys.platform == 'win32', 'Windows refuses reads while a writer holds its lock.')
    def test_file_held_open_by_a_writer_process_is_read_after_the_writer_closes_it(self):
        path = make_dgs_file(self.directory / 'being-saved.h5')
        opened = self.directory / 'opened'
        writer = start_python('''
            import sys, time
            from pathlib import Path
            import h5py
            with h5py.File(sys.argv[1], 'r+') as f:  # The default lock, as the dashboard writes.
                Path(sys.argv[2]).touch()
                time.sleep(1.0)
                f.attrs['saved_while_viewed'] = True
            print('closed')
        ''', path, opened)
        self.addCleanup(writer.kill)
        wait_until(lambda: opened.exists() or writer.poll() is not None)
        started = time.monotonic()
        dataset = read_dataset(path)
        waited = time.monotonic() - started
        output, errors = writer.communicate(timeout=30)
        self.assertEqual(writer.returncode, 0, errors)
        self.assertEqual(output.strip(), 'closed')
        self.assertGreater(waited, 0.5)
        self.assertTrue(dataset['attrs']['saved_while_viewed'])
        self.assertFalse(dataset['changing'])

    def test_position_scan_writer_process_is_never_disturbed_by_viewer_reads(self):
        path = self.directory / 'running-position-scan.h5'
        created = self.directory / 'created'
        # Points every 0.2 s: faster than a real Position DGS, whose points each wait for an exposure.
        writer = start_python('''
            import json, sys, time
            from pathlib import Path
            import numpy as np
            from position_dgs import PositionScanStore
            store = PositionScanStore(sys.argv[1])
            store.create(x=[0.0, 0.1, 0.2], y=[0.0, 0.1], c1=[0, 1, 2, 3, 4], c2=[0, 1, 2, 3],
                         wavelengths=np.linspace(800, 900, 64), frames=1, v1=np.zeros((4, 5)), v2=np.zeros((4, 5)),
                         metadata={'gate_coefficients': [1, 0, 0, 1], 'c1_name': 'Top', 'c2_name': 'Bottom'})
            store.set_status('running')
            Path(sys.argv[2]).touch()
            errors, points, index = [], 0, 0
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                iy, ix, j, i = (int(n) for n in np.unravel_index(index % 120, (2, 3, 4, 5)))
                point = (i, j, 0.0, 0.0, 0.0, 0.0, np.random.rand(64), 1e-12, 1e-12, 0.0, 0.0, 0.0,
                         np.random.rand(1, 64))
                try:
                    store.write_point(iy, ix, point)  # Opens the file r+ with the default lock.
                    points += 1
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {exc}')
                index += 1
                time.sleep(0.2)
            store.set_status('complete')
            print(json.dumps({'points': points, 'errors': errors}))
        ''', path, created)
        self.addCleanup(writer.kill)
        wait_until(lambda: created.exists() or writer.poll() is not None)
        # The viewer program itself, in its own process, while the scan writes.
        check = run_program('--check', path)
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
        self.assertIn('Position DGS: 3 x 2 positions, 5 x 4 gates, 64 spectral samples', check.stdout)
        reads = busy = 0
        while writer.poll() is None:
            try:
                dataset = read_dataset(path)
                ReadOnlyScanStore(path).read_from_disk(*dataset['position'])
                reads += 1
            except OSError as exc:  # A read may give up while the scan writes; the scan never fails.
                self.assertIn('stayed busy', str(exc))
                busy += 1
        output, errors = writer.communicate(timeout=30)
        self.assertEqual(writer.returncode, 0, errors)
        summary = json.loads(output.strip().splitlines()[-1])
        self.assertEqual(summary['errors'], [])
        self.assertGreaterEqual(summary['points'], 20)
        self.assertGreater(reads, 5)
        final = read_dataset(path)
        self.assertEqual(final['attrs']['status'], 'complete')
        self.assertEqual(int(final['position_counts'].sum()), min(summary['points'], 120))

    def test_stable_unreadable_file_fails_at_once(self):
        path = self.directory / 'not-hdf5.h5'
        path.write_bytes(b'not an HDF5 file')
        started = time.monotonic()
        with self.assertRaises(OSError):
            read_dataset(path)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_check_option_prints_a_summary_and_reports_unreadable_files(self):
        good = make_dgs_file(self.directory / 'example-DGS.h5')
        bad = self.directory / 'not-hdf5.h5'
        bad.write_bytes(b'not an HDF5 file')
        result = run_program('--check', good, bad)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('Dual-Gate Sweep: 2 x 2 gates, 3 spectral samples; 3 / 4 gate points measured', result.stdout)
        self.assertIn('not-hdf5.h5: cannot be shown', result.stdout)
        usage = run_program('--check')
        self.assertEqual(usage.returncode, 2)
        self.assertIn('usage', usage.stderr)

    def test_device_access_is_refused_in_the_viewer_process(self):
        import socket
        import Pyro4
        import pyvisa
        import serial
        targets = [(socket, 'create_connection'), (pyvisa, 'ResourceManager'), (Pyro4, 'Proxy'), (serial, 'Serial')]
        originals = [(module, name, getattr(module, name)) for module, name in targets]
        try:
            data_plotter.block_device_access()
            for module, name in targets:
                self.assertIs(getattr(module, name), data_plotter._refuse_device_access, name)
        finally:
            for module, name, original in originals:
                setattr(module, name, original)
        with self.assertRaisesRegex(RuntimeError, 'only reads files'):
            data_plotter._refuse_device_access(('127.0.0.1', 9))

    def test_launch_starts_a_detached_lock_free_low_priority_process(self):
        popen = Mock()
        log = self.directory / 'viewer.log'
        with patch.object(data_plotter, 'LOG_FILE', log):
            process = data_plotter.launch([self.directory], popen=popen)
        self.assertIs(process, popen.return_value)
        (command,), options = popen.call_args
        self.assertEqual(command, [sys.executable, str(REPO / 'data_plotter.py'), str(self.directory)])
        self.assertEqual(options['cwd'], str(REPO))
        self.assertEqual(options['env']['HDF5_USE_FILE_LOCKING'], 'FALSE')
        self.assertEqual(options['env']['PYTHONUNBUFFERED'], '1')
        self.assertEqual(options['stdin'], subprocess.DEVNULL)
        self.assertEqual(options['stderr'], subprocess.STDOUT)
        self.assertIn('Data plotter started', log.read_text(encoding='utf-8'))
        if sys.platform == 'win32':
            for flag in (subprocess.CREATE_NO_WINDOW, subprocess.CREATE_NEW_PROCESS_GROUP,
                         subprocess.BELOW_NORMAL_PRIORITY_CLASS):
                self.assertTrue(options['creationflags'] & flag)
        popen.return_value.wait.assert_not_called()
        popen.return_value.communicate.assert_not_called()


class DashboardLauncherTests(unittest.TestCase):
    """The dashboard only starts the separate program; it has no plotter mode any more."""
    setUpClass = classmethod(dashboard_checks.DashboardWorkflowTests.setUpClass.__func__)
    setUp = dashboard_checks.DashboardWorkflowTests.setUp
    close_dashboard = dashboard_checks.DashboardWorkflowTests.close_dashboard
    wait_for = dashboard_checks.DashboardWorkflowTests.wait_for

    def test_dashboard_opens_the_viewer_during_a_measurement_without_changing_it(self):
        dashboard = self.dashboard
        for name in ('mode_slider', 'set_plotter_mode', 'data_plotter', 'plotter_mode'):
            self.assertFalse(hasattr(dashboard, name), name)
        window = dashboard.dual_window
        dashboard.tab_widget.setCurrentIndex(1)
        window.is_running = True
        try:
            dashboard.check_measurement_status()
            tabs = [dashboard.tab_widget.isTabEnabled(index) for index in range(dashboard.tab_widget.count())]
            self.assertTrue(dashboard.plotter_button.isEnabled())
            with patch('data_plotter.launch') as launch:
                dashboard.plotter_button.click()
            launch.assert_called_once_with()
            self.assertIn('Data plotter is starting', dashboard.statusBar().currentMessage())
            self.assertTrue(window.is_running)
            self.assertFalse(window.plotter_mode)
            dashboard.check_measurement_status()
            self.assertEqual(dashboard.tab_widget.currentIndex(), 1)
            self.assertEqual([dashboard.tab_widget.isTabEnabled(index)
                              for index in range(dashboard.tab_widget.count())], tabs)
        finally:
            window.is_running = False
            dashboard.check_measurement_status()

    def test_a_failed_start_is_reported_in_the_status_bar(self):
        with patch('data_plotter.launch', side_effect=OSError('python not found')):
            self.dashboard.plotter_button.click()
        self.assertIn('Could not start the Data plotter: python not found', self.dashboard.statusBar().currentMessage())


if __name__ == '__main__':
    unittest.main()
