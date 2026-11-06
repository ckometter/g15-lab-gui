"""Verify the B/V control equation, persistence, and simulated acquisition."""
import json
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from PyQt6 import QtWidgets

import test_dashboard_workflow as dashboard_checks
from measurements.B_V_sweep_gui import control_to_gate_voltages


class ControlEquationTests(unittest.TestCase):
    def test_gate_targets_satisfy_the_control_equation(self):
        control = np.array([-.2, 0, .3])
        for a, b in ((1, 0), (0, 1), (.5, .5), (.5, -.5), (2, -3)):
            with self.subTest(a=a, b=b):
                ch1, ch2 = control_to_gate_voltages(control, a, b)
                np.testing.assert_allclose(a * ch1 + b * ch2, control)
                np.testing.assert_allclose(-b * ch1 + a * ch2, 0, atol=1e-15)
        ch1, ch2 = control_to_gate_voltages(.2, .5, .5)
        self.assertAlmostEqual(ch1, .2)
        self.assertAlmostEqual(ch2, .2)

    def test_undefined_and_nonfinite_controls_are_rejected(self):
        for control, a, b in ((1, 0, 0), (1, np.nan, 1), (1, 1, np.inf),
                              (np.inf, 1, 0), ([0, np.nan], 1, 1)):
            with self.subTest(control=control, a=a, b=b):
                with self.assertRaises(ValueError):
                    control_to_gate_voltages(control, a, b)


class BVControlWorkflowTests(unittest.TestCase):
    setUpClass = classmethod(dashboard_checks.DashboardWorkflowTests.setUpClass.__func__)
    setUp = dashboard_checks.DashboardWorkflowTests.setUp
    close_dashboard = dashboard_checks.DashboardWorkflowTests.close_dashboard
    wait_for = dashboard_checks.DashboardWorkflowTests.wait_for

    def rebuild_dashboard(self, storage):
        self.close_dashboard()
        (self.directory / 'value_storage').write_text(json.dumps(storage))
        self.station = dashboard_checks.Station()
        self.dashboard = dashboard_checks.LabMainWindow(self.station)
        self.app.processEvents()

    def test_saved_singular_dgs_matrix_opens_without_errors_and_blocks_only_dgs(self):
        singular = dict(coeff_a=0, coeff_b=0, coeff_c=1, coeff_d=0)
        self.rebuild_dashboard({
            'shared': dict(singular, c1_name='field', c2_name='doping'),
            'bv_sweep': dict(control_coeff_ch1=.5, control_coeff_ch2=.5),
        })
        self.errors.assert_not_called()
        for window in (self.dashboard.dual_window, self.dashboard.position_dgs_window):
            self.assertEqual([getattr(window, key).value() for key in singular], [0, 0, 1, 0])
            self.assertFalse(window.begin_btn.isEnabled())
            self.assertIn('singular', window.det_label.text())
            self.assertTrue(np.isnan(window.v1_grid).all())
            self.assertTrue(np.isnan(window.v2_grid).all())
        self.assertTrue(self.dashboard.bv_window.btn_begin.isEnabled())

        window = self.dashboard.dual_window
        was_paused = getattr(self.station, 'live_status_paused', False)
        fail = Mock(side_effect=AssertionError('Invalid DGS must not acquire or ramp'))
        with patch.object(self.station.smu.ch1.set_r, 'volt', fail), \
             patch.object(self.station.smu.ch2.set_r, 'volt', fail), \
             patch.object(self.station.camera.remote_lf, 'get_spectrum', fail):
            window.begin_sweep()  # Also protect direct calls to the controller.
            self.assertFalse(window.is_running)
            self.assertIsNone(window.sweep_worker)
            self.assertEqual(getattr(self.station, 'live_status_paused', False), was_paused)
            fail.assert_not_called()
        self.errors.assert_called_once()

    def test_editing_singular_dgs_coefficients_is_recoverable_without_hardware_writes(self):
        window = self.dashboard.dual_window
        fail = Mock(side_effect=AssertionError('Editing a matrix must not move hardware'))
        with patch.object(self.station.smu.ch1.set_r, 'volt', fail), \
             patch.object(self.station.smu.ch2.set_r, 'volt', fail), \
             patch.object(self.station.camera.remote_lf, 'get_spectrum', fail):
            window.coeff_a.setValue(0)
            window.coeff_a.editingFinished.emit()
            self.assertFalse(window.begin_btn.isEnabled())
            self.assertTrue(np.isnan(window.v1_grid).all())
            # A manual ramp completion or reset must not enable an invalid scan.
            window.on_ramp_finished()
            self.assertFalse(window.begin_btn.isEnabled())
            window.reset_sweep()
            self.assertFalse(window.begin_btn.isEnabled())
            window.coeff_a.setValue(.5)
            window.coeff_a.editingFinished.emit()
            self.assertTrue(window.begin_btn.isEnabled())
            np.testing.assert_allclose(window.v1_grid, 2 * np.broadcast_to(window.c1_coords, window.v1_grid.shape))
            window.reload_gui_from_storage()
            self.assertEqual(window.coeff_a.value(), .5)
            self.assertTrue(window.begin_btn.isEnabled())
            fail.assert_not_called()
        self.errors.assert_not_called()

    def test_dgs_and_bv_coefficients_survive_edits_tab_changes_and_restart_independently(self):
        dual, bv = self.dashboard.dual_window, self.dashboard.bv_window
        matrix = dict(coeff_a=.6, coeff_b=.2, coeff_c=-.4, coeff_d=.8)
        for key, value in matrix.items():
            getattr(dual, key).setValue(value)
        dual.on_sweep_range_changed()
        dual.c1_name_edit.setText('CH1')
        dual.c2_name_edit.setText('CH2')
        # Coordinate labels must not reset user-chosen matrix coefficients.
        self.assertEqual({key: getattr(dual, key).value() for key in matrix}, matrix)
        dual.save_storage()
        bv.control_coeff_ch1.setValue(.5)
        bv.control_coeff_ch2.setValue(-.5)
        bv.on_sweep_range_changed()
        path = self.directory / 'value_storage'
        saved = json.loads(path.read_text())
        self.assertEqual({key: saved['dual_gate'][key] for key in matrix}, matrix)
        self.assertEqual(saved['bv_sweep']['control_coeff_ch1'], .5)
        self.assertEqual(saved['bv_sweep']['control_coeff_ch2'], -.5)
        # Legacy shared values must not override the DGS-owned matrix.
        saved['shared'].update(dict(coeff_a=0, coeff_b=0, coeff_c=1, coeff_d=0))
        path.write_text(json.dumps(saved))
        for index in (1, 2, 1, 2):
            self.dashboard.tab_widget.setCurrentIndex(index)
            self.app.processEvents()
        self.assertEqual({key: getattr(dual, key).value() for key in matrix}, matrix)
        self.assertEqual(bv.control_coeff_ch1.value(), .5)
        self.assertEqual(bv.control_coeff_ch2.value(), -.5)
        self.rebuild_dashboard(json.loads(path.read_text()))
        self.assertEqual({key: getattr(self.dashboard.dual_window, key).value() for key in matrix}, matrix)
        self.assertEqual(self.dashboard.bv_window.control_coeff_ch1.value(), .5)
        self.assertEqual(self.dashboard.bv_window.control_coeff_ch2.value(), -.5)
        self.errors.assert_not_called()

    def test_simplified_controls_persist_separately_from_dgs_and_ignore_legacy_mapping(self):
        window = self.dashboard.bv_window
        for name in ('v_fixed', 'coeff_a', 'coeff_b', 'coeff_c', 'coeff_d',
                     'btn_ch1_only', 'btn_ch2_only'):
            self.assertFalse(hasattr(window, name), name)
        body = self.dashboard.tab_widget.widget(2)
        button_labels = {button.text().lower() for button in body.findChildren(QtWidgets.QPushButton)}
        self.assertNotIn('channel 1 only', button_labels)
        self.assertNotIn('channel 2 only', button_labels)
        self.assertEqual(window.gatemap_win.getAxis('bottom').labelText, 'Control variable')

        path = self.directory / 'value_storage'
        storage = json.loads(path.read_text())
        shared_mapping = dict(coeff_a=.2, coeff_b=.8, coeff_c=-.7, coeff_d=.3,
                              c1_name='field', c2_name='doping')
        storage['shared'].update(shared_mapping)
        storage['bv_sweep'] = {'v_fixed': 17, 'bv_coeff_a': 4, 'bv_coeff_b': 3,
                               'bv_coeff_c': 2, 'bv_coeff_d': 1,
                               'variable_order': 'Hold V, scan B'}
        path.write_text(json.dumps(storage))
        window.reload_gui_from_storage()
        self.assertEqual(window.control_coeff_ch1.value(), 1)
        self.assertEqual(window.control_coeff_ch2.value(), 0)
        self.assertEqual(window.variable_order.currentText(), 'Hold Control, scan B')
        window.control_coeff_ch1.setValue(.5)
        window.control_coeff_ch2.setValue(-.5)
        window.on_sweep_range_changed()
        saved = json.loads(path.read_text())
        self.assertEqual({key: saved['shared'][key] for key in shared_mapping}, shared_mapping)
        self.assertEqual(saved['bv_sweep']['control_coeff_ch1'], .5)
        self.assertEqual(saved['bv_sweep']['control_coeff_ch2'], -.5)
        self.assertNotIn('v_fixed', saved['bv_sweep'])
        window.control_coeff_ch1.setValue(1)
        window.reload_gui_from_storage()
        self.assertEqual(window.control_coeff_ch1.value(), .5)
        np.testing.assert_allclose(.5 * window.v1_grid - .5 * window.v2_grid,
                                   np.broadcast_to(window.v_coords, window.v1_grid.shape))

    def test_invalid_control_path_cannot_start_or_contact_acquisition_devices(self):
        window = self.dashboard.bv_window
        was_paused = getattr(self.station, 'live_status_paused', False)
        fail = Mock(side_effect=AssertionError('Invalid control must not contact devices'))
        with patch.object(QtWidgets.QMessageBox, 'warning') as warning, \
             patch.object(self.station.monochromator, 'wavelength', fail), \
             patch.object(self.station.camera.remote_lf, 'get_wavelengths_cache', fail), \
             patch.object(self.station.camera, 'exposure_time', fail), \
             patch.object(self.station.smu.ch1.set_r, 'volt', fail), \
             patch.object(self.station.smu.ch2.set_r, 'volt', fail):
            for a, b in ((0, 0), (.0001, 0)):
                window.control_coeff_ch1.setValue(a)
                window.control_coeff_ch2.setValue(b)
                window.v_start.setValue(0)
                window.v_end.setValue(.1)
                window.v_pts.setValue(2)
                window.update_coordinates_from_ui()
                self.assertTrue(window.control_mapping_status.text())
                window.begin_sweep()
                self.assertFalse(window.is_running)
                self.assertIsNone(window.sweep_worker)
                self.assertEqual(getattr(self.station, 'live_status_paused', False), was_paused)
            self.assertEqual(warning.call_count, 2)
            fail.assert_not_called()

    def test_simulated_sweep_saves_spectra_and_the_actual_control_mapping(self):
        window = self.dashboard.bv_window
        self.dashboard.single_window.num_spectra_box.setValue(2)
        self.dashboard.single_window.save_value_storage()
        window.control_coeff_ch1.setValue(.5)
        window.control_coeff_ch2.setValue(.5)
        window.v_start.setValue(-.005)
        window.v_end.setValue(.005)
        window.v_pts.setValue(2)
        window.b_start.setValue(0)
        window.b_end.setValue(0)
        window.b_pts.setValue(1)
        window.end_ramp_ground.setChecked(False)
        window.end_ramp_zero.setChecked(False)
        window.begin_sweep()
        self.assertFalse(window.control_coeff_ch1.isEnabled())
        self.wait_for(lambda: not window.is_running)
        self.assertTrue(window.data_measured_mask.all())
        self.assertEqual(window.data_spectra.shape[:2], (1, 2))
        self.assertEqual(window.data_raw_spectra.shape[:3], (1, 2, 2))
        # Simulated voltage measurements include instrument noise.
        np.testing.assert_allclose(window.data_measured_v1, [[-.005, .005]], atol=.001)
        np.testing.assert_allclose(window.data_measured_v2, [[-.005, .005]], atol=.001)
        np.testing.assert_allclose(window.v1_grid, [[-.005, .005]])
        np.testing.assert_allclose(window.v2_grid, [[-.005, .005]])
        self.errors.assert_not_called()

        # Later edits must not rewrite the mapping of an already acquired run.
        window.control_coeff_ch1.setValue(1)
        window.control_coeff_ch2.setValue(0)
        window.update_coordinates_from_ui()
        saved_path = self.directory / 'control-sweep.h5'
        window.save_sweep_data(file_path=str(saved_path))
        with h5py.File(saved_path) as data:
            self.assertEqual(data.attrs['control_variable_definition'], 'Control = a*CH1 + b*CH2')
            self.assertEqual(data.attrs['control_coeff_ch1'], .5)
            self.assertEqual(data.attrs['control_coeff_ch2'], .5)
            self.assertIn('control_gate_mapping', data.attrs)
            np.testing.assert_allclose(data['v1_grid'], [[-.005, .005]])
            np.testing.assert_allclose(data['v2_grid'], [[-.005, .005]])
            np.testing.assert_allclose(.5 * data['v1_grid'][:] + .5 * data['v2_grid'][:],
                                       data['v_coords'][:][None, :])
            self.assertTrue(data['measured_mask'][:].all())
            self.assertEqual(data['raw_intensities'].shape[:3], (1, 2, 2))
            self.assertNotIn('v_fixed', data.attrs)


if __name__ == '__main__':
    unittest.main()
