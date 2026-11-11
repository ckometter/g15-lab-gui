"""Four-probe transport: script parity, lock-in checks, simulation and dashboard runs.

Everything uses simulated instruments or fake I/O; no hardware is opened.
"""
import json
import math
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import transport_lockin as tl
from drivers.qpg_uvoltdaq import UVoltDacChannel
from drivers.stanford_sr830 import CURRENT_SENSITIVITIES, StanfordSR830

SETTINGS = {
    "current_lockin": "lockin", "current_mode": "direct", "preamp_sensitivity": 1e-6, "voltage_gain": 1.0,
    "excitation_divider": 1.0, "series_resistance": 0.0, "time_constant": "keep", "settle_tc": 3.0,
    "configure_excitation": False, "frequency": 17.777, "amplitude": 0.1,
}


def lockin(label, input_config="a", reference="external", frequency=17.777, time_constant=0.3):
    return {"label": label, "input_config": input_config, "reference_source": reference,
            "frequency": frequency, "time_constant": time_constant}


class TransportFormulaTests(unittest.TestCase):
    def test_resistances_match_the_qcodes_qpg_scripts(self):
        ix, vx = 12.5e-9, 3.1e-5
        # transport_measurement_v2.py: r = vx / ix if abs(ix) > 1e-30 else nan
        self.assertAlmostEqual(float(tl.four_probe_resistance(vx, ix)), vx / ix)
        self.assertTrue(math.isnan(float(tl.four_probe_resistance(vx, 0.0))))
        # transport_measurement.py: r4 = z3 / z1 / v_preamp and r2 = 0.016 / z1
        z1, z3, v_preamp = 8e-9, 2.4e-3, 100
        v = float(tl.voltage_from_reading(z3, v_preamp))
        self.assertAlmostEqual(float(tl.four_probe_resistance(v, z1)), z3 / z1 / v_preamp)
        self.assertAlmostEqual(float(tl.two_probe_resistance(0.016, z1)), 0.016 / z1)
        self.assertAlmostEqual(float(tl.two_probe_resistance(1.0, 1e-7, series_resistance=1e7)), 0.0)

    def test_current_conversion_and_settle(self):
        np.testing.assert_allclose(tl.current_from_reading([2e-9, -1e-9], "direct", 1e-6), [2e-9, -1e-9])
        self.assertAlmostEqual(float(tl.current_from_reading(0.25, "preamp", 1e-7)), 2.5e-8)
        # The script's settle_s = 3 * max(tau_I, tau_V); the extra settle is added.
        self.assertAlmostEqual(tl.point_wait([0.1, 0.3], 3.0), 0.9)
        self.assertAlmostEqual(tl.point_wait([0.1, 0.3], 3.0, extra=0.25), 1.15)
        self.assertEqual(tl.sensitivity_for(3e-7, True), 5e-7)
        self.assertEqual(tl.sensitivity_for(1.0, True), CURRENT_SENSITIVITIES[-1])
        self.assertEqual(tl.lockin_roles("lockin2"), {"current": "lockin2", "voltage": "lockin"})
        with self.assertRaises(ValueError):
            tl.lockin_roles("lockin3")

    def test_settings_are_validated_before_any_hardware_access(self):
        tl.validate_transport_settings(SETTINGS)
        for change, message in (({"current_mode": "preamp", "preamp_sensitivity": 0.0}, "preamp sensitivity"),
                                ({"voltage_gain": -1.0}, "gain"),
                                ({"time_constant": 0.25}, "time constant"),
                                ({"configure_excitation": True, "amplitude": 6.0}, "amplitude"),
                                ({"series_resistance": -5.0}, "series")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                tl.validate_transport_settings(dict(SETTINGS, **change))

    def test_input_modes_must_match_the_wiring_and_references_are_checked(self):
        good = lockin("Lock-in 1", "I 1M", "internal"), lockin("Lock-in 2", "a-b")
        self.assertEqual(tl.check_lockin_inputs(*good, SETTINGS), [])
        with self.assertRaisesRegex(RuntimeError, "voltage input 'a'.*Nothing was moved"):
            tl.check_lockin_inputs(lockin("Lock-in 1", "a"), good[1], SETTINGS)
        with self.assertRaisesRegex(RuntimeError, "current preamp is selected"):
            tl.check_lockin_inputs(good[0], good[1], dict(SETTINGS, current_mode="preamp"))
        with self.assertRaisesRegex(RuntimeError, "four-probe voltage but is set to current input"):
            tl.check_lockin_inputs(good[0], lockin("Lock-in 2", "I 100M"), SETTINGS)
        both_internal = tl.check_lockin_inputs(good[0], lockin("Lock-in 2", reference="internal"), SETTINGS)
        self.assertIn("not phase-locked", both_internal[0])
        drift = tl.check_lockin_inputs(good[0], lockin("Lock-in 2", frequency=19.0, time_constant=1.0), SETTINGS)
        self.assertTrue(any("not following" in text for text in drift))
        self.assertTrue(any("time constants differ" in text for text in drift))

    def test_simulated_signals_recover_the_sample_resistance(self):
        sample = tl.SimulatedTransportSample()
        voltages = (-0.3, -1.2, 0.0)
        for mode, gain in (("direct", 1.0), ("preamp", 100.0)):
            settings = dict(SETTINGS, current_mode=mode, preamp_sensitivity=1e-7, voltage_gain=gain)
            signals = tl.SimulatedTransportSignals(sample, lambda: 0.004, lambda: voltages, settings)
            current_reading, voltage_reading = signals.current_input(), signals.voltage_input()
            i = tl.current_from_reading(current_reading.real, mode, 1e-7)
            v = tl.voltage_from_reading(voltage_reading.real, gain)
            # Both signals lag equally, so the in-phase ratio is exactly R4.
            self.assertAlmostEqual(float(tl.four_probe_resistance(v, i)), float(sample.r4(*voltages)), delta=1e-6)
            magnitude = tl.current_from_reading(abs(current_reading), mode, 1e-7)
            self.assertAlmostEqual(float(tl.two_probe_resistance(0.004, magnitude)), float(sample.r2(*voltages)),
                                   delta=1e-6)
            self.assertNotEqual(current_reading.imag, 0.0)
        # The peak sits at charge neutrality and grows with the displacement field.
        self.assertGreater(sample.r4(0.5, -0.5, 0.0), sample.r4(0.5, 0.5, 0.0))
        self.assertGreater(sample.r4(1.0, -1.0, 0.0), sample.r4(0.2, -0.2, 0.0))


class LockinDeviceTests(unittest.TestCase):
    def test_simulated_current_input_reads_amperes(self):
        lockin = StanfordSR830("lockin_current_test", address="simulated")
        self.addCleanup(lockin.close)
        lockin.sensitivity(1e-3)
        lockin.input_config("I 1M")
        self.assertEqual((lockin.sensitivity(), lockin.sensitivity.unit, lockin.X.unit), (1e-9, "A", "A"))
        with self.assertRaises(ValueError):
            lockin.sensitivity(1e-3)  # a voltage range is not a current range
        lockin.signal_source = lambda tc: complex(3e-7, 1e-9)
        lockin.autorange(30)
        self.assertEqual(lockin.sensitivity(), 5e-7)
        self.assertEqual(lockin.snap_xy(), (3e-7, 1e-9))
        lockin.input_config("a")  # same range index: 500 nA <-> 500 mV
        self.assertEqual((lockin.sensitivity(), lockin.sensitivity.unit), (0.5, "V"))

    def test_second_lockin_is_a_device_manager_device(self):
        from qcodes.station import Station
        import station_setup
        from station_devices import DEVICE_LABELS, DEVICE_NAMES, initialize_device
        self.assertEqual(station_setup.LOCKIN2_ADDRESS, "GPIB0::10::INSTR")
        self.assertIn("lockin2", DEVICE_NAMES)
        self.assertIn("transport voltage", DEVICE_LABELS["lockin2"])
        station = Station()
        self.addCleanup(lambda: [c.close() for c in list(station.components.values())])
        self.assertEqual(initialize_device(station, "lockin2", simulated=True), "SIMULATED")
        self.assertTrue(station.lockin2.simulated)


try:
    import test_dashboard_workflow as dashboard_checks
except Exception:  # pragma: no cover - GUI dependencies missing
    dashboard_checks = None


@unittest.skipIf(dashboard_checks is None, "Qt dashboard dependencies are unavailable")
class TransportDashboardTests(unittest.TestCase):
    if dashboard_checks is not None:
        setUpClass = classmethod(dashboard_checks.DashboardWorkflowTests.setUpClass.__func__)
        setUp = dashboard_checks.DashboardWorkflowTests.setUp
        close_dashboard = dashboard_checks.DashboardWorkflowTests.close_dashboard
        wait_for = dashboard_checks.DashboardWorkflowTests.wait_for

    def window(self, **values):
        from lab_gui import TRANSPORT_TAB
        self.dashboard.tab_widget.setCurrentIndex(TRANSPORT_TAB)
        window = self.dashboard.transport_window
        defaults = dict(x_points=5, y_points=4, line_settle=0.0)
        defaults.update(values)
        widgets = {"x_points": window.x_points_spin, "y_points": window.y_points_spin,
                   "line_settle": window.line_settle_spin, "point_settle": window.point_settle_spin,
                   "max_ts": window.max_ts_spin, "preamp": window.preamp_spin, "gain": window.voltage_gain_spin,
                   "series": window.series_spin}
        for key, value in defaults.items():
            widgets[key].setValue(value)
        window.refresh_preview()
        return window

    def run_to_end(self, window, timeout=60):
        window.begin_sweep()
        self.wait_for(lambda: not window.busy(), timeout)
        self.app.processEvents()

    def files(self):
        return sorted(self.directory.rglob("*.h5"))

    def model_r4(self, data):
        vt, vb, vs = (data[f"target_v_{name}"][()] for name in ("top", "bottom", "sample"))
        return tl.SimulatedTransportSample().r4(vt, vb, vs)

    def test_defaults_port_the_script_grid_wiring_and_limits(self):
        window = self.window(x_points=20, y_points=40)
        settings = window.collect_settings()
        self.assertEqual(settings["wiring"], {"top": "dac:1", "bottom": "dac:0", "sample": "none"})
        self.assertEqual((settings["coordinate_mode"], settings["fixed"], settings["fast_axis"]),
                         ("direct", "sample", "y"))
        plan = window.build_plan(settings)
        np.testing.assert_allclose(plan.x, np.linspace(-0.8, 0, 20))   # top gate, slow
        np.testing.assert_allclose(plan.y, np.linspace(-4.5, 0, 40))   # back gate, fast
        self.assertEqual(settings["max_top_sample"], 0.8)
        self.assertTrue(window._preview_ok, window.preview_label.text())

    def test_simulated_map_saves_resistance_and_ramps_to_zero(self):
        window = self.window()
        self.run_to_end(window)
        self.assertEqual(window.run_outcome, "complete", window.run_message)
        self.assertTrue(window.measured_mask.all())
        files = self.files()
        self.assertEqual(len(files), 1)
        self.assertIn("transport_4pt_V_top_m0p8-0V_V_bottom_m4p5-0V_5x4", files[0].name)
        with h5py.File(files[0]) as data:
            self.assertEqual(data.attrs["schema"], "g15/transport_lockin/1")
            self.assertEqual(data.attrs["status"], "complete")
            self.assertEqual(data.attrs["current_lockin"], "lockin")
            self.assertEqual(json.loads(data.attrs["device_modes"])["lockin2"], "simulated")
            self.assertEqual(json.loads(data.attrs["lockin_warnings"]), [])
            self.assertEqual(data["resistance_4pt"].attrs["units"], "Ohm")
            self.assertEqual(data["lockin_current_x"].attrs["units"], "A")
            measured = data["resistance_4pt"][()]
            np.testing.assert_allclose(measured, self.model_r4(data), rtol=1e-2)
            self.assertTrue(np.all(data["measured_mask"][()]))
        self.assertEqual(window.make_electrodes(window.collect_settings()).setpoints(),
                         {"top": 0.0, "bottom": 0.0, "sample": 0.0})
        window.open_saved(str(files[0]))
        np.testing.assert_allclose(window.data["r4"], measured)
        self.errors.assert_not_called()

    def test_preamp_and_voltage_gain_are_converted(self):
        window = self.window(preamp=1e-5, gain=100.0)
        window._set_combo(window.current_mode_combo, "preamp")
        window.refresh_preview()
        self.run_to_end(window)
        self.assertEqual(window.run_outcome, "complete", window.run_message)
        with h5py.File(self.files()[0]) as data:
            np.testing.assert_allclose(data["resistance_4pt"][()], self.model_r4(data), rtol=1e-2)
            np.testing.assert_allclose(data["current_x"][()], data["lockin_current_x"][()] * 1e-5)
            np.testing.assert_allclose(data["voltage_x"][()], data["lockin_voltage_x"][()] / 100.0)
            self.assertEqual(data["lockin_current_x"].attrs["units"], "V")
            self.assertEqual(int(data.attrs["lockin_overload_points"]), 0)
        self.assertEqual(self.station.lockin.input_config(), "a")
        self.errors.assert_not_called()

    def test_saturating_preamp_counts_overloaded_points(self):
        # 1e-7 A/V turns ~0.5 uA into ~5 V, beyond the SR830's 1 V range: the reading clips.
        window = self.window(preamp=1e-7)
        window._set_combo(window.current_mode_combo, "preamp")
        window.refresh_preview()
        self.run_to_end(window)
        self.assertEqual(window.run_outcome, "complete", window.run_message)
        self.assertGreater(window._overloads, 0)
        self.assertIn("overloaded", window.lbl_status.text())
        with h5py.File(self.files()[0]) as data:
            self.assertEqual(int(data.attrs["lockin_overload_points"]), window._overloads)
            self.assertTrue(np.any(data["status_current"][()] & 0b0111))

    def test_input_mode_mismatch_blocks_before_any_electrode_moves(self):
        from measurements.transport_gui import TransportWindow
        window = self.window()
        self.station.lockin.input_config("a")  # the current lock-in reads volts
        with patch.object(TransportWindow, "bind_simulation"), \
             patch.object(UVoltDacChannel, "_write_step") as write:
            self.run_to_end(window)
            write.assert_not_called()
        self.assertEqual(window.run_outcome, "error")
        self.assertIn("voltage input 'a'", window.run_message)
        self.assertEqual(self.errors.call_count, 1)
        self.assertFalse(window.measured_mask.any())

    def test_reference_warnings_are_logged_and_saved(self):
        from measurements.transport_gui import TransportWindow
        original = TransportWindow.bind_simulation

        def unlocked(window_self, settings, electrodes, plan=None):
            original(window_self, settings, electrodes, plan)
            self.station.lockin2.reference_source("internal")
        window = self.window()
        with patch.object(TransportWindow, "bind_simulation", unlocked):
            self.run_to_end(window)
        self.assertEqual(window.run_outcome, "complete", window.run_message)
        self.assertIn("not phase-locked", window.log_view.toPlainText())
        with h5py.File(self.files()[0]) as data:
            self.assertIn("not phase-locked", json.loads(data.attrs["lockin_warnings"])[0])

    def test_running_transport_locks_other_tabs_and_stop_keeps_partial_data(self):
        from lab_gui import TRANSPORT_TAB
        window = self.window(x_points=10, y_points=10, point_settle=0.03)
        window.begin_sweep()
        self.wait_for(lambda: window.measured_mask is not None and window.measured_mask.sum() >= 3)
        self.dashboard.check_measurement_status()
        self.assertTrue(self.dashboard.measurement_active())
        self.assertFalse(self.dashboard.prepare_device_change())
        for index in range(self.dashboard.tab_widget.count()):
            self.assertEqual(self.dashboard.tab_widget.isTabEnabled(index), index == TRANSPORT_TAB)
        window.stop_sweep()
        self.wait_for(lambda: not window.busy())
        self.app.processEvents()
        self.assertEqual(window.run_outcome, "stopped")
        measured = int(window.measured_mask.sum())
        self.assertTrue(0 < measured < 100)
        files = self.files()
        self.assertIn("_interrupted", files[0].name)
        with h5py.File(files[0]) as data:
            self.assertEqual(int(data["measured_mask"][()].sum()), measured)
        # Stop holds the voltages (the first line runs at V_top = -0.8 V).
        self.assertNotEqual(window.make_electrodes(window.collect_settings()).setpoints()["top"], 0.0)
        self.dashboard.check_measurement_status()
        self.assertTrue(all(self.dashboard.tab_widget.isTabEnabled(i)
                            for i in range(self.dashboard.tab_widget.count())))
        self.errors.assert_not_called()

    def test_out_of_limit_top_gate_is_blocked_before_any_output_write(self):
        window = self.window(max_ts=0.5)
        self.assertFalse(window._preview_ok)
        self.assertIn("dielectric limit", window.preview_label.text())
        with patch.object(UVoltDacChannel, "_write_step") as write:
            window.begin_sweep()
            self.app.processEvents()
            write.assert_not_called()
        self.assertFalse(window.is_running)
        self.assertEqual(self.errors.call_count, 1)

    def test_autorange_records_each_points_sensitivity(self):
        window = self.window()
        window.autorange_current_check.setChecked(True)
        window.autorange_voltage_check.setChecked(True)
        self.run_to_end(window)
        self.assertEqual(window.run_outcome, "complete", window.run_message)
        self.assertTrue(np.all(np.isfinite(window.data["sensitivity_current"])))
        self.assertTrue(np.all(np.isfinite(window.data["sensitivity_voltage"])))
        self.assertTrue(np.all(window.data["sensitivity_current"] <= 1e-6))

    def test_settings_persist_and_reload_without_instrument_commands(self):
        from lab_gui import TRANSPORT_TAB
        window = self.window()
        window._set_combo(window.current_lockin_combo, "lockin2")
        window._set_combo(window.current_mode_combo, "preamp")
        window.preamp_spin.setValue(2e-8)
        window.voltage_gain_spin.setValue(100.0)
        window._set_combo(window.tc_combo, 1.0)
        self.dashboard.tab_widget.setCurrentIndex(0)
        stored = json.loads((self.directory / "value_storage").read_text())["transport"]
        self.assertEqual((stored["current_lockin"], stored["current_mode"]), ("lockin2", "preamp"))
        self.assertAlmostEqual(stored["preamp_sensitivity"], 2e-8)
        with patch.object(UVoltDacChannel, "_write_step") as write:
            self.close_dashboard()
            self.station = dashboard_checks.Station()
            self.dashboard = dashboard_checks.LabMainWindow(self.station)
            self.dashboard.tab_widget.setCurrentIndex(TRANSPORT_TAB)
            write.assert_not_called()
        window = self.dashboard.transport_window
        self.assertEqual(window.current_lockin_combo.currentData(), "lockin2")
        self.assertEqual(window.voltage_lockin_label.text(), "Lock-in 1")
        self.assertAlmostEqual(window.preamp_spin.value(), 2e-8)
        self.assertTrue(window.preamp_spin.isEnabled())
        self.assertEqual(window.tc_combo.currentData(), 1.0)

    def test_live_readout_converts_and_flags_wrong_input(self):
        window = self.window()
        worker = window.live_status_worker
        worker._last_config = -1e9
        window.on_live_status(worker.poll())
        text = window.live_lockin_label.text()
        self.assertIn("R4", text)
        self.assertNotIn("⚠", text)
        self.station.lockin.input_config("a")
        worker._last_config = -1e9
        window.on_live_status(worker.poll())
        self.assertIn("voltage input 'a'", window.live_lockin_label.text())

    def test_log_scale_shows_decades_and_keeps_cuts_linear(self):
        window = self.window()
        self.run_to_end(window)
        window.log_scale_check.setChecked(True)
        np.testing.assert_allclose(window.display_arrays()["r4"], np.log10(np.abs(window.data["r4"])))
        np.testing.assert_allclose(window.cut_arrays()["r4"], window.data["r4"])
        self.assertIn("log", window.r4_map_plot.plotItem.titleLabel.text)

    def test_transport_axes_keep_volts_and_adaptive_ticks(self):
        import pyqtgraph as pg
        window = self.window()
        plots = [value for value in vars(window).values() if isinstance(value, pg.PlotWidget)]
        self.assertGreaterEqual(len(plots), 11)
        for plot in plots:
            for side in ("bottom", "left"):
                axis = plot.getAxis(side)
                self.assertIsNone(axis._tickLevels)
                if axis.labelUnits == "V":
                    axis.setRange(9.0001, 9.0009)
                    self.assertEqual(axis.autoSIPrefixScale, 1)


if __name__ == "__main__":
    unittest.main()
