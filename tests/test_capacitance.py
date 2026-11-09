"""Capacitance bridge: original-formula parity, drivers, protection and dashboard runs.

Everything uses simulated instruments or fake I/O; no hardware is opened.
"""
import json
import math
import unittest
import warnings
from unittest.mock import patch

import h5py
import numpy as np

import capacitance_bridge as cb
from capacitance_electrodes import ElectrodeSet, MeasurementStopped, RunControl, validate_wiring
from drivers.qpg_uvoltdaq import QPGuVoltDAQ, UVoltDacChannel
from drivers.siglent_sdg import SiglentSDG2042X, parse_fields, parse_number
from drivers.stanford_sr830 import StanfordSR830
from gate_safety import GateRampCancelled


# --- Verbatim copies of the qcodes-qpg capacitance_measurement.py formulas ----
def original_vb_fixed(p0, n0, delta, vb):
    return vb - (n0 * delta - p0) / (1.0 - delta ** 2), vb - 0.5 * (n0 - p0) / (1.0 - delta)


def original_vt_fixed(p0, n0, delta, vt):
    return (n0 * delta - p0) / (1.0 - delta ** 2) + vt, vt - 0.5 * (n0 + p0) / (1.0 + delta)


def original_vs_fixed(p0, n0, delta, vs):
    return vs + 0.5 * (n0 + p0) / (1.0 + delta), vs + 0.5 * (n0 - p0) / (1.0 - delta)


def original_vec_phase(s):
    phase = -1.0 * np.degrees(np.arctan2(s[1, 0], s[0, 0]))
    if phase < 0:
        phase = 360 + phase
    return phase


def original_off_balance(M, vb, offset, ac_scale):
    M = np.linalg.det(M)
    vbx, vby = np.array(vb).flatten()
    vdx, vdy = np.array(offset).flatten()
    Dtg = (vby * vdx - vbx * vdy) / (vbx ** 2 + vby ** 2) / M * ac_scale
    Ctg = -1.0 * (vbx * vdx + vby * vdy) / (vbx ** 2 + vby ** 2) / M * ac_scale
    return Ctg, Dtg


class CoordinateTests(unittest.TestCase):
    def test_transforms_match_the_original_script_for_every_fixed_electrode(self):
        rng = np.random.default_rng(3)
        n0, p0 = rng.uniform(-4, 4, 64), rng.uniform(-4, 4, 64)
        for delta in (0.0, 0.214285, -0.45):
            for v_fixed in (-0.415, 0.0, 1.7):
                with self.subTest(delta=delta, v_fixed=v_fixed):
                    vt, vb, vs = cb.electrode_voltages(n0, p0, delta=delta, v_fixed=v_fixed, fixed="bottom")
                    np.testing.assert_allclose((vt, vs), original_vb_fixed(p0, n0, delta, v_fixed))
                    np.testing.assert_allclose(vb, v_fixed)
                    vt, vb, vs = cb.electrode_voltages(n0, p0, delta=delta, v_fixed=v_fixed, fixed="top")
                    np.testing.assert_allclose((vb, vs), original_vt_fixed(p0, n0, delta, v_fixed))
                    np.testing.assert_allclose(vt, v_fixed)
                    vt, vb, vs = cb.electrode_voltages(n0, p0, delta=delta, v_fixed=v_fixed, fixed="sample")
                    np.testing.assert_allclose((vt, vb), original_vs_fixed(p0, n0, delta, v_fixed))
                    back = cb.density_field(vt, vb, vs, delta=delta)
                    np.testing.assert_allclose(back, (n0, p0), atol=1e-12)

    def test_direct_mode_sweeps_the_two_non_fixed_electrodes_in_script_order(self):
        self.assertEqual(cb.swept_electrodes("bottom"), ("top", "sample"))
        self.assertEqual(cb.swept_electrodes("top"), ("bottom", "sample"))
        self.assertEqual(cb.swept_electrodes("sample"), ("top", "bottom"))
        vt, vb, vs = cb.direct_voltages(0.3, -0.2, v_fixed=0.1, fixed="bottom")
        self.assertEqual((float(vt), float(vb), float(vs)), (0.3, 0.1, -0.2))
        self.assertEqual(cb.coordinate_names("direct", "sample"), ("V_top", "V_bottom"))

    def test_invalid_asymmetry_and_electrode_are_rejected(self):
        for delta in (1, -1, 1.5, math.nan):
            with self.assertRaises(ValueError):
                cb.electrode_voltages(0, 0, delta=delta, v_fixed=0, fixed="sample")
        with self.assertRaises(ValueError):
            cb.electrode_voltages(0, 0, delta=0, v_fixed=0, fixed="gate")

    def test_plan_order_lines_and_serpentine(self):
        common = dict(mode="density", fixed="sample", delta=0, v_fixed=0, x_start=-1, x_end=1,
                      x_points=3, y_start=0, y_end=1, y_points=2)
        plan = cb.build_sweep_plan(**common)
        self.assertEqual(plan.order.tolist(), [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        plan = cb.build_sweep_plan(**common, serpentine=True)
        self.assertEqual(plan.order.tolist(), [[0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [1, 0]])
        plan = cb.build_sweep_plan(**common, fast_axis="y")
        self.assertEqual([len(points) for _, points in plan.lines()], [2, 2, 2])
        self.assertEqual(plan.shape, (2, 3))
        np.testing.assert_allclose(plan.n0, np.tile([-1, 0, 1], (2, 1)))

    def test_limits_reject_targets_and_dielectric_voltages_without_clipping(self):
        limits = cb.ElectrodeLimits(max_abs=2, max_top_sample=1.5, max_bottom_sample=3)
        limits.check([1.9], [-1.9], [0.5])
        with self.assertRaisesRegex(ValueError, "Top gate would reach 2.1"):
            limits.check([2.1], [0], [0])
        with self.assertRaisesRegex(ValueError, "Nothing was moved"):
            limits.check([1.0], [0], [-0.8])
        with self.assertRaises(ValueError):
            limits.check([math.nan], [0], [0])
        with self.assertRaisesRegex(ValueError, "10 V limit"):
            cb.ElectrodeLimits(max_abs=20).check([12], [0], [0], output_caps={"top": 10})

    def test_coordinated_path_steps_are_bounded_and_end_exactly(self):
        path = cb.interpolate_path({"top": -5, "bottom": -5, "sample": 0},
                                   {"top": 5, "bottom": 5, "sample": 0}, 0.05)
        self.assertEqual(len(path), 200)
        self.assertEqual(path[-1], {"top": 5, "bottom": 5, "sample": 0})
        steps = np.diff([[-5, -5, 0]] + [[p["top"], p["bottom"], p["sample"]] for p in path], axis=0)
        self.assertLessEqual(np.abs(steps).max(), 0.05 + 1e-12)
        self.assertTrue(all(math.isclose(p["top"], p["bottom"]) for p in path))


class FakeLinearBridge:
    """m = M s + c with M a rotation and scale, like the ideal bridge."""

    def __init__(self, gain=2e-3, phase=0.6, offset=(3e-5, -1e-5)):
        self.M = gain * np.array([[math.cos(phase), -math.sin(phase)], [math.sin(phase), math.cos(phase)]])
        self.c = np.array(offset)
        self.s = np.zeros(2)

    def excite(self, s):
        self.s = np.array(s, dtype=float)

    def measure(self):
        return tuple(self.M @ self.s + self.c)


class BridgeAlgorithmTests(unittest.TestCase):
    def test_linear_bridge_balances_in_one_iteration_at_the_exact_point(self):
        fake = FakeLinearBridge()
        result = cb.LinearBridge(fake.excite, fake.measure, ac_scale=50.0).balance()
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        np.testing.assert_allclose(result.balance_vector, -np.linalg.inv(fake.M) @ fake.c, rtol=1e-9)
        np.testing.assert_allclose(result.matrix, fake.M, rtol=1e-9)

    def test_results_use_the_original_capacitance_and_offbalance_formulas(self):
        fake = FakeLinearBridge(offset=(4e-5, 2e-5))
        result = cb.LinearBridge(fake.excite, fake.measure, ac_scale=12.5).balance()
        vb = result.balance_vector
        self.assertAlmostEqual(result.capacitance, -vb[0] * 12.5)
        self.assertAlmostEqual(result.dissipation, vb[1] * 12.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)  # the original used np.matrix
            expected = original_off_balance(np.matrix(result.matrix), np.array(vb).reshape(2, 1),
                                            np.array(result.offset).reshape(2, 1), 12.5)
        np.testing.assert_allclose(result.offbalance_coefficients, expected, rtol=1e-12)
        c_coeff, d_coeff = expected
        c, d = result.offbalance(2e-6, -1e-6)
        self.assertAlmostEqual(c, c_coeff * 2e-6 + d_coeff * -1e-6 + result.capacitance)
        self.assertAlmostEqual(d, d_coeff * 2e-6 - c_coeff * -1e-6 + result.dissipation)

    def test_vector_phase_and_scale_match_the_original(self):
        for vector in ((0.1, 0.1), (-0.1, -0.1), (0.0, 0.2), (-0.3, 0.0), (0.2, -0.05)):
            self.assertAlmostEqual(cb.vector_phase_deg(vector),
                                   original_vec_phase(np.array(vector).reshape(2, 1)) % 360)
        self.assertAlmostEqual(cb.ac_scale_factor(20, 40, 0.1), 10 ** (-(20 - 40) / 20.0) / 0.1)
        self.assertEqual(cb.lockin_wait_time(0.3, 6), 1.5)
        self.assertAlmostEqual(cb.lockin_wait_time(0.3, 12), 2.1)
        self.assertAlmostEqual(cb.lockin_wait_time(0.3, 24), 3.0)

    def test_simulated_bridge_recovers_capacitance_and_offbalance_values(self):
        state = {"ch1_amplitude": 0.1, "ch1_phase": 0.0, "ch2_amplitude": 0.3, "ch2_phase": 0.0}
        volts = {"v": (0.4, -0.3, 0.0)}
        signal = cb.SimulatedBridgeSignal(lambda: state, lambda: volts["v"], lambda: (40.0, 20.0))

        def excite(s):
            state["ch1_phase"] = cb.vector_phase_deg(s)
            state["ch2_amplitude"] = float(np.linalg.norm(s))

        def measure():
            value = signal()
            return value.real, value.imag
        result = cb.LinearBridge(excite, measure, ac_scale=cb.ac_scale_factor(20, 40, 0.1)).balance()
        truth = signal.sample.complex_capacitance(*volts["v"])
        self.assertAlmostEqual(result.capacitance, truth.real, places=9)
        self.assertAlmostEqual(result.dissipation, -truth.imag, places=9)
        for vt in np.linspace(-0.6, 0.6, 13):
            volts["v"] = (vt, -0.3, 0.0)
            c, d = result.offbalance(*measure())
            truth = signal.sample.complex_capacitance(*volts["v"])
            self.assertLess(abs(c - truth.real), 1e-3)
            self.assertLess(abs(d + truth.imag), 1e-3)

    def test_out_of_range_minimum_amplitude_cancel_and_nonconvergence(self):
        weak = FakeLinearBridge(gain=1e-9)
        with self.assertRaisesRegex(cb.BalanceError, "out of range"):
            cb.LinearBridge(weak.excite, weak.measure).balance()
        tiny = FakeLinearBridge(offset=(1e-9, 0))
        with self.assertRaisesRegex(cb.BalanceError, "below the AWG minimum"):
            cb.LinearBridge(tiny.excite, tiny.measure).balance()
        fake = FakeLinearBridge()
        with self.assertRaises(cb.BalanceCancelled):
            cb.LinearBridge(fake.excite, fake.measure, keep_running=lambda: False).balance()
        sheared = FakeLinearBridge()
        sheared.M = np.array([[2e-3, 1.2e-3], [0.0, 2e-3]])
        result = cb.LinearBridge(sheared.excite, sheared.measure, tolerance=1e-15, iterations=2).balance()
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 2)
        self.assertIn("maximum number of iterations", result.message)
        with self.assertRaisesRegex(ValueError, "below the AWG minimum"):
            cb.LinearBridge(fake.excite, fake.measure, s1=(1e-4, 0))

    def test_rebalance_reuses_the_response_matrix(self):
        fake = FakeLinearBridge()
        first = cb.LinearBridge(fake.excite, fake.measure).balance()
        fake.c = fake.c + np.array([5e-6, 5e-6])
        calls = []
        bridge = cb.LinearBridge(fake.excite, lambda: calls.append(1) or fake.measure())
        second = bridge.rebalance(first)
        np.testing.assert_allclose(second.matrix, first.matrix)
        self.assertTrue(second.converged)
        self.assertEqual(len(calls), 2)  # start point plus one iteration; no probe vectors


class DriverTests(unittest.TestCase):
    def test_siglent_replies_are_parsed_with_units(self):
        first, fields = parse_fields("C1:BSWV WVTP,SINE,FRQ,7777.7HZ,PERI,0.000128572S,AMP,0.1V,"
                                     "AMPVRMS,0.0353553Vrms,OFST,0V,HLEV,0.05V,LLEV,-0.05V,PHSE,0")
        self.assertIsNone(first)
        self.assertEqual(parse_number(fields["FRQ"]), 7777.7)
        self.assertEqual(parse_number(fields["AMP"]), 0.1)
        self.assertEqual(parse_number("-2.5e-3V"), -0.0025)
        self.assertEqual(parse_fields("C2:OUTP ON,LOAD,HZ,PLRT,NOR"), ("ON", {"LOAD": "HZ", "PLRT": "NOR"}))

    def test_siglent_scpi_commands_and_readback(self):
        awg = SiglentSDG2042X("awg_scpi_test", address="simulated")
        self.addCleanup(awg.close)
        awg.simulated = False
        self.addCleanup(setattr, awg, "simulated", True)
        replies = {"C1:BSWV?": "C1:BSWV WVTP,SINE,FRQ,1234.5HZ,AMP,0.25V,OFST,0V,PHSE,45.5",
                   "C1:OUTP?": "C1:OUTP ON,LOAD,HZ,PLRT,NOR"}
        writes = []
        with patch.object(awg, "ask", side_effect=lambda cmd: replies[cmd]), \
             patch.object(awg, "write", side_effect=writes.append):
            self.assertEqual(awg.ch1.frequency(), 1234.5)
            self.assertEqual(awg.ch1.phase(), 45.5)
            self.assertTrue(awg.ch1.enabled())
            self.assertEqual(awg.ch1.load(), "HZ")
            self.assertEqual(awg.ch1.state()["amplitude"], 0.25)
            awg.ch2.amplitude(0.0123)
            awg.ch1.phase(315.0)
            awg.ch2.enabled(False)
            with self.assertRaises(ValueError):
                awg.ch1.amplitude(30)
        self.assertEqual(writes, ["C2:BSWV AMP,0.0123", "C1:BSWV PHSE,315", "C2:OUTP OFF"])

    def test_simulated_lockin_values_autorange_and_overload_bits(self):
        lockin = StanfordSR830("lockin_sim_test", address="simulated")
        self.addCleanup(lockin.close)
        with self.assertRaises(ValueError):
            lockin.time_constant(0.25)
        lockin.sensitivity(1e-5)
        lockin.signal_source = lambda tc: complex(3e-4, -1e-4)
        x, y = lockin.snap_xy()
        self.assertAlmostEqual(x, 1.09e-5)
        self.assertEqual(lockin.status_byte() & 0b100, 0b100)
        self.assertEqual(lockin.status_byte(), 0)
        lockin.autorange(8)
        self.assertGreater(lockin.sensitivity(), abs(complex(3e-4, -1e-4)) / 0.9)
        self.assertEqual(lockin.snap_xy(), (3e-4, -1e-4))

    def test_dac_ramps_in_bounded_steps_verifies_and_cancels(self):
        dac = QPGuVoltDAQ("dac_scpi_test", address="simulated")
        self.addCleanup(dac.close)
        dac.simulated = False
        self.addCleanup(setattr, dac, "simulated", True)
        state = {"v": 0.0, "stale": False}
        writes = []

        def ask(cmd):
            if cmd == "GET_DAC,0":
                return "0.0" if state["stale"] else f"{state['v']:.9e}"
            if cmd.startswith("SET,0,"):
                state["v"] = float(cmd.split(",")[2])
                writes.append(state["v"])
                return "OK"
            raise AssertionError(cmd)
        dac.dac0.ramp_rate(1000.0)
        with patch.object(dac, "ask", side_effect=ask):
            dac.dac0.voltage(0.2)
            np.testing.assert_allclose(writes, [0.05, 0.1, 0.15, 0.2])
            with self.assertRaises(ValueError):
                dac.dac0.voltage(10.5)
            self.assertEqual(len(writes), 4)
            # Cancel once two steps of the return ramp were written: 0.2 -> 0.15 -> 0.1.
            with self.assertRaises(GateRampCancelled):
                dac.dac0.ramp_to(0.0, keep_running=lambda: len(writes) < 6)
            self.assertAlmostEqual(state["v"], 0.1)
            self.assertEqual(len(writes), 6)
            state["stale"] = True
            with self.assertRaisesRegex(RuntimeError, "not confirmed"):
                dac.dac0.voltage(0.04)


class ElectrodeTests(unittest.TestCase):
    def setUp(self):
        from qcodes.station import Station
        from station_devices import initialize_device
        self.station = Station()
        for name in ("smu", "dac"):
            initialize_device(self.station, name, simulated=True)
        self.addCleanup(lambda: [c.close() for c in list(self.station.components.values())])

    def test_wiring_requires_driven_swept_electrodes_and_unique_outputs(self):
        validate_wiring({"top": "dac:0", "bottom": "dac:1", "sample": "none"}, "sample")
        with self.assertRaisesRegex(ValueError, "must be driven"):
            validate_wiring({"top": "dac:0", "bottom": "none", "sample": "dac:1"}, "sample")
        with self.assertRaisesRegex(ValueError, "more than one electrode"):
            validate_wiring({"top": "dac:0", "bottom": "dac:0", "sample": "dac:1"}, "sample")

    def test_coordinated_move_keeps_both_gates_together(self):
        smu = self.station.smu
        for channel in (smu.ch1, smu.ch2):
            channel.output(True)
        electrodes = ElectrodeSet(self.station, {"top": "smu:1", "bottom": "smu:2", "sample": "dac:1"},
                                  fixed="sample", fixed_voltage=0.0,
                                  limits=cb.ElectrodeLimits(10, 10, 10), dac_ramp_rate=10)
        control = RunControl()
        electrodes.preflight()
        electrodes.move_to({"top": -5, "bottom": -5, "sample": 0}, control)
        history = []
        for role, output in electrodes.outputs.items():
            original = output.ramp_to

            def recorded(value, control, original=original):
                original(value, control)
                history.append(electrodes.setpoints())
            output.ramp_to = recorded
        electrodes.move_to({"top": 5, "bottom": 5, "sample": 0}, control)
        self.assertEqual(electrodes.setpoints(), {"top": 5.0, "bottom": 5.0, "sample": 0.0})
        # Sequential channel ramps would have put 10 V between the gates.
        self.assertLessEqual(max(abs(p["top"] - p["bottom"]) for p in history), 0.05 + 1e-9)

    def test_smu_output_must_be_on_and_external_electrode_cannot_move(self):
        electrodes = ElectrodeSet(self.station, {"top": "smu:1", "bottom": "dac:0", "sample": "none"},
                                  fixed="sample", fixed_voltage=0.2,
                                  limits=cb.ElectrodeLimits(), dac_ramp_rate=1)
        with self.assertRaisesRegex(RuntimeError, "output is OFF"):
            electrodes.preflight()
        with self.assertRaisesRegex(RuntimeError, "not driven"):
            electrodes.outputs["sample"].ramp_to(0.0, RunControl())

    def test_stop_interrupts_a_move_and_holds_the_reached_voltage(self):
        electrodes = ElectrodeSet(self.station, {"top": "dac:0", "bottom": "dac:3", "sample": "dac:1"},
                                  fixed="bottom", fixed_voltage=0.0, limits=cb.ElectrodeLimits(),
                                  dac_ramp_rate=10)
        control = RunControl()
        moves = []

        def progress(values):
            moves.append(values)
            if len(moves) == 5:
                control.stop()
        with self.assertRaises(MeasurementStopped):
            electrodes.move_to({"top": 1.0, "bottom": 0.0, "sample": -1.0}, control, progress)
        self.assertAlmostEqual(electrodes.setpoints()["top"], 0.25)


try:
    import test_dashboard_workflow as dashboard_checks
except Exception:  # pragma: no cover - GUI dependencies missing
    dashboard_checks = None


@unittest.skipIf(dashboard_checks is None, "Qt dashboard dependencies are unavailable")
class CapacitanceDashboardTests(unittest.TestCase):
    if dashboard_checks is not None:
        setUpClass = classmethod(dashboard_checks.DashboardWorkflowTests.setUpClass.__func__)
        setUp = dashboard_checks.DashboardWorkflowTests.setUp
        close_dashboard = dashboard_checks.DashboardWorkflowTests.close_dashboard
        wait_for = dashboard_checks.DashboardWorkflowTests.wait_for

    def window(self, **values):
        from lab_gui import CAPACITANCE_TAB
        self.dashboard.tab_widget.setCurrentIndex(CAPACITANCE_TAB)
        window = self.dashboard.capacitance_window
        defaults = dict(x_points=11, y_points=2, point_settle=0.0, line_settle=0.0)
        defaults.update(values)
        widgets = {"x_points": window.x_points_spin, "y_points": window.y_points_spin,
                   "point_settle": window.point_settle_spin, "line_settle": window.line_settle_spin,
                   "max_abs": window.max_abs_spin, "x_start": window.x_start_spin, "x_end": window.x_end_spin}
        for key, value in defaults.items():
            widgets[key].setValue(value)
        window.refresh_preview()
        return window

    def run_to_end(self, window, timeout=30):
        window.begin_sweep()
        self.wait_for(lambda: not window.busy(), timeout)
        self.app.processEvents()

    def files(self):
        return sorted(self.directory.rglob("*.h5"))

    def test_simulated_map_saves_calibrated_data_and_ramps_to_zero(self):
        window = self.window()
        self.assertTrue(window._preview_ok, window.preview_label.text())
        self.run_to_end(window)
        self.assertEqual(window.run_outcome, "complete", window.run_message)
        self.assertTrue(window.measured_mask.all())
        files = self.files()
        self.assertEqual(len(files), 1)
        self.assertIn("capacitance_C_P_n0_m1-1V_p0_m1-1V_11x2", files[0].name)
        with h5py.File(files[0]) as data:
            self.assertEqual(data.attrs["status"], "complete")
            self.assertEqual(data.attrs["fixed_electrode"], "bottom")
            self.assertEqual(json.loads(data.attrs["device_modes"])["lockin"], "simulated")
            self.assertEqual(data["capacitance_ratio"].shape, (2, 11))
            self.assertTrue(bool(data["balance/converged"][0]))
            self.assertTrue(np.all(data["measured_mask"][()]))
            vt, vb, vs = (data[f"target_v_{name}"][()] for name in ("top", "bottom", "sample"))
            measured = data["capacitance_ratio"][()]
        truth = np.vectorize(lambda a, b, c: cb.SimulatedSample("penetration").complex_capacitance(a, b, c).real)
        np.testing.assert_allclose(measured, truth(vt, vb, vs), atol=2e-3)
        self.assertEqual(window.make_electrodes(window.collect_settings()).setpoints(),
                         {"top": 0.0, "bottom": 0.0, "sample": 0.0})
        window.open_saved(str(files[0]))
        np.testing.assert_allclose(window.data["capacitance"], measured)
        self.errors.assert_not_called()

    def test_running_measurement_locks_other_tabs_and_stop_keeps_partial_data(self):
        from lab_gui import CAPACITANCE_TAB
        window = self.window(x_points=21, y_points=3, point_settle=0.03)
        window.begin_sweep()
        self.wait_for(lambda: window.measured_mask is not None and window.measured_mask.sum() >= 3)
        self.dashboard.check_measurement_status()
        self.assertTrue(self.dashboard.measurement_active())
        self.assertFalse(self.dashboard.prepare_device_change())
        for index in range(self.dashboard.tab_widget.count()):
            self.assertEqual(self.dashboard.tab_widget.isTabEnabled(index), index == CAPACITANCE_TAB)
        window.stop_sweep()
        self.wait_for(lambda: not window.busy())
        self.app.processEvents()
        self.assertEqual(window.run_outcome, "stopped")
        measured = int(window.measured_mask.sum())
        self.assertTrue(0 < measured < 63)
        files = self.files()
        self.assertEqual(len(files), 1)
        self.assertIn("_interrupted", files[0].name)
        with h5py.File(files[0]) as data:
            self.assertEqual(int(data["measured_mask"][()].sum()), measured)
            self.assertEqual(data.attrs["status"], "stopped")
        # Stop holds the voltages: no ramp to zero after an incomplete run.
        self.assertNotEqual(window.make_electrodes(window.collect_settings()).setpoints()["top"], 0.0)
        self.dashboard.check_measurement_status()
        self.assertTrue(all(self.dashboard.tab_widget.isTabEnabled(i)
                            for i in range(self.dashboard.tab_widget.count())))
        self.errors.assert_not_called()

    def test_manual_awg_balance_and_coordinated_moves(self):
        import measurements.capacitance_gui as capacitance_gui
        # A signal named "event" would shadow QObject.event(), which Qt calls on every event.
        self.assertNotIn("event", vars(capacitance_gui.CapacitanceTask))
        window = self.window()

        def finish():
            self.wait_for(lambda: not window.busy())
            self.app.processEvents()
        window.run_awg_action("apply")
        finish()
        self.assertTrue(self.station.awg.ch1.enabled() and self.station.awg.ch2.enabled())
        window.goto_x_spin.setValue(0.4)
        window.goto_y_spin.setValue(-0.2)
        window.goto_point()
        finish()
        targets = window.coordinate_targets(window.collect_settings(), 0.4, -0.2)
        setpoints = window.make_electrodes(window.collect_settings()).setpoints()
        for role, value in targets.items():
            self.assertAlmostEqual(setpoints[role], value)
        window.balance_now()
        finish()
        self.assertTrue(window.last_balance["converged"])
        self.assertEqual(window.balances, [])  # manual balances are not run calibrations
        window.ramp_to_zero()
        finish()
        self.assertEqual(window.make_electrodes(window.collect_settings()).setpoints(),
                         {"top": 0.0, "bottom": 0.0, "sample": 0.0})
        window.run_awg_action("off")
        finish()
        self.assertFalse(self.station.awg.ch1.enabled() or self.station.awg.ch2.enabled())
        self.errors.assert_not_called()

    def test_out_of_limit_sweep_is_blocked_before_any_output_write(self):
        window = self.window(max_abs=0.5)
        self.assertFalse(window._preview_ok)
        self.assertFalse(window.btn_begin.isEnabled())
        self.assertIn("outside", window.preview_label.text())
        with patch.object(UVoltDacChannel, "_write_step") as write, \
             patch.object(self.station.awg.ch1, "_set_bswv") as awg_write:
            window.begin_sweep()
            self.app.processEvents()
            write.assert_not_called()
            awg_write.assert_not_called()
        self.assertFalse(window.is_running)
        self.assertEqual(self.errors.call_count, 1)

    def test_settings_persist_and_reload_without_instrument_commands(self):
        from lab_gui import CAPACITANCE_TAB
        window = self.window()
        window.v_fixed_spin.setValue(-0.415)
        window.delta_spin.setValue(0.214285)
        window.wiring_combos["sample"].setCurrentIndex(window.wiring_combos["sample"].findData("dac:2"))
        window._set_combo(window.rebalance_combo, "line")
        window.c_ref_spin.setValue(2.5)
        self.dashboard.tab_widget.setCurrentIndex(0)
        stored = json.loads((self.directory / "value_storage").read_text())["capacitance"]
        self.assertEqual(stored["v_fixed"], -0.415)
        self.assertEqual(stored["wiring"]["sample"], "dac:2")
        self.assertAlmostEqual(stored["c_ref"], 2.5e-12)
        with patch.object(UVoltDacChannel, "_write_step") as write:
            self.close_dashboard()
            self.station = dashboard_checks.Station()
            self.dashboard = dashboard_checks.LabMainWindow(self.station)
            self.dashboard.tab_widget.setCurrentIndex(CAPACITANCE_TAB)
            write.assert_not_called()
        window = self.dashboard.capacitance_window
        self.assertEqual(window.v_fixed_spin.value(), -0.415)
        self.assertEqual(window.delta_spin.value(), 0.214285)
        self.assertEqual(window.wiring_combos["sample"].currentData(), "dac:2")
        self.assertEqual(window.rebalance_combo.currentData(), "line")
        self.assertAlmostEqual(window.c_ref_spin.value(), 2.5)

    def test_rebalance_every_line_records_each_calibration(self):
        window = self.window(y_points=3)
        window._set_combo(window.rebalance_combo, "line")
        self.run_to_end(window)
        self.assertEqual(window.run_outcome, "complete", window.run_message)
        self.assertEqual(len(window.balances), 3)
        np.testing.assert_array_equal(window.data["balance_index"], np.repeat([[0], [1], [2]], 11, axis=1))
        with h5py.File(self.files()[0]) as data:
            self.assertEqual(data["balance/matrix"].shape, (3, 2, 2))

    def test_replaced_lockin_is_used_by_the_next_measurement(self):
        from station_devices import configure_device, remove_device
        window = self.window()
        old = self.station.lockin
        self.assertTrue(self.dashboard.prepare_device_change())
        remove_device(self.station, "lockin")
        configure_device(self.station, "lockin", "Simulation")
        self.dashboard.finish_device_change()
        self.assertIsNot(self.station.lockin, old)
        self.run_to_end(window)
        self.assertEqual(window.run_outcome, "complete", window.run_message)
        self.assertIsNotNone(self.station.lockin.signal_source)

    def test_capacitance_axes_keep_volts_and_adaptive_ticks(self):
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
