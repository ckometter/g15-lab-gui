"""Whole-grid gate safety and ramp/settle/acquire ordering on fake devices."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import types
import unittest
from unittest.mock import Mock, patch
import numpy as np

from gate_safety import GateRampCancelled, GATE_MAX_ABS_VOLTAGE_V
from measurements import B_V_sweep_gui as bv_module
from measurements import photoluminescence_map_gui as pl_module
from measurements import scalar_map_gui as scalar_module


class FakeChannel:
    def __init__(self, number, events):
        self.channel_num = number
        self.events = events
        self.ramp_stop_requested = False
        self.mode = 'VOLT'
        self.value = 0.0
        self.source_mode = lambda: self.mode
        self.volt = lambda: self.value
        self.measure_volt = lambda: self.value
        self.measure_curr = lambda: 0.0
        self.set_r = types.SimpleNamespace(volt=Mock(side_effect=self.ramp))

    def ramp(self, target):
        self.events.append(('ramp', self.channel_num, target))
        self.value = target

    def output(self, *args):
        if args:
            self.events.append(('output', self.channel_num, args[0]))
        return True


class GateWorkerSafetyTests(unittest.TestCase):
    def setUp(self):
        guard = patch('pyvisa.ResourceManager', side_effect=AssertionError('Physical hardware forbidden'))
        guard.start()
        self.addCleanup(guard.stop)
        self.events = []
        self.errors = []
        smu = types.SimpleNamespace(
            ch1=FakeChannel(1, self.events), ch2=FakeChannel(2, self.events),
            simulated=True, return_to_local=lambda: None,
            enforce_current_compliance=lambda: self.events.append(('compliance',)),
        )
        magnet = types.SimpleNamespace(
            simulated=True, _sim_sweep_rate=60.0, SWITCH_HEATER_FITTED=False,
            field_sweep_rate=lambda *args: self.events.append(('magnet_rate', args)) if args else 60,
            output_field=lambda: 0.0, is_sweeping=lambda: False,
            field=lambda value: self.events.append(('magnet', value)),
            target_field_param=lambda value: self.events.append(('magnet_target', value)),
            goto_set=lambda: self.events.append(('magnet_go',)),
            hold=lambda: self.events.append(('magnet_hold',)),
        )
        self.station = types.SimpleNamespace(
            smu=smu, magnet=magnet,
            camera=types.SimpleNamespace(
                num_frames=lambda value: self.events.append(('camera_frames', value)),
                exposure_time=lambda value: self.events.append(('camera_exposure', value))),
            monochromator=types.SimpleNamespace(wavelength=lambda: 800),
            powermeter=types.SimpleNamespace(power=self.acquire_power),
            piezo300=types.SimpleNamespace(simulated=True,
                set_voltage=lambda axis, value: self.events.append(('piezo', axis, value))),
        )
        for module in (bv_module, pl_module):
            acquire = patch.object(module, 'acquire_sweep_spectrum', side_effect=self.acquire)
            acquire.start()
            self.addCleanup(acquire.stop)

    def acquire(self, *args, **kwargs):
        self.events.append(('acquire',))
        return np.zeros(4), np.zeros((1, 4))

    def acquire_power(self):
        self.events.append(('acquire',))
        return 1e-9

    def map_worker(self, module, **overrides):
        params = dict(x_start=0, x_end=.1, x_pts=2, y_start=0, y_end=0, y_pts=1,
                      x_param='V1', y_param='V2', settling_time=0,
                      num_pixels=4, wl_axis=np.arange(4), num_spectra=1)
        params.update(overrides)
        cls = module.PLMapWorker if module is pl_module else module.ScalarMapWorker
        worker = cls(self.station, params)
        worker.error.connect(self.errors.append)
        return worker

    def bv_worker(self, **overrides):
        params = dict(control_coeff_ch1=1, control_coeff_ch2=0, v_start=0, v_end=.1, v_pts=2,
                      b_start=0, b_end=0, b_pts=1,
                      variable_order='Hold B, scan Control', serpentine=False,
                      settling_time=0, exposure_time=.01, num_pixels=4,
                      faraday_comp=False, faraday_rot=0, end_ramp_zero=False,
                      end_ramp_ground=False, num_spectra=1, wl_axis=np.arange(4))
        params.update(overrides)
        worker = bv_module.BVSweepWorker(self.station, params)
        worker.error_occurred.connect(self.errors.append)
        return worker

    def test_later_invalid_map_target_blocks_all_movement_and_camera_configuration(self):
        for module in (pl_module, scalar_module):
            for params in (dict(x_end=GATE_MAX_ABS_VOLTAGE_V + 1, y_param='position Y'),
                           dict(x_param='Vc1', y_param='Vc2', x_end=GATE_MAX_ABS_VOLTAGE_V, y_end=GATE_MAX_ABS_VOLTAGE_V,
                                y_pts=2, coeff_a=1, coeff_b=1),
                           dict(x_param='Vc1', x_end=1, coeff_c=GATE_MAX_ABS_VOLTAGE_V + 1)):
                with self.subTest(module=module.__name__, params=params):
                    self.events.clear()
                    self.errors.clear()
                    self.map_worker(module, **params).run()
                    self.assertEqual(self.events, [])
                    self.assertEqual(len(self.errors), 1)
                    self.assertIn(str(GATE_MAX_ABS_VOLTAGE_V), self.errors[0])

    def test_invalid_bv_grid_blocks_every_device_even_with_cleanup_options(self):
        for params in (dict(v_end=GATE_MAX_ABS_VOLTAGE_V + 1),
                       dict(control_coeff_ch1=0, control_coeff_ch2=0),
                       dict(v_end=GATE_MAX_ABS_VOLTAGE_V, control_coeff_ch1=.5),
                       dict(control_coeff_ch2=float('nan'))):
            with self.subTest(params=params):
                self.events.clear()
                self.errors.clear()
                self.bv_worker(**params, end_ramp_ground=True, end_ramp_zero=True).run()
                self.assertEqual(self.events, [])
                self.assertEqual(len(self.errors), 1)

    def test_forward_coordinate_mapping_is_preserved_in_map_workers(self):
        for make in (
            lambda: self.map_worker(pl_module, coeff_a=.5, x_param='Vc1', x_start=20, x_end=20, x_pts=1),
            lambda: self.map_worker(scalar_module, coeff_a=.5, x_param='Vc1', x_start=20, x_end=20, x_pts=1),
        ):
            self.events.clear()
            self.errors.clear()
            make().run()
            self.assertFalse(self.errors)
            self.assertIn(('ramp', 1, 10), self.events)
            self.assertIn(('acquire',), self.events)

    def test_bv_sweeps_the_defined_control_and_acquires_every_field_control_pair(self):
        for order in ('Hold B, scan Control', 'Hold Control, scan B'):
            for serpentine in (False, True):
                with self.subTest(order=order, serpentine=serpentine):
                    self.events.clear()
                    self.errors.clear()
                    worker = self.bv_worker(control_coeff_ch1=.5, control_coeff_ch2=-.5,
                                            v_start=-.1, v_end=.1, v_pts=3,
                                            b_start=-.1, b_end=.1, b_pts=2,
                                            variable_order=order, serpentine=serpentine)
                    points = []
                    worker.point_measured.connect(lambda *args: points.append(args))
                    worker.run()
                    self.assertFalse(self.errors)
                    self.assertEqual(len(points), 6)
                    self.assertEqual(sum(e[0] == 'acquire' for e in self.events), 6)
                    self.assertEqual({(p[0], p[1]) for p in points},
                                     {(i, j) for i in range(3) for j in range(2)})
                    for point in points:
                        control, ch1, ch2 = point[2], point[4], point[5]
                        self.assertAlmostEqual(.5 * ch1 - .5 * ch2, control)
                        self.assertAlmostEqual(ch1, control)
                        self.assertAlmostEqual(ch2, -control)

    def test_current_source_mode_preflight_blocks_all_changes(self):
        self.station.smu.ch1.mode = 'CURR'
        for make in (self.bv_worker, lambda: self.map_worker(pl_module), lambda: self.map_worker(scalar_module)):
            self.events.clear()
            self.errors.clear()
            make().run()
            self.assertEqual(self.events, [])
            self.assertTrue(self.errors)

    def test_failed_or_cancelled_ramps_never_acquire_or_retry(self):
        for exception in (RuntimeError('SMU failed'), GateRampCancelled('Stopped')):
            for make in (self.bv_worker, lambda: self.map_worker(pl_module), lambda: self.map_worker(scalar_module)):
                self.events.clear()
                self.errors.clear()
                ramp = self.station.smu.ch1.set_r.volt
                ramp.reset_mock()
                ramp.side_effect = exception
                make().run()
                self.assertNotIn(('acquire',), self.events)
                self.assertEqual(ramp.call_count, 1)
                self.assertTrue(self.errors)

    def test_stop_during_first_channel_never_clears_stop_for_second_channel(self):
        for make in (self.bv_worker, lambda: self.map_worker(pl_module), lambda: self.map_worker(scalar_module)):
            self.events.clear()
            self.errors.clear()
            worker = make()
            second = self.station.smu.ch2.set_r.volt
            second.reset_mock()
            self.station.smu.ch1.set_r.volt.side_effect = lambda target: worker.stop()
            worker.run()
            self.assertNotIn(('acquire',), self.events)
            self.assertEqual(second.call_count, 0)
            self.assertTrue(self.station.smu.ch2.ramp_stop_requested)
            self.assertFalse(self.errors)

    def test_zero_settling_still_checks_stop_before_acquisition(self):
        for make in (self.bv_worker, lambda: self.map_worker(pl_module), lambda: self.map_worker(scalar_module)):
            self.events.clear()
            self.errors.clear()
            worker = make()
            self.station.smu.ch1.set_r.volt.side_effect = self.station.smu.ch1.ramp
            self.station.smu.ch2.set_r.volt.side_effect = lambda target: worker.stop()
            worker.run()
            self.assertNotIn(('acquire',), self.events)
            self.assertFalse(self.errors)

    def test_bv_stop_holds_gates_even_with_end_ground_selected(self):
        # Stop is not a new permission to ramp/disconnect gates. Grounding is
        # reserved for a successfully completed sweep or a new explicit action.
        for stop_stage in ('ramp', 'settle', 'magnet'):
            with self.subTest(stop_stage=stop_stage):
                self.events.clear()
                self.errors.clear()
                worker = self.bv_worker(end_ramp_ground=True, settling_time=.02,
                                        b_start=1 if stop_stage == 'magnet' else 0)
                self.station.smu.ch1.set_r.volt.side_effect = self.station.smu.ch1.ramp
                self.station.smu.ch2.set_r.volt.side_effect = self.station.smu.ch2.ramp
                if stop_stage == 'ramp':
                    def cancel(target):
                        self.station.smu.ch1.ramp(target)
                        worker.stop()
                        raise GateRampCancelled('Stopped mid-ramp')
                    self.station.smu.ch1.set_r.volt.side_effect = cancel
                if stop_stage == 'magnet':
                    self.station.magnet.goto_set = worker.stop
                    self.station.magnet.is_sweeping = lambda: True
                with patch.object(bv_module.time, 'sleep', side_effect=lambda delay: worker.stop()):
                    worker.run()
                self.assertFalse(self.errors)
                self.assertNotIn(('acquire',), self.events)
                self.assertFalse(any(e[0] == 'output' and e[2] is False for e in self.events))
                expected_ramps = 0 if stop_stage == 'magnet' else (1 if stop_stage == 'ramp' else 2)
                self.assertEqual(sum(e[0] == 'ramp' for e in self.events), expected_ramps)

    def test_bv_successful_completion_still_ramps_to_ground_before_output_off(self):
        self.bv_worker(end_ramp_ground=True).run()
        self.assertFalse(self.errors)
        channel_events = [e for e in self.events if e[0] in ('ramp', 'output')]
        self.assertEqual(channel_events[-4:], [('ramp', 1, 0), ('ramp', 2, 0),
                                               ('output', 1, False), ('output', 2, False)])

    def test_each_point_ramps_both_channels_then_waits_then_acquires(self):
        for module, make in (
            (bv_module, lambda: self.bv_worker(settling_time=.03)),
            (pl_module, lambda: self.map_worker(pl_module, settling_time=.03)),
            (scalar_module, lambda: self.map_worker(scalar_module, settling_time=.03)),
        ):
            with self.subTest(module=module.__name__):
                self.events.clear()
                self.errors.clear()
                elapsed = [0.0]
                def sleep(duration):
                    elapsed[0] += duration
                    self.events.append(('settle', duration))
                with patch.object(module.time, 'time', side_effect=lambda: elapsed[0]), \
                     patch.object(module.time, 'sleep', side_effect=sleep):
                    make().run()
                self.assertFalse(self.errors)
                points = [i for i, event in enumerate(self.events) if event[0] == 'acquire']
                self.assertEqual(len(points), 2)
                last = 0
                for end in points:
                    section = self.events[last:end]
                    ramps = [i for i, event in enumerate(section) if event[0] == 'ramp']
                    waits = [i for i, event in enumerate(section) if event[0] == 'settle']
                    self.assertEqual(len(ramps), 2)
                    self.assertTrue(waits)
                    self.assertGreater(min(waits), max(ramps))
                    self.assertGreaterEqual(sum(section[i][1] for i in waits), .03)
                    last = end + 1


if __name__ == '__main__': unittest.main()
