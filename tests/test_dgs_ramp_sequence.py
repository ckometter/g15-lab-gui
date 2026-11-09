"""Physical gate limits and cancellation ordering with fake instruments only."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PyQt6 import QtCore

from gate_safety import GateRampCancelled, GATE_MAX_ABS_VOLTAGE_V
from measurements import Dual_gate_sweep_gui as dgs
from position_dgs import PositionSweepWorker


DIRECT = QtCore.Qt.ConnectionType.DirectConnection


class DgsRampSequenceTests(unittest.TestCase):
    def setUp(self):
        self.params = dict(a=1., b=0., c=0., d=1.,
                           c1_start=.2, c1_end=.2, c1_pts=1,
                           c2_start=.3, c2_end=.3, c2_pts=1,
                           sweep_mode='Hold C2, sweep C1', serpentine=False,
                           exposure_time=.01, settling_time=0., num_pixels=3, num_spectra=1)
        self.station = SimpleNamespace(smu=Mock(), camera=Mock(), monochromator=Mock(),
                                       piezo300=Mock(spec=['get_axis_status', 'get_voltage', 'set_voltage']))
        for axis in (1, 2):
            channel = getattr(self.station.smu, f'ch{axis}')
            channel.source_mode.return_value = 'VOLT'
            channel.volt.return_value = 0.
            channel.ramp_stop_requested = False
            channel.measure_curr.return_value = 0.
            channel.measure_volt.return_value = 0.
            channel.output.return_value = True
        offsets = {1: 9., 2: 32.}
        self.station.piezo300.get_axis_status.side_effect = lambda axis: {'offset': offsets[axis], 'mode': 'off'}
        self.station.piezo300.get_voltage.side_effect = offsets.__getitem__
        self.station.piezo300.set_voltage.side_effect = offsets.__setitem__
        self.scan = dict(x=np.array([9.]), y=np.array([32.]), move_rate=.5, return_rate=.2, settle=0)
        self.store = Mock()
        self.store.position_finished.return_value = True
        self.acquisition = self.enterContext(patch.object(dgs, 'acquire_sweep_spectrum',
                                            return_value=(np.ones(3), np.ones((1, 3)))))
        # Do not let persisted acquisition settings influence test behavior.
        self.enterContext(patch.object(dgs.os.path, 'exists', return_value=False))

    def worker(self, position=False):
        worker = (PositionSweepWorker(self.station, self.params.copy(), self.scan, self.store)
                  if position else dgs.SweepWorker(self.station, self.params.copy()))
        worker.points, worker.errors = [], []
        if position:
            worker.position_point.connect(lambda *point: worker.points.append(point), DIRECT)
        else:
            worker.point_measured.connect(lambda *point: worker.points.append(point), DIRECT)
        worker.error_occurred.connect(worker.errors.append, DIRECT)
        return worker

    def assert_no_measurement_writes(self):
        self.station.smu.ch1.set_r.volt.assert_not_called()
        self.station.smu.ch2.set_r.volt.assert_not_called()
        self.station.smu.enforce_current_compliance.assert_not_called()
        self.station.piezo300.set_voltage.assert_not_called()
        self.station.camera.exposure_time.assert_not_called()
        self.station.camera.num_frames.assert_not_called()
        self.acquisition.assert_not_called()

    def test_inverse_transform_corner_exceeding_physical_limit_rejected_before_writes(self):
        # The inverse map doubles CH1 beyond the instrument range.
        self.params.update(a=.5, c1_start=0., c1_end=GATE_MAX_ABS_VOLTAGE_V, c1_pts=11,
                           c2_start=-1., c2_end=1., c2_pts=3)
        with self.assertRaisesRegex(ValueError, str(GATE_MAX_ABS_VOLTAGE_V)):
            dgs.validate_dgs_parameters(self.params)
        for position in (False, True):
            with self.subTest(position=position):
                worker = self.worker(position)
                worker.run() if position else worker._do_run()
                message = worker.failure if position else ' '.join(worker.errors)
                self.assertIn(str(GATE_MAX_ABS_VOLTAGE_V), message)
                self.assert_no_measurement_writes()
                self.assertEqual(worker.points, [])

    def test_actual_current_source_mode_blocks_dgs_and_psdgs_before_writes(self):
        self.station.smu.ch2.source_mode.return_value = 'CURR'
        for position in (False, True):
            with self.subTest(position=position):
                worker = self.worker(position)
                worker.run() if position else worker._do_run()
                message = worker.failure if position else ' '.join(worker.errors)
                self.assertIn('Channel 2', message)
                self.assertIn('voltage-source', message)
                self.assert_no_measurement_writes()

    def test_existing_out_of_bounds_voltage_blocks_scanner_before_position_scan(self):
        self.station.smu.ch2.volt.return_value = GATE_MAX_ABS_VOLTAGE_V + 1
        self.scan['x'] = np.array([9.1])
        worker = self.worker(position=True)
        worker.run()
        self.assertIn(str(GATE_MAX_ABS_VOLTAGE_V), worker.failure)
        self.assert_no_measurement_writes()
        self.station.piezo300.set_voltage.assert_not_called()

    def test_both_ramps_complete_before_full_gui_settling_then_spectrum(self):
        self.params['settling_time'] = .12
        worker = self.worker()
        elapsed, events = [0.], []
        def ramp(axis, target):
            events.append(('ramp', axis, target, elapsed[0]))
            elapsed[0] += .7
        self.station.smu.ch1.set_r.volt.side_effect = lambda target: ramp(1, target)
        self.station.smu.ch2.set_r.volt.side_effect = lambda target: ramp(2, target)
        def wait(seconds):
            events.append(('settle', seconds, elapsed[0]))
            elapsed[0] += seconds
        def acquire(*args, **kwargs):
            events.append(('spectrum', elapsed[0]))
            return np.ones(3), np.ones((1, 3))
        self.acquisition.side_effect = acquire
        clock = SimpleNamespace(time=lambda: elapsed[0], sleep=wait)
        with patch.object(dgs, 'time', clock):
            worker._do_run()
        self.assertEqual(worker.errors, [])
        self.assertEqual([entry[0] for entry in events[:2]], ['ramp', 'ramp'])
        self.assertEqual([entry[1] for entry in events[:2]], [1, 2])
        self.assertTrue(all(entry[0] == 'settle' for entry in events[2:-1]))
        self.assertEqual(events[-1][0], 'spectrum')
        self.assertGreaterEqual(events[-1][1] - 1.4, .12)
        self.assertEqual(len(worker.points), 1)

    def test_ramp_failure_or_unrequested_cancellation_never_records_or_continues(self):
        for failure in (RuntimeError('Gate setpoint readback failed'), GateRampCancelled('Output switched off')):
            for position in (False, True):
                with self.subTest(failure=type(failure).__name__, position=position):
                    self.station.smu.ch1.set_r.volt.reset_mock()
                    self.station.smu.ch2.set_r.volt.reset_mock()
                    self.station.smu.ch1.set_r.volt.side_effect = failure
                    worker = self.worker(position)
                    worker.run() if position else worker._do_run()
                    message = worker.failure if position else ' '.join(worker.errors)
                    self.assertIn(str(failure), message)
                    self.assertEqual(worker.points, [])
                    self.station.smu.ch1.set_r.volt.assert_called_once()
                    self.station.smu.ch2.set_r.volt.assert_not_called()
                    self.acquisition.assert_not_called()
                    self.store.write_point.assert_not_called()

    def test_pause_resume_retries_both_gate_ramps_before_recording_dgs_and_psdgs(self):
        for position in (False, True):
            for immediate_resume in (False, True):
                with self.subTest(position=position, immediate_resume=immediate_resume):
                    self.acquisition.reset_mock()
                    self.store.reset_mock()
                    self.store.position_finished.return_value = True
                    worker = self.worker(position)
                    events, paused, attempts = [], threading.Event(), [0]
                    worker.status_updated.connect(
                        lambda text: paused.set() if text == 'Status: Paused' else None, DIRECT)
                    def ramp1(target):
                        attempts[0] += 1
                        events.append(('ramp1', attempts[0]))
                        if attempts[0] == 1:
                            worker.pause()
                            if immediate_resume:
                                worker.resume()
                            raise GateRampCancelled('Paused inside a gate ramp')
                    self.station.smu.ch1.set_r.volt.side_effect = ramp1
                    self.station.smu.ch2.set_r.volt.side_effect = lambda target: events.append(('ramp2', target))
                    self.acquisition.side_effect = lambda *args, **kwargs: (
                        events.append(('acquire',)) or (np.ones(3), np.ones((1, 3))))
                    thread = threading.Thread(target=worker.run if position else worker._do_run, daemon=True)
                    thread.start()
                    try:
                        if not immediate_resume:
                            self.assertTrue(paused.wait(2), 'Worker did not acknowledge pause')
                            self.acquisition.assert_not_called()
                            self.assertEqual(worker.points, [])
                            worker.resume()
                        thread.join(2)
                        self.assertFalse(thread.is_alive(), 'Worker failed to finish after resume')
                    finally:
                        worker.stop()
                        thread.join(2)
                    self.assertEqual(worker.errors, [])
                    if position:
                        self.assertEqual(worker.failure, '')
                        self.assertTrue(worker.completed)
                        self.store.write_point.assert_called_once()
                    self.assertEqual(events, [('ramp1', 1), ('ramp1', 2), ('ramp2', .3), ('acquire',)])
                    self.assertEqual(len(worker.points), 1)

    def test_stop_during_gate_ramp_exits_dgs_and_psdgs_without_spectrum(self):
        for position in (False, True):
            with self.subTest(position=position):
                worker = self.worker(position)
                def stopped(target):
                    worker.stop()
                    raise GateRampCancelled('Stop requested')
                self.station.smu.ch1.set_r.volt.side_effect = stopped
                worker.run() if position else worker._do_run()
                self.assertEqual(worker.errors, [])
                self.assertEqual(worker.points, [])
                self.acquisition.assert_not_called()
                self.store.write_point.assert_not_called()
                if position:
                    self.assertEqual(worker.failure, '')
                    self.assertFalse(worker.completed)
                    self.assertEqual(self.store.set_status.call_args.args, ('interrupted',))

    def test_stop_between_readiness_check_and_ramp_preparation_cannot_be_erased(self):
        worker = self.worker()
        checks = [0]
        def readiness():
            checks[0] += 1
            if checks[0] == 2:
                # Reproduce Stop immediately after a successful readiness read,
                # before the worker would clear the channel cancellation flags.
                worker.stop()
            return True
        worker.check_pause_and_stop = readiness
        worker._do_run()
        self.station.smu.ch1.set_r.volt.assert_not_called()
        self.station.smu.ch2.set_r.volt.assert_not_called()
        self.acquisition.assert_not_called()
        self.assertEqual(worker.errors, [])
        self.assertEqual(worker.points, [])
        self.assertTrue(self.station.smu.ch1.ramp_stop_requested)
        self.assertTrue(self.station.smu.ch2.ramp_stop_requested)

    def track_setpoints(self, events, on_write=None):
        """Stateful fake channels: set_r.volt changes what volt() reads back."""
        setpoints = {1: 0., 2: 0.}
        for axis in (1, 2):
            channel = getattr(self.station.smu, f'ch{axis}')
            channel.volt.side_effect = lambda axis=axis: setpoints[axis]

            def write(target, axis=axis):
                if on_write is not None:
                    on_write(axis, target)
                setpoints[axis] = target
                events.append(('ramp', axis, target))
            channel.set_r.volt.side_effect = write
        return setpoints

    def test_optional_final_ramp_moves_both_gates_together_before_finished(self):
        self.params['ramp_to_zero'] = True
        worker = self.worker()
        events = []
        setpoints = self.track_setpoints(events)
        worker.finished.connect(lambda: events.append(('finished',)), DIRECT)
        worker._do_run()
        self.assertEqual(worker.errors, [])
        self.assertEqual(len(worker.points), 1)
        self.assertEqual(events[:2], [('ramp', 1, .2), ('ramp', 2, .3)])
        self.assertEqual(events[-1], ('finished',))
        final = events[2:-1]
        self.assertEqual([axis for _, axis, _ in final], [1, 2] * (len(final) // 2))
        self.assertEqual(final[-2:], [('ramp', 1, 0.), ('ramp', 2, 0.)])
        self.assertEqual(setpoints, {1: 0., 2: 0.})
        previous = {1: .2, 2: .3}
        for _, axis, target in final:
            self.assertLessEqual(abs(target - previous[axis]), .05 + 1e-12)
            previous[axis] = target
        # Both gates stay on the straight line to 0 V: V1/V2 keeps the 0.2/0.3 ratio.
        for (_, _, v1), (_, _, v2) in zip(final[0:-2:2], final[1:-2:2]):
            self.assertAlmostEqual(v1 / v2, .2 / .3)

    def test_final_ramp_is_off_by_default_and_never_runs_inside_position_scans(self):
        for position, enabled in ((False, False), (True, True)):
            with self.subTest(position=position, enabled=enabled):
                self.store.reset_mock()
                self.store.position_finished.return_value = True
                if enabled:
                    self.params['ramp_to_zero'] = True
                worker = self.worker(position)
                events = []
                setpoints = self.track_setpoints(events)
                worker.run() if position else worker._do_run()
                self.assertEqual(events, [('ramp', 1, .2), ('ramp', 2, .3)])
                self.assertEqual(setpoints, {1: .2, 2: .3})
                if position:
                    self.assertTrue(worker.completed)

    def test_stop_during_final_ramp_holds_reached_voltages_without_finished(self):
        self.params['ramp_to_zero'] = True
        worker = self.worker()
        events, writes = [], [0]

        def stop_on_fifth_write(axis, target):
            writes[0] += 1
            if writes[0] == 5:  # two sweep writes, then two final-ramp steps
                worker.stop()
                raise GateRampCancelled('Stop requested')
        setpoints = self.track_setpoints(events, stop_on_fifth_write)
        worker.finished.connect(lambda: events.append(('finished',)), DIRECT)
        worker._do_run()
        self.assertEqual(worker.errors, [])
        self.assertNotIn(('finished',), events)
        self.assertEqual(len(events), 4)
        self.assertEqual(setpoints, {1: events[2][2], 2: events[3][2]})
        self.assertGreater(setpoints[2], 0)

    def test_final_ramp_failure_reports_that_the_sweep_data_is_complete(self):
        self.params['ramp_to_zero'] = True
        worker = self.worker()
        events, writes = [], [0]

        def fail_in_final_ramp(axis, target):
            writes[0] += 1
            if writes[0] == 3:
                raise RuntimeError('setpoint not confirmed')
        self.track_setpoints(events, fail_in_final_ramp)
        worker._do_run()
        self.assertEqual(len(worker.points), 1)
        self.assertEqual(len(worker.errors), 1)
        self.assertIn('Sweep data is complete', worker.errors[0])
        self.assertIn('setpoint not confirmed', worker.errors[0])

    def test_stop_before_manual_worker_run_issues_no_set_commands(self):
        for parameter in ('volt', 'curr', 'output'):
            with self.subTest(parameter=parameter):
                channel = Mock()
                worker = dgs.SingleRampWorker(channel, .2, parameter)
                worker.stop()
                worker.run()
                channel.set_r.volt.assert_not_called()
                channel.set_r.curr.assert_not_called()
                # An output readback after cancellation is allowed, never a SET.
                self.assertTrue(all(not call.args and not call.kwargs for call in channel.output.call_args_list))
                channel.parent.visa_handle.clear.assert_not_called()
                self.assertTrue(channel.ramp_stop_requested)


if __name__ == '__main__':
    unittest.main()
