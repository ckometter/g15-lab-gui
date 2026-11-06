"""Failure propagation tests with mocked instruments only."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from measurement_acquisition import acquire_sweep_spectrum
from measurements.Dual_gate_sweep_gui import SweepWorker
from measurements.B_V_sweep_gui import BVSweepWorker
from measurements.photoluminescence_map_gui import PLMapWorker
from drivers.remote_client import ClientLightFieldCamera


class AcquisitionTests(unittest.TestCase):
    def test_preserves_sum_average_and_raw_frames(self):
        camera = MagicMock()
        camera.remote_lf.get_spectrum.return_value = ([[10, 20], [30, 40]], [1100, 1101])
        for average, expected in ((False, [40, 60]), (True, [20, 30])):
            spectrum, raw = acquire_sweep_spectrum(camera, pixels=2, frames=2, average=average)
            np.testing.assert_array_equal(spectrum, expected)
            np.testing.assert_array_equal(raw, [[10, 20], [30, 40]])

    def test_missing_invalid_and_changed_data_cannot_be_saved_as_a_point(self):
        camera = MagicMock()
        for response in (None, ([], []), ([[1]], [1100]), ([[np.nan, 2]], [1100, 1101]),
                         ([[1, 2]], [1101, 1102])):
            with self.subTest(response=response):
                camera.remote_lf.get_spectrum.return_value = response
                with self.assertRaises((RuntimeError, ValueError)):
                    acquire_sweep_spectrum(camera, pixels=2, frames=1, average=False,
                                           wavelengths=[1100, 1101])

    def test_explicit_simulation_still_supports_multiple_frames(self):
        camera = ClientLightFieldCamera("test_sim_camera", simulated=True)
        self.addCleanup(camera.close)
        camera.num_frames(3)
        _, raw = acquire_sweep_spectrum(camera, pixels=512, frames=3, average=True)
        self.assertEqual(raw.shape, (3, 512))

    def test_all_three_sweep_workers_stop_without_emitting_a_fake_point(self):
        params = dict(a=1, b=0, c=0, d=1, c1_start=0, c1_end=0, c1_pts=1,
                      c2_start=0, c2_end=0, c2_pts=1, sweep_mode="Hold C2, sweep C1",
                      serpentine=False, exposure_time=0.05, settling_time=0,
                      num_pixels=2, wl_axis=[1100, 1101], v_start=0, v_end=0, v_pts=1,
                      control_coeff_ch1=1, control_coeff_ch2=0,
                      b_start=0, b_end=0, b_pts=1, variable_order="Hold B, scan Control",
                      faraday_comp=False, faraday_rot=0, end_ramp_zero=False, end_ramp_ground=False,
                      x_start=0, x_end=0, x_pts=1, y_start=0, y_end=0, y_pts=1,
                      x_param="position X", y_param="position Y")
        for worker_cls in (SweepWorker, BVSweepWorker, PLMapWorker):
            with self.subTest(worker=worker_cls.__name__):
                station = SimpleNamespace(smu=MagicMock(), camera=MagicMock(),
                                          monochromator=MagicMock(), magnet=MagicMock(),
                                          piezo300=MagicMock())
                for channel in (station.smu.ch1, station.smu.ch2):
                    channel.source_mode.return_value = "VOLT"
                    channel.volt.return_value = 0.
                    channel.ramp_stop_requested = False
                station.magnet.simulated = True
                station.magnet._sim_sweep_rate = 1
                station.magnet.output_field.return_value = 0.0
                station.magnet.SWITCH_HEATER_FITTED = False
                station.magnet.is_sweeping.return_value = False
                station.camera.remote_lf.get_spectrum.side_effect = RuntimeError("NIRvana disconnected")
                worker = worker_cls(station, params)
                points, errors = [], []
                worker.point_measured.connect(lambda *args: points.append(args))
                error_signal = worker.error if worker_cls == PLMapWorker else worker.error_occurred
                error_signal.connect(errors.append)
                with patch("os.path.exists", return_value=False):
                    if worker_cls == PLMapWorker:
                        with patch.object(worker, "apply_sweep_values"):
                            worker.run()
                    else:
                        worker.run()
                station.camera.remote_lf.get_spectrum.assert_called_once()
                self.assertEqual(points, [])
                self.assertEqual(errors, ["NIRvana disconnected"])


if __name__ == "__main__":
    unittest.main()
