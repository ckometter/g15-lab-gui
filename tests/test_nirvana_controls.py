"""Offline LightField simulation and real loopback TCP; never lab connections."""
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from drivers.nirvana_control_client import NIRvanaControlClient
from drivers.remote_client import NIRvanaSocketWrapper
from nirvana_server.NIRvana_control import LightFieldController
from nirvana_server.NIRvana_device_server import Server


class Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def emit(self, args=None):
        for handler in self.handlers:
            handler(None, args)


class FakeExperiment:
    def __init__(self):
        self.values = {"grating": "[150 g/mm][0]", "wl": 1100., "exposure": 50., "frames": 1}
        self.choices = ["[150 g/mm][0]", "[600 g/mm][1]"]
        self.ImageDataSetReceived = Event()
        self.ExperimentCompleted = Event()
        self.IsRunning = False
        self.set_calls = []
        self.acquire_calls = 0
        self.preview_calls = 0
        self.csv_path = None
        self.SystemColumnCalibration = SimpleNamespace(Length=2, Get=lambda i: [1100., 1101.][i])

    def Exists(self, key):
        return key in self.values

    def GetValue(self, key):
        return self.values[key]

    def IsValid(self, key, value):
        return 300 <= value <= 2500 if key == "wl" else True

    def SetValue(self, key, value):
        self.set_calls.append((key, value))
        self.values[key] = value

    def GetCurrentCapabilities(self, key):
        return self.choices

    def Preview(self):
        self.preview_calls += 1
        self.IsRunning = True

    def Stop(self):
        self.IsRunning = False

    def Acquire(self):
        self.acquire_calls += 1
        self.csv_path.write_text("Frame,Wavelength,Intensity\n1,1100,10\n1,1101,20\n")
        self.ExperimentCompleted.emit()

    def emit_frame(self):
        frame = SimpleNamespace(Width=2, Height=2, GetData=lambda: [1., 2., 3., 4.])
        dataset = SimpleNamespace(Frames=1, Regions=SimpleNamespace(Length=1), GetFrame=lambda *args: frame)
        self.ImageDataSetReceived.emit(SimpleNamespace(ImageDataSet=dataset))


class ControlTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.experiment = FakeExperiment()
        self.controller = LightFieldController(
            self.experiment, SimpleNamespace(GratingSelected="grating", GratingCenterWavelength="wl"),
            SimpleNamespace(ShutterTimingExposureTime="exposure"),
            SimpleNamespace(AcquisitionFramesToStore="frames"), temp.name)
        self.experiment.csv_path = self.controller.csv_path

    def start_server(self, controls=True):
        server = Server(("127.0.0.1", 0), self.controller, controls=controls)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.addCleanup(stop)
        return server.server_address[1]

    def test_readback_lists_real_choices_and_never_writes_or_deletes_csv(self):
        self.controller.csv_path.write_text("existing measurement")
        client = NIRvanaControlClient("127.0.0.1", self.start_server())
        status = client.get_status()
        self.assertEqual(status["gratings"], self.experiment.choices)
        self.assertEqual(status["wl"], 1100.)
        self.assertTrue(status["capabilities"]["grating"])
        self.assertEqual(self.experiment.set_calls, [])
        self.assertEqual(self.controller.csv_path.read_text(), "existing measurement")

    def test_exact_grating_and_wavelength_set_and_readback_over_tcp(self):
        client = NIRvanaControlClient("127.0.0.1", self.start_server())
        self.assertEqual(client.set_grating(self.experiment.choices[1])["grating"], self.experiment.choices[1])
        self.assertEqual(client.set_wavelength(1300)["wl"], 1300)
        self.assertEqual(client.get_wavelength(), 1300)
        self.assertEqual(self.experiment.set_calls, [("grating", self.experiment.choices[1]), ("wl", 1300.)])

    def test_legacy_grating_setting_name_is_supported(self):
        self.controller.spectrometer_settings = SimpleNamespace(Grating="grating")
        status = self.controller.command({"command": "set_grating", "value": self.experiment.choices[1]})
        self.assertEqual(status["grating"], self.experiment.choices[1])
        self.assertFalse(status["capabilities"]["wavelength"])

    def test_invalid_targets_and_busy_experiment_never_move(self):
        for command, value in (("set_grating", "600"), ("set_wavelength", float("nan")), ("set_wavelength", 99999)):
            with self.assertRaises(ValueError):
                self.controller.command({"command": command, "value": value})
        self.experiment.IsRunning = True
        with self.assertRaisesRegex(RuntimeError, "acquiring"):
            self.controller.command({"command": "set_wavelength", "value": 1300})
        self.assertEqual(self.experiment.set_calls, [])

    def test_busy_command_is_rejected_instead_of_queued(self):
        with self.controller.exclusive():
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.controller.command({"command": "set_grating", "value": self.experiment.choices[1]})
        self.assertEqual(self.experiment.set_calls, [])

    def test_readback_mismatch_is_an_error_not_success(self):
        self.experiment.SetValue = Mock()
        with self.assertRaisesRegex(RuntimeError, "readback does not match"):
            self.controller.command({"command": "set_wavelength", "value": 1300})

    def test_unknown_acquisition_command_keeps_existing_csv(self):
        self.controller.csv_path.write_text("existing measurement")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.controller.acquire_csv({"command": "get_status"}, Mock())
        self.assertEqual(self.controller.csv_path.read_text(), "existing measurement")
        self.assertEqual(self.experiment.acquire_calls, 0)

    def test_original_acquisition_client_works_with_replacement_server(self):
        camera = NIRvanaSocketWrapper("127.0.0.1", self.start_server(controls=False))
        intensities, wavelengths = camera.get_spectrum()
        self.assertEqual(intensities, [[10., 20.]])
        self.assertEqual(wavelengths, [1100., 1101.])
        self.assertFalse(self.controller.csv_path.exists())

    def test_fragmented_json_request_is_assembled(self):
        port = self.start_server()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as conn:
            conn.sendall(b'{"command":')
            conn.sendall(b'"get_status"}\n')
            received = b""
            while not received.endswith(b"\n"):
                received += conn.recv(65536)
        self.assertEqual(json.loads(received)["wl"], 1100)

    def test_native_preview_uses_memory_stops_and_never_acquires_or_writes(self):
        camera = NIRvanaSocketWrapper("127.0.0.1", 1, enable_controls=True)
        camera.controls = NIRvanaControlClient("127.0.0.1", self.start_server())
        camera.start_preview()
        try:
            self.assertIsNone(camera.get_latest_preview_data())
            self.experiment.emit_frame()
            self.assertEqual(camera.get_latest_preview_data(), [4., 6.])
            self.assertEqual(camera.get_wavelengths_cache(), [1100., 1101.])
            self.assertIsNone(camera.get_latest_preview_data())  # No duplicate frame.
            with self.assertRaisesRegex(RuntimeError, "busy"):
                camera.set_grating(self.experiment.choices[1])
            with self.assertRaisesRegex(RuntimeError, "already acquiring"):
                camera.get_spectrum()
        finally:
            camera.stop_preview()
        self.assertFalse(self.experiment.IsRunning)
        self.assertEqual(self.experiment.acquire_calls, 0)
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertTrue(camera._acquisition_lock.acquire(blocking=False))
        camera._acquisition_lock.release()

    def test_lost_preview_client_expires_and_wrong_client_cannot_stop_it(self):
        self.controller.command({"command": "start_preview", "token": "owner", "exposure_time": 50})
        with self.assertRaisesRegex(RuntimeError, "another client"):
            self.controller.command({"command": "stop_preview", "token": "someone else"})
        self.controller.preview_touched = time.monotonic() - 16
        self.controller.expire_preview()
        self.assertIsNone(self.controller.preview_token)
        self.assertFalse(self.experiment.IsRunning)

    def test_missing_control_server_does_not_probe_acquisition_port(self):
        camera = NIRvanaSocketWrapper("offline.test", 65432, enable_controls=True)
        with patch("socket.create_connection", side_effect=OSError("offline")) as connect:
            status = camera.get_status()
        connect.assert_called_once_with(("offline.test", 65433), timeout=3.)
        self.assertEqual(status["status_source"], "local_settings")
        self.assertIsNone(status["grating"])

    def test_failed_preview_never_falls_back_to_csv_acquisition(self):
        camera = NIRvanaSocketWrapper("offline.test", 65432, enable_controls=True)
        with patch.object(camera.controls, "request", side_effect=RuntimeError("offline")), patch.object(camera, "_acquire") as acquire:
            with self.assertRaisesRegex(RuntimeError, "offline"):
                camera.start_preview()
            acquire.assert_not_called()
        self.assertIsNone(camera._preview_token)
        self.assertFalse(camera._acquisition_lock.locked())


if __name__ == "__main__":
    unittest.main()
