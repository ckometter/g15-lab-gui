"""Offline regression tests; no sockets connect to laboratory equipment."""
import socket
import threading
import unittest
from unittest.mock import MagicMock, patch

import NIRvana_send_trigger as trig
from drivers.remote_client import NIRvanaSocketWrapper, ClientLightFieldCamera


class NIRvanaSocketTests(unittest.TestCase):
    def setUp(self):
        self.client = NIRvanaSocketWrapper("nirvana.test", 65432)
        self.sock = MagicMock()
        self.sock.__enter__.return_value = self.sock
        self.patcher = patch("socket.socket", return_value=self.sock)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def respond(self, data):
        self.sock.recv.side_effect = [data[:13], data[13:], b""]
        return self.client.get_spectrum()

    def test_multiframe_csv_is_sorted_without_losing_precision(self):
        self.client.set_num_frames(2)
        intensities, wavelengths = self.respond(
            b"Frame,Wavelength,Intensity\n2,1101,40.5\n1,1101,20\n"
            b"2,1100,30\n1,1100,10\n"
        )
        self.assertEqual(wavelengths, [1100, 1101])
        self.assertEqual(intensities, [[10, 20], [30, 40.5]])
        self.assertEqual(self.client.get_wavelengths_cache(), wavelengths)
        self.sock.connect.assert_called_once_with(("nirvana.test", 65432))
        self.assertEqual([c.args[0] for c in self.sock.settimeout.call_args_list], [10, 600])
        self.sock.__exit__.assert_called_once()
        self.assertIsNone(self.client._active_socket)

    def test_direct_trigger_sends_supplied_params_unchanged(self):
        params = {"exposure_time": 0.05, "num_frames": 1, "extra_server_field": "test"}
        self.sock.recv.side_effect = [
            b"Frame,Wavelength,Intensity\n1,1100,42\n", b""
        ]
        wavelengths, intensities = trig.trigger(
            params, device_ip="nirvana.test", port=65432
        )
        self.assertEqual(wavelengths.tolist(), [1100])
        self.assertEqual(intensities.tolist(), [[42]])
        self.sock.sendall.assert_called_once_with(
            b'{"exposure_time": 0.05, "num_frames": 1, "extra_server_field": "test"}'
        )

    def test_dns_failure_has_endpoint_and_phase(self):
        self.sock.connect.side_effect = socket.gaierror("hostname not found")
        with self.assertRaisesRegex(RuntimeError, "nirvana.test:65432 failed while connecting"):
            self.client.get_spectrum()
        self.sock.__exit__.assert_called_once()
        self.assertIsNone(self.client._active_socket)

    def test_empty_response_is_an_error(self):
        self.sock.recv.side_effect = [b""]
        with self.assertRaisesRegex(RuntimeError, "receiving CSV.*0 bytes received"):
            self.client.get_spectrum()

    def test_verification_checks_two_spectra_without_changing_measurement_settings(self):
        self.client.set_exposure_time(2.5)
        self.client.set_num_frames(7)
        csv = b"Frame,Wavelength,Intensity\n1,1100,10\n1,1101,20\n"
        self.sock.recv.side_effect = [csv, b"", csv, b""]
        report = self.client.verify_acquisition()
        self.assertIn("Two consecutive", report)
        self.assertEqual(self.sock.connect.call_count, 2)
        for call in self.sock.sendall.call_args_list:
            self.assertEqual(call.args[0], b'{"command": "start_acquisition", "exposure_time": 50.0, "num_frames": 1}')
        self.assertEqual(self.client.get_exposure_time(), 2.5)
        self.assertEqual(self.client.get_num_frames(), 7)
        self.assertEqual(self.client.acquisition_health["state"], "verified")
        self.assertIsNotNone(self.client.acquisition_health["last_success"])
        self.assertEqual(self.client.get_wavelengths_cache(), [1100, 1101])

    def test_failed_second_check_and_later_recovery(self):
        csv = b"Frame,Wavelength,Intensity\n1,1100,10\n"
        self.sock.recv.side_effect = [csv, b"", b""]
        with self.assertRaisesRegex(RuntimeError, "Server closed"):
            self.client.verify_acquisition()
        self.assertEqual(self.client.acquisition_health["state"], "failed")
        self.sock.recv.side_effect = [csv, b"", csv, b""]
        self.client.verify_acquisition()
        self.assertEqual(self.client.acquisition_health["state"], "verified")

    def test_check_rejects_different_calibrations_and_runtime_failure_revokes_health(self):
        csv = b"Frame,Wavelength,Intensity\n1,1100,10\n"
        self.sock.recv.side_effect = [csv, b"", csv.replace(b"1100", b"1101"), b""]
        with self.assertRaisesRegex(ValueError, "calibration changed"):
            self.client.verify_acquisition()
        self.assertEqual(self.client.acquisition_health["state"], "failed")
        self.respond(csv)
        self.assertEqual(self.client.acquisition_health["state"], "verified")
        self.sock.recv.side_effect = [b""]
        with self.assertRaises(RuntimeError):
            self.client.get_spectrum()
        self.assertEqual(self.client.acquisition_health["state"], "failed")

    def test_recheck_cannot_send_commands_during_an_acquisition(self):
        self.client._acquisition_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.client.verify_acquisition()
            with self.assertRaisesRegex(RuntimeError, "already acquiring"):
                self.client.get_spectrum()
            self.sock.connect.assert_not_called()
        finally:
            self.client._acquisition_lock.release()

    def test_timeout_does_not_accept_partial_csv(self):
        self.sock.recv.side_effect = [b"Frame,Wavelength,Intensity\n1,1100,10\n", socket.timeout("timed out")]
        with self.assertRaisesRegex(RuntimeError, "receiving CSV.*bytes received"):
            self.client.get_spectrum()

    def test_text_server_error_is_visible(self):
        with self.assertRaisesRegex(RuntimeError, "camera unavailable"):
            self.respond(b"ERROR: camera unavailable\n")

    def test_duplicate_pixels_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate wavelengths"):
            self.respond(b"Frame,Wavelength,Intensity\n1,1100,10\n1,1100,20\n")

    def test_mismatched_grids_are_rejected(self):
        self.client.set_num_frames(2)
        with self.assertRaisesRegex(RuntimeError, "different wavelength grids"):
            self.respond(b"Frame,Wavelength,Intensity\n1,1100,10\n2,1101,20\n")

    def test_missing_frame_is_rejected(self):
        self.client.set_num_frames(2)
        with self.assertRaisesRegex(RuntimeError, "Requested 2 frames; received 1"):
            self.respond(b"Frame,Wavelength,Intensity\n1,1100,10\n")

    def test_next_acquisition_uses_updated_exposure_and_frame_count(self):
        self.client.set_exposure_time(0.05)
        self.client.set_num_frames(1)
        self.respond(b"Frame,Wavelength,Intensity\n1,1100,10\n")
        self.client.set_exposure_time(0.2)
        self.client.set_num_frames(2)
        self.respond(
            b"Frame,Wavelength,Intensity\n1,1100,20\n2,1100,30\n"
        )
        sent = [call.args[0] for call in self.sock.sendall.call_args_list]
        self.assertEqual(len(sent), 2)
        self.assertIn(b'"exposure_time": 50.0', sent[0])
        self.assertIn(b'"num_frames": 1', sent[0])
        self.assertIn(b'"exposure_time": 200.0', sent[1])
        self.assertIn(b'"num_frames": 2', sent[1])

    def test_nonfinite_wavelength_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.respond(b"Frame,Wavelength,Intensity\n1,nan,10\n")

    def test_no_fabricated_calibration_or_telemetry(self):
        self.assertIsNone(self.client.get_wavelengths_cache())
        with patch.object(self.client.controls, "get_status", side_effect=RuntimeError("not installed")):
            status = self.client.get_status()
        self.assertEqual(status["status_source"], "local_settings")
        for key in ("temp", "target_temp", "status", "gain", "grating", "wl"):
            self.assertIsNone(status[key])
        self.sock.connect.assert_not_called()

    def test_camera_abort_reaches_socket(self):
        camera = MagicMock()
        camera.simulated = False
        camera.spectrometer_type = "lightfieldG11"
        camera.remote_lf = self.client
        self.client._active_socket = self.sock
        ClientLightFieldCamera.abort_acquisition(camera)
        self.sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        self.sock.close.assert_called_once()


class LocalProtocolTests(unittest.TestCase):
    def test_legacy_live_uses_only_original_commands_and_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            listener.settimeout(3)
            received = []
            def serve():
                for index in range(3):
                    conn, _ = listener.accept()
                    with conn:
                        received.append(conn.recv(4096))
                        conn.sendall(('Frame,Wavelength,Intensity\n1,1100,{}\n1,1101,{}\n'.format(index, index + 1)).encode())
            thread = threading.Thread(target=serve)
            thread.start()
            client = NIRvanaSocketWrapper('127.0.0.1', listener.getsockname()[1])
            try:
                with patch.object(client.controls, 'request', side_effect=AssertionError('New server is not installed')):
                    self.assertTrue(client.get_status()['capabilities']['preview'])
                    client.start_preview()
                    try:
                        for index in range(3):
                            self.assertEqual(client.get_latest_preview_data(), [index, index + 1])
                            self.assertEqual(client.get_wavelengths_cache(), [1100., 1101.])
                        with self.assertRaisesRegex(RuntimeError, 'already acquiring'):
                            client.get_spectrum()
                    finally:
                        client.stop_preview()
            finally:
                thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(received, [b'{"command": "start_acquisition", "exposure_time": 100.0, "num_frames": 1}'] * 3)
            self.assertFalse(client._acquisition_lock.locked())

    def test_trigger_talks_to_a_real_local_tcp_server(self):
        # A loopback server verifies the wire protocol without lab hardware.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            received = []

            def serve():
                conn, _ = listener.accept()
                with conn:
                    received.append(conn.recv(4096))
                    conn.sendall(
                        b"Frame,Wavelength,Intensity\n1,1100,10\n1,1101,20\n"
                    )

            thread = threading.Thread(target=serve)
            thread.start()
            try:
                wavelengths, intensities = trig.trigger(
                    {"exposure_time": 50, "num_frames": 1},
                    device_ip="127.0.0.1",
                    port=port,
                )
            finally:
                thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(
                received, [b'{"exposure_time": 50, "num_frames": 1}']
            )
            self.assertEqual(wavelengths.tolist(), [1100, 1101])
            self.assertEqual(intensities.tolist(), [[10, 20]])


if __name__ == "__main__":
    unittest.main()
