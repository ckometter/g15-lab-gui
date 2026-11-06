"""Device verification and spectrometer readback without hardware."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from drivers.remote_client import NIRvanaMonoWrapper, PyroProxyWrapper
from station_devices import verify_device, configure_device, remove_device, _physical_powermeter


class DeviceCheckTests(unittest.TestCase):
    def test_powermeter_discovery_does_not_close_existing_visa_connections(self):
        from pyvisa.errors import InvalidSession
        for outcome in ('found', 'absent', 'enumeration_failed'):
            with self.subTest(outcome=outcome):
                closed = []
                manager = Mock()
                manager.close.side_effect = lambda: closed.append(True)
                manager.list_resources.return_value = ('ASRL3::INSTR', 'USB0::0x1313::0x8078::PM::INSTR')
                if outcome == 'enumeration_failed':
                    manager.list_resources.side_effect = RuntimeError('Enumeration failed')
                def read_offset():
                    if closed: raise InvalidSession()
                    return 9.0
                meter = object()
                def connect(*args, **kwargs):
                    self.assertEqual(read_offset(), 9.0)
                    if outcome == 'absent': raise RuntimeError('Power meter absent')
                    return meter
                with patch('pyvisa.ResourceManager', return_value=manager), \
                     patch('station_devices.station_setup.ThorlabsPM100D', side_effect=connect):
                    if outcome == 'found':
                        self.assertIs(_physical_powermeter(), meter)
                    else:
                        with self.assertRaises(RuntimeError): _physical_powermeter()
                manager.close.assert_not_called()
                self.assertEqual(read_offset(), 9.0)

    def test_anc300_invalid_session_retry_replaces_only_its_connection(self):
        from pyvisa.errors import InvalidSession
        old = SimpleNamespace(simulated=False, get_axis_status=Mock(side_effect=InvalidSession()), close=Mock())
        other = SimpleNamespace(close=Mock())
        station = SimpleNamespace(components={'piezo300': old, 'smu': other})
        station.remove_component = lambda name: station.components.pop(name)
        replacement = SimpleNamespace(simulated=False)
        def reconnect(station, name, *, simulated):
            self.assertEqual(name, 'piezo300')
            self.assertFalse(simulated)
            station.components[name] = replacement
            return 'VERIFIED (readback)'
        with patch('station_devices.initialize_device', side_effect=reconnect) as initialize:
            self.assertEqual(configure_device(station, 'piezo300', 'Physical'), 'VERIFIED (readback)')
        initialize.assert_called_once()
        old.close.assert_called_once()
        other.close.assert_not_called()
        self.assertIs(station.components['piezo300'], replacement)

    def test_remote_check_has_bounded_timeout_and_restores_acquisition_timeout(self):
        remote = SimpleNamespace(_pyroTimeout=600)
        def status():
            self.assertEqual(remote._pyroTimeout, 5)
            return {"success": True}
        remote.get_status = status
        camera = SimpleNamespace(simulated=False, remote_lf=remote)
        self.assertEqual(verify_device(camera, "camera"), "VERIFIED (readback)")
        self.assertEqual(remote._pyroTimeout, 600)
        remote.get_status = Mock(side_effect=RuntimeError("Disconnected"))
        with self.assertRaisesRegex(RuntimeError, "Disconnected"):
            verify_device(camera, "camera")
        self.assertEqual(remote._pyroTimeout, 600)

    def test_legacy_nirvana_cannot_claim_to_set_or_read_real_wavelength(self):
        mono = NIRvanaMonoWrapper()
        with self.assertRaises(RuntimeError):
            mono.get_wavelength()
        with self.assertRaises(RuntimeError):
            mono.set_wavelength(1100)

    def test_legacy_monochromator_is_manual_without_connection_or_simulation_fallback(self):
        from qcodes.station import Station
        from device_manager import DeviceManagerWindow
        station = Station()
        self.addCleanup(remove_device, station, 'monochromator')
        with patch('station_setup.SPECTROMETER_TYPE', 'lightfieldG11'), patch('socket.create_connection', side_effect=AssertionError('No control probe allowed')):
            status = configure_device(station, 'monochromator', 'Auto')
            self.assertEqual(status, 'MANUAL (set in LightField)')
            self.assertFalse(station.monochromator.simulated)
            self.assertEqual(configure_device(station, 'monochromator', 'Auto'), status)
            self.assertEqual(DeviceManagerWindow.status_kind(True, status), 'limited')

    def test_winspec_grating_round_trip_and_unknown_rejection(self):
        proxy = SimpleNamespace(grating=1)
        remote = PyroProxyWrapper(proxy)
        for grating in (1, 2, 3):
            proxy.grating = grating
            value = remote.get_status()["grating"]
            proxy.grating = None
            remote.set_grating(value)
            self.assertEqual(proxy.grating, grating)
        with self.assertRaises(ValueError):
            remote.set_grating("[600][0][0]")
        self.assertEqual(proxy.grating, 3)


if __name__ == "__main__":
    unittest.main()
