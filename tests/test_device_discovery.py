"""Read-only discovery and reconnect with mocked ports, never real equipment."""
import threading
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import Mock, patch

import device_discovery as discovery
from drivers.attocube_anc150 import AttocubeANC150
from station_devices import configure_device, verify_device


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        discovery._known_ports.clear()
        self.addCleanup(discovery._known_ports.clear)

    def test_identity_probe_only_sends_ver_and_closes_port(self):
        connection = Mock(in_waiting=200)
        connection.read.return_value = b'ver\r\nattocube ANC300 controller version 1.2\r\nANC150 compatibility console\r\nOK\r\n'
        with patch('device_discovery.serial.Serial', return_value=connection):
            result = discovery.identify_attocube('COM3')
        self.assertEqual(result['model'], 'ANC300')
        connection.write.assert_called_once_with(b'ver\r')
        connection.close.assert_called_once()
        self.assertFalse(connection.dtr)
        self.assertFalse(connection.rts)

    def test_legacy_anc150_banner_is_recognized(self):
        connection = Mock(in_waiting=200)
        connection.read.return_value = b'attocube controller version 1.4 build date 2005\r\nserial number 1720362005264\r\nOK\r\n'
        with patch('device_discovery.serial.Serial', return_value=connection):
            self.assertEqual(discovery.identify_attocube('COM10')['model'], 'ANC150')

    def test_replugged_controller_is_found_on_new_port(self):
        inventory = [{'port': 'COM8', 'usb': True}, {'port': 'COM12', 'usb': True}]
        def identify(port):
            return {'port': port, 'model': 'ANC300'} if port == 'COM12' else {'port': port, 'model': 'ANC150'}
        with patch.object(discovery, 'serial_inventory', return_value=inventory), patch.object(discovery, 'identify_attocube', side_effect=identify) as probe:
            self.assertEqual(discovery.find_attocube_address('ANC300', 'COM3'), 'ASRL12::INSTR')
            self.assertEqual([call.args[0] for call in probe.call_args_list], ['COM8', 'COM12'])

    def test_inventory_scan_during_measurement_never_opens_ports(self):
        with patch.object(discovery, 'serial_inventory', return_value=[{'port': 'COM3', 'usb': True}]), patch.object(discovery, 'identify_attocube') as probe:
            report = discovery.scan_attocubes(identify=False)
            probe.assert_not_called()
        self.assertEqual(report['controllers'], [])
        self.assertFalse(report['identity_probed'])

    def test_busy_ports_are_never_probed(self):
        with patch.object(discovery, 'serial_inventory', return_value=[{'port': 'COM3', 'usb': True}]), patch.object(discovery, 'identify_attocube') as probe:
            discovery.scan_attocubes(skip_ports={'COM3'})
            probe.assert_not_called()

    def test_stale_connection_is_reopened_before_simulation_fallback(self):
        old = SimpleNamespace(simulated=False, close=Mock())
        other = object()
        station = SimpleNamespace(components={'piezo300': old, 'other': other})
        station.remove_component = lambda name: station.components.pop(name)
        replacement = SimpleNamespace(simulated=False)
        def initialize(station, name, *, simulated):
            self.assertFalse(simulated)
            station.components[name] = replacement
            return 'VERIFIED (readback)'
        with patch('station_devices.verify_device', side_effect=RuntimeError('Old port disappeared')), patch('station_devices.initialize_device', side_effect=initialize) as reconnect:
            self.assertEqual(configure_device(station, 'piezo300'), 'VERIFIED (readback)')
        reconnect.assert_called_once()
        old.close.assert_called_once()
        self.assertIs(station.components['piezo300'], replacement)
        self.assertIs(station.components['other'], other)

    def test_anc150_read_failure_never_switches_mode_or_returns_zero(self):
        controller = SimpleNamespace(simulated=False, ask=Mock(return_value='Axis not in computer control mode\r\nERROR\r\n> '), write=Mock())
        controller._read_numeric = MethodType(AttocubeANC150._read_numeric, controller)
        for getter in (AttocubeANC150._get_voltage, AttocubeANC150._get_frequency):
            with self.assertRaisesRegex(RuntimeError, 'computer control'):
                getter(controller, 1)
        controller.write.assert_not_called()

    def test_connected_anc150_in_manual_mode_stays_physical(self):
        controller = SimpleNamespace(simulated=False, ask=Mock(return_value='Axis not in computer control mode\r\nERROR\r\n> '), write=Mock())
        station = SimpleNamespace(components={'piezo150': controller})
        with patch('station_devices.initialize_device') as initialize:
            status = configure_device(station, 'piezo150', 'Auto')
            self.assertEqual(status, 'MANUAL (enable computer control)')
            initialize.assert_not_called()
        self.assertIs(station.components['piezo150'], controller)
        controller.write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
