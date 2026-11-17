"""NI-VISA's serial-port lock: serial devices fail fast instead of hanging, and nothing touches a port."""
import contextlib
import ctypes
import io
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import station_devices
import visa_serial_lock
from visa_serial_lock import SerialLockHeld, serial_lock_owned, wait_for_serial_lock


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeStation:
    def __init__(self):
        self.components = {}

    def add_component(self, component, update_snapshot=False):
        self.components[component.name] = component

    def remove_component(self, name):
        return self.components.pop(name)


def fake_driver(name, address=None, **kwargs):
    return SimpleNamespace(name=name, address=address, simulated=address == "simulated", close=Mock())


class WaitForSerialLockTests(unittest.TestCase):
    def setUp(self):
        visa_serial_lock._stuck_at = None
        self.addCleanup(setattr, visa_serial_lock, "_stuck_at", None)
        self.clock = FakeClock()

    def wait(self, owned, device="The ANC150"):
        return wait_for_serial_lock(device, owned=owned, clock=self.clock, sleep=self.clock.sleep)

    def assertWaitedFiveSeconds(self, since):
        # Polls every 0.1 s, so it can overshoot by one poll.
        self.assertGreaterEqual(self.clock.now - since, 5.0)
        self.assertLess(self.clock.now - since, 5.15)

    def test_free_or_uncheckable_lock_returns_at_once(self):
        for state in (False, None):
            with self.subTest(state=state):
                self.assertEqual(self.wait(lambda: state), state)
        self.assertEqual(self.clock.now, 1000.0)

    def test_lock_released_within_the_wait_is_fine(self):
        states = iter([True] * 10 + [False])
        self.assertFalse(self.wait(lambda: next(states)))
        self.assertAlmostEqual(self.clock.now, 1001.0)

    def test_stuck_lock_fails_after_five_seconds_with_the_fix(self):
        with self.assertRaises(SerialLockHeld) as raised:
            self.wait(lambda: True)
        self.assertWaitedFiveSeconds(1000.0)
        message = str(raised.exception)
        for part in ("The ANC150 was not opened", "NI Configuration Manager (nimxs.exe)",
                     "Stop-Process -Name nimxs -Force", "Start-Service mxssvr, nidevldu, NITaggerService",
                     "Connect / Retry", "GPIB and USB devices are not affected"):
            self.assertIn(part, message)

    def test_next_device_fails_at_once_then_waits_again_after_a_minute(self):
        with self.assertRaises(SerialLockHeld):
            self.wait(lambda: True)
        start = self.clock.now
        with self.assertRaisesRegex(SerialLockHeld, "The ANC300 was not opened"):
            self.wait(lambda: True, "The ANC300")
        self.assertEqual(self.clock.now, start)
        self.clock.now += 61
        with self.assertRaises(SerialLockHeld):
            self.wait(lambda: True, "The uVoltDAQ")
        self.assertWaitedFiveSeconds(start + 61)

    def test_a_free_lock_clears_the_stuck_result(self):
        with self.assertRaises(SerialLockHeld):
            self.wait(lambda: True)
        self.assertFalse(self.wait(lambda: False))
        start = self.clock.now
        with self.assertRaises(SerialLockHeld):
            self.wait(lambda: True)
        self.assertWaitedFiveSeconds(start)


@unittest.skipUnless(sys.platform == "win32", "Windows named mutexes")
class SerialLockOwnedTests(unittest.TestCase):
    def test_reads_a_mutex_held_by_another_thread_without_taking_it(self):
        name = f"Local\\g15-lab-gui-test-lock-{os.getpid()}"
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.assertIsNone(serial_lock_owned(name))  # no such mutex: nothing to check

        taken, release, released, close = (threading.Event() for _ in range(4))

        def holder():
            handle = kernel32.CreateMutexW(None, True, name)  # created already owned by this thread
            taken.set()
            release.wait(10)
            kernel32.ReleaseMutex(handle)
            released.set()
            close.wait(10)
            kernel32.CloseHandle(handle)

        thread = threading.Thread(target=holder, daemon=True)
        thread.start()
        self.addCleanup(close.set)
        self.addCleanup(release.set)
        self.assertTrue(taken.wait(5))
        self.assertIs(serial_lock_owned(name), True)
        self.assertIs(serial_lock_owned(name), True)  # reading it twice took nothing
        release.set()
        self.assertTrue(released.wait(5))
        self.assertIs(serial_lock_owned(name), False)
        close.set()
        thread.join(5)
        self.assertIsNone(serial_lock_owned(name))

    def test_the_real_nivisa_lock_can_be_read(self):
        self.assertIn(serial_lock_owned(), (True, False, None))


class DeviceConnectionTests(unittest.TestCase):
    def setUp(self):
        visa_serial_lock._stuck_at = None
        self.addCleanup(setattr, visa_serial_lock, "_stuck_at", None)

    def test_serial_devices_check_the_lock_before_any_port_is_touched(self):
        for name, driver in (("piezo150", "AttocubeANC150"), ("piezo300", "AttocubeANC300"), ("dac", "QPGuVoltDAQ")):
            with self.subTest(name=name):
                station = FakeStation()
                stuck = SerialLockHeld(f"{station_devices.SERIAL_VISA_DEVICES[name]} was not opened")
                with patch("station_devices.wait_for_serial_lock", side_effect=stuck) as wait, \
                     patch(f"station_devices.station_setup.{driver}") as opened, \
                     patch("device_discovery.find_attocube_address") as probe_ports:
                    with self.assertRaises(SerialLockHeld):
                        station_devices.initialize_device(station, name, simulated=False)
                wait.assert_called_once_with(station_devices.SERIAL_VISA_DEVICES[name])
                opened.assert_not_called()
                probe_ports.assert_not_called()
                self.assertEqual(station.components, {})

    def test_a_free_lock_lets_the_serial_device_open(self):
        station = FakeStation()
        with patch("station_devices.wait_for_serial_lock") as wait, \
             patch("station_devices.station_setup.QPGuVoltDAQ", side_effect=fake_driver), \
             patch("station_devices.verify_device", return_value="VERIFIED (readback)"):
            station_devices.initialize_device(station, "dac", simulated=False)
        wait.assert_called_once_with("The uVoltDAQ")
        self.assertEqual(station.components["dac"].address, station_devices.station_setup.DAC_ADDRESS)

    def test_simulated_and_non_serial_devices_skip_the_check(self):
        station = FakeStation()
        with patch("station_devices.wait_for_serial_lock") as wait, \
             patch("station_devices.station_setup.AttocubeANC150", side_effect=fake_driver), \
             patch("station_devices.station_setup.StanfordSR830", side_effect=fake_driver), \
             patch("station_devices.verify_device", return_value="VERIFIED (readback)"):
            station_devices.initialize_device(station, "piezo150", simulated=True)
            station_devices.initialize_device(station, "lockin", simulated=False)
        wait.assert_not_called()
        self.assertEqual(set(station.components), {"piezo150", "lockin"})

    def test_auto_mode_falls_back_to_simulation_and_shows_the_fix(self):
        station = FakeStation()
        output = io.StringIO()

        def stuck(device):
            return wait_for_serial_lock(device, owned=lambda: True, wait=0.0)

        with patch("station_devices.wait_for_serial_lock", side_effect=stuck), \
             patch("station_devices.station_setup.AttocubeANC150", side_effect=fake_driver) as driver, \
             patch("device_discovery.find_attocube_address") as probe_ports, \
             contextlib.redirect_stdout(output):
            status = station_devices.configure_device(station, "piezo150", "Auto")
        self.assertEqual(status, "SIMULATED (physical failed)")
        self.assertTrue(station.components["piezo150"].simulated)
        self.assertEqual([call.kwargs["address"] for call in driver.call_args_list], ["simulated"])
        probe_ports.assert_not_called()
        self.assertIn("SerialLockHeld: The ANC150 was not opened", output.getvalue())
        self.assertIn("Stop-Process -Name nimxs -Force", output.getvalue())

    def test_awg_and_power_meter_search_only_usb(self):
        manager = Mock()
        manager.list_resources.return_value = ("USB0::0xF4EC::0x1102::SDG::INSTR", "USB0::0x1313::0x8078::PM::INSTR")
        with patch("pyvisa.ResourceManager", return_value=manager), \
             patch("station_devices.station_setup.SiglentSDG2042X", side_effect=fake_driver), \
             patch("station_devices.station_setup.ThorlabsPM100D", side_effect=fake_driver):
            self.assertEqual(station_devices._physical_awg().address, "USB0::0xF4EC::0x1102::SDG::INSTR")
            self.assertEqual(station_devices._physical_powermeter().address, "USB0::0x1313::0x8078::PM::INSTR")
        self.assertEqual([call.args for call in manager.list_resources.call_args_list],
                         [("USB?*::INSTR",), ("USB?*::INSTR",)])
        manager.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
