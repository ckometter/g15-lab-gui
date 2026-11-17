"""INITIALIZE for the uVoltDAQ: the driver against a fake serial port, and the Device Manager button.

No hardware: INITIALIZE sets every output to 0 V at once.
"""
import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyvisa
from PyQt6 import QtWidgets

from device_manager import DeviceManagerWindow
from drivers.qpg_uvoltdaq import INITIALIZE_REPLY_TIMEOUT_MS, QPGuVoltDAQ
from gate_safety import GATE_MAX_RAMP_STEP_V


class FakeDacPort:
    """Answers GET_DAC like a uVoltDAQ; the reply to INITIALIZE is a line or nothing."""

    def __init__(self, reply=None, readback=None, stray=0, read_error=None):
        self.timeout = 5000
        self.reply, self.readback, self.read_error = reply, readback or {}, read_error
        self.bytes_in_buffer = stray
        self.written, self.queries, self.discarded, self.read_timeouts = [], [], [], []

    def write(self, cmd):
        self.written.append(cmd)

    def read(self):
        self.read_timeouts.append(self.timeout)
        if self.read_error is not None:
            raise self.read_error
        if self.reply is None:
            raise pyvisa.errors.VisaIOError(pyvisa.constants.StatusCode.error_timeout)
        return self.reply

    def read_bytes(self, count):
        self.discarded.append(count)
        self.bytes_in_buffer = 0
        return b"?" * count

    def query(self, cmd):
        self.queries.append(cmd)
        return f"{self.readback.get(int(cmd.split(',')[1]), 0.0):.6f}"

    def close(self):
        pass


class InitializeDriverTests(unittest.TestCase):
    def connect(self, **port_settings):
        dac = QPGuVoltDAQ("dac_" + self._testMethodName, address="simulated")
        port = FakeDacPort(**port_settings)
        dac.simulated, dac._visa_handle = False, port
        dac.visa_log = logging.getLogger("test.uvoltdaq")

        def close():
            dac.simulated = True
            dac.close()

        self.addCleanup(close)
        return dac, port

    def test_a_one_line_reply_is_returned_and_all_outputs_are_checked(self):
        dac, port = self.connect(reply="OK")
        self.assertEqual(dac.initialize_outputs(), "OK")
        self.assertEqual(port.written, ["INITIALIZE"])
        self.assertEqual(port.read_timeouts, [INITIALIZE_REPLY_TIMEOUT_MS])
        self.assertEqual(port.timeout, 5000)
        self.assertEqual(port.queries, [f"GET_DAC,{channel}" for channel in range(4)])

    def test_no_reply_is_accepted(self):
        dac, port = self.connect(reply=None)
        self.assertEqual(dac.initialize_outputs(), "")
        self.assertEqual(port.timeout, 5000)

    def test_stray_bytes_are_discarded_so_later_queries_stay_in_step(self):
        dac, port = self.connect(reply="OK", stray=4)
        dac.initialize_outputs()
        self.assertEqual(port.discarded, [4])

    def test_an_output_not_at_0_v_afterwards_is_named(self):
        dac, _ = self.connect(reply=None, readback={2: 0.5})
        with self.assertRaisesRegex(RuntimeError, r"INITIALIZE was sent, but DAC2 reads 0\.5 V instead of 0 V"):
            dac.initialize_outputs()

    def test_other_port_errors_are_not_taken_for_no_reply(self):
        error = pyvisa.errors.VisaIOError(pyvisa.constants.StatusCode.error_connection_lost)
        dac, port = self.connect(read_error=error)
        with self.assertRaises(pyvisa.errors.VisaIOError):
            dac.initialize_outputs()
        self.assertEqual(port.timeout, 5000)

    def test_not_sent_while_an_output_ramps(self):
        dac, port = self.connect(reply="OK")
        dac.dac1.ramping = True
        with self.assertRaisesRegex(RuntimeError, "ramping"):
            dac.initialize_outputs()
        self.assertEqual(port.written, [])

    def test_simulated_outputs_ramp_together_then_initialize_to_0_v(self):
        dac = QPGuVoltDAQ("dac_simulated_ramp", address="simulated")
        self.addCleanup(dac.close)
        dac.dac0._sim_voltage, dac.dac2._sim_voltage = 0.12, -0.07
        seen = []
        start = dac.ramp_outputs_to_zero(lambda step, steps: seen.append(dac.setpoints()))
        self.assertEqual(start[0], 0.12)
        self.assertEqual(dac.setpoints(), {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
        self.assertEqual(len(seen), 3)  # 0.12 V in steps of at most 0.05 V
        for previous, now in zip([start] + seen, seen):
            self.assertLessEqual(abs(previous[0] - now[0]), GATE_MAX_RAMP_STEP_V + 1e-12)
            if now[0]:
                self.assertAlmostEqual(now[2] / now[0], -0.07 / 0.12)  # Along a straight line to 0 V.
        dac.dac1._sim_voltage = 0.3
        self.assertEqual(dac.initialize_outputs(), "")
        self.assertEqual(dac.setpoints(), {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})


class InitializeButtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def manager(self, dashboard=None, with_dac=True):
        components = {}
        if with_dac:
            components["dac"] = QPGuVoltDAQ("dac_button_" + self._testMethodName, address="simulated")
            self.addCleanup(components["dac"].close)
        manager = DeviceManagerWindow(SimpleNamespace(components=components), lambda: None)
        self.addCleanup(manager.close)
        manager.dashboard = dashboard
        return manager, components.get("dac")

    def press_initialize(self, manager, choice):
        with patch.object(manager, "choose_dac_initialization", return_value=choice) as ask, \
                patch.object(QtWidgets.QMessageBox, "critical") as critical:
            manager.dac_init_button.click()
            if manager.dac_worker is not None:
                self.assertTrue(manager.dac_worker.wait(5000))
            self.app.processEvents()
        return ask, critical

    def status(self, manager):
        return manager.table.item(manager.rows["dac"], 2).text()

    def test_button_needs_a_connected_dac(self):
        manager, _ = self.manager(with_dac=False)
        self.assertFalse(manager.dac_init_button.isEnabled())
        manager, _ = self.manager()
        self.assertTrue(manager.dac_init_button.isEnabled())
        self.assertIs(manager.connect_buttons["dac"].parentWidget(), manager.dac_init_button.parentWidget())

    def test_dialog_lists_the_setpoints_and_offers_a_ramp_only_when_needed(self):
        manager, _ = self.manager()
        box, buttons = manager.dac_initialize_dialog({0: 0.12, 1: 0.0, 2: -0.07, 3: 0.0}, False, 1.9)
        self.addCleanup(box.deleteLater)
        text = box.text()
        for part in ("from tri-state", "sets all four outputs to 0 V at once",
                     "DAC0 +0.120 V, DAC1 +0.000 V, DAC2 -0.070 V, DAC3 +0.000 V",
                     "DAC0, DAC2 are not at 0 V", "(about 2 s)", "Initialize now drops them to 0 V at once"):
            self.assertIn(part, text)
        self.assertEqual(list(buttons), ["ramp", "now", "cancel"])
        self.assertIs(box.defaultButton(), buttons["ramp"])
        box, buttons = manager.dac_initialize_dialog({channel: 0.0 for channel in range(4)}, True, 0.0)
        self.addCleanup(box.deleteLater)
        self.assertEqual(list(buttons), ["now", "cancel"])
        self.assertEqual(buttons["now"].text(), "Initialize")
        self.assertIn("(simulated)", box.windowTitle())

    def test_initialize_now_sets_every_output_to_0_v(self):
        manager, dac = self.manager()
        ask, critical = self.press_initialize(manager, "now")
        ask.assert_called_once()
        critical.assert_not_called()
        self.assertEqual(dac.setpoints(), {channel: 0.0 for channel in range(4)})
        self.assertEqual(self.status(manager), "SIMULATED (initialized)")
        self.assertTrue(manager.results["dac"][0])
        self.assertIn("INITIALIZE sent; no reply.", manager.results["dac"][2])
        self.assertIn("INITIALIZE sent", manager.application_log.toPlainText())
        self.assertTrue(manager.dac_init_button.isEnabled())
        self.assertTrue(manager.connect_buttons["dac"].isEnabled())

    def test_ramp_first_takes_the_outputs_to_0_v_before_initializing(self):
        manager, dac = self.manager()
        dac.dac0._sim_voltage = 0.12
        ask, _ = self.press_initialize(manager, "ramp")
        self.assertEqual(ask.call_args.args[0][0], 0.12)
        self.assertAlmostEqual(ask.call_args.args[2], 1.2)  # 0.12 V at 0.1 V/s
        self.assertEqual(dac.setpoints(), {channel: 0.0 for channel in range(4)})
        self.assertIn("Ramped to 0 V from DAC0 +0.120 V", manager.results["dac"][2])

    def test_cancel_changes_nothing(self):
        dashboard = MagicMock()
        dashboard.measurement_active.return_value = False
        manager, dac = self.manager(dashboard)
        dac.dac0._sim_voltage = 0.12
        self.press_initialize(manager, None)
        self.assertIsNone(manager.dac_worker)
        self.assertEqual(dac.setpoints()[0], 0.12)
        dashboard.prepare_device_change.assert_not_called()

    def test_refused_while_a_measurement_runs(self):
        dashboard = MagicMock()
        dashboard.measurement_active.return_value = True
        manager, _ = self.manager(dashboard)
        ask, _ = self.press_initialize(manager, "now")
        ask.assert_not_called()
        self.assertIsNone(manager.dac_worker)
        self.assertIn("Finish or stop the active measurement", manager.readiness.text())

    def test_dashboard_polling_is_paused_around_it(self):
        dashboard = MagicMock()
        dashboard.measurement_active.return_value = False
        dashboard.prepare_device_change.return_value = True
        manager, _ = self.manager(dashboard)
        self.press_initialize(manager, "now")
        dashboard.prepare_device_change.assert_called_once()
        dashboard.finish_device_change.assert_called_once()

    def test_nothing_is_sent_when_the_dashboard_cannot_pause(self):
        dashboard = MagicMock()
        dashboard.measurement_active.return_value = False
        dashboard.prepare_device_change.return_value = False
        manager, dac = self.manager(dashboard)
        with patch.object(dac, "initialize_outputs") as initialize:
            self.press_initialize(manager, "now")
        initialize.assert_not_called()
        self.assertIsNone(manager.dac_worker)
        dashboard.finish_device_change.assert_not_called()

    def test_a_failure_is_red_reported_and_still_resumes_the_dashboard(self):
        dashboard = MagicMock()
        dashboard.measurement_active.return_value = False
        dashboard.prepare_device_change.return_value = True
        manager, dac = self.manager(dashboard)
        message = "INITIALIZE was sent, but DAC2 reads 0.5 V instead of 0 V."
        with patch.object(dac, "initialize_outputs", side_effect=RuntimeError(message)):
            _, critical = self.press_initialize(manager, "now")
        critical.assert_called_once()
        self.assertIn(message, critical.call_args.args[2])
        self.assertEqual(self.status(manager), "INITIALIZE FAILED")
        self.assertFalse(manager.results["dac"][0])
        dashboard.finish_device_change.assert_called_once()
        self.assertTrue(manager.dac_init_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
