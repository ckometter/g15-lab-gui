"""The measurement queue: entries, files, records, and Dual-Gate runs back to back.

Simulation and temporary folders only; any hardware or network access fails the test.
"""
import contextlib
import importlib
import json
import os
from datetime import datetime
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import h5py
import numpy as np
from PyQt6 import QtWidgets, QtTest
from qcodes.station import Station

import measurement_queue as mq
from lab_gui import LabMainWindow
from measurements import dgs_queue
from measurements.position_tab import PositionTab

KEYSIGHT = {"1": "smu:1", "2": "smu:2"}
WINDOWS = {1: (-1.0, 1.0), 2: (-1.0, 1.0)}


def dgs_settings(**changes):
    settings = {"c1_name": "Vc1", "c2_name": "Vc2",
                "c1_start": 0.0, "c1_end": 0.01, "c1_pts": 2,
                "c2_start": 0.0, "c2_end": 0.01, "c2_pts": 3,
                "coeff_a": 1.0, "coeff_b": 0.0, "coeff_c": 0.0, "coeff_d": 1.0,
                "sweep_mode": "Hold C1, sweep C2", "serpentine": True, "ramp_to_zero": False,
                "exposure_time": 0.1, "settling_time": 0.05, "filename_wavelength_nm": 1090.0,
                "gate_outputs": dict(KEYSIGHT)}
    settings.update(changes)
    return settings


def problems(settings, **envelope):
    envelope = {"gate_outputs": {1: "smu:1", 2: "smu:2"}, "gate_limits": WINDOWS, **envelope}
    return dgs_queue.entry_problems(settings, **envelope)


class QueueFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_a_queue_file_and_its_record_open_as_the_same_queue(self):
        entries = [mq.QueueEntry("dual_gate", dgs_settings(), "first"),
                   mq.QueueEntry("dual_gate", dgs_settings(c1_end=0.02))]
        path = self.directory / "queue.json"
        mq.save_queue_file(path, entries, {"ramp_to_zero_after_last": False})
        loaded, options = mq.load_queue_file(path)
        self.assertEqual([(e.tab, e.settings, e.note) for e in loaded],
                         [(e.tab, e.settings, e.note) for e in entries])
        self.assertEqual(options, {"ramp_to_zero_after_last": False})
        self.assertFalse((self.directory / "queue.json.tmp").exists())
        # A record also keeps what happened; it opens as the same queue, every entry pending.
        entries[0].result = {"status": mq.DONE, "file": "a.h5"}
        record = mq.QueueRecord(self.directory / "record.json", entries, options,
                                {"dual_gate": {"gate_outputs": KEYSIGHT}})
        self.assertTrue(record.finish("finished", "All 2 entries ran."))
        document = json.loads(record.path.read_text(encoding="utf-8"))
        self.assertEqual(document["run"]["outcome"], "finished")
        self.assertEqual(document["run"]["envelope"], {"dual_gate": {"gate_outputs": KEYSIGHT}})
        self.assertEqual(document["entries"][0]["result"], {"status": "done", "file": "a.h5"})
        reopened, _ = mq.load_queue_file(record.path)
        self.assertEqual([e.settings for e in reopened], [e.settings for e in entries])
        self.assertTrue(all(e.status == mq.PENDING for e in reopened))

    def test_files_that_are_not_queues_are_refused_with_the_reason(self):
        header = {"format": mq.QUEUE_FORMAT, "version": 1}
        cases = {"broken": ("{", "not a readable queue file"),
                 "other": (json.dumps({"format": "x", "version": 1, "entries": []}), "not a measurement queue"),
                 "newer": (json.dumps({**header, "version": 9, "entries": []}), "newer version"),
                 "no entries": (json.dumps(header), "no list of entries"),
                 "bad entry": (json.dumps({**header, "entries": [{"tab": "dual_gate"}]}), "Entry 1")}
        for name, (text, message) in cases.items():
            with self.subTest(name):
                path = self.directory / f"{name}.json"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    mq.load_queue_file(path)

    def test_a_record_that_cannot_be_written_is_logged_once_and_written_later(self):
        log = []
        record = mq.QueueRecord(self.directory / "queue" / "record.json",
                                [mq.QueueEntry("dual_gate", dgs_settings())], mq.DEFAULT_OPTIONS, {},
                                log=log.append)
        with patch("measurement_queue.os.replace", side_effect=PermissionError("in use")):
            self.assertFalse(record.write())
            self.assertFalse(record.write())
        self.assertEqual(len(log), 1)
        self.assertIn("could not write the record", log[0])
        self.assertTrue(record.write())
        self.assertIn("up to date again", log[-1])
        self.assertEqual(json.loads(record.path.read_text(encoding="utf-8"))["run"]["outcome"], "running")

    def test_records_are_dated_by_the_queue_start_and_never_replaced(self):
        started = datetime(2026, 11, 25, 23, 59, 30).astimezone()
        first = mq.record_path(self.directory, started)
        self.assertEqual(first.relative_to(self.directory).as_posix(),
                         "data/2026-11-25/queue/2026_11_25_queue_235930.json")
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_text("{}")
        self.assertEqual(mq.record_path(self.directory, started).name, "2026_11_25_queue_235930_2.json")


class DualGateEntryTests(unittest.TestCase):
    def test_settings_begin_accepts_have_no_problems(self):
        self.assertEqual(problems(dgs_settings()), [])

    def test_every_value_must_be_one_its_field_holds_exactly(self):
        cases = {
            "exposure_time": (20.0, "Exposure: 20 s is outside its field's range, 0.001 to 10 s."),
            "c2_start": (0.00125, "Vc2 start: 0.00125 V has more than 4 decimals"),
            "c1_pts": (2.5, "Vc1 points must be a whole number"),
            "c2_pts": (1001, "Vc2 points: 1001 is outside 1 to 1000."),
            "settling_time": ("0.1", "Settling time must be a finite number"),
            "coeff_a": (float("nan"), "Matrix a must be a finite number"),
            "c1_end": (True, "Vc1 end must be a finite number"),
            "serpentine": ("yes", "Serpentine must be true or false."),
            "sweep_mode": ("Hold both", "Variable order must be"),
            "c1_name": (3, "Virtual gate 1 name must be text."),
        }
        for key, (value, message) in cases.items():
            with self.subTest(key):
                found = problems(dgs_settings(**{key: value}))
                self.assertTrue(any(message in problem for problem in found), found)
        missing = dgs_settings()
        missing.pop("settling_time")
        self.assertIn("Missing setting: settling_time.", problems(missing))
        # A typo in a hand-edited file is refused, not replaced by a default.
        self.assertIn("Unknown setting: c1_ned.", problems(dgs_settings(c1_ned=0.02)))
        self.assertEqual(problems("not a dict"), ["The entry has no settings."])

    def test_the_sweep_gets_begins_own_checks_with_the_tabs_outputs_and_windows(self):
        # VG1 = V1 + V2 and VG2 = V1 - V2: V1 = (VG1 + VG2) / 2 reaches +1.5 V at the corner (2, 1).
        matrix = dict(coeff_b=1.0, coeff_c=1.0, coeff_d=-1.0, c1_end=2.0, c2_end=1.0)
        self.assertIn("Channel 1 would reach +1.5 V, above its highest allowed +1 V.",
                      problems(dgs_settings(**matrix)))
        self.assertEqual(problems(dgs_settings(**matrix), gate_limits={1: (-2.0, 2.0), 2: (-1.0, 1.0)}), [])
        self.assertTrue(any("singular" in p for p in problems(dgs_settings(coeff_d=0.0))))
        self.assertTrue(any("+/-210 V" in p for p in problems(dgs_settings(coeff_a=0.5, c1_end=200.0),
                                                                  gate_limits={})))
        dac = {"gate_outputs": {1: "smu:1", 2: "dac:0"}}
        entry = dgs_settings(gate_outputs={"1": "smu:1", "2": "dac:0"}, c2_end=12.0)
        found = problems(entry, gate_limits={1: (-1.0, 1.0), 2: (-20.0, 20.0)}, dac_connected=False, **dac)
        self.assertTrue(any("±10 V range" in p for p in found), found)
        self.assertIn("The uVoltDAQ is not initialized. Connect it in the Device Manager.", found)
        # Wiring cannot change overnight: the entry must name the outputs the tab drives.
        self.assertTrue(any("made with V2 on Keysight CH2, but the tab drives V2 with uVoltDAQ DAC0" in p
                            for p in problems(dgs_settings(), **dac)))
        self.assertIn("Gate sweeps require both DC sources in voltage mode.",
                      problems(dgs_settings(), source_modes=("CURR", "VOLT")))

    def test_targets_follow_the_sweep_workers_order(self):
        # Hold C1, sweep C2, serpentine: every second row runs backwards.
        np.testing.assert_allclose(dgs_queue.sweep_targets(dgs_settings()),
                                   [[0, 0], [0, .005], [0, .01], [.01, .01], [.01, .005], [.01, 0]])
        settings = dgs_settings(sweep_mode="Hold C2, sweep C1", serpentine=False, c1_pts=3, c2_pts=2)
        np.testing.assert_allclose(dgs_queue.sweep_targets(settings),
                                   [[0, 0], [.005, 0], [.01, 0], [0, .01], [.005, .01], [.01, .01]])
        # Physical voltages through the inverse of the matrix: V1 = VG1 / a.
        single = dgs_settings(coeff_a=2.0, c1_start=0.01, c1_pts=1, c2_pts=1)
        np.testing.assert_allclose(dgs_queue.sweep_targets(single), [[0.005, 0.0]])

    def test_the_summary_is_one_line(self):
        self.assertEqual(dgs_queue.entry_summary(dgs_settings()),
                         "Vc1 0 to 0.01 V (2) × Vc2 0 to 0.01 V (3) · hold Vc1, serpentine · "
                         "0.1 s exposure, 0.05 s settling")
        other = dgs_settings(c1_name="n0", c1_pts=1, coeff_b=0.5, sweep_mode="Hold C2, sweep C1",
                             serpentine=False, ramp_to_zero=True)
        self.assertEqual(dgs_queue.entry_summary(other),
                         "n0 0 V × Vc2 0 to 0.01 V (3) · matrix (1, 0.5; 0, 1) · hold Vc2 · "
                         "0.1 s exposure, 0.05 s settling · then 0 V")
        self.assertEqual(dgs_queue.entry_summary({}), "(these settings cannot be read)")

    def test_the_estimate_starts_where_the_previous_run_left_the_gates(self):
        settings = dgs_settings(c1_start=0.5, c1_pts=1, c2_start=-0.2, c2_pts=1)
        seconds, end = dgs_queue.entry_estimate(settings, (0.0, 0.0), (0.1, 0.2), frames=2, overhead=1.0)
        # Ramps 0.5 V at 0.1 V/s and 0.2 V at 0.2 V/s, 2 x 0.1 s exposure, 0.05 s settling, 1 s overhead.
        self.assertAlmostEqual(seconds, 5 + 1 + 0.2 + 0.05 + 1)
        self.assertEqual(end, (0.5, -0.2))
        ramped, at_zero = dgs_queue.entry_estimate(settings, (0.0, 0.0), (0.1, 0.2), frames=2, overhead=1.0,
                                                   end_ramp=True)
        self.assertAlmostEqual(ramped - seconds, 0.5 / 0.1 + 0.2 / 0.2)
        self.assertEqual(at_zero, (0.0, 0.0))


class QueueDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        directory = self.context.enter_context(tempfile.TemporaryDirectory())
        self.directory = Path(directory)
        (self.directory / "value_storage").write_text(json.dumps({
            "shared": {"exposure_time": 0.01, "settling_time": 0.0},
            "single_spectra": {"num_spectra": 2},
        }))
        for name in ("single_spectra_gui", "Dual_gate_sweep_gui", "B_V_sweep_gui",
                     "photoluminescence_map_gui", "scalar_map_gui", "polarization_control_gui", "position_dgs_gui",
                     "gate_map_gui", "capacitance_gui", "transport_gui"):
            module = importlib.import_module("measurements." + name)
            self.context.enter_context(patch.object(module, "parent_dir", directory))
        self.context.enter_context(patch.object(PositionTab, "save_storage"))
        self.context.enter_context(patch("scanner_safety.wait_for_motion",
                                         side_effect=lambda seconds, keep_running: keep_running()))
        # Any unexpected physical network/VISA access makes the test fail.
        self.context.enter_context(patch("pyvisa.ResourceManager",
                                         side_effect=AssertionError("Hardware access in simulation test")))
        self.context.enter_context(patch("socket.create_connection",
                                         side_effect=AssertionError("Network access in queue test")))
        self.context.enter_context(patch("Pyro4.Proxy", side_effect=AssertionError("Remote access in simulation test")))
        self.context.enter_context(patch.dict(os.environ, {"G15_LEVELMETER_DIR": str(self.directory / "levelmeter")}))
        self.errors = self.context.enter_context(patch.object(QtWidgets.QMessageBox, "critical"))
        self.infos = self.context.enter_context(patch.object(QtWidgets.QMessageBox, "information"))
        self.warnings = self.context.enter_context(patch.object(QtWidgets.QMessageBox, "warning"))
        self.questions = self.context.enter_context(patch.object(
            QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.StandardButton.Yes))
        self.station = Station()
        self.dashboard = LabMainWindow(self.station)
        self._closed = False
        self.addCleanup(self.close_dashboard)
        self.app.processEvents()

    def close_dashboard(self):
        if self._closed:
            return
        self._closed = True
        self.dashboard.close()
        self.app.processEvents()
        for component in list(self.station.components.values()):
            component.close()

    def wait_for(self, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(20)
        self.assertTrue(predicate())

    @property
    def window(self):
        return self.dashboard.dual_window

    @property
    def panel(self):
        return self.dashboard.queue_panel

    def set_sweep(self, c1, c2, settling=0.0, ramp_to_zero=False):
        """(start, end, points) per virtual gate, set in the Dual-Gate tab's fields."""
        self.dashboard.tab_widget.setCurrentIndex(1)
        for axis, (start, end, points) in ((1, c1), (2, c2)):
            getattr(self.window, f"c{axis}_start").setValue(start)
            getattr(self.window, f"c{axis}_end").setValue(end)
            getattr(self.window, f"c{axis}_pts").setValue(points)
        self.window.exposure_box.setValue(0.01)
        self.window.settling_box.setValue(settling)
        self.window.ramp_zero_box.setChecked(ramp_to_zero)

    def add(self):
        count = len(self.panel.entries)
        self.dashboard.queue_add_button.click()
        self.assertEqual(len(self.panel.entries), count + 1, self.warnings.call_args)
        return self.panel.entries[-1]

    def run_and_wait(self, timeout=60):
        self.assertTrue(self.panel.run_queue(), self.errors.call_args)
        self.wait_for(lambda: not self.panel.runner.running(), timeout)
        QtTest.QTest.qWait(50)  # The queue's last message is shown after the run's own handlers.

    def record(self):
        return json.loads(self.panel.runner.record.path.read_text(encoding="utf-8"))

    def gates(self):
        return (self.station.smu.ch1.volt(), self.station.smu.ch2.volt())

    def test_three_ranges_run_back_to_back_with_one_file_each_and_a_record(self):
        ranges = [((0.0, 0.01, 2), (0.0, 0.01, 2)),
                  ((-0.02, 0.02, 3), (0.0, 0.01, 2)),
                  ((0.0, 0.01, 2), (0.01, 0.03, 3))]
        for c1, c2 in ranges:
            self.set_sweep(c1, c2)
            self.add()
        self.assertEqual(self.panel.entries[1].settings["c1_start"], -0.02)
        self.assertEqual(self.panel.table.rowCount(), 3)
        self.run_and_wait()
        # The confirmation named the entries, the time and what happens after the last one.
        self.questions.assert_called_once()
        question = self.questions.call_args.args[2]
        self.assertIn("Run 3 entries (1, 2, 3) on the Dual-Gate Sweep tab, about", question)
        self.assertIn("Both gates ramp to 0 V after the last entry.", question)
        # No dialog waited during the runs: only the queue's own message at the end.
        self.errors.assert_not_called()
        self.assertEqual([call.args[1] for call in self.infos.call_args_list], ["Queue finished"])
        record = self.record()
        self.assertEqual(record["run"]["outcome"], "finished")
        self.assertEqual(record["run"]["message"], "All 3 entries ran.")
        envelope = record["run"]["envelope"]["dual_gate"]
        self.assertEqual(envelope["gate_limits_V"], {"1": [-1.0, 1.0], "2": [-1.0, 1.0]})
        self.assertEqual(envelope["gate_outputs"], KEYSIGHT)
        self.assertEqual([entry["result"]["status"] for entry in record["entries"]], ["done"] * 3)
        self.assertEqual(len(list(self.directory.rglob("*autosave_gate_sweep_*.h5"))), 3)
        self.assertEqual(len(list(self.directory.glob("data/*/*DGS_*.h5"))), 3)
        for number, (entry, (c1, c2)) in enumerate(zip(record["entries"], ranges), 1):
            result = entry["result"]
            for key in ("file", "copy"):
                with self.subTest(entry=number, file=key), h5py.File(result[key]) as data:
                    np.testing.assert_allclose(data["c1_coords"][()], np.linspace(*c1))
                    np.testing.assert_allclose(data["c2_coords"][()], np.linspace(*c2))
                    self.assertTrue(np.all(data["measured_mask"][()]))
                    # The record names the frames per point every run used (Single Spectra's setting).
                    self.assertEqual(data["raw_intensities"].shape[2], envelope["frames_per_point"])
                    self.assertEqual(int(data.attrs["queue_entry"]), number)
                    self.assertEqual(int(data.attrs["queue_entries"]), 3)
                    self.assertEqual(data.attrs["queue_record"], str(self.panel.runner.record.path))
            self.assertRegex(Path(result["copy"]).name, r"^\d{4}_\d\d_\d\d_DGS_RC_.+_\d{6}\.h5$")
            self.assertIn("points in", result["timing"])
        # Entries 1 and 2 left the gates at their last point; the queue ramped to 0 V after entry 3.
        for entry in record["entries"][:2]:
            np.testing.assert_allclose(entry["result"]["gates_after_V"],
                                       dgs_queue.sweep_targets(entry["settings"])[-1])
            self.assertNotIn("end_ramp_by_queue", entry["result"])
        self.assertEqual(record["entries"][2]["result"]["gates_after_V"], [0.0, 0.0])
        self.assertTrue(record["entries"][2]["result"]["end_ramp_by_queue"])
        self.assertEqual(self.gates(), (0.0, 0.0))
        # The tab shows the last entry and is free again.
        self.assertEqual(self.window.c2_end.value(), 0.03)
        self.assertTrue(self.window.c1_start.isEnabled())
        self.assertTrue(self.window.begin_btn.isEnabled())
        self.assertFalse(self.dashboard.measurement_active())
        self.assertTrue(self.panel.state_text(self.panel.entries[0]).startswith("done · "))
        # Another point count in the fields while the maps hold the last run: no slice is cut
        # (this raised in every queued entry with a new grid, and after such an edit by hand).
        self.window.c1_pts.setValue(3)
        self.window.on_sweep_range_changed()
        self.window.on_tab2_crosshair_moved()
        self.window.on_tab3_crosshair_moved()

    def test_a_typed_value_counts_without_enter(self):
        self.set_sweep((0.0, 0.01, 2), (0.0, 0.01, 2))
        box = self.window.c1_end
        box.setFocus()
        box.selectAll()
        QtTest.QTest.keyClicks(box, "0.03")
        self.assertEqual(box.value(), 0.01)  # Not read yet: keyboard tracking is off.
        entry = self.add()
        self.assertEqual(entry.settings["c1_end"], 0.03)
        self.assertEqual(box.value(), 0.03)

    def test_an_entry_begin_would_refuse_keeps_the_whole_queue_from_starting(self):
        self.set_sweep((0.0, 0.01, 2), (0.0, 0.01, 2))
        self.add()
        self.set_sweep((0.0, 0.5, 2), (0.0, 0.01, 2))
        self.add()
        # Narrower windows afterwards: entry 2 would take channel 1 to +0.5 V.
        for box, value in zip(self.window.gate_limit_boxes[1], (-0.2, 0.2)):
            box.setValue(value)
        self.window.on_gate_limits_changed()
        self.assertFalse(self.panel.run_queue())
        self.errors.assert_called_once()
        text = self.errors.call_args.args[2]
        self.assertIn("Nothing was started or moved", text)
        self.assertIn("Entry 2: Channel 1 would reach +0.5 V, above its highest allowed +0.2 V.", text)
        self.assertNotIn("Entry 1", text)
        self.questions.assert_not_called()
        self.assertFalse(self.panel.runner.running())
        self.assertFalse(self.window.is_running)
        self.assertEqual(list(self.directory.rglob("*.h5")), [])
        self.assertEqual(list(self.directory.rglob("*queue_*.json")), [])
        self.assertEqual(self.gates(), (0.0, 0.0))
        self.assertTrue(self.panel.state_text(self.panel.entries[1]).startswith("needs a change: Channel 1"))
        # A tab that shows a saved file does not run the queue either.
        self.window.plotter_mode = True
        try:
            self.assertFalse(self.panel.run_queue())
        finally:
            self.window.plotter_mode = False
        self.assertEqual(self.warnings.call_args.args[1:], ("Queue not started", "The Dual-Gate tab shows a saved file."))
        self.assertFalse(self.panel.runner.running())

    def test_an_error_stops_the_queue_keeps_the_data_and_holds_the_gates(self):
        import measurements.Dual_gate_sweep_gui as dgs
        for c1 in ((0.0, 0.01, 2), (0.02, 0.03, 2), (0.0, 0.01, 2)):
            self.set_sweep(c1, (0.0, 0.01, 2))
            self.add()
        acquire, calls = dgs.acquire_sweep_spectrum, []

        def failing_camera(*args, **kwargs):
            calls.append(1)
            if len(calls) == 6:  # Entry 1 has 4 points: entry 2 fails at its second point.
                raise RuntimeError("camera timed out")
            return acquire(*args, **kwargs)

        with patch.object(dgs, "acquire_sweep_spectrum", side_effect=failing_camera):
            self.run_and_wait()
        self.assertEqual([entry.status for entry in self.panel.entries], [mq.DONE, mq.FAILED, mq.NOT_RUN])
        record = self.record()
        self.assertEqual(record["run"]["outcome"], "failed")
        self.assertEqual(record["run"]["message"], "Entry 2 failed: camera timed out")
        failed = record["entries"][1]["result"]
        self.assertEqual(failed["message"], "camera timed out")
        for key in ("file", "copy"):
            self.assertTrue(failed[key].endswith("_interrupted.h5"), failed[key])
            with h5py.File(failed[key]) as data:
                self.assertEqual(int(data["measured_mask"][()].sum()), 1)
        # Nothing moved the gates after the error: they hold entry 2's second point.
        np.testing.assert_allclose(failed["gates_after_V"], [0.02, 0.01])
        np.testing.assert_allclose(self.gates(), (0.02, 0.01))
        self.assertEqual(record["entries"][2]["result"], {"status": "not run"})
        # The run's own Sweep Error box did not wait for a click; the queue's message came at the end.
        self.assertEqual([call.args[1] for call in self.errors.call_args_list], ["Queue stopped"])
        # Run queue again: the entries not done run, entry 1 is not repeated.
        self.questions.reset_mock()
        self.run_and_wait()
        self.assertIn("Run 2 entries (2, 3)", self.questions.call_args.args[2])
        self.assertEqual([entry.status for entry in self.panel.entries], [mq.DONE] * 3)
        self.assertEqual([entry["result"]["status"] for entry in self.record()["entries"]], ["done", "done"])

    def test_an_unexpected_exception_ends_the_queue_instead_of_holding_the_dashboard(self):
        self.set_sweep((0.0, 0.01, 2), (0.0, 0.01, 2))
        self.add()
        adapter = self.panel.adapters["dual_gate"]
        with patch.object(adapter, "show", side_effect=RuntimeError("broken field")):
            self.run_and_wait()
        result = self.panel.entries[0].result
        self.assertEqual(result["status"], mq.FAILED)
        self.assertEqual(result["message"], "Begin did not start the run: RuntimeError: broken field")
        self.assertFalse(self.dashboard.measurement_active())
        self.assertTrue(self.window.c1_start.isEnabled())
        self.assertEqual(self.gates(), (0.0, 0.0))
        # A named copy that cannot be saved does not hold the queue up: the autosave has the data.
        with patch.object(adapter, "save_named_copy", side_effect=OSError("disk full")):
            self.run_and_wait()
        result = self.panel.entries[0].result
        self.assertEqual(result["status"], mq.DONE)
        self.assertNotIn("copy", result)
        self.assertTrue(Path(result["file"]).exists())

    def test_stop_after_this_run_lets_the_run_finish_then_stops(self):
        for c1 in ((0.0, 0.01, 3), (0.0, 0.02, 2)):
            self.set_sweep(c1, (0.0, 0.01, 3), settling=0.05)
            self.add()
        self.assertTrue(self.panel.run_queue())
        self.wait_for(lambda: self.window.is_running)
        self.panel.stop_after_btn.click()
        self.assertIn("stopping after this run", self.panel.status_lbl.text())
        self.wait_for(lambda: not self.panel.runner.running(), 60)
        self.assertEqual([entry.status for entry in self.panel.entries], [mq.DONE, mq.NOT_RUN])
        record = self.record()
        self.assertEqual((record["run"]["outcome"], record["run"]["message"]),
                         ("stopped", "Stopped after entry 1, as asked."))
        self.errors.assert_not_called()

    def test_reset_on_the_tab_stops_the_run_and_the_queue(self):
        for c1 in ((0.0, 0.01, 4), (0.0, 0.02, 2)):
            self.set_sweep(c1, (0.0, 0.01, 4), settling=0.05)
            self.add()
        self.assertTrue(self.panel.run_queue())
        self.wait_for(lambda: self.window.is_running and self.window.progress_bar.value() > 0)
        self.window.reset_btn.click()
        self.wait_for(lambda: not self.panel.runner.running())
        self.assertEqual([entry.status for entry in self.panel.entries], [mq.STOPPED, mq.NOT_RUN])
        result = self.panel.entries[0].result
        self.assertEqual(result["message"], "Stopped with Reset.")
        self.assertTrue(result["file"].endswith("_interrupted.h5"))
        self.assertEqual(self.record()["run"]["outcome"], "stopped")
        QtTest.QTest.qWait(50)
        self.errors.assert_not_called()
        self.infos.assert_not_called()

    def test_the_dashboard_stays_locked_between_runs(self):
        for c1 in ((0.0, 0.01, 2), (0.0, 0.02, 2)):
            self.set_sweep(c1, (0.0, 0.01, 2))
            self.add()
        seen = []

        def between_runs():
            runner = self.panel.runner
            if runner.state == "waiting" and runner.entries and runner.entries[0].status == mq.DONE:
                self.dashboard.check_measurement_status()
                seen.append((self.dashboard.measurement_active(), self.dashboard.prepare_device_change(),
                             self.dashboard.tab_widget.isTabEnabled(0), self.window.c1_start.isEnabled(),
                             self.window.begin_btn.isEnabled(), self.dashboard.queue_add_button.isEnabled()))

        self.panel.runner.changed.connect(between_runs)
        self.run_and_wait()
        self.assertTrue(seen)
        self.assertEqual(seen[0], (True, False, False, False, False, False))
        self.assertEqual([entry.status for entry in self.panel.entries], [mq.DONE, mq.DONE])

    def test_show_on_tab_and_replace_edit_an_entry_without_instrument_commands(self):
        self.set_sweep((0.0, 0.01, 2), (0.0, 0.01, 2))
        first = self.add()
        self.set_sweep((-0.1, 0.1, 5), (0.0, 0.02, 3), settling=0.2)
        self.add()
        before = json.loads((self.directory / "value_storage").read_text())["dual_gate"]
        rates = (self.window.ch1_ramp_rate.value(), self.window.ch2_ramp_rate.value())
        self.panel.select(0)
        self.assertTrue(self.panel.show_selected())
        self.assertEqual((self.window.c1_start.value(), self.window.c1_end.value(), self.window.c1_pts.value()),
                         (0.0, 0.01, 2))
        self.assertEqual(self.window.settling_box.value(), 0.0)
        stored = json.loads((self.directory / "value_storage").read_text())
        self.assertEqual((stored["dual_gate"]["c1_pts"], stored["shared"]["settling_time"]), (2, 0.0))
        for key in ("ramprate1", "ramprate2", "v1_min", "v1_max", "v2_min", "v2_max"):
            self.assertEqual(stored["dual_gate"].get(key), before.get(key), key)
        self.assertEqual((self.window.ch1_ramp_rate.value(), self.window.ch2_ramp_rate.value()), rates)
        self.assertEqual(self.gates(), (0.0, 0.0))
        self.window.c1_end.setValue(0.05)
        self.assertTrue(self.panel.replace_selected())
        self.assertIs(self.panel.entries[0], first)
        self.assertEqual(first.settings["c1_end"], 0.05)
        self.panel.select(1)
        self.panel.move_selected(-1)
        self.assertIs(self.panel.entries[1], first)
        self.panel.remove_selected()
        self.assertEqual(self.panel.entries, [first])
        self.warnings.assert_not_called()

    def test_a_saved_queue_opens_again_and_bad_values_are_marked(self):
        self.set_sweep((0.0, 0.01, 2), (0.0, 0.01, 2))
        self.add()
        self.panel.entries[0].note = "near the CNP"
        path = self.directory / "my_queue.json"
        self.assertEqual(self.panel.save_queue_file(str(path)), str(path))
        document = json.loads(path.read_text(encoding="utf-8"))
        document["entries"].append(json.loads(json.dumps(document["entries"][0])))
        document["entries"][1]["settings"]["exposure_time"] = 20
        path.write_text(json.dumps(document), encoding="utf-8")
        self.panel.open_queue_file(str(path), replace=True)
        self.assertEqual(len(self.panel.entries), 2)
        self.assertEqual(self.panel.entries[0].note, "near the CNP")
        self.assertEqual(self.panel.state_text(self.panel.entries[0]), "pending")
        self.assertIn("Exposure: 20 s is outside", self.panel.state_text(self.panel.entries[1]))
        self.assertIn("Entry 2 needs a change", self.panel.status_lbl.text())
        self.assertFalse(self.panel.run_queue())
        self.assertIn("Entry 2: Exposure: 20 s is outside", self.errors.call_args.args[2])

    def test_the_entry_fields_match_the_tab_and_position_dgs_has_no_queue_button(self):
        for key, (lowest, highest, decimals) in dgs_queue.NUMBER_FIELDS.items():
            box = getattr(self.window, dgs_queue.FIELD_WIDGETS[key])
            self.assertEqual((box.minimum(), box.maximum(), box.decimals()), (lowest, highest, decimals), key)
        for key, (lowest, highest) in dgs_queue.POINT_FIELDS.items():
            box = getattr(self.window, dgs_queue.FIELD_WIDGETS[key])
            self.assertEqual((box.minimum(), box.maximum()), (lowest, highest), key)
        combo = self.window.sweep_mode_combo
        self.assertEqual(tuple(combo.itemText(index) for index in range(combo.count())), dgs_queue.SWEEP_MODES)
        layout = self.window.begin_btn.parentWidget().layout()
        self.assertEqual(layout.indexOf(self.dashboard.queue_add_button), layout.indexOf(self.window.begin_btn) + 1)
        psdgs = self.dashboard.position_dgs_window
        texts = [button.text() for button in psdgs.begin_btn.parentWidget().findChildren(QtWidgets.QPushButton)]
        self.assertNotIn("+ Add to queue", texts)

    def test_closing_the_dashboard_during_a_queued_run_keeps_its_data_and_record(self):
        for c1 in ((0.0, 0.01, 4), (0.0, 0.02, 2)):
            self.set_sweep(c1, (0.0, 0.01, 4), settling=0.05)
            self.add()
        self.assertTrue(self.panel.run_queue())
        self.wait_for(lambda: self.window.is_running and self.window.progress_bar.value() > 0)
        path = self.panel.runner.record.path
        self.close_dashboard()
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((record["run"]["outcome"], record["run"]["message"]),
                         ("stopped", "The dashboard was closed."))
        first, second = (entry["result"] for entry in record["entries"])
        self.assertEqual((first["status"], second["status"]), ("stopped", "not run"))
        self.assertTrue(first["file"].endswith("_interrupted.h5"))
        with h5py.File(first["file"]) as data:
            self.assertGreater(int(data["measured_mask"][()].sum()), 0)


if __name__ == "__main__":
    unittest.main()
