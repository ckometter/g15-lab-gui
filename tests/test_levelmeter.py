"""Helium level on the dashboard: reading the level meter's file, states and the panel.

Synthetic files in temporary folders only; the real level-meter folder is never read.
"""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import levelmeter_monitor as lm

NETCDF_FILL = 9.969209968386869e36


def write_level_file(folder, times, levels, fill_entries=0):
    """The level meter's layout: group main_data with timestamps, level (mm) and rate."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    times = list(times) + [NETCDF_FILL] * fill_entries
    levels = list(levels) + [NETCDF_FILL] * fill_entries
    with h5py.File(folder / lm.DATA_NAME, "w") as f:
        group = f.create_group("main_data")
        group.create_dataset("timestamps", data=np.array(times, dtype=float), maxshape=(None,))
        group.create_dataset("level", data=np.array(levels, dtype=float), maxshape=(None,)).attrs["units"] = "mm"
        group.create_dataset("rate", data=np.zeros(len(times)), maxshape=(None,))
    return folder / lm.DATA_NAME


def falling(now, hours, start_mm, rate_mm_per_h, step_min=10):
    times = now - np.arange(int(hours * 60 / step_min), -1, -1) * step_min * 60.0
    return times, start_mm + rate_mm_per_h * (times - times[0]) / 3600.0


class LevelmeterFileTests(unittest.TestCase):
    def test_reads_a_copy_drops_fill_values_and_sorts(self):
        with tempfile.TemporaryDirectory() as folder:
            now = time.time()
            path = write_level_file(folder, [now - 600, now - 1200, now], [500.0, 510.0, 495.0], fill_entries=3)
            times, levels = lm.read_history(path)
        np.testing.assert_allclose(times, [now - 1200, now - 600, now])
        np.testing.assert_allclose(levels, [510.0, 500.0, 495.0])

    def test_trend_is_the_24_hour_linear_slope(self):
        now = time.time()
        times, levels = falling(now, 48, 600.0, -2.5)
        self.assertAlmostEqual(lm.trend(times, levels), -2.5, places=6)
        self.assertTrue(np.isnan(lm.trend(times[-1:], levels[-1:])))


class LevelmeterStateTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time()

    def summary(self, rate=-1.0, start=600.0, age_min=2.0, **options):
        times, levels = falling(self.now - age_min * 60, 30, start, rate)
        return lm.summarize(times, levels, now=self.now, folder="C:/levelmeter", **options)

    def test_normal_reading_shows_level_trend_and_age(self):
        summary = self.summary()
        self.assertEqual(summary.severity, lm.NORMAL)
        self.assertRegex(summary.text, r"^LHe 570 mm · -1\.0 mm/h · 2 min ago$")
        self.assertIn("Levelmeter-Stop.bat", summary.detail)

    def test_falling_toward_the_threshold_within_a_day_needs_attention(self):
        summary = self.summary(rate=-12.0, start=600.0)
        self.assertEqual(summary.severity, lm.ATTENTION)
        self.assertIn("230 mm in ~", summary.text)
        self.assertAlmostEqual(summary.hours_to_low, (summary.latest_level - 230.0) / 12.0, places=3)

    def test_below_the_threshold_is_low(self):
        summary = self.summary(rate=-1.0, start=250.0)
        self.assertEqual(summary.severity, lm.ATTENTION)
        self.assertIn("LOW", summary.text)

    def test_no_new_reading_for_30_minutes_is_a_problem(self):
        summary = self.summary(age_min=45)
        self.assertEqual(summary.severity, lm.PROBLEM)
        self.assertIn("NO NEW READING for 45 min", summary.text)

    def test_about_zero_mm_means_meter_off_or_empty(self):
        times = self.now - np.arange(5)[::-1] * 600.0
        summary = lm.summarize(times, np.full(5, -1.0), now=self.now)
        self.assertEqual(summary.severity, lm.PROBLEM)
        self.assertIn("meter off or dewar empty", summary.text)

    def test_switched_off_on_purpose_is_grey_even_when_stale(self):
        summary = self.summary(age_min=300, stopped_since=self.now - 3600 * 5)
        self.assertEqual(summary.severity, lm.INACTIVE)
        self.assertIn("level meter OFF since", summary.text)

    def test_no_data_is_shown_without_an_alarm(self):
        summary = lm.summarize([], [], now=self.now, error="FileNotFoundError: no file")
        self.assertEqual(summary.severity, lm.INACTIVE)
        self.assertEqual(summary.text, "LHe: no level-meter data")
        self.assertIn("FileNotFoundError", summary.detail)

    def test_history_keeps_14_days(self):
        times, levels = falling(self.now, 20 * 24, 700.0, -0.5)
        summary = lm.summarize(times, levels, now=self.now)
        self.assertGreaterEqual(summary.times[0], self.now - 14 * 86400 - 1)


try:
    import test_dashboard_workflow as dashboard_checks
except Exception:  # pragma: no cover - GUI dependencies missing
    dashboard_checks = None


@unittest.skipIf(dashboard_checks is None, "Qt dashboard dependencies are unavailable")
class LevelmeterDashboardTests(unittest.TestCase):
    if dashboard_checks is not None:
        setUpClass = classmethod(dashboard_checks.DashboardWorkflowTests.setUpClass.__func__)
        setUp = dashboard_checks.DashboardWorkflowTests.setUp
        close_dashboard = dashboard_checks.DashboardWorkflowTests.close_dashboard
        wait_for = dashboard_checks.DashboardWorkflowTests.wait_for

    def test_worker_reads_the_folder_and_follows_stop_and_new_readings(self):
        from measurements.levelmeter_panel import LevelmeterWorker
        folder = self.directory / "levelmeter_worker"
        now = time.time()
        write_level_file(folder, *falling(now - 60, 30, 600.0, -1.0))
        received = []
        worker = LevelmeterWorker(folder=folder, poll_s=0.1)
        worker.updated.connect(received.append)
        worker.start()
        self.addCleanup(lambda: (worker.stop(), worker.wait(3000)))
        self.wait_for(lambda: received and received[-1].severity == lm.NORMAL)
        (folder / lm.STOP_NAME).write_text("stopped", encoding="utf-8")
        self.wait_for(lambda: received[-1].severity == lm.INACTIVE)
        (folder / lm.STOP_NAME).unlink()
        write_level_file(folder, *falling(now, 30, 200.0, -1.0))  # a new file: re-read
        self.wait_for(lambda: "LOW" in received[-1].text)

    def test_status_bar_and_dashboard_panel_show_the_level(self):
        from measurements.levelmeter_panel import with_gaps
        # The fixture points the dashboard at a missing folder: grey, no alarm.
        self.wait_for(lambda: "no level-meter data" in self.dashboard.levelmeter_label.text())
        now = time.time()
        times, levels = falling(now - 120, 30, 600.0, -1.0)
        summary = lm.summarize(np.r_[times - 5 * 86400, times], np.r_[levels + 100, levels], now=now)
        self.dashboard.on_levelmeter_update(summary)
        self.assertTrue(self.dashboard.levelmeter_label.text().startswith("LHe 570 mm"))
        self.assertIn("Click to show", self.dashboard.levelmeter_label.toolTip())
        panel = self.dashboard.levelmeter_panel
        x, y = panel.history_curve.getData()
        self.assertEqual(np.count_nonzero(np.isnan(y)), 1)  # the 5-day outage is drawn as a break
        self.assertEqual(np.count_nonzero(np.isfinite(y)), summary.times.size)
        self.assertGreater(panel.fit_curve.getData()[0].size, 2)
        self.assertIn("-1.00 mm/h", panel.trend_plot.plotItem.titleLabel.text)
        np.testing.assert_array_equal(with_gaps([0.0, 600.0], [1.0, 2.0])[1], [1.0, 2.0])
        stale = lm.summarize(times, levels, now=now + 3600)
        self.dashboard.on_levelmeter_update(stale)
        self.assertEqual(self.dashboard.levelmeter_label.severity, lm.PROBLEM)
        self.assertIn("background-color", self.dashboard.levelmeter_label.styleSheet())

    def test_clicking_the_status_shows_the_dashboard_tab(self):
        self.dashboard.tab_widget.setCurrentIndex(0)
        self.dashboard.levelmeter_label.clicked.emit()
        self.assertEqual(self.dashboard.tab_widget.currentIndex(), 5)
        self.assertIn(self.dashboard.levelmeter_panel, self.dashboard.position_tab.findChildren(type(self.dashboard.levelmeter_panel)))

    def test_closing_the_dashboard_stops_the_reader(self):
        worker = self.dashboard.levelmeter_worker
        self.assertTrue(worker.isRunning())
        self.close_dashboard()
        self.assertTrue(worker.wait(3000))
        self.assertFalse(worker.isRunning())
        self.station = dashboard_checks.Station()
        self.dashboard = dashboard_checks.LabMainWindow(self.station)


if __name__ == "__main__":
    unittest.main()
