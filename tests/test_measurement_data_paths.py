"""Offline checks for measurement folders; no laboratory devices are used."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from measurement_data_paths import measurement_data_dir, measurement_start_text, dated_measurement_filename


class MeasurementDataPathsTests(unittest.TestCase):
    def test_filename_uses_start_date_and_does_not_duplicate_prefix(self):
        started = datetime(2026, 9, 26, 23, 59)
        self.assertEqual(dated_measurement_filename('autosave_spectra_000100.h5', started),
                         '2026_09_26_autosave_spectra_000100.h5')
        self.assertEqual(dated_measurement_filename('2026_09_27_custom.h5', started),
                         '2026_09_26_custom.h5')

    def test_run_keeps_its_start_day_for_later_manual_and_automatic_saves(self):
        started_at = datetime(2026, 9, 26, 23, 59, tzinfo=timezone(timedelta(hours=2)))

        with tempfile.TemporaryDirectory() as project_dir:
            manual_dir = Path(measurement_data_dir(project_dir, started_at))
            autosave_dir = Path(
                measurement_data_dir(project_dir, started_at, autosave=True)
            )

            self.assertEqual(manual_dir, Path(project_dir) / "data" / "2026-09-26")
            self.assertEqual(autosave_dir, manual_dir / "autosave")
            self.assertTrue(autosave_dir.is_dir())
            self.assertEqual(
                measurement_start_text(started_at), "2026-09-26T23:59:00+02:00"
            )


if __name__ == "__main__":
    unittest.main()
