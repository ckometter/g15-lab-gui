"""Read-only view of the cryostat's helium level meter for the dashboard.

The level meter is a separate program (Desktop\\Python_Script\\Levelmeter,
started with Levelmeter.bat). It reads the NI-DAQ every 10 minutes into
LevelData.nc and keeps running when the dashboard is closed. The dashboard
never talks to the instrument: it copies that file and reads the copy, so it
never locks the file the level meter appends to, and summarizes the newest
reading, the 24-hour trend and whether the level meter is still working.
LEVELMETER_STOPPED.txt in that folder means it was switched off on purpose
(Levelmeter-Stop.bat). No Qt here, so it is tested offline.
"""
from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

DEFAULT_LEVELMETER_DIR = Path(r"C:\Users\QPG-User\Desktop\Python_Script\Levelmeter")
LEVELMETER_DIR_ENV = "G15_LEVELMETER_DIR"
DATA_NAME = "LevelData.nc"
STOP_NAME = "LEVELMETER_STOPPED.txt"
LOW_LEVEL_MM = 230.0       # the level-meter script's low-level threshold
STALE_AFTER_S = 30 * 60    # readings come every 10 min; none for 30 min means it is not working
SLOPE_WINDOW_H = 24.0      # trend window, as in the level-meter script
HISTORY_DAYS = 14.0        # plotted history, as in the level-meter script
OFF_BELOW_MM = 5.0         # about 0 V (-1 mm): the level meter is off, or the dewar is empty
WARN_WITHIN_H = 24.0       # attention when the trend reaches the threshold within a day
FILL_LIMIT = 1e30          # NetCDF fill values (9.97e36) mark unwritten entries

# Severity of a summary: shown grey, normal, orange or red.
INACTIVE, NORMAL, ATTENTION, PROBLEM = -1, 0, 1, 2


def levelmeter_dir():
    """The level-meter folder; the G15_LEVELMETER_DIR environment variable overrides the default."""
    return Path(os.environ.get(LEVELMETER_DIR_ENV) or DEFAULT_LEVELMETER_DIR)


def read_history(data_file):
    """(unix times, levels in mm) sorted by time, from a copy of the level meter's NetCDF file.

    NetCDF4 files are HDF5 files, so h5py reads them (the netCDF4 package is not needed).
    """
    with tempfile.TemporaryDirectory(prefix="g15_levelmeter_") as folder:
        copy = Path(folder) / DATA_NAME
        shutil.copyfile(data_file, copy)
        with h5py.File(copy, "r") as f:
            group = f["main_data"]
            times = np.asarray(group["timestamps"][()], dtype=float)
            levels = np.asarray(group["level"][()], dtype=float)
    count = min(times.size, levels.size)
    times, levels = times[:count], levels[:count]
    valid = (np.isfinite(times) & np.isfinite(levels) & (times > 0)
             & (np.abs(times) < FILL_LIMIT) & (np.abs(levels) < FILL_LIMIT))
    order = np.argsort(times[valid])
    return times[valid][order], levels[valid][order]


def trend(times, levels, latest=None):
    """Linear slope (mm/h) over the SLOPE_WINDOW_H hours before ``latest``; NaN without two points."""
    times, levels = np.asarray(times, dtype=float), np.asarray(levels, dtype=float)
    if not times.size:
        return math.nan
    latest = float(times[-1]) if latest is None else float(latest)
    recent = times >= latest - SLOPE_WINDOW_H * 3600
    if np.count_nonzero(recent) < 2 or np.ptp(times[recent]) <= 0:
        return math.nan
    slope, _intercept = np.polyfit((times[recent] - latest) / 3600.0, levels[recent], 1)
    return float(slope)


def describe_age(seconds):
    minutes = max(0.0, seconds) / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} h"
    return f"{minutes / 1440:.1f} days"


def describe_hours(hours):
    return f"{hours:.0f} h" if hours < 48 else f"{hours / 24:.1f} days"


@dataclass
class LevelSummary:
    severity: int
    text: str                       # one line for the status bar
    detail: str                     # tooltip and panel text
    folder: str = ""
    latest_time: float = math.nan   # unix time of the newest reading
    latest_level: float = math.nan  # mm
    slope: float = math.nan         # mm/h over the last 24 h
    hours_to_low: float = math.nan  # until LOW_LEVEL_MM at the present slope
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    levels: np.ndarray = field(default_factory=lambda: np.zeros(0))


def summarize(times, levels, *, now, stopped_since=None, error=None, folder=""):
    """What the dashboard shows for this history at time ``now`` (unix seconds)."""
    times, levels = np.asarray(times, dtype=float), np.asarray(levels, dtype=float)
    controls = f"Start or stop it with Levelmeter.bat or Levelmeter-Stop.bat in {folder}." if folder else ""
    if not times.size:
        detail = "No level-meter data could be read" + (f": {error}" if error else ".")
        return LevelSummary(INACTIVE, "LHe: no level-meter data", f"{detail}\n{controls}".strip(), str(folder))
    latest, level = float(times[-1]), float(levels[-1])
    slope = trend(times, levels, latest)
    hours_to_low = ((level - LOW_LEVEL_MM) / -slope
                    if math.isfinite(slope) and slope < 0 and level > LOW_LEVEL_MM else math.nan)
    keep = times >= latest - HISTORY_DAYS * 86400
    summary = LevelSummary(NORMAL, "", "", str(folder), latest, level, slope, hours_to_low, times[keep], levels[keep])
    age = now - latest
    when = datetime.fromtimestamp(latest).strftime("%d %b %H:%M")
    rate = f"{slope:+.1f} mm/h" if math.isfinite(slope) else "rate n/a"
    eta = (f"{LOW_LEVEL_MM:.0f} mm in ~{describe_hours(hours_to_low)}" if math.isfinite(hours_to_low) else "")
    reading = f"Last reading {level:.0f} mm at {when} ({describe_age(age)} ago); 24 h trend {rate}"
    reading += f"; {eta} at this rate." if eta else "."
    if error:
        reading += f"\nThe newest file could not be read ({error}); showing the previous copy."
    if stopped_since is not None:
        since = datetime.fromtimestamp(stopped_since).strftime("%d %b %H:%M")
        summary.severity = INACTIVE
        summary.text = f"LHe level meter OFF since {since} · last {level:.0f} mm"
        summary.detail = f"The level meter was switched off on purpose (Levelmeter-Stop.bat).\n{reading}\n{controls}"
    elif age > STALE_AFTER_S:
        summary.severity = PROBLEM
        summary.text = f"LHe {level:.0f} mm · NO NEW READING for {describe_age(age)}"
        summary.detail = (f"The level meter has not recorded for {describe_age(age)}; it may have stopped. "
                          f"Check Levelmeter-Status.bat.\n{reading}\n{controls}")
    elif level < OFF_BELOW_MM:
        summary.severity = PROBLEM
        summary.text = "LHe ~0 mm · meter off or dewar empty?"
        summary.detail = (f"About 0 mm (0 V): the level meter is switched off, or the dewar is empty.\n"
                          f"{reading}\n{controls}")
    elif level < LOW_LEVEL_MM:
        summary.severity = ATTENTION
        summary.text = f"LHe {level:.0f} mm LOW · {rate}"
        summary.detail = f"Below the {LOW_LEVEL_MM:.0f} mm low-level threshold.\n{reading}\n{controls}"
    elif math.isfinite(hours_to_low) and hours_to_low < WARN_WITHIN_H:
        summary.severity = ATTENTION
        summary.text = f"LHe {level:.0f} mm · {rate} · {eta}"
        summary.detail = f"{reading}\n{controls}"
    else:
        summary.text = f"LHe {level:.0f} mm · {rate} · {describe_age(age)} ago"
        summary.detail = f"{reading}\n{controls}"
    return summary
