"""Shared locations for measurement files saved by the lab GUI."""

from datetime import datetime
from pathlib import Path


def measurement_device_modes(station):
    """Record which data sources were simulated, without querying hardware."""
    return {name: "simulated" if getattr(device, "simulated", False) else "physical"
            for name, device in station.components.items()}


def new_measurement_start(timestamp=None):
    """Capture the local start time, including its UTC offset."""
    if timestamp is None:
        return datetime.now().astimezone()
    return datetime.fromtimestamp(timestamp).astimezone()


def measurement_data_dir(project_dir, started_at=None, *, autosave=False):
    """Return the dated directory for a run, creating it when first saved.

    The date stays tied to the measurement start even if the save happens
    after midnight. A cooldown directory can later be inserted above it.
    """
    if started_at is None:
        started_at = new_measurement_start()
    directory = Path(project_dir) / "data" / started_at.date().isoformat()
    if autosave:
        directory /= "autosave"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def measurement_start_text(started_at):
    """Format the run start for metadata without changing the save timestamp."""
    if started_at is None:
        started_at = new_measurement_start()
    return started_at.isoformat(timespec="seconds")


def dated_measurement_filename(filename, started_at=None):
    """Prefix a suggested name with the run's start date, exactly once."""
    import re
    started_at = started_at or new_measurement_start()
    filename = re.sub(r'^\d{4}_\d{2}_\d{2}_', '', str(filename))
    return started_at.strftime('%Y_%m_%d_') + filename


def dated_measurement_path(filename, started_at=None):
    path = Path(filename)
    return str(path.with_name(dated_measurement_filename(path.name, started_at)))
