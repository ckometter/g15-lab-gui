"""Measurement queue: tab settings run one after another (docs/MEASUREMENT_QUEUE.md).

This module has no Qt: queue entries, the queue file and the run record. A
queue file lists entries. A record is the same document plus what happened
when the queue ran, so a record also opens as a queue. The panel and the
runner are in ``measurements/queue_panel.py``; the Dual-Gate Sweep entries in
``measurements/dgs_queue.py``.
"""
import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

QUEUE_FORMAT = "g15-measurement-queue"
QUEUE_VERSION = 1

# Entry states, kept in entry.result["status"].
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
STOPPED = "stopped"
NOT_RUN = "not run"

# Queue options. After the last entry the gates ramp to 0 V, as that entry's own
# "Ramp gates to 0 V when complete" would (only after a complete sweep).
DEFAULT_OPTIONS = {"ramp_to_zero_after_last": True}


def now_text():
    """The local time with its UTC offset, as the files' measurement_started_at."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class QueueEntry:
    """One queued run: the tab that runs it and the settings put on that tab."""
    tab: str
    settings: dict
    note: str = ""
    # What happened when it ran: status, started_at, ended_at, message, file, copy, ...
    result: dict = field(default_factory=dict)

    @property
    def status(self):
        return self.result.get("status", PENDING)

    def document(self, with_result=False):
        entry = {"tab": self.tab, "note": self.note, "settings": copy.deepcopy(self.settings)}
        if with_result and self.result:
            entry["result"] = copy.deepcopy(self.result)
        return entry


def _json_value(value):
    """numpy numbers and arrays as plain JSON values."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is always installed here
        np = None
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} cannot be written to a queue file.")


def queue_document(entries, options, run=None):
    """The JSON document of a queue file; with ``run``, a record that keeps the results."""
    document = {"format": QUEUE_FORMAT, "version": QUEUE_VERSION, "options": dict(options),
                "entries": [entry.document(with_result=run is not None) for entry in entries]}
    if run is not None:
        document["run"] = copy.deepcopy(run)
    return document


def write_json_atomic(path, document):
    """Write ``document`` next to ``path`` first, then swap it in: no reader sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False, default=_json_value)
    os.replace(temporary, path)


def save_queue_file(path, entries, options):
    write_json_atomic(path, queue_document(entries, options))


def load_queue_file(path):
    """(entries, options) from a queue file or a record. Results are not loaded."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            document = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not a readable queue file: {exc}") from exc
    return parse_queue_document(document, path.name)


def parse_queue_document(document, name="The file"):
    if not isinstance(document, dict) or document.get("format") != QUEUE_FORMAT:
        raise ValueError(f"{name} is not a measurement queue file.")
    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"{name} has no valid format version.")
    if version > QUEUE_VERSION:
        raise ValueError(f"{name} was written by a newer version of the queue (format {version}); "
                         f"this program reads format {QUEUE_VERSION}.")
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"{name} has no list of entries.")
    entries = []
    for number, raw in enumerate(raw_entries, 1):
        if (not isinstance(raw, dict) or not isinstance(raw.get("tab"), str)
                or not isinstance(raw.get("settings"), dict)):
            raise ValueError(f"Entry {number} in {name} needs a tab name and its settings.")
        note = raw.get("note", "")
        entries.append(QueueEntry(raw["tab"], copy.deepcopy(raw["settings"]),
                                  note if isinstance(note, str) else str(note)))
    options = dict(DEFAULT_OPTIONS)
    raw_options = document.get("options")
    if isinstance(raw_options, dict):
        for key, default in DEFAULT_OPTIONS.items():
            if type(raw_options.get(key)) is type(default):
                options[key] = raw_options[key]
    return entries, options


def unique_path(path):
    """``path``, or the same name with _2, _3, ... before the extension if it exists."""
    path = Path(path)
    candidate, number = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        number += 1
    return candidate


def record_path(data_root, started_at):
    """data/<date>/queue/<yyyy_mm_dd>_queue_<HHMMSS>.json, dated by the queue's start."""
    from measurement_data_paths import measurement_data_dir
    directory = Path(measurement_data_dir(data_root, started_at)) / "queue"
    return unique_path(directory / started_at.strftime("%Y_%m_%d_queue_%H%M%S.json"))


class QueueRecord:
    """The record of one queue run: written at the start and after every change.

    It holds the scheduled entries with their results, the options, and the run:
    start, end, outcome and the tab settings the queue checked the entries
    against. A failed write (for example while a program holds the file open) is
    logged once and retried at the next change; it never stops a run.
    """

    def __init__(self, path, entries, options, envelope, log=print):
        self.path = Path(path)
        self.entries = list(entries)
        self.options = dict(options)
        self.run = {"started_at": now_text(), "ended_at": None, "outcome": "running",
                    "message": "", "envelope": copy.deepcopy(envelope)}
        self.log = log
        self.write_failed = False

    def write(self):
        try:
            write_json_atomic(self.path, queue_document(self.entries, self.options, self.run))
        except (OSError, TypeError, ValueError) as exc:
            if not self.write_failed:
                self.log(f"Measurement queue: could not write the record {self.path} ({exc}). "
                         "Retrying at the next change; the queue goes on.")
            self.write_failed = True
            return False
        if self.write_failed:
            self.log(f"Measurement queue: the record {self.path} is up to date again.")
        self.write_failed = False
        return True

    def finish(self, outcome, message=""):
        self.run.update(ended_at=now_text(), outcome=outcome, message=message)
        return self.write()
