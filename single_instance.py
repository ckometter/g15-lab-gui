"""Prevent two copies of the lab dashboard from opening the same hardware."""

import hashlib
import os
import tempfile

from PyQt6.QtCore import QLockFile


def lock_path(project_dir):
    """Use one lock per checkout, regardless of the launch working directory."""
    normalized = os.path.normcase(os.path.realpath(project_dir))
    suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"g15_qcodes_lab_gui_{suffix}.lock")


def acquire_lock(project_dir):
    """Return the held lock, or an unlocked object with a Qt error code."""
    lock = QLockFile(lock_path(project_dir))
    lock.setStaleLockTime(0)
    return lock, lock.tryLock(0)
