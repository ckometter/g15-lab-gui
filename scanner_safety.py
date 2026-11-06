"""Shared, cancellable scanner motion policy for dashboard and measurements."""
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

STORAGE_PATH = Path(__file__).resolve().parent / 'value_storage'
MAX_RAMP_RATE = 1.0
MAX_INCREMENT = 0.05


@dataclass(frozen=True)
class ScannerSettings:
    minimum: float = 0.0
    maximum: float = 60.0
    rate: float = 0.2
    increment: float = 0.01

    def __post_init__(self):
        values = (self.minimum, self.maximum, self.rate, self.increment)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Scanner settings must be finite.')
        if not 0 <= self.minimum < self.maximum <= 150:
            raise ValueError('Scanner limits must satisfy 0 <= minimum < maximum <= 150 V.')
        if not 0.001 <= self.rate <= MAX_RAMP_RATE:
            raise ValueError(f'Scanner rate must be 0.001 to {MAX_RAMP_RATE:g} V/s.')
        if not 0.001 <= self.increment <= MAX_INCREMENT:
            raise ValueError(f'Scanner increment must be 0.001 to {MAX_INCREMENT:g} V.')

    def validate(self, value):
        value = float(value)
        if not math.isfinite(value) or not self.minimum <= value <= self.maximum:
            raise ValueError(f'Scanner voltage {value:g} V is outside the configured '
                             f'{self.minimum:g} to {self.maximum:g} V limits.')
        return value

    @classmethod
    def load(cls):
        try:
            data = json.loads(STORAGE_PATH.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return cls()
        if not isinstance(data, dict) or not isinstance(data.get('position', {}), dict):
            raise ValueError('Saved position settings must be a JSON object.')
        saved = data.get('position', {}).get('scanner_safety', {})
        if not isinstance(saved, dict):
            raise ValueError('Saved scanner safety settings must be a JSON object.')
        return cls(**saved)


def wait_for_motion(seconds, keep_running):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not keep_running():
            return False
        time.sleep(min(0.02, max(0, end - time.monotonic())))
    return keep_running()


def ramp_offsets(scanner, targets, settings, *, rate=None, keep_running=lambda: True,
                 progress=None, write=None, wait=None):
    """Validate every target before motion; never catch up with a larger step.

    Wait BEFORE each increment, including the first and last. Cancellation leaves
    the last verified voltage in place. No rollback or automatic write retries.
    """
    rate = settings.rate if rate is None else float(rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Scanner ramp rate must be finite and positive.')
    rate = min(rate, settings.rate)
    targets = {axis: settings.validate(value) for axis, value in targets.items()}
    if not targets or any(type(axis) is not int or axis not in range(1, 8) for axis in targets):
        raise ValueError('Scanner axes must be integers from 1 to 7.')
    starts = {}
    for axis in targets:
        state = scanner.get_axis_status(axis)
        start = float(state['offset'])
        if not math.isfinite(start) or not 0 <= start <= 150:
            raise RuntimeError('Invalid scanner offset readback; movement blocked.')
        if state['mode'] != 'off':
            raise RuntimeError(f'Scanner axis {axis} is not in DC offset mode.')
        # A tightened limit must still permit a slow return into its range.
        starts[axis] = start
    write = write or scanner.set_voltage
    wait = wait or (lambda seconds: wait_for_motion(seconds, keep_running))
    for axis, target in targets.items():
        start = starts[axis]
        distance = abs(target - start)
        count = math.ceil(distance / min(settings.increment, rate * 0.05))
        previous = start
        for index in range(1, count + 1):
            value = target if index == count else start + (target - start) * index / count
            if not keep_running() or not wait(abs(value - previous) / rate) or not keep_running():
                return False
            with getattr(scanner, 'comm_lock', nullcontext()):
                state = scanner.get_axis_status(axis)
                if state['mode'] != 'off' or not math.isclose(float(state['offset']), previous, abs_tol=0.001, rel_tol=0):
                    raise RuntimeError(f'Scanner axis {axis} changed during ramp; stopped.')
                if not keep_running():
                    return False
                write(axis, value)
                actual = float(scanner.get_voltage(axis))
                if not math.isfinite(actual) or not math.isclose(actual, value, abs_tol=0.001, rel_tol=0):
                    raise RuntimeError(f'Scanner axis {axis} readback mismatch; ramp stopped.')
            previous = value
            if progress:
                progress(axis, actual)
    return keep_running()


class SafeScannerMotion:
    """Driver entry points; a single motion owns the scanner until it finishes."""
    def init_motion_safety(self):
        self.motion_lock = threading.Lock()
        self.scanner_settings_error = ''
        try:
            self.scanner_settings = ScannerSettings.load()
        except (ValueError, TypeError, OSError) as exc:
            self.scanner_settings = ScannerSettings()
            self.scanner_settings_error = f'Invalid saved scanner settings: {exc}. Apply valid settings before moving.'

    def configure_motion(self, settings):
        if not isinstance(settings, ScannerSettings):
            raise TypeError('Expected validated scanner settings.')
        if not self.motion_lock.acquire(blocking=False):
            raise RuntimeError('Scanner is moving; settings cannot change.')
        try:
            self.scanner_settings = settings
            self.scanner_settings_error = ''
        finally:
            self.motion_lock.release()

    def ramp_voltages(self, targets, *, relative=False, **kwargs):
        if not self.motion_lock.acquire(blocking=False):
            raise RuntimeError('Scanner is already moving.')
        try:
            if self.scanner_settings_error:
                raise RuntimeError(self.scanner_settings_error)
            if relative:
                targets = {axis: self.get_voltage(axis) + delta for axis, delta in targets.items()}
            return ramp_offsets(self, targets, self.scanner_settings,
                                write=self._write_voltage_step, **kwargs)
        finally:
            self.motion_lock.release()

    def set_voltage(self, axis, val):
        self.ramp_voltages({axis: val})


def move_scanner(scanner, targets, *, keep_running=lambda: True, rate=None, progress=None, relative=False):
    if scanner is None:
        raise RuntimeError('Scanner is not connected.')
    if isinstance(scanner, SafeScannerMotion):
        return scanner.ramp_voltages(targets, keep_running=keep_running, rate=rate, progress=progress, relative=relative)
    if relative:
        targets = {axis: scanner.get_voltage(axis) + delta for axis, delta in targets.items()}
    return ramp_offsets(scanner, targets, ScannerSettings(), keep_running=keep_running,
                        rate=rate, progress=progress)


def validate_map_scanner_targets(scanner, params, x_values, y_values):
    settings = scanner.scanner_settings if isinstance(scanner, SafeScannerMotion) else ScannerSettings()
    for name, values in ((params['x_param'], x_values), (params['y_param'], y_values)):
        if name.lower().startswith('position '):
            if scanner is None:
                raise RuntimeError('Scanner is not connected.')
            if isinstance(scanner, SafeScannerMotion) and scanner.scanner_settings_error:
                raise RuntimeError(scanner.scanner_settings_error)
            for value in values:
                settings.validate(value)
