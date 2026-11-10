"""Bounded ANC300 raster motion and disk-backed position/dual-gate acquisition."""
from contextlib import nullcontext
import json
import math
from pathlib import Path
import shutil
import threading
import time

import h5py
import numpy as np
from PyQt6 import QtCore

from measurements.Dual_gate_sweep_gui import SweepWorker, validate_dgs_parameters, require_voltage_source_modes
from dgs_gate_outputs import DEFAULT_GATE_OUTPUTS, normalize_gate_outputs
from scanner_safety import SafeScannerMotion

# The PSDGS scanner safety limit lives here. Applies to X/Y offsets, NOT gates.
SCANNER_MAX_VOLTAGE = 60.0
SCANNER_MIN_VOLTAGE = 0.0
SCANNER_RAMP_STEP_V = 0.05
MAX_SCAN_POSITIONS = 1_000_000


def validate_voltage(value):
    value = float(value)
    if not math.isfinite(value) or not SCANNER_MIN_VOLTAGE <= value <= SCANNER_MAX_VOLTAGE:
        raise ValueError(f"Scanner voltage {value:g} V is outside {SCANNER_MIN_VOLTAGE:g}–{SCANNER_MAX_VOLTAGE:g} V. Scan blocked.")
    return value


def scan_axis(start, size, step):
    """Inclusive regular grid; reject a nonintegral size/step rather than overshoot."""
    start = validate_voltage(start)
    size, step = float(size), float(step)
    if not math.isfinite(size) or size < 0 or not math.isfinite(step) or step <= 0:
        raise ValueError("Scan size must be nonnegative and step must be positive.")
    validate_voltage(start + size)
    intervals = size / step
    if not math.isclose(intervals, round(intervals), rel_tol=0, abs_tol=1e-7):
        raise ValueError("Scan size must be an exact multiple of its voltage step.")
    if intervals > 100000:
        raise ValueError("Too many scanner points on one axis (maximum 100001).")
    return np.linspace(start, start + size, int(round(intervals)) + 1)


def read_scanner_offsets(scanner):
    """Read configured X/Y offsets only; movement readiness is checked separately.

    A valid readback must remain visible even when the present controller mode
    or voltage would prevent a scan. Never use cached values or substitute zero.
    """
    with getattr(scanner, 'comm_lock', nullcontext()):
        values = [float(scanner.get_voltage(axis)) for axis in (1, 2)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('Scanner returned a non-finite offset; start values were not updated.')
    return values


def scanner_state(scanner):
    states = []
    with getattr(scanner, 'comm_lock', nullcontext()):
        for axis in (1, 2):
            state = scanner.get_axis_status(axis)
            validate_voltage(state['offset'])
            if state['mode'] != 'off':
                raise RuntimeError(f"Scanner axis {axis} is not in DC offset mode. Set the controller mode manually before scanning.")
            states.append(float(state['offset']))
    return states


def ramp_scanner(scanner, axis, target, rate, keep_running, wait=None):
    """Read the actual offset, ramp in small increments and verify every write."""
    target = validate_voltage(target)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Scanner ramp rate must be positive.")
    if isinstance(scanner, SafeScannerMotion):
        return scanner.ramp_voltages({axis: target}, rate=rate, keep_running=keep_running, wait=wait)
    if wait is None:
        def wait(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if not keep_running():
                    return False
                time.sleep(min(0.02, max(0, end - time.monotonic())))
            return keep_running()
    with getattr(scanner, 'comm_lock', nullcontext()):
        state = scanner.get_axis_status(axis)
        start = validate_voltage(state['offset'])
        if state['mode'] != 'off':
            raise RuntimeError(f"Scanner axis {axis} is not in DC offset mode.")
        distance = abs(target - start)
        if distance < 1e-9:
            return keep_running()
        count = int(math.ceil(distance / min(SCANNER_RAMP_STEP_V, rate * 0.05)))
        previous = start
        for value in np.linspace(start, target, count + 1)[1:]:
            validate_voltage(value)
            if not keep_running() or not wait(abs(value - previous) / rate):
                return False
            scanner.set_voltage(axis, float(value))
            actual = validate_voltage(scanner.get_voltage(axis))
            if not math.isclose(actual, value, abs_tol=0.011, rel_tol=0):
                raise RuntimeError(f"Scanner axis {axis} readback mismatch: requested {value:g}, received {actual:g} V.")
            previous = value
        return True


class PositionScanStore:
    """One HDF5 file with labelled (Y, X, gate2, gate1, [frame], energy) axes.

    Persist every measured point; never allocate the whole position scan in RAM.
    A shared lock also permits browsing completed positions during acquisition.
    """
    fields = ('spectra', 'raw_intensities', 'current_ch1', 'current_ch2',
              'measured_volt_ch1', 'measured_volt_ch2', 'measured_b', 'measured_mask')

    def __init__(self, path):
        self.path = str(path)
        self.lock = threading.RLock()

    def create(self, *, x, y, c1, c2, wavelengths, frames, v1, v2, metadata,
               background=None, reference=None):
        wavelengths = np.asarray(wavelengths, dtype=float)
        if wavelengths.ndim != 1 or not wavelengths.size or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0):
            raise ValueError('A valid camera wavelength calibration is required before scanning.')
        energy = 1239.84193 / np.asarray(wavelengths)
        self.order = np.argsort(energy)
        shape = (len(y), len(x), len(c2), len(c1))
        with self.lock, h5py.File(self.path, 'x') as f:
            f.attrs['schema'] = 'PSDGS-1'
            f.attrs['scan_order'] = 'X forward; slow X return to start; then Y step'
            f.attrs['scanner_max_voltage_V'] = SCANNER_MAX_VOLTAGE
            for key, value in metadata.items():
                f.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
            axes = [('y_voltages', y, 'V'), ('x_voltages', x, 'V'),
                    ('c2_coords', c2, 'V'), ('c1_coords', c1, 'V'),
                    ('frame', np.arange(frames), 'index'), ('energy_axis', energy[self.order], 'eV')]
            for name, values, units in axes:
                ds = f.create_dataset(name, data=values)
                ds.attrs['units'] = units
                ds.make_scale(name)
            f.create_dataset('wavelength_axis', data=np.asarray(wavelengths)[self.order]).attrs['units'] = 'nm'
            f.create_dataset('v1_grid', data=v1).attrs['units'] = 'V'
            f.create_dataset('v2_grid', data=v2).attrs['units'] = 'V'
            for name in self.fields:
                tail = (frames, len(energy)) if name == 'raw_intensities' else (len(energy),) if name == 'spectra' else ()
                dimensions = shape + tail
                labels = [a[0] for a in axes[:4]]
                if name == 'raw_intensities': labels += ['frame', 'energy_axis']
                elif name == 'spectra': labels += ['energy_axis']
                ds = f.create_dataset(name, shape=dimensions,
                                      dtype=bool if name == 'measured_mask' else 'f8',
                                      chunks=(1, 1, 1, 1) + tuple(1 if n == 0 and len(tail) == 2 else size for n, size in enumerate(tail)),
                                      compression='gzip', fillvalue=False if name == 'measured_mask' else np.nan)
                ds.attrs['axis_order'] = ','.join(labels)
                ds.attrs['units'] = ('counts' if name in ('spectra', 'raw_intensities') else
                                     'A' if name.startswith('current') else 'V' if name.startswith('measured_volt') else
                                     'T' if name == 'measured_b' else 'boolean')
                for i, label in enumerate(labels):
                    ds.dims[i].label = label
                    ds.dims[i].attach_scale(f[label])
            f.create_dataset('position_complete', shape=shape[:2], dtype=bool)
            f.create_dataset('position_readback_V', shape=shape[:2] + (2,), dtype='f8', fillvalue=np.nan)
            f['position_readback_V'].attrs['axis_order'] = 'y,x,scanner_axis(X,Y)'
            f.create_dataset('position_signal', shape=shape[:2], dtype='f8', fillvalue=np.nan)
            f.attrs['status'] = 'ready'
            for name, data in [('background', background), ('reference', reference)]:
                if data is not None:
                    group = f.create_group(name)
                    for key, values in data.items(): group.create_dataset(key, data=values)

    def write_point(self, iy, ix, point):
        i, j, c1, c2, v1, v2, spectrum, cur1, cur2, mv1, mv2, field, raw = point
        values = (np.asarray(spectrum)[self.order], np.asarray(raw)[:, self.order],
                  cur1, cur2, mv1, mv2, field)
        with self.lock, h5py.File(self.path, 'r+') as f:
            key = (iy, ix, j, i)
            for name, value in zip(self.fields[:-1], values): f[name][key] = value
            f.flush()
            f['measured_mask'][key] = True

    def position_started(self, iy, ix, readback):
        with self.lock, h5py.File(self.path, 'r+') as f:
            f['position_readback_V'][iy, ix] = readback

    def position_finished(self, iy, ix):
        with self.lock, h5py.File(self.path, 'r+') as f:
            complete = bool(np.all(f['measured_mask'][iy, ix]))
            f['position_complete'][iy, ix] = complete
            mask = f['measured_mask'][iy, ix]
            if np.any(mask):
                f['position_signal'][iy, ix] = np.mean(np.sum(f['spectra'][iy, ix][mask], axis=-1))
            return complete

    def set_status(self, status, error=''):
        with self.lock, h5py.File(self.path, 'r+') as f:
            f.attrs['status'], f.attrs['error'] = status, error

    def read_position(self, iy, ix):
        with self.lock, h5py.File(self.path, 'r') as f:
            return {name: f[name][iy, ix] for name in self.fields}

    def copy_to(self, path):
        with self.lock:
            if Path(path).resolve() != Path(self.path).resolve():
                shutil.copyfile(self.path, path)


class PositionSweepWorker(SweepWorker):
    """Raster positions outside a complete, unchanged standalone DGS per pixel.

    Only SweepWorker sets gate voltages. Only ramp_scanner sets scanner offsets.
    The blocking gate run must finish before the next X move or slow line return.
    """
    position_started = QtCore.pyqtSignal(int, int)
    position_point = QtCore.pyqtSignal(int, int, object)
    position_finished = QtCore.pyqtSignal(int, int)

    def __init__(self, station, params, scan, store):
        super().__init__(station, params)
        self.scan, self.store = scan, store
        self.failure = ''
        self.completed = False

    def run(self):
        try:
            x, y = self.scan['x'], self.scan['y']
            for value in np.concatenate((x, y)): validate_voltage(value)
            if isinstance(self.station.piezo300, SafeScannerMotion):
                for value in np.concatenate((x, y)):
                    self.station.piezo300.scanner_settings.validate(value)
            validate_dgs_parameters(self.params)
            if normalize_gate_outputs(self.params.get('gate_outputs')) != DEFAULT_GATE_OUTPUTS:
                raise ValueError('Position scans drive both gates with the Keysight only.')
            scanner_state(self.station.piezo300)  # Both axes before any write.
            require_voltage_source_modes(self.station.smu)
            self.station.smu.enforce_current_compliance()
            self.store.set_status('running')
            count = 0
            gate_points = self.params['c1_pts'] * self.params['c2_pts']
            total = len(x) * len(y) * gate_points
            scanner = self.station.piezo300
            def move(axis, target, rate):
                return ramp_scanner(scanner, axis, target, rate, self.check_pause_and_stop)
            for iy, yv in enumerate(y):
                # This return is deliberately completed BEFORE changing Y.
                self.status_updated.emit('Status: Slowly returning X to the line start...')
                if not move(1, x[0], self.scan['return_rate']): break
                if not move(2, yv, self.scan['move_rate']): break
                for ix, xv in enumerate(x):
                    if not move(1, xv, self.scan['move_rate']): break
                    end = time.monotonic() + self.scan['settle']
                    while time.monotonic() < end and self.check_pause_and_stop(): time.sleep(0.02)
                    if not self.check_pause_and_stop(): break
                    self.store.position_started(iy, ix, scanner_state(scanner))
                    self.position_started.emit(iy, ix)
                    position_status = (f'Status: DGS at scanner X={xv:g} V, Y={yv:g} V '
                                       f'(pixel {iy * len(x) + ix + 1}/{len(x) * len(y)})')
                    self.status_updated.emit(f'{position_status}: 0/{gate_points} gate points')
                    gate = SweepWorker(self.station, self.params)
                    gate._control = self
                    gate.check_pause_and_stop = self.check_pause_and_stop
                    errors = []
                    def gate_progress(current, per_pixel_total):
                        self.status_updated.emit(f'{position_status}: {current}/{per_pixel_total} gate points')
                    def record(*point):
                        nonlocal count
                        try:
                            self.store.write_point(iy, ix, point)
                            self.position_point.emit(iy, ix, point)
                            count += 1
                            self.progress_updated.emit(count, total)
                        except Exception as exc:
                            errors.append(str(exc))
                            self.stop()
                    gate.point_measured.connect(record, QtCore.Qt.ConnectionType.DirectConnection)
                    gate.error_occurred.connect(errors.append, QtCore.Qt.ConnectionType.DirectConnection)
                    gate.progress_updated.connect(gate_progress, QtCore.Qt.ConnectionType.DirectConnection)
                    gate.voltage_updated.connect(self.voltage_updated.emit, QtCore.Qt.ConnectionType.DirectConnection)
                    gate._do_run()
                    if errors: raise RuntimeError(errors[0])
                    complete = self.store.position_finished(iy, ix)
                    if self._is_running and not complete:
                        raise RuntimeError(f'Incomplete DGS at scanner X={xv:g} V, Y={yv:g} V. '
                                           'Scan stopped before moving to another position; partial data retained.')
                    self.position_finished.emit(iy, ix)
                    if not self.check_pause_and_stop(): break
                if not self.check_pause_and_stop(): break
                # Also return slowly after the final line. Leave Y at its last line.
                if not move(1, x[0], self.scan['return_rate']): break
            self.completed = self._is_running and count == total
            self.store.set_status('complete' if self.completed else 'interrupted')
        except Exception as exc:
            self.failure = str(exc)
            try: self.store.set_status('error', self.failure)
            except Exception: pass
        finally:
            try: self.station.smu.return_to_local()
            except Exception: pass
            # One completion signal for the entire scan, including errors/stops.
            self.finished.emit()
