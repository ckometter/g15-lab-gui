"""Data plotter: a separate, read-only program for the dashboard's saved measurement files.

It shows Spectrum, Dual-Gate Sweep, B/V Sweep, Vector Map, Scalar Map, Position DGS,
Capacitance and Transport files, each in that tab's own analysis view.

Run it on its own, also while the dashboard is measuring:

    python data_plotter.py [FILE or FOLDER]
    python data_plotter.py --check FILE...    print what would be shown; no window

The dashboard's "Open Data plotter" button starts the same program. Because it is a
separate process, loading and plotting never compete with an acquisition for the
dashboard's GUI thread or for h5py's lock (one per process), and an error here cannot
stop a measurement. This program:

- opens HDF5 files read-only, without HDF5 file locks, and only for the length of a read;
- retries a read while another program is writing the file;
- reads in background-priority threads of a below-normal-priority process;
- uses simulated devices only, refuses instrument, serial-port and network connections,
  and keeps the analysis windows' settings in a private temporary folder.

File contents never become measurement settings. See docs/DATA_PLOTTER_AND_LIVE.md.
"""
if __name__ == '__main__':
    import os as _os
    # HDF5 reads this once, when the library loads, so it must come before any h5py import.
    # Then no open in this process takes a lock; a lock would make a writer's open fail.
    _os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import ctypes
import functools
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets, sip

from measurements.B_V_sweep_gui import BVSweepWindow
from measurements.capacitance_gui import FILE_SCHEMA as CAPACITANCE_SCHEMA
from measurements.capacitance_gui import CapacitanceWindow, parse_capacitance_file
from measurements.gate_map_gui import GateMapWindow
from measurements.photoluminescence_map_gui import PhotoluminescenceMapWindow
from measurements.position_dgs_gui import PositionDGSWindow
from measurements.scalar_map_gui import ScalarMapWindow
from measurements.transport_gui import FILE_SCHEMA as TRANSPORT_SCHEMA
from measurements.transport_gui import TransportWindow, parse_transport_file
from position_dgs import PositionScanStore


REPO_DIR = Path(__file__).resolve().parent
# One tab per measurement tab of the dashboard that saves files, in the dashboard's order.
TAB_TITLES = {'spectrum': 'Spectrum', 'dgs': 'Dual-Gate Sweep', 'bv': 'B/V Sweep', 'vector_map': 'Vector Map',
              'scalar_map': 'Scalar Map', 'position_dgs': 'Position DGS', 'capacitance': 'Capacitance',
              'transport': 'Transport'}
# Capacitance and Transport files name their format; their own tabs' readers parse them.
GATE_MAP_FORMATS = {CAPACITANCE_SCHEMA: ('capacitance', parse_capacitance_file),
                    TRANSPORT_SCHEMA: ('transport', parse_transport_file)}
# The analysis windows keep their settings (value_storage) and data folders in parent_dir.
SETTINGS_MODULES = tuple('measurements.' + name for name in (
    'single_spectra_gui', 'Dual_gate_sweep_gui', 'B_V_sweep_gui', 'photoluminescence_map_gui', 'scalar_map_gui',
    'position_dgs_gui', 'gate_map_gui', 'capacitance_gui', 'transport_gui'))
UNSUPPORTED = ('Choose a saved measurement: Spectrum, Dual-Gate Sweep, B/V Sweep, Vector Map, Scalar Map, '
               'Position DGS, Capacitance or Transport HDF5 file.')
DATA_ATTRIBUTES = {'raw_intensities': 'data_raw_spectra', 'spectra': 'data_spectra'}
# Output of a viewer started by the dashboard (it has no console).
LOG_FILE = Path(tempfile.gettempdir()) / 'g15_data_plotter.log'
# Pauses before each retry of a read that met a writer: about 5.5 s in all. A writer holds
# its lock for milliseconds per Position DGS point, and for seconds during a DGS autosave.
RETRY_DELAYS_S = (0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0)
# A file that changes during every read is shown from its last complete read after this many.
MAX_CHANGED_READS = 3
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
THREAD_MODE_BACKGROUND_BEGIN = 0x00010000


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _panel(window):
    widget = window.centralWidget()
    return widget if widget is not None else window._panel_widget


def _measurement_controls(window):
    """What the viewer hides: Single Spectra's control boxes, or another tab's left control column."""
    if hasattr(window, 'set_plotter_mode'):
        return _panel(window).findChildren(QtWidgets.QGroupBox)
    panel = _panel(window)
    if isinstance(panel, QtWidgets.QMainWindow):
        # Tabs whose centralWidget() returns the window itself.
        panel = QtWidgets.QMainWindow.centralWidget(panel)
    first = panel.layout().itemAt(0).widget()
    return [first.widget(0) if isinstance(first, QtWidgets.QSplitter) else first]


def _stop_threads(window):
    """Stop a window's device pollers; the viewer has nothing to poll."""
    threads = [value for value in vars(window).values()
               if isinstance(value, QtCore.QThread) and value.isRunning()]
    for thread in threads:
        if hasattr(thread, 'stop'):
            thread.stop()
        else:
            thread.requestInterruption()
    for thread in threads:
        thread.wait(10000)


def _axis(handle, name):
    data = np.asarray(handle[name], dtype=float)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError(f'{name} must contain a finite, nonempty coordinate axis.')
    return data


def _calibration(handle):
    result = SimpleNamespace()
    for group, prefix in (('background', 'bg'), ('reference', 'ref'), ('signal', 'signal')):
        x = y = raw = None
        if group in handle:
            x = _axis(handle[group], 'wavelength')
            if np.any(x <= 0):
                raise ValueError(f'{group} wavelengths must be positive.')
            y = np.asarray(handle[group]['intensity'], dtype=float)
            if y.shape != x.shape:
                raise ValueError(f'{group} wavelength and intensity lengths differ.')
            raw = np.asarray(handle[group]['raw_intensities']) if 'raw_intensities' in handle[group] else y[None, :]
            if raw.ndim == 1:
                raw = raw[None, :]  # Legacy single-frame Spectrum exports.
            if raw.ndim != 2 or not raw.shape[0] or raw.shape[-1] != x.size:
                raise ValueError(f'{group} raw frames do not match its wavelength axis.')
            if not np.isfinite(y).all() or not np.isfinite(raw).all():
                raise ValueError(f'{group} contains non-finite intensity values.')
            # Spectrum processing uses np.interp, which requires increasing
            # wavelengths. Reorder every aligned array together, in memory only.
            order = np.argsort(x)
            x, y, raw = x[order], y[order], raw[:, order]
            if np.any(np.diff(x) <= 0):
                raise ValueError(f'{group} wavelengths must be unique.')
        setattr(result, prefix + '_x', x)
        setattr(result, prefix + '_y', y)
        setattr(result, prefix + '_y_raw', raw)
    return result


# --- Reading files that another program may be writing ---------------------------------

def _file_state(path):
    state = os.stat(path)
    return state.st_size, state.st_mtime_ns


def _writer_busy(exc):
    """True for the errors a read gets while another program has the file open for writing.

    On Windows a writer's HDF5 lock refuses every read of the file: 'Permission denied'.
    """
    text = str(exc).lower()
    return isinstance(exc, PermissionError) or 'permission denied' in text or 'unable to lock' in text


def read_hdf5(path, read, *, delays=RETRY_DELAYS_S, should_stop=None):
    """Return ``(read(file), changing)`` from a read-only, lock-free, short-lived handle.

    While a writer has the file open, its HDF5 lock makes every read fail; a read that
    overlaps a write can also meet a half-updated file. Both are retried. A file that
    does not change and still fails to read is reported at once. A read counts when the
    file did not change while it ran; ``changing`` is True when it kept changing and the
    last complete read is returned instead.
    """
    error, latest, changed = None, None, 0
    for delay in (0.0, *delays):
        if delay:
            if should_stop is not None and should_stop():
                break
            time.sleep(delay)
        before = _file_state(path)
        try:
            with h5py.File(path, 'r', locking=False) as f:
                result = read(f)
        except Exception as exc:
            if not _writer_busy(exc) and _file_state(path) == before:
                raise
            error = exc
            continue
        if _file_state(path) == before:
            return result, False
        latest, changed = result, changed + 1
        if changed >= MAX_CHANGED_READS:
            break
    if changed:
        return latest, True
    raise OSError(f'{Path(path).name} stayed busy: another program kept it open for writing. '
                  f'Try again in a moment. ({error})') from error


def read_position(f, iy, ix):
    """One scanner position's DGS. A writer sets a point's mask after its data, so reading
    the mask first means no point is marked measured before its data could be read."""
    names = ('measured_mask',) + tuple(name for name in PositionScanStore.fields if name != 'measured_mask')
    return {name: f[name][iy, ix] for name in names}


def _grid_fields(f, shape, required, optional=()):
    """Per-point arrays of one sweep shape; optional ones may be absent from older files."""
    data = {}
    for name in (*required, *optional):
        if name not in f:
            if name in optional:
                continue
            raise ValueError(f'The file is missing {name}.')
        values = np.asarray(f[name])
        if values.shape != shape:
            raise ValueError(f'{name} shape does not match the recorded coordinates.')
        data[name] = values
    return data


def _spectra_fields(f, shape, samples):
    """Spectra (shape + samples) and, if saved, their raw frames (shape + frames + samples)."""
    if 'spectra' not in f:
        raise ValueError('The file is missing spectra.')
    data = {'spectra': np.asarray(f['spectra'])}
    if data['spectra'].shape != shape + (samples,):
        raise ValueError('spectra shape does not match the recorded coordinates.')
    if 'raw_intensities' in f:
        raw = np.asarray(f['raw_intensities'])
        if raw.ndim != len(shape) + 2 or raw.shape[:len(shape)] != shape or raw.shape[-1] != samples:
            raise ValueError('Raw frame dimensions do not match the recorded coordinates.')
        data['raw_intensities'] = raw
    return data


def _recorded_scan_params(f):
    """The X/Y parameters a map recorded in its embedded settings (value_storage), if any."""
    try:
        storage = json.loads(_text(f['metadata/value_storage'][()]))
    except Exception:
        return {}
    for section in (storage, *storage.values()) if isinstance(storage, dict) else ():
        params = section.get('params') if isinstance(section, dict) else None
        if isinstance(params, dict):
            return params
    return {}


def _read_bv(f, result):
    result['kind'] = 'bv'
    v, b, energy = _axis(f, 'v_coords'), _axis(f, 'b_coords'), _axis(f, 'energy_axis')
    if np.any(energy <= 0) or np.any(np.diff(energy) <= 0):
        raise ValueError('The saved energy axis must be positive and strictly increasing.')
    wavelength = _axis(f, 'wavelength_axis') if 'wavelength_axis' in f else 1239.84193 / energy
    if wavelength.size != energy.size:
        raise ValueError('Energy and wavelength axes have different lengths.')
    shape = (b.size, v.size)
    result['axes'] = {'v_coords': v, 'b_coords': b, 'energy_axis': energy, 'wavelength_axis': wavelength}
    result['grids'] = {name: values.astype(float) for name, values in
                       _grid_fields(f, shape, ('v1_grid', 'v2_grid')).items()}
    result['data'] = {**_spectra_fields(f, shape, energy.size),
                      **_grid_fields(f, shape, ('current_ch1', 'current_ch2', 'measured_mask'),
                                     ('measured_v1', 'measured_v2', 'measured_b'))}
    result['control'] = [float(f.attrs[name]) if name in f.attrs else None
                         for name in ('control_coeff_ch1', 'control_coeff_ch2')]
    return result


def _read_map(f, result, kind):
    result['kind'] = kind
    x, y = _axis(f, 'x_coords'), _axis(f, 'y_coords')
    shape = (y.size, x.size)
    result['axes'] = {'x_coords': x, 'y_coords': y}
    fields = ('current_ch1', 'current_ch2', 'measured_v1', 'measured_v2', 'measured_mask')
    if kind == 'vector_map':
        wavelengths = _axis(f, 'wavelengths')
        if np.any(wavelengths <= 0):
            raise ValueError('Wavelengths must be positive.')
        result['axes']['wavelengths'] = wavelengths
        result['data'] = {**_spectra_fields(f, shape, wavelengths.size),
                          **_grid_fields(f, shape, fields, ('measured_b',))}
    else:
        result['data'] = _grid_fields(f, shape, ('power', *fields), ('measured_b',))
    result['params'] = _recorded_scan_params(f)
    return result


def _read_file(f, path, position=None):
    result = {'path': path, 'attrs': dict(f.attrs), 'calibration': _calibration(f)}
    schema = _text(f.attrs.get('schema', ''))
    if schema in GATE_MAP_FORMATS:
        result['kind'], parse = GATE_MAP_FORMATS[schema]
        result['loaded'] = parse(f)
        return result
    if schema == 'PSDGS-1':
        result['kind'] = 'position_dgs'
    elif all(key in f for key in ('c1_coords', 'c2_coords', 'spectra', 'energy_axis')):
        if f['spectra'].ndim != 3:
            raise ValueError('This file is not a two-dimensional dual gate sweep.')
        result['kind'] = 'dgs'
    elif all(key in f for key in ('v_coords', 'b_coords', 'spectra', 'energy_axis')):
        return _read_bv(f, result)
    elif all(key in f for key in ('x_coords', 'y_coords', 'spectra', 'wavelengths')):
        return _read_map(f, result, 'vector_map')
    elif all(key in f for key in ('x_coords', 'y_coords', 'power')):
        return _read_map(f, result, 'scalar_map')
    elif any(key in f for key in ('signal', 'background', 'reference')):
        # Checked last: sweeps and maps save the background and reference groups too.
        result['kind'] = 'spectrum'
        return result
    else:
        raise ValueError(UNSUPPORTED)
    kind = result['kind']

    axes = {name: _axis(f, name) for name in ('c1_coords', 'c2_coords', 'energy_axis')}
    if np.any(axes['energy_axis'] <= 0):
        raise ValueError('Energy coordinates must be positive.')
    # Older DGS exports recorded energy only. Its inverse is the same
    # calibration, rather than a guessed spectrometer centre wavelength.
    axes['wavelength_axis'] = (1239.84193 / axes['energy_axis']
                               if kind == 'dgs' and 'wavelength_axis' not in f
                               else _axis(f, 'wavelength_axis'))
    if axes['energy_axis'].size != axes['wavelength_axis'].size:
        raise ValueError('Energy and wavelength axes have different lengths.')
    if np.any(axes['energy_axis'] <= 0) or np.any(axes['wavelength_axis'] <= 0):
        raise ValueError('Energy and wavelength coordinates must be positive.')
    if np.any(np.diff(axes['energy_axis']) <= 0):
        raise ValueError('The saved gate scan energy axis must be strictly increasing.')
    result['axes'] = axes
    gate_shape = (axes['c2_coords'].size, axes['c1_coords'].size)
    result['grids'] = {}
    for name in ('v1_grid', 'v2_grid'):
        grid = np.asarray(f[name], dtype=float)
        if grid.shape != gate_shape:
            raise ValueError(f'{name} shape does not match the gate coordinates.')
        result['grids'][name] = grid
    if kind == 'position_dgs':
        result['scan_x'], result['scan_y'] = _axis(f, 'x_voltages'), _axis(f, 'y_voltages')
        shape = (len(result['scan_y']), len(result['scan_x'])) + gate_shape
        # Only masks are read for the overview; spectra remain on disk and
        # the pixel selector reads one DGS at a time.
        if f['measured_mask'].shape != shape:
            raise ValueError('Position mask shape does not match the scanner and gate coordinates.')
        counts = np.zeros(shape[:2], dtype=int)
        for iy in range(shape[0]):
            counts[iy] = np.count_nonzero(f['measured_mask'][iy], axis=(-2, -1))
        result['position_counts'] = counts
        coefficients = f.attrs.get('gate_coefficients', '[1, 0, 0, 1]')
        result['coefficients'] = json.loads(_text(coefficients)) if isinstance(coefficients, (str, bytes)) else list(coefficients)
    else:
        shape = gate_shape
        result['coefficients'] = [float(f.attrs.get('coeff_' + letter, default))
                                  for letter, default in zip('abcd', (1, 0, 0, 1))]
    if len(result['coefficients']) != 4 or not np.isfinite(result['coefficients']).all():
        raise ValueError('The file must provide four finite gate coefficients.')
    result['data'] = {}
    for name in PositionScanStore.fields:
        if name not in f:
            if kind == 'dgs' and name in ('raw_intensities', 'measured_b', 'measured_volt_ch1', 'measured_volt_ch2'):
                continue
            raise ValueError(f'The file is missing {name}.')
        dataset = f[name]
        if name == 'spectra':
            expected = shape + (axes['energy_axis'].size,)
        elif name == 'raw_intensities':
            if dataset.ndim != len(shape) + 2:
                raise ValueError('Raw frame dimensions do not match the gate scan.')
            expected = shape + dataset.shape[-2:]
            if dataset.shape[-1] != axes['energy_axis'].size:
                raise ValueError('Raw frame wavelength length does not match the energy axis.')
        else:
            expected = shape
        if dataset.shape != expected:
            raise ValueError(f'{name} shape does not match the recorded coordinates.')
        if kind == 'dgs':
            result['data'][name] = dataset[:]
    if kind == 'position_dgs':
        # The pixel shown first is read here too, in the loading thread.
        if position is None or not all(0 <= int(i) < n for i, n in zip(position, counts.shape)):
            measured = np.flatnonzero(counts)
            position = np.unravel_index(measured[0], counts.shape) if measured.size else (0, 0)
        result['position'] = (int(position[0]), int(position[1]))
        result['position_data'] = read_position(f, *result['position'])
    return result


def read_dataset(path, position=None, *, should_stop=None):
    """Validate and read a saved file for display: no Qt objects, devices or writes.

    ``position`` (iy, ix) picks the Position DGS pixel to read with the file.
    """
    path = str(Path(path).resolve())
    result, changing = read_hdf5(path, lambda f: _read_file(f, path, position), should_stop=should_stop)
    result['changing'] = changing
    result['read_at'] = time.strftime('%H:%M:%S')
    return result


def _measured(mask):
    mask = np.asarray(mask, dtype=bool)
    return f'{np.count_nonzero(mask)} / {mask.size} points measured'


def describe(data):
    """One line about a dataset, as --check prints it."""
    kind = data['kind']
    title = TAB_TITLES[kind]
    if kind == 'spectrum':
        calibration = data['calibration']
        parts = [f'{label} {len(getattr(calibration, prefix + "_x"))} samples'
                 for prefix, label in (('signal', 'signal'), ('bg', 'background'), ('ref', 'reference'))
                 if getattr(calibration, prefix + '_x') is not None]
        text = 'Spectrum: ' + ', '.join(parts)
    elif kind == 'bv':
        axes = data['axes']
        text = (f"{title}: {len(axes['v_coords'])} control x {len(axes['b_coords'])} field points, "
                f"{len(axes['energy_axis'])} spectral samples; {_measured(data['data']['measured_mask'])}")
    elif kind in ('vector_map', 'scalar_map'):
        axes = data['axes']
        text = f"{title}: {len(axes['x_coords'])} x {len(axes['y_coords'])} points"
        if kind == 'vector_map':
            text += f", {len(axes['wavelengths'])} spectral samples"
        text += f"; {_measured(data['data']['measured_mask'])}"
    elif kind in ('capacitance', 'transport'):
        loaded = data['loaded']
        text = f"{title}: {len(loaded['x'])} x {len(loaded['y'])} points; {_measured(loaded['mask'])}"
        if 'status' in data['attrs']:
            text += f"; recorded status {_text(data['attrs']['status'])}"
    else:
        axes = data['axes']
        gates = (f"{len(axes['c1_coords'])} x {len(axes['c2_coords'])} gates, "
                 f"{len(axes['energy_axis'])} spectral samples")
        if kind == 'dgs':
            mask = data['data']['measured_mask']
            text = f'{TAB_TITLES[kind]}: {gates}; {np.count_nonzero(mask)} / {mask.size} gate points measured'
        else:
            counts = data['position_counts']
            text = (f'{TAB_TITLES[kind]}: {counts.shape[1]} x {counts.shape[0]} positions, {gates}; '
                    f'{np.count_nonzero(counts)} / {counts.size} positions with data')
            if 'status' in data['attrs']:
                text += f"; recorded status {_text(data['attrs']['status'])}"
    if data.get('changing'):
        text += ' (the file changed while it was read)'
    return text


# --- This process: priorities, no devices, private settings -----------------------------

@functools.lru_cache(maxsize=1)
def _kernel32():
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetThreadPriority.argtypes = (ctypes.c_void_p, ctypes.c_int)
    return kernel32


def lower_process_priority():
    """Windows: below-normal CPU priority, so a measurement program always runs first."""
    if sys.platform == 'win32':
        kernel32 = _kernel32()
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)


def background_priority():
    """Windows: lowest disk and CPU priority for the calling thread, which reads files."""
    if sys.platform == 'win32':
        kernel32 = _kernel32()
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN)


def _refuse_device_access(*_args, **_kwargs):
    raise RuntimeError('The Data plotter only reads files: it never opens instruments, '
                       'serial ports or network connections.')


def block_device_access():
    """Make instrument (VISA), serial-port, spectrometer-server and network connections fail."""
    import socket
    socket.create_connection = _refuse_device_access
    for module_name, attribute in (('pyvisa', 'ResourceManager'), ('Pyro4', 'Proxy'), ('serial', 'Serial')):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        setattr(module, attribute, _refuse_device_access)


def use_private_settings(folder, source=REPO_DIR):
    """Keep the analysis windows' settings and folders in ``folder``, never the dashboard's.

    Building those windows writes value_storage. A snapshot copy keeps display choices.
    """
    try:
        shutil.copyfile(Path(source) / 'value_storage', Path(folder) / 'value_storage')
    except OSError:
        pass  # Missing: the windows use their defaults.
    for name in SETTINGS_MODULES:
        importlib.import_module(name).parent_dir = str(folder)


def simulated_station():
    """A station of simulated devices, as the analysis windows expect; none is ever real."""
    from qcodes.station import Station
    from station_devices import ensure_simulated_devices
    station = Station()
    ensure_simulated_devices(station)
    smu = station.components.get('smu')
    for channel in (getattr(smu, 'ch1', None), getattr(smu, 'ch2', None)):
        if channel is not None:
            # The SMU driver stores ramp settings in the dashboard's value_storage itself.
            channel._save_param_to_storage = lambda key, value: None
    return station


def launch(paths=(), popen=subprocess.Popen):
    """Start the Data plotter as a separate program and return at once (dashboard button).

    It has no console; its output goes to LOG_FILE. Closing the dashboard leaves it open.
    """
    command = [sys.executable, str(REPO_DIR / 'data_plotter.py'), *map(str, paths)]
    # Unbuffered, so the log is complete up to the moment of any crash.
    environment = dict(os.environ, HDF5_USE_FILE_LOCKING='FALSE', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    flags = 0
    if sys.platform == 'win32':
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(f"\n--- Data plotter started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log.flush()
        return popen(command, cwd=str(REPO_DIR), env=environment, stdin=subprocess.DEVNULL,
                     stdout=log, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)


# --- Worker threads ---------------------------------------------------------------------

class DataLoadWorker(QtCore.QThread):
    loaded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, path, position=None, parent=None):
        super().__init__(parent)
        self.path = path
        self.position = position

    def run(self):
        background_priority()
        try:
            result = read_dataset(self.path, self.position, should_stop=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.loaded.emit(result)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class PositionReadWorker(QtCore.QThread):
    """Reads one scanner position's DGS for the Position DGS tab."""
    loaded = QtCore.pyqtSignal(object, int, int, object)
    failed = QtCore.pyqtSignal(object, int, int, str)

    def __init__(self, store, iy, ix, parent=None):
        super().__init__(parent)
        self.store, self.iy, self.ix = store, iy, ix

    def run(self):
        background_priority()
        try:
            data = self.store.read_from_disk(self.iy, self.ix, should_stop=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.loaded.emit(self.store, self.iy, self.ix, data)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(self.store, self.iy, self.ix, str(exc))


class ReadOnlyScanStore:
    """A Position DGS file for display: read-only, lock-free, one position at a time.

    Unlike PositionScanStore it has no write methods. A position read in a worker thread
    is handed over with provide(), so the GUI thread does not wait for the disk.
    """
    fields = PositionScanStore.fields

    def __init__(self, path):
        self.path = str(path)
        self._ready = {}

    def provide(self, iy, ix, data):
        self._ready = {(iy, ix): data}  # Only one: a position's raw frames can be large.

    def has(self, iy, ix):
        return (iy, ix) in self._ready

    def read_from_disk(self, iy, ix, should_stop=None):
        data, _changing = read_hdf5(self.path, lambda f: read_position(f, iy, ix), should_stop=should_stop)
        return data

    def read_position(self, iy, ix):
        """The Position DGS tab's read: the provided data, else a direct read."""
        data = self._ready.pop((iy, ix), None)
        return self.read_from_disk(iy, ix) if data is None else data


class PlotterPositionDGSWindow(PositionDGSWindow):
    """Position DGS analysis whose clicked pixels are read in a worker thread."""

    def __init__(self, station, embed=False):
        self._position_reader = None
        self._wanted_position = None
        super().__init__(station, embed=embed)

    def select_position(self, iy, ix):
        store = self.store
        if store is None or not hasattr(store, 'provide') or store.has(iy, ix):
            self._wanted_position = None  # This request supersedes any pending one.
            super().select_position(iy, ix)
            return
        self._wanted_position = (iy, ix)
        self.selected_label.setText(f'Reading the DGS at scanner X={self.scan_x[ix]:g} V, '
                                    f'Y={self.scan_y[iy]:g} V from the file…')
        if self._position_reader is None:
            self._read_wanted_position()

    def _read_wanted_position(self):
        reader = PositionReadWorker(self.store, *self._wanted_position, parent=self)
        reader.loaded.connect(self._position_loaded)
        reader.failed.connect(self._position_failed)
        reader.finished.connect(self._position_reader_finished)
        self._position_reader = reader
        reader.start()

    def _position_loaded(self, store, iy, ix, data):
        if store is self.store and self._wanted_position == (iy, ix):
            store.provide(iy, ix, data)
            self.select_position(iy, ix)

    def _position_failed(self, store, iy, ix, message):
        if store is self.store and self._wanted_position == (iy, ix):
            self._wanted_position = None
            self.status_lbl.setText(f'Cannot read selected position: {message}')
            if self.selected_position is not None:
                self.update_selected_label()

    def _position_reader_finished(self):
        reader, self._position_reader = self._position_reader, None
        if reader is not None:
            reader.deleteLater()
        # The newest click is read next; clicks in between are skipped.
        if self._wanted_position is not None and self.store is not None:
            self._read_wanted_position()

    def position_reader_busy(self):
        return self._position_reader is not None

    def cancel_position_reads(self):
        self._wanted_position = None
        if self._position_reader is not None:
            self._position_reader.requestInterruption()


# --- Display -----------------------------------------------------------------------------

def _set_control(widget, value):
    with QtCore.QSignalBlocker(widget):
        if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            widget.setValue(value)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.setCurrentIndex(value)
        elif isinstance(widget, QtWidgets.QAbstractButton):
            widget.setChecked(value)
        else:
            widget.setText(value)


def update_gate_labels(window):
    """Apply recorded labels without invoking the measurement coordinate editor."""
    c1, c2 = window.c1_name_edit.text(), window.c2_name_edit.text()
    labels = {'gatemap_win': (c1, c2), 'vslice_win': (None, c2), 'hslice_win': (None, c1),
              't2_heatmap_win': (c1, c2), 't3_heatmap_win': (c1, c2),
              't2_hslice_win': (c1, None), 't2_vslice_win': (c2, None),
              't3_hslice_win': (c1, None), 't3_vslice_win': (c2, None)}
    for name, pair in labels.items():
        for axis, label in zip(('bottom', 'left'), pair):
            if label is not None:
                getattr(window, name).setLabel(axis, label, units='V')
    window.ctrl_c1_box.setTitle(c1)
    window.ctrl_c2_box.setTitle(c2)
    window.v1_formula_label.setText(c1 + ' =')
    window.v2_formula_label.setText(c2 + ' =')
    window.update_det_label()
    window.update_step_calculations()


def mask_unmeasured(window):
    """Make missing gates transparent in offline analysis, never alter the file."""
    measured = np.asarray(window.data_measured_mask, dtype=bool)
    for field in PositionScanStore.fields:
        if field == 'measured_mask':
            continue
        attribute = {'raw_intensities': 'data_raw_spectra', 'spectra': 'data_spectra'}.get(field, 'data_' + field)
        values = getattr(window, attribute, None)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if values.shape[:2] == measured.shape:
            expanded = measured.reshape(measured.shape + (1,) * (values.ndim - measured.ndim))
            setattr(window, attribute, np.where(expanded, values, np.nan))


def redraw_gate_data(window, initialize_regions=False):
    if np.any(window.data_measured_mask):
        if initialize_regions:
            # A prior run can have intensity bounds wholly above this file's
            # counts, clipping its whole gate map to zero. Start saved-file
            # analysis with all measured energy/counts, then allow refinement.
            measured = window.data_spectra[window.data_measured_mask]
            low = min(0.0, float(np.nanmin(measured)))
            high = max(low + 1e-9, float(np.nanmax(measured)))
            with QtCore.QSignalBlocker(window.spectrum_v_region):
                window.spectrum_v_region.setRegion((float(window.energy_axis.min()), float(window.energy_axis.max())))
            with QtCore.QSignalBlocker(window.spectrum_h_region):
                window.spectrum_h_region.setRegion((low, high))
        # Establish valid integration regions before computing the gate map,
        # then create slice images before assigning their voltage/energy rects.
        window.reset_plot_scaling(reset_spectrum=True)
        window.update_heatmaps_and_slices()
        window.on_crosshair_moved()
        window.on_tab2_crosshair_moved()
        window.on_tab3_crosshair_moved()
        window.reset_plot_scaling(reset_spectrum=False)
    else:
        for name, item in vars(window).items():
            if name.endswith(('_img', '_curve')) and hasattr(item, 'clear'):
                item.clear()


class WindowSnapshot:
    """The displayed file's state, restored when another file fails to display."""
    state_names = ('store', '_recorded_calibration', 'single_spectra_window', 'selected_position',
                   '_filename_context', 'measurement_started_at', 'measurement_device_modes',
                   'selected_energy', 'point_gatemap_1', 'point_gatemap_2', '_grid_signature',
                   'signal_x', 'signal_y', 'signal_y_raw', 'bg_x', 'bg_y', 'bg_y_raw',
                   'ref_x', 'ref_y', 'ref_y_raw', '_displaying_live_frame', '_live_autoscale', 'plotter_file',
                   'plotter_metadata', 'plotter_changing', 'plotter_read_at', 'scan_x', 'scan_y',
                   'position_counts', 'wavelength_axis', 'data_measured_volt_ch1', 'data_measured_volt_ch2',
                   # Sweeps and maps start with None or empty arrays; restore those too.
                   'data_spectra', 'data_raw_spectra', 'data_power', 'data_measured_mask', 'data_current_ch1',
                   'data_current_ch2', 'data_measured_v1', 'data_measured_v2', 'data_measured_b',
                   'x_coords', 'y_coords', 'wavelengths',
                   # Capacitance and Transport keep a run's plan, settings and data.
                   'plan', 'run_settings', 'data', 'measured_mask', 'loaded_path', 'run_outcome', 'run_message',
                   'balances', 'last_balance')

    def __init__(self, window):
        names = set(self.state_names) | {key for key, value in vars(window).items() if isinstance(value, np.ndarray)}
        self.state = {key: getattr(window, key) for key in names if hasattr(window, key)}
        self.absent = names - self.state.keys()
        self.controls = []
        self.enabled = []
        self.visibility = [(widget, widget.isHidden()) for widget in _measurement_controls(window)]
        widgets = _panel(window).findChildren(QtWidgets.QWidget)
        popup = getattr(window, 'view_control_popup', None)
        if popup is not None:
            widgets += popup.findChildren(QtWidgets.QWidget)
        for widget in widgets:
            self.enabled.append((widget, not widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_ForceDisabled)))
            if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                self.controls.append((widget, widget.value()))
            elif isinstance(widget, QtWidgets.QComboBox):
                self.controls.append((widget, widget.currentIndex()))
            elif isinstance(widget, QtWidgets.QAbstractButton) and widget.isCheckable():
                self.controls.append((widget, widget.isChecked()))
            elif isinstance(widget, (QtWidgets.QLabel, QtWidgets.QLineEdit)) and not isinstance(widget.parent(), QtWidgets.QAbstractSpinBox):
                self.controls.append((widget, widget.text()))
        self.views = [(plot.getViewBox(), plot.getViewBox().getState())
                      for plot in _panel(window).findChildren(pg.PlotWidget)]
        self.regions = [(item, item.getRegion()) for item in vars(window).values()
                        if isinstance(item, pg.LinearRegionItem)]

    def restore(self, window):
        for name, value in self.state.items():
            setattr(window, name, value)
        for name in self.absent:
            if hasattr(window, name):
                delattr(window, name)
        for widget, value in self.controls:
            _set_control(widget, value)
        for widget, enabled in self.enabled:
            widget.setEnabled(enabled)
        for widget, hidden in self.visibility:
            widget.setHidden(hidden)
        for item, region in self.regions:
            with QtCore.QSignalBlocker(item):
                item.setRegion(region)
        if hasattr(window, 'c1_coords'):
            update_gate_labels(window)
            redraw_gate_data(window)
            if hasattr(window, 'position_image'):
                if window.store is not None:
                    window._grid_signature = None
                    window.update_position_map()
                    if window.selected_position is not None:
                        iy, ix = window.selected_position
                        window.position_marker.setData([window.scan_x[ix]], [window.scan_y[iy]])
                else:
                    window.position_image.clear()
                    window.position_grid.clear()
                    window.position_marker.clear()
        elif isinstance(window, BVSweepWindow):
            redraw_gate_data(window)
        elif isinstance(window, PhotoluminescenceMapWindow):
            if window.data_spectra is not None:
                redraw_vector_map(window)
        elif isinstance(window, ScalarMapWindow):
            if window.data_power is not None:
                redraw_scalar_map(window)
        elif isinstance(window, GateMapWindow):
            redraw_gate_map(window)
        else:
            window.replot()
        for view, state in self.views:
            view.setState(state)
        if hasattr(window, 'c1_coords') and np.any(window.data_measured_mask):
            window.update_crosshair_values(self.state['point_gatemap_1'], self.state['point_gatemap_2'],
                                           self.state['selected_energy'])
            window.on_crosshair_moved()
        # Restoring a ViewBox emits auto-range signals; the user's previous
        # live-autoscale choice is authoritative over those programmatic events.
        if '_live_autoscale' in self.state:
            window._live_autoscale = self.state['_live_autoscale']


def _set_data(window, data):
    for name, values in data.items():
        setattr(window, DATA_ATTRIBUTES.get(name, 'data_' + name), values)
    window.data_measured_mask = np.asarray(window.data_measured_mask, dtype=bool)


def _count_range(window):
    """All measured counts, so a previous run's narrower count window cannot clip this file."""
    measured = window.data_spectra[window.data_measured_mask]
    low = min(0.0, float(np.nanmin(measured))) if measured.size else 0.0
    high = max(low + 1e-9, float(np.nanmax(measured))) if measured.size else 1.0
    return low, high


def _first_measured_point(window):
    measured = np.argwhere(window.data_measured_mask)
    iy, ix = measured[0] if len(measured) else (0, 0)
    return float(window.x_coords[ix]), float(window.y_coords[iy])


def _show_bv(window, data):
    """B/V Sweep: its layout is the DGS tab's, with control and field as the two axes."""
    for name, values in {**data['axes'], **data['grids']}.items():
        setattr(window, name, values)
    window.energy_sort_idx = np.arange(len(window.energy_axis))  # Saved in energy order.
    window.single_spectra_window = data['calibration']
    _set_data(window, data['data'])
    shape = (len(window.b_coords), len(window.v_coords))
    for name in ('measured_v1', 'measured_v2', 'measured_b'):
        if name not in data['data']:
            setattr(window, 'data_' + name, np.zeros(shape))
    if 'raw_intensities' not in data['data']:
        window.data_raw_spectra = window.data_spectra[:, :, None, :]
    for prefix, coords in (('v', window.v_coords), ('b', window.b_coords)):
        for suffix, value in (('start', coords[0]), ('end', coords[-1]), ('pts', len(coords))):
            _set_control(getattr(window, f'{prefix}_{suffix}'), value)
    for box, value in zip((window.control_coeff_ch1, window.control_coeff_ch2), data['control']):
        if value is not None:
            _set_control(box, value)
    redraw_gate_data(window, initialize_regions=True)


def _show_map(window, data):
    """Vector Map or Scalar Map. Unmeasured points keep the value 0 the tab saved for them."""
    for name, values in data['axes'].items():
        setattr(window, name, values)
    _set_data(window, data['data'])
    shape = (len(window.y_coords), len(window.x_coords))
    if 'measured_b' not in data['data']:
        window.data_measured_b = np.zeros(shape)
    for combo, key in ((window.x_param_cb, 'x_param'), (window.y_param_cb, 'y_param')):
        index = combo.findText(str(data['params'].get(key, '')))
        if index >= 0:
            _set_control(combo, index)
    if data['kind'] == 'vector_map':
        window.single_spectra_window = data['calibration']
        if 'raw_intensities' not in data['data']:
            window.data_raw_spectra = window.data_spectra[:, :, None, :]
        redraw_vector_map(window, initialize_regions=True)
    else:
        redraw_scalar_map(window)


def redraw_vector_map(window, initialize_regions=False):
    rows, columns, samples = window.data_spectra.shape
    if initialize_regions:
        energies = 1239.84193 / window.wavelengths
        with QtCore.QSignalBlocker(window.integration_region):
            window.integration_region.setRegion((float(energies.min()), float(energies.max())))
        with QtCore.QSignalBlocker(window.integration_h_region):
            window.integration_h_region.setRegion(_count_range(window))
    # Size the images before reset_plot_bounds assigns their rectangles, as a scan start does.
    window.map_img.setImage(np.zeros((rows, columns)), autoLevels=True)
    window.wl_vs_y_img.setImage(np.zeros((rows, samples)), autoLevels=True)
    window.wl_vs_x_img.setImage(np.zeros((columns, samples)), autoLevels=True)
    window.reset_plot_bounds()
    window.update_map_plot()
    x, y = _first_measured_point(window)
    window.update_crosshair_values(x, y)
    window.refresh_slice_plots(x, y)


def redraw_scalar_map(window):
    window.update_map_plot()
    window.reset_plot_bounds()
    window.update_crosshair_values(*_first_measured_point(window))


def redraw_gate_map(window):
    """Capacitance or Transport: redraw the restored run, as its Open does."""
    plan = window.plan
    if plan is None:
        return
    window.reset_view()
    window._plots_dirty = True
    window._refresh_plots_if_dirty()
    for image, bar, grid in ((window.vt_map_img, window.vt_map_bar, plan.vt),
                             (window.vb_map_img, window.vb_map_bar, plan.vb),
                             (window.vs_map_img, window.vs_map_bar, plan.vs)):
        window._set_map(image, bar, grid, plan.x, plan.y)


def apply_dataset(window, data):
    """Populate display data only; preserve the exact recorded physical grids."""
    attrs = data['attrs']
    kind = data['kind']
    if kind == 'spectrum':
        for name, value in vars(data['calibration']).items():
            setattr(window, name, value)
        window._displaying_live_frame = False
        window.replot()
        window.scale_plots()
    elif kind == 'bv':
        _show_bv(window, data)
    elif kind in ('vector_map', 'scalar_map'):
        _show_map(window, data)
    elif kind in ('capacitance', 'transport'):
        # The tab's own Open, minus the file read: that happened in the loading thread.
        window.show_loaded(data['path'], data['loaded'], window.plan_for(data['loaded']))
    else:
        for name, values in {**data['axes'], **data['grids']}.items():
            setattr(window, name, values)
        window.linear_energy_axis = np.linspace(window.energy_axis[0], window.energy_axis[-1], len(window.energy_axis))
        window.energy_sort_idx = np.arange(len(window.energy_axis))
        for axis in (1, 2):
            coords = getattr(window, f'c{axis}_coords')
            _set_control(getattr(window, f'c{axis}_name_edit'), _text(attrs.get(f'c{axis}_name', f'Gate {axis}')))
            for suffix, value in (('start', coords[0]), ('end', coords[-1]), ('pts', len(coords))):
                _set_control(getattr(window, f'c{axis}_{suffix}'), value)
        for key, value in zip('abcd', data['coefficients']):
            _set_control(getattr(window, 'coeff_' + key), value)
        update_gate_labels(window)
        if data['kind'] == 'position_dgs':
            window.scan_x, window.scan_y = data['scan_x'], data['scan_y']
            window.position_counts = data['position_counts']
            window.store = ReadOnlyScanStore(data['path'])
            window.store.provide(*data['position'], data['position_data'])
            window._recorded_calibration = data['calibration']
            window._grid_signature = None
            window.update_position_map()
            window.select_position(*data['position'])
            redraw_gate_data(window, initialize_regions=True)
        else:
            # The analysis calibration belongs to this file, not to whatever
            # Spectrum dataset was last opened in the neighbouring tab.
            window.single_spectra_window = data['calibration']
            for name, values in data['data'].items():
                attribute = {'raw_intensities': 'data_raw_spectra', 'spectra': 'data_spectra'}.get(name, 'data_' + name)
                setattr(window, attribute, values)
            window.data_measured_mask = window.data_measured_mask.astype(bool)
            gate_shape = (len(window.c2_coords), len(window.c1_coords))
            for name in ('measured_b', 'measured_volt_ch1', 'measured_volt_ch2'):
                if name not in data['data']:
                    setattr(window, 'data_' + name, np.full(gate_shape, np.nan))
            if 'raw_intensities' not in data['data']:
                window.data_raw_spectra = window.data_spectra[:, :, None, :]
            mask_unmeasured(window)
            redraw_gate_data(window, initialize_regions=True)
    window.plotter_file = data['path']
    window.plotter_metadata = data['attrs']
    window.plotter_changing = data.get('changing', False)
    window.plotter_read_at = data.get('read_at')
    label = getattr(window, 'status_lbl', None)
    if label is None and kind in ('bv', 'vector_map', 'scalar_map'):
        label = getattr(window, 'lbl_status', None)
    if label is not None:
        label.setText('Saved data: ' + Path(data['path']).name)


# --- The viewer window --------------------------------------------------------------------

class DataPlotterWindow(QtWidgets.QMainWindow):
    """Saved files of every measurement tab of the dashboard, for analysis only.

    It hosts the dashboard's own analysis windows, in plotter mode, on a station of
    simulated devices. Device pollers are stopped and acquisition controls hidden.
    """
    def __init__(self, station, start=None):
        super().__init__()
        from measurements.Dual_gate_sweep_gui import MainWindow as DualGateWindow
        from measurements.single_spectra_gui import SingleSpectraWindow
        self.station = station
        self.worker = None
        self.model = None
        self._loading = False
        self._next_load = None
        self._close_requested = False
        self.setWindowTitle('G15 Data plotter (read-only)')
        self.setMinimumSize(1400, 850)
        self.single_window = SingleSpectraWindow(station, embed=True)
        self.dual_window = DualGateWindow(station, embed=True)
        self.bv_window = BVSweepWindow(station, embed=True)
        self.pl_map_window = PhotoluminescenceMapWindow(station, embed=True)
        self.scalar_map_window = ScalarMapWindow(station, embed=True)
        self.position_dgs_window = PlotterPositionDGSWindow(station, embed=True)
        self.capacitance_window = CapacitanceWindow(station, embed=True)
        self.transport_window = TransportWindow(station, embed=True)
        self.windows = {'spectrum': self.single_window, 'dgs': self.dual_window, 'bv': self.bv_window,
                        'vector_map': self.pl_map_window, 'scalar_map': self.scalar_map_window,
                        'position_dgs': self.position_dgs_window, 'capacitance': self.capacitance_window,
                        'transport': self.transport_window}
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.TabPosition.West)
        for kind, window in self.windows.items():
            window.plotter_mode = True
            # QTabWidget reparents the central widgets away from their QMainWindow wrappers.
            window._panel_widget = window.centralWidget()
            self.tab_widget.addTab(window._panel_widget, TAB_TITLES[kind])
        self.setCentralWidget(self.tab_widget)
        for window in (self.dual_window, self.bv_window, self.pl_map_window, self.position_dgs_window):
            window.single_spectra_window = self.single_window
        self.single_window.set_plotter_mode(True)
        for window in self.windows.values():
            _stop_threads(window)
        for window in self.windows.values():
            # Reclaim the measurement-form space for data analysis. Source
            # selection, crosshairs, current plots and the pixel map stay active.
            for control in _measurement_controls(window):
                control.setEnabled(False)
                control.hide()
        self._build_browser()
        notice = QtWidgets.QLabel('Read-only: files are opened without locks and never written; '
                                  'simulated devices only.  ')
        notice.setStyleSheet('color: #475569;')
        self.statusBar().addPermanentWidget(notice)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        from theme import SCIENTIFIC_THEME_QSS
        self.setStyleSheet(SCIENTIFIC_THEME_QSS)
        from gui_safety import install_parameter_wheel_guard
        install_parameter_wheel_guard(self)
        self._open_browser()
        start = Path(start) if start else REPO_DIR / 'data'
        if start.is_file():
            self.set_folder(start.parent)
            QtCore.QTimer.singleShot(0, lambda: self.load_file(start))
        else:
            self.set_folder(start if start.is_dir() else Path.cwd())
        self.show_current_file()

    def _build_browser(self):
        self.dock = QtWidgets.QDockWidget('Saved measurement files', self)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        self.folder = QtWidgets.QLineEdit()
        self.folder.setReadOnly(True)
        layout.addWidget(self.folder)
        buttons = QtWidgets.QHBoxLayout()
        choose = QtWidgets.QPushButton('Browse folder…')
        choose.clicked.connect(self.choose_folder)
        buttons.addWidget(choose)
        open_button = QtWidgets.QPushButton('Open file…')
        open_button.clicked.connect(self.choose_file)
        buttons.addWidget(open_button)
        layout.addLayout(buttons)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText('Filter filenames (wildcards allowed)')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter_files)
        layout.addWidget(self.search)
        layout.addWidget(QtWidgets.QLabel('Double-click or press Enter to open a measurement.'))
        self.tree = QtWidgets.QTreeView()
        self.tree.setSortingEnabled(True)
        self.tree.doubleClicked.connect(self.open_index)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.open_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Return'), self.tree)
        self.open_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        self.open_shortcut.activated.connect(lambda: self.open_index(self.tree.currentIndex()))
        layout.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton('Fit plots')
        self.fit_button.setToolTip('Fit the current analysis plots to their data without changing integration regions.')
        self.fit_button.clicked.connect(self.fit_plots)
        row.addWidget(self.fit_button)
        self.reload_button = QtWidgets.QPushButton('Reload')
        self.reload_button.setToolTip('Read the displayed file again, for example to show points '
                                      'that a running measurement has added since.')
        self.reload_button.clicked.connect(self.reload)
        self.reload_button.setEnabled(False)
        row.addWidget(self.reload_button)
        layout.addLayout(row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.info = QtWidgets.QLabel()
        self.info.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.info.setWordWrap(True)
        # Long names must not widen the panel at the plots' expense; drag its edge instead.
        self.info.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.info)
        self.dock.setWidget(body)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.resizeDocks([self.dock], [420], QtCore.Qt.Orientation.Horizontal)

    def _open_browser(self):
        self.model = QtGui.QFileSystemModel(self)
        self.model.setNameFilters(['*.h5', '*.hdf5'])
        self.model.setNameFilterDisables(False)
        self.filter_files()
        self.model.setReadOnly(True)
        self.tree.setModel(self.model)
        for column in (1, 2, 3):
            self.tree.hideColumn(column)

    def filter_files(self):
        if self.model is not None:
            query = self.search.text().strip()
            self.model.setNameFilters([f'*{query}*.h5', f'*{query}*.hdf5'])

    def set_folder(self, path):
        self.folder.setText(str(path))
        if self.model is not None:
            self.tree.setRootIndex(self.model.setRootPath(str(path)))

    def choose_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Browse saved measurements', self.folder.text())
        if path:
            self.set_folder(path)

    def choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open saved measurement', self.folder.text(),
                                                        'Measurement data (*.h5 *.hdf5)')
        if path:
            self.load_file(path)

    def open_index(self, index):
        if self.model is not None and index.isValid() and not self.model.isDir(index):
            self.load_file(self.model.filePath(index))

    def current_window(self):
        widget = self.tab_widget.currentWidget()
        return next((window for window in self.windows.values() if _panel(window) is widget), None)

    def busy(self):
        # A finished thread may still have its loaded/failed signal queued for the GUI.
        return self._loading or (self.worker is not None and self.worker.isRunning())

    def load_file(self, path, position=None):
        """Read a file in a worker thread. A file chosen meanwhile is read next (latest wins)."""
        if self._close_requested:
            return
        if self.busy():
            self._next_load = (str(path), position)
            self.info.setText(f'Opening {Path(path).name} once the current file has loaded…')
            return
        self._loading = True
        self.info.setText('Loading ' + Path(path).name + '…')
        if self.worker is not None:
            self.worker.wait()
            self.worker.deleteLater()
        self.worker = DataLoadWorker(str(path), position, self)
        self.worker.loaded.connect(self.display)
        self.worker.failed.connect(self.load_failed)
        self.worker.finished.connect(self._loading_finished)
        self.progress.show()
        self.fit_button.setEnabled(False)
        self.reload_button.setEnabled(False)
        self.worker.start()

    def _loading_finished(self):
        self._loading = False
        self.progress.hide()
        self.fit_button.setEnabled(True)
        self.reload_button.setEnabled(getattr(self.current_window(), 'plotter_file', None) is not None)
        if self._next_load is not None:
            path, position = self._next_load
            self._next_load = None
            self.load_file(path, position)

    def reload(self):
        window = self.current_window()
        path = getattr(window, 'plotter_file', None)
        if path:
            self.load_file(path, getattr(window, 'selected_position', None))

    def display(self, data):
        window = self.windows[data['kind']]
        previous = WindowSnapshot(window)
        try:
            apply_dataset(window, data)
        except Exception as exc:
            previous.restore(window)
            self.load_failed(str(exc))
            return
        self.tab_widget.setCurrentWidget(_panel(window))
        self.show_current_file()

    def fit_plots(self):
        window = self.current_window()
        if window is None or self.busy():
            return
        if hasattr(window, 'scale_plots'):
            window.scale_plots()
        elif hasattr(window, 'autoscale'):
            window.autoscale()
        else:  # The maps: every plot, without touching integration regions.
            for plot in _panel(window).findChildren(pg.PlotWidget):
                plot.enableAutoRange()
        if hasattr(window, 'position_plot'):
            window.position_plot.autoRange()

    def on_tab_changed(self, _index):
        current = self.current_window()
        for window in self.windows.values():
            popup = getattr(window, 'view_control_popup', None)
            if window is not current and popup is not None and popup.isVisible():
                popup.hide()
        self.reload_button.setEnabled(not self.busy() and getattr(current, 'plotter_file', None) is not None)
        self.show_current_file()

    def show_current_file(self):
        window = self.current_window()
        path = getattr(window, 'plotter_file', None)
        attrs = getattr(window, 'plotter_metadata', {})
        details = ''
        if path and hasattr(window, 'c1_coords'):
            details = '\n'.join(f"{getattr(window, f'c{axis}_name_edit').text()}: "
                                f"{getattr(window, f'c{axis}_coords')[0]:g} to "
                                f"{getattr(window, f'c{axis}_coords')[-1]:g} V "
                                f"({len(getattr(window, f'c{axis}_coords'))} points)"
                                for axis in (1, 2)) + '\n'
            if hasattr(window, 'position_counts'):
                measured = int(np.sum(window.position_counts))
                total = window.position_counts.size * len(window.c1_coords) * len(window.c2_coords)
                coverage_label = 'Measured gates across scan'
            else:
                measured = int(np.count_nonzero(window.data_measured_mask))
                total = window.data_measured_mask.size
                coverage_label = 'Measured gates'
            details += f'{coverage_label}: {measured} / {total} ({measured / total:.0%})\n'
            details += f'Spectral samples: {len(window.energy_axis)}\n'
            if hasattr(window, 'position_counts'):
                counts = window.position_counts
                details += f'Scanner positions with data: {np.count_nonzero(counts)} / {counts.size}\n'
        elif path and not hasattr(window, 'set_plotter_mode'):
            details = self._sweep_details(window)
        elif path:
            for prefix, label in (('signal', 'Signal'), ('bg', 'Background'), ('ref', 'Reference')):
                axis = getattr(window, prefix + '_x', None)
                if axis is not None:
                    details += f'{label}: {len(axis)} samples, {np.min(axis):g}–{np.max(axis):g} nm\n'
        exposure = attrs.get('exposure_time', attrs.get('exposure_time_s'))
        if exposure is not None:
            details += f'Recorded exposure: {exposure} s\n'
        if path:
            started = attrs.get('measurement_started_at')
            if started is not None:
                details += f'Measurement started: {_text(started)}\n'
            status = attrs.get('status')
            if status is not None:
                details += f'Recorded scan status: {_text(status)}\n'
                if _text(status) in ('ready', 'running'):
                    details += 'The scan was still running: Reload shows newer positions.\n'
            try:
                modes = json.loads(_text(attrs.get('device_modes', '{}')))
            except (TypeError, ValueError):
                modes = None
            if isinstance(modes, dict) and modes:
                simulated = sorted(str(name) for name, mode in modes.items() if mode == 'simulated')
                details += ('Recorded simulated devices: ' + ', '.join(simulated) if simulated else
                            'Recorded device modes: no simulated devices listed') + '\n'
            else:
                details += 'Recorded device modes: unavailable\n'
            if getattr(window, 'plotter_changing', False):
                details += 'The file changed while it was read (a measurement is writing it).\n'
            read_at = getattr(window, 'plotter_read_at', None)
            if read_at:
                details += f'Read at {read_at}.\n'
        self.info.setText(('Loaded: ' + Path(path).name + '\n' if path else 'Open a saved file to start analysis.\n')
                          + details + 'Read-only analysis: files are never locked or written, '
                          'and no device is contacted.')
        self.info.setToolTip(str(path) if path else '')

    def load_failed(self, message):
        self.info.setText('Could not load file: ' + message)
        QtWidgets.QMessageBox.warning(self, 'Cannot open measurement data', message)

    def closeEvent(self, event):
        if self.busy() or self.position_dgs_window.position_reader_busy():
            # Never block the window on a read: ask the threads to stop, then close.
            self._close_requested = True
            self._next_load = None
            if self.worker is not None:
                self.worker.requestInterruption()
            self.position_dgs_window.cancel_position_reads()
            self.info.setText('Closing when the current read ends…')
            event.ignore()
            QtCore.QTimer.singleShot(100, self.close)
            return
        self.shutdown()
        super().closeEvent(event)

    @staticmethod
    def _sweep_details(window):
        """Axes and coverage of a B/V Sweep, map, Capacitance or Transport file."""
        if isinstance(window, GateMapWindow):
            names = getattr(window.plan, 'names', ('X', 'Y'))
            axes = ((names[0], window.x_coords, 'V'), (names[1], window.y_coords, 'V'))
            mask = window.measured_mask
        elif isinstance(window, BVSweepWindow):
            axes = (('Control', window.v_coords, 'V'), ('Field', window.b_coords, 'T'))
            mask = window.data_measured_mask
        else:
            axes = ((window.x_param_cb.currentText(), window.x_coords, ''),
                    (window.y_param_cb.currentText(), window.y_coords, ''))
            mask = window.data_measured_mask
        lines = [f'{name}: {coords[0]:g} to {coords[-1]:g}{" " + unit if unit else ""} ({len(coords)} points)'
                 for name, coords, unit in axes]
        measured, total = int(np.count_nonzero(mask)), int(np.size(mask))
        lines.append(f'Measured points: {measured} / {total} ({measured / max(total, 1):.0%})')
        if measured < total and not isinstance(window, GateMapWindow):
            lines.append('Unmeasured points show 0, as the tab saved them.')
        spectra = getattr(window, 'data_spectra', None)
        if spectra is not None and not isinstance(window, GateMapWindow):
            lines.append(f'Spectral samples: {np.shape(spectra)[-1]}')
        return '\n'.join(lines) + '\n'

    def shutdown(self):
        """Release threads and the native browser model while Qt and Python are alive."""
        for window in self.windows.values():
            _stop_threads(window)
        if self.model is not None:
            model, self.model = self.model, None
            self.tree.setModel(None)
            # Destroy synchronously in its GUI thread; a deferred delete could
            # remain pending after the final application event loop exits.
            sip.delete(model)
        if self.worker is not None:
            self.worker.wait()
            worker, self.worker = self.worker, None
            sip.delete(worker)
        from gui_safety import uninstall_parameter_wheel_guard
        for window in (*self.windows.values(), self):
            uninstall_parameter_wheel_guard(window)


# --- Program --------------------------------------------------------------------------------

def check_files(paths):
    """--check: read each file as the viewer would and print one line; no window, no devices."""
    status = 0
    for path in paths:
        try:
            data = read_dataset(path)
        except Exception as exc:
            print(f'{path}: cannot be shown: {exc}')
            status = 1
        else:
            print(f"{data['path']}: {describe(data)}")
    return status


def install_error_handler(window):
    """Log unexpected errors and keep the viewer open (PyQt would otherwise end it)."""
    def report(kind, value, trace):
        traceback.print_exception(kind, value, trace)
        try:
            window.statusBar().showMessage(f'Unexpected error: {value}. Details in {LOG_FILE} '
                                           'or the console.', 20000)
        except RuntimeError:
            pass
    sys.excepthook = report


def main(argv=None):
    """Run the Data plotter program; see the module docstring."""
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = 'usage: python data_plotter.py [FILE or FOLDER] | --check FILE...'
    lower_process_priority()
    block_device_access()
    if argv[:1] == ['--check']:
        if len(argv) < 2:
            print(usage, file=sys.stderr)
            return 2
        return check_files(argv[1:])
    if len(argv) > 1 or (argv and argv[0].startswith('-')):
        print(usage, file=sys.stderr)
        return 2
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('G15 Data plotter')
    with tempfile.TemporaryDirectory(prefix='g15_data_plotter_', ignore_cleanup_errors=True) as settings:
        use_private_settings(settings)
        station = simulated_station()
        try:
            window = DataPlotterWindow(station, start=argv[0] if argv else None)
            install_error_handler(window)
            window.show()
            return app.exec()
        finally:
            for component in list(station.components.values()):
                try:
                    component.close()
                except Exception:
                    pass


if __name__ == '__main__':
    # measurements.position_dgs_gui imports this module by name; give it this running copy.
    sys.modules.setdefault('data_plotter', sys.modules[__name__])
    sys.exit(main())
