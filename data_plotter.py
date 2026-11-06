"""Read-only saved-data browsing using the dashboard's existing analysis plots.

File contents never become measurement settings.  Measurement arrays and form
values are retained in memory and restored before returning to measurement.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets, sip

from position_dgs import PositionScanStore


PLOTTER_TABS = {0: 'Spectrum', 1: 'DGS', 7: 'Position DGS'}


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _panel(window):
    widget = window.centralWidget()
    return widget if widget is not None else window._panel_widget


def _measurement_controls(window):
    if hasattr(window, 'c1_coords'):
        return [_panel(window).layout().itemAt(0).widget()]
    return _panel(window).findChildren(QtWidgets.QGroupBox)


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


def read_dataset(path):
    """Validate first; no Qt objects, devices, writes, or metadata execution."""
    path = str(Path(path).resolve())
    with h5py.File(path, 'r') as f:
        result = {'path': path, 'attrs': dict(f.attrs), 'calibration': _calibration(f)}
        if _text(f.attrs.get('schema', '')) == 'PSDGS-1':
            result['tab'] = 7
        elif all(key in f for key in ('c1_coords', 'c2_coords', 'spectra', 'energy_axis')):
            if f['spectra'].ndim != 3:
                raise ValueError('This file is not a two-dimensional dual gate sweep.')
            result['tab'] = 1
        elif any(key in f for key in ('signal', 'background', 'reference')):
            result['tab'] = 0
            return result
        else:
            raise ValueError('Choose a saved Spectrum, DGS, or Position DGS HDF5 file.')

        axes = {name: _axis(f, name) for name in ('c1_coords', 'c2_coords', 'energy_axis')}
        if np.any(axes['energy_axis'] <= 0):
            raise ValueError('Energy coordinates must be positive.')
        # Older DGS exports recorded energy only. Its inverse is the same
        # calibration, rather than a guessed spectrometer centre wavelength.
        axes['wavelength_axis'] = (1239.84193 / axes['energy_axis']
                                   if result['tab'] == 1 and 'wavelength_axis' not in f
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
        if result['tab'] == 7:
            result['scan_x'], result['scan_y'] = _axis(f, 'x_voltages'), _axis(f, 'y_voltages')
            shape = (len(result['scan_y']), len(result['scan_x'])) + gate_shape
            # Only masks are read for the overview; spectra remain on disk and
            # the existing pixel selector loads one DGS at a time.
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
                if result['tab'] == 1 and name in ('raw_intensities', 'measured_b', 'measured_volt_ch1', 'measured_volt_ch2'):
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
            if result['tab'] == 1:
                result['data'][name] = dataset[:]
        return result


class DataLoadWorker(QtCore.QThread):
    loaded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            result = read_dataset(self.path)
            if not self.isInterruptionRequested():
                self.loaded.emit(result)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


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
    """Keep references to measurement arrays, avoiding duplicate full scan cubes."""
    state_names = ('store', '_recorded_calibration', 'single_spectra_window', 'selected_position',
                   '_filename_context', 'measurement_started_at', 'measurement_device_modes',
                   'selected_energy', 'point_gatemap_1', 'point_gatemap_2', '_grid_signature',
                   'signal_x', 'signal_y', 'signal_y_raw', 'bg_x', 'bg_y', 'bg_y_raw',
                   'ref_x', 'ref_y', 'ref_y_raw', '_displaying_live_frame', '_live_autoscale', 'plotter_file',
                   'plotter_metadata', 'scan_x', 'scan_y', 'position_counts', 'wavelength_axis',
                   'data_measured_volt_ch1', 'data_measured_volt_ch2')

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


def apply_dataset(window, data):
    """Populate display data only; preserve the exact recorded physical grids."""
    attrs = data['attrs']
    if data['tab'] == 0:
        for name, value in vars(data['calibration']).items():
            setattr(window, name, value)
        window._displaying_live_frame = False
        window.replot()
        window.scale_plots()
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
        if data['tab'] == 7:
            window.scan_x, window.scan_y = data['scan_x'], data['scan_y']
            window.position_counts = data['position_counts']
            window.store = PositionScanStore(data['path'])
            window._recorded_calibration = data['calibration']
            window._grid_signature = None
            window.update_position_map()
            measured_positions = np.flatnonzero(window.position_counts)
            first = np.unravel_index(measured_positions[0], window.position_counts.shape) if measured_positions.size else (0, 0)
            window.select_position(*first)
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
    window.status_lbl.setText('Saved data: ' + Path(data['path']).name)


class DataPlotter(QtCore.QObject):
    """File browser and data-only adapters, owned by LabMainWindow."""
    def __init__(self, dashboard):
        super().__init__(dashboard)
        self.dashboard = dashboard
        self.windows = {0: dashboard.single_window, 1: dashboard.dual_window, 7: dashboard.position_dgs_window}
        self.snapshots = {}
        self.worker = None
        self._loading = False
        self.dock = QtWidgets.QDockWidget('Saved measurement files', dashboard)
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
        # QFileSystemModel owns a native directory-gatherer thread. Do not
        # start it in Measurement mode, and never leave its destruction to
        # Python interpreter teardown after Qt has begun shutting down.
        self.model = None
        self.tree = QtWidgets.QTreeView()
        self.tree.setSortingEnabled(True)
        self.tree.doubleClicked.connect(self.open_index)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.open_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Return'), self.tree)
        self.open_shortcut.activated.connect(lambda: self.open_index(self.tree.currentIndex()))
        layout.addWidget(self.tree, 1)
        self.fit_button = QtWidgets.QPushButton('Fit plots')
        self.fit_button.setToolTip('Fit the current analysis plots to their data without changing integration regions.')
        self.fit_button.clicked.connect(self.fit_plots)
        layout.addWidget(self.fit_button)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.info = QtWidgets.QLabel('Read-only analysis. Acquisition controls and device polling are off.')
        self.info.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.dock.setWidget(body)
        dashboard.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.dock.hide()
        self.set_folder(Path(__file__).parent / 'data')

    def filter_files(self):
        if self.model is not None:
            query = self.search.text().strip()
            self.model.setNameFilters([f'*{query}*.h5', f'*{query}*.hdf5'])

    def fit_plots(self):
        window = self.windows.get(self.dashboard.tab_widget.currentIndex())
        if self.dashboard.plotter_mode and window is not None and not self.busy():
            window.scale_plots()
            if hasattr(window, 'position_plot'):
                window.position_plot.autoRange()

    def _loading_finished(self):
        self._loading = False
        self.progress.hide()
        self.tree.setEnabled(True)
        self.fit_button.setEnabled(True)
        self.dashboard.mode_slider.setEnabled(True)

    def set_folder(self, path):
        self.folder.setText(str(path))
        if self.model is not None:
            self.tree.setRootIndex(self.model.setRootPath(str(path)))

    def _open_browser(self):
        if self.model is not None:
            return
        self.model = QtGui.QFileSystemModel(self)
        self.model.setNameFilters(['*.h5', '*.hdf5'])
        self.model.setNameFilterDisables(False)
        self.filter_files()
        self.model.setReadOnly(True)
        self.tree.setModel(self.model)
        for column in (1, 2, 3):
            self.tree.hideColumn(column)
        self.set_folder(self.folder.text())

    def shutdown(self):
        """Finish native browser resources while Qt and Python are still alive."""
        if self.busy():
            raise RuntimeError('Wait for file loading to finish before closing the browser.')
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

    def choose_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self.dashboard, 'Browse saved measurements', self.folder.text())
        if path:
            self.set_folder(path)

    def choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.dashboard, 'Open saved measurement', self.folder.text(), 'Measurement data (*.h5 *.hdf5)')
        if path:
            self.load_file(path)

    def open_index(self, index):
        if self.model is not None and index.isValid() and not self.model.isDir(index):
            self.load_file(self.model.filePath(index))

    def busy(self):
        # A finished thread may still have its loaded/failed signal queued for
        # the GUI. Keep measurement mode locked until that result is handled.
        return self._loading or (self.worker is not None and self.worker.isRunning())

    def load_file(self, path):
        if not self.dashboard.plotter_mode or self.busy():
            return
        self._loading = True
        self.info.setText('Loading ' + Path(path).name + '…')
        if self.worker is not None:
            self.worker.wait()
            self.worker.deleteLater()
        self.worker = DataLoadWorker(path, self)
        self.worker.loaded.connect(self.display)
        self.worker.failed.connect(self.load_failed)
        self.worker.finished.connect(self._loading_finished)
        self.progress.show()
        self.tree.setEnabled(False)
        self.fit_button.setEnabled(False)
        self.dashboard.mode_slider.setEnabled(False)
        self.worker.start()

    def display(self, data):
        if not self.dashboard.plotter_mode:
            return
        window = self.windows[data['tab']]
        previous = WindowSnapshot(window)
        try:
            apply_dataset(window, data)
        except Exception as exc:
            previous.restore(window)
            self.load_failed(str(exc))
            return
        self.dashboard.tab_widget.setCurrentIndex(data['tab'])
        self.show_current_file()

    def show_current_file(self):
        window = self.windows.get(self.dashboard.tab_widget.currentIndex())
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
        self.info.setText(('Loaded: ' + str(path) + '\n' if path else 'Open a saved file to start analysis.\n') + details +
                          'Read-only analysis. Acquisition controls and device polling are off.')

    def load_failed(self, message):
        self.info.setText('Could not load file: ' + message)
        QtWidgets.QMessageBox.warning(self.dashboard, 'Cannot open measurement data', message)

    def enter(self):
        self._open_browser()
        self.snapshots = {index: WindowSnapshot(window) for index, window in self.windows.items()}
        for index, window in self.windows.items():
            if index == 0 and hasattr(window, 'set_plotter_mode'):
                window.set_plotter_mode(True)
            # Reclaim the measurement-form space for data analysis. Source
            # selection, crosshairs, current plots and the pixel map stay active.
            for control in _measurement_controls(window):
                control.setEnabled(False)
                control.hide()
        self.dock.show()
        self.show_current_file()

    def leave(self):
        for index, window in self.windows.items():
            if index not in self.snapshots:
                continue
            if index == 0 and hasattr(window, 'set_plotter_mode'):
                window.set_plotter_mode(False)
            self.snapshots[index].restore(window)
        self.snapshots.clear()
        self.dock.hide()
        self.shutdown()
