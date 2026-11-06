"""Position scan with a complete dual-gate sweep at each ANC300 X/Y offset."""
import json
import os
from pathlib import Path
import uuid
from datetime import datetime
from types import SimpleNamespace

import h5py
import numpy as np
import pyqtgraph as pg
from pyvisa.errors import InvalidSession
from PyQt6 import QtCore, QtWidgets

from measurements.Dual_gate_sweep_gui import MainWindow, transform_c_to_v
from measurements.utils import configure_plot_axes
from measurement_data_paths import dated_measurement_filename, dated_measurement_path
from measurement_data_paths import measurement_data_dir, measurement_start_text
from measurement_filenames import dgs_filename
from position_dgs import (SCANNER_MAX_VOLTAGE, MAX_SCAN_POSITIONS, PositionScanStore,
                          PositionSweepWorker, scan_axis, scanner_state, read_scanner_offsets)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PositionDGSWindow(MainWindow):
    editable_current_compliance = False

    def __init__(self, station, embed=False):
        self.store = None
        self._recorded_calibration = None
        self._scan_origin_loaded = False
        self.selected_position = None
        self._pending_scan = None
        super().__init__(station, embed=embed)
        self.setWindowTitle('Position Dual-Gate Sweep')
        self.live_status_worker.pause()
        self.scan_box = QtWidgets.QGroupBox('Position scan · ANC300 X/Y')
        layout = QtWidgets.QGridLayout(self.scan_box)
        self.scan_inputs = {}
        for col, text in enumerate(('Axis', 'Start (V)', 'Size / span (V)', 'Step (V)')):
            layout.addWidget(QtWidgets.QLabel(text), 0, col)
        for row, axis in enumerate(('x', 'y'), 1):
            layout.addWidget(QtWidgets.QLabel(axis.upper()), row, 0)
            for col, (name, default) in enumerate((('start', 0), ('size', 0.1), ('step', 0.1)), 1):
                box = QtWidgets.QDoubleSpinBox()
                box.setDecimals(4)
                # Permit invalid requests to be entered, then reject explicitly;
                # never silently clamp a requested scan at the safety limit.
                box.setRange(-1000 if name == 'start' else 0.0001 if name == 'step' else 0, 1000)
                box.setValue(default)
                box.setKeyboardTracking(False)
                box.setToolTip({
                    'start': 'First scanner offset to measure. Reading offsets fills this field without moving the scanner.',
                    'size': 'Total voltage span: end = start + size. Zero keeps this axis fixed.',
                    'step': 'Voltage spacing between measured positions. Example: size 0.2 V and step 0.1 V gives 3 positions, including both endpoints.',
                }[name])
                self.scan_inputs[axis + '_' + name] = box
                layout.addWidget(box, row, col)
        for row, (key, title, default) in enumerate((
                ('move_rate', 'Forward / Y rate (V/s)', 0.5),
                ('return_rate', 'Slow X return (V/s)', 0.2),
                ('settle', 'Position settling (s)', 0.2)), 3):
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(3)
            box.setRange(0 if key == 'settle' else 0.001, 10 if key == 'settle' else 1)
            box.setValue(default)
            self.scan_inputs[key] = box
            layout.addWidget(QtWidgets.QLabel(title), row, 0, 1, 2)
            layout.addWidget(box, row, 2, 1, 2)
        self.scan_inputs['settle'].setToolTip('Wait after reaching each scanner position, before starting its complete dual-gate sweep. Gate settling is set separately below.')
        for key in ('move_rate', 'return_rate'):
            self.scan_inputs[key].setToolTip('Also capped by the scanner safety rate saved in the Dashboard.')
        self.serpentine_box.setToolTip('Applies to the gate sweep within each scanner pixel. Scanner lines always run forward in X, return X slowly, then step Y.')
        self.read_origin_btn = QtWidgets.QPushButton('Use current scanner offsets')
        self.read_origin_btn.setToolTip('Read the connected ANC300 X/Y offsets into the Start fields. Does not move either axis or change Size or Step.')
        self.read_origin_btn.clicked.connect(self.read_scan_origin)
        layout.addWidget(self.read_origin_btn, 6, 0, 1, 4)
        self.scan_info = QtWidgets.QLabel(f'Limit: 0–{SCANNER_MAX_VOLTAGE:g} V. Read current offsets before starting.')
        self.scan_info.setWordWrap(True)
        layout.addWidget(self.scan_info, 7, 0, 1, 4)
        self.load_scan_btn = QtWidgets.QPushButton('Open saved position scan…')
        self.load_scan_btn.clicked.connect(self.open_scan)
        layout.addWidget(self.load_scan_btn, 8, 0, 1, 4)

        scroll = self.centralWidget().layout().itemAt(0).widget()
        scroll.widget().layout().insertWidget(0, self.scan_box)

        # Separate small position box, leaving the existing DGS plots intact.
        right_layout = self.tab1.layout()
        self.position_plot = pg.PlotWidget(title='Scan completion · click a pixel to load its DGS')
        self.position_plot.setFixedHeight(180)
        self.position_plot.setLabel('bottom', 'Scanner X offset', units='V')
        self.position_plot.setLabel('left', 'Scanner Y offset', units='V')
        configure_plot_axes(self.position_plot)
        self.position_image = pg.ImageItem(axisOrder='row-major')
        self.position_image.setColorMap(pg.colormap.get('viridis'))
        self.position_plot.addItem(self.position_image)
        self.position_grid = pg.PlotCurveItem(pen=pg.mkPen('#64748b', width=1), connect='finite')
        self.position_plot.addItem(self.position_grid)
        self.position_plot.getViewBox().setAspectLocked(True)
        self.position_marker = pg.ScatterPlotItem(size=14, pen=pg.mkPen('r', width=2), brush=None)
        self.position_plot.addItem(self.position_marker)
        # The DGS optical tab uses a grid; add the position selector below it.
        right_layout.addWidget(self.position_plot, right_layout.rowCount(), 0, 1, right_layout.columnCount())
        self.position_plot.scene().sigMouseClicked.connect(self.position_clicked)
        self.follow_position = QtWidgets.QCheckBox('Follow current scan pixel')
        self.follow_position.setChecked(True)
        layout.addWidget(self.follow_position, 9, 0, 1, 4)
        self.selected_label = QtWidgets.QLabel('No scan data yet')
        self.selected_label.setWordWrap(True)
        layout.addWidget(self.selected_label, 10, 0, 1, 4)
        self.scan_preview = QtWidgets.QLabel()
        self.scan_preview.setWordWrap(True)
        layout.addWidget(self.scan_preview, 11, 0, 1, 4)
        for key, box in self.scan_inputs.items():
            if key.startswith(('x_', 'y_')):
                box.valueChanged.connect(self.update_scan_preview)
        self.update_scan_preview()
        self.reset_btn.setText('Stop scan')
        self.begin_btn.setText('Begin position scan')
        self.status_lbl.setWordWrap(True)
        self.lbl_expected_end.setWordWrap(True)

    def save_storage(self):
        # This independent panel must not overwrite the ordinary DGS settings.
        pass

    def reload_gui_from_storage(self):
        if getattr(self, "plotter_mode", False):
            return
        # Retain this panel's gate settings and displayed data across tab changes.
        if hasattr(self, 'scan_inputs') and not self._scan_origin_loaded and not self.is_running:
            self.read_scan_origin()

    def read_scan_origin(self):
        if getattr(self, "plotter_mode", False):
            return
        if self.is_running:
            return
        try:
            x, y = read_scanner_offsets(self.station.piezo300)
            self.scan_inputs['x_start'].setValue(x)
            self.scan_inputs['y_start'].setValue(y)
            self._scan_origin_loaded = True
            mode = 'SIMULATED' if self.station.piezo300.simulated else 'Physical'
            read_time = datetime.now().strftime('%H:%M:%S')
            self.scan_info.setText(f'{mode} ANC300 offsets copied to Start at {read_time}: X={x:g} V, Y={y:g} V. No movement commanded.')
            self.scan_info.setStyleSheet('color: #7c3aed;' if mode == 'SIMULATED' else 'color: #047857;')
            if mode == 'SIMULATED':
                self.scan_info.setText(self.scan_info.text() + ' To read physical offsets, reconnect ANC300 in Devices / Errors.')
        except Exception as exc:
            self._scan_origin_loaded = False
            if isinstance(exc, InvalidSession) or 'invalid session handle' in str(exc).lower():
                self.scan_info.setText('ANC300 connection is closed. In Devices / Errors, use Connect / Retry for Attocube ANC300 while idle. Start values were not updated; no movement was commanded.')
            else:
                self.scan_info.setText(f'Scanner readback unavailable: {exc}. Start values were not updated. Check ANC300 in Devices / Errors, then retry.')
            self.scan_info.setStyleSheet('color: #b91c1c;')
        self.update_scan_preview()

    def update_scan_preview(self):
        """Preview geometry without asking any device for a value."""
        try:
            parts, total = [], 1
            for axis in ('x', 'y'):
                coords = scan_axis(*(self.scan_inputs[axis + '_' + name].value() for name in ('start', 'size', 'step')))
                parts.append(f'{axis.upper()}: {coords[0]:g} → {coords[-1]:g} V ({len(coords)} positions)')
                total *= len(coords)
            if total > MAX_SCAN_POSITIONS:
                raise ValueError(f'Too many positions (maximum {MAX_SCAN_POSITIONS:,}). Increase Step.')
            self.scan_preview.setText('\n'.join(parts) + f'\n{total} positions total; one complete DGS at each. Limit: 0–{SCANNER_MAX_VOLTAGE:g} V.'
                                      '\nFinish DGS → next X. End of line → slow X return → step Y.')
            self.scan_preview.setStyleSheet('color: #475569;')
        except ValueError as exc:
            self.scan_preview.setText(f'Scan settings invalid: {exc}')
            self.scan_preview.setStyleSheet('color: #b91c1c;')

    def begin_sweep(self):
        if getattr(self, "plotter_mode", False):
            return
        if self.ramp_worker is not None and self.ramp_worker.isRunning():
            return
        if self.is_running:
            if self.is_paused: self.pause_sweep()
            return
        dashboard = getattr(self, 'dashboard', None)
        if (self.ramp_worker and self.ramp_worker.isRunning()) or (dashboard and dashboard.measurement_active()):
            QtWidgets.QMessageBox.warning(self, 'Device busy', 'Finish the active measurement or movement before starting a position scan.')
            return
        try:
            if not self._scan_origin_loaded:
                raise ValueError('Read the current scanner offsets before starting, then choose the scan origin and size.')
            scan = {key: box.value() for key, box in self.scan_inputs.items()}
            for axis in ('x', 'y'):
                scan[axis] = scan_axis(scan[axis + '_start'], scan[axis + '_size'], scan[axis + '_step'])
            if len(scan['x']) * len(scan['y']) > MAX_SCAN_POSITIONS:
                raise ValueError(f'Scan has too many position pixels (maximum {MAX_SCAN_POSITIONS:,}). Increase the step size.')
            if scan['return_rate'] > scan['move_rate']:
                raise ValueError('The slow return rate must not exceed the forward rate.')
            for value in np.concatenate((scan['x'], scan['y'])):
                self.station.piezo300.scanner_settings.validate(value)
            scanner_state(self.station.piezo300)
            transform_c_to_v(0, 0, self.coeff_a.value(), self.coeff_b.value(), self.coeff_c.value(), self.coeff_d.value())
            if self.ch1_mode.currentText() != 'VOLT' or self.ch2_mode.currentText() != 'VOLT':
                raise ValueError('Dual-gate position scans require both Keysight source modes to be VOLT.')
            self._pending_scan = scan
            super().begin_sweep()
        except Exception as exc:
            self.is_running = False
            self.station.live_status_paused = False
            self.set_inputs_enabled(True)
            self.begin_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.status_lbl.setText(f'Scan blocked: {exc}')
            QtWidgets.QMessageBox.critical(self, 'Position scan blocked', str(exc))

    def create_sweep_worker(self, params):
        scan = self._pending_scan
        self.scan_x, self.scan_y = scan['x'], scan['y']
        self.position_counts = np.zeros((len(self.scan_y), len(self.scan_x)))
        self.selected_position = None
        directory = measurement_data_dir(parent_dir, self.measurement_started_at, autosave=True)
        path = Path(directory) / dated_measurement_filename(('autosave_PSDGS_' + self.measurement_started_at.strftime('%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.h5'), self.measurement_started_at)
        store = PositionScanStore(path)
        metadata = {
            'measurement_started_at': measurement_start_text(self.measurement_started_at),
            'device_modes': self.measurement_device_modes,
            'exposure_time': params['exposure_time'], 'num_acquisitions': params['num_spectra'],
            'c1_name': self.c1_name_edit.text(), 'c2_name': self.c2_name_edit.text(),
            'filename_central_wavelength_nm': self.filename_wavelength_box.value(),
            'filename_wavelength_source': 'manual',
            'initial_field_T': self._filename_context['field'],
            'gate_coefficients': [params[k] for k in ('a', 'b', 'c', 'd')],
            'gate_sweep_mode': params['sweep_mode'], 'gate_serpentine': params['serpentine'],
            'gate_settling_s': params['settling_time'],
            'scanner_move_rate_V_s': scan['move_rate'], 'scanner_return_rate_V_s': scan['return_rate'],
            'position_settling_s': scan['settle'],
        }
        storage = Path(parent_dir) / 'value_storage'
        if storage.exists(): metadata['value_storage'] = storage.read_text(encoding='utf-8')
        def calibration(prefix):
            window = self.single_spectra_window
            x, y = getattr(window, prefix + '_x', None), getattr(window, prefix + '_y', None)
            if x is None or y is None: return None
            raw = getattr(window, prefix + '_y_raw', None)
            return {'wavelength': x, 'intensity': y, 'raw_intensities': np.asarray([y]) if raw is None else raw}
        store.create(x=self.scan_x, y=self.scan_y, c1=self.c1_coords, c2=self.c2_coords,
                     wavelengths=params['wl_axis'], frames=params['num_spectra'],
                     v1=self.v1_grid, v2=self.v2_grid, metadata=metadata,
                     background=calibration('bg'), reference=calibration('ref'))
        self.store = store
        self.load_recorded_calibration()
        worker = PositionSweepWorker(self.station, params, scan, store)
        worker.position_started.connect(self.on_position_started)
        worker.position_point.connect(self.on_position_point)
        worker.position_finished.connect(self.on_position_finished)
        self.update_position_map()
        return worker

    def set_inputs_enabled(self, enabled):
        super().set_inputs_enabled(enabled)
        self.c1_name_edit.setEnabled(enabled)
        self.c2_name_edit.setEnabled(enabled)
        if hasattr(self, 'scan_inputs'):
            for box in self.scan_inputs.values(): box.setEnabled(enabled)
            self.read_origin_btn.setEnabled(enabled)
            self.load_scan_btn.setEnabled(enabled)
        self.filename_wavelength_box.setEnabled(enabled)

    def on_position_started(self, iy, ix):
        if self.follow_position.isChecked() or self.selected_position is None:
            self.select_position(iy, ix)

    def on_position_point(self, iy, ix, point):
        self.position_counts[iy, ix] += 1
        self.update_position_map()
        if self.selected_position == (iy, ix):
            self.on_point_measured(*point)
            self.update_selected_label()

    def on_position_finished(self, iy, ix):
        if self.selected_position == (iy, ix): self.select_position(iy, ix)

    def on_progress_updated(self, current, total):
        self.progress_bar.setValue(int(100 * current / total))
        self.lbl_expected_end.setText(f'{current}/{total} gate points across {len(self.scan_x) * len(self.scan_y)} positions')

    def update_position_map(self):
        self.position_image.setImage(self.position_counts, levels=(0, max(1, len(self.c1_coords) * len(self.c2_coords))))
        dx = self.scan_x[1] - self.scan_x[0] if len(self.scan_x) > 1 else self.scan_inputs['x_step'].value()
        dy = self.scan_y[1] - self.scan_y[0] if len(self.scan_y) > 1 else self.scan_inputs['y_step'].value()
        self.position_image.setRect(QtCore.QRectF(self.scan_x[0] - dx/2, self.scan_y[0] - dy/2, len(self.scan_x)*dx, len(self.scan_y)*dy))
        signature = (id(self.scan_x), id(self.scan_y), dx, dy)
        if signature != getattr(self, '_grid_signature', None):
            self._grid_signature = signature
            xe = np.r_[self.scan_x - dx/2, self.scan_x[-1] + dx/2]
            ye = np.r_[self.scan_y - dy/2, self.scan_y[-1] + dy/2]
            gx, gy = [], []
            for edge in xe:
                gx.extend((edge, edge, np.nan)); gy.extend((ye[0], ye[-1], np.nan))
            for edge in ye:
                gx.extend((xe[0], xe[-1], np.nan)); gy.extend((edge, edge, np.nan))
            self.position_grid.setData(gx, gy)
            self.position_plot.autoRange()

    def position_clicked(self, event):
        if self.store is None or event.button() != QtCore.Qt.MouseButton.LeftButton: return
        if not self.position_plot.getViewBox().sceneBoundingRect().contains(event.scenePos()): return
        point = self.position_plot.getViewBox().mapSceneToView(event.scenePos())
        ix, iy = int(np.argmin(abs(self.scan_x - point.x()))), int(np.argmin(abs(self.scan_y - point.y())))
        self.follow_position.setChecked(False)
        self.select_position(iy, ix)

    def select_position(self, iy, ix):
        if self.store is None: return
        try:
            data = self.store.read_position(iy, ix)
        except Exception as exc:
            self.status_lbl.setText(f'Cannot read selected position: {exc}')
            return
        names = {'raw_intensities': 'data_raw_spectra', 'spectra': 'data_spectra'}
        for name, values in data.items():
            setattr(self, names.get(name, 'data_' + name), np.nan_to_num(values))
        if self.plotter_mode:
            from data_plotter import mask_unmeasured
            mask_unmeasured(self)
        self.selected_position = (iy, ix)
        self.position_marker.setData([self.scan_x[ix]], [self.scan_y[iy]])
        self.update_selected_label()
        # Clear stale plots even if the selected pixel has no measured gates yet.
        if not np.any(self.data_measured_mask):
            for img in (self.gatemap_img, self.vslice_img, self.hslice_img, self.t2_heatmap_img, self.t3_heatmap_img): img.clear()
            self.spectrum_curve.clear()
        elif self.plotter_mode:
            from data_plotter import redraw_gate_data
            redraw_gate_data(self)
        else:
            self.update_heatmaps_and_slices()
            self.reset_plot_scaling(reset_spectrum=True)
            self.on_crosshair_moved()
            self.on_tab2_crosshair_moved()
            self.on_tab3_crosshair_moved()

    def update_selected_label(self):
        iy, ix = self.selected_position
        measured = np.count_nonzero(self.data_measured_mask)
        total = len(self.c1_coords) * len(self.c2_coords)
        self.selected_label.setText(f'Displayed DGS at scanner X={self.scan_x[ix]:g} V, Y={self.scan_y[iy]:g} V · {measured}/{total} gate points')

    def pause_sweep(self):
        if not self.is_running: return
        self.is_paused = not self.is_paused
        if self.is_paused: self.sweep_worker.pause()
        else: self.sweep_worker.resume()
        self.pause_btn.setText('Resume' if self.is_paused else 'Interrupt')
        self.status_lbl.setText('Pause requested; the current acquisition may finish.' if self.is_paused else 'Status: Scanning...')
        # Keep all manual movement and competing polling locked while paused.

    def reset_sweep(self):
        if self.is_running:
            self.sweep_worker.stop()
            self.pause_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.status_lbl.setText('Stopping after the current operation; no return movement will be commanded.')

    def on_sweep_finished(self):
        worker = self.sweep_worker
        self.is_running = self.is_paused = False
        self.station.live_status_paused = False
        self.begin_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText('Interrupt')
        self.reset_btn.setEnabled(True)
        self.save_btn.setEnabled(self.store is not None)
        self.set_inputs_enabled(True)
        if self.selected_position is not None: self.select_position(*self.selected_position)
        state = 'complete' if worker.completed else 'interrupted'
        self.status_lbl.setText(f'Position scan {state}. All acquired points saved automatically.')
        self.status_lbl.setToolTip(self.store.path)
        if worker.failure:
            self.status_lbl.setText(f'Position scan error: {worker.failure}. Partial data retained.')
            QtWidgets.QMessageBox.critical(self, 'Position scan error', worker.failure)

    def trigger_autosave(self, was_interrupted=False):
        if getattr(self, "plotter_mode", False):
            return
        # Acquisition streams directly into the recovery file after each point.
        pass

    def default_sweep_filename(self):
        return dgs_filename(**self._filename_context,
                            wavelength=self.filename_wavelength_box.value(),
                            positions=(self.scan_x, self.scan_y))

    def save_sweep_data(self, file_path=None):
        if getattr(self, "plotter_mode", False):
            return
        if self.store is None: return
        if isinstance(file_path, bool): file_path = None
        if file_path is None:
            directory = measurement_data_dir(parent_dir, getattr(self, 'measurement_started_at', None))
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save entire position scan', str(Path(directory) / dated_measurement_filename(self.default_sweep_filename(), getattr(self, "measurement_started_at", None))), 'HDF5 (*.h5)')
            if not file_path: return
            file_path = dated_measurement_path(file_path, getattr(self, "measurement_started_at", None))
        try:
            self.store.copy_to(file_path)
            with self.store.lock, h5py.File(file_path, 'r+') as f:
                f.attrs['filename_central_wavelength_nm'] = self.filename_wavelength_box.value()
            self.status_lbl.setText(f'Saved all positions: {Path(file_path).name}')
            self.status_lbl.setToolTip(str(file_path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Save error', str(exc))

    def open_scan(self, file_path=None):
        if self.is_running: return
        if not file_path or isinstance(file_path, bool):
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open position scan', measurement_data_dir(parent_dir), 'HDF5 (*.h5)')
            if not file_path: return
        try:
            with h5py.File(file_path, 'r') as f:
                if f.attrs.get('schema') != 'PSDGS-1': raise ValueError('This file is not a PSDGS dataset.')
                for attr, key in [('scan_x', 'x_voltages'), ('scan_y', 'y_voltages'),
                                  ('c1_coords', 'c1_coords'), ('c2_coords', 'c2_coords'),
                                  ('v1_grid', 'v1_grid'), ('v2_grid', 'v2_grid'),
                                  ('energy_axis', 'energy_axis'), ('wavelength_axis', 'wavelength_axis')]:
                    setattr(self, attr, f[key][:])
                self.linear_energy_axis = np.linspace(self.energy_axis[0], self.energy_axis[-1], len(self.energy_axis))
                self.position_counts = np.zeros((len(self.scan_y), len(self.scan_x)))
                for iy in range(len(self.scan_y)):
                    for ix in range(len(self.scan_x)):
                        self.position_counts[iy, ix] = np.count_nonzero(f['measured_mask'][iy, ix])
                self._filename_context = {'exposure': f.attrs['exposure_time'], 'acquisitions': f.attrs['num_acquisitions'],
                    'field': f.attrs.get('initial_field_T', 0),
                    'axes': [(f.attrs['c1_name'], self.c1_coords.copy()), (f.attrs['c2_name'], self.c2_coords.copy())]}
                self.filename_wavelength_box.setValue(f.attrs.get('filename_central_wavelength_nm', 1090))
                self.measurement_started_at = datetime.fromisoformat(f.attrs['measurement_started_at'])
                for edit, name in ((self.c1_name_edit, f.attrs['c1_name']), (self.c2_name_edit, f.attrs['c2_name'])):
                    with QtCore.QSignalBlocker(edit): edit.setText(name)
                for axis, coords in ((1, self.c1_coords), (2, self.c2_coords)):
                    for suffix, value in [('start', coords[0]), ('end', coords[-1]), ('pts', len(coords))]:
                        box = getattr(self, f'c{axis}_{suffix}')
                        with QtCore.QSignalBlocker(box): box.setValue(value)
                self.update_variable_names()
                # Restore the recorded transform after the normal name handler,
                # which can otherwise force physical-channel names to identity.
                for key, value in zip(('a', 'b', 'c', 'd'), json.loads(f.attrs['gate_coefficients'])):
                    box = getattr(self, 'coeff_' + key)
                    with QtCore.QSignalBlocker(box): box.setValue(value)
                self.update_coordinates_from_ui()
            self.store = PositionScanStore(file_path)
            self.load_recorded_calibration()
            self.update_position_map()
            self.select_position(0, 0)
            self.save_btn.setEnabled(True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, 'Open scan error', str(exc))

    def load_recorded_calibration(self):
        recorded = SimpleNamespace()
        with self.store.lock, h5py.File(self.store.path, 'r') as f:
            for group, prefix in [('background', 'bg'), ('reference', 'ref')]:
                for suffix, key in [('x', 'wavelength'), ('y', 'intensity')]:
                    setattr(recorded, prefix + '_' + suffix, f[group][key][:] if group in f else None)
        self._recorded_calibration = recorded

    def get_processed_spectra(self):
        # Display recorded calibration even if Single Spectra has since changed.
        live = self.single_spectra_window
        if self._recorded_calibration is not None:
            self.single_spectra_window = self._recorded_calibration
        try:
            return super().get_processed_spectra()
        finally:
            self.single_spectra_window = live
