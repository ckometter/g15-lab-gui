"""Background scanner ramps and counted stepper moves."""
from PyQt6 import QtCore, QtWidgets

from scanner_safety import ScannerSettings, move_scanner, wait_for_motion

STEPPER_AXES = {1: 'X', 2: 'Y', 3: 'Z'}
# Counted stepper moves are sent as bursts: one "stepu/stepd <axis> <n>" command makes n steps at the
# controller's frequency. A burst lasts at most BURST_S, so Stop or releasing an arrow takes effect
# within that time, and the step count stays exact. (One command per single step was ~8x slower than
# the front panel: every step also needs a mode command and frequency/amplitude readbacks.)
BURST_S = 0.25
BURST_MARGIN_S = 0.02    # extra wait so a burst has finished before the next command
ERROR_STYLE = 'color: #b91c1c; font-weight: bold;'


class ScannerMotionWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, float)
    error = QtCore.pyqtSignal(str)

    def __init__(self, controller, targets, relative=False, parent=None):
        super().__init__(parent)
        self.controller, self.targets, self.relative = controller, targets, relative
        self.completed = False
        self.failure = ''
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True
        self.requestInterruption()

    def run(self):
        try:
            self.completed = move_scanner(self.controller, self.targets, relative=self.relative,
                keep_running=lambda: not self.stop_requested, progress=self.progress.emit)
        except Exception as exc:
            self.failure = str(exc)
            self.error.emit(str(exc))


class StepperMotionWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, controller, targets=None, jog=None, relative=False, parent=None):
        super().__init__(parent)
        self.controller, self.targets = controller, targets
        self.jog, self.relative = jog, relative
        self.completed = False
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True
        self.requestInterruption()

    def run(self):
        controller = self.controller
        if controller is None:
            self.error.emit('Stepper is not connected.')
            return
        if not controller.motion_lock.acquire(blocking=False):
            self.error.emit('Stepper is already moving.')
            return
        axis = None
        try:
            # Check every axis before any of them moves. Steps still to make per axis; None = until stopped.
            axes = [self.jog[0]] if self.jog else list(self.targets)
            remaining, frequency = {}, {}
            for axis in axes:
                frequency[axis] = controller.validate_drive(axis)
                if self.jog:
                    remaining[axis] = None
                    continue
                target = self.targets[axis]
                if type(target) is not int:
                    raise ValueError('Stepper coordinates must be integer step counts.')
                if self.relative:
                    remaining[axis] = target  # arrows and relative moves need no known coordinate
                else:
                    current = controller.pseudo_position(axis)
                    if current is None:
                        raise RuntimeError('Stepper coordinate is unknown. Set zero before an absolute Goto.')
                    remaining[axis] = target - current
            for axis in axes:
                while not self.stop_requested and remaining[axis] != 0:
                    left = remaining[axis]
                    direction = self.jog[1] if left is None else (1 if left > 0 else -1)
                    count = max(1, int(frequency[axis] * BURST_S))
                    if left is not None:
                        count = min(count, abs(left))
                    # step() reads back and checks the settings again: the front panel may have changed them.
                    frequency[axis] = controller.step(axis, direction * count)
                    if left is not None:
                        remaining[axis] = left - direction * count
                    self.progress.emit()
                    # Let the controller finish this burst before the next command or the end of the move.
                    wait_for_motion(count / frequency[axis] + BURST_MARGIN_S, lambda: True)
                if self.stop_requested:
                    return
            self.completed = True
        except Exception as exc:
            self.error.emit(f'{STEPPER_AXES.get(axis, "Stepper")}: {exc}' if axis else str(exc))
        finally:
            controller.motion_lock.release()


class PositionMotionControls:
    def scanner_moving(self):
        return getattr(self, '_scanner_busy', False)

    def stepper_moving(self):
        return getattr(self, '_stepper_busy', False)

    def motion_active(self):
        return self.scanner_moving() or self.stepper_moving()

    def build_scanner_motion_ui(self, layout):
        self.scanner_motion_controls = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self.scanner_motion_controls)
        grid.setContentsMargins(0, 0, 0, 0)
        self.scan_targets = {}
        for column, axis in enumerate(('x', 'y')):
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(3)
            # Explicit validation on submit, so an invalid target is not clamped.
            box.setRange(-10000, 10000)
            box.setSuffix(' V')
            box.setKeyboardTracking(False)
            self.scan_targets[axis] = box
            grid.addWidget(QtWidgets.QLabel(f'{axis.upper()} target / delta'), 0, column)
            grid.addWidget(box, 1, column)
        self.setscan_btn = QtWidgets.QPushButton('Setscan X, Y (absolute V)')
        self.setscan_btn.clicked.connect(lambda: self.start_scanner_motion(
            {i: self.scan_targets[a].value() for i, a in enumerate(('x', 'y'), 1)}))
        self.scanner_goto_btn = QtWidgets.QPushButton('Goto relative X, Y (delta V)')
        self.scanner_goto_btn.clicked.connect(lambda: self.start_scanner_motion(
            {i: self.scan_targets[a].value() for i, a in enumerate(('x', 'y'), 1)}, relative=True))
        grid.addWidget(self.setscan_btn, 2, 0)
        grid.addWidget(self.scanner_goto_btn, 2, 1)
        self.scan_command = QtWidgets.QLineEdit()
        self.scan_command.setPlaceholderText('setscan 10, 20  or  goto -0.1, 0.2  (V)')
        self.scan_command.returnPressed.connect(self.execute_scan_command)
        grid.addWidget(self.scan_command, 3, 0, 1, 2)
        layout.addWidget(self.scanner_motion_controls)
        self.scanner_stop_btn = QtWidgets.QPushButton('Stop scanner ramp')
        self.scanner_stop_btn.setEnabled(False)
        self.scanner_stop_btn.clicked.connect(lambda: self.scanner_worker.stop() if self.scanner_worker else None)
        layout.addWidget(self.scanner_stop_btn)
        self.scanner_settings_group = QtWidgets.QGroupBox('Scanner safety settings (all axes and scans)')
        form = QtWidgets.QFormLayout(self.scanner_settings_group)
        self.scanner_setting_inputs = {}
        for key, label, limits in (
                ('minimum', 'Minimum voltage (V)', (0, 150)),
                ('maximum', 'Maximum voltage (V)', (0, 150)),
                ('rate', 'Maximum ramp rate (V/s)', (0.001, 1)),
                ('increment', 'Maximum increment (V)', (0.001, 0.05))):
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(3)
            box.setRange(*limits)
            box.setValue(getattr(ScannerSettings(), key))
            box.setKeyboardTracking(False)
            self.scanner_setting_inputs[key] = box
            form.addRow(label, box)
        self.apply_scanner_settings_btn = QtWidgets.QPushButton('Apply and save safety settings')
        self.apply_scanner_settings_btn.clicked.connect(self.apply_scanner_settings)
        form.addRow(self.apply_scanner_settings_btn)
        layout.addWidget(self.scanner_settings_group)
        note = QtWidgets.QLabel('All moves ramp slowly, including arrow buttons, Enter, setscan and measurement scans. '
                               'Stop holds the reached voltage. Targets outside the saved limits are rejected.')
        note.setWordWrap(True)
        layout.addWidget(note)

    def sync_scanner_settings(self):
        settings = getattr(self.scanner_controller, 'scanner_settings', ScannerSettings())
        for key, box in self.scanner_setting_inputs.items():
            box.setValue(getattr(settings, key))
        error = getattr(self.scanner_controller, 'scanner_settings_error', '')
        if error:
            self._scanner_command_error = error

    def apply_scanner_settings(self):
        try:
            if self.scanner_controller is None:
                raise RuntimeError('Scanner is not connected.')
            settings = ScannerSettings(**{key: box.value() for key, box in self.scanner_setting_inputs.items()})
            self.scanner_controller.configure_motion(settings)
            saved = self.save_storage()
            self._scanner_command_error = ('Safety settings applied and saved; no voltage changed.' if saved else
                                           'Safety settings applied for this session, but saving failed.')
        except Exception as exc:
            self._scanner_command_error = str(exc)
        self.refresh_scanner_readbacks()

    def execute_scan_command(self):
        try:
            command, values = self.scan_command.text().strip().split(None, 1)
            x, y = (float(value.strip()) for value in values.split(','))
            if command.lower() not in ('setscan', 'goto'):
                raise ValueError('Use setscan X,Y (absolute V) or goto dX,dY (relative V).')
            self.start_scanner_motion({1: x, 2: y}, relative=command.lower() == 'goto')
        except Exception as exc:
            self._scanner_command_error = str(exc)
            self.refresh_scanner_readbacks()

    def start_scanner_motion(self, targets, relative=False):
        if self.scanner_moving():
            return
        self._scanner_busy = True
        if self.scanner_worker:
            self.scanner_worker.deleteLater()
        self._scanner_command_error = 'Ramping slowly...'
        self.scanner_worker = ScannerMotionWorker(self.scanner_controller, targets, relative, self)
        self.scanner_worker.progress.connect(self.scanner_motion_progress)
        self.scanner_worker.error.connect(self.scanner_motion_error)
        self.scanner_worker.finished.connect(self.scanner_motion_finished)
        self.scanner_motion_controls.setEnabled(False)
        self.scanner_settings_group.setEnabled(False)
        self.btn_reset_scanner.setEnabled(False)
        self.scanner_stop_btn.setEnabled(True)
        self.scanner_worker.start()
        self.refresh_scanner_readbacks()
        self.update_motion_lockout()

    def scanner_motion_progress(self, axis, value):
        self._scanner_command_error = f'Ramping slowly: axis {axis}, {value:.3f} V'
        # The worker already verified this offset. Do not perform serial queries
        # in the GUI thread for every increment; Stop must remain responsive.
        name = {1: 'x', 2: 'y'}.get(axis)
        if name:
            label = getattr(self, f'v{name}_label')
            label.setText(f'V{name}: {value:.3f} V')
            label.setToolTip('Offset verified during the current ramp.')
        prefix = 'SIMULATED ANC300' if getattr(self.scanner_controller, 'simulated', False) else 'ANC300'
        self.scanner_status.setText(prefix + '\n' + self._scanner_command_error)

    def scanner_motion_error(self, message):
        self._scanner_command_error = f'Scanner motion failed: {message}'

    def scanner_motion_finished(self):
        self._scanner_busy = False
        if self.scanner_worker.failure:
            self._scanner_command_error = f'Scanner motion failed: {self.scanner_worker.failure}'
        elif self.scanner_worker.completed:
            self._scanner_command_error = 'Scanner ramp complete.'
        elif self.scanner_worker.stop_requested:
            self._scanner_command_error = 'Scanner ramp stopped at the reached voltage.'
        self.scanner_motion_controls.setEnabled(True)
        self.scanner_settings_group.setEnabled(True)
        self.btn_reset_scanner.setEnabled(True)
        self.scanner_stop_btn.setEnabled(False)
        self.refresh_scanner_readbacks()
        self.update_motion_lockout()

    def build_stepper_motion_ui(self, layout):
        self.stepper_position_label = QtWidgets.QLabel()
        layout.addWidget(self.stepper_position_label)
        note = QtWidgets.QLabel('Pseudo-coordinate in commanded steps, relative to session start / Set zero. '
                               'No position sensor; step size and slip are not measured.')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.stepper_targets = {}
        for axis in ('x', 'y', 'z'):
            box = QtWidgets.QSpinBox()
            box.setRange(-1000000, 1000000)
            box.setPrefix(axis.upper() + ': ')
            self.stepper_targets[axis] = box
            row.addWidget(box)
        layout.addLayout(row)
        self.stepper_relative = QtWidgets.QCheckBox('Relative move (delta steps)')
        layout.addWidget(self.stepper_relative)
        row = QtWidgets.QHBoxLayout()
        self.stepper_goto_btn = QtWidgets.QPushButton('Goto X, Y, Z')
        self.stepper_goto_btn.clicked.connect(lambda: self.start_stepper_motion(
            targets={i: self.stepper_targets[a].value() for i, a in enumerate(('x', 'y', 'z'), 1)},
            relative=self.stepper_relative.isChecked()))
        self.stepper_zero_btn = QtWidgets.QPushButton('Set zero (no motion)')
        self.stepper_zero_btn.clicked.connect(self.zero_stepper_coordinates)
        self.stepper_stop_btn = QtWidgets.QPushButton('Stop stepper')
        self.stepper_stop_btn.clicked.connect(self.stop_stepper_motion)
        self.stepper_stop_btn.setEnabled(False)
        for button in (self.stepper_goto_btn, self.stepper_zero_btn, self.stepper_stop_btn):
            row.addWidget(button)
        layout.addLayout(row)
        self.stepper_status = QtWidgets.QLabel('Idle. Click an arrow: 1 step. Hold it: keeps stepping until '
                                               'released. Limits 100 Hz / 120 V.')
        self.stepper_status.setWordWrap(True)
        layout.addWidget(self.stepper_status)
        self.refresh_stepper_coordinates()

    def refresh_stepper_coordinates(self):
        controller = self.stepper_controller
        values = [controller.pseudo_position(a) if controller else None for a in (1, 2, 3)]
        self.stepper_position_label.setText('Pseudo position: ' + '   '.join(
            f'{axis}: {value if value is not None else "unknown"}' for axis, value in zip('XYZ', values)))
        group = getattr(self, 'stepper_group', None)
        if group is not None:
            group.setTitle('Stepper (ANC150): not connected' if controller is None else
                           'Stepper (ANC150): SIMULATED, nothing moves'
                           if getattr(controller, 'simulated', False) else 'Stepper (ANC150)')

    def zero_stepper_coordinates(self):
        try:
            if self.stepper_moving():
                raise RuntimeError('Stop stepper motion before zeroing.')
            if self.stepper_controller is None:
                raise RuntimeError('Stepper is not connected.')
            self.stepper_controller.zero_pseudo_position()
            self.stepper_status.setText('Pseudo-coordinate set to zero. No motion.')
        except Exception as exc:
            self.stepper_status.setText(str(exc))
        self.refresh_stepper_coordinates()

    def start_stepper_motion(self, targets=None, jog=None, relative=False):
        if self.stepper_moving():
            return
        self._stepper_busy = True
        if self.stepper_worker:
            self.stepper_worker.deleteLater()
        self._stepper_error = ''
        self.stepper_worker = StepperMotionWorker(self.stepper_controller, targets, jog, relative, self)
        self.stepper_worker.progress.connect(self.refresh_stepper_coordinates)
        self.stepper_worker.error.connect(self.stepper_motion_error)
        self.stepper_worker.finished.connect(self.stepper_motion_finished)
        for widget in [self.stepper_goto_btn, self.stepper_zero_btn, *self.f_edits, *self.a_edits]:
            widget.setEnabled(False)
        self.stepper_stop_btn.setEnabled(True)
        self.stepper_status.setStyleSheet('')
        self.stepper_status.setText('Moving in counted steps...')
        self.stepper_worker.start()
        self.update_motion_lockout()

    def stepper_motion_error(self, message):
        self._stepper_error = message
        self.stepper_status.setText(message)
        self.stepper_status.setStyleSheet(ERROR_STYLE)
        print(f'Stepper stopped: {message}')  # also in the Device Manager's application log

    def stepper_motion_finished(self):
        self._stepper_busy = False
        for widget in [self.stepper_goto_btn, self.stepper_zero_btn, *self.f_edits, *self.a_edits]:
            widget.setEnabled(True)
        self.stepper_stop_btn.setEnabled(False)
        text = 'Stepper move complete.' if self.stepper_worker.completed else 'Stepper stopped.'
        if getattr(self.stepper_controller, 'simulated', False):
            text += ' SIMULATED ANC150: no hardware moved. Connect it in the Device Manager.'
        self.stepper_status.setText(self._stepper_error or text)
        self.refresh_stepper_coordinates()
        self.update_motion_lockout()

    def stop_stepper_motion(self):
        self.long_press_timer.stop()
        self.active_button = None
        self.is_continuous_moving = False
        if self.stepper_worker:
            self.stepper_worker.stop()

    def update_motion_lockout(self):
        dashboard = self.window()
        if hasattr(dashboard, 'check_measurement_status'):
            dashboard.check_measurement_status()

    def hideEvent(self, event):
        # A held arrow must not continue jogging after the dashboard is hidden.
        if hasattr(self, 'long_press_timer'):
            self.stop_stepper_motion()
        super().hideEvent(event)
