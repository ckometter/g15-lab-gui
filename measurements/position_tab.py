import os
import json
import math
from PyQt6 import QtWidgets, QtCore, QtGui
from scanner_safety import ScannerSettings
from measurements.position_motion import PositionMotionControls

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SwitchHeaterWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, float, float)
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, magnet, target_state, parent=None):
        super().__init__(parent)
        self.magnet = magnet
        self.target_state = target_state # 'on' or 'off'
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def run(self):
        try:
            self.stop_requested = False
            if self.target_state == 'on':
                # Check switch heater state
                heater = self.magnet._get_switch_heater()
                if heater.startswith('off'):
                    output_f = self.magnet._get_output_field()
                    persistent_f = self.magnet._get_persistent_magnet_field()
                    # If leads do not match coil field, match them first
                    if not math.isclose(output_f, persistent_f, abs_tol=0.01):
                        self.status.emit("Matching leads to coil...")
                        self.magnet._set_target_field(persistent_f)
                        self.magnet.goto_set()
                        self.msleep(1000)
                        while self.magnet._get_is_sweeping() and not self.stop_requested:
                            self.msleep(100)
                            current_f = self.magnet._get_output_field()
                            try: volts = self.magnet.voltage()
                            except Exception: volts = 0.0
                            try: amps = self.magnet.current()
                            except Exception: amps = 0.0
                            self.progress.emit(current_f, volts, amps)
                        self.magnet.hold()
                    
                    if self.stop_requested:
                        self.magnet.hold()
                        self.finished.emit()
                        return
                        
                    self.status.emit("Turning switch heater on...")
                    self.magnet._set_switch_heater('on')
            else:
                # Turning OFF
                # We can only turn off if field is at zero
                current_field = self.magnet._get_output_field()
                if abs(current_field) < 0.0005:
                    self.status.emit("Turning switch heater off...")
                    self.magnet._set_switch_heater('off')
                else:
                    raise RuntimeError(f"Cannot turn off switch heater: field is not at zero ({current_field:.4f} T).")

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class MagnetRampWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, float, float)
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, magnet, target_field, sweep_rate, parent=None):
        super().__init__(parent)
        self.magnet = magnet
        self.target_field = target_field
        self.sweep_rate = sweep_rate
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def run(self):
        try:
            self.stop_requested = False
            # 1. Set sweep rate (convert T/s to T/min for driver)
            self.magnet.field_sweep_rate(self.sweep_rate * 60.0)
            
            # 2. Check switch heater state and match leads if needed
            if getattr(self.magnet, 'SWITCH_HEATER_FITTED', False):
                heater = self.magnet._get_switch_heater()
                if heater.startswith('off'):
                    output_f = self.magnet._get_output_field()
                    persistent_f = self.magnet._get_persistent_magnet_field()
                    # If leads do not match coil field, match them first
                    if not math.isclose(output_f, persistent_f, abs_tol=0.01):
                        self.status.emit("Matching leads to coil...")
                        self.magnet._set_target_field(persistent_f)
                        self.magnet.goto_set()
                        self.msleep(1000)
                        while self.magnet._get_is_sweeping() and not self.stop_requested:
                            self.msleep(100)
                            current_f = self.magnet._get_output_field()
                            try: volts = self.magnet.voltage()
                            except Exception: volts = 0.0
                            try: amps = self.magnet.current()
                            except Exception: amps = 0.0
                            self.progress.emit(current_f, volts, amps)
                        self.magnet.hold()
                    
                    if self.stop_requested:
                        self.magnet.hold()
                        self.finished.emit()
                        return
                        
                    self.status.emit("Turning switch heater on...")
                    self.magnet._set_switch_heater('on')

            # 3. Ramp to target
            self.status.emit(f"Ramping to {self.target_field:.4f} T...")
            self.magnet._set_target_field(self.target_field)
            self.magnet.goto_set()
            self.msleep(1000)

            # Loop while sweeping
            while self.magnet._get_is_sweeping() and not self.stop_requested:
                self.msleep(200)
                current_val = self.magnet._get_output_field()
                try: volts = self.magnet.voltage()
                except Exception: volts = 0.0
                try: amps = self.magnet.current()
                except Exception: amps = 0.0
                self.progress.emit(current_val, volts, amps)

            # Hold magnet
            self.magnet.hold()

            # Turn off heater if fitting (Disabled: persistent mode cannot be enabled)
            # if getattr(self.magnet, 'SWITCH_HEATER_FITTED', False) and not self.stop_requested:
            #     self.status.emit("Turning switch heater off...")
            #     self.magnet._set_switch_heater('off')

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

# Reuse button style from Dual_gate_sweep_gui (assuming it's defined elsewhere or replicate basic style)
button_style = """
QPushButton {
    background-color: #ffffff;
    border: 1px solid #cccccc;
    font-weight: bold;
    border-radius: 0px;
    min-width: 30px;
    min-height: 30px;
    padding: 0px;
    color: #000000;
}
QPushButton:hover {
    background-color: #eeeeee;
}
QPushButton:pressed {
    background-color: #cccccc;
}
"""

class PositionTab(PositionMotionControls, QtWidgets.QWidget):
    """Measurement tab providing stepper and scanner controls.

    Left panel: Stepper (X/Y/Z) with directional buttons arranged
    to represent movement. Right panel: Scanner (X/Y/Z) with similar layout.
    Includes step‑size inputs and voltage displays for the scanner.
    """

    def __init__(self, station, parent=None):
        super().__init__(parent)
        # Assign hardware controllers from the provided station
        self.stepper_controller = getattr(station, 'piezo150', None)
        self.scanner_controller = getattr(station, 'piezo300', None)
        self.magnet_controller = getattr(station, 'magnet', None)
        self.magnet_ramping = False
        self.magnet_worker = None
        self._scanner_states = {}
        self._scanner_command_error = ""
        self.scanner_worker = None
        self.stepper_worker = None
        self._stepper_error = ''
        self.setup_ui()
        self.load_storage()

        # Initialize magnet UI values
        if self.magnet_controller is not None:
            try:
                target_f = self.magnet_controller.target_field_param()
                self.magnet_setpoint.blockSignals(True)
                self.magnet_setpoint.setValue(target_f)
                self.magnet_setpoint.blockSignals(False)
            except Exception as e:
                print(f"Error reading initial target field: {e}")
            self.poll_magnet_parameters()
        else:
            self.magnet_live_field_lbl.setText("No Magnet")
            self.magnet_volt_lbl.setText("N/A")
            self.magnet_amp_lbl.setText("N/A")
            self.magnet_mode_combo.setEnabled(False)
            self.magnet_setpoint.setEnabled(False)
            self.magnet_ramp_rate.setEnabled(False)
            self.magnet_zero_btn.setEnabled(False)
            self.magnet_heater_btn.setEnabled(False)

        # Setup polling timer (every 1 second) to refresh actual voltages on sample
        self.polling_timer = QtCore.QTimer(self)
        self.polling_timer.timeout.connect(self.poll_voltages)
        self.polling_timer.start(1000)

        # Setup timer for detecting long button presses (continuous movement)
        self.long_press_timer = QtCore.QTimer(self)
        self.long_press_timer.setSingleShot(True)
        self.long_press_timer.timeout.connect(self.on_long_press_timeout)
        self.active_button = None
        self.active_axis = None
        self.active_direction = 0
        self.is_continuous_moving = False

    def refresh_scanner_readbacks(self):
        controller = self.scanner_controller
        lock = getattr(controller, 'comm_lock', None)
        if lock is not None and not lock.acquire(blocking=False):
            return
        try:
            messages = []
            for axis, number in (('x', 1), ('y', 2)):
                label = getattr(self, f'v{axis}_label')
                spin = getattr(self, f'v{axis}_set_spin')
                negative, positive = ((self.s_btn_left, self.s_btn_right) if axis == 'x'
                                      else (self.s_btn_down, self.s_btn_up))
                try:
                    if controller is None:
                        raise RuntimeError('not connected')
                    state = controller.get_axis_status(number)
                    offset = float(state['offset'])
                    output = float(state['output'])
                    if not math.isfinite(offset) or not math.isfinite(output) or not 0 <= offset <= 150:
                        raise ValueError('invalid voltage readback')
                    self._scanner_states[axis] = state
                    label.setText(f"V{axis}: {offset:.3f} V")
                    label.setToolTip(f"Controller DC offset: {offset:.3f} V; measured output: {output:.3f} V; mode: {state['mode']}")
                    if not spin.hasFocus() and not self.scanner_moving():
                        spin.blockSignals(True)
                        spin.setValue(offset)
                        spin.blockSignals(False)
                    offset_ready = state['mode'] == 'off'
                    ready = offset_ready and not self.scanner_moving()
                    spin.setEnabled(ready)
                    settings = getattr(controller, 'scanner_settings', ScannerSettings())
                    negative.setEnabled(ready and offset > settings.minimum)
                    positive.setEnabled(ready and offset < settings.maximum)
                    mode_text = "offset enabled" if offset_ready else f"mode {state['mode']}"
                    messages.append(f"{axis.upper()}: output {output:.3f} V, {mode_text}")
                    if not offset_ready:
                        messages.append(f"{axis.upper()} is not in offset mode; controls disabled")
                except Exception as exc:
                    self._scanner_states.pop(axis, None)
                    label.setText(f"V{axis}: unavailable")
                    label.setToolTip(str(exc))
                    spin.setEnabled(False)
                    spin.lineEdit().setText('Unavailable')
                    negative.setEnabled(False)
                    positive.setEnabled(False)
                    messages.append(f"{axis.upper()}: {exc}")
            prefix = 'SIMULATED ANC300' if getattr(controller, 'simulated', False) else 'ANC300 readback'
            self.scanner_status.setText(prefix + '\n' + '\n'.join(messages)
                                        + ('\n' + self._scanner_command_error if self._scanner_command_error else ''))
        finally:
            if lock is not None:
                lock.release()

    def poll_voltages(self):
        self.refresh_scanner_readbacks()
        self.refresh_stepper_coordinates()
        if self.stepper_moving():
            return
        # Poll stepper frequencies and amplitudes
        if self.stepper_controller is None:
            self.sf1_label.setText("Fx: N/A")
            self.sf2_label.setText("Fy: N/A")
            self.sf3_label.setText("Fz: N/A")
            self.am1_label.setText("Ax: N/A")
            self.am2_label.setText("Ay: N/A")
            self.am3_label.setText("Az: N/A")
        else:
            is_sim = getattr(self.stepper_controller, 'simulated', False)
            axis_chars = {1: 'x', 2: 'y', 3: 'z'}
            for axis, label in zip([1, 2, 3], [self.sf1_label, self.sf2_label, self.sf3_label]):
                try:
                    ch = axis_chars[axis]
                    if is_sim:
                        label.setText(f"F{ch}: Simulation")
                    else:
                        freq = getattr(self.stepper_controller, f'frequency_ch{axis}')()
                        if freq > 0:
                            label.setText(f"F{ch}: {freq:.0f} Hz")
                        else:
                            label.setText(f"F{ch}: Off")
                except Exception:
                    label.setText(f"F{ch}: unavailable")
            for axis, label in zip([1, 2, 3], [self.am1_label, self.am2_label, self.am3_label]):
                try:
                    ch = axis_chars[axis]
                    if is_sim:
                        label.setText(f"A{ch}: Simulation")
                    else:
                        amp = getattr(self.stepper_controller, f'voltage_ch{axis}')()
                        if amp > 0:
                            label.setText(f"A{ch}: {amp:.1f} V")
                        else:
                            label.setText(f"A{ch}: Off")
                except Exception:
                    label.setText(f"A{ch}: unavailable")

        # Poll magnet parameters every 3 ticks (3 seconds) to avoid freezing the GUI
        if self.magnet_controller is not None and not self.magnet_ramping:
            if not hasattr(self, 'magnet_poll_counter'):
                self.magnet_poll_counter = 0
            self.magnet_poll_counter += 1
            if self.magnet_poll_counter >= 3:
                self.magnet_poll_counter = 0
                self.poll_magnet_parameters()

    def poll_magnet_parameters(self):
        if self.magnet_controller is None:
            return
        try:
            current_field = self.magnet_controller.field()
            self.magnet_live_field_lbl.setText(f"{current_field:.4f} T")
            
            voltage = self.magnet_controller.voltage()
            self.magnet_volt_lbl.setText(f"{voltage:.4f} V")
            
            current = self.magnet_controller.current()
            self.magnet_amp_lbl.setText(f"{current:.4f} A")
            
            mode = self.magnet_controller.control_mode()
            self.magnet_mode_combo.blockSignals(True)
            self.magnet_mode_combo.setCurrentText(mode)
            self.magnet_mode_combo.blockSignals(False)
            
            rate = self.magnet_controller.field_sweep_rate() / 60.0
            self.magnet_ramp_rate.blockSignals(True)
            self.magnet_ramp_rate.setValue(rate)
            self.magnet_ramp_rate.blockSignals(False)
            
            # Activity readback — display only, no automatic commands
            activity = self.magnet_controller.activity()
            if activity == 'clamped':
                self.magnet_activity_lbl.setText("CLAMPED ⚠")
                self.magnet_activity_lbl.setStyleSheet("color: #dc2626; font-weight: bold;")
            else:
                self.magnet_activity_lbl.setText(activity)
                self.magnet_activity_lbl.setStyleSheet("color: #000000; font-weight: bold;")

            # Switch heater status button update
            heater_status = self.magnet_controller.switch_heater()
            self.magnet_heater_btn.setStyleSheet("font-weight: bold; min-height: 25px;")
            if 'no switch heater' in heater_status:
                self.magnet_heater_btn.setText("No Heater")
                self.magnet_heater_btn.setEnabled(False)
            elif 'on' in heater_status:
                self.magnet_heater_btn.setText("ON")
                if abs(current_field) < 0.0005:
                    self.magnet_heater_btn.setEnabled(True)
                else:
                    self.magnet_heater_btn.setEnabled(False)
            else:
                self.magnet_heater_btn.setText("OFF")
                self.magnet_heater_btn.setEnabled(True)
        except Exception as e:
            print(f"Error polling magnet parameters: {e}")
            self.magnet_live_field_lbl.setText("Error")
            self.magnet_volt_lbl.setText("Error")
            self.magnet_amp_lbl.setText("Error")

    def setup_ui(self):
        # Main layout containing stepper and scanner groups
        outer_layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(content)
        self.panels_layout = main_layout  # add_panel() appends to the right of the controls
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        # ----- Stepper Group -----
        stepper_group = QtWidgets.QGroupBox("Stepper (ANC150)")
        
        # Grid/Cross layout for XY buttons (wrapped in a fixed-size widget to keep it nicely quadratic)
        cross_widget = QtWidgets.QWidget()
        cross_widget.setFixedSize(160, 160)
        cross_layout = QtWidgets.QGridLayout(cross_widget)
        cross_layout.setContentsMargins(10, 10, 10, 10)
        cross_layout.setSpacing(10)
        
        self.btn_up = QtWidgets.QPushButton("Y ↑")
        self.btn_down = QtWidgets.QPushButton("Y ↓")
        self.btn_left = QtWidgets.QPushButton("X ←")
        self.btn_right = QtWidgets.QPushButton("X →")
        for btn in (self.btn_up, self.btn_down, self.btn_left, self.btn_right):
            btn.setStyleSheet(button_style)
            btn.setFixedSize(40, 40)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        cross_layout.addWidget(self.btn_up, 0, 1)
        cross_layout.addWidget(self.btn_left, 1, 0)
        cross_layout.addWidget(self.btn_right, 1, 2)
        cross_layout.addWidget(self.btn_down, 2, 1)
        
        # Z axis layout (aligned on the right side of the cross layout)
        z_stepper_layout = QtWidgets.QVBoxLayout()
        self.btn_z_up = QtWidgets.QPushButton("Z ↑")
        self.btn_z_down = QtWidgets.QPushButton("Z ↓")
        for btn in (self.btn_z_up, self.btn_z_down):
            btn.setStyleSheet(button_style)
            btn.setFixedSize(65, 30)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        z_stepper_layout.addWidget(self.btn_z_up)
        z_stepper_layout.addWidget(self.btn_z_down)
        z_stepper_layout.addStretch() # Push Z buttons up to align with XY cross
        # Combine buttons at the top horizontally: XY cross on left, Z axis on right (aligned to top)
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(cross_widget, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        buttons_layout.addLayout(z_stepper_layout)
        
        # Assemble stepper group layout vertically
        stepper_group_layout = QtWidgets.QVBoxLayout()
        stepper_group_layout.addLayout(buttons_layout) # Buttons on top
        
        # Frequency and Amplitude display & controls for each axis
        self.stepper_params_layout = QtWidgets.QGridLayout()
        
        # Headers/Titles
        freq_title = QtWidgets.QLabel("Step Frequency")
        freq_title.setStyleSheet("font-weight: bold; color: #000000;")
        amp_title = QtWidgets.QLabel("Step Amplitude")
        amp_title.setStyleSheet("font-weight: bold; color: #000000;")
        
        self.stepper_params_layout.addWidget(freq_title, 0, 0, 1, 3)
        self.stepper_params_layout.addWidget(amp_title, 0, 3, 1, 3)
        
        self.sf1_label = QtWidgets.QLabel("Fx: 0 Hz")
        self.sf2_label = QtWidgets.QLabel("Fy: 0 Hz")
        self.sf3_label = QtWidgets.QLabel("Fz: 0 Hz")
        
        self.am1_label = QtWidgets.QLabel("Ax: 0 V")
        self.am2_label = QtWidgets.QLabel("Ay: 0 V")
        self.am3_label = QtWidgets.QLabel("Az: 0 V")
        
        self.f_edits = []
        self.a_edits = []
        
        for i, axis in enumerate([1, 2, 3], start=1):
            # Frequency controls
            f_polled_label = getattr(self, f'sf{axis}_label')
            f_edit = QtWidgets.QLineEdit()
            f_edit.setPlaceholderText("Set Hz")
            f_edit.setValidator(QtGui.QIntValidator(1, 100))
            f_edit.setFixedWidth(55)
            f_edit.returnPressed.connect(lambda ax=axis, edit=f_edit: self.set_stepper_frequency(ax, edit))
            self.f_edits.append(f_edit)
            
            # Amplitude controls
            a_polled_label = getattr(self, f'am{axis}_label')
            a_edit = QtWidgets.QLineEdit()
            a_edit.setPlaceholderText("Set V")
            a_edit.setValidator(QtGui.QIntValidator(0, 120))
            a_edit.setFixedWidth(55)
            a_edit.returnPressed.connect(lambda ax=axis, edit=a_edit: self.set_stepper_amplitude(ax, edit))
            self.a_edits.append(a_edit)
            
            # Add to grid
            self.stepper_params_layout.addWidget(f_polled_label, i, 0)
            self.stepper_params_layout.addWidget(f_edit, i, 1)
            self.stepper_params_layout.addWidget(QtWidgets.QLabel("Hz"), i, 2)
            
            self.stepper_params_layout.addWidget(a_polled_label, i, 3)
            self.stepper_params_layout.addWidget(a_edit, i, 4)
            self.stepper_params_layout.addWidget(QtWidgets.QLabel("V"), i, 5)
            
        stepper_group_layout.addLayout(self.stepper_params_layout)
        self.build_stepper_motion_ui(stepper_group_layout)
        stepper_group_layout.addStretch() # Push everything to the top
        stepper_group.setLayout(stepper_group_layout)

        # ----- Scanner Group -----
        scanner_group = QtWidgets.QGroupBox("Scanner (ANC300)")
        # XY cross layout (wrapped in a fixed-size widget to keep it nicely quadratic)
        scanner_cross_widget = QtWidgets.QWidget()
        scanner_cross_widget.setFixedSize(160, 160)
        scanner_top_layout = QtWidgets.QGridLayout(scanner_cross_widget)
        scanner_top_layout.setContentsMargins(10, 10, 10, 10)
        scanner_top_layout.setSpacing(10)
        
        self.s_btn_up = QtWidgets.QPushButton("Y ↑")
        self.s_btn_down = QtWidgets.QPushButton("Y ↓")
        self.s_btn_left = QtWidgets.QPushButton("X ←")
        self.s_btn_right = QtWidgets.QPushButton("X →")
        for btn in (self.s_btn_up, self.s_btn_down, self.s_btn_left, self.s_btn_right):
            btn.setStyleSheet(button_style)
            btn.setFixedSize(40, 40)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        scanner_top_layout.addWidget(self.s_btn_up, 0, 1)
        scanner_top_layout.addWidget(self.s_btn_left, 1, 0)
        scanner_top_layout.addWidget(self.s_btn_right, 1, 2)
        scanner_top_layout.addWidget(self.s_btn_down, 2, 1)
        
        # Step size input for scanner (below buttons)
        scanner_step_layout = QtWidgets.QHBoxLayout()
        scanner_step_layout.addWidget(QtWidgets.QLabel("Step size:"))
        self.s_step_size_edit = QtWidgets.QLineEdit("0.1")
        self.s_step_size_edit.setValidator(QtGui.QDoubleValidator(bottom=0))
        scanner_step_layout.addWidget(self.s_step_size_edit)
        
        # Voltage display (below step size)
        scanner_volt_layout = QtWidgets.QGridLayout()
        scanner_volt_layout.setSpacing(6)

        self.vx_label = QtWidgets.QLabel("Vx: awaiting readback")
        self.vx_label.setStyleSheet("font-weight: bold; color: #000000;")
        scanner_volt_layout.addWidget(self.vx_label, 0, 0)

        self.vx_set_spin = QtWidgets.QDoubleSpinBox()
        self.vx_set_spin.setRange(-10000.0, 10000.0)
        self.vx_set_spin.setDecimals(3)
        self.vx_set_spin.setSingleStep(0.1)
        self.vx_set_spin.setSuffix(" V")
        self.vx_set_spin.setKeyboardTracking(False)
        self.vx_set_spin.lineEdit().returnPressed.connect(self.on_scanner_vx_enter)
        scanner_volt_layout.addWidget(self.vx_set_spin, 0, 1)

        self.vy_label = QtWidgets.QLabel("Vy: awaiting readback")
        self.vy_label.setStyleSheet("font-weight: bold; color: #000000;")
        scanner_volt_layout.addWidget(self.vy_label, 1, 0)

        self.vy_set_spin = QtWidgets.QDoubleSpinBox()
        self.vy_set_spin.setRange(-10000.0, 10000.0)
        self.vy_set_spin.setDecimals(3)
        self.vy_set_spin.setSingleStep(0.1)
        self.vy_set_spin.setSuffix(" V")
        self.vy_set_spin.setKeyboardTracking(False)
        self.vy_set_spin.lineEdit().returnPressed.connect(self.on_scanner_vy_enter)
        scanner_volt_layout.addWidget(self.vy_set_spin, 1, 1)
        
        # Reset scanner communication button
        self.btn_reset_scanner = QtWidgets.QPushButton("🔄 Reset Scanner Comm")
        self.btn_reset_scanner.setStyleSheet("""
            QPushButton {
                background-color: #f3f4f6;
                color: #1f2937;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
            }
            QPushButton:pressed {
                background-color: #d1d5db;
            }
        """)
        self.btn_reset_scanner.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.btn_reset_scanner.clicked.connect(self.on_reset_scanner_clicked)

        # Combine scanner layouts vertically: buttons, step size, voltage inputs, reset button
        scanner_group_layout = QtWidgets.QVBoxLayout()
        scanner_group_layout.addWidget(scanner_cross_widget) # Buttons on top
        scanner_group_layout.addLayout(scanner_step_layout) # Then step size
        scanner_group_layout.addLayout(scanner_volt_layout) # Then voltage inputs
        self.scanner_status = QtWidgets.QLabel("Reading ANC300 voltages...")
        self.scanner_status.setWordWrap(True)
        scanner_group_layout.addWidget(self.scanner_status)
        self.build_scanner_motion_ui(scanner_group_layout)
        scanner_group_layout.addWidget(self.btn_reset_scanner) # Reset comm button
        scanner_group_layout.addStretch() # Push everything to the top
        scanner_group.setLayout(scanner_group_layout)

        # ----- Magnet Group -----
        magnet_group = QtWidgets.QGroupBox("Magnet")
        magnet_group_layout = QtWidgets.QVBoxLayout()
        
        # Grid for Setpoint, Ramp Rate, and Readbacks
        mag_grid = QtWidgets.QGridLayout()
        mag_grid.setSpacing(6)
        
        # Setpoint
        mag_grid.addWidget(QtWidgets.QLabel("Setpoint (T):"), 0, 0)
        self.magnet_setpoint = QtWidgets.QDoubleSpinBox()
        self.magnet_setpoint.setRange(-15.0, 15.0)
        self.magnet_setpoint.setDecimals(4)
        self.magnet_setpoint.setSingleStep(0.01)
        self.magnet_setpoint.setValue(0.0)
        self.magnet_setpoint.setKeyboardTracking(False)
        self.magnet_setpoint.lineEdit().returnPressed.connect(self.on_magnet_setpoint_enter)
        self.magnet_setpoint.valueChanged.connect(lambda _: self.save_storage())
        mag_grid.addWidget(self.magnet_setpoint, 0, 1)
        
        # Ramp Rate (Read-only as requested)
        mag_grid.addWidget(QtWidgets.QLabel("Ramp Rate (T/s):"), 1, 0)
        self.magnet_ramp_rate = QtWidgets.QDoubleSpinBox()
        self.magnet_ramp_rate.setRange(0.0001, 10.0)
        self.magnet_ramp_rate.setDecimals(4)
        self.magnet_ramp_rate.setReadOnly(True)
        self.magnet_ramp_rate.setEnabled(False)
        mag_grid.addWidget(self.magnet_ramp_rate, 1, 1)

        # Voltage (V) readback
        mag_grid.addWidget(QtWidgets.QLabel("Voltage:"), 2, 0)
        self.magnet_volt_lbl = QtWidgets.QLabel("-- V")
        self.magnet_volt_lbl.setStyleSheet("color: #000000; font-weight: bold;")
        mag_grid.addWidget(self.magnet_volt_lbl, 2, 1)

        # Current (A) readback
        mag_grid.addWidget(QtWidgets.QLabel("Current:"), 3, 0)
        self.magnet_amp_lbl = QtWidgets.QLabel("-- A")
        self.magnet_amp_lbl.setStyleSheet("color: #000000; font-weight: bold;")
        mag_grid.addWidget(self.magnet_amp_lbl, 3, 1)

        # Field (T) readback
        mag_grid.addWidget(QtWidgets.QLabel("Field:"), 4, 0)
        self.magnet_live_field_lbl = QtWidgets.QLabel("-- T")
        self.magnet_live_field_lbl.setStyleSheet("color: #000000; font-weight: bold;")
        mag_grid.addWidget(self.magnet_live_field_lbl, 4, 1)

        # IPS Mode Selection
        mag_grid.addWidget(QtWidgets.QLabel("IPS Mode:"), 5, 0)
        self.magnet_mode_combo = QtWidgets.QComboBox()
        self.magnet_mode_combo.addItems([
            'Local & Locked',
            'Remote & Locked',
            'Local & Unlocked',
            'Remote & Unlocked'
        ])
        self.magnet_mode_combo.currentTextChanged.connect(self.on_magnet_mode_changed)
        mag_grid.addWidget(self.magnet_mode_combo, 5, 1)
        
        # Switch Heater Button (placed below IPS Mode, in the same style)
        mag_grid.addWidget(QtWidgets.QLabel("Switch Heater:"), 6, 0)
        self.magnet_heater_btn = QtWidgets.QPushButton("Heater: --")
        self.magnet_heater_btn.clicked.connect(self.on_magnet_heater_clicked)
        self.magnet_heater_btn.setStyleSheet("font-weight: bold; min-height: 25px;")
        mag_grid.addWidget(self.magnet_heater_btn, 6, 1)

        # Activity readback
        mag_grid.addWidget(QtWidgets.QLabel("Activity:"), 7, 0)
        self.magnet_activity_lbl = QtWidgets.QLabel("--")
        self.magnet_activity_lbl.setStyleSheet("color: #000000; font-weight: bold;")
        mag_grid.addWidget(self.magnet_activity_lbl, 7, 1)
        
        magnet_group_layout.addLayout(mag_grid)
        
        # Status Label
        self.magnet_status_lbl = QtWidgets.QLabel("Status: Idle")
        self.magnet_status_lbl.setStyleSheet("color: #000000; font-weight: bold; font-size: 11px;")
        magnet_group_layout.addWidget(self.magnet_status_lbl)
        
        # Buttons layout
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.magnet_zero_btn = QtWidgets.QPushButton("Go to Zero")
        self.magnet_zero_btn.setStyleSheet("background-color: #059669; color: #ffffff; border: 1px solid #059669; font-weight: bold; min-height: 25px; border-radius: 0px;")
        self.magnet_zero_btn.clicked.connect(self.on_magnet_zero_clicked)
        btn_layout.addWidget(self.magnet_zero_btn)
        
        self.magnet_stop_btn = QtWidgets.QPushButton("Stop Ramp")
        self.magnet_stop_btn.setEnabled(False)
        self.magnet_stop_btn.setStyleSheet("background-color: #000000; color: #ffffff; border: 1px solid #000000; font-weight: bold; min-height: 25px; border-radius: 0px;")
        self.magnet_stop_btn.clicked.connect(self.on_magnet_stop_clicked)
        btn_layout.addWidget(self.magnet_stop_btn)
        
        magnet_group_layout.addLayout(btn_layout)
        
        magnet_group_layout.addStretch()
        magnet_group.setLayout(magnet_group_layout)

        # Left column layout containing Stepper on top and Magnet below
        left_column_layout = QtWidgets.QVBoxLayout()
        left_column_layout.addWidget(stepper_group, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        left_column_layout.addWidget(magnet_group, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        left_column_layout.addStretch()

        # Add groups and sublayouts to main layout
        main_layout.addLayout(left_column_layout)
        main_layout.addWidget(scanner_group, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        # Connect signals for stepper buttons (with long-press continuous move support)
        self.btn_up.pressed.connect(lambda: self.on_button_pressed(self.btn_up, 'y', +1))
        self.btn_up.released.connect(lambda: self.on_button_released(self.btn_up, 'y', +1))

        self.btn_down.pressed.connect(lambda: self.on_button_pressed(self.btn_down, 'y', -1))
        self.btn_down.released.connect(lambda: self.on_button_released(self.btn_down, 'y', -1))

        self.btn_left.pressed.connect(lambda: self.on_button_pressed(self.btn_left, 'x', -1))
        self.btn_left.released.connect(lambda: self.on_button_released(self.btn_left, 'x', -1))

        self.btn_right.pressed.connect(lambda: self.on_button_pressed(self.btn_right, 'x', +1))
        self.btn_right.released.connect(lambda: self.on_button_released(self.btn_right, 'x', +1))

        self.btn_z_up.pressed.connect(lambda: self.on_button_pressed(self.btn_z_up, 'z', +1))
        self.btn_z_up.released.connect(lambda: self.on_button_released(self.btn_z_up, 'z', +1))

        self.btn_z_down.pressed.connect(lambda: self.on_button_pressed(self.btn_z_down, 'z', -1))
        self.btn_z_down.released.connect(lambda: self.on_button_released(self.btn_z_down, 'z', -1))

        # Connect signals for scanner buttons
        self.s_btn_up.clicked.connect(lambda: self.move_scanner('y', +1))
        self.s_btn_down.clicked.connect(lambda: self.move_scanner('y', -1))
        self.s_btn_left.clicked.connect(lambda: self.move_scanner('x', -1))
        self.s_btn_right.clicked.connect(lambda: self.move_scanner('x', +1))
        



    def add_panel(self, widget, stretch=1):
        """Show a read-only panel (e.g. the helium level) to the right of the motion controls."""
        self.panels_layout.addWidget(widget, stretch)

    def _get_step(self, line_edit: QtWidgets.QLineEdit) -> float:
        try:
            return float(line_edit.text())
        except ValueError:
            return 0.0

    def move_stepper(self, axis: str, direction: int):
        self.start_stepper_motion(targets={dict(x=1, y=2, z=3)[axis]: direction}, relative=True)

    def _change_scanner_voltage(self, axis, *, target=None, delta=None):
        self.start_scanner_motion({dict(x=1, y=2, z=3)[axis]: delta if delta is not None else target},
                                  relative=delta is not None)

    def move_scanner(self, axis: str, direction: int):
        try:
            step = float(self.s_step_size_edit.text())
            if not math.isfinite(step) or step <= 0 or direction not in (-1, 1):
                raise ValueError('Use a positive finite step size.')
        except ValueError as exc:
            self.scanner_status.setText(str(exc))
            return
        self._change_scanner_voltage(axis.lower(), delta=direction * step)

    def set_scanner_voltage(self, axis: str, target_volt: float):
        self._change_scanner_voltage(axis.lower(), target=target_volt)

    def on_scanner_vx_enter(self):
        if hasattr(self, 'vx_set_spin'):
            val = self.vx_set_spin.value()
            self.set_scanner_voltage('x', val)

    def on_scanner_vy_enter(self):
        if hasattr(self, 'vy_set_spin'):
            val = self.vy_set_spin.value()
            self.set_scanner_voltage('y', val)

    def on_reset_scanner_clicked(self):
        if self.scanner_controller is None or getattr(self.scanner_controller, 'simulated', True):
            QtWidgets.QMessageBox.information(
                self, "Scanner Reset",
                "Scanner controller is simulated or not initialized."
            )
            return
        
        try:
            success = False
            if hasattr(self.scanner_controller, 'reset_communication'):
                success = self.scanner_controller.reset_communication()
            else:
                lock = getattr(self.scanner_controller, 'comm_lock', None)
                if lock:
                    with lock:
                        if hasattr(self.scanner_controller, '_clear_input_buffer'):
                            self.scanner_controller._clear_input_buffer()
                        if hasattr(self.scanner_controller, 'visa_handle') and self.scanner_controller.visa_handle:
                            try:
                                self.scanner_controller.visa_handle.clear()
                            except Exception:
                                pass
                success = True

            if success:
                # Force immediate re-poll of scanner voltages
                self.poll_voltages()
                QtWidgets.QMessageBox.information(
                    self, "Scanner Reset",
                    "Scanner communication reset successfully!"
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Scanner Reset",
                    "Failed to reset scanner communication. Please check hardware connection."
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Scanner Reset Error",
                f"Error resetting scanner communication:\n{e}"
            )

    def load_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        magnet_setpoint = 0.0
        magnet_ramp_rate = 0.0083
        try:
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    data = json.load(f)
                if "position" in data:
                    magnet_setpoint = data["position"].get("magnet_setpoint", 0.0)
                    magnet_ramp_rate = data["position"].get("magnet_ramp_rate", 0.0083)
        except Exception as e:
            print(f"Error loading value_storage in PositionTab: {e}")
        # Saved offsets describe an old session, not the controller's present
        # output. Opening/reloading this tab reads hardware and never sets it.
        self.sync_scanner_settings()
        self.refresh_scanner_readbacks()

        # Restore magnet UI values from storage
        self.magnet_setpoint.blockSignals(True)
        self.magnet_setpoint.setValue(magnet_setpoint)
        self.magnet_setpoint.blockSignals(False)

    def save_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        try:
            data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    data = json.load(f)
            if "position" not in data:
                data["position"] = {}
            from dataclasses import asdict
            # Routine tab/close saves must not silently replace rejected safety
            # settings with defaults. Only an explicit Apply repairs them.
            if self.scanner_controller is not None and not self.scanner_controller.scanner_settings_error:
                data["position"]["scanner_safety"] = asdict(self.scanner_controller.scanner_settings)
            data["position"]["magnet_setpoint"] = self.magnet_setpoint.value()
            data["position"]["magnet_ramp_rate"] = self.magnet_ramp_rate.value()
            with open(storage_path, "w") as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving value_storage in PositionTab: {e}")
            return False

    def reload_gui_from_storage(self):
        self.load_storage()

    def on_button_pressed(self, btn, axis: str, direction: int):
        if self.stepper_moving():
            return
        self.active_button = btn
        self.active_axis = axis
        self.active_direction = direction
        self.is_continuous_moving = False
        self.long_press_timer.start(500) # 500 ms threshold for continuous move

    def on_button_released(self, btn, axis: str, direction: int):
        if self.active_button == btn:
            self.long_press_timer.stop()
            if self.is_continuous_moving:
                self.stop_continuous_stepper(axis)
            else:
                self.move_stepper(axis, direction)
            self.active_button = None
            self.is_continuous_moving = False

    def on_long_press_timeout(self):
        if self.active_button is not None:
            self.is_continuous_moving = True
            self.start_continuous_stepper(self.active_axis, self.active_direction)

    def start_continuous_stepper(self, axis: str, direction: int):
        # Repeated single pulses keep the pseudo-coordinate countable.
        self.start_stepper_motion(jog=(dict(x=1, y=2, z=3)[axis], direction))

    def stop_continuous_stepper(self, axis: str):
        self.stop_stepper_motion()

    def set_stepper_frequency(self, axis: int, edit: QtWidgets.QLineEdit):
        if self.stepper_controller is None:
            print("Stepper controller not initialized.")
            return
        text = edit.text()
        if not text:
            return
        try:
            val = int(text)
            if not 1 <= val <= 100:
                raise ValueError('Stepper frequency must be 1 to 100 Hz.')
            getattr(self.stepper_controller, f'frequency_ch{axis}')(val)
            edit.clear()
            edit.clearFocus()
        except Exception as e:
            self.stepper_status.setText(f"Failed to set stepper frequency: {e}")

    def set_stepper_amplitude(self, axis: int, edit: QtWidgets.QLineEdit):
        if self.stepper_controller is None:
            print("Stepper controller not initialized.")
            return
        text = edit.text()
        if not text:
            return
        try:
            val = float(text)
            if not math.isfinite(val) or not 0 <= val <= 120:
                raise ValueError('Stepper amplitude must be 0 to 120 V.')
            getattr(self.stepper_controller, f'voltage_ch{axis}')(val)
            edit.clear()
            edit.clearFocus()
        except Exception as e:
            self.stepper_status.setText(f"Failed to set stepper amplitude: {e}")

    def on_magnet_setpoint_enter(self):
        """Triggered when user presses Enter inside the magnet setpoint spinbox."""
        if self.magnet_controller is None:
            return
        
        target = self.magnet_setpoint.value()
        rate = self.magnet_ramp_rate.value()
        
        self.magnet_ramping = True
        self.magnet_setpoint.setEnabled(False)
        self.magnet_ramp_rate.setEnabled(False)
        self.magnet_mode_combo.setEnabled(False)
        self.magnet_zero_btn.setEnabled(False)
        self.magnet_heater_btn.setEnabled(False)
        self.magnet_stop_btn.setEnabled(True)
        self.magnet_status_lbl.setText(f"Status: Starting ramp to {target:.4f} T...")
        
        self.magnet_worker = MagnetRampWorker(self.magnet_controller, target, rate, parent=self)
        self.magnet_worker.progress.connect(self.on_magnet_progress)
        self.magnet_worker.status.connect(self.on_magnet_status)
        self.magnet_worker.finished.connect(self.on_magnet_ramp_finished)
        self.magnet_worker.error.connect(self.on_magnet_ramp_error)
        self.magnet_worker.start()

    def on_magnet_progress(self, val, volts, amps):
        self.magnet_live_field_lbl.setText(f"{val:.4f} T")
        self.magnet_volt_lbl.setText(f"{volts:.4f} V")
        self.magnet_amp_lbl.setText(f"{amps:.4f} A")
        target = self.magnet_setpoint.value()
        self.magnet_status_lbl.setText(f"Ramping to {target:.4f} T (Currently at {val:.4f} T)")

    def on_magnet_status(self, msg):
        self.magnet_status_lbl.setText(msg)

    def on_magnet_ramp_finished(self):
        if self.magnet_worker and self.magnet_worker.stop_requested:
            self.magnet_status_lbl.setText("Status: Ramping stopped/interrupted.")
        else:
            self.magnet_status_lbl.setText("Status: Ramping complete.")
        
        self.poll_magnet_parameters()
        self.magnet_ramping = False
        self.magnet_setpoint.setEnabled(True)
        self.magnet_ramp_rate.setEnabled(False)
        self.magnet_mode_combo.setEnabled(True)
        self.magnet_zero_btn.setEnabled(True)
        self.magnet_heater_btn.setEnabled(True)
        self.magnet_stop_btn.setEnabled(False)

    def on_magnet_ramp_error(self, err_msg):
        self.magnet_status_lbl.setText("Status: Ramping error.")
        self.poll_magnet_parameters()
        self.magnet_ramping = False
        self.magnet_setpoint.setEnabled(True)
        self.magnet_ramp_rate.setEnabled(False)
        self.magnet_mode_combo.setEnabled(True)
        self.magnet_zero_btn.setEnabled(True)
        self.magnet_heater_btn.setEnabled(True)
        self.magnet_stop_btn.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Magnet Ramping Error", f"Error ramping magnet:\n{err_msg}")

    def on_magnet_stop_clicked(self):
        if self.magnet_worker:
            self.magnet_worker.stop()
            self.magnet_status_lbl.setText("Status: Interrupting ramp...")

    def on_magnet_zero_clicked(self):
        """Action triggered by the 'Go to Zero' button."""
        if self.magnet_controller is None:
            return
        self.magnet_setpoint.setValue(0.0)
        self.on_magnet_setpoint_enter()

    def on_magnet_mode_changed(self, text):
        """Triggered when the user changes the IPS Mode combobox selection."""
        if self.magnet_controller is None:
            return
        try:
            self.magnet_controller.control_mode(text)
            self.magnet_status_lbl.setText(f"Status: Mode set to {text}")
        except Exception as e:
            print(f"Error setting control mode: {e}")
            self.magnet_status_lbl.setText("Status: Error setting mode")

    def on_magnet_heater_clicked(self):
        """Action triggered by the switch heater button."""
        if self.magnet_controller is None:
            return
        
        heater_status = self.magnet_controller.switch_heater()
        target_state = 'off' if 'on' in heater_status else 'on'
        
        self.magnet_ramping = True
        self.magnet_setpoint.setEnabled(False)
        self.magnet_ramp_rate.setEnabled(False)
        self.magnet_mode_combo.setEnabled(False)
        self.magnet_zero_btn.setEnabled(False)
        self.magnet_heater_btn.setEnabled(False)
        self.magnet_stop_btn.setEnabled(True)
        
        if target_state == 'on':
            self.magnet_status_lbl.setText("Status: Starting switch heater activation...")
        else:
            self.magnet_status_lbl.setText("Status: Starting switch heater shutdown...")
            
        self.magnet_worker = SwitchHeaterWorker(self.magnet_controller, target_state, parent=self)
        self.magnet_worker.progress.connect(self.on_magnet_progress)
        self.magnet_worker.status.connect(self.on_magnet_status)
        self.magnet_worker.finished.connect(self.on_magnet_ramp_finished)
        self.magnet_worker.error.connect(self.on_magnet_ramp_error)
        self.magnet_worker.start()
