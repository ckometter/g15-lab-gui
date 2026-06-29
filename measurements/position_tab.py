import os
import json
import math
from PyQt6 import QtWidgets, QtCore, QtGui

class MagnetRampWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float)
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
                        while self.magnet._get_is_sweeping() and not self.stop_requested:
                            self.msleep(100)
                            self.progress.emit(self.magnet._get_output_field())
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

            # Loop while sweeping
            while self.magnet._get_is_sweeping() and not self.stop_requested:
                self.msleep(200)
                current_val = self.magnet._get_output_field()
                self.progress.emit(current_val)

            # Hold magnet
            self.magnet.hold()

            # Turn off heater if fitting
            if getattr(self.magnet, 'SWITCH_HEATER_FITTED', False) and not self.stop_requested:
                self.status.emit("Turning switch heater off...")
                self.magnet._set_switch_heater('off')

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

# Reuse button style from Dual_gate_sweep_gui (assuming it's defined elsewhere or replicate basic style)
button_style = """
QPushButton {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    font-weight: bold;
    border-radius: 4px;
    min-width: 40px;
    min-height: 40px;
    color: #1e293b;
}
QPushButton:hover {
    background-color: #f1f5f9;
}
QPushButton:pressed {
    background-color: #e2e8f0;
}
"""

class PositionTab(QtWidgets.QWidget):
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
        self.setup_ui()
        self.load_storage()

        # Initialize magnet UI values
        if self.magnet_controller is not None:
            try:
                current_field = self.magnet_controller.field()
                self.magnet_live_field_lbl.setText(f"B = {current_field:.4f} T")
                
                target_f = self.magnet_controller.target_field_param()
                self.magnet_setpoint.setValue(target_f)
                
                rate = self.magnet_controller.field_sweep_rate() / 60.0
                self.magnet_ramp_rate.setValue(rate)
            except Exception as e:
                print(f"Error reading initial magnet values: {e}")
        else:
            self.magnet_live_field_lbl.setText("B = No Magnet")
            self.magnet_setpoint.setEnabled(False)
            self.magnet_ramp_rate.setEnabled(False)

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

    def poll_voltages(self):
        # Poll scanner voltages
        if self.scanner_controller is None or getattr(self.scanner_controller, 'simulated', True):
            # No real scanner, skip voltage polling
            pass
        else:
            try:
                vx = self.scanner_controller.get_voltage(1)
                self.vx_label.setText(f"Vx: {vx:.3f} V")
            except Exception as e:
                self.vx_label.setText("Vx: Error")
                print(f"Error polling Vx: {e}")
            try:
                vy = self.scanner_controller.get_voltage(2)
                self.vy_label.setText(f"Vy: {vy:.3f} V")
            except Exception as e:
                self.vy_label.setText("Vy: Error")
                print(f"Error polling Vy: {e}")
        # Poll stepper frequencies and amplitudes
        if self.stepper_controller is None:
            self.sf1_label.setText("Fx: Error")
            self.sf2_label.setText("Fy: Error")
            self.sf3_label.setText("Fz: Error")
            self.am1_label.setText("Ax: Error")
            self.am2_label.setText("Ay: Error")
            self.am3_label.setText("Az: Error")
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
                        label.setText(f"F{ch}: {freq:.0f} Hz")
                except Exception:
                    label.setText(f"F{ch}: Error")
            for axis, label in zip([1, 2, 3], [self.am1_label, self.am2_label, self.am3_label]):
                try:
                    ch = axis_chars[axis]
                    if is_sim:
                        label.setText(f"A{ch}: Simulation")
                    else:
                        amp = getattr(self.stepper_controller, f'voltage_ch{axis}')()
                        label.setText(f"A{ch}: {amp:.1f} V")
                except Exception:
                    label.setText(f"A{ch}: Error")

        # Poll magnet field if not currently ramping
        if self.magnet_controller is not None and not self.magnet_ramping:
            try:
                current_field = self.magnet_controller.field()
                self.magnet_live_field_lbl.setText(f"B = {current_field:.4f} T")
            except Exception as e:
                self.magnet_live_field_lbl.setText("B = Error")
                print(f"Error polling magnet field: {e}")

    def setup_ui(self):
        # Main layout containing stepper and scanner groups
        main_layout = QtWidgets.QHBoxLayout(self)

        # ----- Stepper Group -----
        stepper_group = QtWidgets.QGroupBox("Stepper")
        
        # Grid/Cross layout for XY buttons (wrapped in a fixed-size widget to keep it nicely quadratic)
        cross_widget = QtWidgets.QWidget()
        cross_widget.setFixedSize(160, 160)
        cross_layout = QtWidgets.QGridLayout(cross_widget)
        cross_layout.setContentsMargins(0, 0, 0, 0)
        cross_layout.setSpacing(5)
        
        self.btn_up = QtWidgets.QPushButton("Y ↑")
        self.btn_down = QtWidgets.QPushButton("Y ↓")
        self.btn_left = QtWidgets.QPushButton("X ←")
        self.btn_right = QtWidgets.QPushButton("X →")
        for btn in (self.btn_up, self.btn_down, self.btn_left, self.btn_right):
            btn.setStyleSheet(button_style)
            btn.setFixedSize(50, 50)
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
            btn.setFixedSize(80, 30)
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
        freq_title.setStyleSheet("font-weight: bold; color: #475569;")
        amp_title = QtWidgets.QLabel("Step Amplitude")
        amp_title.setStyleSheet("font-weight: bold; color: #475569;")
        
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
            f_edit.setValidator(QtGui.QIntValidator(1, 8000))
            f_edit.setFixedWidth(55)
            f_edit.returnPressed.connect(lambda ax=axis, edit=f_edit: self.set_stepper_frequency(ax, edit))
            self.f_edits.append(f_edit)
            
            # Amplitude controls
            a_polled_label = getattr(self, f'am{axis}_label')
            a_edit = QtWidgets.QLineEdit()
            a_edit.setPlaceholderText("Set V")
            a_edit.setValidator(QtGui.QIntValidator(0, 70))
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
        stepper_group_layout.addStretch() # Push everything to the top
        stepper_group.setLayout(stepper_group_layout)

        # ----- Scanner Group -----
        scanner_group = QtWidgets.QGroupBox("Scanner")
        # XY cross layout (wrapped in a fixed-size widget to keep it nicely quadratic)
        scanner_cross_widget = QtWidgets.QWidget()
        scanner_cross_widget.setFixedSize(160, 160)
        scanner_top_layout = QtWidgets.QGridLayout(scanner_cross_widget)
        scanner_top_layout.setContentsMargins(0, 0, 0, 0)
        scanner_top_layout.setSpacing(5)
        
        self.s_btn_up = QtWidgets.QPushButton("Y ↑")
        self.s_btn_down = QtWidgets.QPushButton("Y ↓")
        self.s_btn_left = QtWidgets.QPushButton("X ←")
        self.s_btn_right = QtWidgets.QPushButton("X →")
        for btn in (self.s_btn_up, self.s_btn_down, self.s_btn_left, self.s_btn_right):
            btn.setStyleSheet(button_style)
            btn.setFixedSize(50, 50)
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
        voltage_layout = QtWidgets.QVBoxLayout()
        self.vx_label = QtWidgets.QLabel("Vx: 0.0 V")
        self.vy_label = QtWidgets.QLabel("Vy: 0.0 V")
        voltage_layout.addWidget(self.vx_label)
        voltage_layout.addWidget(self.vy_label)
        
        # Combine scanner layouts vertically: buttons, then step size, then voltage
        scanner_group_layout = QtWidgets.QVBoxLayout()
        scanner_group_layout.addWidget(scanner_cross_widget) # Buttons on top
        scanner_group_layout.addLayout(scanner_step_layout) # Then step size
        scanner_group_layout.addLayout(voltage_layout) # Then voltage
        scanner_group_layout.addStretch() # Push everything to the top
        scanner_group.setLayout(scanner_group_layout)

        # ----- Magnet Group -----
        magnet_group = QtWidgets.QGroupBox("Magnet")
        magnet_group_layout = QtWidgets.QVBoxLayout()
        
        # Grid for Setpoint and Ramp Rate
        mag_grid = QtWidgets.QGridLayout()
        mag_grid.setSpacing(6)
        
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
        
        mag_grid.addWidget(QtWidgets.QLabel("Ramp Rate (T/s):"), 1, 0)
        self.magnet_ramp_rate = QtWidgets.QDoubleSpinBox()
        self.magnet_ramp_rate.setRange(0.0001, 0.0175)
        self.magnet_ramp_rate.setDecimals(4)
        self.magnet_ramp_rate.setSingleStep(0.0001)
        self.magnet_ramp_rate.setValue(0.0083)
        self.magnet_ramp_rate.setKeyboardTracking(False)
        self.magnet_ramp_rate.valueChanged.connect(lambda _: self.save_storage())
        mag_grid.addWidget(self.magnet_ramp_rate, 1, 1)
        
        magnet_group_layout.addLayout(mag_grid)
        
        # Magnet field display
        self.magnet_live_field_lbl = QtWidgets.QLabel("B = -- T")
        self.magnet_live_field_lbl.setStyleSheet("color: #1e293b; font-weight: bold;")
        magnet_group_layout.addWidget(self.magnet_live_field_lbl)
        
        # Status Label and Stop Button
        self.magnet_status_lbl = QtWidgets.QLabel("Status: Idle")
        self.magnet_status_lbl.setStyleSheet("color: #475569; font-weight: bold; font-size: 11px;")
        magnet_group_layout.addWidget(self.magnet_status_lbl)
        
        self.magnet_stop_btn = QtWidgets.QPushButton("Stop Ramp")
        self.magnet_stop_btn.setEnabled(False)
        self.magnet_stop_btn.setStyleSheet("background-color: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; font-weight: bold; min-height: 25px; border-radius: 4px;")
        self.magnet_stop_btn.clicked.connect(self.on_magnet_stop_clicked)
        magnet_group_layout.addWidget(self.magnet_stop_btn)
        
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
        self.setLayout(main_layout)
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
        
        self.load_storage()



    def _get_step(self, line_edit: QtWidgets.QLineEdit) -> float:
        try:
            return float(line_edit.text())
        except ValueError:
            return 0.0

    # ----- Placeholder hardware interaction -----
    def move_stepper(self, axis: str, direction: int):
        if self.stepper_controller is None:
            print("Stepper controller not initialized.")
            return
        # Map axis letters to driver axis numbers (1=x, 2=y, 3=z)
        axis_map = {'x': 1, 'y': 2, 'z': 3}
        driver_axis = axis_map.get(axis.lower())
        if driver_axis is None:
            print(f"Unknown axis {axis} for stepper.")
            return
        # A quick click triggers exactly 1 step in the specified direction
        try:
            self.stepper_controller.step(driver_axis, direction)
        except Exception as e:
            print(f"Failed to move stepper: {e}")

    def move_scanner(self, axis: str, direction: int):
        step = self._get_step(self.s_step_size_edit)
        if self.scanner_controller is None:
            print("Scanner controller not initialized.")
            return
        # Map axis letters to driver axis numbers (1=x, 2=y, 3=z)
        axis_map = {'x': 1, 'y': 2, 'z': 3}
        driver_axis = axis_map.get(axis.lower())
        if driver_axis is None:
            print(f"Unknown axis {axis} for scanner.")
            return
        # Current voltage from driver (fallback to displayed value if driver missing)
        try:
            current_volt = self.scanner_controller.get_voltage(driver_axis)
        except Exception:
            # Fallback: parse from label
            try:
                if axis == 'x':
                    current_volt = float(self.vx_label.text().split(':')[1].split()[0])
                else:
                    current_volt = float(self.vy_label.text().split(':')[1].split()[0])
            except Exception:
                current_volt = 0.0
        new_volt = current_volt + direction * step
        # Clamp between 0.0 and 150.0
        new_volt = max(0.0, min(150.0, new_volt))
        if new_volt == current_volt:
            return
        try:
            self.scanner_controller.set_voltage(driver_axis, new_volt)
            # Update displayed voltages and save storage only on success
            if axis == 'x':
                self.vx_label.setText(f"Vx: {new_volt:.3f} V")
            elif axis == 'y':
                self.vy_label.setText(f"Vy: {new_volt:.3f} V")
            self.save_storage()
        except Exception as e:
            print(f"Failed to set scanner voltage: {e}")

    def load_storage(self):
        storage_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value_storage")
        vx, vy = 0.0, 0.0
        magnet_setpoint = 0.0
        magnet_ramp_rate = 0.0083
        try:
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    data = json.load(f)
                if "position" in data:
                    vx = data["position"].get("vx", 0.0)
                    vy = data["position"].get("vy", 0.0)
                    magnet_setpoint = data["position"].get("magnet_setpoint", 0.0)
                    magnet_ramp_rate = data["position"].get("magnet_ramp_rate", 0.0083)
        except Exception as e:
            print(f"Error loading value_storage in PositionTab: {e}")
        self.vx_label.setText(f"Vx: {vx:.3f} V")
        self.vy_label.setText(f"Vy: {vy:.3f} V")
        
        # Restore magnet UI values from storage
        self.magnet_setpoint.blockSignals(True)
        self.magnet_setpoint.setValue(magnet_setpoint)
        self.magnet_setpoint.blockSignals(False)

        self.magnet_ramp_rate.blockSignals(True)
        self.magnet_ramp_rate.setValue(magnet_ramp_rate)
        self.magnet_ramp_rate.blockSignals(False)

    def save_storage(self):
        storage_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value_storage")
        try:
            # Parse Vx and Vy from labels
            try:
                vx = float(self.vx_label.text().split(':')[1].split()[0])
            except Exception:
                vx = 0.0
            try:
                vy = float(self.vy_label.text().split(':')[1].split()[0])
            except Exception:
                vy = 0.0

            data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    data = json.load(f)
            if "position" not in data:
                data["position"] = {}
            data["position"]["vx"] = vx
            data["position"]["vy"] = vy
            data["position"]["magnet_setpoint"] = self.magnet_setpoint.value()
            data["position"]["magnet_ramp_rate"] = self.magnet_ramp_rate.value()
            with open(storage_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving value_storage in PositionTab: {e}")

    def reload_gui_from_storage(self):
        self.load_storage()

    def on_button_pressed(self, btn, axis: str, direction: int):
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
        if self.stepper_controller is None:
            print("Stepper controller not initialized for continuous move.")
            return
        axis_map = {'x': 1, 'y': 2, 'z': 3}
        driver_axis = axis_map.get(axis.lower())
        if driver_axis is None:
            return
        try:
            if hasattr(self.stepper_controller, 'start_continuous'):
                self.stepper_controller.start_continuous(driver_axis, direction)
            else:
                print("start_continuous method not supported by the driver.")
        except Exception as e:
            print(f"Failed to start continuous stepper move: {e}")

    def stop_continuous_stepper(self, axis: str):
        if self.stepper_controller is None:
            return
        axis_map = {'x': 1, 'y': 2, 'z': 3}
        driver_axis = axis_map.get(axis.lower())
        if driver_axis is None:
            return
        try:
            self.stepper_controller.stop(driver_axis)
        except Exception as e:
            print(f"Failed to stop continuous stepper move: {e}")

    def set_stepper_frequency(self, axis: int, edit: QtWidgets.QLineEdit):
        if self.stepper_controller is None:
            print("Stepper controller not initialized.")
            return
        text = edit.text()
        if not text:
            return
        try:
            val = int(text)
            getattr(self.stepper_controller, f'frequency_ch{axis}')(val)
            edit.clear()
            edit.clearFocus()
        except ValueError:
            pass
        except Exception as e:
            print(f"Failed to set stepper frequency: {e}")

    def set_stepper_amplitude(self, axis: int, edit: QtWidgets.QLineEdit):
        if self.stepper_controller is None:
            print("Stepper controller not initialized.")
            return
        text = edit.text()
        if not text:
            return
        try:
            val = float(text)
            getattr(self.stepper_controller, f'voltage_ch{axis}')(val)
            edit.clear()
            edit.clearFocus()
        except ValueError:
            pass
        except Exception as e:
            print(f"Failed to set stepper amplitude: {e}")

    def on_magnet_setpoint_enter(self):
        """Triggered when user presses Enter inside the magnet setpoint spinbox."""
        if self.magnet_controller is None:
            return
        
        target = self.magnet_setpoint.value()
        rate = self.magnet_ramp_rate.value()
        
        self.magnet_ramping = True
        self.magnet_setpoint.setEnabled(False)
        self.magnet_ramp_rate.setEnabled(False)
        self.magnet_stop_btn.setEnabled(True)
        self.magnet_status_lbl.setText(f"Status: Starting ramp to {target:.4f} T...")
        
        self.magnet_worker = MagnetRampWorker(self.magnet_controller, target, rate, parent=self)
        self.magnet_worker.progress.connect(self.on_magnet_progress)
        self.magnet_worker.status.connect(self.on_magnet_status)
        self.magnet_worker.finished.connect(self.on_magnet_ramp_finished)
        self.magnet_worker.error.connect(self.on_magnet_ramp_error)
        self.magnet_worker.start()

    def on_magnet_progress(self, val):
        self.magnet_live_field_lbl.setText(f"B = {val:.4f} T")
        target = self.magnet_setpoint.value()
        self.magnet_status_lbl.setText(f"Ramping to {target:.4f} T (Currently at {val:.4f} T)")

    def on_magnet_status(self, msg):
        self.magnet_status_lbl.setText(msg)

    def on_magnet_ramp_finished(self):
        if self.magnet_worker and self.magnet_worker.stop_requested:
            self.magnet_status_lbl.setText("Status: Ramping stopped/interrupted.")
        else:
            self.magnet_status_lbl.setText("Status: Ramping complete.")
            if self.magnet_controller is not None:
                try:
                    current_field = self.magnet_controller.field()
                    self.magnet_live_field_lbl.setText(f"B = {current_field:.4f} T")
                except Exception:
                    pass
        
        self.magnet_ramping = False
        self.magnet_setpoint.setEnabled(True)
        self.magnet_ramp_rate.setEnabled(True)
        self.magnet_stop_btn.setEnabled(False)

    def on_magnet_ramp_error(self, err_msg):
        self.magnet_status_lbl.setText("Status: Ramping error.")
        self.magnet_ramping = False
        self.magnet_setpoint.setEnabled(True)
        self.magnet_ramp_rate.setEnabled(True)
        self.magnet_stop_btn.setEnabled(False)
        QtWidgets.QMessageBox.critical(self, "Magnet Ramping Error", f"Error ramping magnet:\n{err_msg}")

    def on_magnet_stop_clicked(self):
        if self.magnet_worker:
            self.magnet_worker.stop()
            self.magnet_status_lbl.setText("Status: Interrupting ramp...")
