# -*- coding: utf-8 -*-
"""
PyQt6 GUI for Polarization Control.
"""
import os
import sys
import json
import math
import time
from PyQt6 import QtCore, QtGui, QtWidgets

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def normalize_angle(angle):
    a = float(angle) % 360.0
    if a > 180.0:
        a -= 360.0
    a = round(a, 2)
    if a == -180.0:
        a = 180.0
    return a

class PolarizationDial(QtWidgets.QWidget):
    """
    A custom circular dial showing:
    - Target Vertical angle (dashed green line)
    - Target Horizontal angle (dashed red line)
    - Current angle (solid indigo needle with circle tip)
    
    Supports drag-and-drop manual tuning: updates visuals on drag, commands rotator on drop.
    """
    angleDragging = QtCore.pyqtSignal(float)
    angleTuned = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_angle = 0.0
        self.vertical_angle = 0.0
        self.horizontal_angle = 90.0
        self.setMinimumSize(240, 240)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def set_angles(self, current, vertical, horizontal):
        self.current_angle = current
        self.vertical_angle = vertical
        self.horizontal_angle = horizontal
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            angle = self.calculate_angle(event.position())
            self.angleDragging.emit(angle)

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            angle = self.calculate_angle(event.position())
            self.angleDragging.emit(angle)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            angle = self.calculate_angle(event.position())
            self.angleTuned.emit(angle)

    def calculate_angle(self, pos):
        width = self.width()
        height = self.height()
        center = QtCore.QPointF(width / 2.0, height / 2.0)
        dx = pos.x() - center.x()
        dy = -(pos.y() - center.y())  # Invert Y for Cartesian (90° is UP)
        rad = math.atan2(dy, dx)
        angle = math.degrees(rad)
        # Map to (-180, 180]
        angle = round(angle, 2)
        if angle == -180.0:
            angle = 180.0
        return angle

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        size = min(width, height) - 20
        rect = QtCore.QRectF((width - size)/2, (height - size)/2, size, size)

        # Draw circle background
        pen = QtGui.QPen(QtGui.QColor("#cbd5e1"), 2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#f8fafc")))
        painter.drawEllipse(rect)

        center = rect.center()
        
        # Center marker
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#1e293b")))
        painter.drawEllipse(center, 4, 4)

        radius = size / 2.0
        painter.setFont(QtGui.QFont("Segoe UI", 9))

        # Helper to convert Cartesian angles to QPointF
        def get_point(angle_deg, dist):
            rad = math.radians(angle_deg)
            return center + QtCore.QPointF(math.cos(rad) * dist, -math.sin(rad) * dist)

        # Draw ticks and labels
        for angle_deg in range(0, 360, 30):
            p1 = get_point(angle_deg, radius - 8)
            p2 = get_point(angle_deg, radius)
            painter.setPen(QtGui.QPen(QtGui.QColor("#94a3b8"), 1.5))
            painter.drawLine(p1, p2)

            if angle_deg in (0, 90, 180, 270):
                painter.setPen(QtGui.QPen(QtGui.QColor("#475569")))
                label = f"{angle_deg}°"
                tp = get_point(angle_deg, radius - 20)
                painter.drawText(int(tp.x() - 15), int(tp.y() - 10), 30, 20, QtCore.Qt.AlignmentFlag.AlignCenter, label)

        # Draw configured vertical target line (dashed green)
        v_p = get_point(self.vertical_angle, radius - 10)
        painter.setPen(QtGui.QPen(QtGui.QColor("#10b981"), 2, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(center, v_p)

        # Draw configured horizontal target line (dashed red)
        h_p = get_point(self.horizontal_angle, radius - 10)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ef4444"), 2, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(center, h_p)

        # Draw current angle needle (solid indigo)
        curr_p = get_point(self.current_angle, radius - 5)
        painter.setPen(QtGui.QPen(QtGui.QColor("#6366f1"), 3.5))
        painter.drawLine(center, curr_p)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#6366f1")))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(curr_p, 6, 6)


class PolarizationWorker(QtCore.QThread):
    """
    Background worker thread to interact with Thorlabs rotators asynchronously.
    """
    angles_updated = QtCore.pyqtSignal(float, float, float)
    status_updated = QtCore.pyqtSignal(str)
    move_finished = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, station, parent=None):
        super().__init__(parent)
        self.station = station
        self._is_running = True
        self.cmd_queue = []
        self.moving = False
        self.mutex = QtCore.QMutex()
        
        self._sim_exc_angle = 0.0
        self._sim_coll_angle = 0.0
        self._sim_rot3_angle = 0.0

    def queue_move_excitation(self, angle):
        self.mutex.lock()
        self.cmd_queue = [c for c in self.cmd_queue if c[0] != "move_exc"]
        self.cmd_queue.append(("move_exc", angle))
        self.mutex.unlock()

    def queue_move_collection(self, angle):
        self.mutex.lock()
        self.cmd_queue = [c for c in self.cmd_queue if c[0] != "move_coll"]
        self.cmd_queue.append(("move_coll", angle))
        self.mutex.unlock()

    def queue_move_rot3(self, angle):
        self.mutex.lock()
        self.cmd_queue = [c for c in self.cmd_queue if c[0] != "move_rot3"]
        self.cmd_queue.append(("move_rot3", angle))
        self.mutex.unlock()

    def stop(self):
        self.mutex.lock()
        self._is_running = False
        self.mutex.unlock()

    def run(self):
        rotator1 = getattr(self.station, "rotator1", None)
        rotator2 = getattr(self.station, "rotator2", None)
        rotator3 = getattr(self.station, "rotator3", None)

        while True:
            self.mutex.lock()
            if not self._is_running:
                self.mutex.unlock()
                break
            
            cmd = None
            if len(self.cmd_queue) > 0:
                cmd = self.cmd_queue.pop(0)
            self.mutex.unlock()

            if cmd is not None:
                self.moving = True
                cmd_type, angle = cmd
                if cmd_type == "move_exc":
                    if rotator1 and not rotator1.simulated:
                        self.status_updated.emit("Moving Excitation Waveplate...")
                        try:
                            rotator1.angle(angle)
                            self.move_finished.emit("excitation")
                        except Exception as e:
                            self.error_occurred.emit(f"Excitation rotator error: {e}")
                    else:
                        self.status_updated.emit("Simulating Excitation Move...")
                        self.msleep(150)
                        self._sim_exc_angle = angle
                        self.move_finished.emit("excitation")
                elif cmd_type == "move_coll":
                    if rotator2 and not rotator2.simulated:
                        self.status_updated.emit("Moving Collection Waveplate...")
                        try:
                            rotator2.angle(angle)
                            self.move_finished.emit("collection")
                        except Exception as e:
                            self.error_occurred.emit(f"Collection rotator error: {e}")
                    else:
                        self.status_updated.emit("Simulating Collection Move...")
                        self.msleep(150)
                        self._sim_coll_angle = angle
                        self.move_finished.emit("collection")
                elif cmd_type == "move_rot3":
                    if rotator3 and not rotator3.simulated:
                        self.status_updated.emit("Moving Rotator 3...")
                        try:
                            rotator3.angle(angle)
                            self.move_finished.emit("rotator3")
                        except Exception as e:
                            self.error_occurred.emit(f"Rotator 3 error: {e}")
                    else:
                        self.status_updated.emit("Simulating Rotator 3 Move...")
                        self.msleep(150)
                        self._sim_rot3_angle = angle
                        self.move_finished.emit("rotator3")
            else:
                exc_ang = self._sim_exc_angle
                coll_ang = self._sim_coll_angle
                rot3_ang = self._sim_rot3_angle
                
                try:
                    if rotator1 and not rotator1.simulated:
                        exc_ang = rotator1.angle()
                    if rotator2 and not rotator2.simulated:
                        coll_ang = rotator2.angle()
                    if rotator3 and not rotator3.simulated:
                        rot3_ang = rotator3.angle()
                    
                    self.angles_updated.emit(exc_ang, coll_ang, rot3_ang)
                except Exception:
                    pass

            self.moving = False
            self.msleep(300)


class PolarizationControlWindow(QtWidgets.QMainWindow):
    """
    Main Polarization Control Tab GUI
    """
    def __init__(self, station, embed=False):
        super().__init__()
        self.station = station
        self._embed = embed
        self.setWindowTitle("Polarization control")
        self.setMinimumSize(1200, 640)

        self.load_storage()
        self.current_exc_angle = self.stored_settings.get("pol_exc_actual", 0.0)
        self.current_coll_angle = self.stored_settings.get("pol_coll_actual", 0.0)
        self.current_rot3_angle = self.stored_settings.get("pol_rot3_actual", 0.0)
        self.last_saved_exc_angle = self.current_exc_angle
        self.last_saved_coll_angle = self.current_coll_angle
        self.last_saved_rot3_angle = self.current_rot3_angle

        self.setup_ui()
        self.apply_theme()

        # Start Background worker
        self.worker = PolarizationWorker(self.station)
        self.worker._sim_exc_angle = self.current_exc_angle
        self.worker._sim_coll_angle = self.current_coll_angle
        self.worker._sim_rot3_angle = self.current_rot3_angle
        self.worker.angles_updated.connect(self.on_angles_updated)
        self.worker.status_updated.connect(self.on_status_updated)
        self.worker.move_finished.connect(self.on_move_finished)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.start()

    def load_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        defaults = {
            "pol_exc_actual": 0.0,
            "pol_coll_actual": 0.0,
            "pol_rot3_actual": 0.0
        }
        self.stored_settings = defaults.copy()
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    raw_data = json.load(f)
                pol_data = raw_data.get("polarization", {})
                self.stored_settings["pol_exc_actual"] = pol_data.get("pol_exc_actual", raw_data.get("pol_exc_actual", 0.0))
                self.stored_settings["pol_coll_actual"] = pol_data.get("pol_coll_actual", raw_data.get("pol_coll_actual", 0.0))
                self.stored_settings["pol_rot3_actual"] = pol_data.get("pol_rot3_actual", raw_data.get("pol_rot3_actual", 0.0))
            except Exception as e:
                print(f"Error loading value_storage: {e}")

    def save_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        try:
            current_data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    current_data = json.load(f)
            
            if "polarization" not in current_data:
                current_data["polarization"] = {}
            current_data["polarization"]["pol_exc_actual"] = self.current_exc_angle
            current_data["polarization"]["pol_coll_actual"] = self.current_coll_angle
            current_data["polarization"]["pol_rot3_actual"] = self.current_rot3_angle
            
            # Update last saved tracking attributes
            self.last_saved_exc_angle = self.current_exc_angle
            self.last_saved_coll_angle = self.current_coll_angle
            self.last_saved_rot3_angle = self.current_rot3_angle

            # Purge deprecated root-level keys
            deprecated_keys = [
                "exposure_time", "settling_time", "ramprate1", "ramprate2", "ramp_step", 
                "c1_name", "c2_name", "grating", "wavelength", "target_temp", "num_spectra", 
                "average", "accum_method", "plot_mode", "x_axis_scale", "plot_modes", 
                "b_coord_name", "b_start", "b_end", "b_pts", "v_start", "v_end", "v_pts", 
                "v_fixed", "variable_order", "serpentine", "faraday_comp", "faraday_rot", 
                "end_ramp_zero", "end_ramp_ground", "bv_coeff_a", "bv_coeff_b", "bv_coeff_c", 
                "bv_coeff_d", "pol_exc_vertical", "pol_exc_horizontal", "pol_exc_manual", 
                "pol_coll_vertical", "pol_coll_horizontal", "pol_coll_manual",
                "pol_rot3_vertical", "pol_rot3_horizontal", "pol_rot3_manual"
            ]
            for key in deprecated_keys:
                current_data.pop(key, None)

            with open(storage_path, "w") as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print(f"Error saving to value_storage: {e}")

    def reload_gui_from_storage(self):
        self.load_storage()
        self.current_exc_angle = self.stored_settings.get("pol_exc_actual", 0.0)
        self.current_coll_angle = self.stored_settings.get("pol_coll_actual", 0.0)
        self.current_rot3_angle = self.stored_settings.get("pol_rot3_actual", 0.0)
        self.last_saved_exc_angle = self.current_exc_angle
        self.last_saved_coll_angle = self.current_coll_angle
        self.last_saved_rot3_angle = self.current_rot3_angle
        if hasattr(self, "worker"):
            self.worker._sim_exc_angle = self.current_exc_angle
            self.worker._sim_coll_angle = self.current_coll_angle
            self.worker._sim_rot3_angle = self.current_rot3_angle
        self.lbl_exc_angle.setText(f"{self.current_exc_angle:.2f}°")
        self.lbl_coll_angle.setText(f"{self.current_coll_angle:.2f}°")
        self.lbl_rot3_angle.setText(f"{self.current_rot3_angle:.2f}°")
        self.update_dials()

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Split horizontally into Excitation (Left) and Collection (Right)
        cols_layout = QtWidgets.QHBoxLayout()
        cols_layout.setSpacing(20)

        # --- Excitation Card (Left) ---
        exc_card = QtWidgets.QFrame()
        exc_card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        exc_card.setStyleSheet("QFrame { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 0px; }")
        exc_layout = QtWidgets.QVBoxLayout(exc_card)
        exc_layout.setContentsMargins(20, 20, 20, 20)
        exc_layout.setSpacing(15)

        exc_title = QtWidgets.QLabel("Excitation Waveplate (Rotator 1)")
        exc_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #000000; border: none;")
        exc_layout.addWidget(exc_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Visual Dial (Supports mouse drag manual tuning)
        self.exc_dial_plot = PolarizationDial()
        self.exc_dial_plot.angleDragging.connect(self.on_exc_dial_dragging)
        self.exc_dial_plot.angleTuned.connect(self.on_exc_dial_tuned)
        exc_layout.addWidget(self.exc_dial_plot, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Readout text
        self.lbl_exc_angle = QtWidgets.QLabel("0.00°")
        self.lbl_exc_angle.setStyleSheet("font-family: Consolas, monospace; font-size: 28px; font-weight: bold; color: #000000; border: none;")
        self.lbl_exc_angle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        exc_layout.addWidget(self.lbl_exc_angle)

        # Configurations grid
        exc_grid = QtWidgets.QGridLayout()
        exc_grid.setSpacing(10)
        exc_grid.setColumnStretch(1, 1)

        # Vertical and Horizontal Buttons
        h_layout = QtWidgets.QHBoxLayout()
        self.btn_exc_vertical = QtWidgets.QPushButton("Vertical (90.0°)")
        self.btn_exc_vertical.setStyleSheet("background-color: #000000; color: white; font-weight: bold; border-radius: 0px; padding: 8px;")
        self.btn_exc_vertical.clicked.connect(self.move_exc_vertical)
        h_layout.addWidget(self.btn_exc_vertical)

        self.btn_exc_horizontal = QtWidgets.QPushButton("Horizontal (0.0°)")
        self.btn_exc_horizontal.setStyleSheet("background-color: #555555; color: white; font-weight: bold; border-radius: 0px; padding: 8px;")
        self.btn_exc_horizontal.clicked.connect(self.move_exc_horizontal)
        h_layout.addWidget(self.btn_exc_horizontal)
        exc_grid.addLayout(h_layout, 0, 0, 1, 3)

        # Manual Angle input
        lbl_m = QtWidgets.QLabel("Manual Angle:")
        lbl_m.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        exc_grid.addWidget(lbl_m, 1, 0)
        self.exc_manual_spin = QtWidgets.QDoubleSpinBox()
        self.exc_manual_spin.setRange(-10000.0, 10000.0)
        self.exc_manual_spin.setDecimals(2)
        self.exc_manual_spin.setValue(self.stored_settings.get("pol_exc_manual", 0.0))
        self.exc_manual_spin.editingFinished.connect(self.move_exc_manual)
        exc_grid.addWidget(self.exc_manual_spin, 1, 1)
        self.btn_exc_go_manual = QtWidgets.QPushButton("Go")
        self.btn_exc_go_manual.setStyleSheet("background-color: #000000; color: white; font-weight: bold; border-radius: 0px; padding: 4px 12px;")
        self.btn_exc_go_manual.clicked.connect(self.move_exc_manual)
        exc_grid.addWidget(self.btn_exc_go_manual, 1, 2)

        # Relative Angle input
        lbl_rel = QtWidgets.QLabel("Relative Angle:")
        lbl_rel.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        exc_grid.addWidget(lbl_rel, 2, 0)
        self.exc_relative_spin = QtWidgets.QDoubleSpinBox()
        self.exc_relative_spin.setRange(-360.0, 360.0)
        self.exc_relative_spin.setDecimals(2)
        self.exc_relative_spin.setValue(0.0)
        exc_grid.addWidget(self.exc_relative_spin, 2, 1)
        self.btn_exc_go_relative = QtWidgets.QPushButton("Go")
        self.btn_exc_go_relative.setStyleSheet("background-color: #555555; color: white; font-weight: bold; border-radius: 0px; padding: 4px 12px;")
        self.btn_exc_go_relative.clicked.connect(self.move_exc_relative)
        exc_grid.addWidget(self.btn_exc_go_relative, 2, 2)

        lbl_recal = QtWidgets.QLabel("Recalibrate stage:")
        lbl_recal.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        exc_grid.addWidget(lbl_recal, 3, 0)
        self.btn_exc_home = QtWidgets.QPushButton("Home Stage (0.0°)")
        self.btn_exc_home.setStyleSheet("background-color: #ffffff; color: #000000; font-weight: bold; border-radius: 0px; padding: 4px; border: 1px solid #cccccc;")
        self.btn_exc_home.clicked.connect(self.home_exc)
        exc_grid.addWidget(self.btn_exc_home, 3, 1, 1, 2)

        exc_layout.addLayout(exc_grid)
        cols_layout.addWidget(exc_card, 1)


        # --- Collection Card (Right) ---
        coll_card = QtWidgets.QFrame()
        coll_card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        coll_card.setStyleSheet("QFrame { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 0px; }")
        coll_layout = QtWidgets.QVBoxLayout(coll_card)
        coll_layout.setContentsMargins(20, 20, 20, 20)
        coll_layout.setSpacing(15)

        coll_title = QtWidgets.QLabel("Collection Waveplate (Rotator 2)")
        coll_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #000000; border: none;")
        coll_layout.addWidget(coll_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Visual Dial (Supports mouse drag manual tuning)
        self.coll_dial_plot = PolarizationDial()
        self.coll_dial_plot.angleDragging.connect(self.on_coll_dial_dragging)
        self.coll_dial_plot.angleTuned.connect(self.on_coll_dial_tuned)
        coll_layout.addWidget(self.coll_dial_plot, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Readout text
        self.lbl_coll_angle = QtWidgets.QLabel("0.00°")
        self.lbl_coll_angle.setStyleSheet("font-family: Consolas, monospace; font-size: 28px; font-weight: bold; color: #000000; border: none;")
        self.lbl_coll_angle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        coll_layout.addWidget(self.lbl_coll_angle)

        # Configurations grid
        coll_grid = QtWidgets.QGridLayout()
        coll_grid.setSpacing(10)
        coll_grid.setColumnStretch(1, 1)

        # Vertical and Horizontal Buttons
        h_layout2 = QtWidgets.QHBoxLayout()
        self.btn_coll_vertical = QtWidgets.QPushButton("Vertical (90.0°)")
        self.btn_coll_vertical.setStyleSheet("background-color: #000000; color: white; font-weight: bold; border-radius: 0px; padding: 8px;")
        self.btn_coll_vertical.clicked.connect(self.move_coll_vertical)
        h_layout2.addWidget(self.btn_coll_vertical)

        self.btn_coll_horizontal = QtWidgets.QPushButton("Horizontal (0.0°)")
        self.btn_coll_horizontal.setStyleSheet("background-color: #555555; color: white; font-weight: bold; border-radius: 0px; padding: 8px;")
        self.btn_coll_horizontal.clicked.connect(self.move_coll_horizontal)
        h_layout2.addWidget(self.btn_coll_horizontal)
        coll_grid.addLayout(h_layout2, 0, 0, 1, 3)

        # Manual Angle input
        lbl_cm = QtWidgets.QLabel("Manual Angle:")
        lbl_cm.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        coll_grid.addWidget(lbl_cm, 1, 0)
        self.coll_manual_spin = QtWidgets.QDoubleSpinBox()
        self.coll_manual_spin.setRange(-10000.0, 10000.0)
        self.coll_manual_spin.setDecimals(2)
        self.coll_manual_spin.setValue(self.stored_settings.get("pol_coll_manual", 0.0))
        self.coll_manual_spin.editingFinished.connect(self.move_coll_manual)
        coll_grid.addWidget(self.coll_manual_spin, 1, 1)
        self.btn_coll_go_manual = QtWidgets.QPushButton("Go")
        self.btn_coll_go_manual.setStyleSheet("background-color: #000000; color: white; font-weight: bold; border-radius: 0px; padding: 4px 12px;")
        self.btn_coll_go_manual.clicked.connect(self.move_coll_manual)
        coll_grid.addWidget(self.btn_coll_go_manual, 1, 2)

        # Relative Angle input
        lbl_crel = QtWidgets.QLabel("Relative Angle:")
        lbl_crel.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        coll_grid.addWidget(lbl_crel, 2, 0)
        self.coll_relative_spin = QtWidgets.QDoubleSpinBox()
        self.coll_relative_spin.setRange(-360.0, 360.0)
        self.coll_relative_spin.setDecimals(2)
        self.coll_relative_spin.setValue(0.0)
        coll_grid.addWidget(self.coll_relative_spin, 2, 1)
        self.btn_coll_go_relative = QtWidgets.QPushButton("Go")
        self.btn_coll_go_relative.setStyleSheet("background-color: #555555; color: white; font-weight: bold; border-radius: 0px; padding: 4px 12px;")
        self.btn_coll_go_relative.clicked.connect(self.move_coll_relative)
        coll_grid.addWidget(self.btn_coll_go_relative, 2, 2)

        lbl_crecal = QtWidgets.QLabel("Recalibrate stage:")
        lbl_crecal.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        coll_grid.addWidget(lbl_crecal, 3, 0)
        self.btn_coll_home = QtWidgets.QPushButton("Home Stage (0.0°)")
        self.btn_coll_home.setStyleSheet("background-color: #ffffff; color: #000000; font-weight: bold; border-radius: 0px; padding: 4px; border: 1px solid #cccccc;")
        self.btn_coll_home.clicked.connect(self.home_coll)
        coll_grid.addWidget(self.btn_coll_home, 3, 1, 1, 2)

        coll_layout.addLayout(coll_grid)
        cols_layout.addWidget(coll_card, 1)

        # --- Rotator 3 Card (Right) ---
        rot3_card = QtWidgets.QFrame()
        rot3_card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        rot3_card.setStyleSheet("QFrame { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 0px; }")
        rot3_layout = QtWidgets.QVBoxLayout(rot3_card)
        rot3_layout.setContentsMargins(20, 20, 20, 20)
        rot3_layout.setSpacing(15)

        rot3_title = QtWidgets.QLabel("Rotator 3 (Rotator 3)")
        rot3_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #000000; border: none;")
        rot3_layout.addWidget(rot3_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Visual Dial (Supports mouse drag manual tuning)
        self.rot3_dial_plot = PolarizationDial()
        self.rot3_dial_plot.angleDragging.connect(self.on_rot3_dial_dragging)
        self.rot3_dial_plot.angleTuned.connect(self.on_rot3_dial_tuned)
        rot3_layout.addWidget(self.rot3_dial_plot, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Readout text
        self.lbl_rot3_angle = QtWidgets.QLabel("0.00°")
        self.lbl_rot3_angle.setStyleSheet("font-family: Consolas, monospace; font-size: 28px; font-weight: bold; color: #000000; border: none;")
        self.lbl_rot3_angle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rot3_layout.addWidget(self.lbl_rot3_angle)

        # Configurations grid
        rot3_grid = QtWidgets.QGridLayout()
        rot3_grid.setSpacing(10)
        rot3_grid.setColumnStretch(1, 1)

        # Vertical and Horizontal Buttons
        h_layout_rot3 = QtWidgets.QHBoxLayout()
        self.btn_rot3_vertical = QtWidgets.QPushButton("Vertical (90.0°)")
        self.btn_rot3_vertical.setStyleSheet("background-color: #000000; color: white; font-weight: bold; border-radius: 0px; padding: 8px;")
        h_layout_rot3.addWidget(self.btn_rot3_vertical)
        self.btn_rot3_horizontal = QtWidgets.QPushButton("Horizontal (0.0°)")
        self.btn_rot3_horizontal.setStyleSheet("background-color: #555555; color: white; font-weight: bold; border-radius: 0px; padding: 8px;")
        self.btn_rot3_horizontal.clicked.connect(self.move_rot3_horizontal)
        h_layout3 = QtWidgets.QHBoxLayout()
        self.btn_rot3_vertical = QtWidgets.QPushButton("Vertical (90.0°)")
        self.btn_rot3_vertical.setStyleSheet("background-color: #000000; color: white; font-weight: bold; border-radius: 0px; padding: 8px;")
        self.btn_rot3_vertical.clicked.connect(self.move_rot3_vertical)
        h_layout3.addWidget(self.btn_rot3_vertical)
        h_layout3.addWidget(self.btn_rot3_horizontal)
        rot3_grid.addLayout(h_layout3, 0, 0, 1, 3)

        # Manual Angle input
        lbl_r3m = QtWidgets.QLabel("Manual Angle:")
        lbl_r3m.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        rot3_grid.addWidget(lbl_r3m, 1, 0)
        self.rot3_manual_spin = QtWidgets.QDoubleSpinBox()
        self.rot3_manual_spin.setRange(-10000.0, 10000.0)
        self.rot3_manual_spin.setDecimals(2)
        self.rot3_manual_spin.setValue(self.stored_settings.get("pol_rot3_manual", 0.0))
        self.rot3_manual_spin.editingFinished.connect(self.move_rot3_manual)
        rot3_grid.addWidget(self.rot3_manual_spin, 1, 1)
        self.btn_rot3_go_manual = QtWidgets.QPushButton("Go")
        self.btn_rot3_go_manual.setStyleSheet("background-color: #000000; color: white; font-weight: bold; border-radius: 0px; padding: 4px 12px;")
        self.btn_rot3_go_manual.clicked.connect(self.move_rot3_manual)
        rot3_grid.addWidget(self.btn_rot3_go_manual, 1, 2)

        # Relative Angle input
        lbl_r3rel = QtWidgets.QLabel("Relative Angle:")
        lbl_r3rel.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        rot3_grid.addWidget(lbl_r3rel, 2, 0)
        self.rot3_relative_spin = QtWidgets.QDoubleSpinBox()
        self.rot3_relative_spin.setRange(-360.0, 360.0)
        self.rot3_relative_spin.setDecimals(2)
        self.rot3_relative_spin.setValue(0.0)
        rot3_grid.addWidget(self.rot3_relative_spin, 2, 1)
        self.btn_rot3_go_relative = QtWidgets.QPushButton("Go")
        self.btn_rot3_go_relative.setStyleSheet("background-color: #555555; color: white; font-weight: bold; border-radius: 0px; padding: 4px 12px;")
        self.btn_rot3_go_relative.clicked.connect(self.move_rot3_relative)
        rot3_grid.addWidget(self.btn_rot3_go_relative, 2, 2)

        lbl_r3recal = QtWidgets.QLabel("Recalibrate stage:")
        lbl_r3recal.setStyleSheet("border: none; font-weight: bold; color: #000000;")
        rot3_grid.addWidget(lbl_r3recal, 3, 0)
        self.btn_rot3_home = QtWidgets.QPushButton("Home Stage (0.0°)")
        self.btn_rot3_home.setStyleSheet("background-color: #ffffff; color: #000000; font-weight: bold; border-radius: 0px; padding: 4px; border: 1px solid #cccccc;")
        self.btn_rot3_home.clicked.connect(self.home_rot3)
        rot3_grid.addWidget(self.btn_rot3_home, 3, 1, 1, 2)

        rot3_layout.addLayout(rot3_grid)
        cols_layout.addWidget(rot3_card, 1)

        main_layout.addLayout(cols_layout)

        # Bottom status readout
        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #000000; italic; border: none; font-size: 13px;")
        main_layout.addWidget(self.lbl_status)

        self.update_dials()

    def update_dials(self):
        self.exc_dial_plot.set_angles(
            self.current_exc_angle, 
            90.0, 
            0.0
        )
        self.coll_dial_plot.set_angles(
            self.current_coll_angle, 
            90.0, 
            0.0
        )
        self.rot3_dial_plot.set_angles(
            self.current_rot3_angle, 
            90.0, 
            0.0
        )

    def on_settings_changed(self):
        self.update_dials()
        self.save_storage()

    def normalize_spinbox_values(self):
        for spin in (self.exc_manual_spin, self.coll_manual_spin, self.rot3_manual_spin):
            val = normalize_angle(spin.value())
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self.update_dials()
        self.save_storage()

    # Move operations
    def move_exc_vertical(self):
        self.worker.queue_move_excitation(90.0)
        self.save_storage()

    def move_exc_horizontal(self):
        self.worker.queue_move_excitation(0.0)
        self.save_storage()

    def move_coll_vertical(self):
        self.worker.queue_move_collection(90.0)
        self.save_storage()

    def move_coll_horizontal(self):
        self.worker.queue_move_collection(0.0)
        self.save_storage()

    def move_rot3_vertical(self):
        self.worker.queue_move_rot3(90.0)
        self.save_storage()

    def move_rot3_horizontal(self):
        self.worker.queue_move_rot3(0.0)
        self.save_storage()

    def move_exc_manual(self):
        angle = normalize_angle(self.exc_manual_spin.value())
        self.exc_manual_spin.blockSignals(True)
        self.exc_manual_spin.setValue(angle)
        self.exc_manual_spin.blockSignals(False)
        self.worker.queue_move_excitation(angle)
        self.save_storage()

    def home_exc(self):
        self.exc_manual_spin.blockSignals(True)
        self.exc_manual_spin.setValue(0.0)
        self.exc_manual_spin.blockSignals(False)
        self.worker.queue_move_excitation(0.0)
        self.save_storage()

    def move_exc_relative(self):
        rel_angle = self.exc_relative_spin.value()
        target_angle = normalize_angle(self.current_exc_angle + rel_angle)
        self.exc_manual_spin.blockSignals(True)
        self.exc_manual_spin.setValue(target_angle)
        self.exc_manual_spin.blockSignals(False)
        self.worker.queue_move_excitation(target_angle)
        self.save_storage()

    def move_coll_manual(self):
        angle = normalize_angle(self.coll_manual_spin.value())
        self.coll_manual_spin.blockSignals(True)
        self.coll_manual_spin.setValue(angle)
        self.coll_manual_spin.blockSignals(False)
        self.worker.queue_move_collection(angle)
        self.save_storage()

    def home_coll(self):
        self.coll_manual_spin.blockSignals(True)
        self.coll_manual_spin.setValue(0.0)
        self.coll_manual_spin.blockSignals(False)
        self.worker.queue_move_collection(0.0)
        self.save_storage()

    def move_coll_relative(self):
        rel_angle = self.coll_relative_spin.value()
        target_angle = normalize_angle(self.current_coll_angle + rel_angle)
        self.coll_manual_spin.blockSignals(True)
        self.coll_manual_spin.setValue(target_angle)
        self.coll_manual_spin.blockSignals(False)
        self.worker.queue_move_collection(target_angle)
        self.save_storage()

    def move_rot3_manual(self):
        angle = normalize_angle(self.rot3_manual_spin.value())
        self.rot3_manual_spin.blockSignals(True)
        self.rot3_manual_spin.setValue(angle)
        self.rot3_manual_spin.blockSignals(False)
        self.worker.queue_move_rot3(angle)
        self.save_storage()

    def home_rot3(self):
        self.rot3_manual_spin.blockSignals(True)
        self.rot3_manual_spin.setValue(0.0)
        self.rot3_manual_spin.blockSignals(False)
        self.worker.queue_move_rot3(0.0)
        self.save_storage()

    def move_rot3_relative(self):
        rel_angle = self.rot3_relative_spin.value()
        target_angle = normalize_angle(self.current_rot3_angle + rel_angle)
        self.rot3_manual_spin.blockSignals(True)
        self.rot3_manual_spin.setValue(target_angle)
        self.rot3_manual_spin.blockSignals(False)
        self.worker.queue_move_rot3(target_angle)
        self.save_storage()

    # Visual dial drag manual tuning callbacks (dragging vs dropping)
    def on_exc_dial_dragging(self, angle):
        self.current_exc_angle = angle
        self.lbl_exc_angle.setText(f"{angle:.2f}°")
        self.exc_manual_spin.blockSignals(True)
        self.exc_manual_spin.setValue(angle)
        self.exc_manual_spin.blockSignals(False)
        self.update_dials()

    def on_exc_dial_tuned(self, angle):
        self.current_exc_angle = angle
        self.lbl_exc_angle.setText(f"{angle:.2f}°")
        self.exc_manual_spin.blockSignals(True)
        self.exc_manual_spin.setValue(angle)
        self.exc_manual_spin.blockSignals(False)
        self.update_dials()
        # Trigger the rotator move command only when mouse is released (dropped)
        self.worker.queue_move_excitation(angle)
        self.save_storage()

    def on_coll_dial_dragging(self, angle):
        self.current_coll_angle = angle
        self.lbl_coll_angle.setText(f"{angle:.2f}°")
        self.coll_manual_spin.blockSignals(True)
        self.coll_manual_spin.setValue(angle)
        self.coll_manual_spin.blockSignals(False)
        self.update_dials()

    def on_coll_dial_tuned(self, angle):
        self.current_coll_angle = angle
        self.lbl_coll_angle.setText(f"{angle:.2f}°")
        self.coll_manual_spin.blockSignals(True)
        self.coll_manual_spin.setValue(angle)
        self.coll_manual_spin.blockSignals(False)
        self.update_dials()
        # Trigger the rotator move command only when mouse is released (dropped)
        self.worker.queue_move_collection(angle)
        self.save_storage()

    def on_rot3_dial_dragging(self, angle):
        self.current_rot3_angle = angle
        self.lbl_rot3_angle.setText(f"{angle:.2f}°")
        self.rot3_manual_spin.blockSignals(True)
        self.rot3_manual_spin.setValue(angle)
        self.rot3_manual_spin.blockSignals(False)
        self.update_dials()

    def on_rot3_dial_tuned(self, angle):
        self.current_rot3_angle = angle
        self.lbl_rot3_angle.setText(f"{angle:.2f}°")
        self.rot3_manual_spin.blockSignals(True)
        self.rot3_manual_spin.setValue(angle)
        self.rot3_manual_spin.blockSignals(False)
        self.update_dials()
        # Trigger the rotator move command only when mouse is released (dropped)
        self.worker.queue_move_rot3(angle)
        self.save_storage()

    # Worker feedback slots
    def on_angles_updated(self, exc_angle, coll_angle, rot3_angle):
        self.current_exc_angle = exc_angle
        self.current_coll_angle = coll_angle
        self.current_rot3_angle = rot3_angle
        self.lbl_exc_angle.setText(f"{exc_angle:.2f}°")
        self.lbl_coll_angle.setText(f"{coll_angle:.2f}°")
        self.lbl_rot3_angle.setText(f"{rot3_angle:.2f}°")
        self.update_dials()
        # Only save if the angle has changed from what is currently saved in storage
        if (abs(exc_angle - getattr(self, "last_saved_exc_angle", -999.0)) > 0.005 or
            abs(coll_angle - getattr(self, "last_saved_coll_angle", -999.0)) > 0.005 or
            abs(rot3_angle - getattr(self, "last_saved_rot3_angle", -999.0)) > 0.005):
            self.save_storage()

    def on_status_updated(self, text):
        self.lbl_status.setText(f"Status: {text}")

    def on_move_finished(self, source):
        self.lbl_status.setText(f"Status: Move complete ({source})")

    def on_error_occurred(self, err):
        self.lbl_status.setText(f"Status: Error - {err}")
        QtWidgets.QMessageBox.warning(self, "Hardware Error", f"An error occurred during move:\n{err}")

    def apply_theme(self):
        from theme import SCIENTIFIC_THEME_QSS
        self.setStyleSheet(SCIENTIFIC_THEME_QSS)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.save_storage()
        event.accept()

if __name__ == '__main__':
    from station_setup import initialize_station
    station = initialize_station(simulated=False)
    app = QtWidgets.QApplication(sys.argv)
    win = PolarizationControlWindow(station)
    win.show()
    sys.exit(app.exec())
