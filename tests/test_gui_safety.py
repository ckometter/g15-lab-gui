"""Scroll gestures must never modify a parameter, even when focused."""
import os
import unittest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6 import QtCore, QtGui, QtWidgets
from gui_safety import install_parameter_wheel_guard, uninstall_parameter_wheel_guard


class ParameterWheelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wheel(self, widget):
        event = QtGui.QWheelEvent(QtCore.QPointF(4, 4),
            QtCore.QPointF(widget.mapToGlobal(QtCore.QPoint(4, 4))),
            QtCore.QPoint(), QtCore.QPoint(0, -120),
            QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase, False)
        self.app.sendEvent(widget, event)

    def test_controls_and_their_editors_ignore_wheel_but_panel_scrolls(self):
        root = QtWidgets.QWidget()
        self.addCleanup(root.close)
        layout = QtWidgets.QVBoxLayout(root)
        scroll = QtWidgets.QScrollArea()
        layout.addWidget(scroll)
        panel = QtWidgets.QWidget()
        panel.setMinimumSize(300, 1400)
        body = QtWidgets.QVBoxLayout(panel)
        spin = QtWidgets.QDoubleSpinBox()
        spin.setValue(7.)
        combo = QtWidgets.QComboBox()
        combo.addItems(['VOLT', 'CURR'])
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setValue(42)
        for control in (spin, combo, slider): body.addWidget(control)
        scroll.setWidget(panel)
        install_parameter_wheel_guard(root)
        root.resize(400, 300)
        root.show()
        self.app.processEvents()
        for widget in (spin, spin.lineEdit(), combo, slider):
            widget.setFocus()
            self.wheel(widget)
        self.assertEqual(spin.value(), 7.)
        self.assertEqual(combo.currentText(), 'VOLT')
        self.assertEqual(slider.value(), 42)
        self.assertGreater(scroll.verticalScrollBar().value(), 0)
        # Explicit edits remain possible.
        spin.setValue(8.)
        combo.setCurrentIndex(1)
        self.assertEqual(spin.value(), 8.)
        self.assertEqual(combo.currentText(), 'CURR')

    def test_controls_created_after_installation_are_protected(self):
        root = QtWidgets.QWidget()
        self.addCleanup(root.close)
        install_parameter_wheel_guard(root)
        spin = QtWidgets.QSpinBox(root)
        spin.setValue(12)
        self.wheel(spin)
        self.assertEqual(spin.value(), 12)

    def test_uninstall_and_reinstall_are_idempotent_and_protect_new_controls(self):
        root = QtWidgets.QWidget()
        self.addCleanup(root.close)
        self.addCleanup(uninstall_parameter_wheel_guard, root)
        guard = install_parameter_wheel_guard(root)
        self.assertIs(install_parameter_wheel_guard(root), guard)
        uninstall_parameter_wheel_guard(root)
        uninstall_parameter_wheel_guard(root)
        self.assertFalse(guard._installed)
        self.assertIs(install_parameter_wheel_guard(root), guard)
        self.assertTrue(guard._installed)
        spin = QtWidgets.QSpinBox(root)
        spin.setValue(9)
        self.wheel(spin)
        self.assertEqual(spin.value(), 9)


if __name__ == '__main__':
    unittest.main()
