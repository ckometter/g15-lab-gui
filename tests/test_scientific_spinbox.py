"""Typed entry in ScientificDoubleSpinBox, which replaces QDoubleSpinBox in the dashboard."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from measurements.Dual_gate_sweep_gui import ScientificDoubleSpinBox

ACCEPTABLE = QtGui.QValidator.State.Acceptable
INTERMEDIATE = QtGui.QValidator.State.Intermediate


def type_text(widget, text):
    """Key presses for ``text``; QTest.keyClicks crashes on characters such as µ and Ω."""
    for char in text:
        if char.isascii():
            QtTest.QTest.keyClicks(widget, char)
            continue
        for kind in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease):
            event = QtGui.QKeyEvent(kind, QtCore.Qt.Key.Key_unknown, QtCore.Qt.KeyboardModifier.NoModifier, char)
            QtWidgets.QApplication.sendEvent(widget, event)


class ScientificDoubleSpinBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def box(self, suffix="", prefix="", minimum=-1.0, maximum=1e12, value=1e-7):
        box = ScientificDoubleSpinBox()
        box.setKeyboardTracking(False)
        box.setRange(minimum, maximum)
        box.setPrefix(prefix)
        box.setSuffix(suffix)
        box.setValue(value)
        self.addCleanup(box.deleteLater)
        return box

    def type_and_enter(self, box, text):
        box.selectAll()
        type_text(box, text)
        QtTest.QTest.keyClick(box, QtCore.Qt.Key.Key_Return)

    def test_typed_numbers_are_taken_with_or_without_a_unit_suffix(self):
        for prefix, suffix in (("", ""), ("", " A"), ("", " V/s"), ("X: ", " V")):
            for typed in ("1e-8", "0.00000001", "10e-9", "1.0E-08"):
                with self.subTest(prefix=prefix, suffix=suffix, typed=typed):
                    box = self.box(suffix, prefix)
                    self.type_and_enter(box, typed)
                    self.assertEqual(box.value(), 1e-8)
                    self.assertEqual(box.text(), f"{prefix}1.0e-8{suffix}")

    def test_every_way_of_editing_the_number_is_taken(self):
        # On the old box only replacing the whole field, unit included, was taken.
        def ctrl_a(box):
            QtTest.QTest.keyClick(box, QtCore.Qt.Key.Key_A, QtCore.Qt.KeyboardModifier.ControlModifier)
            type_text(box, "1e-8")

        def whole_field(box):
            box.lineEdit().selectAll()
            type_text(box, "1e-8")

        def one_digit(box):
            box.lineEdit().deselect()
            box.lineEdit().setCursorPosition(len("1.0e-7"))
            QtTest.QTest.keyClick(box, QtCore.Qt.Key.Key_Backspace)
            type_text(box, "8")

        for edit in (ctrl_a, whole_field, one_digit):
            with self.subTest(edit=edit.__name__):
                box = self.box(" A")
                self.assertEqual(box.text(), "1.0e-7 A")
                edit(box)
                QtTest.QTest.keyClick(box, QtCore.Qt.Key.Key_Return)
                self.assertEqual(box.value(), 1e-8)
                self.assertEqual(box.text(), "1.0e-8 A")

    def test_si_prefixes_scale_numbers_in_a_box_with_a_unit(self):
        cases = ((" A", "10 nA", 1e-8), (" A", "10nA", 1e-8), (" A", "10n", 1e-8), (" A", "1 nA", 1e-9),
                 (" A", "2.5 uA", 2.5e-6), (" A", "2.5 µA", 2.5e-6), (" A", "1 A", 1.0), (" A", "3 pA", 3e-12),
                 (" s", "50 ms", 0.05), (" Hz", "1.5k", 1500.0), (" Ω", "2 MΩ", 2e6),
                 (" A/V", "1 nA/V", 1e-9), (" pF", "5 pF", 5.0))
        for suffix, typed, expected in cases:
            with self.subTest(suffix=suffix, typed=typed):
                box = self.box(suffix)
                self.type_and_enter(box, typed)
                self.assertEqual(box.value(), expected)

    def test_unreadable_text_is_kept_as_typed_and_enter_keeps_the_previous_value(self):
        # Qt drops a keystroke its validator calls Invalid: "1 E-9" would become 1. It must stay unread.
        # Qt's fixup would also delete a decimal comma: 1,5 must not become 15.
        cases = ((" A", "1 E-9"), (" A", "10 x"), (" A", "ten"), (" A", "10 NA"), ("", "5m"),
                 (" pF", "5p"), (" nm", "500n"), (" V", "1,5"), ("", "1,5"))
        for suffix, typed in cases:
            with self.subTest(suffix=suffix, typed=typed):
                box = self.box(suffix)
                box.selectAll()
                type_text(box, typed)
                self.assertEqual(box.text(), typed + suffix)
                QtTest.QTest.keyClick(box, QtCore.Qt.Key.Key_Return)
                self.assertEqual(box.value(), 1e-7)
                self.assertEqual(box.text(), "1.0e-7" + suffix)

    def test_a_number_outside_the_range_is_not_taken(self):
        for typed in ("5", "-2", "5 A", "0.5 nA"):
            with self.subTest(typed=typed):
                box = self.box(" A", minimum=1e-9, maximum=3.0)
                self.type_and_enter(box, typed)
                self.assertEqual(box.value(), 1e-7)

    def test_validate_and_value_from_text_get_the_whole_text(self):
        box = self.box(" V", prefix="X: ")
        self.assertEqual(box.validate("X: 2.5 V", 4)[0], ACCEPTABLE)
        self.assertEqual(box.validate("X: 2.5 x V", 4)[0], INTERMEDIATE)
        self.assertEqual(box.valueFromText("X: 2.5 V"), 2.5)
        self.assertEqual(box.valueFromText("2.5"), 2.5)
        self.assertEqual(box.valueFromText("X: 25 mV V"), 0.025)

    def test_special_value_text_stands_for_the_minimum(self):
        box = self.box(" A")
        box.setSpecialValueText("Readback unavailable")
        box.setValue(-1.0)
        self.assertEqual(box.text(), "Readback unavailable")
        self.assertEqual(box.validate("Readback unavailable", 0)[0], ACCEPTABLE)
        self.assertEqual(box.valueFromText("Readback unavailable"), -1.0)


if __name__ == "__main__":
    unittest.main()
