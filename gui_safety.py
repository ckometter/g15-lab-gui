"""Prevent scrolling a panel from changing instrument parameters."""
import weakref

from PyQt6 import QtCore, QtGui, QtWidgets, sip


class ParameterWheelGuard(QtCore.QObject):
    def __init__(self, root):
        super().__init__(root)
        self._root = weakref.ref(root)
        self._installed = False

    def eventFilter(self, obj, event):
        if event.type() != QtCore.QEvent.Type.Wheel:
            return False
        if not isinstance(obj, QtWidgets.QWidget):
            return False
        root = self._root()
        if root is None or sip.isdeleted(root):
            return False
        if obj is not root and not root.isAncestorOf(obj):
            return False
        control = obj
        while control is not None and control is not root:
            if isinstance(control, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox,
                                    QtWidgets.QSlider, QtWidgets.QDial)):
                # Forward the gesture to the surrounding panel, even when the
                # parameter has keyboard focus. Explicit clicks/typing still work.
                panel = control.parentWidget()
                while panel is not None:
                    if isinstance(panel, QtWidgets.QAbstractScrollArea):
                        viewport = panel.viewport()
                        forwarded = QtGui.QWheelEvent(
                            QtCore.QPointF(viewport.mapFromGlobal(event.globalPosition().toPoint())),
                            event.globalPosition(), event.pixelDelta(), event.angleDelta(),
                            event.buttons(), event.modifiers(), event.phase(), event.inverted())
                        QtWidgets.QApplication.sendEvent(viewport, forwarded)
                        break
                    panel = panel.parentWidget()
                return True
            control = control.parentWidget()
        return False


def install_parameter_wheel_guard(widget):
    """Install once; dynamically created controls are covered too."""
    guard = getattr(widget, '_parameter_wheel_guard', None)
    if guard is None or sip.isdeleted(guard):
        guard = ParameterWheelGuard(widget)
        widget._parameter_wheel_guard = guard
    if not guard._installed:
        QtWidgets.QApplication.instance().installEventFilter(guard)
        guard._installed = True
    return guard


def uninstall_parameter_wheel_guard(widget):
    """Detach Python callbacks before accepted-window/native Qt shutdown."""
    guard = getattr(widget, '_parameter_wheel_guard', None)
    if guard is None or sip.isdeleted(guard) or not guard._installed:
        return
    app = QtWidgets.QApplication.instance()
    if app is not None and not sip.isdeleted(app):
        app.removeEventFilter(guard)
    guard._installed = False
