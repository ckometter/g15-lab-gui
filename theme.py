# -*- coding: utf-8 -*-
"""
Unified scientific instrumentation theme for the QCoDeS Lab Dashboard.
Implements the user's design specifications for a premium light theme.
"""

SCIENTIFIC_THEME_QSS = """
/* Global Widget Styles */
QWidget {
    background-color: #FFFFFF;
    color: #0F172A;
    font-family: 'Roboto', 'Montserrat', 'Segoe UI', sans-serif;
    font-size: 12px;
}

/* ScrollBars */
QScrollBar:vertical {
    border: none;
    background: #FFFFFF;
    width: 12px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #CBD5E1;
    min-height: 20px;
    border-radius: 0px;
    border: 1px solid #E2E8F0;
}
QScrollBar::handle:vertical:hover {
    background: #94A3B8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    border: none;
    background: #FFFFFF;
    height: 12px;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background: #CBD5E1;
    min-width: 20px;
    border-radius: 0px;
    border: 1px solid #E2E8F0;
}
QScrollBar::handle:horizontal:hover {
    background: #94A3B8;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #CBD5E1;
    border-radius: 0px;
    background-color: #F8FAFC;
    padding: 2px;
}
QTabBar::tab {
    background-color: #E2E8F0;
    color: #64748B;
    border: 1px solid #CBD5E1;
    border-radius: 0px;
    padding: 6px 12px;
    margin-right: 2px;
    font-weight: normal;
}
QTabBar::tab:selected {
    background-color: #F8FAFC;
    color: #0F172A;
    border-bottom-color: transparent;
    font-weight: bold;
}
QTabBar::tab:hover {
    background-color: #F1F5F9;
    color: #0F172A;
}
QTabBar::tab:disabled {
    color: #CBD5E1;
    background-color: #F8FAFC;
    border-color: #E2E8F0;
}

/* GroupBox Grouping */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 0px;
    margin-top: 12px;
    color: #0F172A;
    font-weight: bold;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: border;
    subcontrol-position: top center;
    padding: 0 8px;
    background-color: #FFFFFF;
    top: -10px;
}

/* Labels */
QLabel {
    color: #0F172A;
    background-color: transparent;
}

/* Input Fields (SpinBoxes, Combos, LineEdits) */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 0px;
    color: #0F172A;
    padding: 3px 5px;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 12px;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
    background-color: #F8FAFC;
    color: #94A3B8;
    border-color: #E2E8F0;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    background-color: #FFFFFF;
    border: 1px solid #3B82F6;
}

/* ComboBox Dropdown Styling */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 18px;
    border-left-width: 1px;
    border-left-color: #CBD5E1;
    border-left-style: solid;
    background-color: #F1F5F9;
}
QComboBox::drop-down:hover {
    background-color: #E2E8F0;
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #64748B;
}

/* Hide SpinBox Buttons (Up/Down Arrows) */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 0px;
    height: 0px;
    border: none;
    background: transparent;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0px;
    height: 0px;
    border: none;
    background: transparent;
}

/* Action Buttons */
QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:1 #F1F5F9);
    color: #0F172A;
    border: 1px solid #CBD5E1;
    border-radius: 0px;
    padding: 6px 12px;
    font-weight: bold;
    font-size: 12px;
}
QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F1F5F9, stop:1 #E2E8F0);
    border-color: #94A3B8;
}
QPushButton:pressed {
    background-color: #E2E8F0;
    border-color: #3B82F6;
}
QPushButton:disabled {
    background-color: #F8FAFC;
    color: #94A3B8;
    border-color: #E2E8F0;
}

/* Special Button Overrides */
/* Begin Button */
QPushButton#begin_btn, QPushButton#btn_begin, QPushButton#btn_start {
    background-color: #10B981; /* Solid green */
    color: #FFFFFF;
    border: 1px solid #059669;
}
QPushButton#begin_btn:hover, QPushButton#btn_begin:hover, QPushButton#btn_start:hover {
    background-color: #059669;
}
QPushButton#begin_btn:pressed, QPushButton#btn_begin:pressed, QPushButton#btn_start:pressed {
    background-color: #047857;
}

/* Interrupt Button */
QPushButton#pause_btn, QPushButton#btn_pause, QPushButton#btn_stop {
    background-color: #EF4444; /* Solid red */
    color: #FFFFFF;
    border: 1px solid #DC2626;
}
QPushButton#pause_btn:hover, QPushButton#btn_pause:hover, QPushButton#btn_stop:hover {
    background-color: #DC2626;
}
QPushButton#pause_btn:pressed, QPushButton#btn_pause:pressed, QPushButton#btn_stop:pressed {
    background-color: #B91C1C;
}

/* Reset Button */
QPushButton#reset_btn, QPushButton#btn_reset {
    background-color: #3B82F6; /* Solid blue */
    color: #FFFFFF;
    border: 1px solid #2563EB;
}
QPushButton#reset_btn:hover, QPushButton#btn_reset:hover {
    background-color: #2563EB;
}
QPushButton#reset_btn:pressed, QPushButton#btn_reset:pressed {
    background-color: #1D4ED8;
}

/* Progress Bar */
QProgressBar {
    background-color: #E2E8F0;
    border: 1px solid #CBD5E1;
    border-radius: 0px;
    text-align: center;
    color: #0F172A;
    font-weight: bold;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #10B981; /* Bright green */
    border-radius: 0px;
}

/* CheckBox & RadioButton */
QCheckBox, QRadioButton {
    background-color: transparent;
    color: #0F172A;
}
QCheckBox::indicator, QRadioButton::indicator {
    border: 1px solid #CBD5E1;
    border-radius: 0px;
    background-color: #FFFFFF;
    width: 14px;
    height: 14px;
}
QCheckBox::indicator:checked {
    background-color: #10B981;
    border-color: #059669;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QRadioButton::indicator:checked {
    background-color: #10B981;
    border-color: #059669;
}

/* StatusBar */
QStatusBar {
    background-color: #F1F5F9;
    border-top: 1px solid #CBD5E1;
    color: #64748B;
}
"""
