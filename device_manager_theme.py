"""A dark palette scoped to the device manager, including its popups."""

COLORS = {
    "ready": "#62d6a3", "failed": "#ff818b", "busy": "#7cbaff",
    "simulated": "#c3a4ff", "limited": "#f1c66c", "idle": "#929caf",
}

DEVICE_MANAGER_QSS = """
QMainWindow, QWidget#managerBody { background: #090b0f; color: #e6eaf2; }
QWidget { color: #e6eaf2; font-family: 'Segoe UI'; font-size: 13px; }
QLabel { background: transparent; }
QLabel#title { font-size: 25px; font-weight: 600; }
QLabel#subtitle, QLabel#footnote { color: #a5afc0; }
QLabel#summary { background: #141923; border: 1px solid #293244;
    border-radius: 7px; padding: 10px 14px; }
QLabel#spectrometer { background: #111823; border: 1px solid #293e59;
    border-radius: 7px; padding: 12px; }
QTableWidget { background: #101318; alternate-background-color: #14181f;
    gridline-color: #242b36; border: 1px solid #29313e; border-radius: 6px;
    selection-background-color: #26364b; selection-color: #ffffff; }
QTableWidget::item { padding: 8px; }
QHeaderView::section { background: #1b212b; color: #aab7ca;
    padding: 10px; border: none; border-bottom: 1px solid #303949; font-weight: 600; }
QComboBox { background: #1a202b; border: 1px solid #364153;
    border-radius: 4px; padding: 6px 10px; min-width: 104px; }
QComboBox QAbstractItemView { background: #1a202b; color: #e6eaf2;
    selection-background-color: #2c4361; }
QComboBox:disabled { color: #9ca6b8; border-color: #283140; }
QPushButton { background: #222b39; border: 1px solid #3d4a60;
    border-radius: 5px; padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #2c3d54; border-color: #709bcc; }
QPushButton:pressed { background: #172337; }
QPushButton:disabled { background: #141921; color: #646f80; border-color: #272f3b; }
QPushButton#primary { background: #397fd0; border-color: #67a6ef; color: #ffffff; }
QPushButton#primary:hover { background: #498ddd; }
QPushButton#primary:disabled { background: #192638; color: #76869c; border-color: #2b3b50; }
QPlainTextEdit { background: #0e1117; color: #c9d4e5; border: 1px solid #29313e;
    border-radius: 5px; padding: 10px; font-family: Consolas; font-size: 12px; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: #131821; color: #95a4b9; padding: 9px 16px; }
QTabBar::tab:selected { background: #233248; color: #e6eaf2; }
QScrollBar:vertical { background: #101318; width: 12px; }
QScrollBar::handle:vertical { background: #39465a; min-height: 30px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QToolTip { background: #222c3a; color: #e6eaf2; border: 1px solid #50617b; }
QMessageBox { background: #141923; }
"""
