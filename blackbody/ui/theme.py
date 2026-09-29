"""Dark, neutral application theme. Greys stay neutral so footage and fire colours read true;
the ember accent marks the active state and keyframes are the familiar amber diamond."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = '#1b1b1d'        # window
PANEL = '#232326'     # panels, docks
FIELD = '#2c2c30'     # inputs
FIELD_HI = '#35353a'
LINE = '#3a3a40'
TEXT = '#d9d9dc'
MUTED = '#8d8d94'
ACCENT = '#ff7a2f'    # ember
ACCENT_DIM = '#8a4a26'
KEY = '#f2c14e'       # keyframes
GOOD = '#5fbf7a'
BAD = '#e5534b'
VIEWER = '#121213'    # viewport surround, neutral for colour judgement

STYLE = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {PANEL}; padding: 5px 8px; border-bottom: 1px solid {LINE}; font-weight: 600; }}
QToolTip {{ background: #2e2e33; color: {TEXT}; border: 1px solid {LINE}; padding: 5px; }}
QMenuBar {{ background: {BG}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 8px; }}
QTabWidget::pane {{ border: 0; border-top: 1px solid {LINE}; }}
QTabBar::tab {{ background: transparent; padding: 6px 12px; color: {MUTED}; border: 0; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QScrollArea {{ border: 0; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #45454c; min-height: 24px; border-radius: 4px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #45454c; min-width: 24px; border-radius: 4px; margin: 2px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {FIELD}; border: 1px solid {LINE}; border-radius: 3px; padding: 2px 5px; min-height: 18px;
    selection-background-color: {ACCENT_DIM}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: 0; }}
QComboBox::drop-down {{ border: 0; width: 16px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {LINE}; selection-background-color: {ACCENT_DIM}; }}
QPushButton {{ background: {FIELD}; border: 1px solid {LINE}; border-radius: 3px; padding: 4px 12px; }}
QPushButton:hover {{ background: {FIELD_HI}; }}
QPushButton:pressed {{ background: #404047; }}
QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #5d5d63; }}
QPushButton#primary {{ background: {ACCENT}; color: #1b1107; border: 0; font-weight: 600; }}
QPushButton#primary:hover {{ background: #ff8d4d; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 3px; padding: 3px 6px; }}
QToolButton:hover {{ background: {FIELD_HI}; }}
QToolButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QToolBar {{ background: {PANEL}; border: 0; spacing: 2px; padding: 2px; }}
QSlider::groove:horizontal {{ height: 4px; background: {FIELD}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 10px; margin: -4px 0; border-radius: 5px; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {LINE}; border-radius: 3px; background: {FIELD}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QGroupBox {{ border: 0; margin-top: 18px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 2px; top: 2px; color: {MUTED}; }}
QTreeWidget, QListWidget {{ background: {PANEL}; border: 0; outline: 0; }}
QTreeWidget::item, QListWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: {ACCENT_DIM}; color: {TEXT}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {LINE}; }}
QProgressBar {{ background: {FIELD}; border: 1px solid {LINE}; border-radius: 3px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QLabel#section {{ color: {MUTED}; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
QLabel#hint {{ color: {MUTED}; }}
QFrame#sep {{ background: {LINE}; max-height: 1px; min-height: 1px; }}
"""


def apply(app: QApplication):
    app.setStyle('Fusion')
    pal = QPalette()
    for role, col in ((QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, FIELD),
                      (QPalette.AlternateBase, PANEL), (QPalette.Text, TEXT), (QPalette.Button, FIELD),
                      (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT_DIM), (QPalette.HighlightedText, TEXT),
                      (QPalette.ToolTipBase, PANEL), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, MUTED)):
        pal.setColor(role, QColor(col))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor('#5d5d63'))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#5d5d63'))
    app.setPalette(pal)
    f = app.font()
    f.setPointSizeF(9.0)
    app.setFont(f)
    app.setStyleSheet(STYLE)


def mono_font(size=9.0):
    f = QFont('Consolas')
    f.setStyleHint(QFont.Monospace)
    f.setPointSizeF(size)
    return f
