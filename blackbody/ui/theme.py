"""Dark, neutral application theme. Greys stay neutral so footage and fire colours read true;
the ember accent marks the active state and keyframes are the familiar amber diamond."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QApplication

BG = '#121214'        # window chrome
PANEL = '#19191c'     # panels, docks
CARD = '#202024'      # groups of settings, cards
FIELD = '#29292e'     # inputs
FIELD_HI = '#323238'
LINE = '#2b2b31'
LINE_HI = '#3d3d45'
TEXT = '#e8e8eb'
MUTED = '#9b9ba4'
FAINT = '#686870'
ACCENT = '#ff7a2f'    # ember
ACCENT_HI = '#ff9452'
ACCENT_DIM = '#6b3415'
ACCENT_FILL = '#3a2416'   # number-field fill
CHANGED = '#ffb27a'   # a setting changed from the preset
KEY = '#f2c14e'       # keyframes
GOOD = '#5fbf7a'
BAD = '#e5534b'
VIEWER = '#0c0c0d'    # viewport surround, neutral for colour judgement

# what each kind of object in the scene is drawn with, in the viewer and the lists
OBJECT_COLOURS = {'emitter': '#ff9a4a', 'collider': '#78beff', 'light': '#ffdc78', 'fabric': '#96d2ff', 'matter': '#e3c08a',
                  'strands': '#9fd36a', 'shot': '#e8b25a'}


def _asset_dir():
    d = Path(tempfile.gettempdir()) / 'blackbody-ui-2'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _draw_asset(path, w, h, draw):
    img = QImage(w * 2, h * 2, QImage.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(2.0)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    draw(p)
    p.end()
    img.save(str(path))


def _assets():
    """Small images the style sheet needs (arrows, the tick), drawn once into a temp folder."""
    d = _asset_dir()

    def chevron(col, down=True):
        def f(p):
            p.setPen(QPen(QColor(col), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            path = QPainterPath()
            if down:
                path.moveTo(2.5, 4.0)
                path.lineTo(5.0, 6.5)
                path.lineTo(7.5, 4.0)
            else:
                path.moveTo(4.0, 2.5)
                path.lineTo(6.5, 5.0)
                path.lineTo(4.0, 7.5)
            p.drawPath(path)
        return f

    def tick(p):
        p.setPen(QPen(QColor('#1b1107'), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        path = QPainterPath()
        path.moveTo(3.2, 7.2)
        path.lineTo(5.8, 9.8)
        path.lineTo(10.8, 4.2)
        p.drawPath(path)

    out = {}
    for name, w, h, draw in (('down', 10, 10, chevron(MUTED)), ('down_hi', 10, 10, chevron(TEXT)),
                             ('right', 10, 10, chevron(MUTED, False)), ('tick', 14, 14, tick)):
        f = d / f'{name}.png'
        _draw_asset(f, w, h, draw)
        out[name] = f.as_posix()
    return out


def stylesheet(a):
    return f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QMainWindow::separator {{ background: {BG}; width: 4px; height: 4px; }}
QMainWindow::separator:hover {{ background: {ACCENT_DIM}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget > QWidget {{ background: {PANEL}; }}
QToolTip {{ background: #26262b; color: {TEXT}; border: 1px solid {LINE_HI}; padding: 6px 8px; border-radius: 4px; }}
QMenuBar {{ background: {BG}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 9px; border-radius: 4px; background: transparent; }}
QMenuBar::item:selected {{ background: {FIELD_HI}; }}
QMenu {{ background: #1f1f23; border: 1px solid {LINE_HI}; padding: 5px; border-radius: 6px; }}
QMenu::item {{ padding: 6px 26px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu::item:disabled {{ color: {FAINT}; }}
QMenu::separator {{ height: 1px; background: {LINE_HI}; margin: 5px 8px; }}
QMenu::right-arrow {{ image: url({a['right']}); width: 10px; height: 10px; }}
QTabWidget::pane {{ border: 0; border-top: 1px solid {LINE}; }}
QTabBar::tab {{ background: transparent; padding: 7px 14px; color: {MUTED}; border: 0; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QScrollArea {{ border: 0; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3a3a42; min-height: 28px; border-radius: 3px; margin: 2px 3px; }}
QScrollBar::handle:vertical:hover {{ background: #4a4a54; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #3a3a42; min-width: 28px; border-radius: 3px; margin: 3px 2px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {FIELD}; border: 1px solid {FIELD}; border-radius: 5px; padding: 3px 7px; min-height: 18px;
    selection-background-color: {ACCENT_DIM}; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ background: {FIELD_HI}; border-color: {FIELD_HI}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit#search {{ background: {FIELD}; border-radius: 15px; padding: 5px 12px; min-height: 18px; }}
QLineEdit#search:focus {{ border-color: {ACCENT}; background: {FIELD}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: 0; }}
QComboBox {{ padding-right: 20px; }}
QComboBox::drop-down {{ border: 0; width: 20px; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url({a['down']}); width: 10px; height: 10px; }}
QComboBox::down-arrow:hover {{ image: url({a['down_hi']}); }}
QComboBox QAbstractItemView {{ background: #1f1f23; border: 1px solid {LINE_HI}; padding: 4px; outline: 0;
    selection-background-color: {ACCENT_DIM}; border-radius: 4px; }}
QPushButton {{ background: {FIELD}; border: 1px solid {FIELD}; border-radius: 5px; padding: 5px 14px; }}
QPushButton:hover {{ background: {FIELD_HI}; border-color: {LINE_HI}; }}
QPushButton:pressed {{ background: #3e3e46; }}
QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {FAINT}; background: {PANEL}; }}
QPushButton::menu-indicator {{ image: url({a['down']}); subcontrol-position: right center; subcontrol-origin: padding; right: 6px; width: 10px; }}
QPushButton#primary {{ background: {ACCENT}; color: #1b1107; border: 0; font-weight: 600; padding: 6px 18px; }}
QPushButton#primary:hover {{ background: {ACCENT_HI}; }}
QPushButton#primary:pressed {{ background: #e86a22; }}
QPushButton#ghost {{ background: transparent; border: 1px solid transparent; }}
QPushButton#ghost:hover {{ background: {FIELD}; border-color: {FIELD}; }}
QPushButton#chip {{ background: transparent; border: 1px solid {LINE_HI}; border-radius: 11px; padding: 2px 9px; color: {MUTED}; }}
QPushButton#chip:hover {{ color: {TEXT}; border-color: {FAINT}; }}
QPushButton#chip:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; color: {TEXT}; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 4px 7px; }}
QToolButton:hover {{ background: {FIELD_HI}; }}
QToolButton:pressed {{ background: #3e3e46; }}
QToolButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#seg {{ border-radius: 0; padding: 4px 11px; color: {MUTED}; background: {FIELD}; border: 0; }}
QToolButton#seg:hover {{ color: {TEXT}; background: {FIELD_HI}; }}
QToolButton#seg:checked {{ color: #1b1107; background: {ACCENT}; font-weight: 600; }}
QToolButton#segFirst {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; border-radius: 0;
    padding: 4px 11px; color: {MUTED}; background: {FIELD}; border: 0; }}
QToolButton#segFirst:hover {{ color: {TEXT}; background: {FIELD_HI}; }}
QToolButton#segFirst:checked {{ color: #1b1107; background: {ACCENT}; font-weight: 600; }}
QToolButton#segLast {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; border-radius: 0;
    padding: 4px 11px; color: {MUTED}; background: {FIELD}; border: 0; }}
QToolButton#segLast:hover {{ color: {TEXT}; background: {FIELD_HI}; }}
QToolButton#segLast:checked {{ color: #1b1107; background: {ACCENT}; font-weight: 600; }}
QToolButton#toggle {{ color: {MUTED}; padding: 4px 9px; }}
QToolButton#toggle:checked {{ color: {TEXT}; background: {FIELD_HI}; border-color: {LINE_HI}; }}
QToolBar {{ background: {BG}; border: 0; spacing: 4px; padding: 0; }}
QSlider::groove:horizontal {{ height: 4px; background: {FIELD}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 12px; margin: -4px 0; border-radius: 6px; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {LINE_HI}; border-radius: 4px; background: {FIELD}; }}
QCheckBox::indicator:hover {{ border-color: {FAINT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({a['tick']}); }}
QCheckBox:disabled {{ color: {FAINT}; }}
QCheckBox::indicator:disabled {{ background: {PANEL}; border-color: {LINE}; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 8px; margin-top: 22px; padding: 10px 8px 8px 8px; background: {CARD}; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 4px; top: 2px; color: {MUTED}; }}
QTreeWidget, QListWidget, QListView {{ background: transparent; border: 0; outline: 0; }}
QTreeWidget::item, QListWidget::item {{ padding: 4px 3px; border-radius: 4px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {FIELD}; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: {ACCENT_DIM}; color: {TEXT}; }}
QTreeWidget::branch {{ background: transparent; }}
QStatusBar {{ background: {BG}; color: {MUTED}; border-top: 1px solid {LINE}; }}
QStatusBar::item {{ border: 0; }}
QProgressBar {{ background: {FIELD}; border: 0; border-radius: 4px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QLabel#section {{ color: {MUTED}; font-weight: 600; letter-spacing: 1px; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#faint {{ color: {FAINT}; }}
QLabel#title {{ font-size: 13pt; font-weight: 600; }}
QLabel#pill {{ background: {FIELD}; color: {MUTED}; border-radius: 10px; padding: 3px 10px; }}
QFrame#sep {{ background: {LINE}; max-height: 1px; min-height: 1px; }}
QFrame#vsep {{ background: {LINE_HI}; max-width: 1px; min-width: 1px; margin: 6px 4px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px; }}
"""


STYLE = ''


def apply(app: QApplication):
    global STYLE
    app.setStyle('Fusion')
    pal = QPalette()
    for role, col in ((QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, FIELD),
                      (QPalette.AlternateBase, PANEL), (QPalette.Text, TEXT), (QPalette.Button, FIELD),
                      (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT_DIM), (QPalette.HighlightedText, TEXT),
                      (QPalette.ToolTipBase, PANEL), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, FAINT),
                      (QPalette.Link, ACCENT_HI)):
        pal.setColor(role, QColor(col))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(FAINT))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(FAINT))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(FAINT))
    app.setPalette(pal)
    f = app.font()
    f.setPointSizeF(9.5)
    app.setFont(f)
    STYLE = stylesheet(_assets())
    app.setStyleSheet(STYLE)


def mono_font(size=9.0):
    f = QFont('Consolas')
    f.setStyleHint(QFont.Monospace)
    f.setPointSizeF(size)
    return f


def font(size=9.5, weight=QFont.Normal):
    f = QFont(QApplication.font())
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


def rounded(rect: QRectF, r):
    path = QPainterPath()
    path.addRoundedRect(rect, r, r)
    return path


def dot(p: QPainter, c: QPointF, r, col):
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(col))
    p.drawEllipse(c, r, r)
