"""Application entry point."""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

import blackbody


def run(argv=None):
    argv = list(argv or [])
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    QCoreApplication.setOrganizationName('Blackbody')
    QCoreApplication.setApplicationName(blackbody.APP_NAME)
    QCoreApplication.setApplicationVersion(blackbody.__version__)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication([sys.argv[0]] + argv)
    from . import icons, theme
    theme.apply(app)
    app.setWindowIcon(icons.app_icon())
    from .main_window import MainWindow
    from .worker import EngineWorker
    worker = EngineWorker()
    worker.start()
    path = next((a for a in argv if a.lower().endswith('.bbfire')), None)
    win = MainWindow(worker, open_path=path)
    win.show()
    code = app.exec()
    if worker.isRunning():
        worker.stop()
    return code
