"""The window tested without a GPU: MainWindow offscreen with a stand-in for the engine's worker (uikit.FakeWorker: every
scene and frame sent to it comes back as the preset's thumbnail, with made-up stats), its settings and its data (your
blocks, presets, surfaces, the autosave) in a folder of the run's own, and every message box answered without waiting. A
test fails if any of the window's slots raised meanwhile (Qt only prints those)."""
import sys
import traceback

import pytest

import uikit
from uikit import pump

if uikit.QApplication is None:      # (no Qt here: these tests are left out)
    collect_ignore_glob = ['test_*.py']
else:
    from PySide6.QtCore import QCoreApplication, QSettings, QStandardPaths
    from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope='session')
def qapp(tmp_path_factory):
    """The application, its settings (an ini file) and its data folder in the run's own folder."""
    home = tmp_path_factory.mktemp('ui-home')
    QCoreApplication.setOrganizationName('BlackbodyUITest')
    QCoreApplication.setApplicationName('Blackbody')
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(home / 'settings'))
    data = home / 'data'
    real = QStandardPaths.writableLocation

    def where(kind):
        if kind == QStandardPaths.AppDataLocation:
            return str(data)
        return real(kind)

    QStandardPaths.writableLocation = staticmethod(where)
    app = QApplication.instance() or QApplication([])
    from blackbody.ui import theme
    theme.apply(app)
    yield app
    QStandardPaths.writableLocation = staticmethod(real)


@pytest.fixture
def slot_errors(monkeypatch):
    """What the window's slots raised (Qt prints those and carries on): checked empty after each test that uses it."""
    errors = []
    monkeypatch.setattr(sys, 'excepthook', lambda t, v, tb: errors.append(''.join(traceback.format_exception(t, v, tb))))
    yield errors
    assert not errors, '\n'.join(errors)


@pytest.fixture
def win(qapp, slot_errors, monkeypatch):
    """The main window as at a first start (closed cleanly last time: no Recover question), its stand-in worker in
    `win.worker`, playback stopped."""
    from blackbody.scene import blocks
    from blackbody.ui.main_window import MainWindow
    monkeypatch.setattr(blocks, 'DIR', None)
    monkeypatch.setattr(QMessageBox, 'information', staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(lambda *a, **k: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(lambda *a, **k: QMessageBox.Yes))
    s = QSettings()
    s.clear()
    s.setValue('ui/welcome_done', True)
    s.setValue('ui/clean_exit', True)
    s.sync()
    w = MainWindow(uikit.FakeWorker())
    w._autosave.stop()
    w.resize(1600, 950)
    w.show()
    pump(0.4)
    w.doc.set_playing(False)
    pump()
    yield w
    w.doc.set_playing(False)
    w.doc.undo.setClean()
    w.close()
    w.deleteLater()
    pump()
