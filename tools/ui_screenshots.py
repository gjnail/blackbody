"""Drive the app offscreen like a user would and save screenshots.

python tools/ui_screenshots.py [footage] [out_dir]
Imports footage, loads a fire, places it, tracks it, saves/reopens the project, renders from the
app's job path, and grabs the window along the way.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if sys.platform == 'win32':
    os.environ.setdefault('QT_QPA_FONTDIR', 'C:/Windows/Fonts')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication, QSettings
from PySide6.QtWidgets import QApplication


def pump(app, seconds, until=None):
    t = time.time()
    while time.time() - t < seconds:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return False


def main():
    import builtins
    import faulthandler
    footage = sys.argv[1] if len(sys.argv) > 1 else None
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / 'out' / 'ui')
    out.mkdir(parents=True, exist_ok=True)
    stacks = open(out / 'stacks.txt', 'w')
    faulthandler.dump_traceback_later(240, repeat=True, file=stacks)
    t0 = time.time()
    _print = builtins.print
    builtins.print = lambda *a, **k: _print(f'[{time.time() - t0:6.1f}s]', *a, flush=True)
    QCoreApplication.setOrganizationName('BlackbodyTest')
    QCoreApplication.setApplicationName('Blackbody')
    app = QApplication([sys.argv[0]])
    from blackbody.ui import theme
    theme.apply(app)
    from blackbody.ui.main_window import MainWindow
    from blackbody.ui.worker import EngineWorker
    s = QSettings()
    s.setValue('ui/welcome_done', True)
    s.remove('ui/geometry')
    s.remove('ui/state')
    w = EngineWorker()
    w.start()
    win = MainWindow(w)
    win.resize(1680, 980)
    win.show()
    win.doc.set_playing(False)   # the app plays on start; these screenshots want still, refined frames
    got = []
    w.frameReady.connect(lambda img, f, st: got.append((f, st.get('refined'))))
    pump(app, 30, until=lambda: any(r for _, r in got))
    print('first refined frame after', len(got), 'frames')
    if footage:
        win.doc.import_footage(footage)
        pump(app, 20, until=lambda: win.doc.footage_info is not None)
        print('footage:', (win.doc.footage_info or {}).get('describe'))
        win.doc.set(('camera', 'anchor_x'), 0.38, merge=False)
        win.doc.set(('camera', 'anchor_y'), 0.8, merge=False)
        win.doc.set(('camera', 'scale'), 1.3, merge=False)
        win.doc.set_frame(win.doc.scene.start + 20)
        got.clear()
        pump(app, 40, until=lambda: any(r for _, r in got))
    win.doc.select(('section', 'composite'))
    pump(app, 1)
    win.grab().save(str(out / 'app_composite.png'))
    # tracking
    if footage:
        from blackbody.ui.tracking import track_fire_base
        track_fire_base(win)
        pump(app, 60, until=lambda: bool(win.doc.scene.track))
        pts = (win.doc.scene.track or {}).get('points', {})
        print('tracked frames:', len(pts))
    # save and reopen
    proj = out / 'shot.bbfire'
    win.doc.save(str(proj))
    win.open_path(str(proj))
    pump(app, 3)
    print('reopened:', win.doc.scene.name, 'footage' if win.doc.scene.footage else 'no footage',
          'track' if win.doc.scene.track else 'no track')
    # render through the app's job path (small and quick)
    from blackbody.render.job import Output
    spec = {'scene': win.doc.scene.copy(), 'frames': (win.doc.scene.start, win.doc.scene.start + 5),
            'outputs': [Output('exr', str(out / 'job' / 'fire.####.exr'), 'element'),
                        Output('video', str(out / 'job' / 'comp.mov'), 'composite', 'prores422hq')],
            'samples': 1, 'motion_blur': True}
    spec['scene'].data['render']['final_scale'] = 0.5
    done = []
    w.jobDone.connect(lambda written, cancelled, err: done.append((written, cancelled, err)))
    w.post('job', spec)
    pump(app, 300, until=lambda: bool(done))
    if done:
        written, cancelled, err = done[0]
        print('job:', len(written), 'files', 'cancelled' if cancelled else '', err.splitlines()[0] if err else '')
    win.doc.select(('emitter', 0))
    got.clear()
    pump(app, 30, until=lambda: any(r for _, r in got))
    win.grab().save(str(out / 'app_emitter.png'))
    w.stop()
    print('screens in', out)


if __name__ == '__main__':
    main()
