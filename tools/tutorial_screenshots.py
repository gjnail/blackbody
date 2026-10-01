"""Drive the app offscreen through docs/tutorial.md and save its screenshots.

python tools/tutorial_screenshots.py [out_dir]
Follows the tutorial's steps on the practice plate (made with make_test_footage.py if missing),
saves the project as out/campfire_shot.bbfire, and writes the images to docs/tutorial.
Uses its own settings, so the app's real settings are left alone.

00_result.png is frame 65 of the tutorial's own render of that project:
python -m blackbody render out/campfire_shot.bbfire -o out/renders/comp.mp4
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if sys.platform == 'win32':
    os.environ.setdefault('QT_QPA_FONTDIR', 'C:/Windows/Fonts')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QCoreApplication, QPoint, QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication

PLATE = ROOT / 'out' / 'practice_plate.mp4'
PROJECT = ROOT / 'out' / 'campfire_shot.bbfire'
# The tutorial's values. The practice plate's ground has enough texture to track only on the left.
BASE = (0.30, 0.90)
SCALE = 0.65
SHOT_FRAME = 30
COMPOSITE = {'haze': 1.2, 'light_cast': 0.9, 'grain': 0.25}
WIND = 0.6
FLARE = ((40, 1.0), (55, 2.5))
LIQUID, LIQUID_FRAME = 'fountain', 60


def pump(app, seconds, until=None):
    t = time.time()
    while time.time() - t < seconds:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return False


def qimage_to_pil(img):
    img = img.convertToFormat(img.Format.Format_RGBA8888)
    return Image.frombuffer('RGBA', (img.width(), img.height()), bytes(img.constBits()), 'raw', 'RGBA', img.bytesPerLine(), 1).convert('RGB')


def font(size):
    for name in ('segoeuib.ttf', 'arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def badge(draw, xy, n, r=14):
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 138, 40), outline=(20, 20, 22), width=2)
    draw.text((x, y), str(n), fill=(20, 20, 22), font=font(17), anchor='mm')


def rect_in(win, widget):
    p = widget.mapTo(win, QPoint(0, 0))
    return p.x(), p.y(), widget.width(), widget.height()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'docs' / 'tutorial')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(*a):
        print(f'[{time.time() - t0:6.1f}s]', *a, flush=True)

    if not PLATE.exists():
        import make_test_footage
        make_test_footage.main(str(PLATE))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    settings_dir = tempfile.mkdtemp(prefix='bb_tutorial_')
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, settings_dir)
    QCoreApplication.setOrganizationName('BlackbodyTutorial')
    QCoreApplication.setApplicationName('Blackbody')
    app = QApplication([sys.argv[0]])
    from blackbody.ui import theme
    theme.apply(app)
    from blackbody.ui.export_dialog import ExportDialog
    from blackbody.ui.main_window import MainWindow
    from blackbody.ui.tracking import track_fire_base
    from blackbody.ui.worker import EngineWorker
    QSettings().setValue('ui/welcome_done', False)
    QSettings().setValue('ui/quality_index', 0)
    w = EngineWorker()
    w.start()
    win = MainWindow(w)
    win._autosave.stop()
    win.resize(1600, 960)
    win.show()
    doc = win.doc
    win.doc.set_playing(False)   # the app plays on start; these screenshots want still, refined frames
    got = []
    w.frameReady.connect(lambda img, f, st: got.append((f, bool(st.get('refined')))))

    def settle(frame=None, seconds=120):
        got.clear()
        ok = pump(app, seconds, until=lambda: any(r and (frame is None or f == frame) for f, r in got))
        pump(app, 0.3)
        if not ok:
            log('  (timed out waiting for a refined frame)')

    def grab(widget=None):
        return qimage_to_pil((widget or win).grab().toImage())

    def save(img, name):
        img.save(out / name)
        log('  saved', name, img.size)

    def viewer_frame():
        """The frame itself, without the viewer's margins and readout."""
        r = win.viewport.frame_rect()
        return grab(win.viewport).crop((int(r.x()) + 1, int(r.y()) + 1, int(r.right()), int(r.bottom())))

    # 1. first launch: the default campfire and the Getting started card
    settle()
    save(grab(), '01_first_launch.png')
    win._close_welcome()

    # 2. import the practice plate
    doc.import_footage(str(PLATE))
    pump(app, 30, until=lambda: doc.footage_info is not None)
    log('footage:', (doc.footage_info or {}).get('describe'))

    # 3. load the campfire, keeping the shot
    win.library.keep.setChecked(True)
    doc.load_preset('campfire', keep_shot=True)

    # 4. place it
    doc.set(('camera', 'anchor_x'), BASE[0], merge=False)
    doc.set(('camera', 'anchor_y'), BASE[1], merge=False)
    doc.set(('camera', 'scale'), SCALE, merge=False)
    doc.set_frame(SHOT_FRAME)
    doc.select(('emitter', 0))
    settle(SHOT_FRAME)
    tour = grab()
    d = ImageDraw.Draw(tour)
    lib, view, props, tl = (rect_in(win, x) for x in (win.library, win.viewport, win.props, win.timeline))
    comp_btn = rect_in(win, win.mode_group.button(0))
    for n, x, y in ((1, lib[0] + 100, lib[1] + 20), (2, comp_btn[0] - 18, comp_btn[1] + comp_btn[3] // 2),
                    (3, view[0] + view[2] - 28, view[1] + 26), (4, props[0] + 41, props[1] + 66),
                    (5, tl[0] + tl[2] // 2, tl[1] + 14)):
        badge(d, (x, y), n)
    save(tour, '02_window.png')

    # the handles, close up
    vx, vy, vw, vh = view
    base, top = win.viewport._handles()[:2]
    bx, by, ty = vx + int(base.x()), vy + int(base.y()), vy + int(top.y())
    y0, y1 = max(vy, ty - 30), min(vy + vh, by + 50)
    half = int((y1 - y0) * 0.8)
    save(grab().crop((max(vx, bx - half), y0, min(vx + vw, bx + half), y1)), '03_handles.png')

    # 5-6. look at it: composite, alpha, heat
    win.guides_btn.setChecked(False)
    tiles = []
    for mode in ('composite', 'alpha', 'heat'):
        win.set_mode(mode)
        settle(SHOT_FRAME)
        tiles.append(viewer_frame())
    win.set_mode('composite')
    win.guides_btn.setChecked(True)
    tw = 520
    th = int(tiles[0].height * tw / tiles[0].width)
    strip = Image.new('RGB', (tw * 3 + 16, th), (24, 24, 26))
    for i, img in enumerate(tiles):
        strip.paste(img.resize((tw, th), Image.LANCZOS), (i * (tw + 8), 0))
    save(strip, '04_view_modes.png')

    # 7. make it sit in the shot
    doc.select(('section', 'composite'))
    for k, v in COMPOSITE.items():
        doc.set(('composite', k), v, merge=False)
    doc.set(('motion', 'wind_speed'), WIND, merge=False)
    settle(SHOT_FRAME)
    pr = grab(win.props)
    save(pr.crop((0, 0, pr.width, min(pr.height, 640))), '05_composite.png')

    # 8. track the fire base through the pan
    track_fire_base(win)
    pump(app, 120, until=lambda: bool(doc.scene.track))
    pump(app, 0.5)
    log('track:', len((doc.scene.track or {}).get('points', {})), 'frames ·', win.msg.text())
    doc.set_frame(80)
    settle(80)
    save(viewer_frame(), '06_tracked.png')

    # 9. make it flare up
    path = ('combustion', 'fuel_scale')
    doc.select(('section', 'combustion'))
    for f, v in FLARE:
        doc.set_frame(f)
        pump(app, 0.2)
        if not doc.is_animated(path):
            doc.toggle_key(path)
        doc.set(path, v, merge=False)
    doc.set_frame(FLARE[1][0])
    settle(FLARE[1][0], seconds=180)
    save(grab(), '07_keyframes.png')

    # 10. save and render
    PROJECT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(PROJECT))
    log('saved project', PROJECT)
    dlg = ExportDialog(doc, win)
    dlg.folder.setText(str(Path('D:/Shots/campfire_shot')))
    for box, on in ((dlg.exr, True), (dlg.png, False), (dlg.mov, False), (dlg.comp, True), (dlg.vdb, False)):
        box.setChecked(on)
    dlg.comp_profile.setCurrentIndex(max(0, dlg.comp_profile.findData('h264')))
    dlg.show()
    pump(app, 0.5)
    save(grab(dlg), '08_render_dialog.png')
    dlg.close()

    # 12. a liquid in the same shot
    doc.load_preset(LIQUID, keep_shot=True)
    doc.set_frame(LIQUID_FRAME)
    doc.select(('section', 'water'))
    settle(LIQUID_FRAME, seconds=300)
    save(grab(), '09_liquid.png')

    w.stop()
    log('screens in', out)
    # remove the throwaway settings, and the presets folder the Library made under the throwaway name
    shutil.rmtree(settings_dir, ignore_errors=True)
    data = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
    if 'BlackbodyTutorial' in data.parts:
        shutil.rmtree(data.parent if data.parent.name == 'BlackbodyTutorial' else data, ignore_errors=True)
    os._exit(0)  # skip Qt's teardown of the still-connected document signals


if __name__ == '__main__':
    main()
