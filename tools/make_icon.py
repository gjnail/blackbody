"""Write the application icon (.ico and .png) from the vector icon used in the app."""
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize
from PySide6.QtWidgets import QApplication

app = QApplication([])
from blackbody.ui.icons import app_icon  # noqa: E402

out = ROOT / 'blackbody' / 'assets'
out.mkdir(parents=True, exist_ok=True)
icon = app_icon()
icon.pixmap(QSize(256, 256)).save(str(out / 'blackbody.png'))
from PIL import Image  # noqa: E402

im = Image.open(out / 'blackbody.png')
im.save(out / 'blackbody.ico', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print('wrote', out / 'blackbody.ico')
