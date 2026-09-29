"""Tile images into a labelled contact sheet."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
files = sys.argv[2:]
cols = 3
tw, th = 480, 270
rows = (len(files) + cols - 1) // cols
sheet = Image.new('RGB', (cols * tw, rows * th), (20, 20, 20))
d = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(f).convert('RGB').resize((tw, th), Image.LANCZOS)
    x, y = (i % cols) * tw, (i // cols) * th
    sheet.paste(im, (x, y))
    d.text((x + 6, y + 4), Path(f).stem, fill=(220, 220, 220))
sheet.save(sys.argv[1])
