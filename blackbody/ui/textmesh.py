"""Words and pictures as solids: type some text in any font installed on the computer, or pick a logo or picture
(SVG, PNG, JPG), and get a 3D shape (an OBJ mesh) that burns, pours, collides or floats like any other mesh.

A picture's shape is traced from it: its transparency if it has any, else whatever stands out from the
background (dark on light or light on dark, judged from its border), at a threshold.

The outlines come from Qt's font engine; overlapping parts are merged, the curves thinned to a tolerance well
under a simulation cell, each letter's face triangulated (holes and all: the eye of an e, the counters of a B)
and the outline extruded into a closed solid. Letters stand on the ground (y = 0), centred on x, facing +z,
and are kept under the triangle count that gets the exact distance bake (engine/mesh.FAST_ABOVE).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QSize, QStandardPaths, Qt
from PySide6.QtGui import QFont, QFontDatabase, QFontInfo, QFontMetricsF, QImage, QPainter, QPainterPath, QPolygonF

EM = 1000.0          # pixel size the outlines are taken at, so Qt's curve flattening is far finer than needed
TOLERANCE = 0.006    # outline tolerance, as a fraction of the letter height
MAX_TRIANGLES = 2900  # under engine/mesh.FAST_ABOVE: small meshes get the exact bake
VERSION = 1          # bump when the geometry changes, so old files are made again
PREFERRED = ('Arial Black', 'Impact', 'Segoe UI Black', 'Helvetica Neue', 'Arial', 'DejaVu Sans')


def text_dir():
    return Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / 'text'


def default_family():
    have = set(QFontDatabase.families())
    return next((f for f in PREFERRED if f in have), QFont().family())


def qfont(family, bold=False, italic=False):
    f = QFont(family)
    f.setPixelSize(int(EM))
    f.setBold(bool(bold))
    f.setItalic(bool(italic))
    f.setKerning(True)
    return f


def outline(text, family, bold=False, italic=False):
    """The text's outline as a QPainterPath in font pixels (y down), every line centred, overlaps merged
    (odd-even fill), and the cap height in the same units."""
    font = qfont(family, bold, italic)
    fm = QFontMetricsF(font)
    path = QPainterPath()
    for k, line in enumerate(text.split('\n')):
        if line.strip():
            path.addText(QPointF(-fm.horizontalAdvance(line) / 2, k * fm.lineSpacing()), font, line)
    cap = fm.capHeight()
    if cap <= 0:
        cap = 0.72 * fm.ascent()
    return path.simplified(), cap


# -- outlines --------------------------------------------------------------------------------------------------

def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _chain(pts, tol):
    """Douglas-Peucker on an open chain (its ends are kept)."""
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        mid = pts[a + 1:b] - pts[a]
        n = float(np.hypot(*seg))
        if n < 1e-12:
            d = np.hypot(mid[:, 0], mid[:, 1])
        else:
            d = np.abs(seg[0] * mid[:, 1] - seg[1] * mid[:, 0]) / n
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


def simplify(pts, tol):
    """A closed outline thinned to within `tol` of the original."""
    if len(pts) <= 4:
        return pts
    far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    ring = np.concatenate([pts, pts[:1]])
    a = _chain(ring[:far + 1], tol)
    b = _chain(ring[far:], tol)
    return np.concatenate([a[:-1], b[:-1]])


def _clean(pts, eps):
    """Drop repeated points and points on a straight line between their neighbours."""
    out = [p for k, p in enumerate(pts) if k == 0 or np.hypot(*(p - pts[k - 1])) > eps]
    while len(out) > 1 and np.hypot(*(out[0] - out[-1])) <= eps:
        out.pop()
    changed = True
    while changed and len(out) > 3:
        changed = False
        for k in range(len(out)):
            a, b, c = out[k - 1], out[k], out[(k + 1) % len(out)]
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) <= eps * eps:
                del out[k]
                changed = True
                break
    return np.array(out, float).reshape(-1, 2)


def contours(text, family, bold=False, italic=False, height=0.4):
    """Closed outlines of the text in metres (y up): capital letters `height` tall, the whole text standing on
    y = 0 and centred on x = 0. Raises ValueError if there is nothing to draw."""
    path, cap = outline(text, family, bold, italic)
    s = float(height) / cap
    raw = []
    for poly in path.toSubpathPolygons():
        pts = np.array([(p.x() * s, -p.y() * s) for p in poly], float)
        if len(pts) >= 3:
            raw.append(pts)
    if not raw:
        raise ValueError('There is nothing to make: type some letters.')
    return polish(raw, height, 'There is nothing to make: type some letters.')


def polish(raw, height, empty='There is nothing to make.', min_area=0.004):
    """Outlines thinned (coarser until the shape fits the exact bake), specks dropped, the whole standing on y = 0
    and centred on x = 0."""
    tol = TOLERANCE * height
    for _ in range(8):   # coarser until the letters fit the exact bake
        out = []
        for pts in raw:
            c = _clean(simplify(_clean(pts, 1e-4 * height), tol), 1e-4 * height)
            if len(c) >= 3 and abs(_area(c)) > (min_area * height) ** 2:
                out.append(c)
        if sum(len(c) for c in out) * 4 <= MAX_TRIANGLES or tol > 0.03 * height:
            break
        tol *= 1.4
    if not out:
        raise ValueError(empty)
    allp = np.concatenate(out)
    lo, hi = allp.min(0), allp.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -lo[1]])
    return [c + shift for c in out]


IMAGE_FILTER = 'Logos and pictures (*.svg *.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff)'


def image_coverage(path, invert=False, max_px=720):
    """How much of each pixel is the shape (0..1), from a picture's transparency or else its contrast with its border.
    Pictures are drawn at up to max_px on their longer side (SVGs at exactly that)."""
    path = str(path)
    if path.lower().endswith('.svg'):
        from PySide6.QtSvg import QSvgRenderer
        r = QSvgRenderer(path)
        if not r.isValid():
            raise ValueError(f'{Path(path).name} is not an SVG this can read.')
        size = r.defaultSize()
        w, h = max(size.width(), 1), max(size.height(), 1)
        k = max_px / max(w, h)
        img = QImage(QSize(max(1, int(w * k)), max(1, int(h * k))), QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        r.render(p)
        p.end()
    else:
        img = QImage(path)
        if img.isNull():
            raise ValueError(f'{Path(path).name} is not a picture this can read.')
        if max(img.width(), img.height()) > max_px:
            img = img.scaled(max_px, max_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    img = img.convertToFormat(QImage.Format_RGBA8888)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4]
    a = a.reshape(img.height(), img.width(), 4).astype(np.float32) / 255.0
    alpha = a[..., 3]
    if alpha.min() < 0.98:   # it has transparency: that is the shape
        cov = alpha
    else:
        lum = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
        border = np.concatenate([lum[0], lum[-1], lum[:, 0], lum[:, -1]])
        cov = 1.0 - lum if border.mean() > 0.5 else lum   # dark on light, or light on dark
    return 1.0 - cov if invert else cov


def trace(cov, threshold=0.5):
    """Closed outlines where the coverage crosses the threshold (marching squares, pixel units: x right, y down)."""
    f = np.pad(np.asarray(cov, np.float32), 1)
    t = float(threshold)
    b = f > t
    case = (b[:-1, :-1].astype(np.int32) * 8 + b[:-1, 1:] * 4 + b[1:, 1:] * 2 + b[1:, :-1])
    ii, jj = np.nonzero((case > 0) & (case < 15))
    # the crossing on each edge of a cell: top (h, i, j), bottom (h, i+1, j), left (v, i, j), right (v, i, j+1)
    def cross(a, c):
        d = c - a
        return np.clip((t - a) / np.where(np.abs(d) < 1e-9, 1e-9, d), 0.0, 1.0)
    point = {}
    links = {}
    seg = {1: (('l',), ('b',)), 2: (('b',), ('r',)), 3: (('l',), ('r',)), 4: (('t',), ('r',)), 6: (('t',), ('b',)),
           7: (('l',), ('t',)), 8: (('l',), ('t',)), 9: (('t',), ('b',)), 11: (('t',), ('r',)), 12: (('l',), ('r',)),
           13: (('b',), ('r',)), 14: (('l',), ('b',))}
    tl, tr, br, bl = f[ii, jj], f[ii, jj + 1], f[ii + 1, jj + 1], f[ii + 1, jj]
    xt, xb = cross(tl, tr), cross(bl, br)
    yl, yr = cross(tl, bl), cross(tr, br)
    centre = (tl + tr + br + bl) / 4
    for n in range(len(ii)):
        i, j, c = int(ii[n]), int(jj[n]), int(case[ii[n], jj[n]])
        ids = {'t': ('h', i, j), 'b': ('h', i + 1, j), 'l': ('v', i, j), 'r': ('v', i, j + 1)}
        pos = {'t': (j + xt[n], i), 'b': (j + xb[n], i + 1), 'l': (j, i + yl[n]), 'r': (j + 1, i + yr[n])}
        if c in (5, 10):   # a saddle: the middle decides which corners join
            joined = (centre[n] > t) == (c == 5)
            pairs = [('l', 't'), ('b', 'r')] if joined else [('l', 'b'), ('t', 'r')]
        else:
            pairs = [(seg[c][0][0], seg[c][1][0])]
        for a_, b_ in pairs:
            ka, kb = ids[a_], ids[b_]
            point[ka], point[kb] = pos[a_], pos[b_]
            links.setdefault(ka, []).append(kb)
            links.setdefault(kb, []).append(ka)
    out = []
    seen = set()
    for start in links:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = [k for k in links[cur] if k != prev and k not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 3:
            out.append(np.array([point[k] for k in loop], float) - 1.0)   # back to unpadded pixels
    return out


def image_contours(path, height=1.0, invert=False, threshold=0.5, max_px=720):
    """A picture's shape as closed outlines in metres (y up), `height` tall, standing on y = 0 and centred on x = 0."""
    cov = image_coverage(path, invert, max_px)
    raw = [np.column_stack([c[:, 0], -c[:, 1]]) for c in trace(cov, threshold)]
    if not raw:
        raise ValueError('Nothing stands out in that picture: try Invert, or another threshold.')
    allp = np.concatenate(raw)
    tall = float(np.ptp(allp[:, 1]))
    if tall <= 0:
        raise ValueError('Nothing stands out in that picture: try Invert, or another threshold.')
    s = float(height) / tall
    return polish([c * s for c in raw], height, 'Nothing stands out in that picture: try Invert, or another threshold.',
                  min_area=0.01)


def contour_path(cs):
    """Outlines as a QPainterPath (odd-even fill, y down), for drawing a preview."""
    path = QPainterPath()
    path.setFillRule(Qt.OddEvenFill)
    for c in cs:
        path.addPolygon(QPolygonF([QPointF(float(x), float(-y)) for x, y in c]))
        path.closeSubpath()
    return path


def _inside(pt, poly):
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    hit = ((ys > y) != (yn > y)) & (x < (xn - xs) * (y - ys) / np.where(yn == ys, 1e-30, yn - ys) + xs)
    return bool(np.count_nonzero(hit) % 2)


def faces(cs):
    """The outlines grouped into faces: [(outer, [holes])], outer anticlockwise and holes clockwise (y up). With
    overlaps merged, an outline inside an odd number of others is a hole in the smallest of them."""
    areas = [abs(_area(c)) for c in cs]
    inside = [[j for j in range(len(cs)) if j != i and areas[j] > areas[i] and _inside(cs[i][0], cs[j])] for i in range(len(cs))]
    depth = [len(x) for x in inside]
    groups = {i: [] for i in range(len(cs)) if depth[i] % 2 == 0}
    for i in range(len(cs)):
        if depth[i] % 2:
            parents = [j for j in inside[i] if depth[j] == depth[i] - 1]
            if parents:
                groups[min(parents, key=areas.__getitem__)].append(i)
    out = []
    for i, hs in groups.items():
        o = cs[i] if _area(cs[i]) > 0 else cs[i][::-1]
        out.append((o, [cs[h] if _area(cs[h]) < 0 else cs[h][::-1] for h in hs]))
    return out


# -- triangulation (a face with holes: bridge each hole to the outline, then clip ears) -------------------------

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _in_triangle(p, a, b, c, eps):
    return _cross(a, b, p) >= -eps and _cross(b, c, p) >= -eps and _cross(c, a, p) >= -eps


def _bridge(V, ring, hole, eps):
    """Join a hole (clockwise) into the ring (anticlockwise) through its rightmost point (Eberly's method)."""
    m = max(range(len(hole)), key=lambda k: (V[hole[k]][0], -V[hole[k]][1]))
    M = V[hole[m]]
    n = len(ring)
    best_x, pk, hit = np.inf, None, None
    for k in range(n):
        a, b = V[ring[k]], V[ring[(k + 1) % n]]
        if a[1] <= M[1] <= b[1] and a[1] < b[1]:   # edges going up: the outline's right-hand side seen from inside
            x = a[0] + (M[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if M[0] - eps <= x < best_x:
                best_x = x
                pk = k if a[0] >= b[0] else (k + 1) % n
                hit = (k, (k + 1) % n)
    if pk is None:   # numerically lost: join to the nearest point of the ring
        pk = min(range(n), key=lambda k: float(np.hypot(*(V[ring[k]] - M))))
    else:
        I = np.array([best_x, M[1]])
        P = V[ring[pk]]
        if abs(I[0] - V[ring[hit[0]]][0]) <= eps and abs(I[1] - V[ring[hit[0]]][1]) <= eps:
            pk = hit[0]
        elif abs(I[0] - V[ring[hit[1]]][0]) <= eps and abs(I[1] - V[ring[hit[1]]][1]) <= eps:
            pk = hit[1]
        else:
            # a reflex point of the ring inside the triangle M, I, P would block the bridge: take the one nearest
            # the ray instead
            tri = (M, I, P) if _cross(M, I, P) > 0 else (M, P, I)
            best = None
            for k in range(n):
                R = V[ring[k]]
                if k == pk or R[0] < M[0] - eps:
                    continue
                if _cross(V[ring[k - 1]], R, V[ring[(k + 1) % n]]) > eps:
                    continue   # convex
                if not _in_triangle(R, *tri, eps):
                    continue
                key = (abs(R[1] - M[1]) / max(R[0] - M[0], 1e-12), float(np.hypot(*(R - M))))
                if best is None or key < best[0]:
                    best = (key, k)
            if best is not None:
                pk = best[1]
    return ring[:pk + 1] + hole[m:] + hole[:m] + [hole[m], ring[pk]] + ring[pk + 1:]


def _earclip(V, ring, eps):
    idx = list(ring)
    tris = []
    start = 0
    while len(idx) > 3:
        n = len(idx)
        conv = [_cross(V[idx[k - 1]], V[idx[k]], V[idx[(k + 1) % n]]) for k in range(n)]
        reflex = [idx[k] for k in range(n) if conv[k] <= eps]
        found = None
        for t in range(n):
            k = (start + t) % n
            if conv[k] <= eps:
                continue
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % n]
            A, B, C = V[a], V[b], V[c]
            ok = True
            for j in reflex:
                if j in (a, b, c):
                    continue
                P = V[j]
                if (abs(P - A) <= eps).all() or (abs(P - B) <= eps).all() or (abs(P - C) <= eps).all():
                    continue   # the other end of a bridge
                if _in_triangle(P, A, B, C, eps):
                    ok = False
                    break
            if ok:
                found = k
                break
        if found is None:   # numerical trouble: clip the most convex corner anyway
            found = int(np.argmax(conv))
            if conv[found] <= 0:
                tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
                return tris
        tris.append((idx[found - 1], idx[found], idx[(found + 1) % n]))
        del idx[found]
        start = found % max(len(idx), 1)
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def triangulate(outer, holes):
    """Triangles over a face (anticlockwise outline, clockwise holes), as index triples into the outline's points
    followed by each hole's."""
    V = np.concatenate([outer] + list(holes))
    eps = 1e-9 * max(1.0, float(np.abs(V).max()) ** 2)
    ring = list(range(len(outer)))
    offs = np.cumsum([len(outer)] + [len(h) for h in holes])
    order = sorted(range(len(holes)), key=lambda k: -float(holes[k][:, 0].max()))
    for h in order:
        ring = _bridge(V, ring, list(range(int(offs[h]), int(offs[h]) + len(holes[h]))), eps)
    return _earclip(V, ring, eps)


def extrude(groups, depth):
    """A closed solid from faces: front at z = +depth/2, back at -depth/2, walls round every outline. Returns
    vertices (n, 3) and triangles (m, 3), wound outwards."""
    verts, tris = [], []
    zf, zb = depth / 2, -depth / 2
    base = 0
    for outer, holes in groups:
        loops = [outer] + list(holes)
        P = np.concatenate(loops)
        n = len(P)
        verts.append(np.column_stack([P, np.full(n, zf)]))
        verts.append(np.column_stack([P, np.full(n, zb)]))
        for a, b, c in triangulate(outer, holes):
            tris.append((base + a, base + b, base + c))
            tris.append((base + n + a, base + n + c, base + n + b))
        o = 0
        for loop in loops:
            m = len(loop)
            for i in range(m):
                j = (i + 1) % m
                fi, fj, bi, bj = base + o + i, base + o + j, base + n + o + i, base + n + o + j
                tris.append((bi, bj, fj))
                tris.append((bi, fj, fi))
            o += m
        base += 2 * n
    return np.concatenate(verts), np.array(tris, np.int64).reshape(-1, 3)


# -- files -----------------------------------------------------------------------------------------------------

def spec_of(text, family, bold=False, italic=False, height=0.4, depth=0.1):
    return {'text': str(text), 'font': str(family), 'bold': bool(bold), 'italic': bool(italic),
            'height': round(float(height), 4), 'depth': round(float(depth), 4)}


def image_spec(image, height=1.0, depth=0.1, invert=False, threshold=0.5):
    return {'kind': 'image', 'image': str(image), 'invert': bool(invert), 'threshold': round(float(threshold), 3),
            'height': round(float(height), 4), 'depth': round(float(depth), 4)}


def label_of(spec):
    """An object name for a shape: its words, or the picture's name, in quotes."""
    t = ' '.join(str(spec.get('text') or Path(spec.get('image', 'shape')).stem).split())
    return f'“{t[:24]}{"…" if len(t) > 24 else ""}”'


def make(spec):
    """The OBJ for a text spec (see spec_of), made now or found from before, with the spec beside it (.json) so the
    text can be edited later. Returns (path, (width, height, depth) of the letters in metres, triangles)."""
    image = spec.get('kind') == 'image'
    if image:   # the picture as it is now: a changed file makes a new shape
        st = Path(spec['image']).stat()
        used = f'{st.st_size}:{st.st_mtime_ns}'
        name = Path(spec['image']).stem
    else:
        used = QFontInfo(qfont(spec['font'], spec['bold'], spec['italic'])).family()   # the font really drawn, if that one is missing
        name = spec['text']
    key = hashlib.sha1(json.dumps(dict(spec, v=VERSION, used=used), sort_keys=True).encode()).hexdigest()[:10]
    slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')[:24] or ('shape' if image else 'text')
    folder = text_dir()
    path = folder / f'{slug}_{key}.obj'
    info = path.with_suffix('.json')
    if path.exists() and info.exists():
        try:
            meta = json.loads(info.read_text(encoding='utf-8'))
            return str(path), tuple(meta['extent']), int(meta['triangles'])
        except (OSError, ValueError, KeyError):
            pass
    if image:
        cs = image_contours(spec['image'], spec['height'], spec['invert'], spec['threshold'])
    else:
        cs = contours(spec['text'], spec['font'], spec['bold'], spec['italic'], spec['height'])
    verts, tris = extrude(faces(cs), spec['depth'])
    extent = tuple(round(float(x), 4) for x in verts.max(0) - verts.min(0))
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f'# Blackbody shape from {Path(spec["image"]).name}\n' if image else f'# Blackbody text: {spec["text"]!r} in {spec["font"]}\n']
    lines += [f'v {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in verts]
    lines += [f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in tris]
    tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(lines), encoding='utf-8')
    tmp.replace(path)
    info.write_text(json.dumps(dict(spec, extent=extent, triangles=int(len(tris))), indent=1), encoding='utf-8')
    return str(path), extent, int(len(tris))


def spec_for(mesh):
    """The text spec of a mesh made here, or None for any other mesh."""
    if not mesh or not str(mesh).lower().endswith('.obj'):
        return None
    info = Path(str(mesh).split('#')[0]).with_suffix('.json')
    try:
        meta = json.loads(info.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get('kind') == 'image' and meta.get('image'):
        return image_spec(meta['image'], meta.get('height', 1.0), meta.get('depth', 0.1), meta.get('invert', False),
                          meta.get('threshold', 0.5))
    if 'text' not in meta or 'font' not in meta:
        return None
    return spec_of(meta['text'], meta['font'], meta.get('bold', False), meta.get('italic', False), meta.get('height', 0.4),
                   meta.get('depth', 0.1))
