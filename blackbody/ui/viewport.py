"""The viewer: the rendered frame, guides projected through the shot camera, and direct manipulation."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

from ..engine import camera as cam
from . import theme

MODE_NAMES = {'composite': 'Composite', 'fire': 'Fire over black', 'alpha': 'Alpha', 'emission': 'Emission',
              'heat': 'Heat (haze)', 'depth': 'Depth', 'temperature': 'Temperature'}


def _circle(center, radius_vec_a, radius_vec_b, n=48):
    t = np.linspace(0, 2 * np.pi, n + 1)
    return center[None, :] + np.cos(t)[:, None] * radius_vec_a[None, :] + np.sin(t)[:, None] * radius_vec_b[None, :]


_MESH_EDGES = {}


def mesh_edges(path, limit=900):
    """A sample of a mesh's edges (k, 2, 3), in mesh units, cached per file; None if it cannot be read."""
    import os
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    hit = _MESH_EDGES.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        from ..engine.mesh import load_mesh
        v, t = load_mesh(path)
    except Exception:
        return None
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    if len(e) > limit:
        e = e[np.linspace(0, len(e) - 1, limit).astype(int)]
    segs = v[e].astype(float)
    _MESH_EDGES[path] = (key, segs)
    return segs


def _rot_y(pts, yaw_deg):
    """Rotate object-space offsets about y into fire-local space (matches the solver's rotation)."""
    a = np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    q = np.asarray(pts, float)
    return np.stack([c * q[..., 0] + s * q[..., 2], q[..., 1], -s * q[..., 0] + c * q[..., 2]], axis=-1)


def shape_lines(shape, pos, size, end=None, yaw=0.0, mesh=None):
    """Polylines (lists of Nx3 arrays) outlining an emitter or collider shape in fire-local space."""
    p = np.asarray(pos, float)
    s = np.asarray(size, float)
    if shape == 'mesh':
        segs = mesh_edges(mesh) if mesh else None
        if segs is None:
            return shape_lines('sphere', pos, (0.1, 0.1, 0.1))
        return [p + _rot_y(seg * s, yaw) for seg in segs]
    if shape != 'capsule' and yaw:
        return [p + _rot_y(line - p, yaw) for line in shape_lines(shape, pos, size, end)]
    X, Y, Z = np.eye(3)
    out = []
    if shape == 'box':
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s + p
        for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
            out.append(np.array([c[a], c[b]]))
    elif shape == 'sphere':
        out.append(_circle(p, X * s[0], Z * s[2]))
        out.append(_circle(p, X * s[0], Y * s[1]))
        out.append(_circle(p, Z * s[2], Y * s[1]))
    elif shape in ('cylinder', 'cone'):
        top_r = 0.0 if shape == 'cone' else s[0]
        out.append(_circle(p - Y * s[1], X * s[0], Z * s[0]))
        if top_r > 0:
            out.append(_circle(p + Y * s[1], X * top_r, Z * top_r))
        for ang in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
            d = np.array([math.cos(ang), 0, math.sin(ang)])
            out.append(np.array([p - Y * s[1] + d * s[0], p + Y * s[1] + d * top_r]))
    elif shape == 'ring':
        out.append(_circle(p, X * s[0], Z * s[0]))
        out.append(_circle(p, X * (s[0] + s[1]), Z * (s[0] + s[1])))
        out.append(_circle(p, X * max(s[0] - s[1], 0.0), Z * max(s[0] - s[1], 0.0)))
    elif shape == 'capsule':
        e = np.asarray(end, float)
        r = s[0]
        out.append(np.array([p, e]))
        for c in (p, e):
            out.append(_circle(c, X * r, Z * r, 24))
            out.append(_circle(c, X * r, Y * r, 24))
    return out


class Viewport(QWidget):
    frameClicked = Signal()

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.image = None
        self.stats = {}
        self.mode = 'composite'
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.guides = True
        self.sim_msg = ''
        self._drag = None
        self._hover = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 200)
        self.setAutoFillBackground(False)
        doc.paramChanged.connect(lambda *_: self.update())
        doc.sceneReplaced.connect(self.update)
        doc.structureChanged.connect(self.update)
        doc.selectionChanged.connect(lambda *_: self.update())
        doc.frameChanged.connect(lambda *_: self.update())

    # -- data in ---------------------------------------------------------------------------------------

    def set_frame_image(self, img: QImage, frame, stats):
        self.image = img
        self.stats = stats
        self.update()

    # -- geometry -----------------------------------------------------------------------------------------

    def out_size(self):
        return self.doc.scene.output_size()

    def frame_rect(self):
        W, H = self.out_size()
        vw, vh = max(1, self.width()), max(1, self.height() - 0)
        s = min(vw / W, vh / H) * 0.94 * self.zoom
        w, h = W * s, H * s
        return QRectF((vw - w) / 2 + self.pan.x(), (vh - h) / 2 + self.pan.y(), w, h)

    def to_widget(self, px):
        r = self.frame_rect()
        W, H = self.out_size()
        return QPointF(r.x() + px[0] / W * r.width(), r.y() + px[1] / H * r.height())

    def to_frame(self, pt):
        r = self.frame_rect()
        W, H = self.out_size()
        return ((pt.x() - r.x()) / r.width() * W, (pt.y() - r.y()) / r.height() * H)

    def camstate(self):
        sc = self.doc.scene
        spec, fire = sc.camera(self.doc.frame)
        W, H = self.out_size()
        return cam.compute(spec, W / H, fire), fire, spec

    def _project_local(self, cs, fire, pts):
        L = fire.local_to_world()
        pts = np.atleast_2d(np.asarray(pts, float))
        wpts = pts @ L[:3, :3].T + L[:3, 3]
        W, H = self.out_size()
        return cam.project(cs, wpts, W, H)

    def _handles(self):
        """Screen positions of the fire base (anchor) and the scale handle (top of the box)."""
        cs, fire, spec = self.camstate()
        sy = self.doc.scene.domain_size()[1]
        px, ok = self._project_local(cs, fire, [[0, 0, 0], [0, sy, 0]])
        return self.to_widget(px[0]), self.to_widget(px[1]), ok.all(), spec

    # -- painting -------------------------------------------------------------------------------------------

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.VIEWER))
        r = self.frame_rect()
        if self.image is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(r, self.image)
        else:
            p.fillRect(r, QColor('#000'))
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)
        if self.guides:
            p.save()
            p.setClipRect(r.adjusted(-1, -1, 1, 1))  # guides stay inside the frame
            try:
                self._paint_guides(p)
            except Exception:
                pass
            p.restore()
        self._paint_hud(p, r)

    def _poly(self, p, cs, fire, pts, pen):
        px, ok = self._project_local(cs, fire, pts)
        if not ok.all():
            return
        path = QPainterPath()
        path.moveTo(self.to_widget(px[0]))
        for q in px[1:]:
            path.lineTo(self.to_widget(q))
        p.setPen(pen)
        p.drawPath(path)

    def _lines(self, p, cs, fire, lines, pen):
        """Draw many polylines with one projection (mesh wireframes have hundreds)."""
        if not lines:
            return
        sizes = [len(ln) for ln in lines]
        px, ok = self._project_local(cs, fire, np.concatenate(lines))
        path = QPainterPath()
        i = 0
        for n in sizes:
            if ok[i:i + n].all():
                path.moveTo(self.to_widget(px[i]))
                for q in px[i + 1:i + n]:
                    path.lineTo(self.to_widget(q))
            i += n
        p.setPen(pen)
        p.drawPath(path)

    def _paint_guides(self, p):
        sc = self.doc.scene
        cs, fire, spec = self.camstate()
        sx, sy, sz = sc.domain_size()
        # ground grid around the fire
        grid_pen = QPen(QColor(255, 255, 255, 22), 1)
        ext = max(sx, sz) * 1.5
        step = 10 ** math.floor(math.log10(max(ext / 4, 1e-3)))
        n = int(ext / step)
        for i in range(-n, n + 1):
            self._poly(p, cs, fire, [[i * step, 0, -ext], [i * step, 0, ext]], grid_pen)
            self._poly(p, cs, fire, [[-ext, 0, i * step], [ext, 0, i * step]], grid_pen)
        # simulation box
        box = QPen(QColor(255, 255, 255, 70), 1, Qt.DashLine)
        for line in shape_lines('box', (0, sy / 2, 0), (sx / 2, sy / 2, sz / 2)):
            self._poly(p, cs, fire, line, box)
        sel = self.doc.selection
        for i, em in enumerate(sc.emitters):
            if not em['enabled']:
                continue
            g = lambda k: sc.get(('emitter', i, k), self.doc.frame)
            is_sel = sel == ('emitter', i)
            pen = QPen(QColor(theme.ACCENT) if is_sel else QColor(255, 170, 90, 140), 1.6 if is_sel else 1.0)
            self._lines(p, cs, fire, shape_lines(em['shape'], g('position'), g('size'), g('end'), g('yaw'),
                                                 sc.mesh_path(em['mesh'])), pen)
        for i, c in enumerate(sc.colliders):
            if not c['enabled']:
                continue
            g = lambda k: sc.get(('collider', i, k), self.doc.frame)
            is_sel = sel == ('collider', i)
            col = QColor(255, 140, 80) if c.get('burnable') else QColor(120, 190, 255)
            col.setAlpha(230 if is_sel else 120)
            pen = QPen(col, 1.6 if is_sel else 1.0)
            self._lines(p, cs, fire, shape_lines(c['shape'], g('position'), g('size'), None, g('yaw'), sc.mesh_path(c['mesh'])), pen)
            op = np.asarray(g('opening'), float)
            if (op > 0).all():  # the doorway or window cut through it
                at = np.asarray(g('position'), float) + _rot_y(np.asarray(g('opening_at'), float), g('yaw'))
                self._lines(p, cs, fire, shape_lines('box', at, op, None, g('yaw')), QPen(col, 1.0, Qt.DotLine))
        # rotate and resize handles of the selected emitter or collider
        oh = self._object_handles()
        if oh is not None:
            drag = (self._drag or {}).get('kind')
            ring = QPen(QColor(theme.ACCENT), 1.0, Qt.DashLine)
            self._lines(p, cs, fire, [oh['ring']], ring)
            for key, shape in (('rot', 'circle'), ('size', 'square')):
                q = oh.get(key)
                if q is None:
                    continue
                hot = self._hover == key or drag == key
                p.setPen(QPen(QColor(theme.ACCENT), 1.4))
                p.setBrush(QColor(theme.ACCENT) if hot else QColor(20, 20, 24))
                if shape == 'circle':
                    p.drawEllipse(q, 5, 5)
                else:
                    p.drawRect(QRectF(q.x() - 4.5, q.y() - 4.5, 9, 9))
        # tracked path of the fire base
        if sc.track and sc.track.get('points'):
            W, H = self.out_size()
            off = sc.track.get('offset', (0.0, 0.0))
            pts = sorted(sc.track['points'].items())
            path = QPainterPath()
            for j, (fr, (x, y)) in enumerate(pts):
                q = self.to_widget(((x + off[0]) * W, (y + off[1]) * H))
                path.moveTo(q) if j == 0 else path.lineTo(q)
            p.setPen(QPen(QColor(120, 220, 140, 150), 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        # placement handles
        base, top, ok, spec = self._handles()
        if ok and spec.use_anchor:
            hot = self._hover == 'anchor' or (self._drag or {}).get('kind') == 'anchor'
            p.setPen(QPen(QColor(theme.ACCENT if hot else '#ffffff'), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(base, 7, 7)
            p.drawLine(base + QPointF(-12, 0), base + QPointF(-4, 0))
            p.drawLine(base + QPointF(4, 0), base + QPointF(12, 0))
            p.drawLine(base + QPointF(0, -12), base + QPointF(0, -4))
            p.drawLine(base + QPointF(0, 4), base + QPointF(0, 12))
            hot = self._hover == 'scale' or (self._drag or {}).get('kind') == 'scale'
            p.setPen(QPen(QColor(theme.ACCENT if hot else '#ffffff'), 1.2))
            p.setBrush(QColor(theme.ACCENT if hot else '#ffffff'))
            p.drawRect(QRectF(top.x() - 4, top.y() - 4, 8, 8))

    def _paint_hud(self, p, r):
        st = self.stats
        f = QFont(theme.mono_font(8.5))
        p.setFont(f)
        lines = [f'{MODE_NAMES.get(self.mode, self.mode)}   frame {self.doc.frame}']
        if st:
            dims = st.get('dims') or (0, 0, 0)
            lines.append(f'{dims[0]}×{dims[1]}×{dims[2]} voxels · {st.get("cell_mm", 0):.1f} mm · {st.get("substeps", 0)} substeps')
            if st.get('kind') == 'liquid':
                ww = st.get('whitewater', 0)
                lines.append(f'{st.get("particles", 0) / 1e6:.2f} M particles · '
                             + (f'{ww / 1e3:.0f} k whitewater' if ww >= 10000 else f'{ww} whitewater'))
                if st.get('particle_limit'):
                    lines.append('particle limit reached: sources are held back (Liquid › Particle limit, advanced)')
            lines.append(f'sim {st.get("sim_ms", 0):.0f} ms · render {st.get("render_ms", 0):.0f} ms · max {st.get("max_speed", 0):.1f} m/s')
            if st.get('refined'):
                lines.append('refined (4 samples)')
        if self.sim_msg:
            lines.append(self.sim_msg)
        p.setPen(QColor(0, 0, 0, 160))
        y = 18
        for ln in lines:
            p.drawText(QPointF(13, y + 1), ln)
            y += 15
        p.setPen(QColor(230, 230, 232, 220))
        y = 18
        for ln in lines:
            p.drawText(QPointF(12, y), ln)
            y += 15
        W, H = self.out_size()
        p.setPen(QColor(255, 255, 255, 90))
        p.drawText(QPointF(r.right() - 90, r.bottom() + 14), f'{W}×{H}')

    # -- hit testing ------------------------------------------------------------------------------------------

    def _hit(self, pos):
        try:
            base, top, ok, spec = self._handles()
        except Exception:
            return None
        if ok and spec.use_anchor:
            if (pos - base).manhattanLength() < 16:
                return 'anchor'
            if (pos - top).manhattanLength() < 14:
                return 'scale'
        oh = self._object_handles()
        if oh is not None:
            for key in ('rot', 'size'):
                q = oh.get(key)
                if q is not None and (q - pos).manhattanLength() < 12:
                    return key
        sc = self.doc.scene
        cs, fire, _ = self.camstate()
        best, bd = None, 12.0
        for kind, items in (('emitter', sc.emitters), ('collider', sc.colliders)):
            for i, it in enumerate(items):
                if not it['enabled']:
                    continue
                px, ok = self._project_local(cs, fire, [sc.get((kind, i, 'position'), self.doc.frame)])
                if ok[0]:
                    d = (self.to_widget(px[0]) - pos).manhattanLength()
                    if d < bd:
                        best, bd = (kind, i), d
        return best

    def _object_handles(self):
        """Screen positions of the selected emitter's or collider's handles: a ring around it with a
        knob to turn it, and a square above it to resize it. None if nothing suitable is selected."""
        sel = self.doc.selection
        if not sel or sel[0] not in ('emitter', 'collider'):
            return None
        sc = self.doc.scene
        kind, i = sel
        items = sc.emitters if kind == 'emitter' else sc.colliders
        if i >= len(items) or not items[i]['enabled']:
            return None
        it = items[i]
        g = lambda k: sc.get((kind, i, k), self.doc.frame)
        pos = np.asarray(g('position'), float)
        size = np.asarray(g('size'), float)
        if it['shape'] == 'mesh':
            segs = mesh_edges(sc.mesh_path(it['mesh']))
            ext = (np.abs(segs).reshape(-1, 3).max(0) if segs is not None else np.full(3, 0.5)) * size
        elif it['shape'] == 'sphere' and kind == 'collider':
            ext = np.full(3, size[0])
        elif it['shape'] in ('cylinder', 'cone') or (it['shape'] == 'sphere'):
            ext = np.array([size[0], size[1], size[0] if it['shape'] != 'sphere' else size[2]])
        elif it['shape'] == 'ring':
            ext = np.array([size[0] + size[1], size[1], size[0] + size[1]])
        elif it['shape'] == 'capsule':
            ext = np.full(3, size[0])
        else:
            ext = size
        radius = max(float(np.hypot(ext[0], ext[2])), 0.05) * 1.15
        cs, fire, _ = self.camstate()
        ring = _circle(pos, np.array([radius, 0, 0]), np.array([0, 0, radius]), 64)
        out = {'ring': ring}
        pts = [pos + np.array([0.0, max(ext[1], 0.05) * 1.3 + 0.02 * radius, 0.0])]
        if it['shape'] != 'capsule':
            pts.append(pos + _rot_y(np.array([radius, 0.0, 0.0]), g('yaw')))
        px, ok = self._project_local(cs, fire, pts)
        if ok[0]:
            out['size'] = self.to_widget(px[0])
        if len(pts) > 1 and ok[1]:
            out['rot'] = self.to_widget(px[1])
        out['centre'] = pos
        return out

    # -- interaction --------------------------------------------------------------------------------------------

    def mousePressEvent(self, e):
        pos = e.position()
        mods = e.modifiers()
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and mods & Qt.ShiftModifier and mods & Qt.ControlModifier):
            self._drag = {'kind': 'pan', 'start': pos, 'pan': QPointF(self.pan)}
            return
        if e.button() == Qt.LeftButton and mods & Qt.AltModifier:
            sc = self.doc.scene
            self._drag = {'kind': 'orbit', 'start': pos, 'yaw': sc.get(('camera', 'yaw'), self.doc.frame),
                          'pitch': sc.get(('camera', 'pitch'), self.doc.frame)}
            return
        if e.button() == Qt.RightButton and mods & Qt.AltModifier:
            self._drag = {'kind': 'dolly', 'start': pos, 'dist': self.doc.scene.get(('camera', 'distance'), self.doc.frame)}
            return
        if e.button() == Qt.LeftButton:
            hit = self._hit(pos)
            if hit == 'anchor':
                self._drag = {'kind': 'anchor'}
            elif hit == 'scale':
                base, top, ok, spec = self._handles()
                self._drag = {'kind': 'scale', 'base': base, 'd0': max(8.0, base.y() - pos.y()), 's0': spec.scale}
            elif hit in ('rot', 'size'):
                kind, i = self.doc.selection
                sc = self.doc.scene
                c = np.asarray(sc.get((kind, i, 'position'), self.doc.frame), float)
                gp = self._ground_point(pos, c[1])
                self._drag = {'kind': hit, 'what': kind, 'i': i, 'c': c, 'sy': pos.y(),
                              'yaw0': sc.get((kind, i, 'yaw'), self.doc.frame),
                              'size0': np.asarray(sc.get((kind, i, 'size'), self.doc.frame), float),
                              'a0': None if gp is None else math.atan2(gp[2] - c[2], gp[0] - c[0])}
            elif isinstance(hit, tuple):
                self.doc.select(hit)
                kind, i = hit
                sc = self.doc.scene
                p0 = np.asarray(sc.get((kind, i, 'position'), self.doc.frame), float)
                e0 = np.asarray(sc.get(('emitter', i, 'end'), self.doc.frame), float) if kind == 'emitter' else p0
                self._drag = {'kind': 'move', 'what': kind, 'i': i, 'y': float(p0[1]), 'start_world': self._ground_point(pos, p0[1]),
                              'p0': p0, 'e0': e0, 'vertical': bool(mods & Qt.ShiftModifier), 'sy': pos.y()}
            self.update()

    def _ground_point(self, pos, y_local):
        """Fire-local point where the pixel ray under `pos` meets the horizontal plane at height y."""
        cs, fire, _ = self.camstate()
        W, H = self.out_size()
        fx, fy = self.to_frame(pos)
        o, d = cam.pixel_ray(cs, fx, fy, W, H)
        L = fire.local_to_world()
        Li = np.linalg.inv(L)
        o_l = (Li @ np.append(o, 1.0))[:3]
        d_l = Li[:3, :3] @ d
        if abs(d_l[1]) < 1e-6:
            return None
        t = (y_local - o_l[1]) / d_l[1]
        if t <= 0:
            return None
        return o_l + d_l * t

    def mouseMoveEvent(self, e):
        pos = e.position()
        d = self._drag
        if d is None:
            h = self._hit(pos)
            h = h if isinstance(h, str) else None
            if h != self._hover:
                self._hover = h
                cursors = {'anchor': Qt.SizeAllCursor, 'scale': Qt.SizeVerCursor, 'size': Qt.SizeVerCursor, 'rot': Qt.OpenHandCursor}
                self.setCursor(cursors.get(h, Qt.ArrowCursor))
                self.update()
            return
        k = d['kind']
        if k == 'pan':
            self.pan = d['pan'] + (pos - d['start'])
            self.update()
        elif k == 'orbit':
            dx = pos.x() - d['start'].x()
            dy = pos.y() - d['start'].y()
            self.doc.set(('camera', 'yaw'), d['yaw'] - dx * 0.3)
            self.doc.set(('camera', 'pitch'), max(-30.0, min(89.0, d['pitch'] + dy * 0.2)))
        elif k == 'dolly':
            dy = pos.y() - d['start'].y()
            self.doc.set(('camera', 'distance'), max(0.3, d['dist'] * math.exp(dy * 0.005)))
        elif k == 'anchor':
            W, H = self.out_size()
            fx, fy = self.to_frame(pos)
            sc = self.doc.scene
            if sc.track and sc.track.get('points'):
                from ..scene.model import track_point
                tp = track_point(sc.track, self.doc.frame)
                off = (fx / W - tp[0], fy / H - tp[1])

                def fn(s, off=off):
                    s.track['offset'] = off
                self.doc.edit('Move fire', fn, merge_key=('track-offset', self.doc._gen))
            else:
                self.doc.set(('camera', 'anchor_x'), fx / W)
                self.doc.set(('camera', 'anchor_y'), fy / H)
        elif k == 'scale':
            dist = max(4.0, d['base'].y() - pos.y())
            self.doc.set(('camera', 'scale'), max(0.02, d['s0'] * dist / d['d0']))
        elif k == 'move':
            i, what = d['i'], d['what']
            if d['vertical']:
                dy = (d['sy'] - pos.y()) * 0.01 * max(self.doc.scene.domain_size()) / 3
                new = d['p0'] + np.array([0.0, dy, 0.0])
            else:
                hitp = self._ground_point(pos, d['y'])
                if hitp is None or d['start_world'] is None:
                    return
                new = d['p0'] + (hitp - d['start_world']) * np.array([1.0, 0.0, 1.0])
            delta = new - d['p0']
            self.doc.set((what, i, 'position'), tuple(float(x) for x in new))
            if what == 'emitter' and self.doc.scene.emitters[i]['shape'] == 'capsule':
                self.doc.set(('emitter', i, 'end'), tuple(float(x) for x in d['e0'] + delta))
        elif k == 'rot':
            gp = self._ground_point(pos, d['c'][1])
            if gp is None or d['a0'] is None:
                return
            a = math.atan2(gp[2] - d['c'][2], gp[0] - d['c'][0])
            # a positive rotation turns +x toward -z, so the angle on the ground runs the other way
            yaw = d['yaw0'] - math.degrees(a - d['a0'])
            yaw = (yaw + 180.0) % 360.0 - 180.0
            if e.modifiers() & Qt.ShiftModifier:
                yaw = round(yaw / 15.0) * 15.0
            self.doc.set((d['what'], d['i'], 'yaw'), yaw)
        elif k == 'size':
            f = math.exp((d['sy'] - pos.y()) * 0.01)
            self.doc.set((d['what'], d['i'], 'size'), tuple(float(max(x * f, 1e-3)) for x in d['size0']))

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            self.doc.end_drag()
            self.update()

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and self._hit(e.position()) is None:
            self.fit()

    def wheelEvent(self, e):
        f = 1.15 ** (e.angleDelta().y() / 120.0)
        pos = e.position()
        c = QPointF(self.width() / 2, self.height() / 2)
        old = self.zoom
        self.zoom = max(0.1, min(16.0, self.zoom * f))
        k = self.zoom / old
        self.pan = (self.pan + c - pos) * k - (c - pos)
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F:
            self.fit()
        elif e.key() == Qt.Key_G:
            self.guides = not self.guides
            self.update()
        else:
            super().keyPressEvent(e)

    def fit(self):
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def contextMenuEvent(self, e):
        if e.modifiers() & Qt.AltModifier:
            return
        m = QMenu(self)
        add = m.addMenu('Add emitter')
        from .panels import add_collider, add_emitter
        for shape, label in (('sphere', 'Sphere'), ('cylinder', 'Disc'), ('box', 'Box'), ('capsule', 'Line'), ('ring', 'Ring'), ('cone', 'Cone'),
                             ('mesh', 'Mesh…')):
            add.addAction(label, lambda s=shape: add_emitter(self.doc, s, self))
        m.addAction('Add collider box', lambda: self.doc.add_collider('box'))
        m.addAction('Add mesh collider…', lambda: add_collider(self.doc, 'mesh', self))
        m.addSeparator()
        a = m.addAction('Show guides', lambda: (setattr(self, 'guides', not self.guides), self.update()))
        a.setCheckable(True)
        a.setChecked(self.guides)
        m.addAction('Fit view  (F)', self.fit)
        m.exec(e.globalPos())
