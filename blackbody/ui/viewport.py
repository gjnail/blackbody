"""The viewer: the rendered frame, guides projected through the shot camera, and direct manipulation."""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QMenu, QWidget

from ..engine import camera as cam
from . import theme

MODE_NAMES = {'composite': 'Composite', 'fire': 'Effect over black', 'alpha': 'Alpha', 'emission': 'Emission',
              'heat': 'Heat (haze)', 'depth': 'Depth', 'temperature': 'Temperature'}


def _circle(center, radius_vec_a, radius_vec_b, n=48):
    t = np.linspace(0, 2 * np.pi, n + 1)
    return center[None, :] + np.cos(t)[:, None] * radius_vec_a[None, :] + np.sin(t)[:, None] * radius_vec_b[None, :]


_MESH_EDGES = {}


def _heightfield_edges(path, n=20):
    """A coarse grid over a heightfield image (the terrain's outline in the viewer)."""
    from ..engine.mesh import _read_grey
    img = _read_grey(path)
    ys = np.linspace(0, img.shape[0] - 1, n).astype(int)
    xs = np.linspace(0, img.shape[1] - 1, n).astype(int)
    hgt = 0.02 + img[np.ix_(ys, xs)]
    u = np.linspace(-0.5, 0.5, n)
    pts = np.stack([np.broadcast_to(u[None, :], (n, n)), hgt, np.broadcast_to(u[:, None], (n, n))], -1)
    segs = [np.stack([pts[:, :-1], pts[:, 1:]], 2).reshape(-1, 2, 3), np.stack([pts[:-1], pts[1:]], 2).reshape(-1, 2, 3)]
    return np.concatenate(segs)


def mesh_edges(path, limit=900, frame=None):
    """A sample of a mesh's edges (k, 2, 3), in mesh units, cached per source (and frame, for a mesh
    that deforms); None if it cannot be read."""
    from ..engine.mesh import IMAGE_EXTS, MeshLibrary, load_mesh, mesh_deforms, split_source
    try:
        key = (path, MeshLibrary._stamp(path), int(frame) if (frame is not None and mesh_deforms(path)) else None)
    except OSError:
        return None
    hit = _MESH_EDGES.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        from pathlib import Path
        if Path(split_source(path)[0]).suffix.lower() in IMAGE_EXTS:
            segs = _heightfield_edges(path)
            _MESH_EDGES[path] = (key, segs)
            return segs
        v, t = load_mesh(path, key[2])
    except Exception:
        return None
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    if len(e) > limit:
        e = e[np.linspace(0, len(e) - 1, limit).astype(int)]
    segs = v[e].astype(float)
    _MESH_EDGES[path] = (key, segs)
    return segs


_VOLUME_BOXES = {}


def volume_box(source, frame=None):
    """The part of a Volume emitter's field that holds anything (its own frame, metres), or None; read
    once per file (the engine's own read fills the same disk cache)."""
    from ..engine.mesh import MeshLibrary
    from ..io.volume import field_source, is_field, load_field
    src = source if is_field(source) else field_source(source)
    try:
        key = (src, MeshLibrary._stamp(src))
    except OSError:
        return None
    if key not in _VOLUME_BOXES:
        try:
            g, _ = load_field(src, frame, 96)
            _VOLUME_BOXES[key] = (g.mesh_min, g.mesh_max)
        except Exception:
            _VOLUME_BOXES[key] = None
    return _VOLUME_BOXES[key]


def fabric_lines(sc, i, frame):
    """The outline of fabric i where it is placed (its rest shape), with its pins marked."""
    f = sc.fabrics[i]
    g = lambda k: sc.get(('fabric', i, k), frame)
    p = np.asarray(g('position'), float)
    yaw = float(g('yaw'))
    if f['shape'] == 'mesh':
        segs = mesh_edges(sc.mesh_path(f['mesh'])) if f['mesh'] else None
        if segs is None:
            return shape_lines('sphere', p, (0.1, 0.1, 0.1))
        return [p + _rot_y(seg * np.asarray(f['scale'], float), yaw) for seg in segs]
    w, h = float(f['width']) / 2, float(f['height']) / 2
    if f['orientation'] == 'lying':
        c = np.array([[-w, 0, -h], [w, 0, -h], [w, 0, h], [-w, 0, h], [-w, 0, -h]])
        top = c[2:4]
    else:
        c = np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0], [-w, -h, 0]])
        top = c[2:4]
    out = [p + _rot_y(c, yaw)]
    pins = {'top': [top], 'side': [np.array([c[0], c[3]])], 'top_corners': [top[:1], top[1:]],
            'corners': [c[k:k + 1] for k in range(4)]}.get(f['pins'], [])
    for seg in pins:
        for q in seg:
            out.append(p + _rot_y(np.array([q + [0, 0.04, 0], q - [0, 0.04, 0]]), yaw))
        if len(seg) == 2:
            out.append(p + _rot_y(np.array(seg) + [0, 0.02, 0], yaw))
    return out


def _rot_y(pts, yaw_deg):
    """Rotate object-space offsets about y into fire-local space (matches the solver's rotation)."""
    a = np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    q = np.asarray(pts, float)
    return np.stack([c * q[..., 0] + s * q[..., 2], q[..., 1], -s * q[..., 0] + c * q[..., 2]], axis=-1)


def shape_lines(shape, pos, size, end=None, yaw=0.0, mesh=None, frame=None):
    """Polylines (lists of Nx3 arrays) outlining an emitter or collider shape in fire-local space."""
    p = np.asarray(pos, float)
    s = np.asarray(size, float)
    if shape == 'mesh':
        segs = mesh_edges(mesh, frame=frame) if mesh else None
        if segs is None:
            return shape_lines('sphere', pos, (0.1, 0.1, 0.1))
        return [p + _rot_y(seg * s, yaw) for seg in segs]
    if shape == 'volume':
        box = volume_box(mesh, frame) if mesh else None
        if box is None:
            return shape_lines('sphere', pos, (0.1, 0.1, 0.1))
        lo, hi = np.asarray(box[0], float) * s, np.asarray(box[1], float) * s
        c = 0.5 * (lo + hi)
        return [p + _rot_y(line, yaw) for line in shape_lines('box', c, 0.5 * (hi - lo))]
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
        self.show_stats = False
        self.mode_labels = {}   # the view's name for this kind of simulation (main window)
        self.drop_hint = None   # text shown while a file is dragged over the window
        self.drop_point = None  # where a dragged building block would land
        self._readout = None     # (text, where) beside the cursor while the gizmo is dragged
        self.path_target = None  # (kind, i) while a path is drawn for it
        self._path_pts = []
        self._path_cursor = None
        self.pathbar = None
        self._rmb_moved = 0.0
        self.roto_mode = False   # drawing and editing roto shapes over the footage
        self.roto_sel = None     # the shape being edited
        self._roto_draft = None  # points of a shape being drawn ([x, y] across and down the frame, 0..1)
        self._roto_live = None   # (shape, points) while a point or the shape is dragged
        self._roto_cursor = None
        self.rotobar = None
        self.sim_msg = ''
        # what holds up the next frame (UI.set_busy), shown as a card over the viewer: (text, fraction) or None
        self.busy = None
        self.loading = None     # name of a scene that has no frame on screen yet
        self._busy_since = 0.0
        self._busy_anim = QTimer(self)
        self._busy_anim.setInterval(50)
        self._busy_anim.timeout.connect(self.update)
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
        doc.viewChanged.connect(self.update)

    # -- data in ---------------------------------------------------------------------------------------

    def set_frame_image(self, img: QImage, frame, stats):
        self.image = img
        self.stats = stats
        self.loading = None
        self.set_busy(None)
        self.update()

    def begin_load(self, name):
        """A different scene replaced the last: drop its picture and show what the engine is doing
        until the new scene's first frame is in."""
        self.image = None
        self.stats = {}
        self.loading = name
        self.busy = None
        self.set_busy('Setting up the simulation', -1.0)

    def set_busy(self, text, frac=-1.0):
        """What holds up the next frame (None when nothing). A short wait shows nothing; one that
        lasts, or any wait for a scene just loaded, shows a card with its progress."""
        if not text:
            self.busy = None
            self._busy_anim.stop()
            self.update()
            return
        if self.busy is None or self.busy[1] == -2.0:
            self._busy_since = time.monotonic()
        self.busy = (text, float(frac))
        if frac == -2.0:   # an error: nothing moves on the card
            self._busy_anim.stop()
        elif not self._busy_anim.isActive():
            self._busy_anim.start()
        self.update()

    def set_drop_hint(self, text, marker=False):
        if text != self.drop_hint:
            self.drop_hint = text
            if not marker:
                self.drop_point = None
            self.update()

    # -- paths ------------------------------------------------------------------------------------------------

    def start_path(self, kind, i):
        sc = self.doc.scene
        items = {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}[kind]
        if i >= len(items):
            return
        self.doc.select((kind, i), force=True)
        self.path_target = (kind, i)
        self._path_pts = [np.asarray(sc.get((kind, i, 'position'), self.doc.frame), float)]
        if self.pathbar is None:
            from .pathbar import PathBar
            self.pathbar = PathBar(self)
        self.pathbar.sync(items[i]['name'], 1, self.doc.frame)
        self.pathbar.show()
        self.place_rotobar()
        self.setCursor(Qt.CrossCursor)
        self.setFocus()
        self.update()

    def path_cancel(self):
        self.path_target = None
        self._path_pts = []
        if self.pathbar is not None:
            self.pathbar.hide()
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def path_finish(self):
        tgt, pts = self.path_target, list(self._path_pts)
        secs = self.pathbar.secs.value() if self.pathbar is not None else 3.0
        self.path_cancel()
        if tgt is None or len(pts) < 2:
            return
        from .actions import path_keys
        kind, i = tgt
        sc = self.doc.scene
        items = {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}[kind]
        name = items[i]['name']
        f0 = self.doc.frame
        end_off = None
        if kind == 'emitter' and items[i].get('shape') == 'capsule':
            end_off = np.asarray(sc.get(('emitter', i, 'end'), f0), float) - pts[0]

        def fn(s):
            s.links = [l for l in getattr(s, 'links', []) if list(l['child']) != [kind, name]]
            curve = path_keys(s, kind, i, pts, f0, secs)
            d = {'emitter': s.emitters, 'collider': s.colliders, 'light': s.lights, 'fabric': s.fabrics}[kind][i]
            d['position'] = curve
            if end_off is not None:
                from ..scene.anim import Curve
                d['end'] = Curve([[f, tuple(float(a + b) for a, b in zip(v, end_off)), it] for f, v, it in curve.keys])
        self.doc.edit(f'Move {name} along a path', fn, structure=True)
        win = self.window()
        if hasattr(win, 'msg'):
            win.msg.setText(f'{name} goes along the path from frame {f0} for {secs:g} s: its keys are on the timeline '
                            '(Animation shows them).')
        self.doc.set_playing(True)

    def _path_ground(self, pos):
        y = float(self._path_pts[0][1]) if self._path_pts else 0.0
        gp = self._ground_point(pos, y)
        return None if gp is None else np.asarray(gp, float)

    def _paint_path(self, p):
        if self.path_target is None:
            return
        cs, fire, _ = self.camstate()
        pts = list(self._path_pts)
        if self._path_cursor is not None:
            pts = pts + [self._path_cursor]
        if len(pts) >= 2:
            self._lines(p, cs, fire, [np.stack(pts)], QPen(QColor(theme.ACCENT), 2.0))
        px, ok = self._project_local(cs, fire, self._path_pts or [np.zeros(3)])
        p.setPen(QPen(QColor(theme.ACCENT), 1.4))
        for k, (q, o) in enumerate(zip(px, ok)):
            if o:
                w = self.to_widget(q)
                p.setBrush(QColor(theme.ACCENT) if k == 0 else QColor(20, 20, 24))
                p.drawEllipse(w, 5, 5)

    # -- roto -------------------------------------------------------------------------------------------------

    def set_roto_mode(self, on):
        self.roto_mode = bool(on)
        self._roto_draft = None
        self._roto_live = None
        if on and self.rotobar is None:
            from .rotobar import RotoBar
            self.rotobar = RotoBar(self)
        if self.rotobar is not None:
            self.rotobar.setVisible(on)
            if on:
                self.rotobar.sync()
        if on and self.roto_sel is None and self.doc.scene.roto:
            self.roto_sel = 0
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def place_rotobar(self):
        if self.pathbar is not None and self.pathbar.isVisible():
            self.pathbar.setFixedWidth(min(560, self.width() - 24))
            self.pathbar.adjustSize()
            self.pathbar.move((self.width() - self.pathbar.width()) // 2, 10)
        if self.rotobar is not None:
            self.rotobar.setFixedWidth(min(max(420, self.rotobar.sizeHint().width()), self.width() - 24))
            self.rotobar.adjustSize()
            self.rotobar.move((self.width() - self.rotobar.width()) // 2, 10)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.place_rotobar()

    def _norm(self, pos):
        W, H = self.out_size()
        fx, fy = self.to_frame(pos)
        return [min(max(fx / W, -0.5), 1.5), min(max(fy / H, -0.5), 1.5)]

    def _scr(self, q):
        W, H = self.out_size()
        return self.to_widget((q[0] * W, q[1] * H))

    def _roto_pts(self, i):
        if self._roto_live is not None and self._roto_live[0] == i:
            return self._roto_live[1]
        from ..scene.roto import points_at
        return points_at(self.doc.scene.roto[i], self.doc.frame)

    def _roto_hit_point(self, pos):
        i = self.roto_sel
        if i is None or i >= len(self.doc.scene.roto):
            return None
        for j, q in enumerate(self._roto_pts(i) or []):
            if (self._scr(q) - pos).manhattanLength() < 10:
                return j
        return None

    def _roto_hit_shape(self, pos):
        for i in reversed(range(len(self.doc.scene.roto))):
            pts = self._roto_pts(i)
            if pts and QPolygonF([self._scr(q) for q in pts]).containsPoint(pos, Qt.OddEvenFill):
                return i
        return None

    def _roto_hit_edge(self, pos):
        i = self.roto_sel
        if i is None or i >= len(self.doc.scene.roto):
            return None
        pts = [self._scr(q) for q in self._roto_pts(i) or []]
        for j in range(len(pts)):
            a, b = pts[j], pts[(j + 1) % len(pts)]
            ab = b - a
            L2 = ab.x() ** 2 + ab.y() ** 2
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((pos - a).x() * ab.x() + (pos - a).y() * ab.y()) / L2))
            q = a + ab * t
            if (q - pos).manhattanLength() < 8:
                return j
        return None

    def _roto_finish(self):
        pts, self._roto_draft = self._roto_draft, None
        if not pts or len(pts) < 3:
            self.update()
            return
        from ..scene.roto import new_shape, set_points
        n = len(self.doc.scene.roto)
        f = self.doc.frame

        def fn(shapes):
            sh = new_shape(f'Roto {n + 1}')
            set_points(sh, f, pts)
            shapes.append(sh)
        self.doc.roto_edit('Add roto shape', fn)
        self.roto_sel = n
        if self.rotobar is not None:
            self.rotobar.sync()
        self.update()

    def roto_delete(self):
        i = self.roto_sel
        if i is None or i >= len(self.doc.scene.roto):
            return
        self.doc.roto_edit('Delete roto shape', lambda shapes: shapes.pop(i))
        self.roto_sel = (i - 1) if i > 0 else (0 if self.doc.scene.roto else None)
        if self.rotobar is not None:
            self.rotobar.sync()
        self.update()

    def _roto_press(self, e):
        pos = e.position()
        if e.button() == Qt.RightButton:
            j = self._roto_hit_point(pos)
            if j is not None and len(self._roto_pts(self.roto_sel)) > 3:
                from ..scene.roto import delete_point
                i = self.roto_sel
                self.doc.roto_edit('Remove roto point', lambda shapes: delete_point(shapes[i], j))
            elif self._roto_draft:
                self._roto_draft.pop()
                if self.rotobar is not None:
                    self.rotobar.sync()
                self.update()
            return
        if e.button() != Qt.LeftButton:
            return
        if self._roto_draft is not None:
            if len(self._roto_draft) >= 3 and (self._scr(self._roto_draft[0]) - pos).manhattanLength() < 12:
                self._roto_finish()
            else:
                self._roto_draft.append(self._norm(pos))
                if self.rotobar is not None:
                    self.rotobar.sync()
            self.update()
            return
        j = self._roto_hit_point(pos)
        if j is not None:
            self._drag = {'kind': 'roto_pt', 'j': j, 'pts': [list(q) for q in self._roto_pts(self.roto_sel)]}
            return
        if e.modifiers() & Qt.ControlModifier:
            j = self._roto_hit_edge(pos)
            if j is not None:
                from ..scene.roto import insert_point
                i, f, xy = self.roto_sel, self.doc.frame, self._norm(pos)
                self.doc.roto_edit('Add roto point', lambda shapes: insert_point(shapes[i], j, f, xy))
                self._drag = {'kind': 'roto_pt', 'j': j + 1, 'pts': [list(q) for q in self._roto_pts(i)]}
                return
        i = self._roto_hit_shape(pos)
        if i is not None:
            self.roto_sel = i
            self._drag = {'kind': 'roto_move', 'start': self._norm(pos), 'pts': [list(q) for q in self._roto_pts(i)]}
            if self.rotobar is not None:
                self.rotobar.sync()
            self.update()
            return
        self.roto_sel = None
        self._roto_draft = [self._norm(pos)]
        if self.rotobar is not None:
            self.rotobar.sync()
        self.update()

    def _roto_drag(self, d, pos):
        q = self._norm(pos)
        if d['kind'] == 'roto_pt':
            pts = [list(x) for x in d['pts']]
            pts[d['j']] = q
        else:
            dx, dy = q[0] - d['start'][0], q[1] - d['start'][1]
            pts = [[x + dx, y + dy] for x, y in d['pts']]
        self._roto_live = (self.roto_sel, pts)
        self.update()

    def _roto_release(self):
        live, self._roto_live = self._roto_live, None
        if live is None:
            return
        from ..scene.roto import set_points
        i, pts = live
        f = self.doc.frame
        self.doc.roto_edit('Move roto shape', lambda shapes: set_points(shapes[i], f, pts))
        if self.rotobar is not None:
            self.rotobar.sync()

    def _paint_roto(self, p):
        shapes = self.doc.scene.roto
        if not shapes and self._roto_draft is None:
            return
        from ..scene.roto import has_key
        p.save()
        for i, sh in enumerate(shapes):
            pts = self._roto_pts(i)
            if not pts:
                continue
            poly = QPolygonF([self._scr(q) for q in pts])
            sel = i == self.roto_sel and self.roto_mode
            on = sh.get('enabled', True)
            if not self.roto_mode:
                p.setPen(QPen(QColor(255, 255, 255, 70 if on else 30), 1.0, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawPolygon(poly)
                continue
            fill = QColor(theme.ACCENT)
            fill.setAlpha((60 if sel else 30) if on else 10)
            p.setBrush(fill)
            p.setPen(QPen(QColor(theme.ACCENT if sel else '#ffffff'), 1.6 if sel else 1.0, Qt.SolidLine if on else Qt.DashLine))
            p.drawPolygon(poly)
            if sel:
                keyed = has_key(sh, self.doc.frame)
                for q in poly:
                    p.setPen(QPen(QColor(theme.ACCENT), 1.2))
                    p.setBrush(QColor(theme.ACCENT) if keyed else QColor(20, 20, 24))
                    p.drawRect(QRectF(q.x() - 3.5, q.y() - 3.5, 7, 7))
            if sh.get('invert'):
                c = poly.boundingRect().center()
                p.setPen(QColor(255, 255, 255, 160))
                p.drawText(QPointF(c.x() - 18, c.y()), 'inverted')
        if self._roto_draft:
            pts = [self._scr(q) for q in self._roto_draft]
            p.setPen(QPen(QColor(theme.ACCENT), 1.6))
            p.setBrush(Qt.NoBrush)
            path = QPainterPath()
            path.moveTo(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            if self._roto_cursor is not None:
                path.lineTo(self._roto_cursor)
            p.drawPath(path)
            for k, q in enumerate(pts):
                p.setBrush(QColor(theme.ACCENT) if k == 0 else QColor(20, 20, 24))
                r = 6.0 if k == 0 else 3.5
                p.drawEllipse(q, r, r)
        p.restore()

    def set_drop_point(self, gp):
        """Where a building block dragged over the viewer would land (fire-local x, z on the ground), or None."""
        if gp != self.drop_point:
            self.drop_point = gp
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
        spec, fire = self.doc.camera(self.doc.frame)
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
        if self.path_target is not None:
            p.save()
            self._paint_path(p)
            p.restore()
        if self.roto_mode or (self.guides and self.doc.scene.roto):
            p.save()
            p.setClipRect(r.adjusted(-1, -1, 1, 1))
            self._paint_roto(p)
            p.restore()
        self._paint_busy(p, r)
        self._paint_hud(p, r)
        self._paint_hints(p, r)

    def _paint_busy(self, p, r):
        if self.busy is None:
            return
        text, frac = self.busy
        waited = time.monotonic() - self._busy_since
        error = frac == -2.0
        if self.loading is None and not error and waited < 0.6:
            return
        head =('Could not show ' + (self.loading or 'this frame')) if error else (
            f'Loading {self.loading}' if self.loading else 'Working')
        note = None
        if error:
            note = 'Try another preset, or Simulation › Restart simulation. The details are in the status bar too.'
        elif text.startswith('Compiling GPU shaders'):
            note = ('Your graphics driver compiles each shader the first time it sees it (after an install or an update). '
                    'The big ones can take several minutes; after that they start at once.')
        elif text.startswith('Pre-roll'):
            note = 'So the effect is already going at the first frame. It is cached: you only wait once.'
        cw = min(460.0, max(260.0, r.width() - 40))
        f_head = QFont(p.font())
        f_head.setPointSizeF(11)
        f_head.setBold(True)
        f_body = QFont(p.font())
        f_body.setPointSizeF(9)
        flags = Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop
        p.setFont(f_body)
        body_r = p.boundingRect(QRectF(0, 0, cw - 32, 400), flags, text)
        note_r = p.boundingRect(QRectF(0, 0, cw - 32, 400), flags, note) if note else QRectF()
        ch = 14 + 26 + body_r.height() + 10 + (0 if error else 14) + (note_r.height() + 4 if note else 0) + 10
        card = QRectF(r.center().x() - cw / 2, r.center().y() - ch / 2, cw, ch)
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor(theme.BAD if error else theme.LINE), 1))
        p.setBrush(QColor(24, 24, 27, 238))
        p.drawRoundedRect(card, 6, 6)
        x, y = card.x() + 16, card.y() + 14
        p.setFont(f_head)
        p.setPen(QColor(theme.BAD if error else theme.TEXT))
        p.drawText(QRectF(x, y, cw - 32 - 50, 22), Qt.AlignLeft | Qt.AlignVCenter, head)
        if not error:
            p.setFont(f_body)
            p.setPen(QColor(theme.MUTED))
            m, sec = divmod(int(waited), 60)
            p.drawText(QRectF(x, y, cw - 32, 22), Qt.AlignRight | Qt.AlignVCenter, f'{m}:{sec:02d}')
        y += 26
        p.setFont(f_body)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(x, y, cw - 32, body_r.height()), flags, text)
        y += body_r.height() + 10
        if not error:
            bar = QRectF(x, y, cw - 32, 4)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.FIELD))
            p.drawRoundedRect(bar, 2, 2)
            p.setBrush(QColor(theme.ACCENT))
            if frac >= 0:
                p.drawRoundedRect(QRectF(bar.x(), bar.y(), bar.width() * min(max(frac, 0.0), 1.0), 4), 2, 2)
            else:   # unknown length: a segment sweeping across
                seg = bar.width() * 0.25
                t = (waited * 0.6) % 1.0
                p.drawRoundedRect(QRectF(bar.x() + (bar.width() + seg) * t - seg, bar.y(), seg, 4).intersected(bar), 2, 2)
            y += 14
        if note:
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(x, y, cw - 32, note_r.height()), flags, note)
        p.restore()

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
                                                 sc.item_source(em) if em['shape'] == 'volume' else sc.mesh_path(em['mesh']),
                                                 self.doc.frame - em.get('mesh_offset', 0.0)), pen)
        for i, l in enumerate(sc.lights):
            if not l['enabled']:
                continue
            g = lambda k, i=i: sc.get(('light', i, k), self.doc.frame)
            is_sel = sel == ('light', i)
            col = QColor(255, 220, 120, 240 if is_sel else 150)
            pen = QPen(col, 1.6 if is_sel else 1.0)
            pos = np.asarray(g('position'), float)
            r = max(float(l['radius']), 0.05)
            lines = shape_lines('sphere', pos, (r, r, r))
            if l['kind'] in ('spot', 'area'):
                aim = np.asarray(g('direction'), float)
                aim = aim / max(float(np.linalg.norm(aim)), 1e-9)
                lines.append(np.stack([pos, pos + aim * max(0.5, 5 * r)]))
            self._lines(p, cs, fire, lines, pen)
        for i, f in enumerate(sc.fabrics):
            if not f['enabled']:
                continue
            is_sel = sel == ('fabric', i)
            pen = QPen(QColor(150, 210, 255, 240 if is_sel else 130), 1.6 if is_sel else 1.0, Qt.DashLine)
            self._lines(p, cs, fire, fabric_lines(sc, i, self.doc.frame), pen)
        floats = self.stats.get('floats') if self.stats.get('frame') == self.doc.frame else None
        for i, c in enumerate(sc.colliders):
            if not c['enabled']:
                continue
            ov = (floats or {}).get(i) if c.get('floating') else None
            g = (lambda k, ov=ov, i=i: (tuple(ov['pos']) if k == 'position' else math.degrees(ov['rot_y']))
                 if ov is not None and k in ('position', 'yaw') else sc.get(('collider', i, k), self.doc.frame))
            is_sel = sel == ('collider', i)
            col = QColor(255, 140, 80) if c.get('burnable') else QColor(120, 190, 255)
            col.setAlpha(230 if is_sel else 120)
            pen = QPen(col, 1.6 if is_sel else 1.0)
            self._lines(p, cs, fire, shape_lines(c['shape'], g('position'), g('size'), None, g('yaw'), sc.mesh_path(c['mesh']),
                                                 self.doc.frame - c.get('mesh_offset', 0.0)), pen)
            op = np.asarray(g('opening'), float)
            if (op > 0).all():  # the doorway or window cut through it
                at = np.asarray(g('position'), float) + _rot_y(np.asarray(g('opening_at'), float), g('yaw'))
                self._lines(p, cs, fire, shape_lines('box', at, op, None, g('yaw')), QPen(col, 1.0, Qt.DotLine))
        # the transform gizmo of the selected object
        gz = self._gizmo()
        if gz is not None:
            self._paint_gizmo(p, gz)
        if self.doc.work_view is not None:
            self._paint_shot_camera(p)
        # tracked path of the fire base
        if sc.track and sc.track.get('points') and self.doc.work_view is None:
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

    def _gizmo(self):
        sel = self.doc.selection
        if not sel or sel[0] not in ('emitter', 'collider', 'light', 'fabric') or self.roto_mode:
            return None
        sc = self.doc.scene
        items = {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}[sel[0]]
        if sel[1] >= len(items) or not items[sel[1]]['enabled']:
            return None
        from .gizmo import Gizmo
        try:
            return Gizmo(self, sel[0], sel[1])
        except Exception:
            return None

    def _paint_gizmo(self, p, gz):
        from .gizmo import AXIS_COLOURS, AXIS_NAMES, arrow_head
        hs = gz.handles()
        cs, fire = gz.cs, gz.fire
        drag = (self._drag or {}).get('key') if (self._drag or {}).get('kind') == 'gz' else None
        hover = self._hover[3:] if isinstance(self._hover, str) and self._hover.startswith('gz:') else None
        if 'ring' in hs:
            hot = 'rot' in (drag, hover)
            self._lines(p, cs, fire, [hs['ring']], QPen(QColor(theme.ACCENT), 1.6 if hot else 1.0, Qt.DashLine))
        p.save()
        for name, col in zip(AXIS_NAMES, AXIS_COLOURS):
            v = hs.get('move_' + name)
            if v is None:
                continue
            a, b = v
            hot = ('move_' + name) in (drag, hover)
            c = QColor('#ffffff' if hot else col)
            p.setPen(QPen(c, 2.6 if hot else 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(a, b)
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawPolygon(QPolygonF(arrow_head(a, b, 9.0)))
        for name, col in zip(AXIS_NAMES, AXIS_COLOURS):
            q = hs.get('scale_' + name)
            if q is None:
                continue
            hot = ('scale_' + name) in (drag, hover)
            p.setPen(QPen(QColor('#ffffff' if hot else col), 1.4))
            p.setBrush(QColor(col) if hot else QColor(16, 16, 20, 220))
            p.drawRect(QRectF(q.x() - 4.5, q.y() - 4.5, 9, 9))
        q = hs.get('rot')
        if q is not None:
            hot = 'rot' in (drag, hover)
            p.setPen(QPen(QColor(theme.ACCENT), 1.4))
            p.setBrush(QColor(theme.ACCENT) if hot else QColor(16, 16, 20, 220))
            p.drawEllipse(q, 5.5, 5.5)
        q = hs.get('centre')
        if q is not None:
            hot = 'centre' in (drag, hover)
            p.setPen(QPen(QColor('#ffffff'), 1.4))
            p.setBrush(QColor(255, 255, 255, 200) if hot else Qt.NoBrush)
            p.drawEllipse(q, 6, 6)
        if self._readout is not None:
            text, at = self._readout
            f = QFont(theme.font(9.0, QFont.DemiBold))
            p.setFont(f)
            w = p.fontMetrics().horizontalAdvance(text) + 16
            box = QRectF(at.x() + 14, at.y() + 12, w, 22)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(12, 12, 14, 220))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor(theme.TEXT))
            p.drawText(box, Qt.AlignCenter, text)
        p.restore()

    def _ray_local(self, pos):
        """The pixel ray under `pos` in the effect's own coordinates: (origin, unit direction)."""
        cs, fire, _ = self.camstate()
        W, H = self.out_size()
        fx, fy = self.to_frame(pos)
        o, d = cam.pixel_ray(cs, fx, fy, W, H)
        Li = np.linalg.inv(fire.local_to_world())
        o_l = (Li @ np.append(o, 1.0))[:3]
        d_l = Li[:3, :3] @ np.asarray(d, float)
        n = float(np.linalg.norm(d_l))
        return o_l, d_l / max(n, 1e-12)

    def _axis_param(self, pos, origin, axis):
        """Where the mouse ray passes closest to the line origin + t * axis: t (None if they are nearly parallel)."""
        o, d = self._ray_local(pos)
        w0 = np.asarray(origin, float) - o
        axis = np.asarray(axis, float)
        b = float(axis @ d)
        den = 1.0 - b * b
        if abs(den) < 1e-5:
            return None
        return (b * float(d @ w0) - float(axis @ w0)) / den

    def _gz_press(self, key, pos):
        from .gizmo import AXES, AXIS_NAMES
        gz = self._gizmo()
        if gz is None:
            return
        d = {'kind': 'gz', 'key': key, 'what': gz.kind, 'i': gz.i, 'pos0': gz.pos.copy(), 'yaw0': gz.yaw, 'L': gz.L,
             'snap': gz.snap()}
        it = gz.item
        g = gz.g
        if gz.capsule:
            d['end0'] = np.asarray(g('end'), float)
        if key.startswith('move_'):
            d['axis'] = AXES[AXIS_NAMES.index(key[-1])]
            d['t0'] = self._axis_param(pos, gz.pos, d['axis'])
        elif key.startswith('scale_'):
            name = key[-1]
            u = dict(gz.scale_axes())[name]
            d['u'], d['name'] = u, name
            d['t0'] = self._axis_param(pos, gz.pos, u)
            if gz.kind == 'fabric':
                d['w0'], d['h0'] = float(g('width')), float(g('height'))
                d['scale0'] = np.asarray(it.get('scale', (1.0, 1.0, 1.0)), float)
            elif gz.kind in ('emitter', 'collider'):
                d['size0'] = np.asarray(g('size'), float)
        elif key == 'rot':
            gp = self._ground_point(pos, gz.pos[1])
            d['a0'] = None if gp is None else math.atan2(gp[2] - gz.pos[2], gp[0] - gz.pos[0])
        elif key == 'centre':
            self._drag = {'kind': 'move', 'what': gz.kind, 'i': gz.i, 'y': float(gz.pos[1]),
                          'start_world': self._ground_point(pos, gz.pos[1]), 'p0': gz.pos.copy(),
                          'e0': d.get('end0', gz.pos.copy()), 'vertical': False, 'sy': pos.y(), 'snap': d['snap']}
            return
        if d.get('t0', 0.0) is None:
            return
        self._drag = d

    def _gz_move(self, d, pos, mods):
        from .gizmo import AXIS_NAMES
        what, i = d['what'], d['i']
        snap = d['snap'] if mods & Qt.ControlModifier else None
        key = d['key']
        if key.startswith('move_'):
            t = self._axis_param(pos, d['pos0'], d['axis'])
            if t is None:
                return
            k = AXIS_NAMES.index(key[-1])
            new = d['pos0'] + d['axis'] * (t - d['t0'])
            if snap:
                new[k] = round(new[k] / snap) * snap
            self.doc.set((what, i, 'position'), tuple(float(x) for x in new))
            if 'end0' in d:
                self.doc.set(('emitter', i, 'end'), tuple(float(x) for x in d['end0'] + (new - d['pos0'])))
            self._readout = (f'{key[-1]}  {new[k]:.3g} m', pos)
        elif key.startswith('scale_'):
            t = self._axis_param(pos, d['pos0'], d['u'])
            if t is None:
                return
            grow = t - d['t0']
            name = d['name']
            k = AXIS_NAMES.index(name)
            it = {'emitter': self.doc.scene.emitters, 'collider': self.doc.scene.colliders,
                  'fabric': self.doc.scene.fabrics}.get(what, [None] * (i + 1))[i]
            if what == 'fabric' and it.get('shape') == 'mesh':
                s0 = d['scale0']
                f = max(0.02, 1.0 + grow / max(d['L'], 1e-6))
                sc_new = s0.copy()
                sc_new[k] = s0[k] * f
                self.doc.set(('fabric', i, 'scale'), tuple(float(x) for x in sc_new))
                self._readout = (f'scale {name}  {sc_new[k]:.2f}\u00d7', pos)
            elif what == 'fabric':
                full0 = d['w0'] if name == 'x' else d['h0']
                full = max(0.02, full0 + 2.0 * grow)
                if snap:
                    full = max(snap, round(full / snap) * snap)
                self.doc.set(('fabric', i, 'width' if name == 'x' else 'height'), float(full))
                self._readout = (f'{"width" if name == "x" else "height"}  {full:.3g} m', pos)
            else:
                s0 = d['size0']
                ext = max(1e-3, s0[k] + grow)
                if snap:
                    ext = max(snap / 2, round(ext * 2 / snap) * snap / 2)
                new = s0.copy()
                shape = it.get('shape') if it else ''
                if shape == 'capsule' or (shape == 'sphere' and what == 'collider'):
                    new[:] = ext
                elif shape in ('cylinder', 'cone', 'ring') and name == 'x':
                    new[0] = new[2] = ext
                else:
                    new[k] = ext
                self.doc.set((what, i, 'size'), tuple(float(x) for x in new))
                self._readout = (f'size {name}  {2 * ext:.3g} m', pos)
        elif key == 'rot':
            gp = self._ground_point(pos, d['pos0'][1])
            if gp is None or d.get('a0') is None:
                return
            a = math.atan2(gp[2] - d['pos0'][2], gp[0] - d['pos0'][0])
            yaw = d['yaw0'] - math.degrees(a - d['a0'])
            yaw = (yaw + 180.0) % 360.0 - 180.0
            if mods & (Qt.ShiftModifier | Qt.ControlModifier):
                yaw = round(yaw / 15.0) * 15.0
            self.doc.set((what, i, 'yaw'), yaw)
            self._readout = (f'turn  {yaw:.0f}\u00b0', pos)

    def _paint_shot_camera(self, p):
        """In the work view: where the shot's camera is and what it sees, as a frustum."""
        sc = self.doc.scene
        spec, fire = sc.camera(self.doc.frame)
        W, H = self.out_size()
        shot = cam.compute(spec, W / H, fire)
        cs, _, _ = self.camstate()
        sy = max(sc.domain_size())
        eye = np.asarray(shot.eye, float)
        corners = []
        for fx, fy in ((0, 0), (W, 0), (W, H), (0, H)):
            o, d = cam.pixel_ray(shot, fx, fy, W, H)
            corners.append(eye + np.asarray(d, float) * sy * 0.35)
        lines = [np.stack([eye, c]) for c in corners] + [np.stack(corners + corners[:1])]
        pen = QPen(QColor(255, 214, 102, 210), 1.3)
        path = QPainterPath()
        for ln in lines:
            px, ok = cam.project(cs, ln, W, H)
            if not ok.all():
                continue
            path.moveTo(self.to_widget(px[0]))
            for q in px[1:]:
                path.lineTo(self.to_widget(q))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        px, ok = cam.project(cs, [eye], W, H)
        if ok[0]:
            q = self.to_widget(px[0])
            p.setPen(QColor(255, 214, 102, 230))
            p.drawText(QPointF(q.x() + 8, q.y() - 6), 'shot camera')

    def _paint_hud(self, p, r):
        st = self.stats
        f = QFont(theme.mono_font(8.5))
        p.setFont(f)
        lines = [f'{self.mode_labels.get(self.mode) or MODE_NAMES.get(self.mode, self.mode)}  ·  frame {self.doc.frame}']
        if self.doc.work_view is not None:
            lines[0] = 'BUILD  ·  ' + lines[0]
            lines.append('drag empty space: look around · right/middle-drag: pan · wheel: closer · F: frame all · '
                         'Ctrl while dragging: snap · right-click a thing: what it can do')
        if st and self.show_stats:
            dims = st.get('dims') or (0, 0, 0)
            lines.append(f'{dims[0]}×{dims[1]}×{dims[2]} voxels · {st.get("cell_mm", 0):.1f} mm · {st.get("substeps", 0)} substeps')
            if st.get('kind') == 'liquid':
                ww = st.get('whitewater', 0)
                lines.append(f'{st.get("particles", 0) / 1e6:.2f} M particles · '
                             + (f'{ww / 1e3:.0f} k whitewater' if ww >= 10000 else f'{ww} whitewater'))
            lines.append(f'sim {st.get("sim_ms", 0):.0f} ms · render {st.get("render_ms", 0):.0f} ms · max {st.get("max_speed", 0):.1f} m/s')
            if st.get('refined'):
                lines.append('refined (4 samples)')
        if st.get('particle_limit'):
            lines.append('particle limit reached: sources are held back (Liquid › Particle limit, advanced)')
        fm = p.fontMetrics()
        w = max(fm.horizontalAdvance(ln) for ln in lines) + 20
        h = 15 * len(lines) + 10
        box = QRectF(10, 10, w, h)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(12, 12, 14, 170))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QColor(230, 230, 232, 225))
        y = box.y() + 17
        for ln in lines:
            p.drawText(QPointF(box.x() + 10, y), ln)
            y += 15
        W, H = self.out_size()
        p.setPen(QColor(255, 255, 255, 80))
        p.drawText(QPointF(r.right() - 90, r.bottom() + 14), f'{W}×{H}')

    def _paint_hints(self, p, r):
        """While a file is dragged over the window: where it goes. With no footage: how to bring some in."""
        if self.drop_hint:
            p.save()
            p.setPen(QPen(QColor(theme.ACCENT), 2, Qt.DashLine))
            p.setBrush(QColor(255, 122, 47, 28))
            p.drawRoundedRect(QRectF(self.rect()).adjusted(8, 8, -8, -8), 12, 12)
            if self.drop_point is not None:
                try:
                    cs, fire, _ = self.camstate()
                    c = np.array([self.drop_point[0], 0.0, self.drop_point[1]])
                    rad = max(self.doc.scene.domain_size()[0] * 0.08, 0.02)
                    ring = _circle(c, np.array([rad, 0, 0]), np.array([0, 0, rad]), 48)
                    self._lines(p, cs, fire, [ring], QPen(QColor(theme.ACCENT), 2.0))
                    self._lines(p, cs, fire, [np.stack([c - [rad * 1.6, 0, 0], c + [rad * 1.6, 0, 0]]),
                                              np.stack([c - [0, 0, rad * 1.6], c + [0, 0, rad * 1.6]])], QPen(QColor(theme.ACCENT), 1.0))
                    px, ok = self._project_local(cs, fire, [c])
                    if ok[0]:
                        q = self.to_widget(px[0])
                        self._pill(p, QPointF(q.x(), q.y() - 34), self.drop_hint, strong=True)
                        p.restore()
                        return
                except Exception:
                    pass
            self._pill(p, QPointF(self.width() / 2, self.height() / 2), self.drop_hint, strong=True)
            p.restore()
            return
        sc = self.doc.scene
        if (self.busy is None and self.image is not None and sc.kind != 'cloud' and not sc.emitters and not sc.colliders
                and self.doc.work_view is None
                and not sc.fabrics and not (sc.kind != 'fire' and (sc.data['liquid']['water_level'] > 0 or sc.data['liquid']['rain'] > 0))):
            self._pill(p, QPointF(r.center().x(), r.center().y()), 'An empty scene · add fire, water or objects from Create, on the left',
                       strong=True)
        if self.busy is None and self.image is not None and not self.doc.scene.footage and not self.doc.footage_info \
                and self.doc.work_view is None:
            self._pill(p, QPointF(r.center().x(), r.top() + 22), 'No footage yet · drop a clip here, or Import footage (Ctrl+I)')

    def _pill(self, p, c, text, strong=False):
        f = QFont(theme.font(10.5 if strong else 9.0, QFont.DemiBold if strong else QFont.Normal))
        p.setFont(f)
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 28
        h = fm.height() + (16 if strong else 10)
        box = QRectF(c.x() - w / 2, c.y() - h / 2, w, h)
        p.setPen(QPen(QColor(theme.ACCENT if strong else theme.LINE_HI), 1))
        p.setBrush(QColor(20, 20, 23, 235 if strong else 200))
        p.drawRoundedRect(box, h / 2, h / 2)
        p.setPen(QColor(theme.TEXT if strong else theme.MUTED))
        p.drawText(box, Qt.AlignCenter, text)

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
        gz = self._gizmo()
        if gz is not None:
            k = gz.hit(pos)
            if k is not None:
                return 'gz:' + k
        sc = self.doc.scene
        cs, fire, _ = self.camstate()
        best, bd = None, 12.0
        for kind, items in (('emitter', sc.emitters), ('collider', sc.colliders), ('light', sc.lights), ('fabric', sc.fabrics)):
            for i, it in enumerate(items):
                if not it['enabled']:
                    continue
                px, ok = self._project_local(cs, fire, [sc.get((kind, i, 'position'), self.doc.frame)])
                if ok[0]:
                    d = (self.to_widget(px[0]) - pos).manhattanLength()
                    if d < bd:
                        best, bd = (kind, i), d
        return best

    # -- interaction --------------------------------------------------------------------------------------------

    def mousePressEvent(self, e):
        pos = e.position()
        mods = e.modifiers()
        if self.roto_mode and not (mods & Qt.AltModifier) and e.button() in (Qt.LeftButton, Qt.RightButton):
            self._roto_press(e)
            return
        if self.path_target is not None and e.button() == Qt.LeftButton:
            gp = self._path_ground(pos)
            if gp is not None:
                self._path_pts.append(gp)
                sc = self.doc.scene
                kind, i = self.path_target
                items = {'emitter': sc.emitters, 'collider': sc.colliders, 'light': sc.lights, 'fabric': sc.fabrics}[kind]
                self.pathbar.sync(items[i]['name'], len(self._path_pts), self.doc.frame)
                self.update()
            return
        self._rmb_moved = 0.0
        wv = self.doc.work_view
        if wv is not None and (e.button() == Qt.MiddleButton or (e.button() == Qt.RightButton and not mods & Qt.AltModifier)):
            self._drag = {'kind': 'wv_pan', 'last': pos}
            return
        if wv is not None and e.button() == Qt.LeftButton and (mods & Qt.AltModifier or self._hit(pos) is None):
            self._drag = {'kind': 'wv_orbit', 'last': pos}
            return
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
            elif isinstance(hit, str) and hit.startswith('gz:'):
                self._gz_press(hit[3:], pos)
            elif isinstance(hit, tuple):
                self.doc.select(hit, force=True)
                kind, i = hit
                sc = self.doc.scene
                p0 = np.asarray(sc.get((kind, i, 'position'), self.doc.frame), float)
                e0 = np.asarray(sc.get(('emitter', i, 'end'), self.doc.frame), float) if kind == 'emitter' else p0
                self._drag = {'kind': 'move', 'what': kind, 'i': i, 'y': float(p0[1]), 'start_world': self._ground_point(pos, p0[1]),
                              'p0': p0, 'e0': e0, 'vertical': bool(mods & Qt.ShiftModifier), 'sy': pos.y()}
                from .gizmo import Gizmo
                try:
                    self._drag['snap'] = Gizmo(self, kind, i).snap()
                except Exception:
                    self._drag['snap'] = 0.1
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
        if d is not None and d['kind'] in ('roto_pt', 'roto_move'):
            self._roto_drag(d, pos)
            return
        if self.path_target is not None and d is None:
            self._path_cursor = self._path_ground(pos)
            self.update()
            return
        if self.roto_mode and d is None:
            self._roto_cursor = pos if self._roto_draft else None
            if self._roto_draft:
                self.update()
            j = self._roto_hit_point(pos)
            self.setCursor(Qt.SizeAllCursor if j is not None else Qt.CrossCursor)
            return
        if d is None:
            h = self._hit(pos)
            h = h if isinstance(h, str) else None
            if h != self._hover:
                self._hover = h
                cursors = {'anchor': Qt.SizeAllCursor, 'scale': Qt.SizeVerCursor, 'gz:rot': Qt.OpenHandCursor, 'gz:centre': Qt.SizeAllCursor}
                cur = cursors.get(h)
                if cur is None and isinstance(h, str) and h.startswith('gz:'):
                    cur = Qt.PointingHandCursor
                self.setCursor(cur or Qt.ArrowCursor)
                self.update()
            return
        k = d['kind']
        if k == 'gz':
            self._gz_move(d, pos, e.modifiers())
            self.update()
            return
        if k in ('wv_orbit', 'wv_pan'):
            delta = pos - d['last']
            d['last'] = pos
            self._rmb_moved += abs(delta.x()) + abs(delta.y())
            wv = self.doc.work_view
            if wv is None:
                return
            if k == 'wv_orbit':
                wv.orbit(delta.x(), delta.y())
            else:
                wv.pan(delta.x(), delta.y(), self.frame_rect().height())
            self.doc.move_work_view()
            return
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
            if e.modifiers() & Qt.ControlModifier and d.get('snap'):
                st = d['snap']
                new = np.array([round(new[0] / st) * st, new[1] if not d['vertical'] else round(new[1] / st) * st,
                                round(new[2] / st) * st])
            delta = new - d['p0']
            self._readout = (f'x {new[0]:.3g}  y {new[1]:.3g}  z {new[2]:.3g} m', pos)
            self.update()
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
        if self._drag is not None and self._drag['kind'] in ('roto_pt', 'roto_move'):
            self._drag = None
            self._roto_release()
            return
        if self._drag is not None:
            self._drag = None
            self._readout = None
            self.doc.end_drag()
            self.update()

    def mouseDoubleClickEvent(self, e):
        if self.path_target is not None:
            self.path_finish()
            return
        if self.roto_mode:
            if self._roto_draft is not None and len(self._roto_draft) >= 3:
                self._roto_finish()
            elif self._roto_draft is None:
                self._roto_press(e)   # a quick second click is a press too (on a point: drag it)
            return
        if e.button() == Qt.LeftButton and self._hit(e.position()) is None:
            self.fit()

    def wheelEvent(self, e):
        if self.doc.work_view is not None:
            self.doc.work_view.dolly(0.87 ** (e.angleDelta().y() / 120.0))
            self.doc.move_work_view()
            return
        f = 1.15 ** (e.angleDelta().y() / 120.0)
        pos = e.position()
        c = QPointF(self.width() / 2, self.height() / 2)
        old = self.zoom
        self.zoom = max(0.1, min(16.0, self.zoom * f))
        k = self.zoom / old
        self.pan = (self.pan + c - pos) * k - (c - pos)
        self.update()

    def keyPressEvent(self, e):
        if self.path_target is not None:
            if e.key() == Qt.Key_Escape:
                self.path_cancel()
                return
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.path_finish()
                return
            if e.key() in (Qt.Key_Backspace, Qt.Key_Delete) and len(self._path_pts) > 1:
                self._path_pts.pop()
                self.update()
                return
        if self.roto_mode:
            if e.key() == Qt.Key_Escape and self._roto_draft is not None:
                self._roto_draft = None
                if self.rotobar is not None:
                    self.rotobar.sync()
                self.update()
                return
            if e.key() in (Qt.Key_Return, Qt.Key_Enter) and self._roto_draft is not None:
                self._roto_finish()
                return
            if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                if self._roto_draft:
                    self._roto_draft.pop()
                    self.update()
                else:
                    self.roto_delete()
                return
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
        if self.doc.work_view is not None:
            from .workview import WorkView
            fr = WorkView.framing(self.doc.scene)
            wv = self.doc.work_view
            wv.distance, wv.target = fr.distance, fr.target
            self.doc.move_work_view()
        self.update()

    def contextMenuEvent(self, e):
        if e.modifiers() & Qt.AltModifier or self.roto_mode or self.path_target is not None:
            return
        if self._rmb_moved > 4:   # that right-drag panned the view
            self._rmb_moved = 0.0
            return
        hit = self._hit(e.pos().toPointF() if hasattr(e.pos(), 'toPointF') else QPointF(e.pos()))
        if isinstance(hit, str) and hit.startswith('gz:'):
            hit = self.doc.selection
        if isinstance(hit, tuple) and hit[0] in ('emitter', 'collider', 'light', 'fabric'):
            self.doc.select(hit, force=True)
            from .actions import fill_menu
            m = QMenu(self)
            fill_menu(m, self.window(), hit, path_mode=self.start_path)
            m.exec(e.globalPos())
            return
        m = QMenu(self)
        add = m.addMenu('Add emitter')
        from .panels import add_collider, add_emitter
        for shape, label in (('sphere', 'Sphere'), ('cylinder', 'Disc'), ('box', 'Box'), ('capsule', 'Line'), ('ring', 'Ring'), ('cone', 'Cone'),
                             ('mesh', 'Mesh…'), ('volume', 'Volume (VDB)…')):
            add.addAction(label, lambda s=shape: add_emitter(self.doc, s, self))
        m.addAction('Add collider box', lambda: self.doc.add_collider('box'))
        m.addAction('Add mesh collider…', lambda: add_collider(self.doc, 'mesh', self))
        m.addSeparator()
        a = m.addAction('Show guides', lambda: (setattr(self, 'guides', not self.guides), self.update()))
        a.setCheckable(True)
        a.setChecked(self.guides)
        m.addAction('Fit view  (F)', self.fit)
        m.exec(e.globalPos())
