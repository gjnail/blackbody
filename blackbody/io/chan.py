"""Camera tracks in the .chan format (Nuke, Blender, SynthEyes, 3DEqualizer, PFTrack exports).

Each line: frame tx ty tz rx ry rz [vertical_fov]. Translation in scene units (assumed metres,
y up), rotation in degrees. Nuke's default rotation order is ZXY; pass another order if your
tracker used one.
"""
from __future__ import annotations

import math

import numpy as np

from ..engine.camera import rot_x, rot_y, rot_z
from ..scene.anim import Curve


def read_chan(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or parts[0].startswith('#'):
                continue
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError('no camera keys found')
    return rows


def _matrix(rx, ry, rz, order):
    R = {'X': rot_x(math.radians(rx)), 'Y': rot_y(math.radians(ry)), 'Z': rot_z(math.radians(rz))}
    # order 'ZXY' means rotate about Z first, then X, then Y (matrix = Ry @ Rx @ Rz)
    m = np.eye(3)
    for axis in order:
        m = R[axis] @ m
    return m


def _to_xyz_euler(m):
    """Decompose R = Rz @ Ry @ Rx into (rx, ry, rz) degrees (Blackbody's free-camera convention)."""
    sy = -m[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(m[2, 1], m[2, 2])
        rz = math.atan2(m[1, 0], m[0, 0])
    else:
        rx = math.atan2(-m[1, 2], m[1, 1])
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def chan_keys(path, order='ZXY', sensor_mm=36.0, aspect=16 / 9):
    rows = read_chan(path)
    pos, rot, focal = [], [], []
    prev = None
    for r in rows:
        f = r[0]
        tx, ty, tz, rx, ry, rz = r[1:7]
        m = _matrix(rx, ry, rz, order.upper())
        e = np.array(_to_xyz_euler(m))
        if prev is not None:  # keep Euler angles continuous so interpolation does not spin the long way
            e = prev + ((e - prev + 180.0) % 360.0 - 180.0)
        prev = e
        pos.append([f, (tx, ty, tz), 'linear'])
        rot.append([f, tuple(float(x) for x in e), 'linear'])
        if len(r) >= 8 and r[7] > 0:
            vfov = math.radians(r[7])
            hfov = 2 * math.atan(math.tan(vfov / 2) * aspect)
            focal.append([f, sensor_mm / (2 * math.tan(hfov / 2)), 'linear'])
    return pos, rot, focal


def apply_chan(doc, path, order='ZXY'):
    sc = doc.scene
    W, H = sc.output_size()
    pos, rot, focal = chan_keys(path, order, sc.data['camera']['sensor_mm'], W / H)

    def fn(s):
        c = s.data['camera']
        c['mode'] = 'free'
        c['use_anchor'] = False
        c['position'] = Curve(pos)
        c['rotation'] = Curve(rot)
        if focal:
            c['focal_mm'] = Curve(focal)
    doc.edit('Import camera track', fn, structure=True)
    doc.sceneReplaced.emit()
    return len(pos)
