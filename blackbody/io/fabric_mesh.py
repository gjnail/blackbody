"""Fabric as meshes for other programs: every frame's cloth (world space, metres, y up), as it has burnt
so far (burnt-through triangles left out), as one OBJ per frame or one USD file for the shot.

USD: /World/Fabric/<name> per fabric, a mesh with time-sampled points, topology (it changes as holes
burn through), normals and velocities, constant UVs (`st`, the weave coordinates in metres) and a
per-vertex `burn` primvar (0 untouched .. 1 burnt through) for shading the char.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..engine import camera as cam


def _safe(name):
    s = re.sub(r'[^A-Za-z0-9_]', '_', name or 'Fabric')
    return s if s[:1].isalpha() else 'F_' + s


def fabric_meshes(engine, scene, frame):
    """[{name, points, normals, velocities, uv, burn, faces}] for each enabled fabric at `frame` (live or
    cached), in world space."""
    cl = engine.cloth
    if not cl.active:
        return []
    arrays = None
    if scene.kind == 'liquid':
        # (a liquid scene's frames keep the cloth with the liquid)
        if engine.sim_frame != frame:
            entry = engine.cache.get(frame) or {}
            arrays = {k[6:]: v for k, v in entry.items() if k.startswith('cloth_') and k != 'cloth_state'} or None
    else:
        vol, live = engine.volume_for(scene, frame)
        c = getattr(vol, 'cloth', None) if vol is not None else None
        arrays = None if c is None or isinstance(c, str) else c
    if arrays is None:
        rd = lambda name: np.frombuffer(engine.gpu.read_buffer(cl.bufs[name]), np.float32).reshape(-1, 4)[:cl.built.n]
        X, N, S, V = rd('X'), rd('N'), rd('S'), rd('V')
    else:
        X, N, S, V = (np.asarray(arrays[k], np.float32) for k in ('X', 'N', 'S', 'V'))
    spec, fire = scene.camera(frame)
    R = cam.rot_y(math.radians(fire.yaw))
    pos = np.asarray(fire.position, np.float32)
    B = cl.built
    names = [d['name'] for d in scene.fabrics if d['enabled'] and not (d['shape'] == 'mesh' and not d['mesh'])]
    out = []
    for f, (a, n) in enumerate(B.ranges):
        tris = B.tris[B.tris[:, 3] == f][:, :3].astype(np.int64) - a
        burn = np.clip(S[a:a + n, 1], 0.0, 1.0)
        alive = (burn[tris] < 1.0).all(1)
        out.append({
            'name': names[f] if f < len(names) else f'Fabric{f + 1}',
            'points': (X[a:a + n, :3] @ R.T + pos).astype(np.float32),
            'normals': (N[a:a + n, :3] @ R.T).astype(np.float32),
            'velocities': (V[a:a + n, :3] @ R.T).astype(np.float32),
            'uv': B.uv[a:a + n, :2].astype(np.float32),
            'burn': burn.astype(np.float32),
            'faces': tris[alive].astype(np.int32),
        })
    return out


def write_obj(path, meshes):
    """One frame of every fabric as a Wavefront OBJ (an object per fabric, with UVs and normals)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = 1
    with open(path, 'w', encoding='ascii', newline='\n') as fh:
        fh.write('# Blackbody fabric\n')
        for m in meshes:
            fh.write(f'o {_safe(m["name"])}\n')
            fh.write(''.join(f'v {a:.5f} {b:.5f} {c:.5f}\n' for a, b, c in m['points']))
            fh.write(''.join(f'vt {a:.5f} {b:.5f}\n' for a, b in m['uv']))
            fh.write(''.join(f'vn {a:.4f} {b:.4f} {c:.4f}\n' for a, b, c in m['normals']))
            fc = m['faces'] + base
            fh.write(''.join(f'f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n' for a, b, c in fc))
            base += len(m['points'])
    return path


class UsdWriter:
    """The whole shot's fabric in one USD file (see the module notes)."""

    def __init__(self, path, fps):
        from pxr import Usd, UsdGeom
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Usd.Stage.CreateNew(str(self.path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetTimeCodesPerSecond(fps)
        self.stage.SetFramesPerSecond(fps)
        UsdGeom.Xform.Define(self.stage, '/World')
        UsdGeom.Xform.Define(self.stage, '/World/Fabric')
        self.meshes = {}
        self.first = None
        self.last = None

    def _mesh(self, m):
        from pxr import Sdf, UsdGeom, Vt
        key = _safe(m['name'])
        if key not in self.meshes:
            mesh = UsdGeom.Mesh.Define(self.stage, f'/World/Fabric/{key}')
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
            pv = UsdGeom.PrimvarsAPI(mesh)
            st = pv.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(m['uv'], np.float32)))
            burn = pv.CreatePrimvar('burn', Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
            self.meshes[key] = (mesh, burn)
        return self.meshes[key]

    def add(self, frame, meshes):
        from pxr import Gf, Vt
        t = float(frame)
        self.first = t if self.first is None else min(self.first, t)
        self.last = t if self.last is None else max(self.last, t)
        for m in meshes:
            mesh, burn = self._mesh(m)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(m['points'])), t)
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(m['faces']), 3, np.int32)), t)
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(m['faces'].ravel(), np.int32)), t)
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(m['normals'])), t)
            mesh.SetNormalsInterpolation('vertex')
            mesh.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(m['velocities'])), t)
            burn.Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(m['burn'])), t)
            p = m['points']
            if len(p):
                lo, hi = p.min(0), p.max(0)
                mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]), t)

    def close(self):
        if self.first is not None:
            self.stage.SetStartTimeCode(self.first)
            self.stage.SetEndTimeCode(self.last)
        self.stage.GetRootLayer().Save()
        return self.path
