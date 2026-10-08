"""The liquid surface as a polygon mesh, for lighting and rendering the water in Blender, Houdini,
Maya or any other 3D package.

The mesh comes from the same signed-distance surface the renderer traces (the render surface grid,
at final quality), by surface nets: a vertex inside every grid cube the surface passes through, at
the mean of the points where the surface crosses the cube's edges, and a quad across every grid
edge the surface crosses. The result is watertight, with even, well-shaped faces. Each vertex
carries the liquid's velocity there (m/s), for motion blur.

Writers:
  OBJ   one file per frame (name.####.obj): positions, normals and faces; universal.
  USD   one file for the whole shot (.usd, .usdc or .usda): a mesh whose points, faces, normals and
        velocities are sampled per frame, plus the spray, foam and bubbles as point clouds (widths
        from the droplet size). Opens in Houdini (Solaris), Blender, Maya, Omniverse.
Coordinates are the scene's world metres, y up, as the camera export (io/camera_out.py) and the USD scene
(io/scene_usd.py) use, so the surface lines up with them.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# the 12 edges of a unit cube: (corner offset, axis)
_EDGES = [((0, 0, 0), 0), ((0, 1, 0), 0), ((0, 0, 1), 0), ((0, 1, 1), 0),
          ((0, 0, 0), 1), ((1, 0, 0), 1), ((0, 0, 1), 1), ((1, 0, 1), 1),
          ((0, 0, 0), 2), ((1, 0, 0), 2), ((0, 1, 0), 2), ((1, 1, 0), 2)]


def surface_nets(phi, vel=None, iso=0.0):
    """Mesh of the iso-surface of phi (z, y, x) on a unit lattice (node i at x = i).

    Returns vertices (n, 3) as (x, y, z) in node units, quads (m, 4) indexing them (outward winding,
    toward positive phi), and per-vertex velocity (n, 3) if vel (z, y, x, 3) is given."""
    f = np.asarray(phi, np.float32) - iso
    nz, ny, nx = f.shape
    if min(nx, ny, nz) < 2:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.int64), np.zeros((0, 3), np.float32)
    inside = f < 0.0
    # cubes the surface passes through
    c = inside[:-1, :-1, :-1].astype(np.int8)
    tot = np.zeros(c.shape, np.int8)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                tot += inside[dz:nz - 1 + dz, dy:ny - 1 + dy, dx:nx - 1 + dx]
    active = (tot > 0) & (tot < 8)
    idx = np.full(active.shape, -1, np.int64)
    cz, cy, cx = np.nonzero(active)
    nv = len(cz)
    idx[cz, cy, cx] = np.arange(nv)
    # vertex: the mean of the edge crossings in each active cube
    acc = np.zeros((nv, 3), np.float64)
    cnt = np.zeros(nv, np.float64)
    vacc = np.zeros((nv, 3), np.float64) if vel is not None else None
    for (ox, oy, oz), ax in _EDGES:
        a = f[cz + oz, cy + oy, cx + ox]
        d = [0, 0, 0]
        d[ax] = 1
        b = f[cz + oz + d[2], cy + oy + d[1], cx + ox + d[0]]
        cross = (a < 0.0) != (b < 0.0)
        if not cross.any():
            continue
        t = np.where(cross, a / np.where(cross, a - b, 1.0), 0.0)
        p = np.stack([cx + ox, cy + oy, cz + oz], -1).astype(np.float64)
        p[:, ax] += t
        acc[cross] += p[cross]
        cnt[cross] += 1.0
        if vacc is not None:
            va = vel[cz + oz, cy + oy, cx + ox]
            vb = vel[cz + oz + d[2], cy + oy + d[1], cx + ox + d[0]]
            vacc[cross] += (va + (vb - va) * t[:, None])[cross]
    cnt = np.maximum(cnt, 1.0)
    verts = (acc / cnt[:, None]).astype(np.float32)
    vv = (vacc / cnt[:, None]).astype(np.float32) if vacc is not None else np.zeros((nv, 3), np.float32)
    # faces: one quad across every lattice edge the surface crosses, joining the four cubes around it
    quads = []
    for ax in range(3):
        a = f
        if ax == 0:
            s0, s1 = inside[:, :, :-1], inside[:, :, 1:]
        elif ax == 1:
            s0, s1 = inside[:, :-1, :], inside[:, 1:, :]
        else:
            s0, s1 = inside[:-1, :, :], inside[1:, :, :]
        crossing = s0 != s1
        ez, ey, ex = np.nonzero(crossing)
        # the four cubes sharing the edge: offset by -1/0 in the two other axes
        o1, o2 = [(1, 2), (0, 2), (0, 1)][ax]
        pos = np.stack([ex, ey, ez], -1)
        cubes = []
        for d1, d2 in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
            q = pos.copy()
            q[:, o1] += d1
            q[:, o2] += d2
            cubes.append(q)
        ok = np.ones(len(pos), bool)
        for q in cubes:
            ok &= (q >= 0).all(-1) & (q[:, 0] < nx - 1) & (q[:, 1] < ny - 1) & (q[:, 2] < nz - 1)
        ids = np.stack([idx[np.clip(q[:, 2], 0, nz - 2), np.clip(q[:, 1], 0, ny - 2), np.clip(q[:, 0], 0, nx - 2)]
                        for q in cubes], -1)
        ok &= (ids >= 0).all(-1)
        ids = ids[ok]
        flip = s0[ez, ey, ex][ok]   # inside -> outside along +axis: keep, else reverse
        # consistent outward winding for the handedness of each axis pair
        if ax == 1:
            flip = ~flip
        ids = np.where(flip[:, None], ids, ids[:, ::-1])
        quads.append(ids)
    faces = np.concatenate(quads) if quads else np.zeros((0, 4), np.int64)
    return verts, faces, vv


def vertex_normals(verts, faces):
    n = np.zeros_like(verts, dtype=np.float64)
    if len(faces):
        v = verts[faces]
        fn = np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 1])
        for k in range(4):
            np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    return (n / np.maximum(ln, 1e-12)).astype(np.float32)


def liquid_surface_mesh(engine, scene, frame, whitewater=True):
    """The liquid surface at `frame` in world metres: dict(points, faces, normals, velocities, and
    optionally spray/foam/bubbles point positions)."""
    from ..engine import camera as cam
    vol, _ = engine._liquid_view(scene, frame)
    if vol is None:
        raise RuntimeError(f'frame {frame} is neither simulated nor cached')
    look = scene.water_look(frame, final=True)
    lr = engine.liquid_r
    with engine.gpu.batch() as b:
        lr.build(b, vol, look)
    surf = engine.gpu.read(lr.surf).astype(np.float32)        # (z, y, x, 4): distance (surface cells), velocity
    phi = surf[..., 0]
    # crop to where the surface is (the mesher only needs the band around it)
    near = np.abs(phi) < 1.5
    if not near.any():
        out = dict(points=np.zeros((0, 3), np.float32), faces=np.zeros((0, 4), np.int64),
                   normals=np.zeros((0, 3), np.float32), velocities=np.zeros((0, 3), np.float32))
    else:
        zz, yy, xx = (np.nonzero(near.any(axis=(1, 2)))[0], np.nonzero(near.any(axis=(0, 2)))[0], np.nonzero(near.any(axis=(0, 1)))[0])
        z0, z1 = max(zz[0] - 1, 0), min(zz[-1] + 2, phi.shape[0])
        y0, y1 = max(yy[0] - 1, 0), min(yy[-1] + 2, phi.shape[1])
        x0, x1 = max(xx[0] - 1, 0), min(xx[-1] + 2, phi.shape[2])
        verts, faces, vv = surface_nets(phi[z0:z1, y0:y1, x0:x1], surf[z0:z1, y0:y1, x0:x1, 1:4])
        verts = verts + np.array([x0, y0, z0], np.float32)
        spec, fire = scene.camera(frame)
        vs = vol.h * vol.dims[0] / lr.nf[0]
        local = np.asarray(vol.origin, np.float32) + (verts + 0.5) * vs
        R = cam.rot_y(math.radians(fire.yaw)).astype(np.float32)
        pts = local @ R.T + np.asarray(fire.position, np.float32)
        vel = vv @ R.T
        nrm = vertex_normals(pts, faces)
        out = dict(points=pts.astype(np.float32), faces=faces, normals=nrm, velocities=vel.astype(np.float32))
    if whitewater and vol.ww is not None and vol.ww_count:
        pk = np.frombuffer(engine.gpu.read_buffer(vol.ww, vol.ww_count * 16), np.uint32).reshape(-1, 4)
        spec, fire = scene.camera(frame)
        R = cam.rot_y(math.radians(fire.yaw))
        q = np.stack([(pk[:, 0] & 0xFFFF), (pk[:, 0] >> 16), (pk[:, 1] & 0xFFFF)], -1).astype(np.float32) / 65535.0
        code = (pk[:, 1] >> 16).astype(np.float32) / 65535.0 * 3.0
        cls = np.minimum(np.floor(code + 1e-4), 2).astype(int)
        local = np.asarray(vol.origin, np.float32) + q * np.asarray(vol.dims, np.float32) * vol.h
        world = (local @ R.T + np.asarray(fire.position)).astype(np.float32)
        for k, name in enumerate(('spray', 'foam', 'bubbles')):
            out[name] = world[cls == k]
    return out


def write_obj(path, mesh):
    """One frame as Wavefront OBJ (positions, normals, quads)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p, n, f = mesh['points'], mesh['normals'], mesh['faces'] + 1
    with open(path, 'w', encoding='ascii', newline='\n') as fh:
        fh.write('# Blackbody liquid surface\n')
        fh.write(''.join(f'v {a:.5f} {b:.5f} {c:.5f}\n' for a, b, c in p))
        fh.write(''.join(f'vn {a:.4f} {b:.4f} {c:.4f}\n' for a, b, c in n))
        fh.write(''.join(f'f {a}//{a} {b}//{b} {c}//{c} {d}//{d}\n' for a, b, c, d in f))
    return path


class UsdWriter:
    """A whole shot of the liquid in one USD file: /World/Liquid/Surface (mesh, time-sampled points,
    topology, normals and velocities) and /World/Liquid/{Spray,Foam,Bubbles} (points)."""

    def __init__(self, path, fps, droplet_size=0.003):
        from pxr import Sdf, Usd, UsdGeom
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Usd.Stage.CreateNew(str(self.path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetTimeCodesPerSecond(fps)
        self.stage.SetFramesPerSecond(fps)
        UsdGeom.Xform.Define(self.stage, '/World')
        UsdGeom.Xform.Define(self.stage, '/World/Liquid')
        self.mesh = UsdGeom.Mesh.Define(self.stage, '/World/Liquid/Surface')
        self.mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        self.mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
        self.points = {}
        for name in ('Spray', 'Foam', 'Bubbles'):
            self.points[name.lower()] = UsdGeom.Points.Define(self.stage, f'/World/Liquid/{name}')
        self.width = float(droplet_size)
        self.first = None
        self.last = None
        self._Sdf = Sdf

    def add(self, frame, mesh):
        from pxr import Gf, Vt
        t = float(frame)
        self.first = t if self.first is None else min(self.first, t)
        self.last = t if self.last is None else max(self.last, t)
        m = self.mesh
        m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh['points']), t)
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(mesh['faces']), 4, np.int32)), t)
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(mesh['faces'].astype(np.int32).ravel()), t)
        m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh['normals']), t)
        m.SetNormalsInterpolation('vertex')
        m.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(mesh['velocities']), t)
        if len(mesh['points']):
            lo, hi = mesh['points'].min(0), mesh['points'].max(0)
            m.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]), t)
        for name, pts in self.points.items():
            p = mesh.get(name)
            if p is None:
                p = np.zeros((0, 3), np.float32)
            pts.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p, np.float32)), t)
            w = self.width * (6.0 if name == 'foam' else (1.5 if name == 'bubbles' else 1.0))
            pts.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(len(p), w, np.float32)), t)

    def close(self):
        if self.first is not None:
            self.stage.SetStartTimeCode(self.first)
            self.stage.SetEndTimeCode(self.last)
        self.stage.GetRootLayer().Save()
        return self.path
