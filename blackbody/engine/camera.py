"""Camera and placement math (numpy, row-major matrices; WebGPU clip space with z in [0, 1]).

World space is y-up, metres. The fire's base sits at the fire object's position. The camera is a
physical lens (focal length and sensor width). On top of the projection, an optional 2D placement
transform (anchor, scale, roll) moves the fire's base point onto a chosen spot in the frame and
scales it there, which is how a fire is placed onto footage without a solved camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2.0)
    return np.array([[f / aspect, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, far / (near - far), near * far / (near - far)],
                     [0, 0, -1, 0]], np.float64)


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, np.float64)
    f = np.asarray(target, np.float64) - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(f, (0.0, 0.0, 1.0))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_xyz(rx, ry, rz):
    """Rotation matrix for Euler angles in degrees applied X, then Y, then Z (Nuke/Maya 'XYZ')."""
    return rot_z(math.radians(rz)) @ rot_y(math.radians(ry)) @ rot_x(math.radians(rx))


@dataclass
class CameraSpec:
    mode: str = 'orbit'              # 'orbit' around the fire, or 'free' (position + rotation)
    yaw: float = 0.0                 # degrees, orbit
    pitch: float = 6.0               # degrees above the horizon, orbit
    distance: float = 6.0            # metres from the target, orbit
    target_y: float = 1.0            # metres, orbit look-at height
    position: tuple = (0.0, 1.6, 6.0)
    rotation: tuple = (0.0, 0.0, 0.0)  # degrees, XYZ order
    focal_mm: float = 35.0
    sensor_mm: float = 36.0          # sensor width
    shift: tuple = (0.0, 0.0)        # lens shift in NDC units
    anchor: tuple = (0.5, 0.85)      # where the fire base lands, as a fraction of the frame (y down)
    scale: float = 1.0
    roll: float = 0.0                # degrees, 2D roll about the anchor
    use_anchor: bool = True
    near: float = 0.05
    far: float = 2000.0


@dataclass
class FireXform:
    position: tuple = (0.0, 0.0, 0.0)
    yaw: float = 0.0  # degrees

    def local_to_world(self):
        m = np.eye(4)
        m[:3, :3] = rot_y(math.radians(self.yaw))
        m[:3, 3] = self.position
        return m


@dataclass
class CameraState:
    view: np.ndarray
    proj: np.ndarray          # includes lens shift and the 2D placement transform
    view_proj: np.ndarray
    inv_view_proj: np.ndarray
    eye: np.ndarray
    aspect: float
    hfov: float


def hfov_of(spec: CameraSpec):
    return 2.0 * math.atan(spec.sensor_mm / (2.0 * max(spec.focal_mm, 1e-3)))


def compute(spec: CameraSpec, aspect: float, fire: FireXform | None = None) -> CameraState:
    fire = fire or FireXform()
    base = np.array(fire.position, np.float64)
    if spec.mode == 'orbit':
        yaw, pitch = math.radians(spec.yaw), math.radians(spec.pitch)
        target = base + np.array([0.0, spec.target_y, 0.0])
        eye = target + spec.distance * np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch),
                                                 math.cos(pitch) * math.cos(yaw)])
        view = look_at(eye, target)
    else:
        R = euler_xyz(*spec.rotation)
        eye = np.array(spec.position, np.float64)
        view = np.eye(4)
        view[:3, :3] = R.T
        view[:3, 3] = -R.T @ eye
    hfov = hfov_of(spec)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) / aspect)
    P = perspective(vfov, aspect, spec.near, spec.far)
    P[0, 2] += spec.shift[0]
    P[1, 2] += spec.shift[1]
    if spec.use_anchor:
        clip = P @ view @ np.append(base, 1.0)
        ndc0 = clip[:2] / clip[3] if abs(clip[3]) > 1e-9 else np.zeros(2)
        ax, ay = spec.anchor
        A = np.array([ax * 2.0 - 1.0, 1.0 - ay * 2.0])
        s = spec.scale
        r = math.radians(spec.roll)
        # 2D map in NDC: A + Asp^-1 R S Asp (p - ndc0), Asp = diag(aspect, 1)
        Asp = np.diag([aspect, 1.0])
        M = np.linalg.inv(Asp) @ np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]]) @ (s * Asp)
        t = A - M @ ndc0
        K = np.eye(4)
        K[:2, :2] = M
        K[0, 3], K[1, 3] = t
        P = K @ P
    vp = P @ view
    return CameraState(view=view, proj=P, view_proj=vp, inv_view_proj=np.linalg.inv(vp), eye=eye,
                       aspect=aspect, hfov=hfov)


def project(cam: CameraState, pts, width, height):
    """World points (N, 3) -> pixel coords (N, 2) and a visibility mask (in front of the camera)."""
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ cam.view_proj.T
    w = h[:, 3:4]
    ok = w[:, 0] > 1e-6
    ndc = h[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)
    px = np.stack([(ndc[:, 0] + 1) * 0.5 * width, (1 - ndc[:, 1]) * 0.5 * height], axis=1)
    return px, ok


def pixel_ray(cam: CameraState, x, y, width, height):
    """World-space ray (origin, unit direction) through pixel (x, y)."""
    ndc = np.array([x / width * 2 - 1, 1 - y / height * 2])
    a = cam.inv_view_proj @ np.array([ndc[0], ndc[1], 0.0, 1.0])
    b = cam.inv_view_proj @ np.array([ndc[0], ndc[1], 1.0, 1.0])
    a = a[:3] / a[3]
    b = b[:3] / b[3]
    d = b - a
    return a, d / np.linalg.norm(d)


def world_to_grid(fire: FireXform, grid_origin_local, h):
    """Affine 4x4 mapping world metres to grid cell coordinates."""
    L = fire.local_to_world()
    inv = np.linalg.inv(L)
    T = np.eye(4)
    T[:3, 3] = -np.asarray(grid_origin_local, np.float64)
    S = np.diag([1.0 / h, 1.0 / h, 1.0 / h, 1.0])
    return S @ T @ inv


def sun_direction(azimuth_deg, elevation_deg):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
