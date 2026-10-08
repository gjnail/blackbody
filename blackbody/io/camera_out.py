"""The shot's camera for other programs: as a USD camera (io/scene_usd.py), a Nuke .chan file, and the camera matrices
in an EXR's header (worldToCamera, worldToNDC).

Blackbody's camera is a physical lens (focal length, sensor width) with an optional lens shift, and on top of it the
2D placement that puts the effect's base on a chosen spot in the frame (anchor, scale, roll: engine/camera.py). That
placement is a turn, a zoom and a slide of the picture, so it is written as a real camera would hold it: the turn as a
roll of the camera about its view axis, the zoom in the focal length, and the slide (with the lens shift) as the film
back's offset. A USD camera holds all three, so its picture lines up with Blackbody's exactly. A .chan file has no film
offset: it holds the turn and the zoom, and the slide is said (`offset_px`). Neither holds the footage's lens distortion
(Composite › Lens distortion, which bends the element toward the frame's edges): the USD camera carries it as
blackbody:lens_k1, and the render says so (job.output_notes).

The camera's own frame looks down -z with +y up (OpenGL's and USD's convention), in the scene's metres, y up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..engine import camera as cam


@dataclass
class CameraFrame:
    to_world: np.ndarray      # 4x4 camera -> world (rigid; column vectors)
    view: np.ndarray          # 4x4 world -> camera (its inverse)
    view_proj: np.ndarray     # 4x4 world -> clip, exactly as Blackbody renders (WebGPU clip space: z in 0..1)
    focal_mm: float           # the focal length that gives the picture's zoom (the lens's, times the placement's scale)
    h_aperture: float         # mm: the sensor's width
    v_aperture: float         # mm: its height (width over the picture's aspect)
    offset_mm: tuple          # (x, y) film back offset (mm), as USD's aperture offsets: the window moves, the picture
                              # moves the other way
    near: float               # m
    far: float                # m
    aspect: float             # width over height
    size: tuple               # (width, height) pixels
    lens_k1: float = 0.0      # the footage's lens distortion the element is bent by (composite.wgsl undistort)

    @property
    def vfov(self):
        """The vertical field of view (degrees), as a .chan file holds it."""
        return math.degrees(2.0 * math.atan(0.5 * self.v_aperture / max(self.focal_mm, 1e-9)))

    @property
    def offset_px(self):
        """How far (pixels, x right and y down) the picture is slid off the lens's centre: what a .chan cannot hold."""
        w, h = self.size
        return (-self.offset_mm[0] / self.h_aperture * w, self.offset_mm[1] / self.v_aperture * h)


def camera_at(scene, frame, size=None):
    """The shot's camera at `frame` (CameraFrame), for an output `size` (width, height) or the scene's."""
    W, H = size or scene.output_size()
    aspect = W / H
    spec, fire = scene.camera(frame)
    cs = cam.compute(spec, aspect, fire)
    P = np.asarray(cs.proj, float)
    # P maps a point (x, y, z) of the camera's frame to ndc = A (x, y) / -z + c (its last row is (0, 0, -1, 0) and
    # the placement only adds a 2D affine map): in units of half the picture's height (x times the aspect) that is
    # B (x, y) / -z + o with B = s f R(roll), a zoom and a turn
    asp = np.diag([aspect, 1.0])
    B = asp @ P[:2, :2]
    o = asp @ (-P[:2, 2])
    zoom = math.sqrt(max(float(np.linalg.det(B)), 1e-18))
    roll = math.atan2(B[1, 0], B[0, 0])
    # the turn as the camera's own roll: its frame turned by -roll about its view axis
    turn = np.eye(4)
    turn[:2, :2] = [[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]]
    view = turn @ np.asarray(cs.view, float)
    to_world = np.linalg.inv(view)
    to_world[3] = (0.0, 0.0, 0.0, 1.0)
    h_ap = float(spec.sensor_mm)
    v_ap = h_ap / aspect
    focal = zoom * 0.5 * v_ap
    off = tuple(float(-x * 0.5 * v_ap) for x in o)
    return CameraFrame(to_world=to_world, view=view, view_proj=np.asarray(cs.view_proj, float), focal_mm=float(focal),
                       h_aperture=h_ap, v_aperture=v_ap, offset_mm=off, near=float(spec.near), far=float(spec.far),
                       aspect=aspect, size=(int(W), int(H)), lens_k1=lens_k1(scene))


def lens_k1(scene):
    """The footage's lens distortion the picture is bent by (Composite › Lens distortion): x_d = x_u (1 + k1 r_u^2), r
    from the frame's centre in half-diagonals (composite.wgsl undistort)."""
    return float(scene.data['composite'].get('lens_k1', 0.0))


def slide_px(scene, frames, size=None):
    """The largest slide of the picture off the lens's centre (pixels) over `frames`: what a .chan leaves out."""
    return max((math.hypot(*camera_at(scene, f, size).offset_px) for f in frames), default=0.0)


def slide_note(path, px):
    """What a .chan leaves out of a picture slid `px` off the lens's centre."""
    return (f'{path}: the picture is slid up to {px:.0f} px off the lens\'s centre (the Anchor, or a lens shift), which '
            'a .chan cannot hold: in Nuke, set the Camera\'s window translate, or use the USD camera (it holds the slide '
            'as its film offset) or the EXRs\' worldToNDC.')


def exr_attrs(cf: CameraFrame):
    """The camera as an EXR's standard header attributes (OpenEXR's worldToCamera and worldToNDC, 4x4 floats in Imath's
    row-vector order: a point times the matrix). worldToNDC is exactly the render's: its x and y over its w are the
    picture's -1..1, y up."""
    return {'worldToCamera': np.ascontiguousarray(cf.view.T, np.float32),
            'worldToNDC': np.ascontiguousarray(cf.view_proj.T, np.float32)}


# -- .chan ---------------------------------------------------------------------------------------------------------

def zxy_euler(R):
    """(rx, ry, rz) degrees with R = Ry Rx Rz: Nuke's default rotation order ZXY (io/chan.py reads it so)."""
    sa = -float(R[1, 2])
    sa = max(-1.0, min(1.0, sa))
    a = math.asin(sa)
    if abs(sa) < 0.999999:
        b = math.atan2(R[0, 2], R[2, 2])
        c = math.atan2(R[1, 0], R[1, 1])
    else:   # (looking straight up or down: the roll goes into the heading)
        b = math.atan2(-R[2, 0], R[0, 0])
        c = 0.0
    return math.degrees(a), math.degrees(b), math.degrees(c)


def chan_rows(frames):
    """[(frame, CameraFrame)] as .chan rows: frame, tx ty tz (m), rx ry rz (degrees, ZXY), vertical field of view
    (degrees). Each angle keeps on from the frame before (no jump from 180 to -180)."""
    rows, prev = [], None
    for f, cf in frames:
        e = np.array(zxy_euler(cf.to_world[:3, :3]))
        if prev is not None:
            e = prev + ((e - prev + 180.0) % 360.0 - 180.0)
        prev = e
        rows.append((f, *(float(x) for x in cf.to_world[:3, 3]), *(float(x) for x in e), cf.vfov))
    return rows


def write_chan(path, frames):
    """Write a camera move as a .chan file (Nuke's Camera › Import chan file, also Blender, Maya and 3DEqualizer
    importers): one line a frame. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='ascii', newline='\n') as fh:
        for r in chan_rows(frames):
            fh.write(f'{r[0]:g}\t' + '\t'.join(f'{x:.6f}' for x in r[1:]) + '\n')
    return path


class ChanWriter:
    """A .chan output: the camera of every frame of the shot, written when the render ends. `said`: the slide (px)
    already said before the render (job.output_notes looks at a few frames); a larger one over every frame written is
    said again at the end (notes)."""

    def __init__(self, path, said=0.0):
        self.path = Path(path)
        self.frames = []
        self.said = float(said)
        self.notes = []

    def add(self, frame, cf: CameraFrame):
        self.frames.append((frame, cf))

    def offset_px(self):
        """The largest slide of the picture off the lens's centre over the shot (pixels), which the file leaves out."""
        return max((math.hypot(*cf.offset_px) for _, cf in self.frames), default=0.0)

    def close(self):
        px = self.offset_px()
        if px > 0.5 and px > self.said + 0.5:
            self.notes.append(slide_note(self.path, px))
        return write_chan(self.path, sorted(self.frames, key=lambda x: x[0]))
