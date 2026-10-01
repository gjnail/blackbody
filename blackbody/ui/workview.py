"""The work view: a camera of your own for building the scene, separate from the shot's camera.

It orbits a target point (fire-local metres) at a distance, and the viewer shows the effect through it
over a plain background (the footage belongs to the shot's camera). It only changes what the viewer
shows: the shot, its camera and every render stay as they are, and since the camera is not part of the
simulation, turning around re-draws cached frames without re-simulating.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..engine import camera as cam


@dataclasses.dataclass
class WorkView:
    yaw: float = 35.0          # degrees around the target (0 looks along -z, like the shot camera's default)
    pitch: float = 22.0        # degrees above the horizon
    distance: float = 6.0      # metres from the target
    target: tuple = (0.0, 1.0, 0.0)   # fire-local metres
    focal_mm: float = 28.0

    @classmethod
    def framing(cls, scene):
        """A view that shows the whole simulation box from above and to one side."""
        sx, sy, sz = scene.domain_size()
        r = math.sqrt(sx * sx + sy * sy + sz * sz) / 2
        return cls(yaw=35.0, pitch=22.0, distance=max(0.2, r * 2.4), target=(0.0, sy * 0.42, 0.0))

    @classmethod
    def behind_shot(cls, scene, frame, back=1.6):
        """A view from behind the shot's camera, looking where it looks, so the change is easy to follow
        (and the shot's camera is in the picture). A free shot camera is turned into the same terms."""
        spec, fire = scene.camera(frame)
        if spec.mode == 'orbit':
            # from behind and a little to the side and above: the shot's camera, and what it sees, are in the picture
            return cls(yaw=spec.yaw + 25.0, pitch=min(80.0, spec.pitch + 10.0), distance=max(0.1, spec.distance * back),
                       target=(0.0, spec.target_y, 0.0), focal_mm=min(spec.focal_mm, 35.0))
        L = fire.local_to_world()
        Li = np.linalg.inv(L)
        eye = (Li @ np.append(np.asarray(spec.position, float), 1.0))[:3]
        R = cam.euler_xyz(*spec.rotation)
        fwd = Li[:3, :3] @ -R[:, 2]
        sy = scene.domain_size()[1]
        # the target: where the camera looks, at the height of the middle of the box (or straight ahead)
        t = (sy * 0.4 - eye[1]) / fwd[1] if abs(fwd[1]) > 1e-3 else 5.0
        t = min(max(t, 0.5), 50.0)
        tgt = eye + fwd * t
        d = eye - tgt
        dist = float(np.linalg.norm(d))
        return cls(yaw=math.degrees(math.atan2(d[0], d[2])) + 25.0,
                   pitch=min(80.0, math.degrees(math.asin(max(-1.0, min(1.0, d[1] / max(dist, 1e-9))))) + 10.0),
                   distance=dist * back, target=tuple(float(x) for x in tgt), focal_mm=min(spec.focal_mm, 35.0))

    @classmethod
    def from_shot(cls, scene, frame):
        """A view from exactly where the shot's camera is, looking the way it looks (its placement in the frame
        aside), turning about a point in front of it at the depth of the middle of the box."""
        spec, fire = scene.camera(frame)
        st = cam.compute(dataclasses.replace(spec, use_anchor=False, shift=(0.0, 0.0)), 16 / 9, fire)
        eye = np.asarray(st.eye, float)
        R = st.view[:3, :3].T
        fwd = -R[:, 2]
        L = fire.local_to_world()
        sy = scene.domain_size()[1]
        centre = (L @ np.array([0.0, sy * 0.4, 0.0, 1.0]))[:3]
        t = float(np.dot(centre - eye, fwd))
        t = min(max(t, 0.3), 500.0) if t > 0 else 5.0
        tgt = eye + fwd * t
        d = eye - tgt
        dist = float(np.linalg.norm(d))
        local = (np.linalg.inv(L) @ np.append(tgt, 1.0))[:3]
        return cls(yaw=math.degrees(math.atan2(d[0], d[2])), pitch=math.degrees(math.asin(max(-1.0, min(1.0, d[1] / max(dist, 1e-9))))),
                   distance=dist, target=tuple(float(x) for x in local), focal_mm=float(spec.focal_mm) * 36.0 / max(float(spec.sensor_mm), 1e-3))

    def shot_values(self, scene, frame):
        """The shot camera settings that see what this view sees: a free camera at the eye, the same field of view on
        the shot's own sensor, no placement in the frame."""
        spec, fire = scene.camera(frame)
        eye, rot = self.eye_and_rotation(fire)
        return {'position': tuple(float(x) for x in eye), 'rotation': tuple(float(x) for x in rot),
                'focal_mm': float(self.focal_mm) * float(spec.sensor_mm) / 36.0, 'near': max(0.001, min(float(spec.near), self.distance * 0.01))}

    def eye_and_rotation(self, fire: cam.FireXform):
        """World-space eye and the XYZ Euler rotation (degrees) of a free camera looking at the target."""
        L = fire.local_to_world()
        tgt = (L @ np.append(np.asarray(self.target, float), 1.0))[:3]
        yaw, pitch = math.radians(self.yaw), math.radians(max(-89.0, min(89.0, self.pitch)))
        eye = tgt + self.distance * np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
        m = cam.look_at(eye, tgt)
        R = m[:3, :3].T   # camera axes in world space (columns): right, up, back
        # R = Rz(c) Ry(b) Rx(a); looking straight along a world axis (b = +-90 degrees) only a - c (or a + c)
        # is defined: take c = 0 there
        if R[2, 0] < -0.999999:
            b, c = math.pi / 2, 0.0
            a = math.atan2(R[0, 1], R[1, 1])
        elif R[2, 0] > 0.999999:
            b, c = -math.pi / 2, 0.0
            a = math.atan2(-R[0, 1], R[1, 1])
        else:
            b = math.asin(-R[2, 0])
            a = math.atan2(R[2, 1], R[2, 2])
            c = math.atan2(R[1, 0], R[0, 0])
        return eye, (math.degrees(a), math.degrees(b), math.degrees(c))

    def spec(self, shot: cam.CameraSpec, fire: cam.FireXform):
        """The shot's camera spec turned into this view's (lens and clipping kept, no placement in the frame)."""
        eye, rot = self.eye_and_rotation(fire)
        r = max(self.distance, 1e-3)
        return dataclasses.replace(shot, mode='free', position=tuple(float(x) for x in eye), rotation=rot, focal_mm=self.focal_mm,
                                   sensor_mm=36.0, use_anchor=False, roll=0.0, shift=(0.0, 0.0),
                                   near=min(shot.near, r * 0.01), far=max(shot.far, r * 50))

    def apply(self, scene):
        """Point a copy of the scene (for the engine) through this view, with no footage behind it."""
        spec, fire = scene.camera(scene.start)
        eye, rot = self.eye_and_rotation(fire)
        c = scene.data['camera']
        for k in list(c.keys()):
            v = c[k]
            if hasattr(v, 'eval'):   # an animated camera setting: the work view holds still
                c[k] = v.eval(scene.start)
        c.update(mode='free', position=tuple(float(x) for x in eye), rotation=rot, focal_mm=self.focal_mm, sensor_mm=36.0,
                 use_anchor=False, roll=0.0, near=min(float(c['near']), self.distance * 0.01),
                 far=max(float(c['far']), self.distance * 50))
        scene.track = None
        scene.footage = None
        comp = scene.data['composite']
        comp['bg'] = (0.035, 0.035, 0.04)
        comp['bg_checker'] = False
        if scene.kind in ('liquid', 'both'):   # no footage to show the objects: draw them as solid grey stand-ins
            scene.data['water']['colliders_look'] = 'shaded'
        return scene

    # -- moves --------------------------------------------------------------------------------------------

    def orbit(self, dx_px, dy_px):
        self.yaw -= dx_px * 0.35
        self.pitch = max(-85.0, min(89.0, self.pitch + dy_px * 0.25))

    def dolly(self, factor):
        self.distance = max(0.02, self.distance * factor)

    def pan(self, dx_px, dy_px, view_h_px):
        """Slide the target across the view (the way the picture moves under the mouse)."""
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        fwd = -np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
        up = np.cross(right, fwd)
        up /= max(np.linalg.norm(up), 1e-9)
        hfov = cam.hfov_of(cam.CameraSpec(focal_mm=self.focal_mm, sensor_mm=36.0))
        m_per_px = 2.0 * self.distance * math.tan(hfov / 2.0) / max(view_h_px * 16 / 9, 1.0)
        t = np.asarray(self.target, float) - right * dx_px * m_per_px + up * dy_px * m_per_px
        self.target = tuple(float(x) for x in t)
