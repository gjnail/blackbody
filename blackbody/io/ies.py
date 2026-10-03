"""Light profiles: IES LM-63 photometric files (.ies), as the makers of real lights publish them (a downlight's beam, a
street light's spread, a wall washer's throw). A file gives the light's intensity (candela) over a grid of directions:
vertical angles from straight down (0°, the light's nadir) to straight up (180°), and horizontal angles round the
vertical from its length (0°); fewer angles where the light is symmetric (one: round; 0–90°: in quarters; 0–180°: two
halves alike).

`load` reads one into a Profile: the table resampled onto the grid the renderers read (PROF_V vertical angles, 0 to
180° inclusive, the light's aim being 0°; PROF_H horizontal angles round it from the light's width axis), as a shape
over its peak, with the peak's candela and the whole light's luminous flux (lumens), worked out from the table.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PROF_V = 64    # vertical angles in a lamp's profile table (0..180°, both ends)
PROF_H = 16    # horizontal angles (0..360°, every 22.5°)


@dataclass(frozen=True)
class Profile:
    table: np.ndarray       # (PROF_V, PROF_H): intensity over the peak, the vertical angle from the aim down the rows
    peak: float             # candela in the brightest direction (the file's values times its multipliers)
    lumens: float           # the light's whole output, from the table
    name: str = ''


def _numbers(text):
    return [float(x) for x in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', text)]


def parse(text: str, name: str = '') -> Profile:
    """A Profile from the text of an IES LM-63 file (1986, 1991, 1995 or 2002)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip().upper().startswith('TILT'):
        i += 1
    if i == len(lines):
        raise ValueError('not an IES file: no TILT line')
    tilt = lines[i].split('=', 1)[1].strip().upper() if '=' in lines[i] else 'NONE'
    nums = _numbers('\n'.join(lines[i + 1:]))
    p = 0
    if tilt == 'INCLUDE':
        # lamp-to-luminaire geometry, then how many tilt angles, the angles and their factors (ignored: the light is
        # taken as aimed as it was measured)
        n = int(nums[p + 1])
        p += 2 + 2 * n
    elif tilt != 'NONE':
        pass   # (TILT=<file>: a separate file; ignored as above)
    if len(nums) - p < 13:
        raise ValueError('IES file too short')
    _n_lamps, _lm_per_lamp, mult, nv, nh, ptype, _units, _w, _l, _h = nums[p:p + 10]
    ballast, _future, _watts = nums[p + 10:p + 13]
    p += 13
    nv, nh = int(nv), int(nh)
    if nv < 1 or nh < 1 or len(nums) - p < nv + nh + nv * nh:
        raise ValueError('IES file: its table is incomplete')
    va = np.asarray(nums[p:p + nv], float)
    p += nv
    ha = np.asarray(nums[p:p + nh], float)
    p += nh
    cd = np.asarray(nums[p:p + nv * nh], float).reshape(nh, nv) * (mult * (ballast if ballast > 0 else 1.0))
    if int(ptype) != 1:
        # (type B and A photometry, floodlights and car lamps measured about a horizontal axis: taken as type C, which
        # turns their pattern a quarter turn; rare in files for set lights)
        pass
    cd = np.maximum(cd, 0.0)
    return _resample(va, ha, cd, name)


def _resample(va, ha, cd, name):
    """The table onto the renderers' grid: vertical angles 0..180° (rows), horizontal 0..360° (columns)."""
    g = np.linspace(0.0, 180.0, PROF_V)
    phi = np.arange(PROF_H) * (360.0 / PROF_H)
    hmax = float(ha[-1]) if len(ha) > 1 else 0.0
    if len(ha) == 1:
        fold = np.zeros_like(phi)
    elif hmax <= 90.0 + 1e-6:
        f = np.mod(phi, 180.0)
        fold = np.where(f > 90.0, 180.0 - f, f)                  # quarters alike
    elif hmax <= 180.0 + 1e-6:
        fold = np.where(phi > 180.0, 360.0 - phi, phi)           # halves alike
    else:
        fold = phi
    fold = np.clip(fold, ha[0], ha[-1]) if len(ha) > 1 else fold
    # per column: the vertical slice at that horizontal angle (between the file's two nearest)
    cols = np.empty((len(phi), len(va)))
    for j, a in enumerate(fold):
        if len(ha) == 1:
            cols[j] = cd[0]
        else:
            k = int(np.clip(np.searchsorted(ha, a) - 1, 0, len(ha) - 2))
            t = 0.0 if ha[k + 1] == ha[k] else (a - ha[k]) / (ha[k + 1] - ha[k])
            cols[j] = cd[k] * (1.0 - t) + cd[k + 1] * t
    table = np.zeros((PROF_V, PROF_H))
    for j in range(PROF_H):
        table[:, j] = np.interp(g, va, cols[j], left=0.0, right=0.0) if len(va) > 1 else cols[j][0]
        if len(va) > 1:
            inside = (g >= va[0] - 1e-6) & (g <= va[-1] + 1e-6)
            table[~inside, j] = 0.0
    peak = float(table.max())
    lumens = _flux(table)
    if peak <= 0.0:
        raise ValueError('IES file: the light is dark in every direction')
    return Profile((table / peak).astype(np.float32), peak, lumens, name)


def _flux(table):
    """The luminous flux of a table of candela (lumens): its intensity over the sphere of directions."""
    g = np.radians(np.linspace(0.0, 180.0, table.shape[0]))
    ring = table.mean(1)                                          # (the mean round each vertical angle)
    # integrate I sin(g) dg dphi = 2 pi int ring(g) sin g dg (trapezoid)
    return float(2.0 * math.pi * np.trapezoid(ring * np.sin(g), g)) if hasattr(np, 'trapezoid') else \
        float(2.0 * math.pi * np.trapz(ring * np.sin(g), g))


@lru_cache(maxsize=16)
def _load(path, mtime):
    return parse(Path(path).read_text(encoding='latin-1'), Path(path).stem)


def load(path) -> Profile:
    """The light profile in an .ies file (cached while the file is unchanged)."""
    p = Path(path)
    return _load(str(p), p.stat().st_mtime)


def frame(aim, spin_deg=0.0):
    """The light's own axes round its aim, as the renderers take them: (width axis, height axis), both across the aim.
    A profile's horizontal angle 0 lies along the width axis; an area light's panel spans them. Spin turns them about
    the aim."""
    a = np.asarray(aim, float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    t0 = np.cross(ref, a)
    t0 /= max(float(np.linalg.norm(t0)), 1e-12)
    t1 = np.cross(a, t0)
    s = math.radians(float(spin_deg))
    return t0 * math.cos(s) + t1 * math.sin(s), -t0 * math.sin(s) + t1 * math.cos(s)


def value(table, aim, axes, w):
    """The profile's shape (0..1) along unit direction w from the light (as the shaders read it: bilinear)."""
    a = np.asarray(aim, float)
    g = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(w, a))))))
    ph = math.degrees(math.atan2(float(np.dot(w, axes[1])), float(np.dot(w, axes[0])))) % 360.0
    gv = g / 180.0 * (PROF_V - 1)
    hv = ph / 360.0 * PROF_H
    i0 = int(gv)
    i1 = min(i0 + 1, PROF_V - 1)
    fv = gv - i0
    j0 = int(hv) % PROF_H
    j1 = (j0 + 1) % PROF_H
    fh = hv - int(hv)
    r0 = table[i0, j0] * (1 - fh) + table[i0, j1] * fh
    r1 = table[i1, j0] * (1 - fh) + table[i1, j1] * fh
    return float(r0 * (1 - fv) + r1 * fv)


def flux_per_peak(kind, cos_outer=-1.0, cos_inner=-1.0, profile=None):
    """Lumens a light gives off per candela of its peak, by its shape: a bare bulb 4 pi, a spot by its cone (smoothstep
    edge, as the shaders take it), a panel pi (Lambertian, one side), a profile by its table."""
    if profile is not None:
        return profile.lumens / profile.peak
    if kind == 'area':
        return math.pi
    if kind == 'spot':
        c = np.linspace(-1.0, 1.0, 4001)
        t = np.clip((c - cos_outer) / max(cos_inner - cos_outer, 1e-9), 0.0, 1.0)
        f = t * t * (3.0 - 2.0 * t)
        return float(2.0 * math.pi * np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(c)))
    return 4.0 * math.pi
