"""OpenColorIO: colour spaces, displays and views from a studio's OCIO config.

Blackbody works in scene-linear Rec.709. With an OCIO config (Composite › OCIO config, or the OCIO
environment variable, or the ACES CG config built into OpenColorIO) it can:
  - show and write display images through any display and view of the config (ACES, AgX, a show
    LUT), baked into a 3D LUT with a log shaper for the viewer and applied exactly on the CPU for
    renders written to disk;
  - read footage in any colour space of the config (camera log, ACEScct...), also through a 3D LUT;
  - write EXRs in another scene-linear space (ACEScg, ACES2065-1).
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger('blackbody.ocio')

# names the scene-linear Rec.709 space goes by in common configs
WORKING_NAMES = ('Linear Rec.709 (sRGB)', 'lin_rec709', 'Utility - Linear - sRGB', 'lin_srgb', 'Linear sRGB',
                 'Linear Rec.709', 'linear', 'Linear', 'scene_linear')
LUT_SIZE = 65
# log2 shaper for scene-linear input: s = (log2(x + 2^LO) - LO) / (HI - LO); x = 0 lands on s = 0
SHAPER_LO = -12.0
SHAPER_HI = 8.0


class OCIOError(ValueError):
    pass


def _ocio():
    try:
        import PyOpenColorIO as OCIO
    except ImportError as ex:  # pragma: no cover
        raise OCIOError('OCIO support needs the opencolorio package (pip install opencolorio).') from ex
    return OCIO


def config_source(path=''):
    """Which config a setting means: the file given, else $OCIO, else OpenColorIO's built-in CG config."""
    if path:
        return str(path)
    return os.environ.get('OCIO') or 'ocio://default'


@lru_cache(maxsize=8)
def _load(source, stamp):
    OCIO = _ocio()
    try:
        return OCIO.Config.CreateFromFile(source)
    except Exception as ex:
        raise OCIOError(f'Cannot read the OCIO config {source}: {ex}') from ex


def load_config(path=''):
    src = config_source(path)
    stamp = os.path.getmtime(src) if not src.startswith('ocio://') and Path(src).exists() else 0.0
    if not src.startswith('ocio://') and not Path(src).exists():
        raise OCIOError(f'OCIO config not found: {src}')
    return _load(src, stamp)


def colour_spaces(cfg):
    return [c.getName() for c in cfg.getColorSpaces()]


def displays(cfg):
    return list(cfg.getDisplays())


def views(cfg, display=''):
    d = display or cfg.getDefaultDisplay()
    return list(cfg.getViews(d))


def looks(cfg):
    return [l.getName() for l in cfg.getLooks()]


def working_space(cfg, name=''):
    """The config's colour space for Blackbody's scene-linear Rec.709 working space."""
    names = colour_spaces(cfg)
    if name:
        if cfg.getColorSpace(name) is None:
            raise OCIOError(f'The OCIO config has no colour space "{name}"')
        return cfg.getColorSpace(name).getName()
    lower = {n.lower(): n for n in names}
    for n in WORKING_NAMES:
        cs = cfg.getColorSpace(n)   # also resolves roles and aliases
        if cs is not None and n != 'scene_linear':
            return cs.getName()
        if n.lower() in lower:
            return lower[n.lower()]
    raise OCIOError('Cannot find a linear Rec.709 colour space in the OCIO config; set Composite › Working space.')


def _display_view(cfg, display, view):
    d = display or cfg.getDefaultDisplay()
    if d not in displays(cfg):
        raise OCIOError(f'The OCIO config has no display "{d}"')
    v = view or cfg.getDefaultView(d)
    if v not in views(cfg, d):
        raise OCIOError(f'Display "{d}" has no view "{v}"')
    return d, v


def view_processor(cfg, working, display='', view='', look=''):
    """CPU processor from the working space to display-encoded values through a display and view."""
    OCIO = _ocio()
    d, v = _display_view(cfg, display, view)
    t = OCIO.DisplayViewTransform(src=working, display=d, view=v)
    if look:
        g = OCIO.GroupTransform()
        g.appendTransform(OCIO.LookTransform(src=working, dst=working, looks=look))
        g.appendTransform(t)
        t = g
    return cfg.getProcessor(t).getDefaultCPUProcessor()


def space_processor(cfg, src, dst):
    return cfg.getProcessor(src, dst).getDefaultCPUProcessor()


def apply(cpu, rgb):
    """Apply a CPU processor to an (..., 3) array (returns float32)."""
    a = np.ascontiguousarray(np.asarray(rgb, np.float32)[..., :3]).copy()
    flat = a.reshape(-1, 3)
    cpu.applyRGB(flat)
    return flat.reshape(a.shape)


def shaper(x, lo=SHAPER_LO, hi=SHAPER_HI):
    x = np.maximum(np.asarray(x, np.float64), 0.0)
    return (np.log2(x + 2.0 ** lo) - lo) / (hi - lo)


def unshaper(s, lo=SHAPER_LO, hi=SHAPER_HI):
    return np.maximum(2.0 ** (lo + np.asarray(s, np.float64) * (hi - lo)) - 2.0 ** lo, 0.0)


def _lut_inputs(size, log_shaper):
    s = np.linspace(0.0, 1.0, size)
    x = unshaper(s) if log_shaper else s
    b, g, r = np.meshgrid(x, x, x, indexing='ij')   # (z = blue, y = green, x = red)
    return np.stack([r, g, b], -1).astype(np.float32)


def bake(cpu, size=LUT_SIZE, log_shaper=True):
    """A 3D LUT of a processor: (size, size, size, 4) float32 indexed [blue][green][red], over the log
    shaper (scene-linear input) or over 0..1 (display-referred or log-encoded input)."""
    out = apply(cpu, _lut_inputs(size, log_shaper))
    lut = np.ones(out.shape[:3] + (4,), np.float32)
    lut[..., :3] = out
    return lut


def is_linear(cfg, space):
    cs = cfg.getColorSpace(space)
    if cs is None:
        raise OCIOError(f'The OCIO config has no colour space "{space}"')
    enc = (cs.getEncoding() or '').lower()
    if enc:
        return enc == 'scene-linear'
    return 'lin' in cs.getName().lower() or cs.getName().startswith('ACES2065') or cs.getName() == 'ACEScg'


class ColourPipeline:
    """The OCIO side of a scene's composite settings, with its LUTs baked once per change."""

    def __init__(self, comp_data):
        c = comp_data
        self.cfg = load_config(c.get('ocio_config', ''))
        self.working = working_space(self.cfg, c.get('ocio_working', ''))
        self.display, self.view = _display_view(self.cfg, c.get('ocio_display', ''), c.get('ocio_view', ''))
        self.look = c.get('ocio_look', '') or ''
        self._view_cpu = None
        self._plate_cpu = {}

    def view_cpu(self):
        if self._view_cpu is None:
            self._view_cpu = view_processor(self.cfg, self.working, self.display, self.view, self.look)
        return self._view_cpu

    def to_display(self, rgb):
        """Scene-linear working-space pixels to display-encoded values (0..1), exactly."""
        return np.clip(apply(self.view_cpu(), rgb), 0.0, 1.0)

    def view_lut(self, size=LUT_SIZE):
        return bake(self.view_cpu(), size, True)

    def plate_lut(self, space, size=LUT_SIZE):
        """LUT taking footage in `space` to the working space, and whether it uses the log shaper."""
        lin = is_linear(self.cfg, space)
        cpu = space_processor(self.cfg, space, self.working)
        return bake(cpu, size, lin), lin

    def to_space(self, rgb, space):
        """Working-space pixels to another colour space (for EXRs)."""
        if not space or space == self.working:
            return np.asarray(rgb, np.float32)
        cpu = self._plate_cpu.get(('out', space))
        if cpu is None:
            cpu = self._plate_cpu[('out', space)] = space_processor(self.cfg, self.working, space)
        return apply(cpu, rgb)


_pipes = {}


def pipeline(comp_data):
    """A cached ColourPipeline for a scene's composite settings (None when OCIO is not in use)."""
    keys = ('ocio_config', 'ocio_working', 'ocio_display', 'ocio_view', 'ocio_look')
    uses = (comp_data.get('view') == 'ocio' or comp_data.get('plate_transform') == 'ocio' or comp_data.get('exr_space'))
    if not uses:
        return None
    src = config_source(comp_data.get('ocio_config', ''))
    stamp = os.path.getmtime(src) if (not src.startswith('ocio://') and Path(src).exists()) else 0.0
    key = tuple(comp_data.get(k, '') for k in keys) + (stamp,)
    p = _pipes.get(key)
    if p is None:
        p = _pipes[key] = ColourPipeline(comp_data)
        if len(_pipes) > 8:
            _pipes.pop(next(iter(_pipes)))
    return p
