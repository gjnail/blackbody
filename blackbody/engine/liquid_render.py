"""Liquid rendering: a smooth surface from the particles, then ray tracing it over the footage.

The surface is a signed distance on a grid finer than the simulation (Surface detail, 1-3x),
built from the packed particles in z slabs so its accumulators stay small, then blurred. The ray
tracer (liq_march.wgsl) writes the same beauty / emission / aux buffers as the fire renderer, so
sampling, bloom, compositing and every output work unchanged.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from . import camera as cam
from .gpu import BU, GPU, SS, Uniforms, ceil_div, groups_1d, kernel_spec, load_wgsl
from .liquid import PACKED_BYTES, WW_PACKED_BYTES
from .solver import MAX_COLLIDERS, pack_colliders

SLAB_NODES = 8 << 20   # surface-grid nodes per accumulation slab

# the march's resources (liq_march.wgsl), and how to build it (precompile.py compiles it ahead of time: the one slow
# kernel of the liquids)
MARCH_BINDINGS = ['tex3d', 'tex2d', 'utex2d', 'smp', 'st2d:rgba16float:w', 'st2d:rgba16float:w', 'st2d:rgba16float:w',
                  'tex3d', 'utex3d', 'tex2d', 'tex2d', 'smp', 'st2d:rgba16float:w', 'tex3d', 'tex3d', 'tex2d', 'tex2d',
                  'tex3d', 'tex3d', 'rbuf', 'rbuf', 'tex2d', 'buf', 'rbuf', 'utex2d', 'tex2d']


def march_spec():
    return kernel_spec('liq_march.wgsl', MARCH_BINDINGS, workgroup=(8, 8, 1))


def glow_colour(kelvin):
    """Linear Rec.709 colour of a blackbody at `kelvin`, luminance 1 at 1300 K and rising with the
    temperature (as T^4, softened), for molten liquids."""
    from .lut import blackbody_lut
    rgb = blackbody_lut(3, max(600.0, kelvin - 1.0), kelvin + 1.0)[1, :3].astype(float)
    return rgb * (max(kelvin, 600.0) / 1300.0) ** 3
ACC_BYTES = 48         # accumulator bytes per node (12 slots)


_LAVA_TABLE = {}


def lava_table(t0, dt, n, response='camera'):
    """(n, 4): blackbody colour (linear Rec.709, luminance 1) and log10 of its luminance relative to 1300 K,
    at t0, t0 + dt, ... K, for the march's molten surfaces (liq_lava.wgsl)."""
    key = (t0, dt, n, response)
    if key not in _LAVA_TABLE:
        from .lut import blackbody_lut, luminance
        lut = blackbody_lut(n, t0, t0 + dt * (n - 1), response=response).astype(np.float64)
        T = t0 + dt * np.arange(n)
        # (below 900 K the colour is a deep red too dim to see; the response's gamut mapping tints it blue there)
        cold = T < 900.0
        lut[cold, :3] = blackbody_lut(3, 899.0, 901.0, response=response)[1, :3]
        lut[:, 3] = np.log10(np.maximum([luminance(t) for t in T], 1e-300) / luminance(1300.0))
        _LAVA_TABLE[key] = [tuple(float(x) for x in r) for r in lut]
    return _LAVA_TABLE[key]


@dataclass
class WaterLook:
    ior: float = 1.333
    color: tuple = (0.70, 0.90, 0.93)   # what white light keeps after travelling `clarity` metres
    clarity: float = 3.0                # metres
    murk: float = 0.0                   # scattering extinction (1/m)
    murk_color: tuple = (0.55, 0.65, 0.60)
    roughness: float = 0.06
    reflection: float = 1.0
    reflect_footage: float = 0.75
    ripple: float = 0.25
    ripple_freq: float = 25.0           # 1/m
    ripple_speed: float = 2.0           # m/s for full ripples
    radius: float = 1.0                 # particle radius, in particle spacings
    influence: float = 2.0              # kernel radius, in particle radii
    smoothing: int = 1
    calm: int = 12                      # passes of smoothing along the surface, in dense liquid
    surface_res: float = 2.0
    wet_darken: float = 0.5
    wet_gloss: float = 0.5
    shadow: float = 1.0
    backdrop: float = 15.0              # metres behind the liquid that refracted rays see
    sky: tuple = (0.55, 0.62, 0.72)
    sun: tuple = (0.0, 0.0, 0.0)        # colour x intensity
    sun_azimuth: float = 35.0
    sun_elevation: float = 40.0
    sun_size: float = 0.02              # apparent radius (rad), softens glints
    exposure: float = 0.0
    whitewater: bool = True
    foam: float = 1.0                   # coverage per unit foam density
    spray: float = 0.35                 # optical depth per unit spray density per cell
    bubbles: float = 0.6                # optical depth per unit bubble density per cell
    foam_color: tuple = (0.92, 0.94, 0.95)
    foam_scale: float = 0.012           # m, cells of the foam lace
    foam_lace: float = 0.6
    droplets: float = 1.0
    droplet_size: float = 0.003         # m
    sheets: float = 0.6                 # anisotropic stretch of the particle kernels
    caustics: float = 1.0
    colliders_look: str = 'holdout'     # 'holdout' (in the footage) or 'shaded' (grey stand-ins)
    environment: str = ''               # lat-long HDRI file
    env_rotation: float = 0.0           # degrees
    env_strength: float = 1.0
    env_sun: bool = False
    sky_image: object = None            # the physical sky (engine/sky.py): (key, HDRI), in place of an environment file
    wind: tuple = (0.0, 0.0, 0.0)       # m/s, fire-local
    whitecaps: float = 1.0              # foam where the sea's crests fold over
    sea_foam: float = 1.0               # how much of the sea's foam shows
    sea_foam_life: float = 12.0         # s, how long the thin foam a whitecap leaves lasts
    foam_streaks: float = 0.7           # the thin foam drawn out along the wind into streaks
    crest_glow: float = 1.0             # sunlight through the thin tops of waves
    crest_glow_color: tuple = (0.1, 0.55, 0.45)
    gusts: float = 0.3                  # patches of stronger wind roughening the ripples
    standin_color: tuple = (0.3, 0.3, 0.3)   # colour of colliders drawn as stand-ins
    bottomless: bool = False            # deep water: the bottom is out of sight (the sea, a lake)
    rainbow: float = 1.0                # the rainbow sunlight makes in spray (1 physical)
    lens_drops: float = 0.0             # drops left on the lens by spray, splashes and rain
    rain: float = 0.0                   # mm/h falling through the view and ringing the water
    rain_drop: float = 2.5              # mm, raindrop diameter
    glow: float = 0.0                   # incandescence of a molten liquid (radiance at glow_temp)
    glow_temp: float = 1300.0           # K
    crust: float = 0.7                  # share of a molten surface covered by dark crust
    crust_scale: float = 0.08           # m, crust plate size
    crust_time: float = 1.5             # s: how long a molten surface takes to skin over and darken
    ropes: float = 1.0                  # how much squeezed crust wrinkles into ropes
    crust_color: tuple = (0.075, 0.073, 0.071)  # albedo of the cooled crust (glassy grey rind)
    crust_kind: str = 'skin'            # 'skin' (pahoehoe: a skin the glow shows through, torn where pulled) or
                                        # 'plates' (a lava lake's or a channel's crust)
    crust_roughness: float = 0.45
    lava_light: float = 1.0             # how strongly the glow lights the ground and footage around it
    lava_cooling: float = 0.0           # 1/s: the simulation's cooling rate (Liquid > Cooling)
    ice: bool = False                   # the liquid has heat (it can freeze): draw its ice (liq_ice_look.wgsl)
    frost: float = 1.0                  # how white frost and rime make ice frozen fast or left below freezing
    ice_cloud: float = 1.0              # how milky cloudy ice looks inside (clear ice stays clear)
    frost_color: tuple = (0.9, 0.93, 0.97)
    crystal_size: float = 0.02          # m, the ice's crystals (its facets and frost feathers)
    ice_melting: float = 0.0            # 0..1: the air is above freezing, so ice is wet and glossy
    step_out: float = 0.35              # surface cells
    step_in: float = 0.6
    max_steps: int = 640
    events: int = 6


@dataclass
class LiquidView:
    """What the renderer needs from a liquid frame (live or cached)."""
    packed: object      # Buffer of packed particles
    count: int
    wet: object         # Texture2D (nx, nz) r32float
    dims: tuple
    h: float
    origin: tuple
    ppc: int
    ww: object = None   # Buffer of packed whitewater particles
    ww_count: int = 0
    open: tuple = (True, True, False)   # sides, top, bottom open (False: solid)
    colliders: tuple = ()               # ColliderGPU list at this frame
    meshes: object = None               # MeshLibrary holding the mesh atlas
    level: float = 0.0                  # open water level (m), 0 = off
    level_blend: float = 8.0            # cells over which the box's surface blends into the open water
    ocean: object = None                # OceanSpec of the sea on the open water (None: flat)
    time: float = 0.0                   # s, the shot's time of this frame (the sea's waves run on it)
    band: object = None                 # narrow band: Buffer of deep-liquid flags (u32 per cell), or None
    dye: object = None                  # Buffer of the packed particles' dye (8 bytes each), or None
    crust: object = None                # a molten liquid's crust field (Texture3D, liquid.crust_field), or None
    ice: object = None                  # Buffer of the packed particles' ice (u32 each: frozen share, cloudiness), or None
    current: tuple = (0.0, 0.0, 0.0)    # m/s, fire-local: the open water's current
    layer: object = None                # the open water round the box (ocean_layer: OceanLayer or LayerView), or None
    sea_sides: tuple = (1.0, 1.0, 1.0, 1.0)   # the box's sides the open water flows through (-x, +x, -z, +z)
    liquid_top: float = -1.0            # cells: the highest particle (for skipping the empty air above it), -1 unknown


def liquid_top_of(packed, dims):
    """Highest particle of a packed particle array (n, 4 uint32), in cells, or -1 without particles."""
    if packed is None or not len(packed):
        return -1.0
    y = (np.asarray(packed)[:, 0] >> 16).max()
    return float(y) / 65535.0 * float(dims[1])


def rain_rings(rate_mm_h, drop_mm, life=0.6, speed=0.25):
    """The march's rain-ring uniforms: impact cell size (m), ring life (s), ring speed (m/s), slope.
    Impacts come one per cell per life on each of two lattices; in heavy rain the rings are cut short
    so none outgrows its cell."""
    if rate_mm_h <= 0.0:
        return 0.0, 1.0, speed, 0.0
    from .liquid import rain_drops_per_m2s
    n = max(rain_drops_per_m2s(rate_mm_h, drop_mm), 1e-6)
    cell = math.sqrt(2.0 / (n * life))
    if speed * life > cell:
        cell = (2.0 * speed / n) ** (1.0 / 3.0)
        life = cell / speed
    return cell, life, speed, 0.35 * min(1.0, drop_mm / 2.5)


MAX_LAMPS = 8


def pack_lamps(u: Uniforms, lamps, gain=1.0, fire_gain=None):
    """The march's lamp uniforms: count, fire lights on, their gain, and 8 lamps of 4 vec4."""
    from .renderer import LAMP_SCALE
    lamps = list(lamps or [])[:MAX_LAMPS]
    u.v4(len(lamps), 1.0 if fire_gain is not None else 0.0)
    u.v4(*(np.asarray(fire_gain if fire_gain is not None else (0.0, 0.0, 0.0), float) * gain), 0.0)
    kinds = {'point': 0.0, 'spot': 1.0, 'area': 2.0}
    for i in range(MAX_LAMPS):
        if i < len(lamps):
            L = lamps[i]
            panel = L['kind'] == 'area' and float(L.get('width', 0.0)) > 0.0 and float(L.get('height', 0.0)) > 0.0
            r = math.sqrt(float(L['width']) * float(L['height']) / math.pi) if panel else float(L['radius'])
            prof = L.get('profile')
            along = float(prof.table[0].mean()) if prof is not None else 1.0   # (a profile: its light along its aim)
            u.v4(*L['position'], max(r, 1e-3))
            u.v4(*(np.asarray(L['power'], float) * (LAMP_SCALE * gain * along)), kinds.get(L['kind'], 0.0))
            u.v4(*L['direction'], L['cos_outer'])
            u.v4(L['cos_inner'], 0.0, 0.0, 0.0)
        else:
            for _ in range(4):
                u.v4(0.0, 0.0, 0.0, 0.0)


@dataclass
class LumeWater:
    """The water Lume traces (stage.wgsl, lume_water.wgsl): its surface, whitewater, caustic map and wet ground
    (textures), and its uniforms (wat[0..11], 48 floats)."""
    surf: object
    ww: object
    caus: object
    wet: object
    data: list
    dye: object = None          # the dye field (liq_dye_resolve.wgsl), when the liquid has one (wat[11].z)


class LiquidRenderer:
    def __init__(self, gpu: GPU, renderer):
        self.gpu = gpu
        self.renderer = renderer
        g = gpu
        self.k_count = g.kernel('liq_surf_count.wgsl', ['rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_deep = g.kernel('liq_surf_deep.wgsl', ['rbuf', 'buf'])
        self.k_splat = g.kernel('liq_surf_splat.wgsl', ['rbuf', 'buf', 'rbuf', 'rbuf', 'rbuf', 'buf'],
                                workgroup=(64, 1, 1))
        self.k_mom = g.kernel('liq_surf_mom.wgsl', ['rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_band = g.kernel('liq_surf_band.wgsl', ['rbuf', 'buf'])
        self.band_buf = None
        self._band_cells = 0
        self.k_aniso = g.kernel('liq_surf_aniso.wgsl', ['rbuf', 'buf'], workgroup=(4, 4, 4))
        self.mom = None
        self.aniso = None
        self._blocks = 0
        self.k_resolve = g.kernel('liq_surf_resolve.wgsl', ['rbuf', 'st3d:rgba16float:w', 'st3d:rgba16float:w', 'rbuf',
                                                                'st3d:rgba16float:w', 'rbuf'])
        # ice (liquid_thermal.py): its frozen share on the surface grid, its cloudiness in the dye field,
        # and its edges kept through the smoothing
        self.k_ice_dye = g.kernel('liq_ice_dye.wgsl', ['rbuf', 'rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_ice_keep = g.kernel('liq_ice_keep.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba16float:w'])
        self._no_pice = g.buffer(16, 'liq-no-ice')
        self._no_iacc = g.buffer(16, 'liq-no-ice-acc')
        self._iacc = None
        self._iacc_nodes = 0
        self._surf_raw = None
        self.ice_buf = None
        self._ice_cap = 0
        self._ice_look = None
        # frost and crystal sparkle over the ice the march drew (liq_ice_shade.wgsl), from copies of its output
        self.k_ice_shade = g.kernel('liq_ice_shade.wgsl', ['tex3d', 'tex3d', 'smp', 'rbuf', 'tex2d', 'tex2d',
                                                           'st2d:rgba16float:w', 'st2d:rgba16float:w', 'tex2d', 'tex2d',
                                                           'tex3d'], workgroup=(8, 8, 1))
        self._ice_tmp = None
        self.k_ww = g.kernel('liq_surf_ww.wgsl', ['rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_blur = g.kernel('liq_surf_blur.wgsl', ['utex3d', 'st3d:rgba16float:w'])
        self.k_flatten = g.kernel('liq_surf_flatten.wgsl', ['tex3d', 'utex3d', 'smp', 'st3d:rgba16float:w'])
        self._k_march = None   # (compiled when the first frame is drawn: k_march)
        self._no_cloth = g.texture2d(1, 1, 'rgba16float', 'liq-no-cloth')   # (fabric: cloth.py layer)
        g.upload(self._no_cloth, np.full((1, 1, 4), -1.0, np.float16))
        # the footage with what stands in it filled in from beside it, for rays that reach the ground behind it
        # (liq_clean.wgsl); a 1x1 that covers nothing without footage or nothing standing in it
        clean = ['tex2d', 'tex2d', 'utex3d', 'smp', 'utex2d', 'st2d:r32float:w', 'st2d:rgba16float:w']
        self.k_clean_cover = g.kernel('liq_clean.wgsl', clean, 'cover', workgroup=(8, 8, 1))
        self.k_clean_fill = g.kernel('liq_clean.wgsl', clean, 'fill', workgroup=(8, 8, 1))
        self._no_clean = g.texture2d(1, 1, 'rgba16float', 'liq-no-clean')
        g.upload(self._no_clean, np.full((1, 1, 4), 60000.0, np.float16))
        self._no_near = g.texture2d(1, 1, 'r32float', 'liq-no-near')
        g.upload(self._no_near, np.full((1, 1, 1), 60000.0, np.float32))
        self._clean = None
        self.k_lens = g.kernel('liq_lens_drops.wgsl', ['tex2d', 'tex2d', 'smp', 'st2d:rgba8unorm:w', 'st2d:rgba16float:w'],
                               workgroup=(8, 8, 1))
        self._lens_tmp = None
        self._no_lights = g.buffer(32, 'liq-no-fire-lights')
        self._no_count = g.buffer(16, 'liq-no-fire-light-count')
        g.write_buffer(self._no_count, np.zeros(4, np.uint32))
        self.k_dye = g.kernel('liq_dye_splat.wgsl', ['rbuf', 'rbuf', 'buf'], workgroup=(64, 1, 1))
        self.k_dye_res = g.kernel('liq_dye_resolve.wgsl', ['rbuf', 'st3d:rgba16float:w'])
        self._no_dye = g.texture3d((1, 1, 1), 'rgba16float', 'liq-no-dye')
        g.upload(self._no_dye, np.zeros((1, 1, 1, 4), np.float16))
        self.dye_tex = self._no_dye
        self._dye_dims = None
        self._dye_acc = None
        self.dye_buf = None
        self._dye_cap = 0
        # a molten liquid's crust field (liq_crust_adv.wgsl): the frame's, or a cached frame's uploaded here
        self._crust_cached = None
        self._no_crust_field = g.texture3d((1, 1, 1), 'rgba32float', 'liq-no-crust-field')
        g.upload(self._no_crust_field, np.zeros((1, 1, 1, 4), np.float32))
        # the march's surface hits for the passes after it (liq_lava.wgsl): 3 vec4 per pixel
        self.gbuf = None
        self.k_lava_shade = g.kernel('liq_lava_shade.wgsl', ['tex3d', 'tex3d', 'smp', 'utex3d', 'rbuf', 'st2d:rgba16float:w',
                                                          'st2d:rgba16float:w', 'tex2d', 'tex2d', 'smp'], workgroup=(8, 8, 1))
        # the light a molten liquid's glow casts on its surroundings (liq_lava_cols.wgsl, liq_lava_lights.wgsl)
        self.k_lava_cols = g.kernel('liq_lava_cols.wgsl', ['utex3d', 'utex3d', 'buf', 'utex3d'], workgroup=(8, 8, 1))
        self.k_lava_lights = g.kernel('liq_lava_lights.wgsl', ['rbuf', 'buf'], workgroup=(8, 8, 1))
        self.k_lava_map = g.kernel('liq_lava_map.wgsl', ['rbuf', 'buf'], workgroup=(8, 8, 1))
        self._lava_map = None
        self._lava_geom = (0.0, 0.0, 1.0, 1.0, 0, 0)
        self._no_lava_lights = g.buffer(48, 'liq-no-lava-lights')
        g.write_buffer(self._no_lava_lights, np.zeros(12, np.float32))
        self._lava_cols = None
        self._lava_lights = None
        # a molten skin's lumps, moved into the surface (liq_lava_disp.wgsl)
        self.k_lava_disp = g.kernel('liq_lava_disp.wgsl', ['tex3d', 'utex3d', 'st3d:rgba16float:w'], workgroup=(4, 4, 4))
        # heat haze over a molten liquid: the air it heats (liq_lava_air.wgsl), traced by the renderer's haze pass
        self.k_lava_air = g.kernel('liq_lava_air.wgsl', ['utex3d', 'utex3d', 'utex3d', 'st3d:rgba16float:w'],
                                   workgroup=(4, 4, 4))
        self._lava_air = None
        self._air_rise = g.texture3d((2, 2, 2), 'rgba16float', 'liq-lava-air-rise')
        g.upload(self._air_rise, np.tile(np.array([0.0, self.LAVA_AIR_RISE, 0.0, 0.0], np.float16), (2, 2, 2, 1)))
        self.crust_field = self._no_crust_field
        self.k_caus = g.kernel('liq_caustics.wgsl', ['tex3d', 'smp', 'buf', 'tex3d', 'tex3d', 'tex2d', 'tex2d', 'smp'],
                               workgroup=(8, 8, 1))
        from .ocean import Sea
        self.ocean = Sea(g)
        self.k_caus_res = g.kernel('liq_caustics_resolve.wgsl', ['rbuf', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        self.k_lume_clear = g.kernel('liq_lume_clear.wgsl', ['st2d:rgba16float:w'] * 4)
        self._one = g.texture2d(1, 1, 'rgba16float', 'liq-one')
        g.upload(self._one, np.ones((1, 1, 4), np.float16))
        self._no_env = g.texture2d(1, 1, 'rgba16float', 'liq-no-env')
        g.upload(self._no_env, np.zeros((1, 1, 4), np.float16))
        self._no_atlas = g.texture3d((1, 1, 1), 'r32float', 'liq-no-atlas')
        g.upload(self._no_atlas, np.full((1, 1, 1, 1), 1.0e6, np.float32))
        self.caus = None
        self.caus_tex = self._one
        self._caus_acc = None
        self._caus_dims = None
        self.env_tex = None
        self._env_key = None
        self.env_info = None
        self.surf = None
        self._surf = []
        self.nf = None
        self.acc = None
        self._acc_nodes = 0
        self.occ = None
        self.deep = None
        self._cells = 0
        self.cache_buf = None
        self._cache_cap = 0
        self.ww_buf = None
        self._ww_cap = 0
        self.ww_tex = None
        self._black = g.texture2d(4, 4, 'rgba8unorm', 'liq-black')
        g.upload(self._black, np.zeros((4, 4, 4), np.uint8))
        self.sig = None
        self._drops_pipeline()

    @property
    def k_march(self):
        """The march (liq_march.wgsl), compiled when the first frame is drawn rather than when the scene is set up:
        the simulation starts meanwhile, and the background precompile (precompile.py) may have it ready by then."""
        if self._k_march is None:
            s = march_spec()
            self._k_march = self.gpu.kernel(s['file'], s['bindings'], s['entry'], None, tuple(s['workgroup']))
        return self._k_march

    @k_march.setter
    def k_march(self, k):
        self._k_march = k

    def _drops_pipeline(self):
        """Spray droplets: streaks drawn over the traced image (premultiplied over the beauty, their
        glints added to the emission that feeds the bloom)."""
        dev = self.gpu.device
        vis = SS.VERTEX | SS.FRAGMENT
        self._drop_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': vis, 'buffer': {'type': 'read-only-storage'}},
            {'binding': 1, 'visibility': vis, 'texture': {'sample_type': 'float', 'view_dimension': '2d'}},
        ])
        module = dev.create_shader_module(code=load_wgsl('liq_drops.wgsl'), label='liq_drops')
        over = {'color': {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'},
                'alpha': {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}}
        add = {'color': {'src_factor': 'one', 'dst_factor': 'one', 'operation': 'add'},
               'alpha': {'src_factor': 'zero', 'dst_factor': 'one', 'operation': 'add'}}
        # the mattes: droplet coverage goes to w (over), the rest is left as it is
        matte = {'color': {'src_factor': 'zero', 'dst_factor': 'one', 'operation': 'add'},
                 'alpha': {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}}
        self._drop_pipe = dev.create_render_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._drop_layout, self.gpu.arena.layout]),
            vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
            fragment={'module': module, 'entry_point': 'fs', 'targets': [
                {'format': 'rgba16float', 'blend': over}, {'format': 'rgba16float', 'blend': add},
                {'format': 'rgba16float', 'blend': matte}]},
            primitive={'topology': 'triangle-strip'}, label='liq-drops')
        self._drop_bg = None
        self._drop_key = None
        # rain: the same streaks, placed through the view instead of read from particles
        self._rain_layout = dev.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': vis, 'texture': {'sample_type': 'float', 'view_dimension': '2d'}},
        ])
        module = dev.create_shader_module(code=load_wgsl('liq_rain_streaks.wgsl'), label='liq_rain_streaks')
        self._rain_pipe = dev.create_render_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[self._rain_layout, self.gpu.arena.layout]),
            vertex={'module': module, 'entry_point': 'vs', 'buffers': []},
            fragment={'module': module, 'entry_point': 'fs', 'targets': [
                {'format': 'rgba16float', 'blend': over}, {'format': 'rgba16float', 'blend': add},
                {'format': 'rgba16float', 'blend': matte}]},
            primitive={'topology': 'triangle-strip'}, label='liq-rain')
        self._rain_bg = None
        self._rain_key = None

    # -- resources -------------------------------------------------------------------------------

    def upload_particles(self, packed: np.ndarray):
        """A buffer holding cached packed particles (for re-rendering cached frames)."""
        n = max(1, len(packed))
        if self.cache_buf is None or self._cache_cap < n:
            if self.cache_buf is not None:
                self.cache_buf.destroy()
            self._cache_cap = int(n * 1.25) + 1024
            self.cache_buf = self.gpu.buffer(self._cache_cap * PACKED_BYTES, 'liq-cached-particles')
        if len(packed):
            self.gpu.write_buffer(self.cache_buf, np.ascontiguousarray(packed, np.uint32))
        return self.cache_buf

    def upload_dye(self, dye: np.ndarray):
        """A buffer holding a cached frame's particle dye (parallel to upload_particles)."""
        n = max(1, len(dye))
        if self.dye_buf is None or self._dye_cap < n:
            if self.dye_buf is not None:
                self.dye_buf.destroy()
            self._dye_cap = int(n * 1.25) + 1024
            self.dye_buf = self.gpu.buffer(self._dye_cap * 8, 'liq-cached-dye')
        if len(dye):
            self.gpu.write_buffer(self.dye_buf, np.ascontiguousarray(dye, np.uint32))
        return self.dye_buf

    def upload_ice(self, ice: np.ndarray):
        """A buffer holding a cached frame's particle ice (u32 each; liquid.read_ice, parallel to upload_particles)."""
        n = max(1, len(ice))
        if self.ice_buf is None or self._ice_cap < n:
            if self.ice_buf is not None:
                self.ice_buf.destroy()
            self._ice_cap = int(n * 1.25) + 1024
            self.ice_buf = self.gpu.buffer(self._ice_cap * 4, 'liq-cached-ice')
        if len(ice):
            self.gpu.write_buffer(self.ice_buf, np.ascontiguousarray(ice, np.uint32))
        return self.ice_buf

    def upload_crust(self, crust: np.ndarray):
        """A texture holding a cached frame's crust field ((nz, ny, nx, 4) float32; liquid.read_crust)."""
        nz, ny, nx = crust.shape[:3]
        if self._crust_cached is None or tuple(self._crust_cached.size) != (nx, ny, nz):
            if self._crust_cached is not None:
                self._crust_cached.destroy()
            self._crust_cached = self.gpu.texture3d((nx, ny, nz), 'rgba32float', 'liq-cached-crust')
        self.gpu.upload(self._crust_cached, np.ascontiguousarray(crust, np.float32))
        return self._crust_cached

    def _build_dye(self, b, view: LiquidView):
        """The dye field on the simulation grid, from the particles (or none)."""
        count = int(view.count)
        ice = view.ice is not None and self._ice_look is not None and self._ice_look.ice
        if (view.dye is None and not ice) or count == 0:
            self.dye_tex = self._no_dye
            return
        n = tuple(view.dims)
        if n != self._dye_dims:
            for x in (self._dye_acc, self.dye_tex if self.dye_tex is not self._no_dye else None):
                if x is not None:
                    x.destroy()
            self._dye_acc = self.gpu.buffer(int(np.prod(n)) * 5 * 4, 'liq-dye-acc')
            self._dye_field = self.gpu.texture3d(n, 'rgba16float', 'liq-dye-field')
            self._dye_dims = n
        b.clear_buffer(self._dye_acc)
        if view.dye is not None:
            b.run(self.k_dye, [view.packed, view.dye, self._dye_acc], Uniforms().v4(*n, count), groups=groups_1d(count))
        if ice:
            # clear ice scatters a few per metre (bubbles), white ice a few hundred (opaque in a centimetre or two)
            gain = max(self._ice_look.ice_cloud, 0.0)
            b.run(self.k_ice_dye, [view.packed, view.ice, self._dye_acc],
                  Uniforms().v4(*n, count).v4(1.0 if view.dye is None else 0.0, 3.0 * gain, 300.0 * gain),
                  groups=groups_1d(count))
        b.run(self.k_dye_res, [self._dye_acc, self._dye_field], Uniforms().v4(*n, 0), n)
        self.dye_tex = self._dye_field

    def upload_whitewater(self, packed: np.ndarray):
        n = max(1, len(packed))
        if self.ww_buf is None or self._ww_cap < n:
            if self.ww_buf is not None:
                self.ww_buf.destroy()
            self._ww_cap = int(n * 1.25) + 1024
            self.ww_buf = self.gpu.buffer(self._ww_cap * WW_PACKED_BYTES, 'liq-cached-whitewater')
        if len(packed):
            self.gpu.write_buffer(self.ww_buf, np.ascontiguousarray(packed, np.uint32))
        return self.ww_buf

    def upload_band(self, bits, cells):
        """A cached frame's deep-liquid flags (packed bits) as a u32-per-cell buffer."""
        if self.band_buf is None or self._band_cells != cells:
            if self.band_buf is not None:
                self.band_buf.destroy()
            self.band_buf = self.gpu.buffer(cells * 4, 'liq-cached-band')
            self._band_cells = cells
        flags = np.unpackbits(np.asarray(bits, np.uint8))[:cells].astype(np.uint32)
        self.gpu.write_buffer(self.band_buf, flags)
        return self.band_buf

    def _ensure_surface(self, nf):
        if nf == self.nf:
            return
        for t in self._surf:
            t.destroy()
        self._surf = [self.gpu.texture3d(nf, 'rgba16float', f'liq-surface{i}') for i in range(2)]
        self.ww_tex = self.gpu.texture3d(nf, 'rgba16float', 'liq-whitewater')
        self._surf.append(self.ww_tex)
        self.heat_tex = self.gpu.texture3d(nf, 'rgba16float', 'liq-heat')
        self._surf.append(self.heat_tex)
        self.nf = nf

    def _ensure_acc(self, nodes):
        if nodes <= self._acc_nodes:
            return
        if self.acc is not None:
            self.acc.destroy()
        self.acc = self.gpu.buffer(nodes * ACC_BYTES, 'liq-surface-acc')
        self._acc_nodes = nodes

    @staticmethod
    def surface_dims(dims, f):
        return tuple(max(8, int(round(d * f))) for d in dims)

    @staticmethod
    def _liquid_top(view, lvl_on):
        """ocx[14].x for the march: how high the liquid can reach in the box (cells), so its rays skip the
        empty air above it (in a big box, over a sea, they ran out of steps before they reached the water);
        -1: unknown, march the whole box."""
        if view.liquid_top < 0.0:
            return -1.0
        top = view.liquid_top
        if lvl_on:
            from .ocean import crest_bound
            crest = crest_bound(view.ocean) if (view.ocean is not None and view.ocean.on) else 0.0
            top = max(top, (view.level + crest) / view.h)
        return top + 2.0

    @staticmethod
    def memory_bytes(dims, f):
        nf = LiquidRenderer.surface_dims(dims, f)
        nodes = nf[0] * nf[1] * nf[2]
        blocks = int(np.prod([(x + 1) // 2 for x in dims]))
        return nodes * 24 + min(nodes, SLAB_NODES) * ACC_BYTES + blocks * 72

    # -- surface ----------------------------------------------------------------------------------

    def build(self, b, view: LiquidView, look: WaterLook):
        """Signed distance and velocity of the liquid on the surface grid, from the particles."""
        n = view.dims
        f = max(0.5, float(look.surface_res))
        nf = self.surface_dims(n, f)
        self._ensure_surface(nf)
        fx = nf[0] / n[0]
        spacing = 1.0 / max(view.ppc, 1) ** (1.0 / 3.0)
        r_cells = max(look.radius, 0.3) * spacing
        R_cells = look.influence * r_cells
        bulk = view.ppc * (4.0 / 3.0 * math.pi * R_cells ** 3) * 0.152
        slab = max(1, min(nf[2], SLAB_NODES // (nf[0] * nf[1])))
        self._ensure_acc(nf[0] * nf[1] * slab)
        count = int(view.count)
        self._ice_look = look
        self.crust_field = view.crust if (view.crust is not None and count > 0) else self._no_crust_field
        ice = view.ice is not None and count > 0 and look.ice
        if ice:
            nodes = nf[0] * nf[1] * slab
            if self._iacc is None or self._iacc_nodes < nodes:
                if self._iacc is not None:
                    self._iacc.destroy()
                self._iacc = self.gpu.buffer(nodes * 4, 'liq-ice-acc')
                self._iacc_nodes = nodes
            if self._surf_raw is None or tuple(self._surf_raw.size) != tuple(nf):
                if self._surf_raw is not None:
                    self._surf_raw.destroy()
                self._surf_raw = self.gpu.texture3d(nf, 'rgba16float', 'liq-surface-unsmoothed')
        # the deep inside of the liquid cannot affect the surface: find it and skip its particles
        cells = int(np.prod(n))
        if cells != self._cells:
            for bf in (self.occ, self.deep):
                if bf is not None:
                    bf.destroy()
            self.occ = self.gpu.buffer(cells * 4, 'liq-surface-occupancy')
            self.deep = self.gpu.buffer(cells * 4, 'liq-surface-deep')
            self._cells = cells
        reach = max(1, math.ceil(R_cells))
        # sheets: the local shape of the liquid around each particle block, for ellipsoid kernels
        nb = tuple((x + 1) // 2 for x in n)
        blocks = int(np.prod(nb))
        if blocks != self._blocks:
            for bf in (self.mom, self.aniso):
                if bf is not None:
                    bf.destroy()
            self.mom = self.gpu.buffer(blocks * 40, 'liq-surface-moments')
            self.aniso = self.gpu.buffer(blocks * 32, 'liq-surface-aniso')
            self._blocks = blocks
        sheets = look.sheets > 0.0 and count > 0
        if sheets:
            b.clear_buffer(self.mom)
            b.run(self.k_mom, [view.packed, self.mom], Uniforms().v4(*n, count), groups=groups_1d(count))
            b.run(self.k_aniso, [self.mom, self.aniso], Uniforms().v4(*n).v4(min(1.0, look.sheets), 1.8, 1.0, 12.0), nb)
        b.clear_buffer(self.occ)
        if count:
            b.run(self.k_count, [view.packed, self.occ], Uniforms().v4(*n, count), groups=groups_1d(count))
        full = math.ceil(0.75 * view.ppc)
        if view.band is not None:
            b.run(self.k_band, [view.band, self.occ], Uniforms().v4(*n, 0.0).v4(full), n)
        b.run(self.k_deep, [self.occ, self.deep], Uniforms().v4(*n, reach)
              .v4(full, *(0.0 if o else 1.0 for o in view.open)), n)
        if view.band is not None:
            b.run(self.k_band, [view.band, self.deep], Uniforms().v4(*n, 1.0).v4(full), n)
        dst = self._surf[0]
        for z0 in range(0, nf[2], slab):
            z1 = min(nf[2], z0 + slab)
            u = (Uniforms().v4(*n, count).v4(*nf, r_cells * fx).v4(z0, z1, R_cells * fx, bulk)
                 .v4(reach, 1.0 if sheets else 0.0, 0.0, 1.0 if ice else 0.0))
            b.clear_buffer(self.acc, 0, nf[0] * nf[1] * (z1 - z0) * ACC_BYTES)
            if ice:
                b.clear_buffer(self._iacc, 0, nf[0] * nf[1] * (z1 - z0) * 4)
            if count:
                b.run(self.k_splat, [view.packed, self.acc, self.deep, self.aniso,
                                     view.ice if ice else self._no_pice, self._iacc if ice else self._no_iacc],
                      u, groups=groups_1d(count))
            if view.ww_count and look.whitewater:
                uw = Uniforms().v4(*n, view.ww_count).v4(*nf, 0).v4(z0, z1, max(1.0, 0.5 * fx))
                b.run(self.k_ww, [view.ww, self.acc], uw, groups=groups_1d(view.ww_count))
            b.run(self.k_resolve, [self.acc, dst, self.ww_tex, self.deep, self.heat_tex,
                                   self._iacc if ice else self._no_iacc], u, (nf[0], nf[1], z1 - z0))
        src = dst
        k = 0
        if ice:
            b.copy_texture(dst, self._surf_raw, nf)
        for _ in range(max(0, int(look.smoothing))):
            k += 1
            out = self._surf[k % 2]
            b.run(self.k_blur, [src, out], Uniforms().v4(*nf, 0), nf)
            src = out
        for _ in range(max(0, int(look.calm))):
            k += 1
            out = self._surf[k % 2]
            b.run(self.k_flatten, [src, self.ww_tex, self.gpu.linear, out], Uniforms().v4(*nf, 1.5).v4(1.0), nf)
            src = out
        if ice and k > 0:
            # ice keeps its edges through the smoothing
            k += 1
            out = self._surf[k % 2]
            b.run(self.k_ice_keep, [self._surf_raw, src, self.heat_tex, out],
                  Uniforms().v4(*nf, 1.0 - 0.45 * look.ice_melting), nf)
            src = out
        if look.glow > 0.0 and count and self.crust_field is not self._no_crust_field and look.crust_kind != 'plates':
            # a molten skin's lumps, in the surface itself (liq_lava_disp.wgsl)
            k += 1
            out = self._surf[k % 2]
            u = Uniforms().v4(*n, view.h).v4(*nf, fx).v4(*view.origin, self.LAVA_RELIEF)
            self.lava_uniforms(u, look, 1.0, 0.0, True)
            b.run(self.k_lava_disp, [src, self.crust_field, out], u, nf)
            src = out
        self.surf = src
        self.fscale = fx
        self._build_dye(b, view)

    # -- environment and caustics -------------------------------------------------------------------

    def environment(self, look: WaterLook):
        """Load (or reuse) the HDRI. Returns None, or (sky ambient colour, sun azimuth, elevation,
        colour) measured from it, rotation included."""
        path = look.environment
        img = None
        if not path and look.sky_image is not None:
            key, img = look.sky_image
        elif not path:
            self.env_info = self._env_key = None   # (loaded and measured again when it is back)
            return None
        else:
            import os
            try:
                key = (path, os.path.getmtime(path))
            except OSError:
                self.env_info = self._env_key = None
                return None
        if key != self._env_key:
            from ..io.hdri import brightest, load_hdri, sky_average
            if img is None:
                img = load_hdri(path)
            if self.env_tex is not None:
                self.env_tex.destroy()
            h, w = img.shape[:2]
            self.env_tex = self.gpu.texture2d(w, h, 'rgba16float', 'liq-hdri')
            rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], -1)
            self.gpu.upload(self.env_tex, np.minimum(rgba, 6.0e4).astype(np.float16))
            az, el, colour = brightest(img)
            self.env_info = (sky_average(img), az, el, colour)
            self._env_key = key
        sky, az, el, colour = self.env_info
        return tuple(x * look.env_strength for x in sky), az + look.env_rotation, el, colour

    def sea(self, b, view: LiquidView, look: WaterLook | None = None):
        """The sea's layers and foam for this frame (nothing without waves)."""
        from .ocean import SeaFoam
        foam = wind = None
        if look is not None:
            foam = SeaFoam(whitecaps=look.whitecaps, life=look.sea_foam_life, streaks=look.foam_streaks)
            wind = (look.wind[0], look.wind[2])
        self.ocean.update(b, view.ocean, view.time, view.h, current=(view.current[0], view.current[2]),
                          wind=wind or (0.0, 0.0), foam=foam)
        if self.nf is not None:
            self.ocean.box_map(b, view.origin, view.dims, view.h, (self.nf[0], self.nf[2]), view.level, view.layer,
                               view.sea_sides)

    def caustics(self, b, view: LiquidView, look: WaterLook, fire: cam.FireXform):
        """Trace key-light photons through the liquid onto the ground (the map the renderer reads
        for caustics and shadows). Uses the surface build() left."""
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        if look.caustics <= 0.0 or max(look.sun) <= 0.0 or sd[1] < 0.03 or view.count == 0:
            self.caus_tex = self._one
            return
        nf = self.nf
        dims = (nf[0], nf[2])
        if dims != self._caus_dims:
            for x in (self.caus, self._caus_acc):
                if x is not None:
                    x.destroy()
            self.caus = self.gpu.texture2d(dims[0], dims[1], 'rgba16float', 'liq-caustics')
            self._caus_acc = self.gpu.buffer(dims[0] * dims[1] * 4, 'liq-caustics-acc')
            self._caus_dims = dims
        ry = cam.rot_y(math.radians(fire.yaw))
        sg = ry.T @ sd
        sg = sg / np.linalg.norm(sg)
        absorb = -np.log(np.clip(np.asarray(look.color, float), 1e-4, 1.0)) / max(look.clarity, 1e-3)
        k = 4   # photons per texel along each side
        n = view.dims
        lvl_on = view.level > 0.0 and view.open[0]
        u = (Uniforms().v4(*n, view.h).v4(*nf, self.fscale).v4(dims[0], dims[1], k)
             .v3(sg, look.ior).v4(*absorb, look.murk)
             .v4(view.level / view.h if lvl_on else -1.0, 1.0 if lvl_on else 0.0, 0.0, view.level_blend)
             .v4(*view.origin))
        self.ocean.uniforms(u, view.level, time=view.time, layer=view.layer, sides=view.sea_sides)
        b.clear_buffer(self._caus_acc)
        b.run(self.k_caus, [self.surf, self.gpu.linear, self._caus_acc, *self.ocean.textures(view.layer), self.gpu.repeat], u,
              (dims[0] * k, dims[1] * k, 1))
        b.run(self.k_caus_res, [self._caus_acc, self.caus], Uniforms().v4(dims[0], dims[1], k * k), (dims[0], dims[1], 1))
        self.caus_tex = self.caus

    # -- Lume ----------------------------------------------------------------------------------------

    def lume_ok(self, view: LiquidView, look: WaterLook):
        """Whether Lume can trace this liquid itself (stage.wgsl lume_water.wgsl): clear, murky or dyed water, honey,
        oil. Not yet: a molten one (its glow and crust), ice, the sea's waves; those the march draws over the set Lume lit."""
        return not (look.glow > 0.0 or (look.ice and view.ice is not None)
                    or (view.ocean is not None and getattr(view.ocean, 'on', False))
                    or view.count <= 0)

    def lume_water(self, view: LiquidView, look: WaterLook, time=0.0):
        """The water for Lume (the surface build() made, the caustic map caustics() traced): LumeWater."""
        lvl_on = view.level > 0.0 and view.open[0]
        absorb = -np.log(np.clip(np.asarray(look.color, float), 1e-4, 1.0)) / max(look.clarity, 1e-3)
        caus_on = self.caus_tex is not self._one
        u = Uniforms()
        u.v4(*view.origin, view.h).v4(*view.dims, self.fscale)
        u.v4(*absorb, look.murk).v4(*look.murk_color, look.ior)
        u.v4(1.0, 1.0 if lvl_on else 0.0, view.level / view.h if lvl_on else -1.0, view.level_blend)
        u.v4(look.ripple, look.ripple_freq, look.ripple_speed, time)
        u.v4(*view.sea_sides)
        u.v4(look.spray * (1.0 - 0.7 * min(1.0, look.droplets)), look.foam, look.bubbles,
             1.0 if (look.whitewater and view.ww_count) else 0.0)
        u.v4(*look.foam_color, look.foam_scale)
        u.v4(look.wet_darken, look.wet_gloss, view.current[0] if lvl_on else 0.0, view.current[2] if lvl_on else 0.0)
        u.v4(look.roughness, look.caustics, 1.0 if caus_on else 0.0, 1.0 if look.bottomless else 0.0)
        dye_on = view.dye is not None and self.dye_tex is not self._no_dye
        u.v4(self._liquid_top(view, lvl_on), look.foam_lace, 1.0 if dye_on else 0.0, 0.0)
        return LumeWater(self.surf, self.ww_tex, self.caus_tex, view.wet, list(u.data), self.dye_tex if dye_on else None)

    def lume_clear(self, b, size):
        """The liquid's own element left empty (Lume drew the water into the stage's picture): for the drops, the rain
        and the composite to go over."""
        r = self.renderer
        w, h = size
        r._ensure_fire(w, h)
        b.run(self.k_lume_clear, [r.beauty, r.emit, r.aux, r.mask], Uniforms().v4(w, h), (w, h, 1))

    # -- tracing ----------------------------------------------------------------------------------

    def drops(self, b, view: LiquidView, camstate: cam.CameraState, fire: cam.FireXform, look: WaterLook, size,
              jitter=(0.0, 0.0), shutter=0.0):
        """Draw the spray particles as droplets over what march() left (call right after it)."""
        if look.droplets <= 0.0 or not look.whitewater or not view.ww_count or view.ww is None:
            return
        r = self.renderer
        key = (id(view.ww), view.ww.size, id(r.aux))
        if self._drop_bg is None or self._drop_key != key:
            self._drop_bg = self.gpu.device.create_bind_group(layout=self._drop_layout, entries=[
                {'binding': 0, 'resource': {'buffer': view.ww.buf, 'offset': 0, 'size': view.ww.size}},
                {'binding': 1, 'resource': r.aux.view},
            ])
            self._drop_key = key
        w, h = size
        P = camstate.proj
        focal_px = 0.5 * h * math.sqrt(abs(P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]) * camstate.aspect)
        gain = 2.0 ** look.exposure
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        u = (Uniforms().m4(camstate.view_proj).m4(fire.local_to_world())
             .v4(*view.dims, view.h).v4(*view.origin, look.droplets)
             .v4(w, h, shutter, focal_px)
             .v4(look.droplet_size, 0.8, look.ior, look.rainbow)
             .v4(*camstate.eye, 0.0)
             .v4(*(np.asarray(look.sky) * gain), 0.0)
             .v4(*(np.asarray(look.sun) * gain), 0.0)
             .v3(sd, 0.0)
             .v4(*jitter))
        off = b.uniform_offset(u)
        rp = b.render_pass(color_attachments=[
            {'view': r.beauty.view, 'load_op': 'load', 'store_op': 'store'},
            {'view': r.emit.view, 'load_op': 'load', 'store_op': 'store'},
            {'view': r.mask.view, 'load_op': 'load', 'store_op': 'store'}])
        rp.set_pipeline(self._drop_pipe)
        rp.set_bind_group(0, self._drop_bg)
        rp.set_bind_group(1, self.gpu.arena.group, [off])
        rp.draw(4, view.ww_count)
        rp.end()

    def rain(self, b, camstate: cam.CameraState, fire: cam.FireXform, look: WaterLook, size, jitter=(0.0, 0.0),
             shutter=0.0, frame=0, wind=(0.0, 0.0, 0.0), ground=True):
        """Draw falling rain over what march() left (streaks over the shutter, or over 1/48 s without
        motion blur: rain is seen as streaks)."""
        if look.rain <= 0.0:
            return
        from .liquid import rain_drops_per_m2s, rain_speed
        r = self.renderer
        key = id(r.aux)
        if self._rain_bg is None or self._rain_key != key:
            self._rain_bg = self.gpu.device.create_bind_group(layout=self._rain_layout, entries=[
                {'binding': 0, 'resource': r.aux.view}])
            self._rain_key = key
        w, h = size
        P = camstate.proj
        focal_px = 0.5 * h * math.sqrt(abs(P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]) * camstate.aspect)
        near, far = 0.3, 40.0
        tx, ty = 1.05 / max(abs(P[0, 0]), 1e-6), 1.05 / max(abs(P[1, 1]), 1e-6)
        volume = 4.0 * tx * ty * (far ** 3 - near ** 3) / 3.0
        vt = rain_speed(look.rain_drop)
        density = rain_drops_per_m2s(look.rain, look.rain_drop) / vt      # drops per cubic metre
        want = density * volume
        count = int(min(want, 600000))
        if count < 1:
            return
        l2w = fire.local_to_world()
        wind_w = l2w[:3, :3] @ np.asarray(wind, float)
        fall = wind_w + np.array([0.0, -vt, 0.0])
        gain = 2.0 ** look.exposure
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        ground_y = float(l2w[1, 3])
        u = (Uniforms().m4(camstate.view_proj).m4(camstate.inv_view_proj)
             .v4(*camstate.eye, float(frame % 65536))
             .v4(*fall, look.rain_drop * 1e-3)
             .v4(w, h, shutter if shutter > 0.0 else 1.0 / 48.0, focal_px)
             .v4(near, far, count, want / count)
             .v4(*(np.asarray(look.sky) * gain), ground_y)
             .v4(*(np.asarray(look.sun) * gain), 1.0 if ground else 0.0)
             .v3(sd, 0.0)
             .v4(*jitter))
        off = b.uniform_offset(u)
        rp = b.render_pass(color_attachments=[
            {'view': r.beauty.view, 'load_op': 'load', 'store_op': 'store'},
            {'view': r.emit.view, 'load_op': 'load', 'store_op': 'store'},
            {'view': r.mask.view, 'load_op': 'load', 'store_op': 'store'}])
        rp.set_pipeline(self._rain_pipe)
        rp.set_bind_group(0, self._rain_bg)
        rp.set_bind_group(1, self.gpu.arena.group, [off])
        rp.draw(4, count)
        rp.end()

    def lens_drops(self, b, look: WaterLook, size, time=0.0, seed=0):
        """Drops on the lens over the finished composite (call right after Renderer.composite)."""
        if look.lens_drops <= 0.0:
            return
        r = self.renderer
        w, h = size
        if self._lens_tmp is None or self._lens_tmp[0].size[:2] != (w, h):
            for x in self._lens_tmp or ():
                x.destroy()
            self._lens_tmp = (self.gpu.texture2d(w, h, 'rgba16float', 'lens-drops-linear'),
                              self.gpu.texture2d(w, h, 'rgba8unorm', 'lens-drops-display'))
        tl, td = self._lens_tmp
        b.copy_texture(r.lin, tl, (w, h, 1))
        b.copy_texture(r.disp, td, (w, h, 1))
        u = Uniforms().v4(w, h, time, look.lens_drops).v4(h / 9.0, 4.0, 1.0 + seed, 0.0)
        b.run(self.k_lens, [tl, td, self.gpu.linear, r.disp, r.lin], u, (w, h, 1))

    def march(self, b, view: LiquidView, camstate: cam.CameraState, fire: cam.FireXform, look: WaterLook, comp, size,
              plate=None, plate_fit=(1.0, 1.0), plate_transform=0, plate_gain=1.0, jitter=(0.0, 0.0), seed=0.0,
              shutter=0.0, ground=True, time=0.0, lamps=(), fire_lights=None, fire_gain=(0.0, 0.0, 0.0), cloth=None,
              standins=None):
        """lamps: the set's lights (Scene.lamps); fire_lights: (buffer, count buffer) of the point lights
        standing in for the fire (Renderer.lights), whose power times fire_gain is in key-light units;
        cloth: the fabric drawn for this pass (Cloth.layer), seen in front of the liquid and through it;
        standins: per collider, its own stand-in colour (r, g, b, 1) or (0, 0, 0, 0) for the look's (stage.standin_colours)."""
        w, h = size
        r = self.renderer
        r._ensure_fire(w, h)
        w2g = cam.world_to_grid(fire, view.origin, view.h)
        g2w = np.linalg.inv(w2g)
        fwd = -camstate.view[2, :3]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        gain = 2.0 ** look.exposure
        absorb = -np.log(np.clip(np.asarray(look.color, float), 1e-4, 1.0)) / max(look.clarity, 1e-3)
        sd = cam.sun_direction(look.sun_azimuth, look.sun_elevation)
        n = view.dims
        u = (Uniforms().m4(camstate.inv_view_proj).m4(camstate.view_proj).m4(w2g).m4(g2w)
             .v4(*n, view.h).v4(*self.nf, self.fscale)
             .v4(w, h, *jitter).v4(seed, shutter, 1.0 if shutter != 0.0 else 0.0, time)
             .v4(1.0 if plate is not None else 0.0, plate_transform, *plate_fit)
             .v4(*comp.bg, 1.0 if comp.bg_checker else 0.0)
             .v4(*camstate.eye, look.backdrop)
             .v4(*fwd, 1.0 if ground else 0.0)
             .v4(look.ior, look.roughness, look.reflection, look.reflect_footage)
             .v4(*absorb, look.murk)
             .v4(*look.murk_color, plate_gain)
             .v4(*(np.asarray(look.sky) * gain), max(8, h / 60))
             .v4(*(np.asarray(look.sun) * gain), look.sun_size)
             .v3(sd, look.shadow)
             .v4(look.ripple, look.ripple_freq, look.ripple_speed)
             .v4(look.wet_darken, 0.0 if look.glow > 0.0 else look.wet_gloss,   # (lava leaves no wet shine)
                 look.crust_scale, 1.0 if look.bottomless else 0.0)
             .v4(look.max_steps, look.step_out, look.step_in, look.events)
             .v4(look.spray * (1.0 - 0.7 * min(1.0, look.droplets)), look.foam, look.bubbles,
                 1.0 if (look.whitewater and view.ww_count) else 0.0)
             .v4(*look.foam_color, look.foam_lace))
        lvl_on = view.level > 0.0 and view.open[0]
        env_on = self.env_tex is not None and bool(look.environment)
        u.v4(*view.origin, 1.0 if look.colliders_look == 'shaded' else 0.0)
        u.v4(view.level / view.h if lvl_on else -1.0, 1.0 if lvl_on else 0.0,
             look.caustics if self.caus_tex is not self._one else 0.0, view.level_blend)
        u.v4(1.0 if env_on else 0.0, math.radians(look.env_rotation), look.env_strength * gain, look.foam_scale)
        u.v4(*(glow_colour(look.glow_temp) * look.glow * gain), look.crust)
        self.ocean.uniforms(u, view.level, whitecaps=look.whitecaps, rainbow=look.rainbow, look=look,
                            pix=2.0 / (max(abs(float(camstate.proj[1, 1])), 1e-6) * h),
                            current=(view.current[0], view.current[2]), time=view.time, layer=view.layer,
                            sides=view.sea_sides)
        u.v4(self._liquid_top(view, lvl_on), 0.0, 0.0, 0.0)   # ocx[14]
        u.v4(1.0 if self.dye_tex is not self._no_dye else 0.0, 48.0, view.current[0] if lvl_on else 0.0,
             view.current[2] if lvl_on else 0.0)
        u.v4(*rain_rings(look.rain, look.rain_drop))
        u.v4(1.0 if (look.ice and view.ice is not None) else 0.0, look.frost, 0.0, look.ice_melting)
        u.v4(*look.frost_color, look.crystal_size)
        pack_lamps(u, lamps, gain, fire_gain if fire_lights is not None else None)
        lava_map = self._lava_glow_lights(b, view, look, gain)
        self.lava_uniforms(u, look, gain, 2.0 / (max(abs(float(camstate.proj[1, 1])), 1e-6) * h),
                           self.crust_field is not self._no_crust_field)
        hold_tex, matte_on, depth_on, kind, scale = r.holdouts(comp)
        u.v4(1.0 if matte_on else 0.0, 1.0 if depth_on else 0.0, kind, scale)
        u.v4(*plate_fit, 0.0, 0.0)
        pack_colliders(u, view.colliders, view.meshes)
        rows = list(standins or [])[:MAX_COLLIDERS]
        for i in range(MAX_COLLIDERS):
            u.v4(*(rows[i] if i < len(rows) else (0.0, 0.0, 0.0, 0.0)))
        atlas = view.meshes.atlas if view.meshes is not None else self._no_atlas
        if self.gbuf is None or self.gbuf.size < w * h * 48:
            if self.gbuf is not None:
                self.gbuf.destroy()
            self.gbuf = self.gpu.buffer(w * h * 48, 'liq-surface-hits')
        b.clear_buffer(self.gbuf, 0, w * h * 48)
        clean = self._clean_plate(b, view, camstate, w2g, look, comp, size, plate, plate_fit, plate_transform, plate_gain,
                                  fwd, ground)
        b.run(self.k_march, [self.surf, plate if plate is not None else self._black, view.wet, self.gpu.linear,
                             r.beauty, r.emit, r.aux, self.ww_tex, atlas, self.caus_tex,
                             self.env_tex if env_on else self._no_env, self.gpu.repeat, r.mask,
                             *self.ocean.textures(view.layer), self.heat_tex, self.dye_tex,
                             *(fire_lights if fire_lights is not None else (self._no_lights, self._no_count)),
                             hold_tex, self.gbuf,
                             lava_map, cloth if cloth is not None else self._no_cloth, clean],
              u, (w, h, 1))
        if look.glow > 0.0:
            # the molten surface the march found, shaded (liq_lava_shade.wgsl)
            lg = w2g[:3, :3] @ np.asarray(sd, float)
            lg = lg / (np.linalg.norm(lg) + 1e-12)
            ul = (Uniforms().m4(g2w).m4(camstate.view_proj)
                  .v4(*n, view.h).v4(*self.nf, self.fscale)
                  .v4(w, h, 1.0 if plate is not None else 0.0, plate_transform)
                  .v4(*view.origin, 0.0)
                  .v4(*camstate.eye, look.reflect_footage)
                  .v4(*fwd, plate_gain)
                  .v4(*(np.asarray(look.sky) * gain), 0.0)
                  .v4(*(np.asarray(look.sun) * gain), look.sun_size)
                  .v4(*lg, 0.0)
                  .v4(1.0 if env_on else 0.0, math.radians(look.env_rotation), look.env_strength * gain, 0.0)
                  .v4(*plate_fit, 0.0, 0.0)
                  .v4(*comp.bg, 0.0))
            self.lava_uniforms(ul, look, gain, 2.0 / (max(abs(float(camstate.proj[1, 1])), 1e-6) * h),
                               self.crust_field is not self._no_crust_field)
            b.run(self.k_lava_shade, [self.surf, self.heat_tex, self.gpu.linear, self.crust_field, self.gbuf,
                                      r.beauty, r.emit, plate if plate is not None else self._black,
                                      self.env_tex if env_on else self._no_env, self.gpu.repeat], ul, (w, h, 1))
        if look.ice and view.ice is not None and look.glow <= 0.0:
            self._ice_shade(b, view, look, w2g, sd, gain, (w, h), camstate)

    def _clean_plate(self, b, view: LiquidView, camstate, w2g, look: WaterLook, comp, size, plate, plate_fit,
                     plate_transform, plate_gain, fwd, ground):
        """The footage with what stands in front of the ground filled in from beside it (liq_clean.wgsl), for the
        march's rays that reach the ground or the backdrop behind it: the colliders drawn in the footage (not as
        stand-ins), and the footage's matte and the surfaces of its depth pass. A 1x1 that covers nothing without."""
        r = self.renderer
        hold_tex, matte_on, depth_on, kind, scale = r.holdouts(comp)
        objects = look.colliders_look != 'shaded' and any(c.holdout for c in list(view.colliders)[:MAX_COLLIDERS])
        if plate is None or not (objects or matte_on or depth_on):
            return self._no_clean
        w, h = size
        if self._clean is None or self._clean[0].size[:2] != (w, h):
            for t in self._clean or ():
                t.destroy()
            self._clean = (self.gpu.texture2d(w, h, 'r32float', 'liq-near'), self.gpu.texture2d(w, h, 'rgba16float', 'liq-clean'))
        near, clean = self._clean
        u = (Uniforms().m4(camstate.inv_view_proj).m4(w2g)
             .v4(*view.dims, view.h).v4(*view.origin, 1.0 if objects else 0.0).v4(w, h)
             .v4(*camstate.eye, look.backdrop).v4(*fwd, 1.0 if ground else 0.0)
             .v4(plate_transform, plate_gain, *plate_fit)
             .v4(1.0 if matte_on else 0.0, 1.0 if depth_on else 0.0, kind, scale).v4(*plate_fit, 0.0, 0.0))
        pack_colliders(u, view.colliders, view.meshes)
        atlas = view.meshes.atlas if view.meshes is not None else self._no_atlas
        res = [plate, hold_tex, atlas, self.gpu.linear]
        b.run(self.k_clean_cover, res + [self._no_near, near, self._no_clean], u, (w, h, 1))
        b.run(self.k_clean_fill, res + [near, self._no_near, clean], u, (w, h, 1))
        return clean

    def _ice_shade(self, b, view: LiquidView, look: WaterLook, w2g, sd, gain, size, camstate):
        """Frost, rime and the sparkle of crystals over the ice the march found (liq_ice_shade.wgsl): it
        reads a copy of what the march wrote and writes over it."""
        w, h = size
        r = self.renderer
        if self._ice_tmp is None or self._ice_tmp[0].size[:2] != (w, h):
            for t in self._ice_tmp or ():
                t.destroy()
            self._ice_tmp = tuple(self.gpu.texture2d(w, h, 'rgba16float', f'liq-ice-{k}') for k in ('beauty', 'emit'))
        tb, te = self._ice_tmp
        b.copy_texture(r.beauty, tb, (w, h, 1))
        b.copy_texture(r.emit, te, (w, h, 1))
        lg = w2g[:3, :3] @ np.asarray(sd, float)
        lg = lg / (np.linalg.norm(lg) + 1e-12)
        u = (Uniforms().m4(camstate.inv_view_proj).m4(w2g)
             .v4(*view.dims, view.h).v4(w, h).v4(*view.origin)
             .v4(*(np.asarray(look.sky) * gain)).v4(*(np.asarray(look.sun) * gain)).v4(*lg)
             .v4(1.0, look.frost, 0.0, look.ice_melting).v4(*look.frost_color, look.crystal_size)
             .v4(1.0, self.fscale, 96, look.ior))
        b.run(self.k_ice_shade, [self.heat_tex, self.dye_tex, self.gpu.linear, self.gbuf, tb, te, r.beauty, r.emit,
                                 r.aux, r.mask, self.surf], u, (w, h, 1))

    LAVA_T0 = 500.0      # K: the blackbody table's first entry
    LAVA_DT = 50.0       # K between entries
    LAVA_N = 32

    LAVA_LIGHTS = 60     # at most this many point lights stand in for a molten surface's glow

    LAVA_MAP = 112       # texels along the longer side of the map of the glow's light on the ground
    LAVA_RELIEF = 1.0    # how strongly a molten skin's lumps are moved into its surface (liq_lava_disp.wgsl)
    LAVA_AIR_DT = 250.0  # K: how much hotter than the air around the air over bared melt is (heat haze)
    LAVA_AIR_HEIGHT = 0.25   # m: the hot air's height scale over the surface
    LAVA_AIR_SPREAD = 0.4    # m the plume spreads per m it rises
    LAVA_AIR_RISE = 0.5      # m/s: the hot air rises (the haze's small eddies drift up with it)

    def lava_haze(self, b, view: LiquidView, look: WaterLook, camstate: cam.CameraState, fire: cam.FireXform, comp,
                  time=0.0):
        """Heat haze over a molten liquid: the air its surface heats (liq_lava_air.wgsl), traced into the
        renderer's haze path (haze_opl.wgsl) as a fire's hot air is. After build(); False when there is no
        haze to composite (not molten, or the look's Heat haze is 0)."""
        if look.glow <= 0.0 or comp.haze <= 0.0 or view.count == 0 or self.nf is None:
            return False
        n = tuple(int(x) for x in view.dims)
        if self._lava_air is None or tuple(self._lava_air.size) != n:
            if self._lava_air is not None:
                self._lava_air.destroy()
            self._lava_air = self.gpu.texture3d(n, 'rgba16float', 'liq-lava-air')
        u = (Uniforms().v4(*view.dims, view.h).v4(*self.nf, self.fscale)
             .v4(self.LAVA_AIR_HEIGHT, self.LAVA_AIR_SPREAD, 0.0, 0.0))
        self.lava_uniforms(u, look, 1.0, 0.0, self.crust_field is not self._no_crust_field)
        b.run(self.k_lava_air, [self.surf, self.heat_tex, self.crust_field, self._lava_air], u, n)
        from .renderer import HAZE_EDDIES, HAZE_EDDY_FREQ
        r = self.renderer
        ow, oh = r.opl.size[:2]
        Ta = 300.0
        u = (Uniforms().m4(camstate.inv_view_proj).m4(cam.world_to_grid(fire, view.origin, view.h))
             .v4(*view.dims, view.h).v4(ow, oh)
             .v4(Ta, Ta + self.LAVA_AIR_DT, Ta + self.LAVA_AIR_DT + 1.0)
             .v4(HAZE_EDDIES, HAZE_EDDY_FREQ * comp.haze_freq, comp.haze_speed, time)
             .v4(*view.origin, 1.0))
        b.run(r.k_opl, [self._lava_air, self._air_rise, r.noise, self.gpu.linear, self.gpu.repeat, r.opl], u, (ow, oh, 1))
        return True

    def _lava_glow_lights(self, b, view: LiquidView, look: WaterLook, gain):
        """The map of a molten surface's glow on the ground around it (liq_lava_map.wgsl), from point lights
        standing in for the surface build() left; an empty one when the liquid is not molten."""
        self._lava_geom = (0.0, 0.0, 1.0, 1.0, 0, 0)
        if look.glow <= 0.0 or look.lava_light <= 0.0 or view.count == 0 or self.nf is None:
            return self._no_lava_lights
        nf = self.nf
        k = max(1, math.ceil(math.sqrt(nf[0] * nf[2] / self.LAVA_LIGHTS)))
        tx, tz = math.ceil(nf[0] / k), math.ceil(nf[2] / k)
        while tx * tz > self.LAVA_LIGHTS:
            k += 1
            tx, tz = math.ceil(nf[0] / k), math.ceil(nf[2] / k)
        cols = nf[0] * nf[2] * 32
        if self._lava_cols is None or self._lava_cols.size < cols:
            if self._lava_cols is not None:
                self._lava_cols.destroy()
            self._lava_cols = self.gpu.buffer(cols, 'liq-lava-columns')
        if self._lava_lights is None:
            self._lava_lights = self.gpu.buffer((1 + 2 * self.LAVA_LIGHTS) * 16, 'liq-lava-lights')
        u = Uniforms().v4(*view.dims, view.h).v4(*nf, self.fscale).v4(*view.origin, 0.0).v4(k, tx, tz, 0.0)
        self.lava_uniforms(u, look, gain, 0.0, self.crust_field is not self._no_crust_field)
        b.run(self.k_lava_cols, [self.surf, self.heat_tex, self._lava_cols, self.crust_field], u, (nf[0], nf[2], 1))
        ds = view.h / self.fscale
        b.run(self.k_lava_lights, [self._lava_cols, self._lava_lights], Uniforms().v4(*nf, 0.0).v4(k, tx, tz, ds),
              (tx, tz, 1))
        # the map: three times the box's footprint, centred on it (past that the glow's light is faint)
        sx, sz = view.dims[0] * view.h * 3.0, view.dims[2] * view.h * 3.0
        cx = view.origin[0] + 0.5 * view.dims[0] * view.h
        cz = view.origin[2] + 0.5 * view.dims[2] * view.h
        mx = self.LAVA_MAP if sx >= sz else max(8, round(self.LAVA_MAP * sx / sz))
        mz = self.LAVA_MAP if sz >= sx else max(8, round(self.LAVA_MAP * sz / sx))
        if self._lava_map is None or self._lava_map.size < mx * mz * 32:
            if self._lava_map is not None:
                self._lava_map.destroy()
            self._lava_map = self.gpu.buffer(self.LAVA_MAP * self.LAVA_MAP * 32, 'liq-lava-light-map')
        self._lava_geom = (cx - 0.5 * sx, cz - 0.5 * sz, sx / mx, sz / mz, mx, mz)
        g = self._lava_geom
        b.run(self.k_lava_map, [self._lava_lights, self._lava_map], Uniforms().v4(*g[:4]).v4(mx, mz, 0.0, 0.0),
              (mx, mz, 1))
        return self._lava_map

    def lava_uniforms(self, u: Uniforms, look: WaterLook, gain, pix, crust_on):
        """The march's molten look (lava[8], lava_bb[32]; liq_lava.wgsl). pix: radians per pixel."""
        on = look.glow > 0.0
        u.v4(1.0 if on else 0.0, look.glow * gain, look.glow_temp, 300.0)
        u.v4(look.crust_scale, look.crust, look.ropes, 1.0 if look.crust_kind == 'plates' else 0.0)
        u.v4(*look.crust_color, look.crust_roughness)
        u.v4(look.crust_time, look.lava_cooling, 1.0 if crust_on else 0.0, look.lava_light)
        u.v4(self.LAVA_T0, 1.0 / self.LAVA_DT, self.LAVA_N, 0.0)
        u.v4(pix, float(os.environ.get('BLACKBODY_LAVA_DEBUG', 0) or 0), 0.0, 0.0)
        g = self._lava_geom
        u.v4(*g[:4])
        u.v4(g[4], g[5], 0.0, 0.0)
        for row in lava_table(self.LAVA_T0, self.LAVA_DT, self.LAVA_N):
            u.v4(*row)
