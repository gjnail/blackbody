"""The Lume benchmark's scenes, in a neutral form both renderers are built from (lume_side.py, mitsuba_side.py).

Units are metres, y up. Every surface uses Lume's material (lume.wgsl lu_eval): albedo, roughness, metalness; a
dielectric's highlight has f0 = 0.04. Clear objects are glass (exact dielectric Fresnel, index `ior`). Lamps are
spheres of radius `radius` whose intensity `power` (rgb, W/sr-like: irradiance at 1 m) falls off as 1 / d^2; the sky
is a uniform radiance. Bounces as Lume counts them: 1 is direct light only."""

SCENES = {
    # A white ball under a uniform sky, nothing else: the furnace test. Every bit of light is the sky's, once or
    # bounced; a renderer that loses or makes energy shows it here.
    'furnace': dict(
        size=(320, 240), camera=dict(eye=(0.0, 0.5, 3.0), target=(0.0, 0.5, 0.0), hfov=40.0),
        sky=1.0, floor=None, bounces=8,
        objects=[dict(shape='sphere', pos=(0.0, 0.5, 0.0), size=(0.5, 0.5, 0.5), alb=(0.8, 0.8, 0.8), rough=0.5)],
        lamps=[]),
    # An open box with red and green sides and two blocks, lit by a lamp inside: light bouncing from wall to wall
    # (colour bleeding, the blocks' shaded sides).
    'cornell': dict(
        size=(320, 320), camera=dict(eye=(0.0, 1.0, 3.6), target=(0.0, 0.9, 0.0), hfov=38.0),
        sky=0.02, floor=dict(alb=(0.7, 0.7, 0.7), rough=0.9), bounces=6,
        objects=[
            dict(shape='box', pos=(0.0, 1.0, -1.05), size=(1.1, 1.0, 0.05), alb=(0.75, 0.75, 0.75), rough=0.9),   # back
            dict(shape='box', pos=(-1.05, 1.0, 0.0), size=(0.05, 1.0, 1.1), alb=(0.7, 0.08, 0.06), rough=0.9),    # left, red
            dict(shape='box', pos=(1.05, 1.0, 0.0), size=(0.05, 1.0, 1.1), alb=(0.1, 0.55, 0.12), rough=0.9),     # right, green
            dict(shape='box', pos=(0.0, 2.05, 0.0), size=(1.1, 0.05, 1.1), alb=(0.75, 0.75, 0.75), rough=0.9),    # ceiling
            dict(shape='box', pos=(-0.35, 0.3, -0.3), size=(0.25, 0.3, 0.25), alb=(0.75, 0.75, 0.75), rough=0.8),
            dict(shape='box', pos=(0.4, 0.2, 0.25), size=(0.2, 0.2, 0.2), alb=(0.75, 0.75, 0.75), rough=0.8),
        ],
        lamps=[dict(pos=(0.0, 1.75, 0.1), radius=0.08, power=(3.0, 2.8, 2.5))]),
    # The same box with direct light only (one bounce), and with two: where the light carried is wrong, these say which bounce.
    'cornell_direct': None,
    'cornell_2': None,
    # Metal balls rough and smooth, two glossy plastic ones, on a floor under a low distant lamp and the sky:
    # highlights and reflections.
    'glossy': dict(
        size=(400, 240), camera=dict(eye=(0.0, 0.9, 3.4), target=(0.0, 0.3, 0.0), hfov=42.0),
        sky=0.15, floor=dict(alb=(0.45, 0.45, 0.45), rough=0.6), bounces=4,
        objects=[
            dict(shape='sphere', pos=(-1.2, 0.3, 0.0), size=(0.3,) * 3, alb=(0.95, 0.75, 0.35), rough=0.08, metal=1.0),
            dict(shape='sphere', pos=(-0.4, 0.3, 0.0), size=(0.3,) * 3, alb=(0.9, 0.9, 0.9), rough=0.3, metal=1.0),
            dict(shape='sphere', pos=(0.4, 0.3, 0.0), size=(0.3,) * 3, alb=(0.75, 0.2, 0.15), rough=0.15),
            dict(shape='sphere', pos=(1.2, 0.3, 0.0), size=(0.3,) * 3, alb=(0.15, 0.3, 0.75), rough=0.6),
        ],
        lamps=[dict(pos=(-6.0, 5.0, 4.0), radius=0.4, power=(60.0, 56.0, 50.0))]),
    # A glass ball on the floor beside a block, under a lamp: refraction, its shadow and the light focused through it.
    'glass': dict(
        size=(320, 240), camera=dict(eye=(0.0, 0.8, 2.6), target=(0.0, 0.25, 0.0), hfov=40.0),
        sky=0.1, floor=dict(alb=(0.6, 0.6, 0.6), rough=0.7), bounces=8,
        objects=[
            dict(shape='sphere', pos=(-0.25, 0.25, 0.0), size=(0.25,) * 3, alb=(1.0, 1.0, 1.0), rough=0.03, clear=1.0, ior=1.5),
            dict(shape='box', pos=(0.45, 0.15, -0.2), size=(0.15, 0.15, 0.15), alb=(0.7, 0.3, 0.2), rough=0.7),
        ],
        lamps=[dict(pos=(-1.5, 2.5, 0.5), radius=0.1, power=(8.0, 7.6, 7.0))]),
}

for _n, _b in (('cornell_direct', 1), ('cornell_2', 2)):
    SCENES[_n] = dict(SCENES['cornell'], bounces=_b)
