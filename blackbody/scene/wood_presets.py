"""Built-in presets about wood (engine/wood.py): a board snapped by a dropped weight, bullets through a row of pine
boards, and the woods side by side.

presets.py registers them (wood_presets(K) returns the specs).
"""
from __future__ import annotations

WOOD_ORDER = ['board_break', 'boards_shot', 'wood_lineup']

DAY = {'sun_on': True, 'sun_intensity': 3.0, 'sun_elevation': 38.0, 'sun_azimuth': 35.0, 'ambient_intensity': 2.2}
STILL = {'buoyancy': 0.6, 'turbulence': 0.8, 'turb_freq': 3.0, 'vorticity': 1.0}
DUST = {'smoke_density': 2.5, 'smoke_albedo': (0.6, 0.55, 0.48), 'exposure': 0.0}


def wood_presets(K):
    return {
        'board_break': {
            'name': 'Breaking a board', 'category': 'Wood', 'size': '20 mm pine, 10 kg from 1 m',
            'blurb': 'A 10 kg weight dropped from a metre onto a pine board across two blocks, slowed down four times: '
                     'the board bends, its fibres give way at staggered places on its underside and it snaps in a '
                     'jagged, splintered break, the two halves folding into the gap. Wood breaks along its grain into '
                     'bundles of fibres (Breaks into: Splinters); try oak, balsa or a heavier weight.',
            'render': {'end': 72},
            'domain': {'size_x': 1.4, 'size_y': 1.2, 'size_z': 1.0, 'resolution': 64, 'preroll': 0.0, 'time_scale': 0.25},
            'motion': STILL, 'shading': DUST,
            'composite': {'backdrop': 'stage', 'floor': 'concrete'},
            'lighting': DAY,
            'camera': {'distance': 0.95, 'target_y': 0.38, 'pitch': 9.0, 'yaw': -28.0, 'focal_mm': 40, 'use_anchor': False},
            'emitters': [],
            'colliders': [
                dict(name='Block', shape='box', position=(-0.22, 0.2, 0.0), size=(0.07, 0.2, 0.12), material='concrete'),
                dict(name='Block 2', shape='box', position=(0.22, 0.2, 0.0), size=(0.07, 0.2, 0.12), material='concrete'),
                dict(name='Board', shape='box', position=(0.0, 0.411, 0.0), size=(0.24, 0.01, 0.075), material='wood',
                     breakable=True, fracture='splinters', pieces=70, dynamic=True),
                dict(name='Weight', shape='sphere', position=(0.0, 1.47, 0.0), size=(0.068, 0.068, 0.068), material='steel',
                     dynamic=True, density=7700.0, release=0.2),
            ],
        },
        'boards_shot': {
            'name': 'Bullets through boards', 'category': 'Wood', 'size': 'six 19 mm pine boards',
            'blurb': 'The classic penetration test: six pine boards stood 2.5 cm apart, a 9 mm pistol and a .308 rifle '
                     'fired into them. The pistol bullet slows through each and only just gets through the last; the '
                     'rifle bullet goes through all six. Each board is holed going in and torn out along its grain going '
                     'out, splinters standing out of its back and flying from it.',
            'render': {'end': 60},
            'domain': {'size_x': 1.2, 'size_y': 1.0, 'size_z': 1.0, 'resolution': 64, 'preroll': 0.0},
            'motion': STILL, 'shading': DUST,
            'composite': {'backdrop': 'stage', 'floor': 'concrete'},
            'lighting': {**DAY, 'sun_azimuth': 215.0},     # (from behind: the boards' torn backs in the light)
            'camera': {'distance': 0.85, 'target_y': 0.475, 'pitch': 4.0, 'yaw': 200.0, 'focal_mm': 35, 'use_anchor': False},
            'emitters': [],
            'colliders': [dict(name=f'Board {k + 1}', shape='box', position=(0.0, 0.45, -0.044 * k), size=(0.15, 0.45, 0.0095),
                               material='wood') for k in range(6)],
            'shots': [dict(name='Pistol', round='9mm', position=(-0.06, 0.6, 3.0), aim=(-0.06, 0.6, -0.1), start=0.4,
                           scatter=0.0),
                      dict(name='Rifle', round='308', position=(0.06, 0.35, 6.0), aim=(0.06, 0.35, -0.1), start=1.2,
                           scatter=0.0, flash=False)],
        },
        'wood_lineup': {
            'name': 'Woods', 'category': 'Wood', 'size': '15 species',
            'blurb': 'Every wood side by side, as boards and as logs sawn through: pine, spruce, Douglas fir, oak, ash, '
                     'maple, birch, walnut, cherry, mahogany, teak, cedar, balsa, plywood and MDF. Their grain, rings, '
                     'knots, pores and rays are their own, right through them (a cut or a break shows the same wood).',
            'render': {'end': 2},
            'domain': {'size_x': 4.0, 'size_y': 1.4, 'size_z': 1.4, 'resolution': 48, 'preroll': 0.0},
            'composite': {'backdrop': 'stage', 'floor': 'studio'},
            'lighting': {**DAY, 'sun_elevation': 30.0},
            'camera': {'distance': 4.6, 'target_y': 0.35, 'pitch': 14.0, 'yaw': 0.0, 'focal_mm': 35, 'use_anchor': False},
            'emitters': [],
            'colliders': [dict(name=f'{k.title()} board', shape='box', position=(-1.75 + 0.25 * n, 0.4, 0.0),
                               size=(0.09, 0.4, 0.012), material=k, yaw=-10.0)
                          for n, k in enumerate(('wood', 'spruce', 'fir', 'oak', 'ash', 'maple', 'birch', 'walnut', 'cherry',
                                                 'mahogany', 'teak', 'cedar', 'balsa', 'plywood', 'mdf'))]
                         + [dict(name=f'{k.title()} log', shape='cylinder', position=(-1.6 + 0.8 * n, 0.08, 0.45),
                                 size=(0.11, 0.08, 0.11), material=k)
                            for n, k in enumerate(('wood', 'oak', 'birch', 'walnut', 'cedar'))],
        },
    }
