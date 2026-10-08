"""Command line: open the app, or render projects and presets headless (batch and render farms).

  blackbody                                   open the app
  blackbody shot.bbfire                       open a project in the app
  blackbody render shot.bbfire -o renders/fire.####.exr
  blackbody render --preset campfire --footage plate.mov -o comp.mov --content composite
  blackbody render shot.bbfire -o vol/fire.####.vdb --frames 1001-1100
  blackbody render shot.bbfire -o renders/fire.deep.####.exr          deep EXR (a name with .deep.)
  blackbody render pond.bbfire -o mesh/pond.usdc                     a liquid's surface as a USD mesh (or .####.obj)
  blackbody render flag.bbfire -o mesh/flag.usdc                     the fabric as a USD mesh (or .####.obj)
  blackbody render shot.bbfire -o usd/shot.scene.usdc                camera, objects, pieces, sand, grass, embers
  blackbody render shot.bbfire -o cam/shot.chan                      the camera for Nuke (or cam.camera.usda: USD)
  blackbody render --preset campfire --usd shot.usd -o fire.####.exr  camera, objects and frame range from USD
  blackbody render --preset campfire --usd shot.usd --usd-offset 1000 -o fire.####.exr   USD 1001-1100 as 1-100

Render farms: simulate once into a shared disk cache (resuming from the last checkpoint if it was
stopped), then render frame ranges on as many machines as you like from that cache:
  blackbody simulate shot.bbfire --cache //server/cache/shot
  blackbody render shot.bbfire --from-cache //server/cache/shot --frames 1001-1050 -o fire.####.exr

  blackbody presets | settings | info
  blackbody precompile          compile the slow GPU shaders now (installers, farms; the app does it in the background)

Exit codes: 0 done, 1 failed, 2 bad arguments, 130 cancelled.
"""
from __future__ import annotations

import argparse
import difflib
import logging
import sys
import time
import traceback
from pathlib import Path

import blackbody


def _fail(msg, code=1):
    print(f'Error: {msg}', file=sys.stderr)
    return code


def _utf8_streams():
    """Farm logs are read as UTF-8; piped output would otherwise use the Windows code page."""
    for s in (sys.stdout, sys.stderr):
        if s is not None and hasattr(s, 'reconfigure'):
            try:
                s.reconfigure(encoding='utf-8', errors='replace')
            except (OSError, ValueError):
                pass


def _frames(text, scene):
    if not text:
        return scene.start, scene.end
    if '-' in text[1:]:
        a, b = text.split('-', 1) if not text.startswith('-') else ('-' + text[1:].split('-', 1)[0], text[1:].split('-', 1)[1])
        return int(a), int(b)
    return int(text), int(text)


def _progress_printer(quiet):
    t0 = time.perf_counter()
    last = {'t': 0.0, 'msg': None}
    tty = sys.stdout is not None and sys.stdout.isatty()

    def cb(frac, msg):
        if quiet or sys.stdout is None:
            return
        now = time.perf_counter()
        el = now - t0
        eta = el / frac - el if frac > 0.01 else 0.0
        if tty:
            if now - last['t'] < 0.25 and frac < 1.0:
                return
            bar = '#' * int(frac * 30)
            sys.stdout.write(f'\r[{bar:<30}] {frac * 100:5.1f}%  {msg:<28} elapsed {el:6.1f}s  left {eta:6.1f}s')
            if frac >= 1.0:
                sys.stdout.write('\n')
        else:
            # piped into a log (render farms): whole lines that parse as "Progress: N%", one per
            # frame, and every few seconds during pre-roll
            new_frame = msg != last['msg'] and not msg.startswith('Simulating')
            if not (new_frame or frac >= 1.0 or now - last['t'] >= 5.0):
                return
            last['msg'] = msg
            sys.stdout.write(f'Progress: {frac * 100:.1f}%  {msg}  (elapsed {el:.1f}s, left {eta:.1f}s)\n')
        last['t'] = now
        sys.stdout.flush()
    return cb


def _notices(order, engines=None, said=None, final=True):
    """Print what the simulation leaves out or cuts short in the layers `order` ([(uid, scene)]), once each: the scenes'
    caps before a run (scene/caps.py) and what was cut to fit the GPU (Scene.memory_notes; `final` or draft), or, given
    their engines (render/layers.LayerEngines), what those said as they ran (Engine.notices). Returns what has been
    said."""
    said = set() if said is None else said
    for uid, sc in order:
        try:
            if engines is None:
                from .scene import caps
                notes = caps.notices(sc) + sc.memory_notes(final)
            else:
                eng = engines.base if uid == 'base' else engines.extra.get(uid)
                notes = eng.notices() if eng is not None else []
        except Exception as ex:   # (saying so must never stop a render)
            notes = [f'Could not check what is left out: {ex}']
        for n in notes:
            line = f'Note: {sc.name}: {n}' if len(order) > 1 else f'Note: {n}'
            if line not in said:
                said.add(line)
                print(line)
    return said


def _check_value(p, text):
    """Validate a --set value strictly (the project loader is lenient, a farm job should not be)."""
    t = text.strip()
    if p.kind == 'bool':
        if t.lower() not in ('1', 'true', 'yes', 'on', '0', 'false', 'no', 'off'):
            raise ValueError(f'"{text}" is not on or off (use true or false)')
    elif p.kind == 'enum':
        vals = [o[0] for o in p.options]
        if t not in vals:
            raise ValueError(f'"{text}" is not one of: {", ".join(vals)}')
    elif p.kind == 'file':
        from pathlib import Path
        if t and not t.startswith('builtin:') and not Path(t).exists():
            raise ValueError(f'there is no file "{text}"')
    elif p.kind in ('float', 'int', 'color', 'vec3'):
        try:
            nums = [float(x) for x in t.replace(',', ' ').split()]
        except ValueError:
            nums = []
        want = 1 if p.kind in ('float', 'int') else 3
        if len(nums) != want:
            raise ValueError(f'"{text}" is not {"a number" if want == 1 else "three numbers"}')
    elif p.kind == 'list':
        import json
        try:
            v = json.loads(t) if t else []
        except ValueError:
            v = None
        if not isinstance(v, list) or not all(isinstance(e, dict) for e in v):
            raise ValueError(f'"{text}" is not a JSON list of settings, e.g. [{{"joint": "rope", "joint_at": [0.4, 0, 0]}}]')
    return t


def _apply_setting(scene, kv):
    """--set SECTION.KEY=VALUE, or emitter.N.KEY=VALUE / collider.N.KEY=VALUE / light.N.KEY=VALUE / fabric.N.KEY=VALUE.
    Returns an error or None."""
    from .scene.params import SECTIONS, param
    key, eq, val = kv.partition('=')
    key = key.strip()
    parts = key.split('.')
    if not eq or len(parts) not in (2, 3):
        return f'--set needs SECTION.KEY=VALUE, got "{kv}".'
    if parts[0] in ('emitter', 'collider', 'light', 'fabric'):
        items = {'emitter': scene.emitters, 'collider': scene.colliders, 'light': scene.lights, 'fabric': scene.fabrics}[parts[0]]
        if len(parts) != 3 or not parts[1].isdigit():
            return f'{parts[0].capitalize()} settings need an index, e.g. {parts[0]}.0.{parts[-1]}={val}.'
        if int(parts[1]) >= len(items):
            return (f'This scene has {len(items)} {parts[0]}{"" if len(items) == 1 else "s"} (numbered from 0), '
                    f'so there is no {parts[0]}.{parts[1]}.')
        path = (parts[0], int(parts[1]), parts[2])
    elif len(parts) == 2:
        path = tuple(parts)
    else:
        path = None
    try:
        p = param(path[0], path[-1]) if path else None
    except KeyError:
        p = None
    if p is None or (p.kind == 'str' and path[-1] == 'name'):
        names = [f'{s}.{q.key}' for s, qs in SECTIONS.items() for q in qs]
        near = difflib.get_close_matches(key, names, n=3)
        return (f'Unknown setting "{key}".' + (f' Did you mean {" or ".join(near)}?' if near else '')
                + ' "blackbody settings" lists them all.')
    try:
        scene.set(path, _check_value(p, val))
    except ValueError as ex:
        return f'Bad value for {key}: {ex}.'
    return None


def cmd_render(args):
    from .engine.engine import Engine
    from .io.footage import Footage
    from .io.video import PROFILES, available_profiles
    from .render.job import (DATA_KINDS, PASSES, RenderJob, check_outputs, infer_output, liquid_layers,
                             output_notes)
    from .scene import Scene, presets

    if args.scene:
        if not Path(args.scene).is_file():
            return _fail(f'No project file at {args.scene}', 2)
        try:
            scene = Scene.load(args.scene)
        except Exception as ex:
            return _fail(f'Could not read the project {args.scene}: {ex}')
    elif args.preset:
        if args.preset not in presets.ORDER:
            return _fail(f'Unknown preset "{args.preset}". Choose from: {", ".join(presets.ORDER)}', 2)
        scene = presets.make(args.preset)
    else:
        return _fail('Give a project file or --preset NAME.', 2)
    footage = None
    fpath = args.footage or (scene.footage or {}).get('path')
    if fpath:
        try:
            footage = Footage(fpath)
        except FileNotFoundError:
            if args.footage:
                return _fail(f'Footage not found: {fpath}', 2)
            return _fail(f"The project's footage is missing: {fpath}. Put it back, or give --footage PATH.")
        except Exception as ex:
            return _fail(f'Could not open the footage {fpath}: {ex}')
        if args.footage:
            scene.footage = {'path': str(Path(fpath).resolve()), 'offset': 0}
            scene.data['render']['width'], scene.data['render']['height'] = footage.width, footage.height
            scene.data['render']['fps'] = footage.fps
            scene.data['render']['end'] = scene.start + footage.frames - 1
    for kv in args.set or []:
        err = _apply_setting(scene, kv)
        if err:
            return _fail(err, 2)
    if args.res:
        scene.data['domain']['resolution'] = args.res
    err = _usd_and_cache(scene, args)
    if err:
        return err
    scene.from_cache = bool(args.from_cache)   # (laid out as the cache was simulated, on any card: Scene.memory_plan)
    if args.size:
        try:
            w, h = (int(x) for x in args.size.lower().split('x'))
        except ValueError:
            return _fail(f'--size needs WIDTHxHEIGHT, e.g. 1920x1080, got "{args.size}".', 2)
        scene.data['render']['width'], scene.data['render']['height'] = w, h
    try:
        first, last = _frames(args.frames, scene)
    except ValueError:
        return _fail(f'--frames needs a frame or a range, e.g. 1001-1100, got "{args.frames}".', 2)
    if last < first:
        return _fail(f'--frames {args.frames} ends before it starts.', 2)
    outputs = []
    for path in args.output or ['renders/' + (scene.name or 'fire').replace(' ', '_').lower() + '.####.exr']:
        try:
            o = infer_output(path, args.content, args.format if args.format not in (None, 'exr', 'png', 'vdb') else None)
        except ValueError as ex:
            return _fail(str(ex), 2)
        if o.kind == 'video':
            usable = available_profiles()
            if o.profile not in usable:
                why = 'is not available in this build' if o.profile in PROFILES else 'is not a video format'
                return _fail(f'"{o.profile}" {why}. Choose from: {", ".join(usable)}', 2)
            ext = PROFILES[o.profile].ext
            if Path(path).suffix.lower() != ext:
                print(f'Note: {o.profile} is written as {ext}, so {path} becomes {Path(path).with_suffix(ext)}.')
        if args.alpha:
            o.alpha_mode = args.alpha
        if args.png_bits:
            o.bits = args.png_bits
        if args.exr_compression:
            o.compression = args.exr_compression
        if args.float:
            o.half = False
        if args.no_passes:
            o.layers = tuple(k for k in o.layers if k not in PASSES)
        if o.kind == 'deep' and args.deep_samples:
            o.deep_samples = args.deep_samples
        if o.kind == 'mesh' and not liquid_layers(scene):
            o.content = 'fabric'   # (no liquid in any layer: the mesh is the fabric)
        if o.content == 'composite' and footage is None and o.kind not in DATA_KINDS:
            print(f'Note: {path} is a composite but there is no footage; the fire is composited over the background colour.')
        outputs.append(o)
    problems = check_outputs(scene, outputs)
    if problems:
        return _fail(' '.join(problems), 2)
    for note in output_notes(scene, outputs):
        print(f'Note: {note}')
    try:
        engine = Engine()
    except Exception as ex:
        return _fail(f'Could not start the GPU engine: {ex}. Update the graphics driver; '
                     '"blackbody info" lists the GPUs Blackbody can see.')
    if not args.quiet:
        dims, h, _ = scene.sim_layout(final=not args.draft)
        print(f'{blackbody.APP_NAME} {blackbody.__version__} on {engine.gpu.name} ({engine.gpu.backend})')
        print(f'Scene "{scene.name}": frames {first}-{last} at {scene.fps:g} fps, {scene.output_size()[0]}x{scene.output_size()[1]}, '
              f'{dims[0]}x{dims[1]}x{dims[2]} voxels ({h * 1000:.1f} mm)')
        for o in outputs:
            print(f'  -> {o.path}  [{o.label()}]')
    from .render.layers import LayerEngines
    order = scene.layer_order() or [('base', scene)]
    said = _notices(order, final=not args.draft)
    engines = LayerEngines(engine)
    job = RenderJob(scene, outputs, engine, frames=(first, last), final=not args.draft, footage=footage,
                    samples=args.samples, motion_blur=False if args.no_motion_blur else None,
                    from_cache=bool(args.from_cache), engines=engines)
    t0 = time.perf_counter()
    try:
        written = job.run(progress=_progress_printer(args.quiet))
    except KeyboardInterrupt:
        job.cancelled = True
        print('\nCancelled.')
        return 130
    except Exception as ex:
        traceback.print_exc()
        return _fail(f'Render failed: {ex}')
    finally:
        _notices(order, engines, said)     # (and what the simulation said as it ran)
        for n in job.notes:                # (and what the writers could not write as it is)
            print(f'Note: {n}')
    if not args.quiet:
        n = last - first + 1
        dt = time.perf_counter() - t0
        print(f'Done: {n} frames in {dt:.1f}s ({dt / max(n, 1):.2f}s per frame). {len(written)} files written.')
    return 0


def cmd_presets(args):
    from .scene.presets import ORDER, PRESETS
    for k in ORDER:
        p = PRESETS[k]
        print(f'{k:14s} {p["name"]:14s} {p["size"]:12s} {p["blurb"]}')
    return 0


def cmd_settings(args):
    """Every name --set accepts, with its default and typical range."""
    from .scene.params import COLLIDER_PARAMS, EMITTER_PARAMS, FABRIC_PARAMS, LIGHT_PARAMS, SECTIONS
    groups = list(SECTIONS.items()) + [('emitter.N', EMITTER_PARAMS), ('collider.N', COLLIDER_PARAMS),
                                       ('light.N', LIGHT_PARAMS), ('fabric.N', FABRIC_PARAMS)]
    if args.section and not any(g == args.section or g.split('.')[0] == args.section for g, _ in groups):
        return _fail(f'Unknown section "{args.section}". Choose from: '
                     f'{", ".join(g.split(".")[0] for g, _ in groups)}', 2)
    print(f'{"SETTING":<34} {"DEFAULT":<16} {"TYPICAL RANGE":<34} NAME IN THE APP')
    for g, ps in groups:
        if args.section and args.section not in (g, g.split('.')[0]):
            continue
        for p in ps:
            if p.kind == 'str':
                continue
            if p.kind == 'enum':
                rng = ' | '.join(o[0] for o in p.options)
            elif p.kind == 'bool':
                rng = 'true | false'
            elif p.kind in ('float', 'int'):
                rng = f'{p.lo:g} to {p.hi:g}' + (f' {p.unit}' if p.unit else '')
            elif p.kind == 'file':
                rng = 'path to an OBJ or STL file'
            elif p.kind == 'list':
                rng = 'JSON list of joint settings'
            else:
                rng = '"x y z"' + (f' {p.unit}' if p.unit else '')
            d = p.default
            if isinstance(d, bool):
                d = 'true' if d else 'false'
            elif p.kind == 'list':
                d = '[]'
            elif isinstance(d, tuple):
                d = ' '.join(f'{x:g}' for x in d)
            elif isinstance(d, float):
                d = f'{d:g}'
            print(f'{g + "." + p.key:<34} {str(d):<16} {rng:<34} {p.label}')
    return 0


def cmd_precompile(args):
    from .engine import precompile
    return precompile.main(quiet=args.quiet)


def cmd_info(args):
    from .engine.gpu import GPU
    from .io.video import available_profiles
    print(f'{blackbody.APP_NAME} {blackbody.__version__}')
    print('GPUs:')
    for a in GPU.adapters():
        print(f'  {a["name"]} ({a["backend"]}, {a["type"]})')
    g = GPU()
    print(f'Using: {g.name} ({g.backend}); float32 filtering: {"yes" if g.float32_filterable else "no"}')
    m, GB = g.memory, 2 ** 30
    print(f'Memory: {m["total"] / GB:.1f} GB ({m["source"]}), {m["free"] / GB:.1f} GB free now; scenes are fitted within '
          f'{g.plan / GB:.1f} GB' if m else 'Memory: not known (BLACKBODY_GPU_MEMORY gives it, in GB); scenes are made as set')
    print('Video formats:')
    for k, p in available_profiles().items():
        print(f'  {k:12s} {p.label}')
    return 0


def _usd_and_cache(scene, args):
    """--usd FILE (camera and meshes from a USD scene; its frame range and rate too, as the app's import does, unless
    --usd-keep-range: with footage, only where the shot starts, the footage keeping its length and rate; --usd-offset N
    turns USD frame F into frame F - N), --cache / --from-cache DIR (the disk cache)."""
    if getattr(args, 'usd', None):
        from .io.usd import import_usd
        footage = bool(getattr(args, 'footage', None) or (scene.footage or {}).get('path'))
        match = False if getattr(args, 'usd_keep_range', False) else ('start' if footage else True)
        try:
            for line in import_usd(scene, args.usd, offset=getattr(args, 'usd_offset', 0) or 0, match_range=match):
                if not getattr(args, 'quiet', False):
                    print(f'USD: {line}')
        except Exception as ex:
            return _fail(f'Could not import {args.usd}: {ex}', 2)
    folder = getattr(args, 'from_cache', None) or getattr(args, 'cache', None)
    if folder:
        if getattr(args, 'from_cache', None) and not Path(folder).is_dir():
            return _fail(f'No disk cache at {folder}; simulate into it first (blackbody simulate ... --cache {folder}).', 2)
        scene.data['domain']['disk_cache'] = True
        scene.data['domain']['cache_dir'] = str(Path(folder).resolve())
    return None


def cmd_simulate(args):
    """Simulate into the disk cache (resuming from its last checkpoint), without rendering."""
    from .engine.engine import Engine
    from .scene import Scene, presets
    if args.scene:
        if not Path(args.scene).is_file():
            return _fail(f'No project file at {args.scene}', 2)
        try:
            scene = Scene.load(args.scene)
        except Exception as ex:
            return _fail(f'Could not read the project {args.scene}: {ex}')
    elif args.preset:
        if args.preset not in presets.ORDER:
            return _fail(f'Unknown preset "{args.preset}". Choose from: {", ".join(presets.ORDER)}', 2)
        scene = presets.make(args.preset)
    else:
        return _fail('Give a project file or --preset NAME.', 2)
    for kv in args.set or []:
        err = _apply_setting(scene, kv)
        if err:
            return _fail(err, 2)
    if args.res:
        scene.data['domain']['resolution'] = args.res
    if not args.cache:
        args.cache = str(Path(args.scene).with_suffix('.bbcache')) if args.scene else 'blackbody.bbcache'
    err = _usd_and_cache(scene, args)
    if err:
        return err
    try:
        first, last = _frames(args.frames, scene)
    except ValueError:
        return _fail(f'--frames needs a frame or a range, e.g. 1001-1100, got "{args.frames}".', 2)
    try:
        engine = Engine()
    except Exception as ex:
        return _fail(f'Could not start the GPU engine: {ex}.')
    final = not args.draft
    engine.prepare(scene, final=final)
    disk = engine.cache.disk
    if not args.quiet:
        dims, h, _ = scene.sim_layout(final=final)
        cps = disk.checkpoints() if disk else []
        print(f'Simulating "{scene.name}" frames {scene.start}-{last} into {disk.folder if disk else "?"} '
              f'({dims[0]}x{dims[1]}x{dims[2]} voxels); {len(disk.frames()) if disk else 0} frames already cached'
              + (f', resuming from frame {max(c for c in cps if c <= last)}' if any(c <= last for c in cps) else ''))
    from .render.layers import LayerEngines
    said = _notices([('base', scene)], final=final)
    t0 = time.perf_counter()
    report = _progress_printer(args.quiet)
    try:
        ok = engine.simulate_to(scene, last, progress=lambda f, fr: report(f, f'Frame {fr}'))
    except KeyboardInterrupt:
        print('\nStopped; the next run resumes from the last checkpoint.')
        return 130
    except Exception as ex:
        traceback.print_exc()
        return _fail(f'Simulation failed: {ex}')
    finally:
        if engine.cache.disk is not None:
            engine.cache.disk.flush()
        _notices([('base', scene)], LayerEngines(engine), said)
    if not args.quiet:
        print(f'Done in {time.perf_counter() - t0:.1f}s: {len(disk.frames())} frames cached, '
              f'{disk.size_bytes() / 1e9:.2f} GB.')
    return 0 if ok else 1


def _usd_args(p):
    p.add_argument('--usd', help='bring in the camera, objects and lights from a USD scene, and its frame range and rate')
    p.add_argument('--usd-offset', type=int, default=0, metavar='FRAMES',
                   help='USD frame F becomes frame F - FRAMES (1000 turns a 1001-1100 shot into 1-100; up to 100000 '
                        'either way)')
    p.add_argument('--usd-keep-range', action='store_true',
                   help="keep the project's (or preset's) frame range and rate instead of the USD scene's")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _utf8_streams()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    if not argv or (argv[0] not in ('render', 'simulate', 'presets', 'settings', 'info', 'precompile', 'gui', '-h',
                                    '--help', '--version')):
        from .ui.app import run
        return run(argv)
    ap = argparse.ArgumentParser(prog='blackbody', description='GPU fire and smoke simulation for compositing into footage.')
    ap.add_argument('--version', action='version', version=f'{blackbody.APP_NAME} {blackbody.__version__}')
    sub = ap.add_subparsers(dest='cmd')
    g = sub.add_parser('gui', help='open the app')
    g.add_argument('scene', nargs='?')
    r = sub.add_parser('render', help='render a project or preset without opening the app')
    r.add_argument('scene', nargs='?', help='.bbfire project file')
    r.add_argument('--preset', help='start from a built-in preset instead of a project')
    r.add_argument('-o', '--output', action='append', help='output path; #### is the frame number. .exr .png .vdb .mov .mp4 .webm, '
                                                           'for liquids and fabric .obj (per frame) or .usd/.usdc/.usda, '
                                                           'name.scene.usdc for the shot as a USD scene (camera, objects, '
                                                           'pieces, sand, grass, embers), .chan for the camera '
                                                           '(repeatable)')
    r.add_argument('--format', help='video profile: prores4444, prores422hq, dnxhr_hq, dnxhr_444, h264, h265, vp9_alpha')
    r.add_argument('--content', choices=('element', 'composite', 'deep', 'scene', 'camera'),
                   help='the fire with alpha, the fire composited over the footage, (for .exr) deep samples, or (for '
                        '.usd) the shot as a USD scene or its camera alone')
    r.add_argument('--footage', help='footage to composite over (sets size, frame rate and length)')
    r.add_argument('--frames', help='frame range, e.g. 1-120 or 1001-1100')
    r.add_argument('--size', help='output size, e.g. 1920x1080')
    r.add_argument('--res', type=int, help='voxels on the longest side of the box')
    r.add_argument('--samples', type=int, help='samples per pixel')
    r.add_argument('--set', action='append', metavar='SECTION.KEY=VALUE',
                   help='override any setting, e.g. motion.wind_speed=3 or emitter.0.fuel=20 (see "blackbody settings")')
    r.add_argument('--alpha', choices=('premultiplied', 'straight'))
    r.add_argument('--png-bits', type=int, choices=(8, 16))
    r.add_argument('--exr-compression', choices=('none', 'zip', 'zips', 'piz', 'dwaa', 'dwab', 'rle', 'zstd'))
    r.add_argument('--float', action='store_true', help='32-bit float EXR instead of half')
    r.add_argument('--no-passes', action='store_true',
                   help='EXRs without the compositing passes (motion vectors, normals, object mattes, Cryptomatte)')
    r.add_argument('--deep-samples', type=int, choices=(8, 16),
                   help='deep EXR samples per pixel (default 8; 16 keeps thick, layered smoke apart, at twice the memory)')
    r.add_argument('--draft', action='store_true', help='interactive quality (fast)')
    r.add_argument('--no-motion-blur', action='store_true')
    _usd_args(r)
    r.add_argument('--cache', help='also write the simulated frames to this disk cache folder')
    r.add_argument('--from-cache', help='render frames already simulated into this disk cache folder (see simulate)')
    r.add_argument('-q', '--quiet', action='store_true')
    sm = sub.add_parser('simulate', help='simulate into a disk cache without rendering (resumes if stopped)')
    sm.add_argument('scene', nargs='?', help='.bbfire project file')
    sm.add_argument('--preset', help='start from a built-in preset instead of a project')
    sm.add_argument('--cache', help='disk cache folder (default: next to the project, NAME.bbcache)')
    sm.add_argument('--frames', help='simulate up to the last frame of this range, e.g. 1-120')
    sm.add_argument('--res', type=int, help='voxels on the longest side of the box')
    sm.add_argument('--set', action='append', metavar='SECTION.KEY=VALUE', help='override any setting')
    _usd_args(sm)
    sm.add_argument('--draft', action='store_true', help='interactive quality (fast)')
    sm.add_argument('-q', '--quiet', action='store_true')
    sub.add_parser('presets', help='list the built-in presets')
    st = sub.add_parser('settings', help='list every setting --set accepts, with defaults and ranges')
    st.add_argument('section', nargs='?', help='only this section, e.g. motion or emitter')
    sub.add_parser('info', help='show GPUs and available video formats')
    pc = sub.add_parser('precompile', help="compile the slow GPU shaders now (the liquid renderer's), so the first "
                                           'scene that needs them does not wait')
    pc.add_argument('-q', '--quiet', action='store_true')
    args = ap.parse_args(argv)
    if args.cmd == 'render':
        return cmd_render(args)
    if args.cmd == 'simulate':
        return cmd_simulate(args)
    if args.cmd == 'presets':
        return cmd_presets(args)
    if args.cmd == 'settings':
        return cmd_settings(args)
    if args.cmd == 'info':
        return cmd_info(args)
    if args.cmd == 'precompile':
        return cmd_precompile(args)
    from .ui.app import run
    return run([args.scene] if getattr(args, 'scene', None) else [])


if __name__ == '__main__':
    sys.exit(main())
