"""Command line: open the app, or render projects and presets headless (batch and render farms).

  blackbody                                   open the app
  blackbody shot.bbfire                       open a project in the app
  blackbody render shot.bbfire -o renders/fire.####.exr
  blackbody render --preset campfire --footage plate.mov -o comp.mov --content composite
  blackbody render shot.bbfire -o vol/fire.####.vdb --frames 1001-1100
  blackbody presets | settings | info

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
    return t


def _apply_setting(scene, kv):
    """--set SECTION.KEY=VALUE, or emitter.N.KEY=VALUE / collider.N.KEY=VALUE. Returns an error or None."""
    from .scene.params import SECTIONS, param
    key, eq, val = kv.partition('=')
    key = key.strip()
    parts = key.split('.')
    if not eq or len(parts) not in (2, 3):
        return f'--set needs SECTION.KEY=VALUE, got "{kv}".'
    if parts[0] in ('emitter', 'collider'):
        items = scene.emitters if parts[0] == 'emitter' else scene.colliders
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
    if p is None or p.kind == 'str':
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
    from .render.job import RenderJob, infer_output
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
        if o.content == 'composite' and footage is None and o.kind != 'vdb':
            print(f'Note: {path} is a composite but there is no footage; the fire is composited over the background colour.')
        outputs.append(o)
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
    job = RenderJob(scene, outputs, engine, frames=(first, last), final=not args.draft, footage=footage,
                    samples=args.samples, motion_blur=False if args.no_motion_blur else None)
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
    from .scene.params import COLLIDER_PARAMS, EMITTER_PARAMS, SECTIONS
    groups = list(SECTIONS.items()) + [('emitter.N', EMITTER_PARAMS), ('collider.N', COLLIDER_PARAMS)]
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
            else:
                rng = '"x y z"' + (f' {p.unit}' if p.unit else '')
            d = p.default
            if isinstance(d, bool):
                d = 'true' if d else 'false'
            elif isinstance(d, tuple):
                d = ' '.join(f'{x:g}' for x in d)
            elif isinstance(d, float):
                d = f'{d:g}'
            print(f'{g + "." + p.key:<34} {str(d):<16} {rng:<34} {p.label}')
    return 0


def cmd_info(args):
    from .engine.gpu import GPU
    from .io.video import available_profiles
    print(f'{blackbody.APP_NAME} {blackbody.__version__}')
    print('GPUs:')
    for a in GPU.adapters():
        print(f'  {a["name"]} ({a["backend"]}, {a["type"]})')
    g = GPU()
    print(f'Using: {g.name} ({g.backend}); float32 filtering: {"yes" if g.float32_filterable else "no"}')
    print('Video formats:')
    for k, p in available_profiles().items():
        print(f'  {k:12s} {p.label}')
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    _utf8_streams()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    if not argv or (argv[0] not in ('render', 'presets', 'settings', 'info', 'gui', '-h', '--help', '--version')):
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
    r.add_argument('-o', '--output', action='append', help='output path; #### is the frame number. .exr .png .vdb .mov .mp4 .webm (repeatable)')
    r.add_argument('--format', help='video profile: prores4444, prores422hq, dnxhr_hq, dnxhr_444, h264, h265, vp9_alpha')
    r.add_argument('--content', choices=('element', 'composite'), help='the fire with alpha, or the fire composited over the footage')
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
    r.add_argument('--draft', action='store_true', help='interactive quality (fast)')
    r.add_argument('--no-motion-blur', action='store_true')
    r.add_argument('-q', '--quiet', action='store_true')
    sub.add_parser('presets', help='list the built-in presets')
    st = sub.add_parser('settings', help='list every setting --set accepts, with defaults and ranges')
    st.add_argument('section', nargs='?', help='only this section, e.g. motion or emitter')
    sub.add_parser('info', help='show GPUs and available video formats')
    args = ap.parse_args(argv)
    if args.cmd == 'render':
        return cmd_render(args)
    if args.cmd == 'presets':
        return cmd_presets(args)
    if args.cmd == 'settings':
        return cmd_settings(args)
    if args.cmd == 'info':
        return cmd_info(args)
    from .ui.app import run
    return run([args.scene] if getattr(args, 'scene', None) else [])


if __name__ == '__main__':
    sys.exit(main())
