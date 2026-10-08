# Contributing

Bug reports, fixes, new presets, footage that breaks it, physics checks and
platform testing are all welcome.

## Rules for changes to the simulation and the look

Blackbody is meant to put believable effects into real footage, so these apply
to every change to a solver, a renderer or a preset:

1. Use real units and real numbers. Settings are in metres, seconds, kelvin,
   kilograms and their combinations. Presets are real things at real sizes,
   such as a 1.2 m campfire or a 4 cm ice cube. If a value is tuned rather
   than physical, say so in its tooltip.
2. Check it against reality. When a change affects how something moves or
   looks, compare it with measurements or textbook values and add a test that
   does the same. `tools/fire_check.py` does this for fire: puffing rate,
   flame height and plume speed. The existing tests do it for waves, boiling,
   evaporation, fall speeds and fabric stiffness.
3. Keep simulations deterministic. The same scene and settings must give the
   same frames on the same machine, because render farms render frame ranges
   of one simulation on several machines. Avoid results that depend on GPU
   thread order; the particle transfers use fixed-point atomics for this
   reason. Nothing that shapes a simulation may depend on how fast the
   machine is (stop on steps or work done, never on elapsed time). When a
   change makes a scene simulate differently, bump `SIM_VERSION` in
   `blackbody/engine/__init__.py`: disk caches simulated by older code are
   then simulated again rather than shown.
4. Mind memory and speed. A scene should only pay for the features it uses.
   Say what a new feature costs (`python tools/benchmark.py --features`
   measures it), and keep big allocations behind the setting that needs them.
5. Don't break saved projects. A new setting gets a default that keeps old
   projects behaving as they did, and a renamed setting should still read its
   old name.
6. Don't add network access or telemetry.

## Building and testing

You need Python 3.12 or 3.13 and a GPU with Vulkan, Direct3D 12 or Metal.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip ...
.venv/bin/pip install pytest
.venv/bin/python -m blackbody                  # the app
.venv/bin/python -m pytest -q                  # the tests
```

`Blackbody.bat` (Windows) and `blackbody.sh` (macOS and Linux) do the same
setup on first run. `python -m blackbody info` lists the GPUs Blackbody can
see. `BLACKBODY_GPU` (part of a GPU's name) and `BLACKBODY_BACKEND` (`Vulkan`,
`D3D12` or `Metal`) pick one.

About half the tests need a GPU, and `tests/conftest.py` marks them `gpu`. On a
machine without one, run `pytest -m "not gpu"`, which is what CI does on
GitHub's machines. Elsewhere a GPU test whose engine cannot start fails rather
than skips (set `BLACKBODY_ALLOW_NO_GPU=1` to skip them). Liquid and fabric
tests can take several minutes, and the first run of each kind of simulation
compiles its shaders, which can take about a minute more (the liquid renderer's
march, cold again after any change to it or its includes: keep its large
functions called from few places, as `tests/test_precompile.py` checks;
`python -m blackbody precompile` compiles it ahead). While you work, run a
single file (`pytest -q tests/test_liquid.py`) or the quick tier
(`pytest -m quick`: the tests that need no GPU, and the GPU tests that took
under 10 s when they last ran on your machine). Run the whole suite
(`pytest -q`) before a pull request, and the slow tier (`pytest -m slow`: a few
frames of every preset, about five minutes, and the command-line renderer)
before a release. A test that runs for more than 10 minutes, shader compiles
aside, stops the run and shows where it was stuck (not while a debugger has
stopped it).

The GPU tests share one engine, rebuilt in place (on the same GPU, its
shaders kept) at the start of each test file and after any test that failed
with it, so what one file's scenes leave behind cannot fail another's.
`tests/test_determinism.py` simulates a small scene of each kind in two
processes at once and compares them exactly (about a minute). The window's
tests (`tests/ui/`) need no GPU: they run the main window offscreen with a
stand-in for the engine (`tests/ui/uikit.py`), its settings and files in a
folder of the run's own.

Useful tools:

- `tools/fire_check.py` measures a fire's puffing, flame height and gas speed
  against real fires.
- `tools/benchmark.py` measures speed and GPU memory at several resolutions
  (`--features` measures what each feature costs).
- `tools/render_presets.py` renders a frame of every preset, or the named
  ones, to compare before and after a change.
- `tools/make_thumbnails.py` renders the preset pictures in the app's Effects
  panel (`blackbody/assets/presets`).
- `tools/tutorial_screenshots.py` drives the app through the tutorial and
  saves its screenshots to `docs/tutorial`.
- `tools/docs_media/` renders the clips and screen recordings in `docs/media`.
  See its README.

## Documentation and the website

Every user-visible feature is documented in the guide it belongs to in
`docs/`: fire, fabric, liquids, ocean, lava, heat, weather, compositing,
scene import, outputs, command line, performance, how it works, or
troubleshooting. `README.md` is the front page, so it only gets a line for a
headline feature. A new preset goes in its guide's preset table; then run
`python tools/build_site.py --presets` with the app's environment to refresh
`docs/presets.md`.

The website is built from `docs/` and `site/` by `tools/build_site.py`. It needs
only `pip install markdown`, and `--serve` previews it on
http://localhost:8000. Pushing to `main` publishes it. In the guides, an image
of `media/gif/NAME.gif` shows as the video `media/video/NAME.mp4` on the site
when there is one. Keep media small (see `tools/docs_media/README.md`),
because every file stays in the repository's history.

## Source layout

```
blackbody/
  cli.py                 command line: render, simulate, presets, settings, info
  engine/                GPU simulation and rendering (wgpu)
    gpu.py               device, buffers, textures and the WGSL kernel loader
    engine.py            the engine: simulate, cache, render, for every kind of scene
    solver.py            the fire and smoke solver (reacting gas on a MAC grid)
    renderer.py          the volume ray-marcher and the compositor
    embers.py            embers and sparks
    liquid*.py           FLIP/APIC liquids: solver, surface rendering, floating bodies, heat
    both_engine.py       fire, liquid and lava in one box
    ocean.py, fft.py     the FFT ocean, and ocean_layer.py the open water around the box
    cloth*.py            fabric: XPBD with multigrid bending, burning and wetting
    cloud*.py, atmos.py  the cloud simulation and its renderer
    weather*.py          snow, sleet, freezing rain, graupel and hail
    mesh.py              meshes and heightfields to signed distance fields
    wgsl/                the compute and render shaders
  io/                    footage, image and video writers, EXR, OpenVDB, USD, OCIO, the tracker, the disk cache
  render/                final renders: one job writes every output
  scene/                 the scene model, settings, keyframes, presets, and Create's building blocks
  ui/                    the Qt (PySide6) app
  assets/                preset pictures, meshes and seabeds, the icon
tests/                   pytest suite (GPU tests are marked gpu: pytest -m "not gpu" runs the rest)
tools/                   checks, benchmarks, thumbnails, screenshots, docs media, the website build
docs/                    the guides (Markdown) and their media
site/                    the website's landing page, styles and scripts
```

## Pull requests

- Keep each pull request to one fix or feature.
- Add tests for new behaviour, especially anything with a physical claim
  behind it.
- Follow the style of the surrounding code.
- Run the tests on a machine with a GPU before pushing, and say which GPU and
  OS you used.
- Update the guide in `docs/` for user-visible changes, and add them to
  `CHANGELOG.md` under "Unreleased".
- For a new preset, say what real thing it is, at what size, and what you
  compared it with.

## Reporting bugs

Include your OS, the output of `python -m blackbody info` (your GPU and
backend), the Blackbody commit, what you did and what happened. A saved
project (`.bbfire`) or the preset name helps most. Footage is only needed when
the problem depends on it; check that you have the right to share it first.

Report anything that could make Blackbody run code from a file, or write or
delete files it shouldn't, privately, as described in
[SECURITY.md](SECURITY.md).

## License

Contributions are licensed under the [MIT License](LICENSE), like the rest of
the project.
