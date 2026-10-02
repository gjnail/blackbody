# Lume benchmark

Measures Lume, Blackbody's path-traced lighting, against ground truth: the same scenes rendered by **Mitsuba 3**, a research path tracer, on the GPU with OptiX. Mitsuba renders them with **Lume's own material** (written as a Mitsuba BSDF in `mitsuba_side.py`), so any difference is in how Lume carries the light, not in the material.

## Setup

Mitsuba runs in its own Python environment, so Blackbody's stays as it is:

```
python -m venv mitsuba-env
mitsuba-env\Scripts\python -m pip install mitsuba numpy pillow
```

## Run

From the repository, in Blackbody's environment:

```
python tools/lume_bench/run.py --mitsuba-python mitsuba-env\Scripts\python.exe
```

It renders every scene in `scenes.py` with Lume (`lume_side.py`) and with Mitsuba (`mitsuba_side.py`, a 16384-sample reference plus timed renders), then writes `out/lume_bench/DATE/report.md` and a picture per scene: Lume, the reference, and where they differ (green equal, red Lume darker, blue Lume brighter, full colour at 20%). `--skip-render` reports on pictures already there; name scenes to run only those.

## The scenes

| Scene | What it checks |
|---|---|
| furnace | A white ball in a uniform sky: energy kept, the sky's light, multiple bounces |
| cornell | An open box with red and green sides and a lamp inside: light bouncing from wall to wall |
| cornell_direct, cornell_2 | The same with one bounce (direct light only) and two: which bounce a difference is in |
| glossy | Rough and smooth metal and plastic balls under a distant lamp: highlights and reflections |
| glass | A glass ball beside a block under a lamp: refraction, its shadow, the light it focuses |

## The measures

- **Bias**: Lume's mean brightness at its most samples over the reference's. 1.000 is exact.
- **Error**: relative mean squared error, `mean((x - ref)^2 / (ref^2 + 0.01))`, at each sample count, for both renderers.
- **Equal time**: each renderer's error after the same render time, and how long each takes to get under 0.01.

The lamps themselves are left out of the measures (Lume does not draw them for the camera; Mitsuba does).

## Results, 2026-10-02 (RTX 3090)

| Scene | Bias | Error, Lume at 1024 samples | Equal-time error, Lume / Mitsuba |
|---|---|---|---|
| furnace | 1.0002 | 0.00004 | (both under a millisecond) |
| cornell | 0.9975 | 0.0027 | 3.0x |
| cornell_direct | 0.9953 | 0.0010 | 1.2x |
| cornell_2 | 0.9966 | 0.0022 | 2.2x |
| glossy | 1.0085 | 0.0019 | 4.9x |
| glass | 0.9743 | 0.0055 | 1.6x |

What is left:

- **Glass**: Lume lets light through glass along shadow rays (by its Fresnel losses and tint), straight through. That is exact for a pane, but a curved piece's focused light (a caustic) is not found: the 2.6% in *glass*.
- **Speed**: Mitsuba traces with the GPU's ray-tracing cores (OptiX); Lume traces in compute shaders, plain shapes exactly and meshes and matter by marching. At equal time Lume has 1.2–5x Mitsuba's error.

What the benchmark has caught and fixed so far: the last bounce dropping the sky's share of light (all scenes up to 8x too dark in sky-lit shade), a path clamp that removed most of an HDRI sun's bounced light, lamps treated as soft points (3% dark near them), rays off a ball's underside slipping under the floor (bright contact rings), and ray offsets of a pixel or two putting surfaces that much nearer the lamps (up to 3% bright).
