# Install and first steps

Install Blackbody, find your way around the window, and put a fire into a shot in five steps. The [tutorial](tutorial.md) then walks through a whole shot, from footage to final render.

## Requirements

- A GPU with Vulkan, Direct3D 12 or Metal. Most cards from the last eight years qualify: NVIDIA, AMD, Intel, and Apple Silicon.
- 4 GB of GPU memory for everyday work; 8 GB or more for high-resolution final renders and big liquid shots.
- Windows 10/11. macOS 12+ and Linux run from source too, but have not been tested yet (see [limitations](troubleshooting.md#limitations)).
- Python 3.10 or newer, to run from source.

## Install and run

Get the source, either with Git:

```bash
git clone https://github.com/gjnail/blackbody.git
cd blackbody
```

or as a ZIP (on GitHub: **Code › Download ZIP**), unpacked anywhere.

Then start it:

- **Windows:** double-click `Blackbody.bat`.
- **macOS and Linux:** run `./blackbody.sh`.

The first run creates a private Python environment in `.venv` and installs the dependencies into it (about 400 MB, a few minutes). After that it starts straight away. The first time each kind of simulation runs, its GPU shaders are compiled, which takes a moment for fire and a few minutes for liquids; a card in the viewer says what it is doing.

Or set it up yourself:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip ...
.venv/bin/python -m blackbody               # the app
.venv/bin/python -m blackbody render --help # the command line
```

**A standalone Windows app.** `pip install pyinstaller`, then `pyinstaller blackbody.spec`, builds `dist/Blackbody/` with `Blackbody.exe` (the app) and `blackbody-cli.exe` (the command line, for batch and render-farm use). The folder runs on machines without Python.

## The window

![The Blackbody window: Effects, viewer, Properties and timeline](tutorial/02_window.png "1 Effects · 2 view bar · 3 viewer · 4 Properties · 5 timeline")

Across the top, the header holds what you do most: **Import footage**, **Open**, **Save**, undo and redo, and **Render**.

1. **Effects and Create.** Two tabs. *Effects* has the built-in fires, smoke, sparks, steam, liquids, seas and skies, in categories (Fire, Blasts, Smoke, Water, Sea, Ice & heat, Sky, Fire + water, and Mine for your own presets) with a search box (**Ctrl+E**). One click loads an effect; **Ctrl+Z** brings back the one before. *Create* (**Ctrl+N**) is for making your own: start an empty scene, and add building blocks to it (see [Build your own](#build-your-own)).
2. **View bar.** What the viewer shows: the composite, the fire over black, its alpha, and passes such as heat and depth (keys **1** to **7**). *Preview* sets the resolution of the live preview, *Guides* shows or hides the handles and the simulation box (**G**), and *Stats* shows the simulation grid, cell size and timings over the picture.
3. **Viewer.** Your frame. Drop a clip, an image sequence or a scene file anywhere on the window to open it.
4. **Properties.** A bar of pages down its left side. **Essentials** has the dozen or so settings that matter most for the kind of effect loaded. **Objects** lists what is in the shot (emitters, colliders, lights, fabrics), with an *Add* menu, and the settings of the one selected. Below those, every section of settings has its own page. The search box at the top (**Ctrl+F**) finds any setting by name, across all the pages, and lets you change it right there. Tick *Advanced* for expert settings.
5. **Timeline.** Play controls, the frame range, and a bar that turns green as frames are simulated and cached.

## Put fire in a shot in five steps

1. **Import your footage.** File › Import footage (Ctrl+I). Video files, image sequences (EXR, DPX, TIFF, PNG) and stills all work. The frame size, frame rate and length follow the footage.
2. **Pick a fire.** Click one in Effects. It plays from its first frame. Until it is ready, a card in the viewer shows what it is doing: the pre-roll, or the first-run shader compile, which can take a few minutes for liquids. With *Keep my shot* on, your footage, camera and output settings stay. Before you import footage, each preset brings its own camera and frame range.

   ![Click a preset in Effects and it simulates live](media/gif/ui-library.gif "Click a preset in Effects: it loads and simulates live.")

3. **Place it.**
   - Drag the ring at the fire's base onto the spot, and drag the square above it to scale.
   - Alt-drag orbits the camera around the fire; Alt + right-drag changes its distance.
   - Drag an emitter to move it along the ground (Shift moves it up and down).

   ![Dragging the ring onto the ground, then the square to scale the fire](media/gif/ui-place.gif "Drag the ring onto the ground, then the square to scale the fire to the shot.")

4. **Match it.** Essentials has the main controls under *Blend with the footage*; Composite has all of them:
   - Fire exposure (Shading)
   - Heat haze
   - Fire light on the footage
   - Glow and grain

   Smoke is lit by the average colour of the footage automatically (Lighting › Match ambient to footage). [Fitting it into your footage](compositing.md) has every tool for this.
5. **Render** (Ctrl+M). Pick one or more outputs: a finished composite, an element with alpha, multi-layer or deep EXRs, or OpenVDB volumes. See [Outputs](outputs.md).

   ![The render dialog](media/gif/ui-render.gif "The render dialog: tick the outputs you want, all written in one pass.")

Space plays. The simulation runs live, so you can change settings while it plays. Frames are cached (the green bar in the timeline): scrubbing back, and changing anything about the look, camera or composite, re-renders from the cache without re-simulating.

## Build your own

Blackbody opens in **Build**: an empty stage you look at with a camera of your own (drag empty space to look around, right- or middle-drag to pan, the wheel to move closer, **F** to frame everything). **Shot**, next to it in the header, is where an effect goes into your footage: the shot's camera, placement on the plate, roto, layers and the render. **Tab** or **W** switches between them; nothing you do in Build moves the shot's camera.

1. **Put something in.** The card on the empty stage starts you with fire, water, cloth, smoke, snow or a solid object. After that, **Create** (the second tab on the left, **Ctrl+N**) has everything: fire sources (a fire, a campfire, a fuel pool, a fire line, a torch, a gas burner, a fireball, a flame jet, a fire whirl, coloured flame, burnable things), smoke, steam and sparks, **liquids** (a pour, a hose jet, a fountain, a block of water, thrown water, a waterfall, a pond, ink, lava), **fabric** (a curtain, a flag, a banner, a canopy, a tablecloth, a falling sheet, a wet towel, your own cloth mesh), **weather** (rain, snow, sleet, hail, freezing rain), **forces** (wind, a fan, an updraft, suction, a vortex: air that pushes smoke, flame, embers and cloth around), **objects** (boxes, balls, pillars, walls, a room with a door, a car, floating crates, terrain, your own meshes) and **lights**. Click one, or drag it into the viewer and drop it where it should stand.
2. **Move and shape it.** The selected thing has a gizmo: drag an arrow to move it along X (red), Y (green) or Z (blue), a square to stretch it along that side, the ring's knob to turn it, and its middle to slide it over the ground. Hold **Ctrl** to snap to a grid sized for the scene. The size or place shows by the cursor as you drag.
3. **Make things happen.** Right-click anything in the viewer (or in the object list, or use the buttons on its page) for what it can do:
   - **Set it on fire**: a box, a chair or a curtain catches at its base and the fire spreads over it.
   - **Make it float**, **make it sink**: in the water, or in a pond put under it if there is none. **Make it hot** (water on it boils) or **freezing** (water on it freezes). **Make it hollow**: a tank, a room or a pipe.
   - **Soak** a cloth (it drips, steams in the heat and will not burn until it dries), **let it go** at this frame, choose what it is **held** by and what it is made of.
   - **Turn into**: any source becomes fire, smoke, steam, water, lava, a fan, an updraft, suction or a vortex, keeping its shape and place.
   - **Start** or **stop** a source at this frame, or give a **short burst** from it.
   - **Attach to** another thing: it then goes wherever that goes (a torch in a moving hand, a flag on a moving pole, fire on a driving car). Moving it by hand keeps its new place relative to what it is attached to.
   - **Move along a path**: click points on the ground where it should go, set how long it takes, and press Enter.
   - **Drop to the ground**, **look at it** (the Build camera turns to it), duplicate, delete.
4. **Tune it** in Properties: everything about the selected thing, the Essentials of the scene, and every other setting a search away.

Generic blocks are sized for the scene; real things (a torch, a car, a curtain) keep their real size. If a block does not fit, the simulation box grows to take it. Whatever you put in simulates together: water puts fire out, fire burns cloth, wind blows the smoke and the flags, lava boils the sea. The header shows what the scene simulates. Every change is one step of undo.

![Building a scene in the Create tab](media/gif/ui-create.gif "Create: start an empty scene, then click building blocks to add them: here a campfire, a curtain and wind.")

## Layers

A shot can hold several effects, each its own simulation with its own box, scale and settings: a campfire in the foreground and a waterfall in the distance, smoke from a chimney behind a burning car. Click **Layer** in the header (**Ctrl+L**) to add one in front of the others, then load an effect into it from **Effects** or build one in **Create**.

The strip over the viewer lists the layers from back to front; the lit one is the layer you are editing, and Properties, Effects and Create all work on it. Click a layer to edit it, double-click to rename it, and right-click to hide it, move it back or forward, duplicate it or delete it. Each layer is placed in the shot by its own ring and square.

Every layer is drawn over the ones behind it as if they were its footage, so its heat haze, glow and light land on them too. The layers share the shot: the footage, the frame size, rate and range, the output look, the holdouts and roto, and the camera track (each layer keeps its own place on it). Renders, the command line and *Export this frame* draw all the layers; element outputs merge them, with the extra passes (emission, depth and so on) from the base layer. Effects that have to touch, such as a hose on a fire, belong in one simulation, not in two layers.

## Animate

Any setting with a **◇** next to it can be keyframed: click the ◇ to key it at the current frame, then change the frame and the value. **Animation** at the right of the timeline opens the animation editor: every animated setting is a row, with its keys on a track and a trace of its value. Drag keys to move them in time (Shift-click to select several), double-click a track to add a key, press **Delete** to remove the selected ones, and right-click to choose how a key eases into the next (*Smooth*, *Linear* or *Hold*). Click a setting's name to bring it up in Properties.

## Working in the app

Every setting has a tooltip. A number is a bar: drag it sideways to change it (Shift for fine steps), or click it to type a value (sums such as `1/24` work too). You can also drag a setting's name to scrub it, and double-click the name for its default. A setting you have changed from the effect you loaded shows its name in orange with a ↺ button that puts the effect's value back. Tick *Advanced* in Properties for expert settings. **Save as preset** (under Effects) keeps your own fires and liquids under *Mine*.

![Scrubbing the wind speed: the flames lean over while the simulation keeps running](media/gif/ui-scrub.gif "Drag a setting's name to scrub it. Here the wind speed leans the flames over while the simulation runs.")

In the viewer, click a source, an object, a light or a piece of fabric to select it and drag it along the ground (Shift: up and down). A selected source or object shows a ring with a knob to turn it and a square to resize it (Shift snaps the turn to 15°).

In **Build** you look at the scene with a camera of your own; the shot's camera is drawn as a yellow frustum so you can see what the shot sees. Only the viewer changes: the shot, its camera and every render stay as they are, and turning the view re-draws cached frames without simulating again. **Shot** (or **Tab**) goes back to the shot's camera over your footage.

**Roto** (in the view bar, or **R**) draws shapes around what is in front of the effect in your footage, and the effect goes behind them: see [Roto](compositing.md#roto).

The view bar switches what the viewer shows, the way a compositor checks an element: the composite, the effect alone over black, its alpha, emission, heat (the haze driver), depth and temperature. In a liquid scene the last views show what the liquid does to the footage (wet ground, shadows, caustics) and how fast it moves.

![Cycling the view modes: composite, fire, alpha, emission, heat, depth, temperature](media/gif/ui-views.gif "Keys 1 to 7 cycle the view: composite, fire, alpha, emission, heat, depth and temperature.")

## Where next

- [Tutorial: your first shot](tutorial.md): a whole shot, from footage to render, in about 30 minutes.
- [Presets](presets.md): all the built-in effects, each a starting point.
- The guides for [fire](fire.md), [liquids](liquids.md), [the sea](ocean.md), [lava](lava.md), [ice and steam](heat.md), [fabric](fabric.md) and [weather and clouds](weather.md).
