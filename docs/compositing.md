# Fitting it into your footage

Everything that makes the effect sit in the plate: what hides it, how it lights the footage, how the camera and lens saw it, and the colour pipeline.

<p>
<img src="media/img/compare-plate.jpg" width="49%" alt="The plate: a brick yard with an oil drum" title="The plate">
<img src="media/img/compare-comp.jpg" width="49%" alt="The same plate with a fire behind the drum, lighting the ground" title="With Blackbody: the drum hides the fire, the fire lights the ground and the wall">
</p>

## Objects in front of the effect

![A fire placed behind the drum in the plate](media/gif/in-your-footage.gif "A fire placed behind the drum in the plate: the drum hides it, its light falls on the ground and the wall, and the footage's noise and grain are matched.")

### Objects in front of the fire

A collider marked *Hides fire* (on by default) hides the fire, smoke and embers behind it, as the real object in the shot would. Put a collider over the car, wall or chair in your footage and the fire goes behind it. Turn it off for helper colliders that are not in the shot.

### Holdouts from the footage

Composite › *Holdout matte* takes a matte sequence of whatever is in front of the fire (any roto or keyed matte, numbered like the footage): the fire, smoke and embers go behind it. *Depth pass* takes the footage's depth (from a 3D scene, a lidar camera or a depth estimator; Z, distance or inverse depth, in any units): the fire is hidden only behind the footage's surfaces, so it can burn in front of an object and behind it, and the fire light lands on the footage's own surfaces, shaped from the depth pass.

## Light from the fire

### Fire light on surfaces

Composite › *Fire light on surfaces* lights the ground, every collider in the shot and (with a depth pass) the footage's own surfaces the way the fire lights the real ones: brightest where they face the flames, falling off with distance, and shaded by colliders and smoke between them and the fire (*Shadows from the fire*). It has a soft shoulder, so a big fire brightens the footage at most about four times.

### Scorch

With spreading fire, Composite › *Scorch* darkens the ground and objects where they have burnt.

### Lights in the set

*+ Light* (in the Outliner) adds a point, spot or area light: a street lamp, a stage spot, a lit window. Up to 8 of them light the smoke and steam, and the smoke shadows them (*Smoke shadows*), so a spot's beam shows in the smoke and a lamp behind the smoke glows through it. Set its *Position* (and *Aim* for a spot or area light), *Colour*, *Colour temperature* and *Intensity at 1 m* (in lux: a 100 W bulb is about 130, a street lamp a few thousand, a stage spot tens of thousands); *Size* softens the light close to it, and a spot's *Cone angle* and *Cone edge* shape its beam. A beam is as sharp as its cone however narrow, and it glows brightest looking into it, as a beam in haze does.
- *In the footage* (on) is for a real light on the set. The footage already shows its light on the real ground and objects, so Blackbody only takes away the light the simulated smoke shadows: the ground darkens under smoke that drifts in front of a street lamp.
- Turn it off for a light added in CG. It then lights the ground, the colliders in the shot and (with a depth pass) the footage's own surfaces too, shadowed by colliders and smoke.

Blackbody judges how much a light changes the footage against the light the footage already has there, the key light and sky from Lighting, so set those to match the shot. EXRs get a `lamps` layer (multiply the plate by 1 + lamps). Lights can be keyframed, and USD lights come in as lights (see [Scene import](scene-import.md#scene-import-usd)).

## Matching the camera

### Highlights like a camera's

In the Standard view, Composite › *Highlights to white* makes over-bright flame behave as it does on a camera sensor, where each colour channel catches some of the others' light: it goes from orange through yellow to white at the hottest cores, instead of clipping to a flat yellow. Footage below the roll-off is never changed. 0 gives the old per-channel roll-off. If whole flames go white, lower *Fire exposure*: real flames are rarely more than a few stops over.

### Haze

Composite › *Visibility* is how far you can see through the air in the shot: about 50 m in fog, a few hundred metres in mist, smog or dusk haze. The fire and smoke then fade into the haze over their distance from the camera, as everything in the footage has, so dark smoke is no darker than the footage's own distant shadows. The haze colour is taken from the footage (its haziest bright spots, usually the sky near the horizon) unless you turn *Haze colour from footage* off. Leave it at 0 for clear air.

### Footage noise

Composite › *Match footage noise* (on by default) measures the footage's noise in every frame: how strong it is at each brightness, how coloured and how coarse. Wherever the fire, its glow or its light change the picture, it adds noise until each pixel is as noisy as footage of that brightness, so the fire is no cleaner than the picture around it. *Grain* adds film grain on top.

### The footage's lens

Composite › *Lens* gives the fire what the camera's lens gave the footage. *Aperture* (the f-number it was shot at) and *Focus distance* (0 focuses on the fire) blur the fire by how far out of focus it is, using the camera's focal length and sensor. *Softness* blurs it as softly as the footage is: compare an edge in the footage with the fire's edges. *Lens distortion* bends the fire with the picture toward the frame edges (negative for the barrel distortion of wide lenses), *Colour fringing* adds the red and blue fringes lenses give bright edges near the corners, and *Halation* the red glow film puts around the brightest light.

### Heat haze

Composite › *Heat haze* is traced through the simulated heat: hot air is thinner than the air around it, so light from the footage behind bends by how much the path through the hot air changes across the picture. The shimmer rises with the plume and ends where the hot air ends. At 1 it has its physical strength, which is subtle for a campfire seen through a normal lens; it grows with focal length.

## Colour management

### Colour management with OCIO

Composite › *View transform: OCIO display and view* uses a display and view from an OpenColorIO config (Composite › *OCIO config*, or the `OCIO` environment variable, or the ACES CG config built into OpenColorIO): ACES, AgX, a show LUT, with an optional *Look*. The viewer uses it through a 65³ LUT; renders written to disk go through OCIO exactly. *Footage colour space: OCIO colour space* reads camera log or any other space of the config, and *EXR colour space* writes EXRs in ACEScg, ACES2065-1 or another scene-linear space.
