"""Simulation and rendering engine (GPU compute through wgpu: Vulkan, Direct3D 12 or Metal)."""

# The simulation's version, part of every disk cache's signature (Scene.sim_signature). Bump it with any change that
# makes the same scene simulate differently (how things move, break, burn or flow), so frames simulated by older code
# are simulated again rather than shown as this code's.
# 1: bottles burst and panes stand whole until hit (afd10c8); rehearsals stop on the work done, not the clock; chunks
#    and meshes share loads by area, standing meshes glued at their feet; panes hit at a corner keep no slivers
# 2: several joints on an object, ropes over several posts and with weight, concave falling meshes as convex parts,
#    hollow fixed cylinders and balls in the rigid world, chains that give way
SIM_VERSION = 2
