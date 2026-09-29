"""Run inside Blender: load a Blackbody VDB with OpenVDB, report its grids, render it with Cycles.

blender --background --factory-startup --python tests/blender_check_vdb.py -- file.vdb out.png report.txt
"""
import sys
import traceback

import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
vdb, out_png = argv[0], argv[1]
log = open(argv[2], 'w') if len(argv) > 2 else None


def say(*a):
    line = ' '.join(str(x) for x in a)
    print(line)
    if log:
        log.write(line + '\n')
        log.flush()


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vol = bpy.data.volumes.new('fire')
    vol.filepath = vdb
    ok = vol.grids.load()
    say('LOAD_OK', ok, 'error:', vol.grids.error_message)
    for g in vol.grids:
        g.load()
        say('GRID', g.name, g.data_type, 'channels', g.channels, 'loaded', g.is_loaded,
            'matrix', [[round(x, 4) for x in r] for r in g.matrix_object])
    obj = bpy.data.objects.new('fire', vol)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    say('BOUNDS', [tuple(round(c, 3) for c in v) for v in obj.bound_box][::6])

    mat = bpy.data.materials.new('fire')
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pv = nt.nodes.new('ShaderNodeVolumePrincipled')
    pv.inputs['Density'].default_value = 4.0
    pv.inputs['Density Attribute'].default_value = 'density'
    pv.inputs['Blackbody Intensity'].default_value = 0.0
    pv.inputs['Color'].default_value = (0.2, 0.2, 0.2, 1)
    att = nt.nodes.new('ShaderNodeAttribute')
    att.attribute_name = 'flame'
    bb = nt.nodes.new('ShaderNodeBlackbody')
    bb.inputs['Temperature'].default_value = 1700
    mul = nt.nodes.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = 40.0
    nt.links.new(att.outputs['Fac'], mul.inputs[0])
    nt.links.new(mul.outputs[0], pv.inputs['Emission Strength'])
    nt.links.new(bb.outputs['Color'], pv.inputs['Emission Color'])
    nt.links.new(pv.outputs['Volume'], out.inputs['Volume'])
    vol.materials.append(mat)

    cam_data = bpy.data.cameras.new('cam')
    cam = bpy.data.objects.new('cam', cam_data)
    bpy.context.scene.collection.objects.link(cam)
    # Blender is z-up: Blackbody's y-up volume lands with its height along Blender's y unless rotated
    obj.rotation_euler = (1.5708, 0.0, 0.0)
    cam.location = (0.0, -6.5, 1.3)
    cam.rotation_euler = (1.5708, 0.0, 0.0)
    bpy.context.scene.camera = cam
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 16
    sc.cycles.device = 'CPU'
    sc.render.resolution_x, sc.render.resolution_y = 480, 270
    sc.render.filepath = out_png
    world = bpy.data.worlds.new('w')
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.02, 0.02, 0.025, 1)
    bpy.ops.render.render(write_still=True)
    say('RENDERED', out_png)


try:
    main()
except Exception:
    say('FAILED', traceback.format_exc())
