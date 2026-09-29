"""Inside Blender: prove the voxel data of a VDB reads correctly (volume-to-mesh vertex count + bounds)."""
import sys
import traceback

import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
vdb, report = argv[0], argv[1]
log = open(report, 'w')


def say(*a):
    log.write(' '.join(str(x) for x in a) + '\n')
    log.flush()


try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.volume_import(filepath=vdb, files=[{'name': vdb.replace('\\', '/').split('/')[-1]}],
                                 directory=vdb.rsplit('\\', 1)[0] + '\\')
    vobj = [o for o in bpy.data.objects if o.type == 'VOLUME'][0]
    dg = bpy.context.evaluated_depsgraph_get()
    ev = vobj.evaluated_get(dg)
    say('EVAL_BOUNDS', [tuple(round(c, 3) for c in v) for v in ev.bound_box][0], [tuple(round(c, 3) for c in v) for v in ev.bound_box][6])
    for grid_name, thr in (('density', 0.05), ('flame', 0.05), ('temperature', 0.2)):
        me = bpy.data.meshes.new('m')
        mobj = bpy.data.objects.new('m_' + grid_name, me)
        bpy.context.scene.collection.objects.link(mobj)
        mod = mobj.modifiers.new('v2m', 'VOLUME_TO_MESH')
        mod.object = vobj
        mod.grid_name = grid_name
        mod.threshold = thr
        mod.resolution_mode = 'GRID'
        dg = bpy.context.evaluated_depsgraph_get()
        mev = mobj.evaluated_get(dg)
        mesh = mev.to_mesh()
        zs = [v.co[1] for v in mesh.vertices]
        say('MESH', grid_name, 'verts', len(mesh.vertices), 'y range', (round(min(zs), 3), round(max(zs), 3)) if zs else None)
except Exception:
    say('FAILED', traceback.format_exc())
