"""Run with Blender's bundled Python: inspect a VDB with the official OpenVDB bindings and write a
reference file for byte-level comparison.  python pyopenvdb_check.py BLENDER_ROOT file.vdb ref.vdb report.txt"""
import os
import sys
import traceback

root, path, ref, report = sys.argv[1:5]
os.add_dll_directory(root)
os.add_dll_directory(os.path.join(root, 'blender.shared'))
log = open(report, 'w')


def say(*a):
    log.write(' '.join(str(x) for x in a) + '\n')
    log.flush()


try:
    import pyopenvdb as vdb
    say('pyopenvdb', getattr(vdb, 'LIBRARY_VERSION', '?'), 'file version', getattr(vdb, 'FILE_FORMAT_VERSION', '?'))
    grids, meta = vdb.readAll(path)
    say('FILE META', dict(meta))
    for g in grids:
        say('GRID', g.name, type(g).__name__, 'active', g.activeVoxelCount(), 'bbox', g.evalActiveVoxelBoundingBox(),
            'leaves', g.leafCount(), 'minmax', g.evalMinMax(), 'meta', {k: v for k, v in g.metadata.items() if k in ('class', 'name')})
    # reference: a small float grid with a known block of values, written by real OpenVDB
    g = vdb.FloatGrid()
    g.name = 'density'
    acc = g.getAccessor()
    for x in range(8, 12):
        for y in range(0, 3):
            for z in range(5, 7):
                acc.setValueOn((x, y, z), float(x + y * 0.1 + z * 0.01))
    g.transform = vdb.createLinearTransform(voxelSize=0.5)
    g.gridClass = vdb.GridClass.FOG_VOLUME
    vdb.write(ref, grids=[g])
    say('REF written', ref, os.path.getsize(ref))
except Exception:
    say('FAILED', traceback.format_exc())
