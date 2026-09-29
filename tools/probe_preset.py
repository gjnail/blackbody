import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from blackbody.engine.engine import Engine
from blackbody.scene import presets
kv = dict(a.split('=', 1) for a in sys.argv[2:])
name = sys.argv[1]
eng = Engine()
sc = presets.make(name)
sc.data['render']['width'], sc.data['render']['height'] = 640, 360
for k, v in kv.items():
    if k == 'frames':
        continue
    if k.startswith('em.'):
        key = k[3:]
        cur = sc.emitters[0][key]
        sc.emitters[0][key] = tuple(float(x) for x in v.split(',')) if isinstance(cur, tuple) else type(cur)(float(v))
    else:
        sec, key = k.split('.')
        sc.set((sec, key), v)
eng.prepare(sc, final=False)
frames = [int(x) for x in kv.get('frames', '6,12,24,48').split(',')] if 'frames' in kv else [6, 12, 24, 48]
imgs = []
for f in frames:
    eng.simulate_to(sc, sc.start + f - 1, cache=False)
    a = eng.solver.read_scalars().astype(np.float32)
    T, F, S, Fl = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    print(f'f{f:2d} vmax {eng.solver.max_speed:5.1f} sub {eng.last_substeps} bbox {eng.solver.bbox} | T max {T.max():.2f} | fuel max {F.max():.2f} | smoke max {S.max():.1f} | flame max {Fl.max():.2f} | hot {(T>0.5).mean()*100:.1f}%')
    eng.render(sc, sc.start + f - 1, (640, 360), mode='fire', samples=2)
    imgs.append(eng.display_image()[..., :3])
sheet = np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:4], 1)], 0) if len(imgs) >= 4 else np.concatenate(imgs, 1)
Image.fromarray(np.ascontiguousarray(sheet)).save(f'out/{name}_probe.png')
