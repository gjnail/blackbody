"""The raster layer: what is drawn as triangles before the fire is marched (the cloth, cloth.py; the grass,
strands.py), into one set of targets with one depth buffer, so each hides what is behind it of the other. Its colour,
aux (view depth, distance along the ray from where the march starts, _, coverage) and glow; the march stops at it
(its aux is the march's limit), and cloth_merge.wgsl puts it under the fire, or cloth_layer.wgsl hands it to the
liquid's march."""
from __future__ import annotations

from .gpu import GPU, Uniforms


class Raster:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.size = None
        self.col = self.aux = self.glow = self.depth = None
        self.fab0 = self.fab1 = None
        self.fabric = None   # the layer on its own, gathered over the anti-aliasing passes (an output)
        self.lq = None
        self.k_merge = gpu.kernel('cloth_merge.wgsl', ['utex2d'] * 8 + ['st2d:rgba16float:w'] * 4, workgroup=(8, 8, 1))
        self.k_layer = None

    def ensure(self, w, h):
        if self.size == (w, h):
            return
        from .gpu import TU
        for name in ('col', 'aux', 'glow', 'depth', 'fab0', 'fab1'):
            t = getattr(self, name, None)
            if t is not None:
                t.destroy()
        g = self.gpu
        usage = TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC | TU.RENDER_ATTACHMENT
        self.col = g.texture2d(w, h, 'rgba16float', 'raster-col', usage)
        self.aux = g.texture2d(w, h, 'rgba16float', 'raster-aux', usage)
        self.glow = g.texture2d(w, h, 'rgba16float', 'raster-glow', usage)
        self.depth = g.texture2d(w, h, 'depth32float', 'raster-depth', TU.RENDER_ATTACHMENT)
        self.fab0 = g.texture2d(w, h, 'rgba16float', 'raster-layer0', usage)
        self.fab1 = g.texture2d(w, h, 'rgba16float', 'raster-layer1', usage)
        self.size = (w, h)
        self.fabric = self.fab0

    def begin(self, b, w, h):
        """A render pass into the layer, cleared, for one anti-aliasing pass. The caller ends it."""
        self.ensure(w, h)
        return b.render_pass(
            color_attachments=[
                {'view': self.col.view, 'load_op': 'clear', 'store_op': 'store', 'clear_value': (0, 0, 0, 0)},
                {'view': self.aux.view, 'load_op': 'clear', 'store_op': 'store', 'clear_value': (0, 0, 0, 0)},
                {'view': self.glow.view, 'load_op': 'clear', 'store_op': 'store', 'clear_value': (0, 0, 0, 0)}],
            depth_stencil_attachment={'view': self.depth.view, 'depth_clear_value': 1.0, 'depth_load_op': 'clear',
                                      'depth_store_op': 'store'})

    def layer(self, b):
        """The drawn layer as one texture for the liquid's march (cloth_layer.wgsl): colour, and distance along the ray
        (m; -1 where there is none)."""
        w, h = self.size
        g = self.gpu
        if self.lq is None or self.lq.size[:2] != (w, h):
            if self.lq is not None:
                self.lq.destroy()
            from .gpu import TU
            self.lq = g.texture2d(w, h, 'rgba16float', 'raster-liquid-layer', TU.TEXTURE_BINDING | TU.STORAGE_BINDING | TU.COPY_SRC)
        if self.k_layer is None:
            self.k_layer = g.kernel('cloth_layer.wgsl', ['utex2d', 'utex2d', 'st2d:rgba16float:w'], workgroup=(8, 8, 1))
        b.run(self.k_layer, [self.col, self.aux, self.lq], Uniforms().v4(w, h), (w, h, 1))
        return self.lq

    def merge(self, b, renderer, pass_index, passes, hold_tolerance=0.0):
        """Put the drawn layer under the marched fire (and gather the layer on its own)."""
        r = renderer
        w, h = self.size
        src_f, dst_f = (self.fab1, self.fab0) if pass_index % 2 == 0 else (self.fab0, self.fab1)
        u = Uniforms().v4(w, h, pass_index, 1.0 / max(1, passes)).v4(hold_tolerance + 1e-3)
        b.run(self.k_merge, [r.beauty, r.emit, r.aux, self.col, self.aux, self.glow, r.mask, src_f,
                             r.dof_b, r.dof_e, r.dof_x, dst_f], u, (w, h, 1))
        b.run(r.k_copy3, [r.dof_b, r.dof_e, r.dof_x, r.beauty, r.emit, r.aux], Uniforms().v4(w, h), (w, h, 1))
        self.fabric = dst_f
