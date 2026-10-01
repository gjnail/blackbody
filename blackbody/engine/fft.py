"""2D FFTs on the GPU: radix-2 Stockham passes over square power-of-two grids.

Data lives in rgba32float 3D textures of n x n x layers texels, each texel two complex numbers
(xy and zw), and every layer is transformed in the same dispatches. The inverse transform is the
plain sum f(x) = sum_k F(k) exp(+2 pi i k x / n), unnormalised; the forward one uses exp(-...),
so forward then inverse multiplies by n^2.
"""
from __future__ import annotations

import math

from .gpu import Uniforms


class FFT2D:
    """Ping-pong textures and the pass kernel for n x n x layers complex pairs."""

    def __init__(self, gpu, n, layers, label='fft'):
        n = int(n)
        if n < 2 or n & (n - 1):
            raise ValueError(f'FFT size must be a power of two, got {n}')
        self.gpu = gpu
        self.n = n
        self.layers = int(layers)
        self.log2 = int(round(math.log2(n)))
        self.a = gpu.texture3d((n, n, self.layers), 'rgba32float', f'{label}-a')
        self.b = gpu.texture3d((n, n, self.layers), 'rgba32float', f'{label}-b')
        self.k = gpu.kernel('ocn_fft.wgsl', ['utex3d', 'st3d:rgba32float:w'], workgroup=(8, 8, 1))
        self._u = {}

    def destroy(self):
        for t in (self.a, self.b):
            t.destroy()

    def _uniforms(self, p, axis, sign):
        key = (p, axis, sign)
        u = self._u.get(key)
        if u is None:
            u = self._u[key] = Uniforms().v4(self.n, p, axis, sign).tobytes()
        return u

    def run(self, b, inverse=True):
        """Transform what is in self.a, in both axes. The passes come in pairs, so the result is
        back in self.a."""
        sign = 1.0 if inverse else -1.0
        src, dst = self.a, self.b
        size = (self.n // 2, self.n, self.layers)
        for axis in (0, 1):
            for s in range(self.log2):
                b.run(self.k, [src, dst], self._uniforms(1 << s, axis, sign), size)
                src, dst = dst, src
        assert src is self.a
        return self.a

    @property
    def nbytes(self):
        return 2 * self.n * self.n * self.layers * 16
