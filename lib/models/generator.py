import math

import chainer
from chainer import Chain, initializers, cuda
from chainer import links as L
from chainer import functions as F

from .util import n_params


class Vanilla(Chain):
    def __init__(self, *, wscale=0.02, out_shape=(3, 32, 32)):
        w = initializers.Normal(wscale)
        super().__init__(
            d0=L.Deconvolution2D(None, 256, 4, 1, 0, initialW=w),
            d1=L.Deconvolution2D(256, 128, 4, 2, 1, initialW=w),
            d2=L.Deconvolution2D(128, 64, 4, 2, 1, initialW=w),
            d3=L.Deconvolution2D(64, out_shape[0], 4, 2, 1, initialW=w),
            bn0=L.BatchNormalization(256),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(64)
        )
        self.out_shape = out_shape
        self.printed = False

    def __call__(self, z, test=False):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        h = F.relu(self.bn0(self.d0(h), test=test))
        h = F.relu(self.bn1(self.d1(h), test=test))
        h = F.relu(self.bn2(self.d2(h), test=test))
        h = F.tanh(self.d3(h))
        assert(h.shape[1:] == self.out_shape)

        if not self.printed:
            print('G(standard) params:', n_params(self))
            self.printed = True

        return h


class Residual(Chain):
    def __init__(self, *, n=1, wscale=0.02, out_shape=(3, 32, 32)):
        w = initializers.Normal(wscale)
        super().__init__(
            d0=L.Deconvolution2D(None, 256, 4, 1, 0, initialW=w),
            #s0=Stage(n, None, 256, 4, 1, 0, w),
            s1=Stage(n, 256, 128, 4, 2, 1, w),
            s2=Stage(n, 128, 64, 4, 2, 1, w),
            d1=L.Deconvolution2D(64, out_shape[0], 4, 2, 1, initialW=w),
            bn0=L.BatchNormalization(256),
        )
        self.out_shape = out_shape
        self.printed = False
        self.n = n

    def __call__(self, z, test=False):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        #h = self.s0(h, test=test)
        h = F.relu(self.bn0(self.d0(h), test=test))
        h = self.s1(h, test=test)
        h = self.s2(h, test=test)
        # h = F.relu(self.bn0(h, test=test))
        h = F.tanh(self.d1(h))
        assert(h.shape[1:] == self.out_shape)

        if not self.printed:
            print('G(residual, n={}) params:'.format(self.n), n_params(self))
            self.printed = True

        return h


class Stage(Chain):
    def __init__(self, n, ic, oc, ksize, stride, pad, w):
        super().__init__()

        blocks = [('b0', Block(ic, oc, ksize, stride, pad, w))]
        blocks += [('b{}'.format(i), Block(oc, oc, 3, 1, 1, w)) for i in range(1, n)]

        for block in blocks:
            self.add_link(*block)

        self.blocks = blocks

    def __call__(self, h, test=False):
        for _, block in self.blocks:
            h = block(h, test=test)
        return h


class Block(chainer.Chain):
    def __init__(self, ic, oc, ksize, stride, pad, w):
        super().__init__(
            d0=L.Deconvolution2D(ic, oc, ksize, stride, pad, initialW=w),
            d1=L.Deconvolution2D(oc, oc, 3, 1, 1, initialW=w),
            bn0=L.BatchNormalization(oc),
            bn1=L.BatchNormalization(oc),
        )
        self.pad = pad
        self.stride = stride
        self.ksize = ksize
        self.transforming = stride > 1 or (stride == 1 and pad == 0)
        self._initialW = w

    def __call__(self, x, test=False):
        # Main path
        h1 = F.relu(self.bn0(self.d0(x), test=test))
        h1 = self.d1(h1)

        # Shortcut path
        h2 = self.project(x, h1)

        # TODO: Separate BN here before addition or one BN after addition?
        h = self.bn1(h1 + h2, test=test)
        h = F.relu(h)
        #return F.relu(h1 + h2)
        return h

        return h

    def project(self, x, h):
        if x.shape == h.shape:
            return x
        if x.shape[1] != h.shape[1]:
            if not hasattr(self, 'proj'):
                print('Deconv projection from {} to {}'.format(x.shape, h.shape))

                in_channels = x.shape[1]
                out_channels = h.shape[1]
                assert (h.shape[2] == h.shape[3])
                assert (x.shape[2] == x.shape[3])
                """
                stride = math.ceil(h.shape[2] / x.shape[2])
                ksize = 4 if stride == 2 or (self.stride == 1 and self.pad ==0) else 3
                pad = 1 if stride == 1 or (self.stride == 1 and self.pad == 0) else self.pad
                """
                stride = self.stride
                ksize = self.ksize
                pad = self.pad

                print('ksize', ksize)
                print('stride', stride)
                print('pad', pad)

                proj = L.Deconvolution2D(in_channels, out_channels, ksize, stride, pad, initialW=self._initialW)
                if self._device_id >= 0:
                    proj.to_gpu(self._device_id)
                self.add_link('proj', proj)
            x = self.proj(x)
        return x
