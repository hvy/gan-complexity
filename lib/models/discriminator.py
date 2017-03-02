import math

import chainer
from chainer import Chain, initializers, Variable
from chainer import links as L
from chainer import functions as F

from .util import MinibatchDiscrimination, n_params


class Vanilla(Chain):
    def __init__(self, *,  wscale=0.02, use_mbd=False):
        w = initializers.Normal(wscale)
        super().__init__(
            c0=L.Convolution2D(None, 64, 4, 2, 1, initialW=w),
            c1=L.Convolution2D(64, 128, 4, 2, 1, initialW=w),
            c2=L.Convolution2D(128, 256, 4, 2, 1, initialW=w),
            bn0=L.BatchNormalization(128),
            bn1=L.BatchNormalization(256),
            #out=L.Convolution2D(256, 1, 4, 1, 0, initialW=w),
            out=L.Linear(None, 1)
        )
        self.printed = False
        self.use_mbd = use_mbd

        if self.use_mbd:
            print('D minibatch discrimination')
            self.add_link('mbd_fc', L.Linear(None, 256))
            self.add_link('mbd', MinibatchDiscrimination(None, 32, 8))
        else:
            print('D no minibatch discriminatin')

        print('D out', self.out)

    def clamp(self, lower=-0.01, upper=0.01):

        """Clamp all parameters, including the batch normalization
        parameters."""

        for params in self.params():
            params_clipped = F.clip(params, lower, upper)
            params.data = params_clipped.data

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))  # n, 64, 16, 16
        h = F.leaky_relu(self.bn0(self.c1(h), test=test))  # n, 128, 8, 8
        h = F.leaky_relu(self.bn1(self.c2(h), test=test))  # n, 256, 4, 4

        if self.use_mbd:
            h = self.mbd_fc(h)
            h = self.mbd(h)

        h = self.out(h)

        assert(h.shape[1:] in [(1, 1, 1), (1,)])
        if not self.printed:
            print('D(standard) params:', n_params(self))
            self.printed = True

        return h


class Residual(Chain):
    def __init__(self, *, n=1, wscale=0.02, use_mbd=False):
        w = initializers.Normal(wscale)
        super().__init__(
            #s0=Stage(n, None, 64, 4, 2, 1, w, skip_fst_bn=True),
            c0=L.Convolution2D(None, 64, 4, 2, 1, initialW=w),
            s1=Stage(n, 64, 128, 4, 2, 1, w, skip_fst_bn=False),
            s2=Stage(n, 128, 256, 4, 2, 1, w, skip_fst_bn=False),
            out=L.Linear(None, 1),
            #bn0=L.BatchNormalization(256)
        )
        self.printed = False
        self.n = n
        self.use_mbd = use_mbd

        if self.use_mbd:
            print('D minibatch discrimination')
            self.add_link('mbd_fc', L.Linear(None, 256))
            self.add_link('mbd', MinibatchDiscrimination(None, 32, 8))
        else:
            print('D no minibatch discriminatin')

        print('D out', self.out)

    def clamp(self, lower=-0.01, upper=0.01):

        """Clamp all parameters, including the batch normalization
        parameters."""

        for params in self.params():
            params_clipped = F.clip(params, lower, upper)
            params.data = params_clipped.data

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))
        h = self.s1(h, test=test)
        h = self.s2(h, test=test)
        #h = F.leaky_relu(self.bn0(h, test=test))

        if self.use_mbd:
            h = self.mbd_fc(h)
            h = self.mbd(h)

        h = self.out(h)

        assert(h.shape[1:] in [(1, 1, 1), (1,)])
        if not self.printed:
            print('D(res, n={}) params:'.format(self.n), n_params(self))
            self.printed = True

        return h


class Stage(Chain):
    def __init__(self, n, ic, oc, ksize, stride, pad, w, skip_fst_bn):
        super().__init__()

        blocks = [('b0', Block(ic, oc, ksize, stride, pad, w, skip_fst_bn))]
        blocks += [('b{}'.format(i), Block(oc, oc, 3, 1, 1, w, skip_fst_bn=False)) for i in range(1, n)]

        for block in blocks:
            self.add_link(*block)

        self.blocks = blocks

    def __call__(self, h, test=False):
        for _, block in self.blocks:
            h = block(h, test=test)
        return h


class Block(chainer.Chain):
    def __init__(self, ic, oc, ksize, stride, pad, w, skip_fst_bn):
        super().__init__(
            c0=L.Convolution2D(ic, oc, ksize, stride, pad, initialW=w),
            c1=L.Convolution2D(oc, oc, 3, 1, 1, initialW=w),
            bn0=L.BatchNormalization(oc),
            #c0_2=L.Convolution2D(ic, oc, ksize, stride, pad, initialW=w),
            #c1_2=L.Convolution2D(oc, oc, 3, 1, 1, initialW=w),
            #bn1_2=L.BatchNormalization(oc)
        )

        if not skip_fst_bn:
            self.add_link('bn1', L.BatchNormalization(oc))

        self.pad = pad
        self.transforming = stride > 1
        self._initialW = w

        print('Discriminator Block stride:', stride)

    def __call__(self, x, test=False):
        h1 = F.leaky_relu(self.bn0(self.c0(x), test=test))
        h1 = self.c1(h1)

        # Shortcut path
        h2 = self.project(x, h1)

        h = h1 + h2

        if hasattr(self, 'bn1'):
            h = self.bn1(h, test=test)


        h = F.leaky_relu(h)
        return h

    def project(self, x, h):
        if x.shape == h.shape:
            return x
        if x.shape[1] != h.shape[1]:  # Halve number of feature maps
            if not hasattr(self, 'proj'):
                print('Conv projection from {} to {}'.format(x.shape, h.shape))

                in_channels = x.shape[1]
                out_channels = h.shape[1]
                assert (h.shape[2] == h.shape[3])
                assert (x.shape[2] == x.shape[3])
                stride = math.ceil(x.shape[2] / h.shape[2])
                ksize = 4 if stride == 2 else 3
                pad = 1 if stride == 1 else self.pad

                print('ksize', ksize)
                print('stride', stride)
                print('pad', pad)

                proj = L.Convolution2D(in_channels, out_channels, ksize, stride, pad, initialW=self._initialW)
                if self._device_id >= 0:
                    proj.to_gpu(self._device_id)
                self.add_link('proj', proj)
            x = self.proj(x)
        return x
