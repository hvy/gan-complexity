import math

import chainer
from chainer import Chain, initializers, Variable
from chainer import functions as F
from chainer import cuda

from .util import MinibatchDiscrimination, n_params

from .layers import Convolution, Deconvolution, Linear, BatchNorm


class VanillaMnist(Chain):
    def __init__(self, *, use_mbd=False):
        super().__init__(
            c1=Convolution(1, 64, 4, 2, 1),
            c2=Convolution(64, 128, 4, 2, 1),
            bn=BatchNorm(128),
            fc=Linear(None, 1)
        )

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c1(x))
        h = F.leaky_relu(self.bn(self.c2(h), test=test))
        h = self.fc(h)
        return h


class ResidualMnist(Chain):
    # TODO: Make sure that batch norm option is passed to Stage
    def __init__(self, *, n=1, use_bn=False, use_mbd=False):
        super().__init__(
            c=Convolution(1, 64, 4, 2, 1),
            s=Stage(n, 64, 128, 4, 2, 1, skip_fst_bn=False),
            fc=Linear(None, 1)
        )
        if use_mbd:
            self.add_link('mbd_fc', Linear(None, 64))
            self.add_link('mbd', MinibatchDiscrimination(None, 32, 8))

        self.n = n
        self.use_bn = use_bn
        self.use_mbd = use_mbd

        # print('D, batch normalization: '.format(use_bn))
        print('D, minibatch discrimination: '.format(use_mbd))

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c(x))
        h = self.s(h, test=test)
        if self.use_mbd:
            h = self.mbd_fc(h)
            h = self.mbd(h)
        h = self.fc(h)
        assert(h.shape[1:] in [(1, 1, 1), (1,)])
        return h


class Vanilla(Chain):
    def __init__(self, *,  use_mbd=False):
        super().__init__(
            c0=Convolution(None, 64, 4, 2, 1),
            c1=Convolution(64, 128, 4, 2, 1),
            c2=Convolution(128, 256, 4, 2, 1),
            bn0=BatchNorm(128),
            bn1=BatchNorm(256),
            #out=Convolution2D(256, 1, 4, 1, 0),
            out=Linear(None, 1)
        )
        self.use_mbd = use_mbd

        if self.use_mbd:
            self.add_link('mbd_fc', Linear(None, 64))
            self.add_link('mbd', MinibatchDiscrimination(None, 32, 8))

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))  # n, 64, 16, 16
        # TODO: Remove me
        h = F.leaky_relu(self.bn0(self.c1(h), test=test))  # n, 128, 8, 8
        # h = F.leaky_relu(self.c1(h))

        # TODO: Remove me
        h = F.leaky_relu(self.bn1(self.c2(h), test=test))
        # h = F.leaky_relu(self.c2(h))

        # Minibatch discrimination
        if self.use_mbd:
            h = self.mbd_fc(h)
            h = self.mbd(h)

        h = self.out(h)
        assert(h.shape[1:] in [(1, 1, 1), (1,)])
        return h


class Residual(Chain):
    def __init__(self, *, n=1, use_mbd=False):
        super().__init__(
            #s0=Stage(n, None, 64, 4, 2, 1, w, skip_fst_bn=True),
            c0=Convolution(None, 64, 4, 2, 1),
            s1=Stage(n, 64, 128, 4, 2, 1, skip_fst_bn=False),
            s2=Stage(n, 128, 256, 4, 2, 1, skip_fst_bn=False),
            out=Linear(None, 1),
            #bn0=BatchNorm(256)
        )
        self.n = n
        self.use_mbd = use_mbd

        if self.use_mbd:
            print('D minibatch discrimination')
            self.add_link('mbd_fc', Linear(None, 64))
            self.add_link('mbd', MinibatchDiscrimination(None, 32, 8))
        else:
            print('D no minibatch discriminatin')

        print('D out', self.out)

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
        return h


class Stage(Chain):
    def __init__(self, n, ic, oc, ksize, stride, pad, skip_fst_bn):
        super().__init__()

        blocks = [('b0', Block(ic, oc, ksize, stride, pad, skip_fst_bn))]
        blocks += [('b{}'.format(i), Block(oc, oc, 3, 1, 1, skip_fst_bn=False)) for i in range(1, n)]

        for block in blocks:
            self.add_link(*block)

        self.blocks = blocks

    def __call__(self, h, test=False):
        for _, block in self.blocks:
            h = block(h, test=test)
        return h


class Block(chainer.Chain):
    def __init__(self, ic, oc, ksize, stride, pad, skip_fst_bn):
        super().__init__(
            c0=Convolution(ic, oc, ksize, stride, pad),
            c1=Convolution(oc, oc, 3, 1, 1),
            bn0=BatchNorm(oc),
            #c0_2=Convolution(ic, oc, ksize, stride, pad),
            #c1_2=Convolution(oc, oc, 3, 1, 1),
            #bn1_2=BatchNorm(oc)
        )

        if not skip_fst_bn:
            self.add_link('bn1', BatchNorm(oc))
        self.add_link('bn2', BatchNorm(oc))

        self.pad = pad
        self.transforming = stride > 1

        print('Discriminator Block stride:', stride)

    def __call__(self, x, test=False):
        # TODO: Replace me back
        # h1 = F.leaky_relu(self.bn0(self.c0(x), test=test))
        h1 = F.leaky_relu(self.c0(x))

        h1 = self.c1(h1)

        # TODO: Replace me back
        # h1 = self.bn1(h1, test=test)

        # Shortcut path
        h2 = self.project(x, h1)

        if x.shape != h1.shape:
            # TODO: Replace me back
            # h2 = self.bn2(h2, test=test)
            pass

        h = h1 + h2
        # h = h1

        """
        if hasattr(self, 'bn1'):
            h = self.bn1(h, test=test)
        """

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

                # Assert feature maps are squares
                assert (h.shape[2] == h.shape[3])
                assert (x.shape[2] == x.shape[3])

                # Standard projection
                """
                stride = math.ceil(x.shape[2] / h.shape[2])
                ksize = 4 if stride == 2 else 3
                pad = 1 if stride == 1 else self.pad
                proj = Convolution(in_channels, out_channels, ksize, stride, pad)
                """

                # Linear projection (1x1 conv)
                proj = Convolution(in_channels, out_channels, 1, 2, 0)

                if self._device_id >= 0:
                    proj.to_gpu(self._device_id)
                self.add_link('proj', proj)

            x = self.proj(x)
        return x
