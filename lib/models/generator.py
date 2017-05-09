import chainer
from chainer import Chain
from chainer import functions as F
from chainer import initializers
from chainer import links as L

from math import log2
from .layers import Convolution, Deconvolution, Linear, BatchNorm


# CIFAR-10

"""
class Standard32(Chain):
    def __init__(self):
        super().__init__(
            fc=Linear(None, 256*4*4),
            d1=Deconvolution(256, 128, 4, 2, 1),
            d2=Deconvolution(128, 64, 4, 2, 1),
            d3=Deconvolution(64, 3, 4, 2, 1),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(64))

    def __call__(self, z, test=False):
        h = self.fc(z)
        h = F.reshape(h, (h.shape[0], 256, 4, 4))
        h = F.relu(self.bn1(self.d1(h), test=test))
        h = F.relu(self.bn2(self.d2(h), test=test))
        h = F.tanh(self.d3(h))
        assert(h.shape[1:] == (3, 32, 32))
        return h

class Standard28(Chain):
    def __init__(self):
        super().__init__(
            fc=Linear(None, 128*7*7),
            d1=Deconvolution(128, 128, 4, 2, 1),
            c1=Convolution(128, 128, 3, 1, 1),
            bn2=BatchNorm(128),
            c2=Convolution(128, 128, 3, 1, 1),
            bn3=BatchNorm(128),
            c3=Convolution(128, 128, 3, 1, 1),
            bn4=BatchNorm(128),
            c4=Convolution(128, 128, 3, 1, 1),
            bn5=BatchNorm(128),
            c5=Convolution(128, 128, 3, 1, 1),
            bn6=BatchNorm(128),
            c6=Convolution(128, 128, 3, 1, 1),
            bn7=BatchNorm(128),
            d2=Deconvolution(128, 1, 4, 2, 1),
            bn1=BatchNorm(128))

    def __call__(self, z, test=False):
        h = self.fc(z)
        h = F.reshape(h, (h.shape[0], 128, 7, 7))
        h = F.relu(self.bn1(self.d1(h), test=test))
        h = F.relu(self.bn2(self.c1(h), test=test))
        h = F.relu(self.bn3(self.c2(h), test=test))
        h = F.relu(self.bn4(self.c3(h), test=test))
        h = F.relu(self.bn5(self.c4(h), test=test))
        h = F.relu(self.bn6(self.c5(h), test=test))
        h = F.relu(self.bn7(self.c6(h), test=test))
        h = F.tanh(self.d2(h))
        assert(h.shape[1:] == (1, 28, 28))
        return h
"""


class VanillaMnist(Chain):
    def __init__(self):
        super().__init__(
            dc1=Deconvolution(None, 128, 7, 1, 0),
            dc2=Deconvolution(128, 64, 4, 2, 1),
            dc3=Deconvolution(64, 1, 4, 2, 1),
            bn1=BatchNorm(128),
            bn2=BatchNorm(64)
        )

    def __call__(self, z, test=False):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.tanh(self.dc3(h))
        assert(h.shape[1:] == (1, 28, 28))
        return h


class ResidualMnist(Chain):
    def __init__(self, *, n=1):
        super().__init__(
            dc1=Deconvolution(None, 128, 7, 1, 0),
            s=Stage(n, 128, 64, 4, 2, 1),
            dc2=Deconvolution(64, 1, 4, 2, 1),
            bn1=BatchNorm(128)
        )
        self.n = n

    def __call__(self, z, test=False):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = self.s(h, test=test)
        h = F.tanh(self.dc2(h))
        assert(h.shape[1:] == (1, 28, 28))
        return h


class Vanilla(Chain):
    def __init__(self, *, out_shape=(3, 32, 32)):

        # Change the padding of one of the deconv layers to match the desired
        # final output shape
        if out_shape[1:] == (32, 32):  # CIFAR-10
            d2_pad = 1
        elif out_shape[1:] == (28, 28):  # MNIST
            d2_pad = 2
        else:
            raise ValueError('Unsupported out shape', out_shape)

        super().__init__(
            d0=Deconvolution(None, 256, 4, 1, 0),
            d1=Deconvolution(256, 128, 4, 2, 1),
            d2=Deconvolution(128, 64, 4, 2, d2_pad),
            d3=Deconvolution(64, out_shape[0], 4, 2, 1),
            bn0=BatchNorm(256),
            bn1=BatchNorm(128),
            bn2=BatchNorm(64)
        )
        self.out_shape = out_shape

    def __call__(self, z, test=False):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        h = F.relu(self.bn0(self.d0(h), test=test))
        h = F.relu(self.bn1(self.d1(h), test=test))
        h = F.relu(self.bn2(self.d2(h), test=test))
        h = F.tanh(self.d3(h))
        assert(h.shape[1:] == self.out_shape)
        return h


class Residual(Chain):
    def __init__(self, *, n=1, out_shape=(3, 32, 32)):

        if out_shape[1:] == (32, 32):  # CIFAR-10
            s2_pad = 1
        elif out_shape[1:] == (28, 28):  # MNIST
            s2_pad = 2
        else:
            raise ValueError('Unsupported out shape', out_shape)

        super().__init__(
            d0=Deconvolution(None, 256, 4, 1, 0),
            s1=Stage(n, 256, 128, 4, 2, 1),
            s2=Stage(n, 128, 64, 4, 2, s2_pad),
            d1=Deconvolution(64, out_shape[0], 4, 2, 1),
            bn0=BatchNorm(256),
        )
        self.out_shape = out_shape
        self.n = n

    def __call__(self, z, test=False):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        #h = self.s0(h, test=test)

        # TODO: Replace me back
        h = F.relu(self.bn0(self.d0(h), test=test))
        # h = F.relu(self.d0(h))

        h = self.s1(h, test=test)
        h = self.s2(h, test=test)
        # h = F.relu(self.bn0(h, test=test))
        h = F.tanh(self.d1(h))
        assert(h.shape[1:] == self.out_shape)

        return h


class Stage(Chain):
    def __init__(self, n, ic, oc, ksize, stride, pad):
        super().__init__()

        blocks = [('b0', Block(ic, oc, ksize, stride, pad))]
        blocks += [('b{}'.format(i), Block(oc, oc, 3, 1, 1)) for i in range(1, n)]

        for block in blocks:
            self.add_link(*block)

        self.blocks = blocks

    def __call__(self, h, test=False):
        for _, block in self.blocks:
            h = block(h, test=test)
        return h


class Block(chainer.Chain):
    def __init__(self, ic, oc, ksize, stride, pad):
        super().__init__(
            d0=Deconvolution(ic, oc, ksize, stride, pad),
            d1=Deconvolution(oc, oc, 3, 1, 1),
            bn0=BatchNorm(oc),
            bn1=BatchNorm(oc),
            bn2=BatchNorm(oc),
        )
        self.pad = pad
        self.stride = stride
        self.ksize = ksize
        self.transforming = stride > 1 or (stride == 1 and pad == 0)

    def __call__(self, x, test=False):
        # Main path

        # TODO: Replace me back
        h1 = F.relu(self.bn0(self.d0(x), test=test))
        # h1 = F.relu(self.d0(x))

        # Uncomment to NOT use BatchNorm before addition
        # h1 = self.d1(h1)
        # Read above

        # TODO: Replace me back
        h1 = self.bn1(self.d1(h1), test=test)
        # h1 = self.d1(h1)

        # Shortcut path
        h2 = self.project(x, h1)
        if x.shape != h1.shape:  # If projection was applied
            # TODO: Replace me back
            h2 = self.bn2(h2, test=test)
            # pass

        h = h1 + h2
        # h = h1

        # Shared BN, after addition
        # h = self.bn1(h, test=test)

        h = F.relu(h)
        return h

    def project(self, x, h):
        if x.shape == h.shape:
            return x
        if x.shape[1] != h.shape[1]:
            if not hasattr(self, 'proj'):
                print('Deconv projection from {} to {}'.format(x.shape, h.shape))

                in_channels = x.shape[1]
                out_channels = h.shape[1]

                # Assert feature maps are squares
                assert (h.shape[2] == h.shape[3])
                assert (x.shape[2] == x.shape[3])

                # Standard projection
                """
                ksize = self.kisze
                stride = self.stride
                pad = self.pad
                proj = Deconvolution(in_channels, out_channels, ksize, stride, pad, outsize=h.shape[2:])
                """

                # Linear projection (1x1 deconv)
                proj = Deconvolution(in_channels, out_channels, 1, 2, 0, outsize=h.shape[2:])

                if self._device_id >= 0:
                    proj.to_gpu(self._device_id)
                self.add_link('proj', proj)

            x = self.proj(x)
        return x
