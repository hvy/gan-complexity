from chainer import initializers
from chainer import links as L


def Convolution(in_channels, out_channels, ksize, stride, pad):
    return L.Convolution2D(in_channels, out_channels, ksize, stride, pad,
                           initialW=initializers.Normal(0.02))


def Deconvolution(in_channels, out_channels, ksize, stride, pad, outsize=None):
    return L.Deconvolution2D(in_channels, out_channels, ksize, stride, pad,
                             initialW=initializers.Normal(0.02), outsize=outsize)


def Linear(in_size, out_size):
    # return L.Linear(in_size, out_size, initialW=initializers.Normal(0.02))

    # Default initialization for linear layers semms
    # to yield better results for CIFAR-10
    return L.Linear(in_size, out_size)


def BatchNorm(in_size):
    return L.BatchNormalization(in_size)
