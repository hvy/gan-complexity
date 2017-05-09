import argparse

from chainer import cuda
from chainer import datasets
from chainer import serializers
import numpy

import models
from inception_score import inception_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist'])
    parser.add_argument('--model-cls', type=str, default='inception',
                        choices=['inception', 'mnist'])
    parser.add_argument('--model-file', type=str, default='inception.model')
    return parser.parse_args()


def load_model(args):
    if args.model_cls == 'inception':
        model = models.Inception()
    elif args.model_cls == 'mnist':
        model = models.MNISTClassifier()
    else:
        raise ValueError('Unknown model {}'.format(args.model_cls))
    serializers.load_hdf5(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    return model


def load_ims(args):
    """ Return images scaled to [-1, 1]."""
    if args.dataset == 'cifar10':
        ims, ims_test = datasets.get_cifar10(ndim=3, withlabel=False, scale=2)
        ims = numpy.concatenate((ims, ims_test))
        ims -= 1.0
    elif args.dataset == 'mnist':
        ims, ims_test = datasets.get_mnist(ndim=3, withlabel=False, scale=2)
        ims = numpy.concatenate((ims, ims_test))
        ims -= 1.0
    else:
        raise ValueError('Unknown dataset {}'.format(args.model_cls))
    if args.samples is not None:
        ims = ims[:args.samples]
    print(ims.shape)
    return ims


def main(args):
    model = load_model(args)
    ims = load_ims(args)

    mean, std = inception_score(model, ims)

    print('Inception score mean:', mean)
    print('Inception score std:', std)


if __name__ == '__main__':
    args = parse_args()
    main(args)
