import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cupy
from chainer import cuda, serializers, Variable

from lib import plot
from lib.models.resnet import GeneratorResidual
from lib.iterators import RandomNoiseIterator, UniformNoiseGenerator, GaussianNoiseGenerator


"""Sample images from a generator and save it as a single tiled image.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='result_0001/generator_1.npz')
    parser.add_argument('--n', type=int, default=100)  # #samples
    parser.add_argument('--n-z', type=int, default=100)
    parser.add_argument('--z', type=str, default='normal')
    parser.add_argument('--z-normal-loc', type=float, default='0')
    parser.add_argument('--z-normal-scale', type=float, default='1')
    parser.add_argument('--out', type=str, default='gen_sample.png')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model = GeneratorResidual()

    # Load parameters from file
    print('Loading parameters from {}...'.format(args.model))
    serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Sample input noise z
    if args.z == 'normal':
        loc = args.z_normal_loc
        scale = args.z_normal_scale
        z_gen = GaussianNoiseGenerator(loc, scale, args.n_z)
    elif args.z == 'uniform':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    z_iter = RandomNoiseIterator(z_gen, args.n)
    z = z_iter.next()

    if args.gpu >= 0:
        z = cuda.to_gpu(z)

    # Sample images from the model
    x = model(Variable(z, volatile=True), test=True)

    if args.gpu >= 0:
        x = cuda.to_cpu(x.data)
    else:
        x = x.data

    # [-1, 1] -> [0, 1]
    x += 1.0
    x /= 2

    print('Generated data of shape {}'.format(x.shape))
    print('Saving image {}...'.format(args.out))

    plot.save_ims(args.out, x)
