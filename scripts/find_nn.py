import math
import argparse

import numpy
from PIL import Image

from chainer import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--im', type=str, default='./result_0032_n0_res_5h_500epochs/images/478_373438.png')
    parser.add_argument('--x', type=int, default=0)
    parser.add_argument('--y', type=int, default=0)
    parser.add_argument('--imsize', type=int, default=32)
    parser.add_argument('--out-original', type=str, default='im_original.png')
    parser.add_argument('--out-nn', type=str, default='im_nn.png')
    return parser.parse_args()


def save_im(filename, ims):
    c, h, w = ims.shape

    ims = numpy.clip(ims * 255, 0.0, 255.0)
    ims = ims.reshape((1, 1, 3, h, w))
    ims = ims.transpose(0, 3, 1, 4, 2)
    ims = ims.reshape((h, w, 3))
    ims = ims.astype(numpy.uint8)

    print('Saving image', filename)

    Image.fromarray(ims).save(filename)


def open_tiled_im(filename):
    im = Image.open(filename)
    im = numpy.asarray(im)
    im = im.transpose((2, 0, 1))
    im = im / 255.0
    return im


if __name__ == '__main__':
    args = parse_args()

    train, _ = datasets.get_cifar10(withlabel=False, ndim=3, scale=1)
    im = open_tiled_im(args.im)

    # Get a single image from the tiles image
    x = args.x
    y = args.y
    w = args.imsize
    h = args.imsize
    im = im[:, h*y:h*(y+1), w*x:w*(x+1)]

    # Find nearest image in L2 distance
    nearest_im = None
    nearest_im_l2 = None
    for t in train:
        l2 = numpy.sqrt((t - im) ** 2).sum()
        if nearest_im_l2 is None or l2 < nearest_im_l2:
            nearest_im = t
            nearest_im_l2 = l2

    save_im(args.out_original, im)
    save_im(args.out_nn, nearest_im)
