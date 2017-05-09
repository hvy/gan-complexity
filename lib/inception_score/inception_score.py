import math

import numpy as np
import scipy.misc

from chainer import cuda
from chainer import Variable


def inception_score(model, ims, batch_size=100, splits=10):
    """Compute the inception score (mean, std) for a set of images with model.

    Default batch_size is 100 and split size is 10. Please refer to the
    official implementation. It is recommended to to use at least 50000
    images to obtain a reliable score.

    Reference:
    https://github.com/openai/improved-gan/blob/master/inception_score/model.py
    """

    expected_in_shape = model.in_shape
    expected_out_shape = model.out_shape
    n_cls = expected_out_shape[0]

    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))

    xp = model.xp
    gpu = model._device_id

    # print('Batch size:', batch_size)
    # print('Total number of images:', n)
    # print('Total number of batches:', n_batches)

    # Compute the softmax predicitions for for all images, split into batches
    # in order to fit in memory

    with cuda.get_device(gpu):
        ys = xp.empty((n, n_cls), dtype=xp.float32)  # Softmax container

        for i in range(n_batches):
            # print('Running batch', i+1, '/', n_batches, '...')
            batch_start = (i * batch_size)
            batch_end = min((i + 1) * batch_size, n)

            ims_batch = ims[batch_start:batch_end]

            # Resize image to the shape expected by the inception module
            if (w, h) != expected_in_shape[1:]:
                ims_batch_resized = np.empty((ims_batch.shape[0], c, expected_in_shape[1], expected_in_shape[2]))
                for i, im in enumerate(ims_batch):
                    im = im.transpose((1, 2, 0))

                    # Note that imresize() rescales the image to [0, 255]
                    im = scipy.misc.imresize(im, expected_in_shape[1:], interp='bilinear')
                    ims_batch_resized[i] = im.transpose((2, 0, 1))
                ims_batch = ims_batch_resized
            else:
                # [-1, 1] -> [0, 255]
                ims_batch += 1
                ims_batch /= 2
                ims_batch *= 255

            # Feed images to the inception module to get the softmax
            # predictions
            ims_batch = xp.asarray(ims_batch, dtype=xp.float32)
            y = model(Variable(ims_batch, volatile=True), test=True)
            y = y.data

            # Print 0s and 1s for debugging purpose meaning, that D is too
            # confident in its predictions causing NaN in the softmax
            # computation.
            # n_zeros = xp.count_nonzero(y == 0)
            # n_ones = xp.count_nonzero(y == 1)
            # print('0:', n_zeros)
            # print('1:', n_ones)
            # print('-----------')
            # print('Total:', y.size)

            # Clip KL for numerical stability
            # http://svitsrv25.epfl.ch/R-doc/library/flexmix/html/KLdiv.html
            eps = 1e-4
            y = xp.clip(y, eps, 1)
            ys[batch_start:batch_end] = y

        # Compute the inception score based on the softmax predictions of the
        # inception module.

        scores = xp.empty((splits), dtype=xp.float32)  # Split inception scores
        for i in range(splits):
            part = ys[(i * n // splits):((i + 1) * n // splits), :]
            kl = part * (xp.log(part) - xp.log(xp.expand_dims(xp.mean(part, 0), 0)))
            kl = xp.mean(xp.sum(kl, 1))
            scores[i] = xp.exp(kl)

        mean, std = xp.mean(scores), xp.std(scores)

    return mean, std


def mnist_orig_score():
    from chainer import datasets
    from chainer import serializers
    from models import MNISTClassifier
    model = MNISTClassifier()
    serializers.load_hdf5('./mnist.model', model)
    train, _ = datasets.get_mnist(withlabel=False, ndim=3, scale=255)
    mean, std = inception_score(model, train)
    return mean, std


if __name__ == '__main__':
    mean, std = mnist_orig_score()
    print('Mean', mean)
    print('Std', std)
