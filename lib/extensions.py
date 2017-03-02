import os

import numpy as np
import chainer
from chainer import training, reporter, cuda, serializers
from chainer.training import extension

from lib import inception_score, plot


class GeneratorSample(extension.Extension):
    def __init__(self, n_samples=100, dirname='images'):
        self._dirname = dirname
        self.n_samples = n_samples
        self.z = None  # Fixed input noise

    def __call__(self, trainer):
        dirname = os.path.join(trainer.out, self._dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        if self.z is None:
            self.z = trainer.updater.sample_z(self.n_samples)

        x = trainer.updater.sample_fixed(self.z)

        filename = '{}_{}.png'.format(trainer.updater.epoch, trainer.updater.iteration)
        filename = os.path.join(dirname, filename)
        plot.save_ims(filename, x)


class InceptionScore(extension.Extension):
    def __init__(self, model_filename='inception_score.model', n_samples=10000,  gpu=-1):
        model = inception_score.Inception()
        serializers.load_hdf5(model_filename, model)
        self._model = model
        self.n_samples = n_samples

        if gpu >= 0:
            model.to_gpu(gpu)

    def __call__(self, trainer):
        x = trainer.updater.sample(self.n_samples)
        mean, std = inception_score.inception_score(self._model, x)
        trainer.reporter.report({
            'inception_score_mean': mean,
            'inception_score_std': std
        })