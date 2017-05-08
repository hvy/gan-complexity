import numpy as np

import chainer
from chainer import training, reporter, cuda
from chainer import functions as F
from chainer import Variable


class GenerativeAdversarialUpdater(training.StandardUpdater):
    def __init__(self, *, iterator, noise_iterator, optimizer_generator,
                 optimizer_discriminator, device=-1):

        if optimizer_generator.target.name is None:
            optimizer_generator.target.name = 'gen'

        if optimizer_discriminator.target.name is None:
            optimizer_discriminator.target.name = 'dis'

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'gen': optimizer_generator,
                      'dis': optimizer_discriminator}

        super().__init__(iterators, optimizers, device=device)

        if device >= 0:
            chainer.cuda.get_device(device).use()
            [optimizer.target.to_gpu() for optimizer in optimizers.values()]

        self.xp = chainer.cuda.cupy if device >= 0 else np

    @property
    def optimizer_generator(self):
        return self._optimizers['gen']

    @property
    def optimizer_discriminator(self):
        return self._optimizers['dis']

    @property
    def generator(self):
        return self._optimizers['gen'].target

    @property
    def discriminator(self):
        return self._optimizers['dis'].target

    @property
    def x(self):
        return self._iterators['main']

    @property
    def z(self):
        return self._iterators['z']

    def next_batch(self, iterator):
        batch = self.converter(iterator.next(), self.device)
        return Variable(batch)

    def sample_z(self, n_samples):
        z_fixed = []
        n_sampled = 0
        while n_sampled < n_samples:
            z = self._iterators['z'].next()
            z_fixed.append(z)
            n_sampled += len(z)
        return np.concatenate(z_fixed, axis=0)[:n_samples]

    def sample_fixed(self, z):
        z = self.converter(z, self.device)
        x = self.generator(Variable(z, volatile=True), test=True)
        x = x.data

        if isinstance(x, cuda.cupy.ndarray):
            x = cuda.to_cpu(x)

        # [-1, 1] -> [0, 1]
        x += 1.0
        x /= 2

        return x


    def sample(self, n_samples):

        """
        Returns:
            numpy.ndarray: Samples of shape (`n_samples`, c, w, h).
        """

        samples = []
        n_sampled = 0

        while n_sampled < n_samples:
            z = self.converter(self._iterators['z'].next(), self.device)
            x = self.generator(Variable(z, volatile=True), test=True).data
            if isinstance(x, cuda.cupy.ndarray):
                x = cuda.to_cpu(x)
            n_sampled += len(x)
            samples.append(x)

        return np.concatenate(samples, axis=0)[:n_samples]


    def update_params(self, losses, report=True):
        for name, loss in losses.items():
            if report:
                reporter.report({'{}/loss'.format(name): loss})

    def update_core(self):
        # p_g
        z_it = self._iterators['z'].next()
        z = self.converter(z_it, self.device)
        x_fake = self.generator(Variable(z), test=False)
        y_fake = self.discriminator(x_fake, test=False)
        d_loss_fake = F.softplus(y_fake)
        d_loss_fake = F.sum(d_loss_fake) / d_loss_fake.shape[0]
        g_loss = F.softplus(-y_fake)
        g_loss = F.sum(g_loss) / g_loss.shape[0]

        # p_data
        x_real_it = self._iterators['main'].next()
        x_real = self.converter(x_real_it, self.device)
        y_real = self.discriminator(Variable(x_real), test=False)
        d_loss_real = F.softplus(-y_real)
        d_loss_real = F.sum(d_loss_real) / d_loss_real.shape[0]

        self._optimizers['dis'].target.cleargrads()
        d_loss_fake.backward()
        self._optimizers['dis'].update()

        self._optimizers['dis'].target.cleargrads()
        d_loss_real.backward()
        self._optimizers['dis'].update()

        self._optimizers['gen'].target.cleargrads()
        g_loss.backward()
        self._optimizers['gen'].update()


        reporter.report({'d_gz': F.sum(F.sigmoid(y_fake)).data / y_fake.shape[0]})

        reporter.report({'d_x': F.sum(F.sigmoid(y_real)).data / y_real.shape[0]})

        reporter.report({'gen/loss': g_loss})
        reporter.report({'dis/loss': (d_loss_fake + d_loss_real) / 2})
