import numpy as np

import chainer
from chainer import training, reporter, cuda
from chainer import functions as F
from chainer import Variable


class WassersteinUpdater(training.StandardUpdater):
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
        z = self.next_batch(self.z)
        x_fake = self.generator(z, test=False)
        y_fake = self.discriminator(x_fake, test=False)
        x_real = self.next_batch(self.x)
        y_real = self.discriminator(x_real, test=False)

        generator_loss = F.sum(F.softplus(-y_fake)) / y_fake.shape[0]
        discriminator_loss_fake = F.sum(F.softplus(y_fake)) / y_fake.shape[0]
        discriminator_loss_real = F.sum(F.softplus(-y_real)) / y_real.shape[0]
        discriminator_loss = discriminator_loss_fake + discriminator_loss_real

        chainer.report({'loss': generator_loss}, self.generator)
        chainer.report({'loss': discriminator_loss}, self.discriminator)
        chainer.report({'loss/real': discriminator_loss_real}, self.discriminator)
        chainer.report({'loss/fake': discriminator_loss_fake}, self.discriminator)

        self.generator.cleargrads()
        generator_loss.backward()
        self.optimizer_generator.update()

        self.discriminator.cleargrads()
        discriminator_loss.backward()
        self.optimizer_discriminator.update()

    def update_core(self):

        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        # Update critic 10 times
        for _ in range(10):
            # Clamp critic parameters
            self.discriminator.clamp()

            # Real images
            x_real = self.next_batch(self.x)
            y_real = self.discriminator(x_real)
            y_real = F.sum(y_real) / y_real.shape[0]
            y_real.grad = self.xp.ones_like(y_real.data)
            _update(self.optimizer_discriminator, y_real)

            # Fake images
            z = self.next_batch(self.z)
            x_fake = self.generator(z)
            y_fake = self.discriminator(x_fake)
            y_fake = F.sum(y_fake) / y_fake.shape[0]
            y_fake.grad = -1 * self.xp.ones_like(y_fake.data)
            _update(self.optimizer_discriminator, y_fake)

            reporter.report({
                'dis/loss/real': y_real,
                'dis/loss/fake': y_fake,
                'dis/loss': y_real - y_fake
            }, self.discriminator)

        # Update generator 1 time
        z = self.next_batch(self.z)
        x_fake = self.generator(z)
        y_fake = self.discriminator(x_fake)
        y_fake.grad = self.xp.ones_like(y_fake.data)
        _update(self.optimizer_generator, y_fake)

        reporter.report({'loss': y_fake}, self.generator)


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
        z = self.next_batch(self.z)
        x_fake = self.generator(z, test=False)
        y_fake = self.discriminator(x_fake, test=False)
        x_real = self.next_batch(self.x)
        y_real = self.discriminator(x_real, test=False)

        generator_loss = F.sum(F.softplus(-y_fake)) / y_fake.shape[0]
        discriminator_loss_fake = F.sum(F.softplus(y_fake)) / y_fake.shape[0]
        discriminator_loss_real = F.sum(F.softplus(-y_real)) / y_real.shape[0]
        discriminator_loss = discriminator_loss_fake + discriminator_loss_real

        chainer.report({'loss': generator_loss}, self.generator)
        chainer.report({'loss': discriminator_loss}, self.discriminator)
        chainer.report({'loss/real': discriminator_loss_real}, self.discriminator)
        chainer.report({'loss/fake': discriminator_loss_fake}, self.discriminator)

        self.generator.cleargrads()
        generator_loss.backward()
        self.optimizer_generator.update()

        self.discriminator.cleargrads()
        discriminator_loss.backward()
        self.optimizer_discriminator.update()


class ChannelWiseUpdater(training.StandardUpdater):
    def __init__(self, *, iterator, noise_iterator, optimizers,
                 device=-1):

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'g0': optimizers[0],
                      'g1': optimizers[1],
                      'g2': optimizers[2],
                      'd0': optimizers[3],
                      'd1': optimizers[4],
                      'd2': optimizers[5],
                      'd': optimizers[6]}

        super().__init__(iterators, optimizers, device=device)

        if device >= 0:
            chainer.cuda.get_device(device).use()
            [optimizer.target.to_gpu() for optimizer in optimizers.values()]

        self.xp = chainer.cuda.cupy if device >= 0 else np

    def next_batch(self, iterator):
        batch = self.converter(iterator.next(), self.device)
        return Variable(batch)

    @property
    def z(self):
        return self._iterators['z']

    @property
    def x(self):
        return self._iterators['main']

    @property
    def g0(self):
        return self._optimizers['g0'].target

    @property
    def g1(self):
        return self._optimizers['g1'].target

    @property
    def g2(self):
        return self._optimizers['g2'].target

    @property
    def d0(self):
        return self._optimizers['d0'].target

    @property
    def d1(self):
        return self._optimizers['d1'].target

    @property
    def d2(self):
        return self._optimizers['d2'].target

    @property
    def d(self):
        return self._optimizers['d'].target

    def update_core(self):
        test=False
        xp = self.xp

        z = self.next_batch(self.z)

        # Generatate LR images
        h0_fake = self.g0(z, test=test)
        h1_fake = self.g1(z, test=test)
        h2_fake = self.g2(z, test=test)

        # Predict LR image sources (fake)
        y0_fake = self.d0(h0_fake, test=test)
        y1_fake = self.d1(h1_fake, test=test)
        y2_fake = self.d2(h2_fake, test=test)

        # Predict LR same-source likeness (fake)
        y_fake = self.d(F.concat((h0_fake, h1_fake, h2_fake), axis=1), test=test)

        # Real LR images
        h_real = self.converter(self._iterators['main'].next(), self.device)
        h0_real = h_real[:,0,:,:][:,xp.newaxis,:,:]
        h1_real = h_real[:,1,:,:][:,xp.newaxis,:,:]
        h2_real = h_real[:,2,:,:][:,xp.newaxis,:,:]

        # Predict LR image sources (real)
        y0_real = self.d0(Variable(h0_real), test=test)
        y1_real = self.d1(Variable(h1_real), test=test)
        y2_real = self.d2(Variable(h2_real), test=test)

        # Predict LR same-source likeness (real)
        y_real = self.d(Variable(h_real), test=test)

        # Compute losses
        g0_loss = F.sum(F.softplus(-y0_fake)) / y0_fake.shape[0]
        g1_loss = F.sum(F.softplus(-y1_fake)) / y1_fake.shape[0]
        g2_loss = F.sum(F.softplus(-y2_fake)) / y2_fake.shape[0]

        d0_loss_fake = F.sum(F.softplus(y0_fake)) / y0_fake.shape[0]
        d0_loss_real = F.sum(F.softplus(-y0_real)) / y0_real.shape[0]
        d0_loss = d0_loss_fake + d0_loss_real

        d1_loss_fake = F.sum(F.softplus(y1_fake)) / y1_fake.shape[0]
        d1_loss_real = F.sum(F.softplus(-y1_real)) / y1_real.shape[0]
        d1_loss = d1_loss_fake + d1_loss_real

        d2_loss_fake = F.sum(F.softplus(y2_fake)) / y2_fake.shape[0]
        d2_loss_real = F.sum(F.softplus(-y2_real)) / y2_real.shape[0]
        d2_loss = d2_loss_fake + d2_loss_real

        d_loss_fake = F.sum(F.softplus(y_fake)) / y_fake.shape[0]
        d_loss_real = F.sum(F.softplus(-y_real)) / y_real.shape[0]
        d_loss = d_loss_fake + d_loss_real

        chainer.report({'loss': g0_loss}, self.g0)
        chainer.report({'loss': g1_loss}, self.g2)
        chainer.report({'loss': g2_loss}, self.g1)

        chainer.report({'loss': d0_loss}, self.d0)
        chainer.report({'loss': d1_loss}, self.d1)
        chainer.report({'loss': d2_loss}, self.d2)

        chainer.report({'loss': d_loss}, self.d)

        self.g0.cleargrads()
        g0_loss.backward()
        self._optimizers['g0'].update()

        self.g1.cleargrads()
        g1_loss.backward()
        self._optimizers['g1'].update()

        self.g2.cleargrads()
        g2_loss.backward()
        self._optimizers['g2'].update()

        self.d0.cleargrads()
        d0_loss.backward()
        self._optimizers['d0'].update()

        self.d1.cleargrads()
        d1_loss.backward()
        self._optimizers['d1'].update()

        self.d2.cleargrads()
        d2_loss.backward()
        self._optimizers['d2'].update()

        self.d.cleargrads()
        d_loss.backward()
        self._optimizers['d'].update()

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
        h0 = self.g0(Variable(z, volatile=True), test=True)
        h1 = self.g1(Variable(z, volatile=True), test=True)
        h2 = self.g2(Variable(z, volatile=True), test=True)
        x = F.concat((h0, h1, h2), axis=1)
        x = x.data

        if isinstance(x, cuda.cupy.ndarray):
            x = cuda.to_cpu(x)

        # [-1, 1] -> [0, 1]
        x += 1.0
        x /= 2

        return x
