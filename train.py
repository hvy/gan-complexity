import os
import argparse

from chainer import datasets, training, iterators, optimizers, optimizer
from chainer.training import updater, extensions

import lib.models.discriminator as D
import lib.models.generator as G
from lib.iterators import RandomNoiseIterator, UniformNoiseGenerator, GaussianNoiseGenerator
from lib.updaters import GenerativeAdversarialUpdater
from lib.extensions import GeneratorSample, InceptionScore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--inception-gpu', type=int, default=1)
    parser.add_argument('--n-z', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', type=str, default='result_0050_vanilla')
    parser.add_argument('--mbd', action='store_true', default=False)
    parser.add_argument('--res', type=int, default=-1)
    return parser.parse_args()


def setup(out):
    if not os.path.exists('./{}'.format(out)):
        os.makedirs('./{}/'.format(out))
    if not os.path.exists('./{}/logs'.format(out)):
        os.makedirs('./{}/logs/'.format(out))
    if not os.path.exists('./{}/images'.format(out)):
        os.makedirs('./{}/images/'.format(out))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    n_z = args.n_z
    batch_size = args.batch_size
    epochs = args.epochs
    gpu = args.gpu
    inception_gpu = args.inception_gpu
    out = args.out
    mbd = args.mbd
    res = args.res

    setup(out)

    # CIFAR-10 images in range [-1, 1] (tanh generator outputs)
    # train, _ = datasets.get_cifar10(withlabel=False, ndim=3, scale=2)
    train, _ = datasets.get_cifar10(withlabel=False, ndim=3, scale=2)
    train -= 1.0
    train_iter = iterators.SerialIterator(train, batch_size)

    z_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, n_z), batch_size)

    optimizer_generator = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_discriminator = optimizers.Adam(alpha=0.0002, beta1=0.5)

    # WGAN optimizers
    # optimizer_generator = optimizers.RMSprop(lr=0.00005)
    # optimizer_discriminator = optimizers.RMSprop(lr=0.00005)

    if res < 0:  # No residual blocks
        optimizer_generator.setup(G.Vanilla())
        optimizer_discriminator.setup(D.Vanilla(use_mbd=mbd))
    else:  # Use residual blocks
        optimizer_generator.setup(G.Residual(n=res))
        optimizer_discriminator.setup(D.Residual(n=res, use_mbd=mbd))

    optimizer_generator.add_hook(optimizer.WeightDecay(0.00001))
    optimizer_discriminator.add_hook(optimizer.WeightDecay(0.00001))

    updater = GenerativeAdversarialUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        device=gpu
    )

    trainer = training.Trainer(updater, stop_trigger=(epochs, 'epoch'), out=out)

    # Logging losses to result/logs/loss
    trainer.extend(
        extensions.LogReport(
            keys=['dis/loss', 'dis/loss/real', 'dis/loss/fake', 'gen/loss'],
            trigger=(1, 'epoch'),  # default (1, 'iteration')
            log_name='logs/loss'
        )
    )

    # Logging Inception Score to result/logs/inception_score
    trainer.extend(
        InceptionScore(
            './lib/inception_score/inception_score.model',
            n_samples=5000,  # Default to 10000 or 50000, 1000 for fast testing
            gpu=inception_gpu
        ),
        trigger=(1, 'epoch')
    )
    inception_score_log_report = extensions.LogReport(
        keys=['inception_score_mean', 'inception_score_std'],
        trigger=(1, 'epoch'),
        log_name='logs/inception_score'
    )
    trainer.extend(inception_score_log_report)

    # Sample images to result/images/
    trainer.extend(GeneratorSample(n_samples=256), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'dis/loss',
            'dis/loss/real', 'dis/loss/fake', 'gen/loss']))
    trainer.extend(extensions.PrintReport(['inception_score_mean', 'inception_score_std'], log_report=inception_score_log_report))
    # trainer.extend(extensions.snapshot_object(optimizer_generator.target, filename='generator_{.updater.epoch}.npz', trigger=(1, 'epoch')))
    trainer.extend(extensions.dump_graph('gen/loss', out_name='gen_loss.dot'))
    trainer.extend(extensions.dump_graph('dis/loss', out_name='dis_loss.dot'))
    # trainer.extend(extensions.dump_graph('dis/loss/real', out_name='dis_loss_real.dot'))
    # trainer.extend(extensions.dump_graph('dis/loss/fake', out_name='dis_loss_fake.dot'))
    trainer.run()
