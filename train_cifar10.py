import os

from chainer import datasets
from chainer import iterators
from chainer import optimizer
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from lib import inception_score
from lib.extensions import GeneratorSample
from lib.extensions import InceptionScore
from lib.iterators import GaussianNoiseGenerator
from lib.iterators import RandomNoiseIterator
from lib.iterators import UniformNoiseGenerator
import lib.models.discriminator as D
import lib.models.generator as G
from lib.updaters import GenerativeAdversarialUpdater
import config
import util


inception_models = {
    'cifar10': inception_score.models.Inception,
    'mnist': inception_score.models.MNISTClassifier
}

if __name__ == '__main__':
    args = config.parse_args()
    util.setup_dirs(args.out)

    dataset = args.dataset

    batch_size = args.batch_size
    epochs = args.epochs

    n_z = args.n_z
    g_res = args.g_res
    d_res = args.d_res
    g_weight_decay = args.g_weight_decay
    d_weight_decay = args.d_weight_decay
    mbd = args.mbd

    out = args.out

    gpu = args.gpu
    inception_gpu = args.inception_gpu
    inception_score_model_cls = args.inception_score_model_cls
    inception_score_model_path = args.inception_score_model_path
    inception_score_model_compression = args.inception_score_model_compression
    inception_score_n_samples = args.inception_score_n_samples


    print('Configurartion', args)

    if dataset == 'cifar10':
        get_dataset = datasets.get_cifar10
    elif dataset == 'mnist':
        get_dataset = datasets.get_mnist
    else:
        raise ValueError('Unknown dataset {}'.format(dataset))

    inception_score_model_cls = inception_models[dataset] if not inception_score_model_cls else getattr(inception_score.models, inception_score_model_cls)

    # Images in range [-1, 1] (tanh generator outputs)
    train, _ = get_dataset(withlabel=False, ndim=3, scale=2)
    train -= 1.0
    print(train.min())
    print(train.max())
    print(train.mean())
    print(train.std())

    train_iter = iterators.SerialIterator(train, batch_size)

    z_iter = RandomNoiseIterator(UniformNoiseGenerator(-1, 1, n_z), batch_size)

    optimizer_generator = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_discriminator = optimizers.Adam(alpha=0.0002, beta1=0.5)

    if g_res < 0:  # No residual blocks
        print('Generator: Standard')
        optimizer_generator.setup(G.Vanilla())
    else:  # Use residual blocks
        print('Generator: Residual (N={})'.format(g_res))
        optimizer_generator.setup(G.Residual(n=g_res, out_shape=train[0].shape))

    if d_res < 0:  # No residual blocks
        print('Discriminator: Standard (Minibatch Discrimination={})'
              .format(mbd))
        optimizer_discriminator.setup(D.Vanilla(use_mbd=mbd))
    else:  # Use residual blocks
        print('Discriminator: Residual (N={}, Minibatch Discrimination={})'
              .format(d_res, mbd))
        optimizer_discriminator.setup(D.Residual(n=d_res, use_mbd=mbd))

    optimizer_generator.add_hook(optimizer.WeightDecay(g_weight_decay))
    optimizer_discriminator.add_hook(optimizer.WeightDecay(d_weight_decay))

    updater = GenerativeAdversarialUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        device=gpu
    )

    trainer = training.Trainer(updater, stop_trigger=(epochs, 'epoch'),
                               out=out)

    # Logging losses to result/logs/loss
    trainer.extend(
        extensions.LogReport(
            keys=['dis/loss', 'dis/loss/real', 'dis/loss/fake', 'gen/loss', 'd_x', 'd_gz'],
            trigger=(100, 'iteration'),  # default (1, 'iteration')
            log_name='logs/loss'
        )
    )

    # Logging Inception Score to result/logs/inception_score

    trainer.extend(
        InceptionScore(
            model_cls=inception_score_model_cls,
            model_filename=inception_score_model_path,
            compression=inception_score_model_compression,
            n_samples=inception_score_n_samples,  # Default 50000
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

    trainer.extend(GeneratorSample(n_samples=100), trigger=(100, 'iteration'))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(['epoch',
                                           'iteration',
                                           'dis/loss',
                                           'dis/loss/real',
                                           'dis/loss/fake',
                                           'gen/loss',
                                           'd_x',
                                           'd_gz',
                                           ]))
    trainer.extend(extensions.PrintReport(['inception_score_mean',
                                           'inception_score_std'],
                                          log_report=inception_score_log_report))
    trainer.extend(extensions.dump_graph('gen/loss', out_name='gen_loss.dot'))
    trainer.extend(extensions.dump_graph('dis/loss', out_name='dis_loss.dot'))
    # trainer.extend(extensions.snapshot_object(optimizer_generator.target, filename='generator_{.updater.epoch}.npz', trigger=(1, 'epoch')))
    trainer.run()
