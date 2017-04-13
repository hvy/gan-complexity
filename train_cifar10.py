from chainer import datasets
from chainer import iterators
from chainer import optimizer
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from lib.extensions import GeneratorSample
from lib.extensions import InceptionScore
from lib.inception_score import Inception
from lib.iterators import GaussianNoiseGenerator
from lib.iterators import RandomNoiseIterator
# from lib.iterators import UniformNoiseGenerator
import lib.models.discriminator as D
import lib.models.generator as G
from lib.updaters import GenerativeAdversarialUpdater
import train_utils


if __name__ == '__main__':
    args = train_utils.parse_args()

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
    inception_score_model_path = args.inception_score_model_path
    inception_score_n_samples = args.inception_score_n_samples
    train_utils.setup_dirs(out)

    print('Configurartion', args)

    # CIFAR-10 images in range [-1, 1] (tanh generator outputs)
    # train, _ = datasets.get_cifar10(withlabel=False, ndim=3, scale=2)
    train, _ = datasets.get_cifar10(withlabel=False, ndim=3, scale=2)
    train -= 1.0
    train_iter = iterators.SerialIterator(train, batch_size)

    z_iter = RandomNoiseIterator(GaussianNoiseGenerator(0, 1, n_z), batch_size)

    optimizer_generator = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_discriminator = optimizers.Adam(alpha=0.0002, beta1=0.5)

    if g_res < 0:  # No residual blocks
        print('Generator: Standard')
        optimizer_generator.setup(G.Vanilla())
    else:  # Use residual blocks
        print('Generator: Residual (N={})'.format(g_res))
        optimizer_generator.setup(G.Residual(n=g_res))

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

    trainer = training.Trainer(updater,
                               stop_trigger=(epochs, 'epoch'),
                               out=out)

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
            model_cls=Inception,
            model_filename=inception_score_model_path,
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

    trainer.extend(GeneratorSample(n_samples=256), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(['epoch',
                                           'iteration',
                                           'dis/loss',
                                           'dis/loss/real',
                                           'dis/loss/fake',
                                           'gen/loss']))
    trainer.extend(extensions.PrintReport(['inception_score_mean',
                                           'inception_score_std'],
                                          log_report=inception_score_log_report))
    trainer.extend(extensions.dump_graph('gen/loss', out_name='gen_loss.dot'))
    trainer.extend(extensions.dump_graph('dis/loss', out_name='dis_loss.dot'))
    # trainer.extend(extensions.snapshot_object(optimizer_generator.target, filename='generator_{.updater.epoch}.npz', trigger=(1, 'epoch')))
    trainer.run()
