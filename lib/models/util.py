import chainer
from chainer import functions as F
from chainer import links as L


def n_params(model):
    for p in model.params():
        print(p.name, p.size)
    return sum(link.size for link in model.params())


class MinibatchDiscrimination(chainer.Chain):

    """Minibatch discrimination layer that is added right before the prediction
    layer in the descriminator. This layer learns the diversity in the data
    based on the diversity in the given batch (thus forcing the generator
    to reproduce the same diversity, avoiding it from collapsing to a single
    points.
    See Improved Techniques for Training GANs.
    https://arxiv.org/abs/1606.03498
    """

    def __init__(self, in_shape, n_kernels, kernel_dim):
        super(MinibatchDiscrimination, self).__init__(
            t=L.Linear(in_shape, n_kernels*kernel_dim)
        )
        self.n_kernels = n_kernels
        self.kernel_dim = kernel_dim

    def __call__(self, x):
        minibatch_size = x.shape[0]

        h = self.t(x)
        #h = x

        #print('minibatch trans', h.shape)

        #self.n_kernels = x.shape[1]
        #self.kernel_dim = x.shape[2] * x.shape[3]

        activation = F.reshape(h, (-1, self.n_kernels, self.kernel_dim))
        #print('activation reshaped', activation.shape)

        activation_ex = F.expand_dims(activation, 3)
        activation_ex_t = F.expand_dims(F.transpose(activation, (1, 2, 0)), 0)
        activation_ex, activation_ex_t = F.broadcast(activation_ex, activation_ex_t)
        diff = activation_ex - activation_ex_t

        xp = chainer.cuda.get_array_module(x.data)
        eps = F.expand_dims(xp.eye(minibatch_size, dtype=xp.float32), 1)
        eps = F.broadcast_to(eps, (minibatch_size, self.n_kernels, minibatch_size))
        sum_diff = F.sum(abs(diff), axis=2)
        sum_diff = F.broadcast_to(sum_diff, eps.shape)
        abs_diff = sum_diff + eps

        minibatch_features = F.sum(F.exp(-abs_diff), 2)

        # minibatch_features = F.reshape(minibatch_features, (64, 16, 4, 4))

        # if x.ndim == 4:
        #     x = F.reshape(x, (minibatch_size, -1))

        return F.concat((x, minibatch_features), axis=1)
