from chainer import functions as F


def pixel_shuffle(x, upscale_factor):
    n, c, w, h = x.shape

    c_out = c // upscale_factor ** 2
    w_out = w * upscale_factor
    h_out = h * upscale_factor

    y = F.reshape(x, (n, c_out, upscale_factor, upscale_factor, w, h))
    y = F.transpose(y, (0, 1, 4, 2, 5, 3))
    y = F.reshape(y, (n, c_out, w_out, h_out))

    return y
