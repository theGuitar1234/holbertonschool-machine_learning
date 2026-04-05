#!/usr/bin/env python3
'''
convolutional forward prop
'''

import numpy as np


def ceil(a):
    """
    ceil function
    """
    b = a // 1
    if a != b:
        return int(b + 1)
    return int(a)


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1, 1)):
    '''
    A_prev - a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
                m - the number of examples
                h_prev - the height of the previous layer
                w_prev - the width of the previous layer
                c_prev - the number of channels in the previous layer

    W - a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution

    b - a numpy.ndarray of shape (1, 1, 1, c_new) containing
        the biases for the convolution

    activation - is an activation function applied to convolution

    padding - a string that is either same or valid,
            indicating the type of padding used
Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[0, 0, 0, c]

    stride - a tuple of (sh, sw) containing the strides for the convolution
            sh - the stride for the height
            sw - the stride for the width

    Returns: the output of the convolutional layer
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ceil(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = ceil(((w_prev - 1) * sw + kw - w_prev) / 2)

    elif padding == 'valid':
        ph, pw = 0, 0

    # output dimensions
    nh = ((h_prev + 2 * ph - kh)) // sh + 1
    nw = ((w_prev + 2 * pw - kw)) // sw + 1

    A_prev_padded = np.pad(A_prev,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    Z = np.zeros((m, nh, nw, c_new))

    for i in range(m):
        a_prev_pad = A_prev_padded[i]
        for h in range(nh):
            for w in range(nw):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    hortiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:hortiz_end]

                    Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[0,
                                                                        0,
                                                                        0,
                                                                        c]

    return activation(Z)
