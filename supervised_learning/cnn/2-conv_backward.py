#!/usr/bin/env python3
'''
convolutional back propogation
'''

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''
    dZ - a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output
            of the convolutional layer
        m - the number of examples
        h_new - the height of the output
        w_new - the width of the output
        c_new - the number of channels in the output
    A_prev - a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
        h_prev - the height of the previous layer
        w_prev - the width of the previous layer
        c_prev - the number of channels in the previous layer
    W - a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution
        kh - the kernel height
        kw - the kernel width
    b - a numpy.ndarray of shape (1, 1, 1, c_new) containing
            the biases applied to the convolution
    padding - a string that is either same or valid,
                indicating the type of padding used
    stride - a tuple of (sh, sw) containing the strides for the convolution
        sh - the stride for the height
        sw - the stride for the width

    Returns: dA_prev, dW, db
    dA_prev is a numpy.ndarray containing
    the partial derivatives with respect to A_prev,
                has the same shape as A_prev
    dW is a numpy.ndarray containing the partial derivatives with respect to W,
                has the same shape as W
    db is a numpy.ndarray containing the partial derivatives with respect to b,
                has the same shape as b
    '''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)), mode='constant')
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end, :] += W[:,
                                                               :,
                                                               :,
                                                               c] * dZ[i,
                                                                       h,
                                                                       w,
                                                                       c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if padding == 'same':
            dA_prev[i, :, :, :] = da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db
