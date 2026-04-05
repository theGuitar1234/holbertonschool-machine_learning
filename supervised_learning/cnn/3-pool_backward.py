#!/usr/bin/env python3
'''
pooling back propogation
'''

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode="max"):
    '''
    dA - a numpy.ndarray of shape (m, h_new, w_new, c_prev) containing
            the partial derivatives with respect to the output of the
            pooling layer
        m - the number of examples
        h_new - the height of the output
        w_new - the width of the output
        c_prev - the number of channels in the previous layer
    A_prev - a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
        h_prev - the height of the previous layer
        w_prev - the width of the previous layer
        c_prev - the number of channels in the previous layer
    kernel_shape - a tuple of (kh, kw) containing the size of
                    the kernel for the pooling
        kh - the kernel height
        kw - the kernel width
    stride - a tuple of (sh, sw) containing the strides for the pooling
        sh - the stride for the height
        sw - the stride for the width
    mode - a string containing either max or avg,
            indicating whether to perform maximum or
            average pooling, respectively
    Returns: the partial derivatives with respect to the previous layer
    '''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    _, h_new, w_new, _ = dA.shape
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):
        a_prev = A_prev[i]
        da = dA[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_prev):
                    patch = a_prev[h * sh: h * sh + kh, w * sw: w * sw + kw, c]
                    if mode == 'max':
                        mask = (patch == np.max(patch))
                        dA_prev[i, h * sh: h * sh + kh,
                                w * sw: w * sw + kw,
                                c] += mask * da[h, w, c]
                    elif mode == 'avg':
                        average = da[h, w, c] / (kh * kw)
                        dA_prev[i, h * sh: h * sh + kh,
                                w * sw: w * sw + kw,
                                c] += np.ones((kh, kw)) * average
    return dA_prev
