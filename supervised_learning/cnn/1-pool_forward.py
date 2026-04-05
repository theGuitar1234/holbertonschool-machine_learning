#!/usr/bin/env python3
'''
pooling forward prop
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


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode="max"):
    '''
    A_prev - a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
        m - the number of examples
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

    Returns: the output of pooling layer
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    nh = (h_prev - kh) // sh + 1
    nw = (w_prev - kw) // sw + 1

    output = np.zeros((m, nh, nw, c_prev))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(nh):
            for w in range(nw):
                for c in range(c_prev):
                    patch = a_prev[h * sh: h * sh + kh, w * sw: w * sw + kw, c]
                    if mode == 'max':
                        output[i, h, w, c] = np.max(patch)
                    elif mode == 'avg':
                        output[i, h, w, c] = np.mean(patch)

    return output
