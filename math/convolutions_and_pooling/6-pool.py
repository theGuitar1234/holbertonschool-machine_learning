#!/usr/bin/env python3
'''
pooling the image
'''

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''
    images - a numpy.ndarray with shape (m, h, w, c) containing multiple images
        m - the number of images
        h - the height in pixels of the images
        w - the width in pixels of the images
        c - the number of channels in the image
    kernel_shape - a tuple of (kh, kw) containing
                   the kernel shape for the pooling
        kh - the height of the kernel
        kw - the width of the kernel
    stride - a tuple of (sh, sw)
        sh - the stride for the height of the image
        sw - the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling

    Returns: a numpy.ndarray containing the pooled images
    '''

    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, c))

    for i in range(h_out):
        for j in range(w_out):
            patch = images[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :]
            if mode == 'max':
                output[:, i, j] = np.max(patch, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j] = np.mean(patch, axis=(1, 2))

    return output
