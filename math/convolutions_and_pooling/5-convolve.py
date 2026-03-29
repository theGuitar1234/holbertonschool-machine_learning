#!/usr/bin/env python3
'''
convolve the image
'''

import numpy as np


def ceil(a):
    '''
    ceil function
    '''

    b = a // 1
    if a != b:
        return int(b+1)
    return int(a)


def convolve(images, kernels, padding='same', stride=(1, 1)):
    '''
    images - a numpy.ndarray with shape (m, h, w, c) containing multiple images
        m - the number of images
        h - the height in pixels of the images
        w - the width in pixels of the images
        c - the number of channels in the image
    kernels - a numpy.ndarray with shape (kh, kw, c, nc) containing
            the kernel for the convolution
        kh - the height of the kernel
        kw - the width of the kernel
        nc - the number of kernels
    padding - either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
            ph - the padding for the height of the image
            pw - the padding for the width of the image
        the image should be padded with 0's
    stride - a tuple of (sh, sw)
        sh - the stride for the height of the image
        sw - the stride for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    '''

    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ceil(((h - 1) * sh + kh - h) / 2)
        pw = ceil(((w - 1) * sw + kw - w) / 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for kernel in range(nc):
                output[:, i, j, kernel] = np.sum(
                    region * kernels[:, :, :, kernel], axis=(1, 2, 3)
                )

    return output
