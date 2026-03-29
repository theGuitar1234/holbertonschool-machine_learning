#!/usr/bin/env python3
"""same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    '''
    images - an ndarray with shape (m, h, w) containing
            multiple grayscale images

            m - the number of images
            h - the height in pixels of the image
            w - the width in pixels of the images

    kernel - an ndarray with shape (kh, kw) containing
            the kernel for the convolution

            kh - the height of the kernel
            kw - the width of the kernel

    Returns: a numpy.ndarray containing the convolved images
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    convolved_out = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            image = images_padded[:, i: i + kh, j: j + kw]
            convolved_out[:, i, j] = np.sum(image*kernel, axis=(1, 2))
    return convolved_out
