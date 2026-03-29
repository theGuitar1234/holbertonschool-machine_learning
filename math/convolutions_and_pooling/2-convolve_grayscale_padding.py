#!/usr/bin/env python3
'''
convolution with padding
'''

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''
    images - a numpy.ndarray with shape (m, h, w) containing
             multiple grayscale images
        m - the number of images
        h - the height in pixels of the images
        w - the width in pixels of the images

    kernel - an ndarray with shape (kh, kw) containing
             the kernel for the convolution
        kh - the height of the kernel
        kw - the width of the kernel

    padding - the tuple of (ph, pw)
        ph - the padding for the height of the image
        pw - the padding for the width of the image
        the image should be padded with 0's

    Returns: an ndarray containing convolved images
    '''

    m, h, w = images.shape
    kh, kw = kernel.shape

    ph, pw = padding

    h_out = (h + 2 * ph) - kh + 1
    w_out = (w + 2 * pw) - kw + 1

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    convolved_out = np.zeros((m, h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            image = images_padded[:, i: i + kh, j: j + kw]
            convolved_out[:, i, j] = np.sum(image*kernel, axis=(1, 2))
    return convolved_out
