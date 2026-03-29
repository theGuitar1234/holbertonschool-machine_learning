#!/usr/bin/env python3
"""valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
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
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    h_out = h - kh + 1
    w_out = w - kw + 1

    convolved_out = np.zeros((m, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            image = images[:, i: i + kh, j: j + kw]
            convolved_out[:, i, j] = np.sum(image*kernel, axis=(1, 2))
    return convolved_out
