#!/usr/bin/env python3
"""PCA color docstring."""

import tensorflow as tf


def pca_color(image, alphas):
    """Docstring for pca_color()."""
    original_dtype = image.dtype
    image = tf.cast(image, tf.float32) / 255.0

    pixels = tf.reshape(image, (-1, 3))
    mean = tf.reduce_mean(pixels, axis=0, keepdims=True)
    centered = pixels - mean

    n = tf.cast(tf.shape(centered)[0], tf.float32)
    covariance = tf.matmul(centered, centered, transpose_a=True) / (n - 1.0)

    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)

    indices = tf.argsort(eigenvalues, direction='DESCENDING')
    eigenvalues = tf.gather(eigenvalues, indices)
    eigenvectors = tf.gather(eigenvectors, indices, axis=1)

    alphas = tf.reshape(tf.convert_to_tensor(alphas, dtype=tf.float32), (3, 1))
    delta = tf.matmul(eigenvectors,
                      alphas * tf.reshape(eigenvalues, (3, 1)))
    delta = tf.reshape(delta, (1, 1, 3))

    augmented = tf.clip_by_value(image + delta, 0.0, 1.0)

    if original_dtype.is_integer:
        augmented = tf.cast(augmented * 255.0, original_dtype)

    return augmented
