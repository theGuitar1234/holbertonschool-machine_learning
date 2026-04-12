#!/usr/bin/env python3
"""random crop"""
import tensorflow as tf


def crop_image(image, size):
    """
    function that performs a random crop of an image
    """
    img = tf.image.random_crop(image, size=size)
    return img
