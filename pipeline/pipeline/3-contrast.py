#!/usr/bin/env python3
"""randomly adjust contrast"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    function to adjust contrast
    """
    img = tf.image.random_contrast(image, lower, upper)
    return img
