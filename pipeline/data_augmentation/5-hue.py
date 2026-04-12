#!/usr/bin/env python3
"""changing hue"""
import tensorflow as tf


def change_hue(image, delta):
    """
    function that changes hue
    """
    img = tf.image.adjust_hue(image, delta)
    return img
