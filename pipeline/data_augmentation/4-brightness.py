#!/usr/bin/env python3
"""changing brightness"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    function that changes brightness
    """
    img = tf.image.random_brightness(image, max_delta)
    return img
