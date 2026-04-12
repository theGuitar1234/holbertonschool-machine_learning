#!/usr/bin/env python3
"""rotates image by 90 degrees"""
import tensorflow as tf


def rotate_image(image):
    """
    function that rotates by 90 degrees
    """
    img = tf.image.rot90(image, k=1)
    return img
