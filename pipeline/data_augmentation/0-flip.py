#!/usr/bin/env python3
"""flip an image horizontally"""
import tensorflow as tf


def flip_image(image):
    """
    function  that flips an image horizontally
    """
    flip = tf.image.flip_left_right(image)
    return flip
