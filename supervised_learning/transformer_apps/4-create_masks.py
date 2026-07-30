#!/usr/bin/env python3
"""
This script defines a function that creates masks for Transformer training.
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    This function creates masks for Transformer training
    """
    encoder_mask = tf.cast(
        tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(
        tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    seq_len_out = target.shape[1]
    lookahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    target_padding_mask = tf.cast(
        tf.math.equal(target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(target_padding_mask, lookahead_mask)

    return encoder_mask, combined_mask, decoder_mask
