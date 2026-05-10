#!/usr/bin/env python3
"""NST that performs tasks for neural style transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer
    """

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        if (type(style_image) is not np.ndarray or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (type(content_image) is not np.ndarray or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Scales image dimensions and values to 0-1"""
        if (type(image) is not np.ndarray or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, c = image.shape

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        new_shape = (h_new, w_new)

        image = np.expand_dims(image, axis=0)

        scaled_image = tf.image.resize(
            image, new_shape, method='bicubic'
        )
        scaled_image = tf.clip_by_value(
            scaled_image / 255, 0, 1
        )

        return scaled_image

    def load_model(self):
        """ loads keras model """
        vgg = tf.keras.applications.VGG19(include_top=False)
        x = vgg.input
        style_outputs = []
        content_output = None
        for i, layer in enumerate(vgg.layers[1:]):
            '''print(type(layer))'''
            if type(layer) is tf.keras.layers.MaxPooling2D:
                ps = layer.pool_size
                layer = tf.keras.layers.AveragePooling2D(pool_size=ps,
                                                         strides=layer.strides,
                                                         padding=layer.padding,
                                                         name=layer.name)
                x = layer(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    style_outputs.append(x)
                if layer.name == self.content_layer:
                    content_output = x
            layer.trainable = False

        outputs = style_outputs + [content_output]
        model = tf.keras.models.Model(vgg.input, outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        calculates gram matrix
        """
        if not (isinstance(input_layer, tf.Tensor)) or\
                isinstance(input_layer, tf.Variable) or\
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        _, h, w, c = input_layer.shape.dims
        n = int(h * w)
        F = tf.reshape(input_layer, (1, n, c))
        gram = tf.matmul(F, F, transpose_a=True)
        gram = gram / tf.cast(n, tf.float32)
        return gram
