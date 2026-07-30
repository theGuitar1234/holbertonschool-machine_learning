#!/usr/bin/env python3
"""Creates a variational autoencoder using Keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Hidden layer units for the encoder.
        latent_dims (int): Dimensions of the latent space.

    Returns:
        encoder: Encoder model outputting latent, mean, log variance.
        decoder: Decoder model.
        auto: Full autoencoder model.
    """
    K = keras.backend

    def sampling(args):
        """Reparameterization trick."""
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(inputs=inputs, outputs=[z, z_mean, z_log_var])
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)
    z, z_mean, z_log_var = encoder(inputs)
    reconstructed = decoder(z)
    auto = keras.Model(inputs=inputs, outputs=reconstructed)
    reconstruction_loss = keras.losses.binary_crossentropy(
        inputs, reconstructed)
    reconstruction_loss *= input_dims
    kl_loss = -0.5 * K.sum(1 + z_log_var -
                           K.square(z_mean) - K.exp(z_log_var), axis=-1)
    auto.add_loss(K.mean(reconstruction_loss + kl_loss))
    auto.compile(optimizer='adam')
    return encoder, decoder, auto
