#!/usr/bin/env python3
"""Builds a variational autoencoder with Keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.

    Args:
        input_dims (int): Number of dimensions in the model input.
        hidden_layers (list): Number of nodes for each encoder hidden layer.
        latent_dims (int): Number of dimensions in the latent space.

    Returns:
        tuple: The encoder, decoder, and full autoencoder models.
    """

    def sampling(args):
        """Sample latent vectors using the reparameterization trick."""
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dims = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dims))

        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation="relu")(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(encoder_input, [z, z_mean, z_log_var])

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation="relu")(decoded)

    decoder_output = keras.layers.Dense(
        input_dims, activation="sigmoid"
    )(decoded)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_output = decoder(z)
    auto = keras.Model(encoder_input, auto_output)

    def vae_loss(input_data, reconstructed_data):
        """Calculate binary cross-entropy plus KL divergence loss."""
        reconstruction_loss = keras.losses.binary_crossentropy(
            input_data, reconstructed_data
        )
        reconstruction_loss *= input_dims

        kl_loss = 1 + z_log_var
        kl_loss -= keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer="adam", loss=vae_loss)

    return encoder, decoder, auto
