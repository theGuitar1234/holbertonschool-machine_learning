#!/usr/bin/env python3
"""Builds a vanilla autoencoder with Keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a vanilla autoencoder.

    Args:
        input_dims (int): Number of dimensions in the model input.
        hidden_layers (list): Number of nodes for each encoder hidden layer.
        latent_dims (int): Number of dimensions in the latent representation.

    Returns:
        tuple: The encoder, decoder, and compiled autoencoder models.
    """
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation="relu")(encoded)

    latent = keras.layers.Dense(latent_dims, activation="relu")(encoded)
    encoder = keras.Model(encoder_input, latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation="relu")(decoded)

    decoder_output = keras.layers.Dense(
        input_dims, activation="sigmoid"
    )(decoded)
    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=(input_dims,))
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, auto_output)
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
