#!/usr/bin/env python3
""" variational autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: dimensions of the model input
        hidden_layers: list of nodes for each hidden encoder layer
        latent_dims: dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        """Reparameterization trick"""
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[z, z_mean, z_log_var]
    )

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    # Full VAE
    z, z_mean, z_log_var = encoder(encoder_input)
    auto_output = decoder(z)

    def vae_loss(y_true, y_pred):
        """VAE loss"""
        reconstruction_loss = keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )
        reconstruction_loss *= input_dims

        kl_loss = 1 + z_log_var
        kl_loss -= keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto = keras.Model(
        inputs=encoder_input,
        outputs=auto_output
    )

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
