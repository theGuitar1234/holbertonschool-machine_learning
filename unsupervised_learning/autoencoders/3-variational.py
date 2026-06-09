#!/usr/bin/env python3
""" variational autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.
    Returns: encoder, decoder, vae
    """

    # Encoder
    input_encoder = keras.Input(shape=(input_dims,))

    encode = input_encoder
    for nodes in hidden_layers:
        encode = keras.layers.Dense(nodes, activation='relu')(encode)

    z_mean = keras.layers.Dense(latent_dims)(encode)
    z_log_sigma = keras.layers.Dense(latent_dims)(encode)

    def sampling(args):
        """ Samples a latent vector """
        z_mean, z_log_sigma = args
        batch = keras.backend.shape(z_mean)[0]
        dims = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dims))
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    encoder = keras.Model(
        inputs=input_encoder,
        outputs=[z, z_mean, z_log_sigma]
    )

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims,))

    decode = input_decoder
    for nodes in reversed(hidden_layers):
        decode = keras.layers.Dense(nodes, activation='relu')(decode)

    output_decoder = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decode)

    decoder = keras.Model(
        inputs=input_decoder,
        outputs=output_decoder
    )

    # Full VAE
    z, z_mean, z_log_sigma = encoder(input_encoder)
    outputs = decoder(z)

    def loss(true, pred):
        """ VAE loss: reconstruction loss + KL divergence """
        reconstruction_loss = keras.losses.binary_crossentropy(true, pred)
        reconstruction_loss *= input_dims

        kl_loss = 1 + z_log_sigma
        kl_loss -= keras.backend.square(z_mean)
        kl_loss -= keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return keras.backend.mean(reconstruction_loss + kl_loss)

    vae = keras.Model(
        inputs=input_encoder,
        outputs=outputs
    )

    vae.compile(optimizer='adam', loss=loss)

    return encoder, decoder, vae
