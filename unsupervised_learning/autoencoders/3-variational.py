#!/usr/bin/env python3
""" variational autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    ARGS:
        -input_dims {integer}: containing the dimensions of the model input

        -hidden_layers {list}: containing the number of nodes
            for each hidden layer in the encoder, respectively

        -latent_dims {integer}: containing the dimensions
            of the latent space representation

    Returns: encoder, decoder, auto
        -encoder is the encoder model

        -decoder is the decoder model

        -auto is the full autoencoder model
    """

    """ Encoder """
    input_encoder = keras.Input(shape=(input_dims,))

    for i in range(len(hidden_layers)):
        if i == 0:
            encode = keras.layers.Dense(hidden_layers[i],
                                        activation='relu')(input_encoder)
        else:
            encode = keras.layers.Dense(hidden_layers[i],
                                        activation='relu')(encode)

    z_mean = keras.layers.Dense(latent_dims)(encode)
    z_log_sigma = keras.layers.Dense(latent_dims)(encode)

    def sampling(z):
        """samling a new points"""
        z_mean, z_log_sigma = z
        batch = keras.backend.shape(z_mean)[0]
        dims = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dims))
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    # Encoder call
    encoder = keras.Model(inputs=input_encoder,
                          outputs=[z, z_mean, z_log_sigma])

    """ _________________________________Decoder____________ """

    input_decoder = keras.Input(shape=(latent_dims, ))

    # hidden layers should be reversed for the decoder

    for j in range(len(hidden_layers)-1, -1, -1):
        if j == len(hidden_layers) - 1:
            decode = keras.layers.Dense(hidden_layers[j],
                                        activation='relu')(input_decoder)
        else:
            decode = keras.layers.Dense(hidden_layers[j],
                                        activation='relu')(decode)

    # Decoder output
    decode = keras.layers.Dense(input_dims,
                                activation='sigmoid')(decode)

    # Decoder call
    decoder = keras.Model(inputs=input_decoder, outputs=decode)

    """ ___________-autoencoder_____________________ """

    def loss(true, pred):
        reconstruction_loss = keras.losses.binary_crossentropy(input_encoder,
                                                               outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) -\
            keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    outputs = decoder(encoder(input_encoder))
    vae = keras.Model(input_encoder, outputs)
    vae.compile(optimizer='adam', loss=loss)
    return encoder, decoder, vae
