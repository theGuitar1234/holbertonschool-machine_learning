#!/usr/bin/env python3
'''Docstring.'''

from tensorflow import keras as K


def proprocess_data(X, Y):
    '''Docstring.'''

    X_p, Y_p = X, Y

    return X_p, Y_p


if __name__ == "__main__":
    train_ds, test_ds = K.datasets.cifar10.load_data()
    
    inputs = K.layers.Input(shape=(32, 32, 3))

    lambd = K.layers.Lambda(
        lambda x: K.backend.resize_images(x, 150//32, 150//32, 'channels_last')
    )(inputs)

    base_model = K.applications.ResNet50(
        weights='imagenet',
        input_shape=(128, 128, 3),
        include_top=False
    )

    base_model.trainable = False

    x = base_model(lambd, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    outputs = K.layers.Dense(1)(x)
    model = K.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

    def learning_rate_schedule(epochs, lr):
        if epochs < 7:
            return lr
        else:
            return lr * 0.9

    model.fit(
        train_ds[0],
        train_ds[1],
        batch_size=32,
        epochs=10,
        validation_data=test_ds,
        callbacks=[K.callbacks.LearningRateScheduler(learning_rate_schedule)]
    )

    model.save('cifar10.h5')
