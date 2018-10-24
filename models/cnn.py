"""
Simple CNN classifier model.
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_model(input_shape=(32, 32, 3), n_classes=10,
                optimizer='Nadam', lr=0.001):

    # Construct the simple CNN model
    conv_args = dict(kernel_size=3, padding='same', activation='relu')
    model = Sequential([
        Conv2D(16, input_shape=input_shape, **conv_args),
        MaxPooling2D(pool_size=2),
        Conv2D(32, **conv_args),
        MaxPooling2D(pool_size=2),
        Conv2D(64, **conv_args),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])

    # Construct the optimizer
    opt = keras.optimizers.Nadam(lr=lr)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model
