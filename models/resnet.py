"""
ResNet models for Keras.
"""

# Externals
import keras
from keras_contrib.applications import resnet

def build_resnet18_cifar(input_shape=(32, 32, 3), n_classes=10,
                         optimizer='Adam', lr=0.001,
                         metrics=['accuracy']):
    """Build the resnet18 model with appropriate settings for CIFAR10"""

    # These are the recommended settings for CIFAR10 from
    # keras_contrib/applications/resnet.py
    model = resnet.ResNet(input_shape=input_shape,
                          classes=n_classes,
                          block='basic',
                          repetitions=[2, 2, 2, 2],
                          include_top=True,
                          initial_strides=(1, 1),
                          initial_kernel_size=(3, 3),
                          initial_pooling=None,
                          top='classification')

    # Construct the optimizer
    if optimizer == 'Adam':
        opt = keras.optimizers.Adam(lr=lr)
    elif optimizer == 'Nadam':
        opt = keras.optimizers.Nadam(lr=lr)
    else:
        raise ValueError('Optimizer %s unsupported' % optimizer)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=metrics)
    return model

def _test():
    model = build_resnet18_cifar()
    model.summary()
