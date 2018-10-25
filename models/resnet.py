"""
ResNet models for Keras.

This currently uses the ResNet implementation from
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py
"""

# Externals
import keras
from keras_contrib.applications import resnet

def build_resnet18_cifar(input_shape=(32, 32, 3), n_classes=10, dropout=None):
    """Build the resnet18 model with appropriate settings for CIFAR10"""

    # These are the recommended settings for CIFAR10 from
    # keras_contrib/applications/resnet.py
    return resnet.ResNet(input_shape=input_shape,
                         classes=n_classes,
                         block='basic',
                         repetitions=[2, 2, 2, 2],
                         include_top=True,
                         dropout=dropout,
                         initial_strides=(1, 1),
                         initial_kernel_size=(3, 3),
                         initial_pooling=None,
                         top='classification')

def _test():
    model = build_resnet18_cifar()
    model.summary()
