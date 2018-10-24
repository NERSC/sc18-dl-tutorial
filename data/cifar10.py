"""
CIFAR10 dataset specification.

https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
"""

# Externals
import keras
from keras.datasets import cifar10

def get_datasets(n_train=None, n_valid=None):
    """
    Load the CIFAR10 data and construct pipeline.
    TODO: add the data augmentation.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert labels to class vectors
    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    return (x_train, y_train), (x_test, y_test)
