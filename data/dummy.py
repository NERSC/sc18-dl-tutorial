"""
Random dummy dataset specification.
"""

# Externals
import numpy as np

def get_datasets(n_train=1024, n_valid=1024,
                 input_shape=(3, 32, 32), target_shape=()):
    x_train = np.random.normal(size=(n_train,) + tuple(input_shape))
    x_valid = np.random.normal(size=(n_valid,) + tuple(input_shape))
    y_train = np.random.normal(size=(n_train,) + tuple(target_shape))
    y_valid = np.random.normal(size=(n_valid,) + tuple(target_shape))
    return (x_train, y_train), (x_valid, y_valid)
