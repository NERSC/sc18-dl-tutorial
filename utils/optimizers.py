"""
Utilty code for constructing optimizers and scheduling learning rates.
"""

# System
import math

# Externals
import keras
import horovod.keras as hvd

def get_optimizer(name, lr, n_ranks=1, scale_lr=True,
                  distributed=False, **opt_args):
    """
    Configure the optimizer and scale the learning rate by n_ranks.
    TODO: add support for sqrt scaling of learning rate.
    TODO: add support for wrapping TF optimizers like LARS.
    """
    # Scale the learning rate
    if scale_lr:
        lr = lr * n_ranks

    # Construct the optimizer
    OptType = getattr(keras.optimizers, name)
    opt = OptType(lr=lr, **opt_args)

    # Distributed optimizer wrapper
    if distributed:
        opt = hvd.DistributedOptimizer(opt)

    return opt
