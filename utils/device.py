"""
Hardware/device configuration
"""

# System
import os

# Haswell settings for now
def configure_session():
    os.environ['OMP_NUM_THREADS'] = '32'
    import keras
    import tensorflow as tf
    config = tf.ConfigProto(
        inter_op_parallelism_threads=2,
        intra_op_parallelism_threads=32
    )
    keras.backend.set_session(tf.Session(config=config))
