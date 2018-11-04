"""
Dataset specification for Tiny-ImageNet:
https://tiny-imagenet.herokuapp.com/
"""

# Externals
from keras.preprocessing.image import ImageDataGenerator

def get_datasets(batch_size, train_dir, valid_dir):
    train_gen = ImageDataGenerator(
        rescale=1./255, horizontal_flip=True,
        width_shift_range=0.2, height_shift_range=0.2
    )
    valid_gen = ImageDataGenerator(rescale=1./255)
    train_iter = train_gen.flow_from_directory(train_dir, batch_size=batch_size,
                                               target_size=(64, 64))
    valid_iter = valid_gen.flow_from_directory(valid_dir, batch_size=batch_size,
                                               target_size=(64, 64))
    return train_iter, valid_iter
