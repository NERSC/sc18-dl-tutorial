# SC18 Tutorial: Deep Learning At Scale

This repository contains the Keras code for the SC18 tutorial: *Deep Learning at Scale*.

It contains specifications for a few datasets, a couple of CNN models, and
all the training code to enable training the models in a distributed fashion
using Horovod.

As part of the tutorial, you will
1. Train a simple CNN to classify images from the CIFAR10 dataset on a single node
2. Train a ResNet model to classify the same images on multiple nodes

## Structure of this repository

**`train.py`** - the main training script which can be steered with YAML
configuration files.

**`data/`** - folder containing the specifications of the datasets. Each dataset
has a corresponding name which is mapped to the specification in `data/__init__.py`

## How to launch the examples.

### Single-node CNN on CIFAR10

`sbatch -N 1 scripts/cifar_cnn.sh`

### Single-node mini ResNet on CIFAR10

`sbatch -N 1 scripts/cifar_resnet.sh`

### Multi-node mini ResNet on CIFAR10

`sbatch -N 8 scripts/cifar_resnet.sh`

### Multi-node ResNet50 on ImageNet-100

`sbatch -N 16 scripts/imagenet_resnet.sh`

## Inspiration

Keras ResNet50 official model:
https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

Horovod ResNet + ImageNet example:
https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py

CIFAR10 CNN and ResNet examples:
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
