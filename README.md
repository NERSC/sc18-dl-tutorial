# SC18 Tutorial: Deep Learning At Scale

This repository contains the Keras code for the SC18 tutorial: *Deep Learning at Scale*.

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

Horovod ResNet + ImageNet example:
https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py

Flexible implementation of ResNet and variants:
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py

CIFAR10 CNN and ResNet examples:
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
