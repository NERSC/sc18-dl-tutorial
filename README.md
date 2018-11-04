# SC18 Tutorial: Deep Learning At Scale

This repository contains the Keras code for the SC18 tutorial: *Deep Learning at Scale*.

## Single-node CNN on CIFAR10

`sbatch -J cifar-cnn -N 1 scripts/batchScript.sh configs/cifar10_cnn.yaml`

## Single-node ResNet18 on CIFAR10

`sbatch -J cifar-resnet -N 1 scripts/batchScript.sh configs/cifar10_resnet.yaml`

## Multi-node ResNet18 on CIFAR10

`sbatch -J cifar-resnet -N 8 scripts/batchScript.sh configs/cifar10_resnet.yaml`

## Single-node ResNet50 on ImageNet-100

`sbatch -J imagenet-resnet -N 1 scripts/batchScript.sh configs/imagenet_resnet.yaml`

## Inspiration

Horovod ResNet + ImageNet example:
https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py

Flexible implementation of ResNet and variants:
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py

CIFAR10 CNN and ResNet examples:
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
