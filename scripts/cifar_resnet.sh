#!/bin/bash
#SBATCH -J cifar-resnet
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
config=configs/cifar10_resnet.yaml
srun -l python train.py $config --distributed
