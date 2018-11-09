#!/bin/bash
#SBATCH -J cifar-cnn
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
config=configs/cifar10_cnn.yaml
srun python train.py $config
