#!/bin/bash
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/cifar10-%j.out

config=configs/cifar10_cnn.yaml
if [ $# -gt 0 ]; then config=$1; fi

. scripts/setup.sh
srun -l python train.py $config --distributed
