#!/bin/bash
#SBATCH -J cifar-resnet
#SBATCH -C knl
#SBATCH -N 1
#SBATCH --reservation=sc18
#SBATCH -q regular
#SBATCH -t 45
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
config=configs/cifar10_resnet.yaml
srun -l python train.py $config --distributed
