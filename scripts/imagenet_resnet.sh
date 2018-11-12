#!/bin/bash
#SBATCH -J imagenet-resnet
#SBATCH -C knl
#SBATCH -N 1
#SBATCH --reservation=sc18
#SBATCH -q regular
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
config=configs/imagenet_resnet.yaml
srun -l python train.py $config --distributed
