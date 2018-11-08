#!/bin/bash
#SBATCH -J tinyimagenet-resnet
#SBATCH -C knl
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

. scripts/setup.sh
config=configs/tinyimagenet_resnet.yaml
srun -u -l python train.py $config --distributed
