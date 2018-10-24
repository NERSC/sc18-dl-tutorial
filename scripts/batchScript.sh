#!/bin/bash
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30

config=configs/cifar10_resnet.yaml
if [ $# -gt 0 ]; then config=$1; fi

. scripts/setup.sh
python train.py $config
