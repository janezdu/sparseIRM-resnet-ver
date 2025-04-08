#!/usr/bin/bash

#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=mncifar
#SBATCH --partition=gpuA100x4
#SBATCH --account=bejr-delta-gpu
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=out/%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none


echo 'hello'
chmod +x mnistcifar-irmv1iht.sh
./mnistcifar-irmv1iht.sh

