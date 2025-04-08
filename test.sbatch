#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bejr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=mncifar
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none    # <- or closest
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=out/%x-%j.out

echo 'hello'

source .venv/bin/activate
which python

chmod +x mnistcifar-irmv1iht.sh
./mnistcifar-irmv1iht.sh

