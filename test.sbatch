#!/usr/bin/bash

#SBATCH --account=bejr-delta-gpu
#SBATCH --time=4:00:00
#SBATCH --job-name=mncifar
#SBATCH --partition=gpuA100x4
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=out/%x-%j.out

echo 'hello'

source .venv/bin/activate
which python

chmod +x mnistcifar-irmv1iht.sh
./mnistcifar-irmv1iht.sh

