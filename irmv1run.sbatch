#!/usr/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=mncifar
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-user=zd16@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=out/%x-%j.out

echo 'hello'
chmod +x irmv1run.sh
./irmv1run.sh
