#!/bin/bach
DATASET=$1
#SBATCH --job-nam=hps_parallel_$DATASET
#SBATCH--output=hps_parallel_$DATASET%j.out
#SBATCH --error=hps_parallel_$DATASET%j.err
#SBATCH --constraint=v100-16g
#SBATCH --nodes=1
#SBATCH ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-tasks=4
#SBATCH --time=00:10:00
#SBATCH --hint=nomultithread
#SBATCH --account=enh@v100
