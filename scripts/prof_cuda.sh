#!/bin/bash
#SBATCH --job-name=prof_cuda
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/prof_cuda_%j.log
#SBATCH --error=logs/errors/prof_cuda_%j.err

module load cuda/11.0
nsys profile --stats=true --output=vtune_reports/prof_cuda_sys ./bin/nbody_cuda 20000 1000 true
ncu --set full --target-processes all --export vtune_reports/prof_cuda_kernel ./bin/nbody_cuda 20000 1000 true
