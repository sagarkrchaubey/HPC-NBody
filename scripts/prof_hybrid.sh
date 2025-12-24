#!/bin/bash
#SBATCH --job-name=prof_hybrid
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=logs/prof_hybrid_%j.log
#SBATCH --error=logs/errors/prof_hybrid_%j.err

module load oneapi/vtune/latest
module load openmpi/4.1.1
export OMP_NUM_THREADS=48
vtune -collect threading -result-dir vtune_reports/prof_hybrid -- mpirun -np 2 ./bin/nbody_hybrid 20000 1000 true
