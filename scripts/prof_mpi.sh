#!/bin/bash
#SBATCH --job-name=prof_mpi
#SBATCH --partition=cpu
#SBATCH --nodes=24
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/prof_mpi_%j.log
#SBATCH --error=logs/errors/prof_mpi_%j.err

module load oneapi/vtune/latest
module load openmpi/4.1.1
vtune -collect hotspots -result-dir vtune_reports/prof_mpi -- mpirun -np 48 ./bin/nbody_mpi 20000 1000 true
