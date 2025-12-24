#!/bin/bash
#SBATCH --job-name=prof_openmp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=logs/prof_openmp_%j.log
#SBATCH --error=logs/errors/prof_openmp_%j.err

. /home/apps/spack/share/spack/setup-env.sh
module load intel-oneapi-vtune/2022.3.0-gcc-12.1.0-pdgi
export OMP_NUM_THREADS=48
vtune -collect hotspots -result-dir vtune_reports/prof_openmp_${SLURM_JOB_ID} -- ./bin/nbody_openmp 20000 1000 true
