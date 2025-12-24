#!/bin/bash
#SBATCH --job-name=prof_serial
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=logs/prof_serial_%j.log
#SBATCH --error=logs/errors/prof_serial_%j.err

module load oneapi/vtune/latest
vtune -collect hotspots -result-dir vtune_reports/prof_serial -- ./bin/nbody_serial 20000 1000 true
