#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J ProbatilityCube
#SBATCH --mail-user=lianming@udel.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load python
#run the application:
srun -n 1 -c 64 --cpu_bind=cores python3 /global/homes/l/lianming/Presto-Color-2/ProbabilityCube.py
