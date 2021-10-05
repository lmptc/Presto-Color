#!/bin/bash
#SBATCH -N 5
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J ProbatilityCube
#SBATCH --mail-user=lianming@udel.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 5 -c 64 --cpu_bind=cores python3 /global/homes/l/lianming/ProbabilityCube.py
