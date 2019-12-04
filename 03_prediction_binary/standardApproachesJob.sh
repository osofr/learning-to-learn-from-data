#!/bin/bash
currDir='/home/aluedtke/funcGrad.jl/examples/ex2'
cd $currDir
# #SBATCH --job-name standardApproaches
# #SBATCH -o out_files/standardApproaches_$SLURM_ARRAY_TASK_ID.out    # File to which STDOUT will be written
# #SBATCH -e out_files/standardApproaches_$SLURM_ARRAY_TASK_ID.err   # File to which STDERR will be written
# #SBATCH --cpus-per-task 50
# #SBATCH --output=$currDir/

ml R
mpirun -v -n 1 R --vanilla "--args settingInd=$SLURM_ARRAY_TASK_ID" < standardApproachesJob.R > standardApproaches_$SLURM_ARRAY_TASK_ID.Rout