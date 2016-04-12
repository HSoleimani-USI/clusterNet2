#!/bin/bash -l
#
#PBS -q qmic
#PBS -N job_name
#PBS -A DD-16-7
#PBS -l select=2:ncpus=16
#PBS -l walltime=01:00:00

# Load modules
#module load intel
#module load icc/2016.1.150-GCC-4.9.3-2.25
#module load imkl

# cd to the directory from where the job was started
cd $PBS_O_WORKDIR

# Your code
echo "Your commands here ..."
mpirun -n 2