#!/bin/bash -l
#
#PBS -N job_name
#PBS -A DD-16-7
#PBS -l select=1:ncpus=16
#PBS -l walltime=01:00:00

#Load modules
module load icc/2016.1.150-GCC-4.9.3-2.25
mdoule load imkl

#cd to dir where job was started
cd $PBS_O_WORKDIR

echo "TEST"

