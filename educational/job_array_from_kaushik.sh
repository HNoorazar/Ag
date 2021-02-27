#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N hello_world

#PBS -l nodes=1:dev:ppn=1,walltime=1:00:00,mem=4gb
#PBS -q fast

#PBS -e /home/kraghavendra/Hello/hello_error.e
#PBS -o /home/kraghavendra/Hello/hello_output.o

#PBS -M k.kapuraghavendra@wsu.edu
#PBS -m abe

#PBS -t 1-10

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla /home/kraghavendra/Hello/Hello_world.R ${PBS_ARRAYID}


echo
echo ${PBS_ARRAYID}
echo "----- DONE -----"
echo

exit 0