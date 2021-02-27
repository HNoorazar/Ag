#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N hardiness

#PBS -l nodes=1:ppn=16,walltime=2:00:00,mem=16gb
#PBS -q fast

#PBS -e /home/kraghavendra/hardiness/error/hardiness_error.e
#PBS -o /home/kraghavendra/hardiness/output/hardiness_output.o

##PBS -M k.kapuraghavendra@wsu.edu
##PBS -m abe

##PBS -t 1-321081%250
##PBS -t $start-$end

# First we ensure a clean running environment:
module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

echo ${PBS_ARRAYID}
Rscript --vanilla /home/kraghavendra/hardiness/hardiness_driver.R 311090 #${PBS_ARRAYID}


echo
echo ${PBS_ARRAYID}
echo "----- DONE -----"
echo

exit 0
