#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N hardiness

#PBS -l nodes=1:ppn=1,walltime=2:00:00,mem=8gb
#PBS -q fast

#PBS -e /home/kraghavendra/hardiness/error/facet_plot_error.e
#PBS -o /home/kraghavendra/hardiness/output/facet_plot_output.o

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
Rscript --vanilla /home/kraghavendra/hardiness/facet_observed.R


echo
echo "----- DONE -----"
echo

exit 0
