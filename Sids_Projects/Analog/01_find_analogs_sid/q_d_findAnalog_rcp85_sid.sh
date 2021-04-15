#!/bin/bash

#PBS -V
#PBS -N parwise_rcp85

#PBS -l nodes=1:ppn=1,walltime=06:00:00
#PBS -l mem=40gb
##PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/sid/analog_pairwise_2021/01_find_analogs/error/parwise_rcp85.e
#PBS -o /home/hnoorazar/sid/analog_pairwise_2021/01_find_analogs/error/parwise_rcp85.o
#PBS -m abe

cd /home/hnoorazar/sid/analog_pairwise_2021/01_find_analogs/
echo
echo We are in the $PWD directory
echo

module purge

# Load R
module load udunits/2.2.20
module load libxml2/2.9.4
module load gdal/2.1.2_gcc proj/4.9.2
module load gcc/7.3.0 r/3.5.1/gcc/7.3.0

Rscript --vanilla d_findAnalog_sid.R rcp85

echo
echo "----- DONE -----"
echo

exit 0


