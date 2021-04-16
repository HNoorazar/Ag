#!/bin/bash

#PBS -V
#PBS -N merge

#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -l mem=40gb
##PBS -q fast

#PBS -k o
#PBS -e /home/hnoorazar/sid/analog_pairwise_2021/01_find_analogs/error/merge.e
#PBS -o /home/hnoorazar/sid/analog_pairwise_2021/01_find_analogs/error/merge.o
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

Rscript --vanilla d_merge_analog_loc_levels_sid.R

echo
echo "----- DONE -----"
echo

exit 0


