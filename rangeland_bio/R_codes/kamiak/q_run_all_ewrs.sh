#!/bin/bash

cd /home/h.noorazar/rangeland_bio/ewrs_R/qsubs

for ewr_func in ddjnonparam generic livpotential movpotential qda_separate sensitivity_separate
do
sbatch ./q_ewr_$ewr_func.sh
done
