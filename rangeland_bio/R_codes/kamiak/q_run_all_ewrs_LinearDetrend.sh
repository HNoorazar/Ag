#!/bin/bash

cd /home/h.noorazar/rangeland_bio/ewrs_R/qsubs

for ewr_func in generic qda sensitivity
do
sbatch ./q_ewr_LinearDetrend_$ewr_func.sh
done