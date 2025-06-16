#!/bin/bash
cd /home/h.noorazar/rangeland_bio/ewrs_R

for ewr_func in generic qda sensitivity
do
  cp q_ewr_temp_LinearDetrend.sh  ./qsubs/q_ewr_LinearDetrend_$ewr_func.sh
  sed -i s/ewr_func/"$ewr_func"/g ./qsubs/q_ewr_LinearDetrend_$ewr_func.sh
done