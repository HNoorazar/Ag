#!/bin/bash
cd /home/h.noorazar/rangeland_bio/01_rolling_variance/

outer=1

for y_ in mean_lb_per_acr anpp_detrendLinReg anpp_detrendDiff anpp_detrendSens
do
  for window_size in 5 6 7 8 9 10
  do
    cp variance_rolling_temp.sh               ./qsubs/variance_rolling_$window_size$y_.sh
    sed -i s/window_size/"$window_size"/g ./qsubs/variance_rolling_$window_size$y_.sh
    sed -i s/y_/"$y_"/g                   ./qsubs/variance_rolling_$window_size$y_.sh
    let "outer+=1"
    let "batch_no+=1"
  done
done