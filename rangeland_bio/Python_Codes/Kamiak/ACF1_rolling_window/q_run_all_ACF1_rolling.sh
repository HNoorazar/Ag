#!/bin/bash

cd /home/h.noorazar/rangeland_bio/rolling_ACF1/qsubs

for y_ in mean_lb_per_acr anpp_detrendLinReg anpp_detrendSens
do
  for window_size in 5 6 7 8 9 10
  do
    sbatch ./ACF1_rolling_$window_size$y_.sh
  done
done

