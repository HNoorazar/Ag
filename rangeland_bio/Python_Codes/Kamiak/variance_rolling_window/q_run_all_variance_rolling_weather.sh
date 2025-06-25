#!/bin/bash
cd /home/h.noorazar/rangeland_bio/rolling_variance/qsubs

for y_ in avg_of_dailyAvgTemp_C temp_detrendLinReg temp_detrendDiff temp_detrendSens precip_mm prec_detrendLinReg prec_detrendDiff prec_detrendSens
do
  for window_size in 5 6 7 8 9 10
  do
    sbatch ./weather_variance_rolling_$window_size$y_.sh
  done
done

