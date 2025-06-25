#!/bin/bash
cd /home/h.noorazar/rangeland_bio/rolling_ACF1/

for y_ in avg_of_dailyAvgTemp_C temp_detrendLinReg temp_detrendDiff temp_detrendSens precip_mm prec_detrendLinReg prec_detrendDiff prec_detrendSens
do
  for window_size in 5 6 7 8 9 10
  do
    cp weather_ACF1_rolling_temp.sh       ./qsubs/weather_ACF1_rolling_$window_size$y_.sh
    sed -i s/window_size/"$window_size"/g ./qsubs/weather_ACF1_rolling_$window_size$y_.sh
    sed -i s/y_/"$y_"/g                   ./qsubs/weather_ACF1_rolling_$window_size$y_.sh
  done
done