#!/bin/bash

cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/qsubs

for plot_what in min mean median max var std ACF1 trends
do
  for variable_set in drought weather
  do
    sbatch ./weather_40Yrs_$plot_what$variable_set.sh
  done
done

