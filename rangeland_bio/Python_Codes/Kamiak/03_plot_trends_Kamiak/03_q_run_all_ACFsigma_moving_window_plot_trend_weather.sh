#!/bin/bash

cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/qsubs

for acf_or_variance in ACF1 variance
do
  for variable_set in weather drought
  do
    sbatch ./03_moving_win_plot_trend_$acf_or_variance$variable_set.sh
  done
done

