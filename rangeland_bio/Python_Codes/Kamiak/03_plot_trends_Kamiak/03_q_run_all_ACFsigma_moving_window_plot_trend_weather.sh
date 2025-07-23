#!/bin/bash

cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/qsubs

for acf_or_variance in ACF1 variance
do
  sbatch ./03_ACFsigma_moving_window_plot_trend_weather_$acf_or_variance.sh
done

