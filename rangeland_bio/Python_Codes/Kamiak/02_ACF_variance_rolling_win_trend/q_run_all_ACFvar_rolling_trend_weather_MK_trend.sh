#!/bin/bash
cd /home/h.noorazar/rangeland_bio/02_ACF_variance_rolling_win_trend/qsubs
for y_ in ACF variance
do
  sbatch ./weather_rolling_trend_$y_.sh
done

