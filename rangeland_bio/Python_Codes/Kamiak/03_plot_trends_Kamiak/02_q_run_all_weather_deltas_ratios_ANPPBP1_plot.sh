#!/bin/bash
cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/qsubs

for diff_or_ratio in diff ratio
do
  sbatch ./02_weather_deltas_ratios_ANPPBP1_plot_$diff_or_ratio.sh
done