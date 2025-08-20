#!/bin/bash
cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/

for diff_or_ratio in diff ratio
do
  cp 02_weather_deltas_ratios_ANPPBP1_plot_templ.sh ./qsubs/02_weather_deltas_ratios_ANPPBP1_plot_$diff_or_ratio.sh
  sed -i s/diff_or_ratio/"$diff_or_ratio"/g         ./qsubs/02_weather_deltas_ratios_ANPPBP1_plot_$diff_or_ratio.sh
done