#!/bin/bash
cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/

for acf_or_variance in ACF1 variance
do
  for variable_set in weather drought
  do
    cp 03_ACFsigma_moving_window_plot_trend_weather_templ.sh ./qsubs/03_ACFsigma_moving_window_plot_trend_weather_$acf_or_variance$variable_set.sh
    sed -i s/acf_or_variance/"$acf_or_variance"/g            ./qsubs/03_ACFsigma_moving_window_plot_trend_weather_$acf_or_variance$variable_set.sh
    sed -i s/variable_set/"$variable_set"/g                  ./qsubs/03_ACFsigma_moving_window_plot_trend_weather_$acf_or_variance$variable_set.sh
  done
done
