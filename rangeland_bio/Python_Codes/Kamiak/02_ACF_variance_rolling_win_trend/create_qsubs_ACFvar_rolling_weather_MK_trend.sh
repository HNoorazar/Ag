#!/bin/bash
cd /home/h.noorazar/rangeland_bio/02_ACF_variance_rolling_win_trend/


for y_ in ACF variance
do
  cp ACFvar_rolling_weather_MK_trend_templ.sh   ./qsubs/weather_rolling_trend_$y_.sh
  sed -i s/y_/"$y_"/g                           ./qsubs/weather_rolling_trend_$y_.sh
done