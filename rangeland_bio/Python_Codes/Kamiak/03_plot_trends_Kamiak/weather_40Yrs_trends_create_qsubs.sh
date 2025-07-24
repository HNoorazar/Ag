#!/bin/bash
cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/

for plot_what in stats ACF1 trends
do
  cp weather_40Yrs_trends_templ.sh  ./qsubs/weather_40Yrs_$plot_what.sh
  sed -i s/plot_what/"$plot_what"/g ./qsubs/weather_40Yrs_$plot_what.sh
done
