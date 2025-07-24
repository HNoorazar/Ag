#!/bin/bash

cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/qsubs

for plot_what in stats ACF1 trends
do
  sbatch ./weather_40Yrs_$plot_what.sh
done

