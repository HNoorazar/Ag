#!/bin/bash
cd /home/h.noorazar/rangeland_bio/plot_bySlope

outer=1
for slope_class in 1 2 3
do
  cp  plot_origGreen_bySlope.sh          ./qsubs/q_origGreen_$outer.sh
  sed -i s/slope_class/"$slope_class"/g  ./qsubs/q_origGreen_$outer.sh
  let "outer+=1"
done