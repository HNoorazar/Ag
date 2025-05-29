#!/bin/bash
cd /home/h.noorazar/rangeland_bio/rolling_ACF1/

outer=1
for window_size in 5 6 7 8 9 10
do
  cp ACF1_rolling_temp.sh               ./qsubs/ACF1_rolling_$window_size.sh
  sed -i s/window_size/"$window_size"/g ./qsubs/ACF1_rolling_$window_size.sh
  let "outer+=1"
  let "batch_no+=1"
done
