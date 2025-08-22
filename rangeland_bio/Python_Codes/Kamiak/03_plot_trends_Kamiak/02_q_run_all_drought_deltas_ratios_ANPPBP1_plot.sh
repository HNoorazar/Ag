#!/bin/bash
cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/qsubs

for diff_or_ratio in diff ratio
do
  for batch_number in {1..14}
  do
    sbatch ./02_drought_ANPPBP1_plot_$diff_or_ratio$batch_number.sh
  done
done