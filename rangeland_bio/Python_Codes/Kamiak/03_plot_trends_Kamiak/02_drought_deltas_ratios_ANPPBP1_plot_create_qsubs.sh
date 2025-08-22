#!/bin/bash
cd /home/h.noorazar/rangeland_bio/03_plot_trends_Kamiak/

for diff_or_ratio in diff ratio
do
  for batch_number in {1..14}
  do
    cp 02_drought_deltas_ratios_ANPPBP1_plot_templ.sh ./qsubs/02_drought_ANPPBP1_plot_$diff_or_ratio$batch_number.sh
    sed -i s/diff_or_ratio/"$diff_or_ratio"/g         ./qsubs/02_drought_ANPPBP1_plot_$diff_or_ratio$batch_number.sh
    sed -i s/batch_number/"$batch_number"/g           ./qsubs/02_drought_ANPPBP1_plot_$diff_or_ratio$batch_number.sh
  done
done