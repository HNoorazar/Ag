#!/bin/bash

cd /home/h.noorazar/NASA/trend/clean_plots_4_DL/qsubs

#for runname in {1..160}
# do
for ML_model in RF DL SVM kNN
do 
  for indeks in EVI NDVI
  do
    for smooth in SG regular
    do
      batch_no=1
      while [ $batch_no -le 40 ]
      do
        sbatch ./trend_DLPlot_$indeks$smooth$batch_no$ML_model.sh
        let "batch_no+=1"
      done
    done
  done
done
