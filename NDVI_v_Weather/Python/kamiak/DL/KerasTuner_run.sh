#!/bin/bash

cd /home/h.noorazar/NDVI_v_Weather/DL_models/qsubs/

for NDVI_lag_or_delta in delta lag
do
  for architecture in paper Network2
  do
    sbatch ./KerasTuner_$architecture$NDVI_lag_or_delta.sh
  done
done