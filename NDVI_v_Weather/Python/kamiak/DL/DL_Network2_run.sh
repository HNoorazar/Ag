#!/bin/bash

cd /home/h.noorazar/NDVI_v_Weather/DL_models/qsubs/
for batch_or_not in noBatch withBatch
do
  for NDVI_lag_or_delta in delta lag
  do
    sbatch ./DL_Network2_$NDVI_lag_or_delta$batch_or_not.sh
  done
done