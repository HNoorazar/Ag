#!/bin/bash

cd /home/h.noorazar/NDVI_v_Weather/DL_models/qsubs/
for batch_or_not in noBatch withBatch
do
  for NDVI_lag_or_delta in delta lag
  do
    sbatch ./DL_NDVI_weather_model_NB1_$NDVI_lag_or_delta$batch_or_not.sh
  done
done