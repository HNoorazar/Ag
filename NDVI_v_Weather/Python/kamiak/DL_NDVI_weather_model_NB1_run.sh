#!/bin/bash

cd /home/h.noorazar/NDVI_v_Weather/DL_models/qsubs/

for NDVI_lag_or_delta in delta lag
do
  sbatch ./DL_NDVI_weather_model_NB1_$NDVI_lag_or_delta.sh
done