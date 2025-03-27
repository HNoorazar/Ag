#!/bin/bash

cd /home/h.noorazar/NDVI_v_Weather/DL_models/qsubs/

for NDVI_lag_or_delta in delta lag
do
  sbatch ./DL_Network2_$NDVI_lag_or_delta.sh
done