#!/bin/bash
cd /home/h.noorazar/NDVI_v_Weather/DL_models

outer=1
for NDVI_lag_or_delta in delta lag
do
  cp DL_Network2_template.sh           ./qsubs/DL_Network2_$NDVI_lag_or_delta.sh
  sed -i s/outer/"$outer"/g                          ./qsubs/DL_Network2_$NDVI_lag_or_delta.sh
  sed -i s/NDVI_lag_or_delta/"$NDVI_lag_or_delta"/g  ./qsubs/DL_Network2_$NDVI_lag_or_delta.sh
  let "outer+=1" 
done

