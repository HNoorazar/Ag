#!/bin/bash
cd /home/h.noorazar/NDVI_v_Weather/DL_models

outer=1
for NDVI_lag_or_delta in delta lag
do
  cp DL_NDVI_weather_model_NB1_template.sh           ./qsubs/DL_NDVI_weather_model_NB1_$NDVI_lag_or_delta.sh
  sed -i s/outer/"$outer"/g                          ./qsubs/DL_NDVI_weather_model_NB1_$NDVI_lag_or_delta.sh
  sed -i s/NDVI_lag_or_delta/"$NDVI_lag_or_delta"/g  ./qsubs/DL_NDVI_weather_model_NB1_$NDVI_lag_or_delta.sh
  let "outer+=1" 
done

