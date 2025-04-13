#!/bin/bash
cd /home/h.noorazar/NDVI_v_Weather/DL_models

outer=1
for NDVI_lag_or_delta in delta lag
do
  for architecture in paper Network2
  do
    cp KerasTuner_template.sh                          ./qsubs/KerasTuner_$architecture$NDVI_lag_or_delta.sh
    sed -i s/outer/"$outer"/g                          ./qsubs/KerasTuner_$architecture$NDVI_lag_or_delta.sh
    sed -i s/NDVI_lag_or_delta/"$NDVI_lag_or_delta"/g  ./qsubs/KerasTuner_$architecture$NDVI_lag_or_delta.sh
    sed -i s/architecture/"$architecture"/g            ./qsubs/KerasTuner_$architecture$NDVI_lag_or_delta.sh
    let "outer+=1" 
  done
done

