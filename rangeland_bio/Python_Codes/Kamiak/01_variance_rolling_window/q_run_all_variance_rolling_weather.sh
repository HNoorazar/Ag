#!/bin/bash
cd /home/h.noorazar/rangeland_bio/01_rolling_variance/qsubs

###for y_ in avg_of_dailyAvgTemp_C temp_detrendLinReg temp_detrendDiff temp_detrendSens precip_mm prec_detrendLinReg prec_detrendDiff prec_detrendSens
for y_ in avg_of_dailyAvgTemp_C \
          max_of_monthlyAvg_of_dailyMaxTemp_C \
          min_of_monthlyAvg_of_dailyMinTemp_C \
          avg_of_monthlymax_of_dailyMaxTemp_C \
          avg_of_monthlymin_of_dailyMinTemp_C \
          avg_of_dailyAvgTemp_C_detrendSens \
          avg_of_monthlymax_of_dailyMaxTemp_C_detrendSens \
          avg_of_monthlymin_of_dailyMinTemp_C_detrendSens \
           max_of_monthlyAvg_of_dailyMaxTemp_C_detrendSens \
           min_of_monthlyAvg_of_dailyMinTemp_C_detrendSens \
           avg_of_dailyAvgTemp_C_detrendLinReg \
           avg_of_monthlymax_of_dailyMaxTemp_C_detrendLinReg \
           avg_of_monthlymin_of_dailyMinTemp_C_detrendLinReg \
           max_of_monthlyAvg_of_dailyMaxTemp_C_detrendLinReg \
           min_of_monthlyAvg_of_dailyMinTemp_C_detrendLinReg \
           precip_mm \
           precip_mm_detrendLinReg \
           precip_mm_detrendSens \
           thi_avg \
           thi_avg_detrendSens \
           thi_avg_detrendLinReg \
           avg_of_dailyAvg_rel_hum \
           avg_of_dailyAvg_rel_hum_detrendSens \
           avg_of_dailyAvg_rel_hum_detrendLinReg
do
  for window_size in 5 6 7 8 9 10
  do
    sbatch ./weather_variance_rolling_$window_size$y_.sh
  done
done

