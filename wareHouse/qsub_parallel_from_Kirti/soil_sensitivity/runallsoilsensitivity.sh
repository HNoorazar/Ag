#!/bin/bash
#for runname in soil_param_b-inf_1.1 soil_param_b-inf_.9 soil_param_Ds_1.1 soil_param_Ds_.9 soil_param_Ds-max_1.1 soil_param_Ds-max_.9 soil_param_soildepth-2_1.1 soil_param_soildepth-2_.9 soil_param_soildepth-3_1.1 soil_param_soildepth-3_.9 soil_param_Ws_1.1 soil_param_Ws_.9

for runname in soil_param_b-inf_.9 soil_param_Ds_1.1 soil_param_Ds_.9 soil_param_Ds-max_1.1 soil_param_Ds-max_.9 soil_param_soildepth-2_1.1 soil_param_soildepth-2_.9 soil_param_soildepth-3_1.1 soil_param_soildepth-3_.9 soil_param_Ws_1.1 soil_param_Ws_.9

#for runname in soil_param_b-inf_1.1
do
cd $runname
for (( i = 1; i <= 87; i++ ))
do
qsub ./qsub_set$i
done
cd ..
done
