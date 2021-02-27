#!/bin/bash
for runname in soil_param_b-inf_1.1 soil_param_b-inf_.9 soil_param_Ds_1.1 soil_param_Ds_.9 soil_param_Ds-max_1.1 soil_param_Ds-max_.9 soil_param_soildepth-2_1.1 soil_param_soildepth-2_.9 soil_param_soildepth-3_1.1 soil_param_soildepth-3_.9 soil_param_Ws_1.1 soil_param_Ws_.9
do
rm -rf $runname
mkdir $runname
cp qsub_template_soilsensitivity ./$runname/qsub_set1
sed -i s/thisjobname/"$runname"/g ./$runname/qsub_set1
cd $runname
for (( i = 2; i <= 87; i++ )) 
do
cp qsub_set1 qsub_set$i
sed -i s/set1/"set$i"/g ./qsub_set$i
done
cd ..
done
