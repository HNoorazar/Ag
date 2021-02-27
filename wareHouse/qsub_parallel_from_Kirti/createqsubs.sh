#!/bin/bash
#Should have the first global param file global_param_CRB_1
for runname in hist hist_co2_437 hist_co2_461 
do
#for i in 2 3 4 5 6 7 8 9 10 11 12
cp qsub_template ./$runname/qsub_set1
sed -i s/thisjobname/"$runname"/g ./$runname/qsub_set1
cd $runname

for (( i = 2; i <= 87; i++ )) 
do
cp qsub_set1 qsub_set$i
sed -i s/set1/"set$i"/g ./qsub_set$i
done
cd ..
done
