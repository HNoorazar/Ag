#!/bin/bash
#Should have the first global param file global_param_CRB_1
counter=-1
declare -a metdataarray=(HistPccsmT  HistPhadcmT  HistPpcmT   HistTcgcmP   HistTipslP HistPcgcmT  HistPipslT   HistTccsmP  HistThadcmP  HistTpcmP)
# declare -a metdataarray=(ccsm3_B1_2020-2049_proj cgcm3.1_t47_B1_2020-2049_proj ipsl_cm4_A1B_2020-2049_proj pcm1_A1B_2020-2049_proj hadcm_B1_2020-2049_proj HistPrecipHadcmB1Temp HistTempHadcmB1Precip)
for runname in HistPccsmT  HistPhadcmT  HistPpcmT   HistTcgcmP   HistTipslP HistPcgcmT  HistPipslT   HistTccsmP  HistThadcmP  HistTpcmP
#ccsm cgcm ipsl pcm hadcm_histcropmix histPhadcmT histThadcmP 
do
let counter=$counter+1
currentmetdata=${metdataarray[$counter]}
#echo $counter  bla  $currentmetdata
#for i in 2 3 4 5 6 7 8 9 10 11 12
cp qsub_TPeffect_Template ./TandPeffects/$runname/qsub_set1
sed -i s/thisjobname/"$runname"/g ./TandPeffects/$runname/qsub_set1
sed -i s/metdatafolder/"$currentmetdata"/g ./TandPeffects/$runname/qsub_set1
cd ./TandPeffects/$runname
for (( i = 2; i <= 87; i++ )) 
do
cp qsub_set1 qsub_set$i
sed -i s/set1/"set$i"/g ./qsub_set$i
done
cd ../../
done
