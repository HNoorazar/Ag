#!/bin/bash
cd /home/h.noorazar/rangeland_bio/ewrs_R

for ewr_func in ddjnonparam generic livpotential movpotential qda sensitivity
do
  cp q_ewr_temp.sh                  ./qsubs/q_ewr_$ewr_func.sh
  sed -i s/ewr_func/"$ewr_func"/g   ./qsubs/q_ewr_$ewr_func.sh
done