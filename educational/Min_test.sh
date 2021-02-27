#!/bin/bash -u
echo "Entering optimize_wb_SLAVE.sh..."
echo "WORKDIR=${1}" 
echo "RESULTDIR=${2}" 
echo "MOCOMRUN=${3}" 
echo "MODELRUN=${4}" 
echo "BASIN=${5}" 
echo "BI=${6}" 
echo "DS=${7}" 
echo "DSMAX=${8}" 
echo "WS=${9}" 
echo "D2=${10}" 
echo "SCRIPTDIR=${11}"

WORKDIR=${1} 
RESULTDIR=${2} 
MOCOMRUN=${3} 
MODELRUN=${4} 
BASIN=${5} 
BI=${6} 
DS=${7} 
DSMAX=${8} 
WS=${9} 
D2=${10} 
SCRIPTDIR=${11}

EACH_SOIL_PIECE=25
source /data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Simulation/Scenario/Kamiak_CRB_calibration/GLOBAL_PATH_ENVIRONMENT.txt
#station_obs_dir=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Datasets/Calibration/adjusted_streamflow_for_calibration
#UH_DIR=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Datasets/Calibration/Uhs
STATS=${WORKDIR}/flow/${BASIN}.stats
#CALIBRATION_SCRIPT_DIR=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Simulation/Scenario/Kamiak_CRB_calibration
#US using GridMet
#GLOBALFILESRC_US=${CALIBRATION_SCRIPT_DIR}/vic_control_mode_no_frozen_gridmet_extvar_calibration.txt
GLOBALFILE_US=${WORKDIR}/US_VIC.conf
#CA using Livneh
#GLOBALFILESRC_CA=${CALIBRATION_SCRIPT_DIR}/vic_control_mode_no_frozen_livneh_var_calibration.txt
GLOBALFILE_CA=${WORKDIR}/CA_VIC.conf

#SIMULATED_REULTS_DIR=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Simulation/Results/Calibration/vic_results_with_calibrated_soils/results/flux
SIMULATED_REULTS_DIR=$WORKDIR/../../../subset_flux
#entire_soil_file=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Simulation/Database/Soil/newsoil_with_soc_corrected_org_density_newUM.txt
#prepost_process_script_dir=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Simulation/Script
#cropscenario_file=${CALIBRATION_SCRIPT_DIR}/.CropSyst_scenario
#basin_cell_list_dir=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Datasets/Calibration/Basins
#CA_cell_list=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Simulation/Database/Soil/CA_cell_list.txt
#US_cell_list=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Simulation/Database/Soil/US_cell_list.txt
SOILFILE_ALLBASIN=$WORKDIR/soil_allbasins.txt
SOIL_SIMULATED=$WORKDIR/soil_already_simulated.txt
SOILFILESRC=$WORKDIR/soil_orig.txt
SOILFILEDEST=$WORKDIR/soil.txt
SOIL_CA=$WORKDIR/CA_soil.txt
SOIL_US=$WORKDIR/US_soil.txt

#irrigation=FALSE
#irrigationfile=/data/adam/mingliang.liu/Projects/BPA_CRB/VIC-CropSyst/Simulation/Database/Management/Umatila_irrigation_parameter_180807.txt
#crop_out=crop
#vic_out=flux

echo "WORKDIR:${WORKDIR}"

if ! [ -f ${WORKDIR}/.CropSyst_scenario ]; then
  cp ${cropscenario_file} ${WORKDIR}/.CropSyst_scenario
fi
if ! [ -f ${SOILFILESRC} ]; then
  #python ${prepost_process_script_dir}/select_subset_soil_parameter_from_cellidlist.py $entire_soil_file ${basin_cell_list_dir}/w${BASIN}.txt ${SOILFILE_ALLBASIN}
  #python ${prepost_process_script_dir}/create_subset_soil_not_simulated.py ${SOILFILE_ALLBASIN} ${SOILFILESRC} ${SOIL_SIMULATED}
  echo '${SOILFILESRC}:'${SOILFILESRC}
  echo 'Should copy from: '"${WORKDIR}/../../../../soil_orig.txt"
  if [ -f ${WORKDIR}/../../../soil_orig.txt ]; then
      cp ${WORKDIR}/../../../soil_orig.txt ${SOILFILESRC}
      echo "cp ${WORKDIR}/../../../soil_orig.txt ${SOILFILESRC}"
  fi
  if [ -f ${WORKDIR}/../../../soil_already_simulated.txt ]; then
      cp ${WORKDIR}/../../../soil_already_simulated.txt ${SOIL_SIMULATED}
      echo "cp ${WORKDIR}/../../../soil_already_simulated.txt ${SOIL_SIMULATED}"
  fi
fi

current_dir=$PWD
rm -Rf flux flow rout_input rout_inp.*
rm -Rf ${WORKDIR}/flux
rm -Rf ${WORKDIR}/flow
cd ${WORKDIR}

# Setup
mkdir -p ${WORKDIR}/flow ${WORKDIR}/flux
awk '{if($1==1){$5 = '"${BI}"'; $6 = '"${DS}"'; $7 = '"${DSMAX}"'; $8 = '"${WS}"'; $95 = '"${D2}"'} print $0;}' "${SOILFILESRC}" >| "${SOILFILEDEST}"

#seperate soil into CA and US part
echo '${SOILFILEDEST}':${SOILFILEDEST}
python ${prepost_process_script_dir}/split_soil_parameter_into_US_CA.py ${SOILFILEDEST} ${CA_cell_list} ${US_cell_list} ${SOIL_CA} ${SOIL_US}

#########################################
##    Run VIC, routing, stats, plot    ##
#########################################
source ${station_obs_dir}/${BASIN}_date_info
#startyear=1979
#endyear=2015
echo "srun singlejob_VIC_Cropsyst.srun..."
#srun -J VIC${BASIN} ${CALIBRATION_SCRIPT_DIR}/singlejob_VIC_Cropsyst.srun ${WORKDIR} ${GLOBALFILESRC} ${SOILFILEDEST} ${startyear} ${endyear} ${RESULTDIR} ${irrigation} ${irrigationfile} ${crop_out} ${vic_out} ${GLOBALFILE}  
#sh ${CALIBRATION_SCRIPT_DIR}/singlejob_VIC_Cropsyst.srun ${WORKDIR} ${GLOBALFILESRC} ${SOILFILEDEST} ${startyear} ${endyear} ${RESULTDIR}/flux ${irrigation} ${irrigationfile} ${crop_out} ${vic_out} ${GLOBALFILE}  

rm -f *_p*_done

numpieces_ca=0
numpieces_us=0
if [ -f ${SOIL_CA} ]; then
    #sh ${prepost_process_script_dir}/split_soil.sh ${SOIL_CA} ${EACH_SOIL_PIECE} SOIL_CA
    numpieces_ca=$( sh ${prepost_process_script_dir}/split_soil.sh ${SOIL_CA} ${EACH_SOIL_PIECE} SOIL_CA )
    for p in $(seq 0 ${numpieces_ca}); do
    sbatch -J CA_VIC_p${p} --partition=adam,kamiak,stockle --account=stockle --nodes=1 --time=0-24:00:00 --ntasks-per-node=1 --mem=1000 ${CALIBRATION_SCRIPT_DIR}/singlejob_VIC_Cropsyst.srun ${WORKDIR} ${GLOBALFILESRC_CA} $WORKDIR/SOIL_CA_p${p}.txt ${startyear} ${endyear} ${RESULTDIR}/flux ${irrigation} ${irrigationfile} ${crop_out} ${vic_out} ${GLOBALFILE_CA}_p${p}
    done
    #sh ${CALIBRATION_SCRIPT_DIR}/singlejob_VIC_Cropsyst.srun ${WORKDIR} ${GLOBALFILESRC_CA} ${SOIL_CA} ${startyear} ${endyear} ${RESULTDIR}/flux ${irrigation} ${irrigationfile} ${crop_out} ${vic_out} ${GLOBALFILE_CA}  
fi
if [ -f ${SOIL_US} ]; then
    #sh ${prepost_process_script_dir}/split_soil.sh ${SOIL_US} ${EACH_SOIL_PIECE} SOIL_US
    numpieces_us=$( sh ${prepost_process_script_dir}/split_soil.sh ${SOIL_US} ${EACH_SOIL_PIECE} SOIL_US )
    for p in $(seq 0 ${numpieces_us}); do
    sbatch -J US_VIC_p${p} --partition=adam,kamiak,stockle --account=stockle --nodes=1 --time=0-24:00:00 --ntasks-per-node=1 --mem=1000 ${CALIBRATION_SCRIPT_DIR}/singlejob_VIC_Cropsyst.srun ${WORKDIR} ${GLOBALFILESRC_US} $WORKDIR/SOIL_US_p${p}.txt ${startyear} ${endyear} ${RESULTDIR}/flux ${irrigation} ${irrigationfile} ${crop_out} ${vic_out} ${GLOBALFILE_US}_p${p}
    done
    #sh ${CALIBRATION_SCRIPT_DIR}/singlejob_VIC_Cropsyst.srun ${WORKDIR} ${GLOBALFILESRC_US} ${SOIL_US} ${startyear} ${endyear} ${RESULTDIR}/flux ${irrigation} ${irrigationfile} ${crop_out} ${vic_out} ${GLOBALFILE_US}  
fi

#sh ${prepost_process_script_dir}/select_and_copy_simulated_results.sh ${SIMULATED_REULTS_DIR} ${startyear} ${endyear} ${SOIL_SIMULATED} ${RESULTDIR}/flux

if [ -f ${SOIL_SIMULATED} ]; then
    sh ${prepost_process_script_dir}/select_and_gen_symlink_simulated_results.sh ${SIMULATED_REULTS_DIR} ${SOIL_SIMULATED} ${RESULTDIR}/flux
fi

#wait untill all pieces are finished
if [ -f ${SOIL_CA} ]; then
    for p in $(seq 0 ${numpieces_ca}); do
        while ! [ -f ${GLOBALFILE_CA}_p${p}_done ]; do
            sleep 30
        done
    done
fi
if [ -f ${SOIL_US} ]; then
    for p in $(seq 0 ${numpieces_us}); do
        while ! [ -f ${GLOBALFILE_US}_p${p}_done ]; do
            sleep 30
        done
    done
fi

#while [ -f ${SOIL_CA} ] && ! [ -f ${GLOBALFILE_CA}_done ]
#do
#    sleep 10
#   echo "sleep ${MOCOMRUN}:${MODELRUN}:US"
#done
#
#while [ -f ${SOIL_US} ] && ! [ -f ${GLOBALFILE_US}_done ]
#do
#    sleep 10
#   echo "sleep ${MOCOMRUN}:${MODELRUN}:CA"
#done

#The above procedure can use sbatch to seperate smaller soil files. The next step will check the wholeness of simulation

echo "$(date +%F\ %H:%M)  Running Routing Model"
sh ${CALIBRATION_SCRIPT_DIR}/run_rout.sh ${WORKDIR} ${RESULTDIR}/flux/flux_ ${BASIN}  
#srun --partition=adam,kamiak,stockle --account=stockle --nodes=1 --time=0-24:00:00 --mem=2000 --cpus-per-task=4 ${CALIBRATION_SCRIPT_DIR}/run_rout.sh ${WORKDIR} ${RESULTDIR}/flux/flux_ ${BASIN}  

if ! [ -f ${UH_DIR}/${BASIN}.uh_s ]; then
  cp ${WORKDIR}/flow/${BASIN}.uh_s ${UH_DIR}/${BASIN}.uh_s
  sleep 5
fi

#190404 allocate large mem
#srun --partition=adam,kamiak,stockle --account=stockle --nodes=1 --time=0-24:00:00 --ntasks-per-node=1 --mem=4000 ${CALIBRATION_SCRIPT_DIR}/run_rout.sh ${WORKDIR} ${RESULTDIR}/flux/flux_ ${BASIN}  


echo "check rout file: ${WORKDIR}/flow/${BASIN}.day"
[[ -e ${WORKDIR}/flow/${BASIN}.day ]] || { echo "Routing failed.  See rout.out for more information." ; exit 1 ; }


echo "$(date +%F\ %H:%M)  Calculating output stats"
sh ${CALIBRATION_SCRIPT_DIR}/run_stat.sh $WORKDIR/flow ${BASIN}
echo "$(date +%F\ %H:%M)  Finish stats"
mv $WORKDIR/flow/tempr2file.txt $RESULTDIR/stats.txt
mv $WORKDIR/flow/"${BASIN}".day .
rm -Rf $RESULTDIR/flux >& /dev/null
cd $current_dir
