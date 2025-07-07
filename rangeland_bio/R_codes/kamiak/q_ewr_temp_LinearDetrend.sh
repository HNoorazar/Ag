#!/bin/bash
#SBATCH --partition=rajagopalan
#SBATCH --requeue
#SBATCH --job-name=ewr_func # Job Name
#SBATCH --time=0-12:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=06GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/h.noorazar/rangeland_bio/ewrs_R/error/ewr_func_LinearDetrend.o
#SBATCH --error=/home/h.noorazar/rangeland_bio/ewrs_R/error/ewr_func_LinearDetrend.e

echo
echo "--- We are now in $PWD, running an R script ..."
echo

# Load R on compute node
module load r/4.1.0

# cd /data/project/agaid/AnalogData_Sid/Creating_Variables_old/
Rscript --vanilla /home/h.noorazar/rangeland_bio/ewrs_R/ewrs_kamiak_LinearDetrend.R ewr_func

echo
echo "----- DONE -----"
echo

exit 0