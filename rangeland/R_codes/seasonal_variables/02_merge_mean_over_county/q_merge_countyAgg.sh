#!/bin/bash

#SBATCH --partition=kamiak,cahnrs,cahnrs_bigmem,cahnrs_gpu,stockle ##rajagopalan
#SBATCH --requeue
#SBATCH --job-name=SeasVGrid_outer # Job Name
#SBATCH --time=00-12:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=01GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/h.noorazar/rangeland/seasonal_variables/02_merge_mean_over_county/error/outer_merge_countyAgg.o
#SBATCH  --error=/home/h.noorazar/rangeland/seasonal_variables/02_merge_mean_over_county/error/outer_merge_countyAgg.e
echo
echo "--- We are now in $PWD, running an R script ..."
echo

# Load R on compute node
module load r/4.1.0
Rscript --vanilla /home/h.noorazar/rangeland/seasonal_variables/02_merge_mean_over_county/d_merge_countyAgg.R

echo
echo "----- DONE -----"
echo

exit 0
