#!/bin/bash
##SBATCH --partition=kamiak
##SBATCH --constraint=cascadelake
#SBATCH --partition=rajagopalan,kamiak,stockle,cahnrs
#SBATCH --requeue
#SBATCH --job-name=weather_ACF1_rolling_window_size # Job Name
#SBATCH --time=0-1:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=16GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000
####SBATCH --array=0-30000%150 # %100 means let it run 150 jobs at a time. 30000 is total number of jobs.
###SBATCH -k o
#SBATCH --output=/home/h.noorazar/rangeland_bio/01_rolling_ACF1/error/weather_ACF1_rolling_window_size_y_.o
#SBATCH  --error=/home/h.noorazar/rangeland_bio/01_rolling_ACF1/error/weather_ACF1_rolling_window_size_y_.e
echo
echo "--- We are now in $PWD, running an R script ..."
echo

# Load R on compute node
# module load r/4.1.0

## module purge         # Kamiak is not similar to Aeolus. purge on its own cannot be loaded. 
                        # Either leave it out or add "module load StdEnv". Lets see if this works. (Feb 22.)
## module load StdEnv

module load gcc/7.3.0
# module load python3/3.5.0
# module load python3/3.7.0
module load anaconda3

# pip install shutup
# pip install dtaidistance
# pip install tslearn

# ----------------------------------------------------------------
# Gathering useful information
# ----------------------------------------------------------------
echo "--------- environment ---------"
env | grep PBS

echo "--------- where am i  ---------"
echo WORKDIR: ${PBS_O_WORKDIR}
echo HOMEDIR: ${PBS_O_HOME}

echo Running time on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo "--------- continue on ---------"
# Split index passed to your script
### SPLIT_NUM="${SLURM_ARRAY_TASK_ID}"
# ----------------------------------------------------------------
# Run python code for matrix
# ----------------------------------------------------------------

python /home/h.noorazar/rangeland_bio/01_rolling_ACF1/weather_ACF1_rolling.py window_size y_


echo Time is `date`
echo
echo "----- DONE -----"
echo

exit 0
