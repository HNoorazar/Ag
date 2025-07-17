#!/bin/bash
##SBATCH --partition=kamiak
##SBATCH --constraint=cascadelake
#SBATCH --partition=rajagopalan,kamiak
#SBATCH --requeue
#SBATCH --job-name=weather_var_roll_window_size_y_ # Job Name
#SBATCH --time=0-1:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=8GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o
#SBATCH --output=/home/h.noorazar/rangeland_bio/01_rolling_variance/error/weather_variance_rolling_window_size_y_.o
#SBATCH  --error=/home/h.noorazar/rangeland_bio/01_rolling_variance/error/weather_variance_rolling_window_size_y_.e
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

# ----------------------------------------------------------------
# Run python code for matrix
# ----------------------------------------------------------------

python /home/h.noorazar/rangeland_bio/01_rolling_variance/weather_variance_rolling.py window_size y_


echo Time is `date`
echo
echo "----- DONE -----"
echo

exit 0
