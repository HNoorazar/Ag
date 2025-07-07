#!/bin/bash

#SBATCH --partition=rajagopalan,cahnrs,cahnrs_bigmem,cahnrs_gpu,kamiak,stockle
#SBATCH --requeue
#SBATCH --job-name=SeasVGrid_outer # Job Name
#SBATCH --time=00-12:00:00    # Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=30GB 
#SBATCH --nodes=1            # Node count required for the job
#SBATCH --ntasks-per-node=1  # Number of tasks to be launched per Node
#SBATCH --ntasks=1           # Number of tasks per array job
#SBATCH --cpus-per-task=1    # Number of threads per task (OMP threads)
####SBATCH --array=0-30000

###SBATCH -k o

## Define path for output & error logs
#SBATCH --error=/home/h.noorazar/NASA/02_remove_outliers_n_jumps/00_intersect_remove_outliers/error/outer_post2008_e
#SBATCH --output=/home/h.noorazar/NASA/02_remove_outliers_n_jumps/00_intersect_remove_outliers/error/outer_post2008_o

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

cd /home/h.noorazar/NASA/02_remove_outliers_n_jumps/00_intersect_remove_outliers/

# ----------------------------------------------------------------
# Run python code for matrix
# ----------------------------------------------------------------

python3 ./00_remove_outliers_intersect_post2008_kamiak.py indeks


