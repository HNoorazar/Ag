#!/bin/bash
#SBATCH --job-name="rangeland climate"
##SBATCH --partition=stockle,kamiak,adam --account stockle
#SBATCH --partition=adam,kamiak --account adam
#SBATCH --time=5-24:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job
#SBATCH --ntasks-per-node=1     ### Nuber of tasks to be launched per Node
#SBATCH --cpus-per-task=1       ### Number of threads per task (OMP threads)
#SBATCH --mem=2000             ### Amount of memory in MB
#SBATCH--array=0-199%40
##SBATCH--array=0-2%2

module load anaconda3/20.02.0
simid=$SLURM_ARRAY_TASK_ID

in_gridid_lat_lon=/weka/data/lab/adam/mingliang.liu/Projects/rangeland/gridmet_sublist_p${simid}.txt
#sys.argv[1] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_file_list.csv"
in_gridmet_path=/weka/data/lab/adam/data/metdata/historical/UI_historical/VIC_Binary_CONUS_1979_to_2022_24thD/
#sys.argv[2] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_temp/"
out_climate_path=/scratch/user/mingliang.liu/20230214_155903/gridmet_monthly_indices/
#sys.argv[3] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_monthly_indices/"


python /weka/data/lab/adam/mingliang.liu/Projects/rangeland/Kamiak_generate_monthly_mean_gridmet_climate_indices.py ${in_gridid_lat_lon} ${in_gridmet_path} ${out_climate_path}
