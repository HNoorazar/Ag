#!/bin/bash
#SBATCH --job-name="rangeland climate"
##SBATCH --partition=stockle,kamiak,adam --account stockle
#SBATCH --partition=adam --account adam
#SBATCH --time=5-24:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job
#SBATCH --ntasks-per-node=1     ### Nuber of tasks to be launched per Node
#SBATCH --cpus-per-task=1       ### Number of threads per task (OMP threads)
#SBATCH --mem=100000             ### Amount of memory in MB
##SBATCH--array=0-119%40
##SBATCH--array=0-2%2

module load anaconda3/20.02.0

in_gridid_lat_lon=/weka/data/lab/adam/mingliang.liu/Projects/rangeland/gridmet_file_list.csv 
#sys.argv[1] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_file_list.csv"
#GRID_ID,lat,lon
in_county_gridmet_count_fraction=/weka/data/lab/adam/mingliang.liu/Projects/rangeland/county_gridmet_range_count_fraction.csv

#in_county_gridmet_count_fraction=/weka/data/lab/adam/mingliang.liu/Projects/rangeland/temp_subset.csv
#sys.argv[2] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/county_gridmet_range_count_fraction.csv"
#county,gridmet,count,county_all,fraction

in_gridmet_path=/scratch/user/mingliang.liu/20230214_155903/gridmet_monthly_indices/
#sys.argv[3] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/gridmet_monthly_indices/"

out_climate_indicex=/weka/data/lab/adam/mingliang.liu/Projects/rangeland/county_gridmet_mean_indices.csv 
#sys.argv[4] #"/home/liuming/mnt/hydronas3/Projects/Rangeland/county_gridmet_mean_indices.csv"

python /weka/data/lab/adam/mingliang.liu/Projects/rangeland/Kamiak_generate_mean_county_gridmet_climate_indices_from_monthly_gridmet.py ${in_gridid_lat_lon} ${in_county_gridmet_count_fraction} ${in_gridmet_path} ${out_climate_indicex}
