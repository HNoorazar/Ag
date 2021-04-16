.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(raster)
library(FNN)
library(RColorBrewer)
library(colorRamps)
library(EnvStats, lib.loc = "~/.local/lib/R3.5.1")

source_path = "/home/hnoorazar/analog_codes/core_analog.R"
source(source_path)
options(digit=9)
options(digits=9)

####################################################################################################

#                Terminal arguments and parameters

####################################################################################################
args = commandArgs(trailingOnly=TRUE)
emission = args[1] # rcp45 rcp85

main_in <- file.path("/data/hydro/users/Hossein/sids_projects/analog/")
in_dir <- paste0(main_in, "01_analogs_perLocModel/")
out_dir <- paste0(main_in, "02_analogs_merged/")
if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }

setwd(in_dir)
getwd()

the_dir <- dir(in_dir, pattern = ".rds")

# remove filenames that aren't of model of interest

NN_dist_tb_list <- the_dir[grep(pattern = "NN_dist_tb_", x = the_dir)]
NN_loc_year_tb_list <- the_dir[grep(pattern = "NN_loc_year_tb_", x = the_dir)]
NN_sigma_tb_list <- the_dir[grep(pattern = "NN_sigma_tb_", x = the_dir)]

time_periods <- c("2030_2070", "2070_2100")

time_emissions <- c("2030_2070_rcp45", "2070_2100_rcp45",
                    "2030_2070_rcp85", "2070_2100_rcp85")

start_time <- Sys.time()

for (time_emission in time_emissions){
  NN_dist_tb_list_int     <- NN_dist_tb_list    [grep(pattern = time_emission, x = NN_dist_tb_list)]
  NN_loc_year_tb_list_int <- NN_loc_year_tb_list[grep(pattern = time_emission, x = NN_loc_year_tb_list)]
  NN_sigma_tb_list_int    <- NN_sigma_tb_list   [grep(pattern = time_emission, x = NN_sigma_tb_list)]
  NN_dist_tb <- data.table()
  NN_loc_year_tb <- data.table()
  NN_sigma_tb <- data.table()
  
  print(length(NN_sigma_tb_list_int))

  for (counter in 1:(length(NN_sigma_tb_list_int))){
    NN_dist_tb <- rbind(NN_dist_tb, data.table(readRDS(NN_dist_tb_list_int[counter])))
    NN_loc_year_tb <- rbind(NN_loc_year_tb,data.table(readRDS(NN_loc_year_tb_list_int[counter])))
    NN_sigma_tb <- rbind(NN_sigma_tb, data.table(readRDS(NN_sigma_tb_list_int[counter])))
  }
  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_",  time_emission, ".rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", time_emission, ".rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", time_emission, ".rds"))
}

print( Sys.time() - start_time)



