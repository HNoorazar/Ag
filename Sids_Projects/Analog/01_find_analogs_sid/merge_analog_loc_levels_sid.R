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

the_dir <- dir(in_dir, pattern = model_ty)

# remove filenames that aren't of model of interest

NN_dist_tb_list <- the_dir[grep(pattern = "NN_dist_tb_", x = the_dir)]
NN_loc_year_tb_list <- the_dir[grep(pattern = "NN_loc_year_tb_", x = the_dir)]
NN_sigma_tb_list <- the_dir[grep(pattern = "NN_sigma_tb_", x = the_dir)]

time_periods <- c("2026_2050", "2051_2075", "2076_2095")

start_time <- Sys.time()

for (time_p in time_periods){
  NN_dist_tb_list <- NN_dist_tb_list[grep(pattern = time_p, x = NN_dist_tb_list)]
  NN_loc_year_tb_list <- NN_loc_year_tb_list[grep(pattern = time_p, x = NN_loc_year_tb_list)]
  NN_sigma_tb_list <- NN_sigma_tb_list[grep(pattern = time_p, x = NN_sigma_tb_list)]

  NN_dist_tb <- data.table()
  NN_loc_year_tb <- data.table()
  NN_sigma_tb <- data.table()

  for (counter in 1:(length(NN_sigma_tb_list))){
    NN_dist_tb_2026_2050 <- rbind(NN_dist_tb_2026_2050, data.table(readRDS(NN_dist_tb_list_2026_2050[counter])))
    NN_loc_year_tb_2026_2050 <- rbind(NN_loc_year_tb_2026_2050,data.table(readRDS(NN_loc_year_tb_2026_2050[counter])))
    NN_sigma_tb_2026_2050 <- rbind(NN_sigma_tb_2026_2050, data.table(readRDS(NN_loc_year_tb_2026_2050[counter])))
  }
  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_",  model_ty, "_", time_p,  ".rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", model_ty, "_", time_p,  ".rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", model_ty, "_", time_p,  ".rds"))
}

# break the fuck into three time periods

# NN_dist_tb_list_2026_2050 <- NN_dist_tb_list[grep(pattern = "2026_2050", x = NN_dist_tb_list)]
# NN_loc_year_tb_list_2026_2050 <- NN_loc_year_tb_list[grep(pattern = "2026_2050", x = NN_loc_year_tb_list)]
# NN_sigma_tb_list_2026_2050 <- NN_sigma_tb_list[grep(pattern = "2026_2050", x = NN_sigma_tb_list)]


# NN_dist_tb_list_2051_2075 <- NN_dist_tb_list[grep(pattern = "2051_2075", x = NN_dist_tb_list)]
# NN_loc_year_tb_list_2051_2075 <- NN_loc_year_tb_list[grep(pattern = "2051_2075", x = NN_loc_year_tb_list)]
# NN_sigma_tb_list_2051_2075 <- NN_sigma_tb_list[grep(pattern = "2051_2075", x = NN_sigma_tb_list)]


# NN_dist_tb_list_2076_2095 <- NN_dist_tb_list[grep(pattern = "2076_2095", x = NN_dist_tb_list)]
# NN_loc_year_tb_list_2076_2095 <- NN_loc_year_tb_list[grep(pattern = "2076_2095", x = NN_loc_year_tb_list)]
# NN_sigma_tb_list_2076_2095 <- NN_sigma_tb_list[grep(pattern = "2076_2095", x = NN_sigma_tb_list)]


# NN_dist_tb_2026_2050 <- data.table()
# NN_loc_year_tb_2026_2050 <- data.table()
# NN_sigma_tb_2026_2050 <- data.table()

# NN_dist_tb_2051_2075 <- data.table()
# NN_loc_year_tb_2051_2075 <- data.table()
# NN_sigma_tb_2051_2075 <- data.table()

# NN_dist_tb_2076_2095 <- data.table()
# NN_loc_year_tb_2076_2095 <- data.table()
# NN_sigma_tb_2076_2095 <- data.table()

end_time <- Sys.time()
print( end_time - start_time)



