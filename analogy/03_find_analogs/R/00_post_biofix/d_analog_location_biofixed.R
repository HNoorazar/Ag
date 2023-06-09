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

################################################################################
################################################################################
# 
#                   Terminal arguments and parameters
# 
################################################################################
args = commandArgs(trailingOnly=TRUE)

precip = args[1]   # w_precip # no_recip
model_ty = args[2]
emission = args[3] # rcp45 rcp85
int_file = args[4]
n_nghs = 47841

print (paste0("precip ", precip))
print (paste0("model_ty ", model_ty))
print (paste0("emission ", emission))
print (paste0("int_file ", int_file))
print (paste0("n_nghs ", n_nghs))
################################################################################
# 
#                     set up proper directories
# 
################################################################################

main_dir <- "/data/hydro/users/Hossein/analog/"

main_us_dir <- paste0(main_dir, "02_features_post_biofix/")
main_local_dir <- file.path(main_us_dir, "future_broken/")
int_local_dir <- file.path(main_local_dir, emission, model_ty, "/")

main_out <- file.path(main_dir, "03_analogs/biofixed/location_level/")
out_dir <- paste0(main_out, precip, "_", emission)

print (paste0("out_dir is : ",  out_dir))
if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }

################################################################################
# 
#                   Read Data
# 
################################################################################

all_dt_usa <- data.table(readRDS(paste0(main_us_dir, "hist_CDD_precip.rds")))
local_dt <- data.table(readRDS(paste0(int_local_dir, int_file)))

################################################################################
start_time <- Sys.time()

# run the model for separate time frames
time_frames = c("2026_2050", "2051_2075", "2076_2095") #
non_numeric_cols <- c("location", "year", "model")

if (precip == "w_precip") {
  numeric_col = c("yearly_precip", "CumDDinF_Aug23")
  } else {
  numeric_col = c("CumDDinF_Aug23")
}

print (numeric_col)
for (time in time_frames){
  if (time == time_frames[1]){
    local_dt_time <- local_dt %>% filter(year >= 2026 & year <= 2050) %>% data.table()
    } else if (time == time_frames[2]) {
      local_dt_time <- local_dt %>% filter(year >= 2051 & year <= 2075) %>% data.table()
    } else if (time == time_frames[3]) {
     local_dt_time <- local_dt %>% filter(year >= 2076 & year <= 2095) %>% data.table()
  }
                                
  information = find_NN_info_biofix(ICV = all_dt_usa, 
                                    historical_dt = all_dt_usa, 
                                    future_dt = local_dt_time, 
                                    n_neighbors = n_nghs, 
                                    numeric_cols = numeric_col, 
                                    non_numeric_cols = non_numeric_cols)

  NN_dist_tb = information[[1]]
  NN_loc_year_tb = information[[2]]
  NN_sigma_tb = information[[3]]

  lat_long <- substr(int_file, 6, 23)

  saveRDS(NN_dist_tb, paste0(out_dir, "/NN_dist_tb_", lat_long, "_", model_ty, "_", time, ".rds"))
  saveRDS(NN_loc_year_tb, paste0(out_dir, "/NN_loc_year_tb_", lat_long, "_", model_ty, "_", time, ".rds"))
  saveRDS(NN_sigma_tb, paste0(out_dir, "/NN_sigma_tb_", lat_long, "_", model_ty, "_", time, ".rds"))
}

end_time <- Sys.time()
print( end_time - start_time)

