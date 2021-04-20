
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(lubridate)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)

source_path = "/Users/hn/Documents/00_GitHub/Ag/Sids_Projects/Analog/core_analog_sid.R"
source_path = "/home/hnoorazar/sid/analog_pairwise_2021/core_analog_sid.R"
source(source_path)

options(digit=9)
options(digits=9)

# Check current folder
print("does this look right?")
getwd()
start_time <- Sys.time()


######################################################################
##                                                                  ##
##                     Terminal arguments                           ##
##                                                                  ##
######################################################################

args = commandArgs(trailingOnly=TRUE)
# carbon_type = args[1] # rcp45 or rcp85
# sigma_bd = args[3]   # sigma cut off for sigma dissimilarity 1 or 2 or 3 or what?
# all_model_names <- args[4]
# time_periods <- args[5]

sigma_bd = 2
time_periods <- c("2030_2070", "2070_2100")
carbon_types <- c("rcp45", "rcp85")
######################################################################
##                                                                  ##
##                     set up directories                           ##
##                                                                  ##
######################################################################

main_in <- "/Users/hn/Documents/01_research_data/Sids_Projects/merged_analog_Aeolus/"
main_in <- "/data/hydro/users/Hossein/sids_projects/analog/02_analogs_merged/"
dt_dir <- main_in

main_out <- file.path("/data/hydro/users/Hossein/sids_projects/analog/03_analog_count/modelsDissolved/")

out_dir <- main_out

if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }
print ("__________________________")
print ("out_dir:")
print (out_dir)
print ("__________________________")
param_dir <- "/home/hnoorazar/analog_codes/parameters/"

######################################################################
##                                                                  ##
##                          read files                              ##
##                                                                  ##
######################################################################
st_time <- Sys.time()

for (emission in carbon_types){
  for (time in time_periods){
    print (paste0(time, " - ", emission))
    
    NNs_name <- paste0(dt_dir, paste("/NN_loc_year_tb", time, emission, sep="_"), ".rds")
    sigma_name <- paste0(dt_dir, paste("/NN_sigma_tb", time, emission, sep="_"),  ".rds")

    NNs <- data.table(readRDS(NNs_name))
    sigmas <- data.table(readRDS(sigma_name))

    print ("line 81 of count_counties_quick.R")
    print (paste0("dim(NNs) is ", dim(NNs)))
    print (paste0("dim(sigmas) is ", dim(sigmas)))
    
    a_model_analog_output <- count_analogs_counties_quick(NNs, sigmas, sigma_bd=sigma_bd)
    print ("line 80")
    a_model_novel_output <- count_novel_quick(NNs, sigmas, novel_bd=sigma_bd)
    print ("line 82")

    saveRDS(a_model_analog_output,paste0(out_dir, paste("analog", time, emission, sep="_"),  ".rds"))
    saveRDS(a_model_novel_output, paste0(out_dir, paste("novel",  time, emission, sep="_"), ".rds"))
  }
}

print (Sys.time() - st_time)


