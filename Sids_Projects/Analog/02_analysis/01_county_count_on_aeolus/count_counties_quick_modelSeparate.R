
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

sigma_bd = 2
######################################################################
##                                                                  ##
##                     set up directories                           ##
##                                                                  ##
######################################################################

dt_dir <- "/Users/hn/Documents/01_research_data/Sids_Projects/merged_analog_Aeolus/"
dt_dir <- "/data/hydro/users/Hossein/sids_projects/analog/01_analogs_perLocModel/"

out_dir <- file.path("/data/hydro/users/Hossein/sids_projects/analog/03_analog_count/modelsSeparate/")


if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }
print ("__________________________")
print ("out_dir:")
print (out_dir)
print ("__________________________")

######################################################################
##                                                                  ##
##                          read files                              ##
##                                                                  ##
######################################################################
st_time <- Sys.time()

list_of_files <- dir(dt_dir)
# remove filenames that aren't data
list_of_files <- list_of_files[grep(pattern = "NN_loc_year_tb_", x = list_of_files)]

for (a_file in list_of_files){
    #
    # break the damn file name and extract emission, model name, time period
    #
    a_file_split <- str_split(string=a_file, pattern = "_")[[1]]
    len_a_file <- length(a_file_split)

    emission <- substr(x=a_file_split[len_a_file], start=1, stop=5)
    time_period <- paste(a_file_split[len_a_file-2], a_file_split[len_a_file-1], sep="_")
    model_name <- paste(a_file_split[5: (len_a_file-3)], collapse="_")

    ############################################################################## end of file name break
    
    NNs_name <- paste0(dt_dir, a_file)
    sigma_name <- paste0(dt_dir, gsub("NN_loc_year_tb", "NN_sigma_tb", a_file))

    NNs <- data.table(readRDS(NNs_name))
    sigmas <- data.table(readRDS(sigma_name))

    print ("line 70 of count_counties_quick.R")
    print (paste0("dim(NNs) is ", dim(NNs)))
    print (paste0("dim(sigmas) is ", dim(sigmas)))
    
    a_model_analog_output <- count_analogs_counties_quick(NNs, sigmas, sigma_bd=sigma_bd)
    a_model_novel_output <- count_novel_quick(NNs, sigmas, novel_bd=sigma_bd)
    
    saveRDS(a_model_analog_output,paste0(out_dir, paste("analog", model_name, time_period, emission, sep="_"),  ".rds"))
    saveRDS(a_model_novel_output, paste0(out_dir, paste("novel",  model_name, time_period, emission, sep="_"), ".rds"))
}

print (Sys.time() - st_time)


