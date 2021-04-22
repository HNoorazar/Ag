rm(list=ls())
library(data.table)
library(ggplot2)
library(dplyr)
options(digits=9)
options(digit=9)


binary_core <- "/Users/hn/Documents/00_GitHub/Ag/read_binary_core/read_binary_core.R"
source(binary_core)

########################################
########
########   directories
########
########################################
JulienOutput_dir_base <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/is_output_of_step_3/"
param_dir <- "/Users/hn/Documents/01_research_data/cold_hardiness/hossein_locations/"

########################################

locations <- data.table(read.csv(paste0(param_dir, "limited_locations_CH.csv")))
locations$location <- paste0(locations$lat, "_", locations$long)

grape_varieties <- dir(path = JulienOutput_dir_base)
models <- dir(paste0(JulienOutput_dir_base, grape_varieties[1]))
emissions <- c("rcp45", "rcp85")

#####
#####
#####
print (grape_varieties)

for (a_variety in grape_varieties){
  CH_data <- data.table()
  for (a_model in models){
    if (a_model == "historical"){
      for (a_location in locations$location){
         file_pathName <- paste0(JulienOutput_dir_base, a_variety, "/", a_model, "/", "data_", a_location, ".csv.gz")
         curr_ch <- data.table::fread(file_pathName)

         curr_ch$location <- a_location
         curr_ch$model <- "Observed"
         curr_ch$emission <- "Observed"

         CH_data <- rbind(CH_data, curr_ch)
        }

      } else{
      for (emission in emissions){
        for (a_location in locations$location){
           file_pathName <- paste0(JulienOutput_dir_base, a_variety, "/", a_model, "/", emission, "/data_", a_location, ".csv.gz")
           curr_ch <- data.table::fread(file_pathName)

           curr_ch$location <- a_location
           curr_ch$model <- a_model
           curr_ch$emission <- emission

           CH_data <- rbind(CH_data, curr_ch)
        }
      }
    }
  }

  # write.csv(CH_data,   file=paste0(JulienOutput_dir_base, "/all_", a_variety, ".csv"),    row.names=F) 
  fwrite(CH_data,      file=paste0(JulienOutput_dir_base, "/all_", a_variety, ".csv.gz"), row.names=F, compress="gzip")
}

