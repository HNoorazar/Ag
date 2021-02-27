# Bloom Pruet
###  Packages  ####
library(shiny)
library(cowplot) # , lib.loc = "r_lib"
library(ggplot2) # , lib.loc = "r_lib"
library(leaflet) # , lib.loc = "r_lib"
library(zoo) # , lib.loc = "r_lib"
library(shinyBS)
library(shinydashboard)
library(purrr)
library(tidyr)
library(dplyr)
library(tibble)
library(stringr)
library(lubridate)
library(rgdal)
library(Cairo)
# library(readr, lib.loc="/home/hnoorazar/R/x86_64-redhat-linux-gnu-library/3.3") # , lib.loc = "r_lib"
options(shiny.usecairo=TRUE)

###  User Defined Functions  ###
source("functions/pruet_core.R")
# source("functions/readRDS.R") ## read RDS data files
# source("functions/wsu_colors.R") ## color functions
# source("functions/storm_plots.R") ## function to plot
# source("functions/runoff_plots.R") ## function to plot
# source("functions/precip_plots.R") ## function to plot
# source("functions/water_year_tools.R") ## Water_year
# source("functions/probability_plots.R") ## Plots
# source("functions/print_prob_plots.R") ## Print Plots
# source("functions/plot_dry_days.R") ## Print Dry Days Plots

### file names and lat, longs  ###
# hydro_map_df <- readRDS("/data/pruett/surface/hydro_spatial.RDS") %>% 
#                 mutate(max_combined = as.numeric(max_combined))

###########################
###                    ####
###    Import Spatial  ####
###                    ####
###########################
###
### Server
###
skagit <- readOGR("geo/Skagit.geo.json", "OGRGeoJSON") # Skagit County
snohomish <- readOGR("geo/Snohomish.geo.json", "OGRGeoJSON") # Snohomish County
whatcom <- readOGR("geo/Whatcom.geo.json", "OGRGeoJSON") # Whatcom County

# Desktop
# skagit <- readOGR("geo/Skagit.geo.json") # Skagit County
# snohomish <- readOGR("geo/Snohomish.geo.json") # Snohomish County
# whatcom <- readOGR("geo/Whatcom.geo.json") # Whatcom County

# Bind all counties ####
counties <- rbind(skagit, snohomish, whatcom, makeUniqueIDs = TRUE)
#################
#               #
#   Map info    #
#               #
#################
spatial_precip <- readRDS("/data/pruett/precip/spatial_prob_time.rds") %>% 
                  group_by (file_name, lat, lng)

# Global part
data_dir <- "/data/hnoorazar/bloom_thresh_frost/just_CM_locs/"

bloom_f_name <- "cm_loc_fullbloom_50percent_day.rds"
frost_f_name <- "cm_loc_frost.rds"
CP_f_name <- "cm_loc_sept_summary_comp.rds"

bloom_dt <- readRDS(paste0(data_dir, bloom_f_name)) %>% 
            group_by(location, lat, long)

frost_dt <- readRDS(paste0(data_dir, frost_f_name)) %>% 
            group_by(location, lat, long)

thresh_dt <- readRDS(paste0(data_dir, CP_f_name)) %>% 
            group_by(location, lat, long)

spatial_bcf <- readRDS(paste0(data_dir, "cm_spatial_bcf.rds")) %>% 
               group_by(location, lat, long)


# spatial_surface <- readRDS("/data/pruett/spatial_prob_surface.rds") %>% 
#                    as_tibble() %>% 
#                    rename(file_name = files) %>% 
#                    mutate(file_name = as.character(file_name)) %>% 
#                    group_by (file_name, lat, lng) %>% 
#                    mutate(exceedance_val = case_when(exceedance == "prob_80" ~ 0.2,
#                                                      exceedance == "prob_90" ~ 0.1,
#                                                      exceedance == "prob_95" ~ 0.05),
#                           prob_median = prob_median - exceedance_val)

# spatial_dry_days <- readRDS("/data/pruett/dry_days/spatial_prob.rds") %>% 
#                     mutate(exceedance_val = case_when(exceedance == "prob_80" ~ 0.2,
#                                                       exceedance == "prob_90" ~ 0.1,
#                                                       exceedance == "prob_95" ~ 0.05),
#                            prob_median = prob_median - exceedance_val)




