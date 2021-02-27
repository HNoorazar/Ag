###  Packages  ####
library(shiny)
library(shinyBS)
library(shinydashboard)
library(purrr)
library(readr)
library(tidyr)
library(dplyr)
library(tibble)
library(ggplot2)
library(cowplot)
library(leaflet)
library(stringr)
library(lubridate)
library(rgdal)
library(zoo)
library(Cairo)
options(shiny.usecairo=TRUE)

###  User Defined Functions  ###
source("functions/read_RDS.R") ## read RDS data files
source("functions/wsu_colors.R") ## color functions
source("functions/storm_plots.R") ## function to plot
source("functions/runoff_plots.R") ## function to plot
source("functions/precip_plots.R") ## function to plot
source("functions/water_year_tools.R") ## Water_year
source("functions/probability_plots.R") ## Plots
source("functions/print_prob_plots.R") ## Print Plots
source("functions/plot_dry_days.R") ## Print Dry Days Plots

###  Data frame of file names and lat, longs  ###
hydro_map_df <- readRDS("data/pruett/surface/hydro_spatial.RDS") %>% 
  mutate(max_combined = as.numeric(max_combined))

### Import Spatial Data ####
# Server
# skagit <- readOGR("geo/Skagit.geo.json", "OGRGeoJSON") # Skagit County
# snohomish <- readOGR("geo/Snohomish.geo.json", "OGRGeoJSON") # Snohomish County
# whatcom <- readOGR("geo/Whatcom.geo.json", "OGRGeoJSON") # Whatcom County

# Desktop
skagit <- readOGR("geo/Skagit.geo.json") # Skagit County
snohomish <- readOGR("geo/Snohomish.geo.json") # Snohomish County
whatcom <- readOGR("geo/Whatcom.geo.json") # Whatcom County

# Bind all counties data ####
counties <- rbind(skagit, snohomish, whatcom, makeUniqueIDs = TRUE)

# Map Data ####
spatial_precip <- read_rds("data/pruett/precip/spatial_prob_time.rds") %>% 
  group_by (file_name, lat, lng)

spatial_surface <- read_rds("data/pruett/spatial_prob_surface.rds") %>% 
  as_tibble() %>% 
  rename(file_name = files) %>% 
  mutate(file_name = as.character(file_name)) %>% 
  group_by (file_name, lat, lng) %>% 
  mutate(exceedance_val = case_when(exceedance == "prob_80" ~ 0.2,
                                    exceedance == "prob_90" ~ 0.1,
                                    exceedance == "prob_95" ~ 0.05),
         prob_median = prob_median - exceedance_val)

spatial_dry_days <- read_rds("data/pruett/dry_days/spatial_prob.rds") %>% 
  mutate(exceedance_val = case_when(exceedance == "prob_80" ~ 0.2,
                                    exceedance == "prob_90" ~ 0.1,
                                    exceedance == "prob_95" ~ 0.05),
         prob_median = prob_median - exceedance_val)


