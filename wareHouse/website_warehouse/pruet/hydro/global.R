###  Packages  ###
require(shiny)
require(shinydashboard)
require(purrr)
require(tidyr)
require(dplyr)
require(tibble)
require(ggplot2)
require(leaflet)
require(stringr)
require(lubridate)
require(rgdal)

###  User Defined Functions  ###
source("functions/read_RDS.R") ## read RDS data files
source("functions/multiplot.R") ## function to plot
source("functions/calc.R") ## calculates return intensity

###  Data frame of file names and lat, longs  ###
map_df <- readRDS("RDS/spatial.rds")

### Import Spatial Data ###
# Server
skagit <- readOGR("geo/Skagit.geo.json", "OGRGeoJSON") # Skagit County
snohomish <- readOGR("geo/Snohomish.geo.json", "OGRGeoJSON") # Snohomish County
whatcom <- readOGR("geo/Whatcom.geo.json", "OGRGeoJSON") # Whatcom County

# Desktop
# skagit <- readOGR("geo/Skagit.geo.json") # Skagit County
# snohomish <- readOGR("geo/Snohomish.geo.json") # Snohomish County
# whatcom <- readOGR("geo/Whatcom.geo.json") # Whatcom County

# Bind all counties data
counties <- rbind(skagit, snohomish, whatcom, makeUniqueIDs = TRUE)

