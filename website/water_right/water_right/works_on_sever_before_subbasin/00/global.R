# Water Rights

library(scales)
library(lattice)
# library(ggmap)
library(jsonlite)
library(raster)

library(data.table)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(maps)
library(rgdal)    # for readOGR and others
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(reshape2)
library(RColorBrewer)
# library(plotly)
# library(Hmisc)

#########################################################
# read county shapefile
# shapefile_dir <- "/data/hnoorazar/bloom/shape_files/tl_2017_us_county/"
# shapefile_dir <- "/data/hnoorazar/bloom/shape_files/tl_2017_us_county_simple/"
# counties <- rgdal::readOGR(dsn=path.expand(shapefile_dir), 
#                            layer = "tl_2017_us_county")

# # Extract just the three states OR: 41, WA:53, ID: 16
# counties <- counties[counties@data$STATEFP %in% c("16", "41", "53"), ]

# #
# # Compute states like so, to put border around states
# states <- aggregate(counties[, "STATEFP"], 
#                     by = list(ID = counties@data$STATEFP), 
#                     FUN = unique, dissolve = T)

# interest_counties <- c("16027", "53001", "53021", "53071",
#                        "41021", "53005", "53025", "53077", 
#                        "41027", "53007", "53037",  
#                        "41049", "53013", "53039", 
#                        "41059", "53017", "53047")

# counties <- counties[counties@data$GEOID %in% interest_counties, ]

wtr_right_dir <- "/data/hnoorazar/water_right/"
data_dir <- paste0(wtr_right_dir, "data/")

spatial_wtr_right <- readRDS(paste0(data_dir, 
                                    "water_right_attributes.rds"))
spatial_wtr_right$colorr <- "#ffff00"
######################################################

RdBu_reverse <- rev(brewer.pal(11, "RdBu"))

