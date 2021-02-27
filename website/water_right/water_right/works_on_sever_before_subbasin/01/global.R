# Water Rights

library(scales)
library(lattice)
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

############################################################
############################################################
#####
#####       Read Shape files
#####
############################################################
wtr_right_dir <- "/data/hnoorazar/water_right/"
shapefile_dir <- paste0(wtr_right_dir, "shapefiles/")
data_dir <- paste0(wtr_right_dir, "data/")


# shapefile_dir <- paste0("/Users/hn/Desktop/", 
#                         "Desktop/Ag/check_point", 
#                         "/water_right/simple_shapefiles/")

# data_dir <- paste0("/Users/hn/Desktop/Desktop/",
#                    "Ag/check_point/water_right/data/"
#                    )

##################
##
## all streams
##
##################
# all_streams_sp <- rgdal::readOGR(dsn=path.expand(
#                                           paste0(shapefile_dir, 
#                                                  "all_streams/")),
#                                 layer = "all_streams")
##################
##
## all basins
##
##################

# all_basins_sp <- rgdal::readOGR(dsn=path.expand(
#                                         paste0(shapefile_dir, 
#                                               "all_basins/")),
#                                 layer = "all_basins")

##################
##
## all subbasins
##
##################

# all_subbasins_sp <- rgdal::readOGR(dsn=path.expand(
#                                           paste0(shapefile_dir, 
#                                                  "all_subbasins/")),
#                                 layer = "all_subbasins")

############################################################
#####
#####       Water Right data
#####
############################################################

spatial_wtr_right <- readRDS(paste0(data_dir, 
                                    "water_right_attributes.rds"))
spatial_wtr_right$colorr <- "#ffff00"


# curr_spatial <- spatial_wtr_right
all_basins <- sort(unique(spatial_wtr_right$WRIA_NM))

subbasins <- c("Ahtanum Creek", 
               "Lmumu-Burbank",
               "Lower Yakima tributaries",
               "tributaries", 
               "Satus Creek",
               "Taneum-Manastash",
               "Toppenish Creek",
               "Wilson-Cherry")


########################################
######
######       Functions
######
########################################

###########################
######
######   construct map
######
###########################
# build_map <- function(data_dt, sub_bas){  
#   data_dt <- data_dt %>%
#              filter(subbasin %in% sub_bas) %>%
#              data.table()

#   mean_lat <- mean(data_dt$lat) 
#   mean_long <- mean(data_dt$long)

#   map <- leaflet() %>%
#          addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#                   attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
#                   layerId = "Satellite",
#                   options= providerTileOptions(opacity = 0.9)) %>%
#          setView(lat = mean_lat, lng = mean_long, zoom = 7) %>%
#          addCircleMarkers(data = data_dt, 
#                           lng = ~ long, lat = ~lat,
#                           label = ~ popup,
#                           # layerId = ~ location,
#                           radius = 3,
#                           color = ~ colorr,
#                           stroke  = FALSE,
#                           fillOpacity = .95 
#                            )
#   return(map)
# }

build_map <- function(data_dt, sub_bas){  
  data_dt <- data_dt %>%
             filter(subbasin %in% sub_bas) %>%
             data.table()

  mean_lat <- mean(data_dt$lat) 
  mean_long <- mean(data_dt$long)

  map <- leaflet() %>%
         addTiles(urlTemplate = paste0("http://server.", 
                                       "arcgisonline.com/", 
                                       "ArcGIS/rest/services", 
                                       "/World_Imagery/MapServer", 
                                       "/tile/{z}/{y}/{x}"),

                  attribution = paste0('Maps by ', 
                                       '<a href="http://',
                                       'www.mapbox.com/">', 
                                       'Mapbox</a>'),
                  layerId = "Satellite",
                  options= providerTileOptions(opacity = 0.9)) %>%
         setView(lat = mean_lat, lng = mean_long, zoom = 7) %>%
         addCircleMarkers(data = data_dt, 
                          lng = ~ long, lat = ~lat,
                          label = ~ popup,
                          # layerId = ~ location,
                          radius = 3,
                          color = ~ colorr,
                          stroke  = FALSE,
                          fillOpacity = .95 
                           )

  # for(ii in 1:length(all_basins_sp@polygons)) {
  #     map <- addPolygons(map = map, 
  #                        data = all_basins_sp, 
  #                        lng = ~all_basins_sp@polygons[[ii]]@Polygons[[1]]@coords[, 1], 
  #                        lat = ~all_basins_sp@polygons[[ii]]@Polygons[[1]]@coords[, 2],
  #                        fill = F, 
  #                        weight = 2, 
  #                        color = "red", 
  #                        group ="Outline")

  #   }

  # for(ii in 1:length(all_subbasins_sp@polygons)) {
  #     map <- addPolygons(map = map, 
  #                        data = all_subbasins_sp, 
  #                        lng = ~all_subbasins_sp@polygons[[ii]]@Polygons[[1]]@coords[, 1], 
  #                        lat = ~all_subbasins_sp@polygons[[ii]]@Polygons[[1]]@coords[, 2],
  #                        fill = F, 
  #                        weight = 2, 
  #                        color = "yellow", 
  #                        group ="Outline")

  #   }

  # for(ii in 1:length(all_streams_sp@lines)) {
  #     map <- addPolylines(map = map, 
  #                         data = all_streams_sp, 
  #                         lng = ~all_streams_sp@lines[[ii]]@Lines[[1]]@coords[, 1], 
  #                         lat = ~all_streams_sp@lines[[ii]]@Lines[[1]]@coords[, 2],
  #                         # fill = F, 
  #                         stroke = TRUE,
  #                         fillOpacity = 0.5, 
  #                         smoothFactor = 0.5, 
  #                         # layerId = "way",
  #                         weight = 2, 
  #                         color = "blue", 
  #                         group ="rivers")
  # }

  return(map)
}