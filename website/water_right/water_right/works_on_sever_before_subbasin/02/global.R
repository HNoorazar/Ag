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
#####################################################
#####
#####       Read Shape files
#####
######################################################
wtr_right_dir <- "/data/hnoorazar/water_right/"

shapefile_dir <- paste0(wtr_right_dir, "shapefiles/")
shapefile_dir <- paste0(wtr_right_dir, "simple_shapefiles/")

data_dir <- paste0(wtr_right_dir, "data/")

# shapefile_dir <- paste0("/Users/hn/Desktop/", 
#                         "Desktop/Ag/check_point/", 
#                         "water_right/shapefiles/")

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
start_time <- Sys.time()
all_streams_sp <- rgdal::readOGR(dsn=path.expand(
                                          paste0(shapefile_dir, 
                                                 "all_streams/")),
                                layer = "all_streams")
print (Sys.time() - start_time)
##################
##
## all basins
##
##################

all_basins_sp <- rgdal::readOGR(dsn=path.expand(
                                        paste0(shapefile_dir, 
                                              "all_basins/")),
                                layer = "all_basins")
##################
##
## all subbasins
##
##################
all_subbasins_sp <- rgdal::readOGR(dsn=path.expand(
                                          paste0(shapefile_dir, 
                                                 "all_subbasins/")),
                                layer = "all_subbasins")

#####################################################
#####
#####       Water Right data
#####
######################################################

spatial_wtr_right <- readRDS(paste0(data_dir,
                            "water_right_attributes.rds")) %>% 
                     data.table()

spatial_wtr_right <- na.omit(spatial_wtr_right, cols=c("subbasin"))

spatial_wtr_right$colorr <- "#ffff00"
spatial_wtr_right <- data.table(spatial_wtr_right)

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

Upper_Yakima_center <- c(46.98, -120.6)
Lower_Yakima_center <- c(46.47, -120.349)
Naches_center <- c(46.69, -120.73)
Wenatchee_center <- c(47.59, -120.58)
Methow_center <- c(48.32, -120.13)
Okanogan_center <- c(48.53, -119.56)
Walla_Walla_center <- c(46.08, -118.34)

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
# base_map <- leaflet() %>%
#             addTiles(urlTemplate = paste0("http://server.", 
#                                           "arcgisonline.com/", 
#                                           "ArcGIS/rest/services", 
#                                           "/World_Imagery/", 
#                                           "MapServer", 
#                                           "/tile/{z}/{y}/{x}"),

#                   attribution = paste0('Maps by ', 
#                                        '<a href="http://',
#                                        'www.mapbox.com/">', 
#                                        'Mapbox</a>'),
#                   layerId = "Satellite",
#                   options= providerTileOptions(opacity = 0.9))

build_map <- function(data_dt, sub_bas){
  if (is.null(sub_bas)){
     map <- leaflet() %>%
            addTiles(urlTemplate = paste0("http://server.", 
                                          "arcgisonline.com/", 
                                          "ArcGIS/rest/services", 
                                          "/World_Imagery/", 
                                          "MapServer", 
                                          "/tile/{z}/{y}/{x}"),

                       attribution = paste0('Maps by ', 
                                          '<a href="http://',
                                          'www.mapbox.com/">', 
                                          'Mapbox</a>'),
                       layerId = "Satellite",
                       options= providerTileOptions(opacity = 0.9)) %>%
                       setView(lat = 46, lng =-121, zoom = 5)

    } else {
      data_dt <- data_dt %>%
                 filter(subbasin %in% sub_bas) %>%
                 data.table()

      if (unique(data_dt$WRIA_NM) == "Walla Walla"){
        # Walla_Walla_center <- c(46.08, -118.34)
          mean_lat <- 46.08
          mean_long <- -118.34

        } else if (unique(data_dt$WRIA_NM) == "Upper Yakima"){
          # Upper_Yakima_center <- c(46.98, -120.6)
          mean_lat <- 46.98 
          mean_long <- -120.6
        
        } else if (unique(data_dt$WRIA_NM) == "Lower Yakima"){
          # Lower_Yakima_center <- c(46.47, -120.349)
          mean_lat <- 46.47
          mean_long <- -120.349
        
        } else if (unique(data_dt$WRIA_NM) == "Naches"){
          # Naches_center <- c(46.69, -120.73)
          mean_lat <- 46.69
          mean_long <- -120.73
        
        } else if (unique(data_dt$WRIA_NM) == "Wenatchee"){
          # Wenatchee_center <- c(47.59, -120.58)
          mean_lat <- 47.59
          mean_long <- -120.58
        
        } else if (unique(data_dt$WRIA_NM) == "Methow"){
          # Methow_center <- c(48.32, -120.13)
          mean_lat <- 48.32
          mean_long <- -120.13
        
        } else if (unique(data_dt$WRIA_NM) == "Okanogan"){
          # Okanogan_center <- c(48.53, -119.56)
          mean_lat <- 48.53
          mean_long <- -119.56
      }
      # mean_lat <- mean(data_dt$lat) 
      # mean_long <- mean(data_dt$long)
      base_map <- leaflet() %>%
                  addTiles(urlTemplate = paste0("http://server.", 
                                                "arcgisonline.com/", 
                                                "ArcGIS/rest/services", 
                                                "/World_Imagery/", 
                                                "MapServer", 
                                                "/tile/{z}/{y}/{x}"),

                  attribution = paste0('Maps by ', 
                                       '<a href="http://',
                                       'www.mapbox.com/">', 
                                       'Mapbox</a>'),
                  layerId = "Satellite",
                  options= providerTileOptions(opacity = 0.9))

      map <- base_map %>% 
             addTiles(urlTemplate = paste0("http://server.", 
                                           "arcgisonline.com/", 
                                           "ArcGIS/rest/services", 
                                           "/World_Imagery/", 
                                           "MapServer", 
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

      # map <- leaflet() %>%
      #        addTiles(urlTemplate = paste0("http://server.", 
      #                                      "arcgisonline.com/", 
      #                                      "ArcGIS/rest/services", 
      #                                      "/World_Imagery/", 
      #                                      "MapServer", 
      #                                      "/tile/{z}/{y}/{x}"),

      #             attribution = paste0('Maps by ', 
      #                                  '<a href="http://',
      #                                  'www.mapbox.com/">', 
      #                                  'Mapbox</a>'),
      #             layerId = "Satellite",
      #             options= providerTileOptions(opacity = 0.9)) %>%
      #        setView(lat = mean_lat, lng = mean_long, zoom = 7) %>%
      #        addCircleMarkers(data = data_dt, 
      #                         lng = ~ long, lat = ~lat,
      #                         label = ~ popup,
      #                         # layerId = ~ location,
      #                         radius = 3,
      #                         color = ~ colorr,
      #                         stroke  = FALSE,
      #                         fillOpacity = .95 
      #                          )
      
      curr_basins_sp <- all_basins_sp[all_basins_sp$WRIA_NM %in% unique(data_dt$WRIA_NM), ]
      
      for(ii in 1:length(curr_basins_sp@polygons)) {
          map <- addPolygons(map = map, 
                             data = curr_basins_sp, 
                             lng = ~curr_basins_sp@polygons[[ii]]@Polygons[[1]]@coords[, 1], 
                             lat = ~curr_basins_sp@polygons[[ii]]@Polygons[[1]]@coords[, 2],
                             fill = F, 
                             weight = 7, 
                             color = "red", 
                             group ="Outline")
        }

      curr_subbasins_sp <- all_subbasins_sp[all_subbasins_sp$Subbasin %in% sub_bas, ]
      
      if (length(curr_subbasins_sp@polygons) > 0 ){
        for(ii in 1:length(curr_subbasins_sp@polygons)) {
            map <- addPolygons(map = map, 
                               data = curr_subbasins_sp, 
                               lng = ~curr_subbasins_sp@polygons[[ii]]@Polygons[[1]]@coords[, 1], 
                               lat = ~curr_subbasins_sp@polygons[[ii]]@Polygons[[1]]@coords[, 2],
                               fill = F, 
                               weight = 1, 
                               color = "yellow", 
                               group ="Outline")
        }
      }
      
      print (paste0("fuck me ", unique(data_dt$WRIA_NM)))
      # curr_stream <- all_streams_sp[all_streams_sp$WRIA %in% unique(data_dt$WRIA_NM), ]

      # for(ii in 1:length(curr_stream@lines)) {
      #    map <- addPolylines(map = base_map, 
      #                   data = curr_stream, 
      #                   lng = ~ curr_stream@lines[[ii]]@Lines[[1]]@coords[, 1], 
      #                   lat = ~ curr_stream@lines[[ii]]@Lines[[1]]@coords[, 2],
      #                   # fill = F, 
      #                   stroke = TRUE,
      #                   fillOpacity = 0.5, 
      #                   smoothFactor = 0.5, 
      #                   # layerId = "way",
      #                   weight = 2, 
      #                   color = "blue", 
      #                   group ="rivers")
      # }

      
      # for(ii in 1:length(all_streams_sp@lines)) {
      #     map <- addPolylines(map = map, 
      #                         data = all_streams_sp, 
      #                         lng = ~ all_streams_sp@lines[[ii]]@Lines[[1]]@coords[, 1], 
      #                         lat = ~ all_streams_sp@lines[[ii]]@Lines[[1]]@coords[, 2],
      #                         # fill = F, 
      #                         stroke = TRUE,
      #                         fillOpacity = 0.5, 
      #                         smoothFactor = 0.5, 
      #                         # layerId = "way",
      #                         weight = 2, 
      #                         color = "blue", 
      #                         group ="rivers")
      # }

    }
  

  

  return(map)
}
