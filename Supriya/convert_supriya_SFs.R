rm(list=ls())
library(data.table)
library(rgdal)
library(sp)
library(dplyr)
library(ggplot2)
library(maps)
library(purrr)
library(scales) # includes pretty_breaks

##############################################################################################################
#
#    Directory setup
#
SF_dir <- paste0("/Users/hn/Documents/01_research_data/Supriya/Shapefiles_landsatcode_NDVIAnalysis/")

########################################
states <- map_data("state")
states_cluster <- subset(states, region %in% c("idaho"))


#####
##### IDb_Canyon
#####

IDb_CanyonCounty <- readOGR(paste0(SF_dir, "IDb_CanyonCounty/IDb_CanyonCounty.shp"),
                            layer = "IDb_CanyonCounty", 
                            GDAL1_integer64_policy = TRUE)

IDb_Canyon_map <- ggplot() +
                  geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
                  geom_polygon(data = IDb_CanyonCounty, aes(x=long, y=lat, group=group), fill = "grey47", color = "red")

plot_w <- 5
plot_h <- 5
ggsave(filename = paste0("map_IDb_Canyon_map.pdf"),
       plot = IDb_Canyon_map, 
       width=plot_w, height=plot_h, units = "in", limitsize = FALSE,
       dpi=400, device = "pdf",
       path=SF_dir)


writeOGR(obj = IDb_CanyonCounty, 
         dsn = paste0(SF_dir, "/new_IDb_Canyon/"), 
         layer="new_IDb_Canyon", 
         driver="ESRI Shapefile")


#####
##### IDb_Canyon_Grid_100m
#####

IDb_Canyon_Grid_100m <- readOGR(paste0(SF_dir, "IDb_Canyon_Grid_100m/IDb_Canyon_Grid_100m.shp"),
                            layer = "IDb_Canyon_Grid_100m", 
                            GDAL1_integer64_policy = TRUE)

IDb_Canyon_Grid_100m_map <- ggplot() +
                            geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
                            geom_polygon(data = IDb_Canyon_Grid_100m, aes(x=long, y=lat, group=group), 
                                         fill = "green", color = "red", size=0.1)

plot_w <- 1000
plot_h <- 1000
ggsave(filename = paste0("map_IDb_Canyon_Grid_100m_map.pdf"),
       plot = IDb_Canyon_Grid_100m_map, 
       width=plot_w, height=plot_h, units = "in", limitsize = FALSE,
       dpi=400, device = "pdf",
       path=SF_dir)


setnames(IDb_Canyon_Grid_100m@data, old=c("id"), new=c("grid_id"))
writeOGR(obj = IDb_Canyon_Grid_100m, 
         dsn = paste0(SF_dir, "/new_IDb_Canyon_Grid_100m/"), 
         layer="new_IDb_Canyon_Grid_100m", 
         driver="ESRI Shapefile")
#####
##### IDb_Canyon_Grid_1km
#####

IDb_Canyon_Grid_1km <- readOGR(paste0(SF_dir, "IDb_Canyon_Grid_1km/IDb_Canyon_Grid_1km.shp"),
                               layer = "IDb_Canyon_Grid_1km", 
                               GDAL1_integer64_policy = TRUE)

IDb_Canyon_Grid_1km_map <- ggplot() +
                           geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group), fill = "grey", color = "black") +
                           geom_polygon(data = IDb_Canyon_Grid_1km, aes(x=long, y=lat, group=group), fill = "yellow", color = "red")


plot_w <- 50
plot_h <- 50
ggsave(filename = paste0("map_IDb_Canyon_Grid_1km_map.pdf"),
       plot = IDb_Canyon_Grid_1km_map, 
       width=plot_w, height=plot_h, units = "in", limitsize = FALSE,
       dpi=400, device = "pdf",
       path=SF_dir)

setnames(IDb_Canyon_Grid_1km@data, old=c("id"), new=c("grid_id"))
writeOGR(obj = IDb_Canyon_Grid_1km, 
         dsn = paste0(SF_dir, "/new_IDb_Canyon_Grid_1km/"), 
         layer="new_IDb_Canyon_Grid_1km", 
         driver="ESRI Shapefile")



