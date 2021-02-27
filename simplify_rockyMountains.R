############      packages      #############

library(data.table)
library(rgdal)
library(dplyr)
library(sp)
library(sf)


data_dir <- "/Users/hn/Documents/01_research_data/Rocky_Mountain/"
rocky <- readOGR(paste0(data_dir, 
                        "/Rocky_Mountain.shp"),
                        layer = "Rocky_Mountain", 
                        GDAL1_integer64_policy = TRUE)

simple_Rocky_Mountain <- rmapshaper::ms_simplify(rocky)


writeOGR(obj=simple_Rocky_Mountain, 
         dsn = paste0("/Users/hn/Documents/01_research_data/simple_Rocky_Mountain/"), 
         layer = "simple_Rocky_Mountain/", 
         driver = "ESRI Shapefile")