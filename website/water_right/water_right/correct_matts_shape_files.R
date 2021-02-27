crs <- CRS("+proj=lcc 
           +lat_1=45.83333333333334 
           +lat_2=47.33333333333334 
           +lat_0=45.33333333333334 
           +lon_0=-120.5 +datum=WGS84")


########################################
########
########
########
shapefile_dir <- paste0("/Users/hn/Desktop/", 
                        "Desktop/Ag/check_point/", 
                        "water_right/shapefiles/")

simple_shapefile_dir <- paste0("/Users/hn/Desktop/", 
                               "Desktop/Ag/check_point/", 
                               "water_right/simple_shapefiles/")

if (dir.exists(file.path(shapefile_dir)) == F){
  dir.create(path=file.path(shapefile_dir), recursive=T)
}

if (dir.exists(file.path(simple_shapefile_dir)) == F){
  dir.create(path=file.path(simple_shapefile_dir), recursive=T)
}

########################################
########
########
########
all_basins_sp <- spTransform(all_basins_sp, 
                             CRS("+proj=longlat +datum=WGS84")
                             )
writeOGR(obj = all_basins_sp, 
         dsn = paste0(shapefile_dir, "all_basins/"), 
         layer="all_basins", 
         driver="ESRI Shapefile")

################################
########
########
########
all_streams_sp <- spTransform(all_streams_sp, 
                             CRS("+proj=longlat +datum=WGS84")
                             )

writeOGR(obj = all_streams_sp, 
         dsn = paste0(shapefile_dir, "all_streams/"), 
         layer="all_streams", 
         driver="ESRI Shapefile")


################################
########
########
########
all_subbasins_sp <- spTransform(all_subbasins_sp, 
                             CRS("+proj=longlat +datum=WGS84")
                             )

writeOGR(obj = all_subbasins_sp, 
         dsn = paste0(shapefile_dir, "all_subbasins/"), 
         layer="all_subbasins", 
         driver="ESRI Shapefile")

####################################
####
#### simplify
####

all_streams_sp <- rgdal::readOGR(dsn=path.expand(
                                          paste0(shapefile_dir, 
                                                 "all_streams/")),
                                layer = "all_streams")

all_basins_sp <- rgdal::readOGR(dsn=path.expand(
                                        paste0(shapefile_dir, 
                                              "all_basins/")),
                                layer = "all_basins")

all_subbasins_sp <- rgdal::readOGR(dsn=path.expand(
                                          paste0(shapefile_dir, 
                                                 "all_subbasins/")),
                                layer = "all_subbasins")
####
#### simplify
####
all_streams_sp <- rmapshaper::ms_simplify(all_streams_sp)
all_basins_sp <- rmapshaper::ms_simplify(all_basins_sp)
all_subbasins_sp <- rmapshaper::ms_simplify(all_subbasins_sp)

####
#### save streams
####
all_streams_dir <- paste0(simple_shapefile_dir,
                          "all_streams/")

if (dir.exists(file.path(all_streams_dir)) == F){
  dir.create(path=file.path(all_streams_dir), recursive=T)
}

writeOGR(obj = all_streams_sp, 
         dsn = all_streams_dir,
         layer="all_streams", 
         driver="ESRI Shapefile")

####
#### save basins
####
all_basins_dir <- paste0(simple_shapefile_dir,
                         "all_basins/")

if (dir.exists(file.path(all_basins_dir)) == F){
  dir.create(path=file.path(all_basins_dir), recursive=T)
}

writeOGR(obj = all_basins_sp, 
         dsn = all_basins_dir,
         layer="all_basins", 
         driver="ESRI Shapefile")

####
#### save subbasin
####
all_subbasins_dir <- paste0(simple_shapefile_dir, 
                            "all_subbasins/")
if (dir.exists(file.path(all_subbasins_dir)) == F){
  dir.create(path=file.path(all_subbasins_dir), recursive=T)
}

writeOGR(obj = all_subbasins_sp, 
         dsn = all_subbasins_dir, 
         layer="all_subbasins", 
         driver="ESRI Shapefile")

