shapefile_dir <- paste0("/Users/hn/Desktop/Desktop", 
                         "/Ag/check_point/water_right", 
                         "/clipped_streams/")

all_streams_sp <- rgdal::readOGR(dsn=path.expand(
                                          paste0(shapefile_dir, 
                                                 "LowerYakima_streams_clipped/")),
                                layer = "all_streams")
