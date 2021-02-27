


water_right <- data.table(readRDS("/Users/hn/Desktop/new_points.rds"))

water_right <- water_right %>% filter(WaRecPhase != "LongForm")
water_right <- water_right %>% filter(WaRecPhase != "Claim")

water_right <- DataCombine::DropNA(water_right, Var="PriorityDa")
water_right <- DataCombine::DropNA(water_right, Var="WaRecRCWCl")

# needed_cols <- c("WaRecID", "Source_Lat", "Source_Lon", 
#                  "PriorityDa", "WaRecRCWCl",
#                  "WRIA_NM", "Subbasin", "Source_NM")

# water_right <- subset(water_right, select=needed_cols)


old_names <- c("WaRecID", "Source_Lat", "Source_Lon", 
               "PriorityDa", 
               "Subbasin", "WRIA_NM", "Source_NM") 

new_names <- c("WR_Doc_ID", "lat", "long", 
               "right_date", 
               "subbasin", "county_type", "stream")

setnames(water_right, old=old_names, new=new_names)

set.seed(124)
lat_norm <- rnorm(dim(water_right)[1], mean=0, sd=.1)
long_norm <- rnorm(dim(water_right)[1], mean=0, sd=.1)

water_right$lat <- water_right$lat + lat_norm
water_right$long <- water_right$long + long_norm

water_right$location <- paste0(water_right$lat, 
                               " E, ", 
                               abs(water_right$long), 
                               " W")

water_right$popup <- paste0(water_right$location, 
                            " ID:", 
                            water_right$WR_Doc_ID)

## Remove rows with missing right_date 

saveRDS(water_right, 
        "/Users/hn/Desktop/water_right_attributes_jiggled.rds")




