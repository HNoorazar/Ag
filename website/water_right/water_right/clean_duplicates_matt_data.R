#
#   duplicated_vec WaRecID
# 1           TRUE 4741945
# 2           TRUE 4741945
# 3           TRUE 4745366
# 4           TRUE 4749804
# 5           TRUE 4749804
# 6           TRUE 4743036
# 7           TRUE 4741920
# 8           TRUE 4745210
#
original_water_right <- readRDS("/Users/hn/Desktop/new_points.rds")

needed_cols <- c("WaRecID", "Source_Lat", "Source_Lon", 
                 "PriorityDa",
                 "WRIA_NM", "Subbasin", "Source_NM")

water_right <- subset(original_water_right, select=needed_cols)


old_names <- c("WaRecID", "Source_Lat", 
              "Source_Lon", "PriorityDa", 
              "Subbasin", "WRIA_NM") 

new_names <- c("WR_Doc_ID", "lat", "long", 
               "date", "subbasin", "county_type")

setnames(water_right, old=old_names, new=new_names)

find_dt <- subset(water_right, select=c("WR_Doc_ID"))
find_dt_logical <- duplicated(find_dt)
find_dt_logical <- data.table(find_dt_logical)

water_right <- cbind(water_right, find_dt_logical)

water_right_no_dups <- water_right %>% 
                       filter(find_dt_logical == FALSE)

water_right_dups <- water_right %>% 
                    filter(find_dt_logical == TRUE)

write.table(water_right_dups, 
            file = "/Users/hn/Desktop/water_right_dups.csv", 
            row.names=FALSE, na="", 
            col.names=TRUE, sep=",")

water_right_no_dups$location <- paste0(water_right_no_dups$lat, 
                                       " E, ", 
                                       abs(water_right_no_dups$long), 
                                       " W")
water_right_no_dups$popup <- paste0(water_right_no_dups$location, 
                                    " ID:", 
                                    water_right_no_dups$WR_Doc_ID)


saveRDS(water_right_no_dups, 
        "/Users/hn/Desktop/water_right_attributes.rds")



