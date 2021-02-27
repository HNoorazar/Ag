spatial_wtr_right <- readRDS("/Users/hn/Desktop/Desktop/Ag/check_point/water_right/data/water_right_attributes_not_jiggled.rds")

spatial_wtr_right <- within(spatial_wtr_right, remove("colorr"))

spatial_wtr_right$popup <- paste0(spatial_wtr_right$lat, " N, ", 
                                  spatial_wtr_right$long, 
                                  " W, ID: ", 
                                  spatial_wtr_right$WaRecID, 
                                  ", ", 
                                  spatial_wtr_right$subbasin,
                                  ", ", 
                                  spatial_wtr_right$PersonLast)

spatial_wtr_right_Wenatchee <- spatial_wtr_right %>% filter(WRIA_NM == "Wenatchee") %>% data.table()
dim(spatial_wtr_right_Wenatchee)
dim(spatial_wtr_right)
saveRDS(spatial_wtr_right_Wenatchee, "/Users/hn/Desktop/Desktop/Ag/check_point/water_right/data/spatial_wtr_right_wenatchee.rds")

spatial_wtr_right_Methow <- spatial_wtr_right %>% filter(WRIA_NM == "Methow") %>% data.table()
dim(spatial_wtr_right_Methow)
dim(spatial_wtr_right)
saveRDS(spatial_wtr_right_Methow, "/Users/hn/Desktop/Desktop/Ag/check_point/water_right/data/spatial_wtr_right_methow.rds")

spatial_wtr_right_Naches <- spatial_wtr_right %>% filter(WRIA_NM == "Naches") %>% data.table()
dim(spatial_wtr_right_Naches)
dim(spatial_wtr_right)
saveRDS(spatial_wtr_right_Naches, "/Users/hn/Desktop/Desktop/Ag/check_point/water_right/data/spatial_wtr_right_naches.rds")

spatial_wtr_right_Upper_Yakima <- spatial_wtr_right %>% filter(WRIA_NM == "Upper Yakima") %>% data.table()
dim(spatial_wtr_right_Upper_Yakima)
dim(spatial_wtr_right)
saveRDS(spatial_wtr_right_Upper_Yakima, "/Users/hn/Desktop/Desktop/Ag/check_point/water_right/data/spatial_wtr_right_upper_yakima.rds")

spatial_wtr_right_Lower_Yakima <- spatial_wtr_right %>% filter(WRIA_NM == "Lower Yakima") %>% data.table()
dim(spatial_wtr_right_Lower_Yakima)
dim(spatial_wtr_right)
saveRDS(spatial_wtr_right_Lower_Yakima, "/Users/hn/Desktop/Desktop/Ag/check_point/water_right/data/spatial_wtr_right_lower_yakima.rds")

spatial_wtr_right_Walla <- spatial_wtr_right %>% filter(WRIA_NM == "Walla Walla") %>% data.table()
dim(spatial_wtr_right_Walla)
dim(spatial_wtr_right)
saveRDS(spatial_wtr_right_Walla, "/Users/hn/Desktop/Desktop/Ag/check_point/water_right/data/spatial_wtr_right_walla.rds")

spatial_wtr_right_okanagon <- spatial_wtr_right %>% filter(WRIA_NM == "Okanogan") %>% data.table()
dim(spatial_wtr_right_okanagon)
dim(spatial_wtr_right)
saveRDS(spatial_wtr_right_okanagon, "/Users/hn/Desktop/Desktop/Ag/check_point/water_right/data/spatial_wtr_right_okanagon.rds")

