

param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/new_params_for_analog/"
conus_tree_fruit <- read.delim(paste0(param_dir, "conus_tree_fruit.txt"), 
                               header = TRUE, sep = ",") %>%
                   data.table()

conus_tree_fruit$location <- paste(conus_tree_fruit$lat, conus_tree_fruit$lon, sep="_")

p_dir <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
us_county_lat_long <- read.csv(paste0(p_dir, "us_county_lat_long.csv"), 
                      header=T, as.is=T)  %>%
                      data.table()
us_county_lat_long$county = gsub("County", "", us_county_lat_long$county)
setnames(us_county_lat_long, old=c("vicclat", "vicclon"), new=c("lat", "lon"))

conus_tree_fruit <- merge(conus_tree_fruit, us_county_lat_long, by=c("lat", "lon"))

write.table(conus_tree_fruit, file = paste0(param_dir, "conus_tree_fruit.csv"), 
            row.names=FALSE, na="", col.names=T, sep=",")


##### the 1293 locations
h_dir <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
hist <- read.delim(paste0(h_dir, "all_us_locations_list.txt"), header=F) %>%
        data.table()
setnames(hist, old=c("V1"), new=c("location"))

us_county_lat_long$location = paste(us_county_lat_long$lat, us_county_lat_long$lon, sep="_")
us_county_lat_long <- within(us_county_lat_long, remove(lat, lon))
hist <- merge(hist, us_county_lat_long, by=c("location"))

write.table(hist, file = paste0(param_dir, "cod_moth_historical_info.csv"), 
            row.names=FALSE, na="", col.names=T, sep=",")


conus_tree_fruit$location <- paste(conus_tree_fruit$lat, conus_tree_fruit$lon, sep="_")
conus_tree_fruit <- subset(conus_tree_fruit, select = c(location, fips, state, county))
hist <- subset(hist, select = c(location, fips, state, county))

hist_counts <- hist %>% 
               group_by(fips, state, county) %>%
               transmute(grid_count = n_distinct(location)) %>%
               unique() %>%
               data.table()

write.table(hist_counts, file = paste0(param_dir, "cod_moth_hist_grid_count_within_counties.csv"), 
            row.names=FALSE, na="", col.names=T, sep=",")


conus_tree_fruit_counts <- conus_tree_fruit %>% group_by(fips, state, county) %>%
                           transmute(grid_count = n_distinct(location)) %>%
                           unique() %>%
                           data.table()

write.table(conus_tree_fruit_counts, file = paste0(param_dir, "conus_fruit_girid_counts_within_counties.csv"), 
            row.names=FALSE, na="", col.names=T, sep=",")


##########################################################
param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/new_params_for_analog/"

hist_grid_count <- read.csv(paste0(param_dir, "cod_moth_hist_grid_count_within_counties.csv"),
                            header=T, as.is=T)  %>%
                            data.table()

hist_grid_count_10 <- hist_grid_count %>% 
                      filter(grid_count >= 10) %>%
                      data.table()

hist_grid_count_5 <- hist_grid_count %>% 
                     filter(grid_count >= 5) %>%
                     data.table()

hist_grid_count_7 <- hist_grid_count %>% 
                     filter(grid_count >= 7) %>%
                     data.table()

hist_grid_count_9 <- hist_grid_count %>% 
                     filter(grid_count >= 9) %>%
                     data.table()

write.table(hist_grid_count_10, file = paste0(param_dir, "hist_counties_with_more_10.csv"), 
            row.names=FALSE, na="", col.names=T, sep=",")


write.table(hist_grid_count_5, file = paste0(param_dir, "hist_counties_with_more_5.csv"), 
            row.names=FALSE, na="", col.names=T, sep=",")

write.table(hist_grid_count_7, file = paste0(param_dir, "hist_counties_with_more_7.csv"), 
            row.names=FALSE, na="", col.names=T, sep=",")

write.table(hist_grid_count_9, file = paste0(param_dir, "hist_counties_with_more_9.csv"), 
            row.names=FALSE, na="", col.names=T, sep=",")

