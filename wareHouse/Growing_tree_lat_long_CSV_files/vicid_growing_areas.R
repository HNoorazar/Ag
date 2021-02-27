rm(list=ls())
library(ggmap)
library(ggpubr)
library(plyr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
library(data.table)
options(digits=9)
options(digit=9)

################################################################################
####################
####################             Local Locations the 295 locations
####################
################################################################################

param_dir <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
local_295 <- read.table(paste0(param_dir, "LocationGroups.csv"), header=TRUE, sep=",")
local_295 <- within(local_295, remove(locationGroup))
setnames(local_295, old=c("latitude", "longitude"), new=c("lat", "long"))

################################################################################
###########                                                          ###########
###########                                                          ###########
###########                                                          ###########
################################################################################

in_dir <- "/Users/hn/Documents/GitHub/Kirti/wareHouse/Growing_tree_lat_long_CSV_files/"
plot_dir <- "/Users/hn/Desktop/"

#####
##### The locations of all farms 
##### in the US given by Kiriti's 
##### email xlsx file
#####
all_west <- read.table(paste0(in_dir, "lat_long_Grid.csv"), header=TRUE, sep=",")
grids <- read.table(paste0(in_dir, "TreeFruitGrids.csv"), header=TRUE, sep=",")
setnames(all_west, old=c("vicidgeo"), new=c("vicid"))

west_0_percent <- merge(all_west, grids) %>% filter(SumTreeFruit > 0)
west_0_percent <- within(west_0_percent, remove(vicid, SumTreeFruit))

west_1_percent <- merge(all_west, grids) %>% filter(SumTreeFruit >= 0.01)
west_1_percent <- within(west_1_percent, remove(vicid, SumTreeFruit))

west_5_percent <- merge(all_west, grids) %>% filter(SumTreeFruit >= .05)
west_5_percent <- within(west_5_percent, remove(vicid, SumTreeFruit))

################################################################################
###########                                                          ###########
###########                                                          ###########
###########                                                          ###########
################################################################################

file_list_0_percent <- paste0("data_", west_0_percent$lat, "_", west_0_percent$long)
file_list_1_percent <- paste0("data_", west_1_percent$lat, "_", west_1_percent$long)
file_list_5_percent <- paste0("data_", west_5_percent$lat, "_", west_5_percent$long)

# there are 5 files are in the 295 files that are not in the all west
# extract them here
D2 <- paste0("data_", local_295$lat, "_", local_295$long)
D2 <- subset(D2, !(D2 %in% file_list_0_percent))
all_west_locs_0 <- c(D2, file_list_0_percent)

D2 <- paste0("data_", local_295$lat, "_", local_295$long)
D2 <- subset(D2, !(D2 %in% file_list_1_percent))
all_west_locs_1 <- c(D2, file_list_1_percent)

D2 <- paste0("data_", local_295$lat, "_", local_295$long)
D2 <- subset(D2, !(D2 %in% file_list_5_percent))
all_west_locs_5 <- c(D2, file_list_5_percent)

main_out <- "/Users/hn/Documents/GitHub/Kirti/Chilling/parameters/"

write.table(x = all_west_locs_0, 
            file = file.path(main_out, "file_list_0_percent.txt"), 
            row.names = F, col.names = F)

write.table(x = all_west_locs_1,
            file = file.path(main_out, "file_list_1_percent.txt"),
            row.names = F, col.names = F)

write.table(x = all_west_locs_5,
            file = file.path(main_out, "file_list_5_percent.txt"), 
            row.names = F, col.names = F)

# test <- read.delim(file = paste0(main_out, "file_list_1_percent.txt"), header = F)
# test <- as.vector(test$V1)

########################################################################

##############      Form location groups, Warmer, cooler, Oregon

########################################################################
all_west_locs <- paste0("data_", all_west$lat, "_", all_west$long)
y <- sapply(all_west_locs, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
west_lats = y[2, ]
west_long = y[3, ]
all_west_lat_long <- data.table(west_lats, west_long)
setnames(all_west_lat_long, 
         old=c("west_lats", "west_long"), 
         new=c("latitude", "longitude"))

pm_dir <- "/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/"
WA_LocationGroups <- read.csv(paste0(pm_dir, "LocationGroups.csv"))
setnames(WA_LocationGroups, 
         old=c("latitude", "longitude"), 
         new=c("lat", "long"))

all_north_west_for_chill <- subset(merged_locs_0_percent,
                                   select=c("lat", "long"))

all_north_west_for_chill <- join(x=all_north_west_for_chill, 
                                 y=WA_LocationGroups, 
                                 type = "left", match = "all")

all_north_west_for_chill[is.na(all_north_west_for_chill)] <- 3
all_north_west_for_chill$location <- paste0(all_north_west_for_chill$lat, "_", 
                                            all_north_west_for_chill$long)
all_north_west_for_chill <- all_north_west_for_chill %>% 
                            filter(!(location %in% monata_sites_lat_long$location))

new_pm_dir <- "/Users/hn/Documents/GitHub/Kirti/Chilling/parameters/"

write.table(x = all_north_west_for_chill,
            file = file.path(new_pm_dir, "LocationGroups_NoMontana.csv"),
            row.names = F, col.names = T, sep=",")

################################################################
##########                 USA Locations
################################################################
all_us_locations <- read.delim(file = paste0(param_dir, "all_us_locations_list.txt"), header = F)
all_us_locations <- as.vector(all_us_locations$V1)

x <- sapply(all_us_locations, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
us_lats = x[1, ]
us_long = x[2, ]
all_us_locations <- data.table(us_lats, us_long)

all_us_locations$lat = as.numeric(all_us_locations$us_lats)
all_us_locations$long = as.numeric(all_us_locations$us_long)

all_us_locations$lat = as.numeric(all_us_locations$lat)
all_us_locations$long = as.numeric(all_us_locations$long)

######################################################

#                   Map PLOTS

#######################################################
states <- map_data("state")

all_us_map <- all_us_locations %>%  
              ggplot() +
              geom_polygon(data = states, 
                           aes(x=long, y=lat, group = group),
                               fill = "grey", color = "red", size=0.1) +
              geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1)

ggsave(filename = "all_us_map.png", plot = all_us_map, device = "png",
       width = 20, height = 12, units = "in", dpi=400, path=plot_dir)

states <- map_data("state")
states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

local_295_map <- local_295 %>%  
                 ggplot() +
                 geom_polygon(data = states_cluster, 
                              aes(x=long, y=lat, group = group),
                                  fill = "grey", color = "red", size=0.1) +
                 geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1)

ggsave(filename = "local_295_map.png", plot = local_295_map, device = "png",
       width = 5, height = 3, units = "in", dpi=400, path=plot_dir)

merged_locs_0_percent <- merge(all_west, grids) %>% filter(SumTreeFruit > 0)
merged_locs_1_percent <- merge(all_west, grids) %>% filter(SumTreeFruit >= 0.01)
merged_locs_5_percent <- merge(all_west, grids) %>% filter(SumTreeFruit >= 0.05)

states <- map_data("state")
states_cluster <- subset(states, region %in% c("oregon", "washington", "idaho"))

city_lat <-  as.numeric(c("45.7054", "47.4235", "44.5646", "48.4110", "44.0521", 
                          "44.9429", "46.0646", "45.5200", "46.6021", "46.2804"))

city_long <- as.numeric(c("-121.5215", "-120.3103", "-123.2620", "-119.5276", 
                          "-123.0868", "-123.0351", "-118.3430", "-122.9366", "-120.5059",
                          "-119.2752"))

city_names = c("Hood River", "Wenatchee", "Corvallis", "Omak", "Eugene", 
               "Salem", "Walla Walla", "Hillsboro", "Yakima", "Richland")

cities <- data.frame(city=city_names, lat=city_lat, long=city_long)

closets_farm_lat <- c(48.40625, 47.40625, 46.28125, 
                      45.53125, 45.71875, 44.59375, 
                      44.03125, 44.96875, 46.03125,
                      46.59375)

closets_farm_long <- c(-119.53125, -120.34375, -119.34375, 
                       -122.90625, -121.53125, -123.28125, 
                       -123.09375, -123.03125, -118.34375, 
                       -120.53125)
close_city <- c("Omak", "Wenatchee", "Richland" , 
                "Hilsboro", "Hood River", "Corvallis", 
                "Eugene", "Salem", "Walla Walla", 
                "Yakima")
close_cities <- data.frame(city=close_city, lat=closets_farm_lat, long=closets_farm_long)

loc_0_percent_map <- merged_locs_0_percent %>%  
                     ggplot() +
                     geom_polygon(data = states_cluster, 
                                  aes(x=long, y=lat, group = group),
                                      fill = "grey", color = "red", size=0.1) +
                     geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1) +
                     geom_point(data= cities, aes(x=long, y=lat), color = "red", size =1 ) + 
                     geom_point(data= close_cities, aes(x=long, y=lat), color = "blue", size=1 )


ggsave(filename = "loc_0_percent_blue_city_red_farm.png", plot = loc_0_percent_map, device = "png",
        width = 5, height = 3, units = "in", dpi=400, path=plot_dir)

################################################

loc_1_percent_map <- merged_locs_1_percent %>%  
                     ggplot() +
                     geom_polygon(data = states_cluster, 
                                  aes(x=long, y=lat, group = group),
                                      fill = "grey", color = "red", size=0.1) +
                     geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1)

ggsave(filename = "loc_1_percent.png", plot = loc_1_percent_map, device = "png",
       width = 5, height = 3, units = "in", dpi=400, path=plot_dir)

loc_5_percent_map <- merged_locs_5_percent %>%  
                     ggplot() +
                     geom_polygon(data = states_cluster, 
                                  aes(x=long, y=lat, group = group),
                                      fill = "grey", color = "red", size=0.1) +
                     geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1)

ggsave(filename = "loc_5_percent.png", plot = loc_5_percent_map, device = "png",
       width = 5, height = 3, units = "in", dpi=400, path=plot_dir)


##################################### Read Mins file to find the fucking 5% sites in OR

# library(foreign)
# Min_counties <- read.dbf("/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/vic_grid_cover_conus/VICID_CO.DBF")
# setnames(Min_counties, old= colnames(Min_counties), new= tolower(colnames(Min_counties)))

Min_counties <- read.csv("/Users/hn/Documents/GitHub/Kirti/codling_moth/code/parameters/us_county_lat_long.csv", 
                           header=T, sep=",")

setnames(Min_counties, old=c("vicclat", "vicclon"), new=c("lat", "long"))
counties_filter <- subset(Min_counties, select= c(vicid, state, lat, long))

##########################################################################

counties_5_perc_merged <- merge(x = merged_locs_5_percent, y = counties_filter, all.x=T)
counties_5_perc_OR <- counties_5_perc_merged %>% filter(state == "OR")

counties_5_perc_OR_map <- counties_5_perc_OR %>%  
                     ggplot() +
                     geom_polygon(data = states_cluster, 
                                  aes(x=long, y=lat, group = group),
                                      fill = "grey", color = "red", size=0.1) +
                     geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1)

ggsave(filename = "counties_5_perc_OR_map.png", plot = counties_5_perc_OR_map, device = "png",
       width = 5, height = 3, units = "in", dpi=400, path=plot_dir)

#####################################
##################################### Read Mins file to find the fucking Monatana Areas
#####################################

counties_0_perc_merged <- merge(x = merged_locs_0_percent, y = counties_filter, all.x=T)

oregon_sites <- counties_0_perc_merged %>% filter(state=="OR")
monata_sites <- counties_0_perc_merged %>% filter(state=="MT")

monata_sites_lat_long <- subset(monata_sites, select=c("lat", "long"))
monata_sites_lat_long$location <- paste0(monata_sites_lat_long$lat, "_", monata_sites_lat_long$long)

oregon_sites <- subset(oregon_sites, select=c("lat", "long"))
oregon_sites$location = paste0(oregon_sites$lat, "_", oregon_sites$long)

write.csv(monata_sites_lat_long, file = "/Users/hn/Desktop/monata_sites_lat_long.csv", row.names=FALSE)

monata_sites <- paste0("data_", monata_sites$lat, "_", monata_sites$long)

write.table(x = monata_sites,
            file = file.path(plot_dir, "monata_sites.txt"),
            row.names = F, col.names = F)


loc_0_percent_map <- merged_locs_0_percent %>%  
                     ggplot() +
                     geom_polygon(data = states_cluster, 
                                  aes(x=long, y=lat, group = grosup),
                                      fill = "grey", color = "red", size=0.1) +
                     geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1)

ggsave(filename = "loc_0_percent.png", plot = loc_0_percent_map, device = "png",
       width = 5, height = 3, units = "in", dpi=400, path=plot_dir)

loc_1_percent_map <- merged_locs_1_percent %>%  
                     ggplot() +
                     geom_polygon(data = states_cluster, 
                                  aes(x=long, y=lat, group = grosup),
                                      fill = "grey", color = "red", size=0.1) +
                     geom_point(aes_string(x = "long", y = "lat"), alpha = 0.8, size=0.1)

ggsave(filename = "loc_1_percent.png", plot = loc_1_percent_map, device = "png",
       width = 5, height = 3, units = "in", dpi=400, path=plot_dir)



