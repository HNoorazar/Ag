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
SF_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                 "00_shapeFiles/0002_final_shapeFiles/000_Eastern_WA/")

double_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/", 
                     "01_NDVI_TS/70_Cloud/00_Eastern_WA_withYear/2Years/", 
                     "06_list_of_2cropped_IDs_4_website/")


output_dir <- paste0("/Users/hn/Documents/00_GitHub/ag_papers/remote_sensing/plots_for_paper/")
if (dir.exists(output_dir) == F) {dir.create(path = output_dir, recursive = T)}

##############################################################################################################
#
#    Parameter setup
#
plot_w = 8
plot_h = 5.5
years <- c(2016, 2017, 2018)

eastern_counties <- c("Okanogan", "Chelan", "Kittitas", "Yakima", "Klickitat", 
                      "Douglas", "Benton", "Ferry", "Lincoln", "Adams", "Franklin",
                      "Walla Walla", "Pend Oreille", "Stevens", "Spokane", "Whitman", "Garfield", 
                      "Columbia", "Asotin", "Grant")
eastern_counties <- tolower(eastern_counties)
##############################################################################################################
#
#    Body
#
states <- map_data("state")
states_cluster <- subset(states, region %in% c("washington"))

states_cluster <- map_data("county", "washington")


mid_range <- function(x) mean(range(x))
county_names <- do.call(rbind, lapply(split(states_cluster, states_cluster$subregion), function(d) {
  data.frame(lat = mid_range(d$lat), long = mid_range(d$long), subregion = unique(d$subregion))
}))

county_names <- county_names %>%
                filter(subregion %in% eastern_counties)

county_names$lat <- county_names$lat + 0.09
county_names$subregion <- stringr::str_to_title(county_names$subregion)

for (year in years){

  SF_name = paste0("Eastern_", year)
  double_name = paste0("doubleCroppedFields_", year,
                       "_onlyAnnuals_JustIrr_LastSrvFalse_NASSIn_SG_win7_Order3")

  SF <- readOGR(paste0(SF_dir, SF_name, "/", SF_name, ".shp"),
                layer = SF_name, 
                GDAL1_integer64_policy = TRUE)

  double_data <- data.table(read.csv(paste0(double_dir, double_name, ".csv")), as.is=TRUE)
  

  unique_IDs <- unique(double_data$ID)

  SF <- SF[SF@data$ID %in% unique_IDs, ]

  ####### Find Centroids
  SF_centers <- rgeos::gCentroid(SF, byid=TRUE)

  crs <- CRS("+proj=lcc 
             +lat_1=45.83333333333334 
             +lat_2=47.33333333333334 
             +lat_0=45.33333333333334 
             +lon_0=-120.5 +datum=WGS84")

  centroid_coord <- spTransform(SF_centers, 
                                CRS("+proj=longlat +datum=WGS84"))

  centroid_coord_dt <- data.table(centroid_coord@coords)
  setnames(centroid_coord_dt, old=c("x", "y"), new=c("long", "lat"))

  centroidMap <- centroid_coord_dt %>%
                 ggplot() +
                 geom_polygon(data = states_cluster, aes(x=long, y=lat, group=group),
                            fill = "grey", color = "black") +
                 geom_point(aes_string(x="long", y="lat"),  color="blue", size=0.7, alpha=0.4) +
                 geom_polygon(data=states_cluster, aes(long, lat, group=group), fill=NA) + #colour = "grey60"
                 geom_text(data = county_names, aes(long, lat, label=subregion), size=2.3) + # angle = 45
                 theme(axis.title.y = element_blank(),
                       axis.title.x = element_blank(),
                       axis.ticks.y = element_blank(), 
                       axis.text.x = element_blank(),
                       axis.text.y = element_blank(),
                       axis.ticks.x = element_blank(),
                       strip.text = element_text(size=12, face="bold"))

  ggsave(filename = paste0(double_name, ".pdf"),
         plot = centroidMap, 
         width=plot_w, height=plot_h, units = "in", 
         dpi=400, device = "pdf",
         path=output_dir)
  
}



