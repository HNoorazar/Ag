###
###   Feb 2021. New plots (first attempt) for Springer.
###   Plot something for forcing period and taking out the fucking bloom. Finally!
###
###
# We could/should create two sets of data for each 
# (of the first two) 
# scenarios above or, we can take care of NAs
# -introduced to data by merging frost and bloom-
# in the plotting functions?
#####################################
rm(list=ls())
.libPaths("/data/hydro/R_libs35")
.libPaths()
library(data.table)
library(dplyr)
library(ggpubr)

options(digits=9)
options(digit=9)
############################################################
###
###             local computer source
###
############################################################
source_dir <- "/Users/hn/Documents/00_GitHub/Ag/Bloom/"
in_dir <- "/Users/hn/Documents/01_research_data/bloom/"
param_dir <- paste0(source_dir, "parameters/")

plot_base_dir <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/01_Springer_1/"
#############################################################
###
###              
###
#############################################################

source_1 <- paste0(source_dir, "bloom_core.R")
source_2 <- paste0(source_dir, "bloom_plot_core.R")
source(source_1)
source(source_2)
#############################################################
###
###               Read data off the disk
###
#############################################################
limited_locations <- read.csv(file = paste0(param_dir, "limited_locations.csv"), 
                        header=TRUE, as.is=TRUE)

limited_locations$location <- paste0(limited_locations$lat, "_", limited_locations$long)
limited_locations <- within(limited_locations, remove(lat, long))

chill_doy_map <- read.csv(paste0(param_dir, "/chill_DoY_map.csv"), as.is=TRUE)
#############################################################
#
# read the data
#

thresh <- readRDS(paste0(in_dir, "sept_summary_comp.rds"))

#############################################################
#
# pick up observed and 2026-2099 time period
#
#############################################################

thresh <- pick_obs_and_F(thresh)

#############################################################
#
#              clean up each data table
#
#############################################################

# for sake of iteration:

thresh <- add_location(thresh)

suppressWarnings({thresh <- within(thresh, remove(lat, long))})

thresh <- thresh %>% filter(location %in% limited_locations$location)
thresh <- dplyr::left_join(x = thresh, y = limited_locations, by = "location")
thresh <- data.table(thresh)

#############################################################
emissions <- c("RCP 4.5", "RCP 8.5")
apple_types <- c("Cripps Pink", "Gala", "Red Deli") 

# apple, cherry, pear; cherry 14 days shift, pear 7 days shift
fruit_types <- c("apple", "cherry", "pear") 
fruit_type <- "apple"
remove_NA <- "no"

thresh <- within(thresh, remove(location))

setnames(thresh, old=c("city"), new=c("location"))

###############################################################
# em <- emissions[2]
# app_tp <- apple_types[1]
# thresh_cut <- 75
# loc <- unique(bloom$location)[1]

start_time <- Sys.time()
plot_threshols <- seq(20, 75, 5) # seq(25, 75, 5)
plot_threshols <- c(45, 75)      # seq(25, 75, 5)


ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")

thresh <- thresh %>% 
          filter(location %in% ict) %>% 
          data.table()

thresh$location <- factor(thresh$location, levels = ict, order=TRUE)

####################################################################

thresh_RCP85 <- thresh %>% filter(emission == "RCP 8.5")
thresh_RCP45 <- thresh %>% filter(emission == "RCP 4.5")

# thresh_RCP85$location  <- paste0(thresh_RCP85$location, " - ", thresh_RCP85$emission)
# thresh_RCP45$location  <- paste0(thresh_RCP45$location, " - ", thresh_RCP45$emission)

thresh_RCP85 <- within(thresh_RCP85, remove(emission))
thresh_RCP45 <- within(thresh_RCP45, remove(emission))


ict <- c("Omak - RCP 8.5", "Yakima - RCP 8.5", "Walla Walla - RCP 8.5", "Eugene - RCP 8.5")
ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")
thresh_RCP85$location <- factor(thresh_RCP85$location, levels = ict, order=TRUE)


ict <- c("Omak - RCP 4.5", "Yakima - RCP 4.5", "Walla Walla - RCP 4.5", "Eugene - RCP 4.5")
ict <- c("Omak", "Yakima", "Walla Walla", "Eugene")
thresh_RCP45$location <- factor(thresh_RCP45$location, levels = ict, order=TRUE)

rm(thresh) # bloom

####################################################################################################
##
##  For testing
##
thresh_cut <- plot_threshols[1]
app_tp <- apple_types[1]
####################################################################################################

for (thresh_cut in plot_threshols){
  col_name <- paste0("thresh_", thresh_cut)
  curr_thresh <- thresh_RCP45

  curr_thresh <- subset(curr_thresh, 
                        select=c("location", "chill_season",
                                 "model",
                                 "time_period", col_name))

  #############################################################
  #
  #        REMOVE NAs
  #
  if (remove_NA == "yes"){
    curr_thresh <- curr_thresh %>% 
                   filter(get(col_name)<365) %>% 
                   data.table()
  }

  for (app_tp in apple_types){

    if (paste0("thresh_", thresh_cut) %in% colnames(curr_thresh)){
      setnames(curr_thresh, 
               old=c(paste0("thresh_", thresh_cut)), 
               new=c("thresh"))
    }

    curr_thresh_melt <- melt(data.table(curr_thresh), 
                             id = c("location", "chill_season", 
                                    "time_period", "model"))

    merged_dt <- rbind(curr_thresh_melt)
    merged_dt <- merged_dt %>% filter(time_period=="future")

    if (fruit_type=="apple"){
       title_ <- paste0(thresh_cut, " CP threshold and ", app_tp)
       } else{
        title_ <- paste0(thresh_cut, " CP threshold and ", fruit_type)
    }
    
    source(source_1)
    source(source_2)
    # merged_plt <- double_cloud_2_rows(d1=merged_dt) # + ggtitle(lab=paste0(title_)) 
    merged_plt <- ThreshCloud_2_rows_forForcing(d1 = merged_dt)

    if (remove_NA=="yes"){
      LP <- "NA_removed"
      } else{
      LP <- "NA_NOTremoved"
    }

    plot_dir <- paste0(plot_base_dir, LP, "/thresh_forceLine/no_obs/fixed_y/", 
                                    fruit_type, "_", col_name, "/")
    
    if (dir.exists(plot_dir) == F) {
        dir.create(path = plot_dir, recursive = T)
        print (plot_dir)
      }

    ggsave(plot=merged_plt,
           filename = paste0(fruit_type, "_RCP45_",
                             gsub(" ", "_", app_tp), "_", thresh_cut, "CP_2rows.png"), 
           width=13, height=10, units = "in", 
           dpi = 400, device = "png",
           path=plot_dir)
    print (plot_dir)
  }
}

#
#   What the hell does the following loop do?
#
#
for (thresh_cut in plot_threshols){
  col_name <- paste0("thresh_", thresh_cut)
  curr_thresh <- thresh_RCP85

  curr_thresh <- subset(curr_thresh, 
                        select=c("location", "chill_season",
                                 "model",
                                 "time_period", col_name))

  #############################################################
  #
  #        REMOVE NAs
  #
  if (remove_NA == "yes"){
    curr_thresh <- curr_thresh %>% 
                   filter(get(col_name) < 365) %>% 
                   data.table()
  }

  for (app_tp in apple_types){
    curr_bloom <- bloom_RCP85 %>% 
                  filter(fruit_type == gsub("\ ", "_", tolower(app_tp))) %>% 
                  data.table()

    if (fruit_type=="apple"){
         title_ <- paste0(app_tp, " bloom shift (")
       } else {
         title_ <- paste0(fruit_type, " bloom shift (")
    }
    suppressWarnings({curr_bloom<-within(curr_bloom, 
                                         remove(fruit_type))})
    setcolorder(curr_bloom, c("location", 
                              "time_period", "chill_season", 
                              "model", 
                              "fifty_perc_chill_DoY"))

    if (paste0("thresh_", thresh_cut) %in% colnames(curr_thresh)){
      setnames(curr_thresh, 
               old=c(paste0("thresh_", thresh_cut)), 
               new=c("thresh"))
    }

    curr_thresh_melt <- melt(data.table(curr_thresh), 
                             id=c("location", 
                                  "chill_season", "time_period", 
                                  "model"))

    curr_bloom_melt <- melt(curr_bloom, 
                            id=c("location", 
                                 "chill_season", "time_period", 
                                 "model"))

    merged_dt <- rbind(curr_thresh_melt, curr_bloom_melt)
    merged_dt <- merged_dt %>% filter(time_period=="future")

    if (fruit_type=="apple"){
       title_ <- paste0(thresh_cut, " CP threshold and ", app_tp, " bloom shifts")
       } else{
        title_ <- paste0(thresh_cut, " CP threshold and ", fruit_type, " bloom shifts")
    }
    
    source(source_1)
    source(source_2)
    merged_plt <- double_cloud_2_rows(d1=merged_dt) # + ggtitle(lab=paste0(title_)) 

    if (remove_NA=="yes"){
      LP <- "NA_removed"
      } else{
      LP <- "NA_NOTremoved"
    }

    plot_dir <- paste0(plot_base_dir, "limited_locations_", LP,
                                    "/bloom_thresh_in_one/no_obs/fixed_y_May_15/", 
                                    fruit_type, "_", col_name, "/2ndloop/")
    if (dir.exists(plot_dir) == F) {
        dir.create(path = plot_dir, recursive = T)
        print (plot_dir)
      }

    ggsave(plot=merged_plt,
           filename = paste0(fruit_type, "_RCP85_",
                             gsub(" ", "_", app_tp), "_", thresh_cut, "CP_2rows.png"), 
           width=15, height=10, units = "in", 
           dpi = 400, device = "png",
           path=plot_dir)
    print (plot_dir)
  }
}


print(Sys.time() - start_time)


