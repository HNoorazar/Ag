remove(list=ls())
library(ggmap)
library(maps)
library(tmap)
library(raster)
library(rgdal)
library(magrittr)
library(sf)
library(RColorBrewer)
library(raster)
library(plyr)
library(dplyr)
library(tigris)
library(ggplot2)
library(ggthemes)

library(choroplethr)
library(choroplethrMaps)



###########3
prac_path = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/ecoregion_design/"
utah = readOGR(dsn=prac_path, layer="eco_l3_ut")
utah@data$id = rownames(utah@data)
utah.points = fortify(utah, region="id") # melt
utah.df = join(utah.points, utah@data, by="id")

ggplot(utah.df) + 
aes(long, lat, group=group, fill=LEVEL3_NAM) + 
geom_polygon() +
geom_path(color="white") +
coord_equal() +
scale_fill_brewer("Utah Ecoregion")
############
############################
###
### read csv files
###
############################
csv_data_dir <- "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday"
csv_data_dir <- paste0(csv_data_dir, "/Help_with_plotting_maps/")

cnt_level <- "USGSDataAtCountyLevel.csv"
cnt_level <- data.table(read.csv(paste0(csv_data_dir, cnt_level)))
names(cnt_level)[names(cnt_level) == "fips"] = "FIPS"

crd_level <- "USGSDataAtCRDLevel.csv"
crd_level <- data.table(read.csv(paste0(csv_data_dir, crd_level)))

stat_level <- "USGSDataAtStateLevel.csv"
stat_level <- data.table(read.csv(paste0(csv_data_dir, stat_level)))
stat_level = stat_level[-51, ] # remove the last row (it is all NA)
############################
###
### read county shape file
###
############################
ds = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/UScounties/"
US_cnt_ly_name = "UScounties"
US_cnt_OGR <- readOGR(dsn=ds, layer = US_cnt_ly_name)
###
### get rid of Alaska and Hawaii
###
US_cnt_OGR = US_cnt_OGR[US_cnt_OGR$STATE_NAME != "Alaska", ]
US_cnt_OGR = US_cnt_OGR[US_cnt_OGR$STATE_NAME != "Hawaii", ]
############################
###
### Practice Plot
###
############################
#US_cnt_OGR@data$id = rownames(US_cnt_OGR@data)
#US_cnt_OGR_points = fortify(US_cnt_OGR, region="id") # melt
#US_cnt_OGR_df = join(US_cnt_OGR_points, US_cnt_OGR@data, by="id")

ggplot(US_cnt_OGR_df) + 
aes(long, lat, group=group, fill=FIPS) + 
ggtitle("Plot of length by dose") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	plot.title = element_text(hjust = 0.5),
	axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())
###########################################################################################
######################                                          ###########################
######################            COUNTY LEVEL                  ###########################
######################                                          ###########################
###########################################################################################
############################
###
### Merge the data
###
############################

######################## prepare data
cnt_level_merge <- merge(US_cnt_OGR, cnt_level, by="FIPS", all.x=T)

cnt_level_merge@data$id = rownames(cnt_level_merge@data)
cnt_level_merge_points = fortify(cnt_level_merge, region="id") # melt
cnt_level_merge_df = join(cnt_level_merge_points, cnt_level_merge@data, by="id")

######################### plot
####
#### A- totalAcres15 (Total Irrigated Acres in 2015)
####
ggplot(cnt_level_merge_df) + 
aes(long, lat, group=group, fill=totalAcres15) + 
ggtitle("Total Irrigated Acres in 2015") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	plot.title = element_text(hjust = 0.5),
	axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())
####
#### B- IrrTotalWith15 (Total Irrigation Water Withdrawal in 2015 in MGD)
####
ggplot(cnt_level_merge_df) + 
aes(long, lat, group=group, fill=IrrTotalWith15) + 
ggtitle("Total Irrigation Water Withdrawal in 2015 in MGD") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	plot.title = element_text(hjust = 0.5),
	axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())
####
#### C- totalAcres10 (Total Irrigated Acres in 2010)
####
ggplot(cnt_level_merge_df) + 
aes(long, lat, group=group, fill=totalAcres10) + 
ggtitle("Total Irrigated Acres in 2010") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	  plot.title = element_text(hjust = 0.5),
	  axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())
####
#### D- IrrTotalWith10 (Total Irrigation Water Withdrawal in 2010 in MGD)
####
ggplot(cnt_level_merge_df) + 
aes(long, lat, group=group, fill=IrrTotalWith10) + 
ggtitle("Total Irrigation Water Withdrawal in 2010 in MGD") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	plot.title = element_text(hjust = 0.5),
	axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())
######################################################################################
#################                                          ###########################
#################             STATE LEVEL                  ###########################
#################                                          ###########################
######################################################################################
stat_level[, state := as.character(state)][state == "AK", state := "Alaska"]
stat_level[, state := as.character(state)][state == "AL", state := "Alabama"]
stat_level[, state := as.character(state)][state == "AR", state := "Arkansas"]
stat_level[, state := as.character(state)][state == "AZ", state := "Arizona"]
stat_level[, state := as.character(state)][state == "CA", state := "California"]
stat_level[, state := as.character(state)][state == "CO", state := "Colorado"]
stat_level[, state := as.character(state)][state == "CT", state := "Connecticut"]

stat_level[, state := as.character(state)][state == "DE", state := "Delaware"]
stat_level[, state := as.character(state)][state == "FL", state := "Florida"]
stat_level[, state := as.character(state)][state == "GA", state := "Georgia"]
stat_level[, state := as.character(state)][state == "HI", state := "Hawaii"]

stat_level[, state := as.character(state)][state == "IA", state := "Iowa"]
stat_level[, state := as.character(state)][state == "ID", state := "Idaho"]
stat_level[, state := as.character(state)][state == "IL", state := "Illinois"]
stat_level[, state := as.character(state)][state == "IN", state := "Indiana"]
stat_level[, state := as.character(state)][state == "KS", state := "Kansas"]
stat_level[, state := as.character(state)][state == "KY", state := "Kentucky"]
stat_level[, state := as.character(state)][state == "LA", state := "Louisiana"]
stat_level[, state := as.character(state)][state == "MA", state := "Massachusetts"]

stat_level[, state := as.character(state)][state == "MD", state := "Maryland"]
stat_level[, state := as.character(state)][state == "ME", state := "Maine"]
stat_level[, state := as.character(state)][state == "MI", state := "Michigan"]
stat_level[, state := as.character(state)][state == "MN", state := "Minnesota"]

stat_level[, state := as.character(state)][state == "MO", state := "Missouri"]
stat_level[, state := as.character(state)][state == "MS", state := "Mississippi"]
stat_level[, state := as.character(state)][state == "MT", state := "Montana"]
stat_level[, state := as.character(state)][state == "NC", state := "North Carolina"]

stat_level[, state := as.character(state)][state == "ND", state := "North Dakota"]
stat_level[, state := as.character(state)][state == "NE", state := "Nebraska"]
stat_level[, state := as.character(state)][state == "NH", state := "New Hampshire"]
stat_level[, state := as.character(state)][state == "NJ", state := "New Jersey"]

stat_level[, state := as.character(state)][state == "NM", state := "New Mexico"]
stat_level[, state := as.character(state)][state == "NV", state := "Nevada"]
stat_level[, state := as.character(state)][state == "NY", state := "New York"]
stat_level[, state := as.character(state)][state == "OH", state := "Ohio"]

stat_level[, state := as.character(state)][state == "OK", state := "Oklahoma"]
stat_level[, state := as.character(state)][state == "OR", state := "Oregon"]
stat_level[, state := as.character(state)][state == "PA", state := "MinPennsylvanianesota"]
stat_level[, state := as.character(state)][state == "RI", state := "Rhode Island"]

stat_level[, state := as.character(state)][state == "SC", state := "South Carolina"]
stat_level[, state := as.character(state)][state == "SD", state := "South Dakota"]
stat_level[, state := as.character(state)][state == "TN", state := "Tennessee"]
stat_level[, state := as.character(state)][state == "TX", state := "Texas"]

stat_level[, state := as.character(state)][state == "UT", state := "Utah"]
stat_level[, state := as.character(state)][state == "VA", state := "Virginia"]
stat_level[, state := as.character(state)][state == "VT", state := "Vermont"]
stat_level[, state := as.character(state)][state == "WA", state := "Washington"]

stat_level[, state := as.character(state)][state == "WI", state := "Wisconsin"]
stat_level[, state := as.character(state)][state == "WV", state := "West Virginia"]
stat_level[, state := as.character(state)][state == "WY", state := "Wyoming"]

names(stat_level)[names(stat_level) == "state"] = "STATE_NAME"
#######################################################################################
########################                                    ###########################
########################                                    ###########################
########################                                    ###########################
#######################################################################################
############################
###
### Merge the data (State Level)
###
############################
######################### prepare data
state_level_merge <- merge(US_cnt_OGR, stat_level, by="STATE_NAME", all.x=T)

state_level_merge@data$id = rownames(state_level_merge@data)
state_level_merge_points = fortify(state_level_merge, region="id") # melt
state_level_merge_df = join(state_level_merge_points, state_level_merge@data, by="id")

######################### plot
####
#### A- totalAcres15 (Total Irrigated Acres in 2015)
####
ggplot(state_level_merge_df) + 
aes(long, lat, group=group, fill=totalAcres15) + 
ggtitle("Total Irrigated Acres in 2015") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	plot.title = element_text(hjust = 0.5),
	axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())
####
#### B- IrrTotalWith15 (Total Irrigation Water Withdrawal in 2015 in MGD)
####
ggplot(state_level_merge_df) + 
aes(long, lat, group=group, fill=IrrTotalWith15) + 
ggtitle("Total Irrigation Water Withdrawal in 2015 in MGD") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	plot.title = element_text(hjust = 0.5),
	axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())
####
#### C- totalAcres10 (Total Irrigated Acres in 2010)
####
ggplot(state_level_merge_df) + 
aes(long, lat, group=group, fill=totalAcres10) + 
ggtitle("Total Irrigated Acres in 2010") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	plot.title = element_text(hjust = 0.5),
	axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())
####
#### D- IrrTotalWith10 (Total Irrigation Water Withdrawal in 2010 in MGD)
####
ggplot(state_level_merge_df) + 
aes(long, lat, group=group, fill=IrrTotalWith10) + 
ggtitle("Total Irrigation Water Withdrawal in 2010 in MGD") + 
geom_polygon() + 
theme_bw() + 
theme(legend.position = "none", 
	plot.title = element_text(hjust = 0.5),
	axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank(),
      axis.title.y=element_blank(),
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank())

#####################################################################################
################                                          ###########################
################            CRD LEVEL                     ###########################
################                                          ###########################
#####################################################################################





