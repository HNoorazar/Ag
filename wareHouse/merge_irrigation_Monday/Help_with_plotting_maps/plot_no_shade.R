### Email Map Help
library(ggmap)
library(maps)
library(tmap)

library(rgdal)
library(magrittr)
library(sf)
library(raster)
library(dplyr)
library(tigris)
library(ggplot2)
library(ggthemes)

library(choroplethr)
library(choroplethrMaps)

### read csv files
csv_data_dir <- "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/Help_with_plotting_maps/"

cnt_level <- "USGSDataAtCountyLevel.csv"
cnt_level <- data.table(read.csv(paste0(csv_data_dir, cnt_level)))
names(cnt_level)[names(cnt_level) == "fips"] = "FIPS"

crd_level <- "USGSDataAtCRDLevel.csv"
crd_level <- data.table(read.csv(paste0(csv_data_dir, crd_level)))

stat_level <- "USGSDataAtStateLevel.csv"
stat_level <- data.table(read.csv(paste0(csv_data_dir, stat_level)))
stat_level = stat_level[-51, ] # remove the last row (it is all NA)

## read us counties map
ds = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/UScounties/"
US_cnt_ly_name = "UScounties"
US_cnt <- read_sf(dsn=path.expand(ds), layer = US_cnt_ly_name, quiet = TRUE)

# get rid of Alaska and Hawaii
US_cnt_main_land = US_cnt[US_cnt$STATE_NAME != "Alaska", ]
US_cnt_main_land = US_cnt_main_land[US_cnt_main_land$STATE_NAME != "Hawaii", ]

#### merge county level
cnt_level_merge <- merge(US_main_land, cnt_level, by="FIPS", all.x=T)

####
#### A- totalAcres15 (total irrigated acres in 2015)
####
plot(cnt_level_merge$geometry, col=cnt_level_merge$totalAcres15)
title( main = "County Level (Total Irrigated Acres in 2015)", cex.main=1)

(cnt_colors <-
   with(cnt_level_merge,
        data.frame(cnt_levels = levels(factor(cnt_level_merge$FIPS)),
                   color = I(brewer.pal(nlevels(cnt_levels), name = 'Dark2')))))   
plot(cnt_level_merge$geometry,
     col = jColors$color[match(jDat$continent, jColors$continent)],
     main = 'custom color scheme based on Dark2', cex = 2)
####
#### B- IrrTotalWith15 (total irrigation water withdrawal in 2015 in MGD)
####
plot(cnt_level_merge$geometry, col=cnt_level_merge$IrrTotalWith15)
title( main = "County Level (Total Irrigation Water Withdrawal in 2015 in MGD)", cex.main=1)
####
#### C- totalAcres10 (total irrigated acres in 2010)
####
plot(cnt_level_merge$geometry, col=cnt_level_merge$totalAcres10)
title( main = "County Level (Total Irrigated Acres in 2010)", cex.main=1)
####
#### D- IrrTotalWith10 (total irrigation water withdrawal in 2010 in MGD)
####
plot(cnt_level_merge$geometry, col=cnt_level_merge$IrrTotalWith10)
title( main = "County Level (Total Irrigation Water Withdrawal in 2010 in MGD)", cex.main=1)
#######################################################################################################
##################################                                          ###########################
##################################                                          ###########################
##################################                                          ###########################
#######################################################################################################
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
#######################################################################################################
##################################                                          ###########################
##################################                                          ###########################
##################################                                          ###########################
#######################################################################################################

#### merge state level
state_level_merge <- merge(US_main_land, stat_level, by="STATE_NAME", all.x=T)
####
#### A- totalAcres15 (total irrigated acres in 2015)
####
plot(state_level_merge$geometry, col=state_level_merge$totalAcres15)
title( main = "State Level (Total Irrigated Acres in 2015)", cex.main=1)
####
#### B- IrrTotalWith15 (total irrigation water withdrawal in 2015 in MGD)
####
plot(state_level_merge$geometry, col=state_level_merge$IrrTotalWith15)
title( main = "State Level (Total Irrigation Water Withdrawal in 2015 in MGD)", cex.main=1)
####
#### C- totalAcres10 (total irrigated acres in 2010)
####
plot(state_level_merge$geometry, col=state_level_merge$totalAcres10)
title( main = "State Level (Total Irrigated Acres in 2010)", cex.main=1)
####
#### D- IrrTotalWith10 (total irrigation water withdrawal in 2010 in MGD)
####
plot(state_level_merge$geometry, col=state_level_merge$IrrTotalWith10)
title( main = "State Level (Total Irrigation Water Withdrawal in 2010 in MGD)", cex.main=1)
#######################################################################################################
##################################                                          ###########################
##################################                                          ###########################
##################################                                          ###########################
#######################################################################################################