### Email Map Help
rm(list=ls())
library(ggmap)
library(maptools)
library(maps)
library(tmap)
library(data.table)
library(rgdal)
library(magrittr)
library(sf)
library(raster)
library(dplyr)
library(tigris)
library(ggplot2)
library(ggthemes)

# library(choroplethr)
# library(choroplethrMaps)

setwd("/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/Help_with_plotting_maps")
### read csv files
# csv_data_dir <- "Data/Processed/"
cnt_level <- "USGSDataAtCountyLevel.csv"
cnt_level <- data.table(read.csv(cnt_level))
names(cnt_level)[names(cnt_level) == "fips"] = "FIPS"

crd_level <- "USGSDataAtCRDLevel.csv"
crd_level <- data.table(read.csv(crd_level))

stat_level <- "USGSDataAtStateLevel.csv"
stat_level <- data.table(read.csv(stat_level))
stat_level = stat_level[-51, ] # remove the last row (it is all NA)

## read us counties map

ds = "/Users/hn/Documents/GitHub/Kirti/wareHouse/merge_irrigation_Monday/UScounties/"
US_cnt_ly_name = "UScounties"
US_cnt <- read_sf(dsn=path.expand(ds), layer = US_cnt_ly_name, quiet = TRUE)

# get rid of Alaska and Hawaii
US_cnt_main_land = US_cnt[US_cnt$STATE_NAME != "Alaska", ]
US_cnt_main_land = US_cnt_main_land[US_cnt_main_land$STATE_NAME != "Hawaii", ]
US_cnt_main_land$FIPS=as.numeric(US_cnt_main_land$FIPS) ### change to numeric so it matched with cnt_level
#### merge county level
cnt_level_merge <- merge(US_cnt_main_land, cnt_level, by="FIPS", all.x=T)
# View(cnt_level_merge)

## plot county maps
library(RColorBrewer)
my.palette <- brewer.pal(n = 7, name = "Blues")
plot(cnt_level_merge["totalAcres15"], 
     pal=brewer.pal(n = 7, name = "Blues"), 
     lwd= .1, breaks = c(0,25,50,100,200,500,1000,1200),
     main = "County Level (Total Irrigated Acres in 2015)",
     cex.main=1)

# summary(cnt_level_merge)
plot(cnt_level_merge["IrrTotalWith15"], 
     pal=brewer.pal(n = 6, name = "Blues"), 
     lwd= .1, 
     breaks = c(0, 1.5, 10, 100, 500, 1000, 2000),
     main = "County Level (Total Irrigation Water Withdrawal in 2015 in MGD)", 
     cex.main=1)

