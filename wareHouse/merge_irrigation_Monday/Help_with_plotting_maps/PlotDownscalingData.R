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
library(data.table)

library(choroplethr)
library(choroplethrMaps)
setwd("/Users/ananth/Documents/Kirti/work/FV_Kirti_FPU_to_CRD")
### read csv files  with USGS data
csv_data_dir <- "Data/Processed/"

cnt_level <- "USGSDataAtCountyLevel.csv"
cnt_level <- data.table(read.csv(paste0(csv_data_dir, cnt_level)))
names(cnt_level)[names(cnt_level) == "fips"] = "FIPS"

crd_level <- "USGSDataAtCRDLevel.csv"
crd_level <- data.table(read.csv(paste0(csv_data_dir, crd_level)))

stat_level <- "USGSDataAtStateLevel.csv"
stat_level <- data.table(read.csv(paste0(csv_data_dir, stat_level)))
stat_level = stat_level[-51, ] # remove the last row (it is all NA)

## read us counties map
ds = "Data/PlottingFromHossein/UScounties/"
US_cnt_ly_name = "UScounties"
US_cnt <- read_sf(dsn=path.expand(ds), layer = US_cnt_ly_name, quiet = TRUE)
county_dataOGR<-readOGR(dsn=ds, layer=US_cnt_ly_name)
US_cntst<-st_as_sf(county_dataOGR)

# get rid of Alaska and Hawaii
US_cnt_main_land = US_cnt[US_cnt$STATE_NAME != "Alaska", ]
US_cnt_main_land = US_cnt_main_land[US_cnt_main_land$STATE_NAME != "Hawaii", ]
US_cnt_main_land$FIPS=as.numeric(as.character(US_cnt_main_land$FIPS)) ### change to numeric so it matched with cnt_level
#### merge county level
cnt_level_merge <- merge(US_cnt_main_land, cnt_level, by="FIPS", all.x=T)
View(cnt_level_merge)


######################
###################### Read the FPU (shapefile) Data off the disk
######################
data_dir = "./IMPACT_FPU_Map/"
layer_name = "fpu2015_polygons_v3_multipart_polygons"
FPU_data <- read_sf(dsn=path.expand(data_dir), layer = layer_name, quiet = TRUE)
FPU_dataOGR <- readOGR(dsn=data_dir, layer=layer_name)
summary(FPU_dataOGR)
FPU_sptosf <- st_as_sf(FPU_dataOGR)

FPU_data <- FPU_sptosf
plot(FPU_dataOGR)
## extract US part of the data from whole world
tofind <- c("_USA")
FPU_data <- FPU_data[grep(paste(tofind, collapse = "|"), FPU_data$FPU2015), ]
FPU_data <- within(FPU_data, remove(FPU2015))

summary(FPU_data)
FPU_dataSub <- FPU_data[c(2:7,9:16), ]

#######################################################################
######
######   make the bbox extent of county data the same as 
######   that of the FPU data, so the polygons can overlap
######
#######################################################################

county_dataOGR@bbox<-FPU_dataOGR@bbox
US_cntst <- st_as_sf(county_dataOGR)

# get rid of Alaska and Hawaii from county data
US_cnt_main_land = US_cnt[US_cnt$STATE_NAME != "Alaska", ]
US_cnt_main_land = US_cnt_main_land[US_cnt_main_land$STATE_NAME != "Hawaii", ]
US_cnt_main_land$FIPS=as.numeric(as.character(US_cnt_main_land$FIPS)) ### change to numeric so it matched with cnt_level
#### merge county level
cnt_level_merge <- merge(US_cnt_main_land, cnt_level, by="FIPS", all.x=T)
#####change NAs to 0
cnt_level_merge[is.na(cnt_level_merge)] <- 0

###############
#######         Plot county data overlayed on FPU boundaries
###############
quartz()
p <- ggplot() +
     geom_sf(data = cnt_level_merge, aes(fill = IrrTotalWith15 ), lwd=.1) + 
     scale_fill_gradient(high = "darkblue", low= "white", name="2015 Total Irrigation Withdrawals (MGD)", 
                         labels = c("0", "", "", "", "100","1000","2000"),
                         breaks = c(0, .5, 1, 10, 100,1000,2000)) +
     theme( panel.grid = element_blank(), panel.grid.major=element_line(colour="transparent")) +
     guides(fill = guide_colourbar(barwidth = 1, barheight = 10)) +
     #theme(legend.position="bottom") +
     geom_sf(data = FPU_dataSub, colour = "black", fill=NA, lwd =1) + 
     geom_sf(data = FPU_dataSub, colour = "black", fill=NA, lwd =1) + 

p

###remove background
p+ coord_sf(datum=NA)+
theme(panel.background=element_blank())
  
 