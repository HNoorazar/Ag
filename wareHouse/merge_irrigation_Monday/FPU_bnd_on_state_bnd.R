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
FPU_USA_Main <- FPU_data[c(2:7,9:16), ]

# FPU_USA_Main = FPU_data[FPU_data$USAFPU != "Alaska", ]
#FPU_USA_Main = FPU_USA_Main[FPU_USA_Main$USAFPU != "Hawaii", ]
#################################### US State map shapefile

data_dir = "./us_state/"
layer_name = "cb_2017_us_state_20m"
state_shape <- read_sf(dsn=path.expand(data_dir), layer = layer_name, quiet = TRUE)

# drop Alaska and Hawaii
state_shape <- state_shape[state_shape$NAME != "Alaska", ]
state_shape <- state_shape[state_shape$NAME != "Hawaii", ]

plot(state_shape$geometry, add=T)
plot(FPU_USA_Main$geometry, lwd=1, border="red")
plot(state_shape$geometry, add=T)
#################################### US county map shapefile
ds = "./UScounties/"
US_cnt_ly_name = "UScounties"
US_cnt <- read_sf(dsn=path.expand(ds), layer = US_cnt_ly_name, quiet = TRUE)

# get rid of Alaska and Hawaii
US_cnt_main_land = US_cnt[US_cnt$STATE_NAME != "Alaska", ]
US_cnt_main_land = US_cnt_main_land[US_cnt_main_land$STATE_NAME != "Hawaii", ]


plot(US_cnt_main_land$geometry, lwd=.1)






