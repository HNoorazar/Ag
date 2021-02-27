rm(list = ls())
library(ggmap)
library(rgdal)
library(tmap)
library(magrittr)
library(sf)
ds = "./UScounties"
ly = "UScounties"
UScounties <- readOGR(dsn = ds, layer = ly)
tt_UScounties <- read_sf(dsn=path.expand(ds), layer = ly, quiet = TRUE)
data_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/IMPACT_FPU_Map/"
layer_name = "fpu2015_polygons_v3_multipart_polygons"
the_shape <- read_sf(dsn=path.expand(data_dir), layer = layer_name, quiet = TRUE)
# extract US
tofind <- c("_USA")
the_shape <- the_shape[grep(paste(tofind, collapse = "|"), the_shape$FPU2015), ]
the_shape = within(the_shape, remove(FPU2015))
the_shape
tt_UScounties
colombia <- subset(the_shape, USAFPU=="Columbia")
pi <- st_intersection(colombia, tt_UScounties)
plot(colombia$geometry, axes=T)
graphics.off()
plot(colombia$geometry, axes=T)
graphics.off()
par("mar")
par(mar=c(1,1,1,1))
plot(colombia$geometry, axes=T)
plot(tt_UScounties$geometry, add =T)
plot(pi$geometry, add =T, color="red")
plot(pi$geometry, add =T, col="red")
library(dplyr)
attArea <- pi %>% mutate(area= st_area(.) %>% as.numeric())
attArea
attArea %>% as_tibble() %>% group_by(tt_UScounties, colombia) %>% summerize(area = sum(area))
ls()
attArea
attArea %>% as_tibble() %>% group_by(NAME, colombia) %>% summerize(area = sum(area))
attArea %>% as_tibble() %>% group_by(NAME, USAFPU) %>% summerize(area = sum(area))
attArea %>% as_tibble() %>% group_by(NAME, USAFPU) %>% summarize(area = sum(area))
tt_UScounties
nevada_counties <- subset(tt_UScounties, STATE_NAME=="Nevada")
nevada_counties
nevada_counties_area <- nevada_counties %>% mutate(area = st_area(.)  %>% as.numeric() )
nevada_counties_area
ls()
attArea
attArea[attArea$NAME == "Elko"]
just_elko = subset(attArea, NAME == "Elko")
just_elko
pi
attArea
dim(attArea)
ls()
tt_UScounties
pi
attArea
