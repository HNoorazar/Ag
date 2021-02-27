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
pi_1 <- st_intersection(colombia, tt_UScounties)
###############3
#graphics.off()
#par("mar")
#par(mar=c(1,1,1,1))
plot(tt_UScounties$geometry, lwd=.1)
plot(colombia$geometry, axes=F, add=T, col ="red", lwd=.1)
plot(pi_1$geometry, add =T, col="green", density=c(1), lwd=.1)

attArea <- pi_1 %>% mutate(area= st_area(.) %>% as.numeric())
attArea
attArea %>% as_tibble() %>% group_by(NAME, USAFPU) %>% summarize(area = sum(area))
tt_UScounties
nevada_counties <- subset(tt_UScounties, STATE_NAME=="Nevada")
nevada_counties
nevada_counties_area <- nevada_counties %>% mutate(area = st_area(.)  %>% as.numeric() )
nevada_counties_area
ls()
attArea

just_elko = subset(attArea, NAME == "Elko")
just_elko
pi
attArea
dim(attArea)
ls()
tt_UScounties
pi
attArea


########
page = readLines('https://www.nass.usda.gov/Data_and_Statistics/County_Data_Files/Frequently_Asked_Questions/county_list.txt')
page <- page[13:4079]

df <- data.frame(matrix(ncol = 5, nrow = 4067))
col_names = c("state", "district", "county", "state_county", "history")
colnames(df) <- col_names

for (row_count in 1:4067){
	df[row_count, 1] = unlist(strsplit(page[row_count], " "))[1]
	df[row_count, 2] = unlist(strsplit(page[row_count], " "))[4]
	df[row_count, 3] = unlist(strsplit(page[row_count], " "))[7]

	df[row_count, 4] = unlist(strsplit(unlist(strsplit(page[row_count], " "))[10], "\t"))[1]
	df[row_count, 5] = unlist(strsplit(unlist(strsplit(page[row_count], " "))[10], "\t"))[7]
}


####
page = readLines('https://www.nass.usda.gov/Data_and_Statistics/County_Data_Files/Frequently_Asked_Questions/county_list.txt')
page <- page[13:4079]

df <- data.frame(matrix(ncol = 5, nrow = 4067))
col_names = c("state", "district", "county", "state_county", "history")
colnames(df) <- col_names

for (row_count in 1:4067){
    a = unlist(strsplit(page[row_count], "\t"))
    a = a[a!=""]
    df[row_count, 5] = a[2]
	
    a = a[1]
    a = unlist(strsplit(a, " "))
    a = a[a!=""]

	df[row_count, 1] = a[1]
	df[row_count, 2] = a[2]
	df[row_count, 3] = a[3]

	len_a = length(a)
	df[row_count, 4] = paste(a[4:length(a)], collapse = ' ')
	
}