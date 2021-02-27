url <- "https://www.nass.usda.gov/Data_and_Statistics/County_Data_Files/Frequently_Asked_Questions/county_list.txt"
url_df <- read_lines(url, skip = 12) %>% 
          data.frame(col = .) %>% 
          mutate(col = str_replace_all(col, "\\t", " ")) %>% 
          separate(col, into = paste0("X", 1:5), sep = "\\s{2,}", extra = "drop") %>%
          na.omit()
colnames(url_df) <- c("state_FIPS", "district_FIPS", "county_FIPS", "state_cnt_name", "flag")
url_df <- data.table(url_df)
url_df <- url_df[url_df$county_FIPS != "000", ]
url_df <- url_df[url_df$county_FIPS != "888", ]
url_df <- url_df[url_df$county_FIPS != "999", ]
url_df <- url_df[url_df$flag == 1]
url_df$FIPS <- with(url_df, paste0(state_FIPS, county_FIPS))
url_df
url_df$FIPS <- with(url_df, paste0(state_FIPS, district_FIPS))
url_df$FIPS <- with(url_df, paste0(state_FIPS, county_FIPS))
url_df$CDR_FIPS <- with(url_df, paste0(state_FIPS, district_FIPS))
url_df
url_dir <- "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/"
saveRDS(url_df, paste0(url_dir, "NASS_county.rds"))
write.csv(url_df, file = paste0(url_dir, "NASS_county.csv" ), row.names=FALSE)
rm(url_dir, url, FPU)
head(US_cnt)
head(url_df)
url_df_1 <- subset(url_df, select = c("FIPS", "CDR_FIPS"))
url_df_1
joint = merge( US_cnt, url_df_1, by= "FIPS" )
joint[joint$STATE_NAME == "Washington", ]
View(joint[joint$STATE_NAME == "Washington", ])
rm(joint)
joint_cnt_crd = merge(US_cnt, url_df, by= "FIPS" )
head(joint_cnt_crd)
plot(joint_cnt_crd$geometry)
# CRD <- unionSpatialPolygons(joint_cnt_crd, joint_cnt_crd.district_FIPS)
# head(joint_cnt_crd)
# CRD <- unionSpatialPolygons(joint_cnt_crd, joint_cnt_crd.district_FIPS)
library("rgdal")
class(polygons)
poly_df <- as.data.frame(joint_cnt_crd)
#CRD <- unionSpatialPolygons(poly_df, poly_df.district_FIPS)
#CRD <- SpatialPolygonsDataFrame(joint_cnt_crd, poly_df)
#typeof(joint_cnt_crd)
#CRD <- aggregate(poly_df, by = district_FIPS)
#CRD <- aggregate(poly_df, by = "district_FIPS")
#CRD <- aggregate(joint_cnt_crd, by = "district_FIPS")
#CRD <- aggregate(joint_cnt_crd, by = district_FIPS)
#CRD <- unionSpatialPolygons(joint_cnt_crd, joint_cnt_crd$district_FIPS)
#CRD <- unionSpatialPolygons(joint_cnt_crd, district_FIPS)
#CRD <- unionSpatialPolygons(joint_cnt_crd, "district_FIPS")
#lps <- getSpPPolygonsLabptSlots(joint_cnt_crd)
#CRD <- unionSpatialPolygons(joint_cnt_crd, "district_FIPS")
CRD_spatial <- as(joint_cnt_crd, 'Spatial')
# CRD <- unionSpatialPolygons(joint_cnt_crd, "district_FIPS")
#CRD <- unionSpatialPolygons(CRD_spatial, "district_FIPS")
#CRD <- unionSpatialPolygons(CRD_spatial, CRD_spatial.district_FIPS)
#CRD <- unionSpatialPolygons(CRD_spatial, "district_FIPS")
#CRD <- unionSpatialPolygons(CRD_spatial, district_FIPS)
#CRD_spatial
#head(CRD_spatial)
#CRD_spatial <- as(joint_cnt_crd, 'Spatial')
#CRD <- unionSpatialPolygons(CRD_spatial, district_FIPS)
#CRD_spatial$district_FIPS
#is.na(CRD_spatial$district_FIPS)
#sum(is.na(CRD_spatial$district_FIPS))
CRD_spatial_DF <- as(CRD_spatial, "data.frame")
unique(CRD_spatial_DF$CDR_FIPS)
CRD_spatial_DF$CRD_FIPS_num = as.numeric(as.character(CRD_spatial_DF$CDR_FIPS))
unique(CRD_spatial_DF$CRD_FIPS_num)
CRD_spatial_DF_aggregated <- aggregate(CRD_spatial_DF[,12],list(CRD_spatial_DF$CDR_FIPS),mean)
row.names(CRD_spatial_DF_aggregated)<-as.character(CRD_spatial_DF_aggregated$Group.1)
head(CRD_spatial_DF_aggregated)
CRD_union <- unionSpatialPolygons(CRD_spatial, CRD_spatial$CDR_FIPS)
CRD_union
CRD_spatial_DF <- as(CRD_spatial, "data.frame")
CRD_spatial_DF$CRD_FIPS_num = as.numeric(as.character(CRD_spatial_DF$CDR_FIPS))
CRD_spatial_DF_aggregated <- aggregate(CRD_spatial_DF[,12],list(CRD_spatial_DF$CDR_FIPS),mean)
row.names(CRD_spatial_DF_aggregated)<-as.character(CRD_spatial_DF_aggregated$Group.1)
# reconvert data frame to spatial polygons
CRD_data_and_shape<-SpatialPolygonsDataFrame(CRD_union, CRD_spatial_DF_aggregated)
plot(CRD_data_and_shape)
head(CRD_spatial_DF_aggregated)

plot(CRD)
head(CRD_spatial)
plot(CRD_spatial)
CRD <- unionSpatialPolygons(CRD_spatial, CRD_spatial$CDR_FIPS)
CRD_fortify <- fortify(CRD)
class(CRD_fortify)
plot(CRD)
head(CRD)
###################
data_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/IMPACT_FPU_Map/"
layer_name = "fpu2015_polygons_v3_multipart_polygons"
FPU_data <- read_sf(dsn=path.expand(data_dir), layer = layer_name, quiet = TRUE)
## extract US part of the data from whole world
tofind <- c("_USA")
FPU_data <- FPU_data[grep(paste(tofind, collapse = "|"), FPU_data$FPU2015), ]
FPU_data <- within(FPU_data, remove(FPU2015))
###################
# plot(FPU_data, col="red")
plot(CRD, add=T, lwd = .2)
par(mar=c(1,1,1,1))
plot(CRD)
plot(FPU_data, add=T)
plot(CRD, add=T)

CRD_sf = st_as_sf(CRD)

plot(CRD_sf)
plot(FPU_data, add=T)
#head(FPU_data)
#head(CRD)
#head(CRD_sf)
#head(FPU_data)
test_FPU <-FPU_data
test_CRDsf <-CRD_sf
test_FPU$bbox
#test_FPU.bbox
#bbox(test_FPU) <- bbox(test_CRDsf)
#bbox(test_CRDsf)
#bbox(test_FPU)
#ls()
bbox(CRD)
#bbox(FPU_data)
#head(bbox(test_FPU))
#head(FPU_data)
#class(FPU_data)
#class(CRD)
FPU_data_sp <- as(FPU_data, 'Spatial')
class(FPU_data_sp)
##plot(FPU_data_sp)
##plot(CRD)
##plot(FPU_data_sp, add=T)
##plot(CRD)
##plot(FPU_data_sp, add=T, col="red")
plot(CRD)
plot(FPU_data_sp, add=T, border = "red")

#writeOGR(CRD, dsn = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/CRD_SP", "CRD_spatial", driver="ESRI Shapefile")

