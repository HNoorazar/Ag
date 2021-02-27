library(ggmap)
library(rgdal)
library(tmap)
library(magrittr)
library(sf)


# read the damn thing.
FPU <- readOGR(dsn = "./IMPACT_FPU_Map/", 
	           layer = "fpu2015_polygons_v3_multipart_polygons")

# subset the USA part
tofind <- c("_USA")
USA_FPU <- FPU[grep(paste(tofind, collapse = "|"), FPU$FPU2015), ]
spplot(USA_FPU, z="USAFPU")

tm_shape(USA_FPU, scale=10) + 
tm_fill("USAFPU", 
	    style="fixed",
	    labels=c("Alaska", "Arkansas", 
	    	     "California", "Colorado", 
	    	     "Columbia", "Great Basin",
	    	     "Great Lakes", "Hawaii",
	    	     "Mississippi", "Missouri", "Northeast", "Ohio",
	    	     "Red Winnipeg", "Rio Grande", "Southeast", "Western Gulf Mexico")
	    ) +
tm_borders("black") +
tm_legend(outside = TRUE) +
tm_layout(frame = FALSE,
	      legend.text.size = .5, 
	      legend.title.size=0.1) 
#######
map <-ggplot() + 
      geom_polygon(data=USA_FPU, aes(x=long, y=lat, group=group), colour = "black", fill = NA) + 
      theme_void()


##################################################################################################
# https://r-spatial.github.io/sf/articles/sf5.html

data_dir = "/Users/hn/Documents/GitHub/Kirti/wareHouse/merge_irrigation_Monday/IMPACT_FPU_Map/"
layer_name = "fpu2015_polygons_v3_multipart_polygons"
tt <- read_sf(dsn=path.expand(data_dir), layer = layer_name, quiet = TRUE)
tofind <- c("_USA")
tt <- tt[grep(paste(tofind, collapse = "|"), tt$FPU2015), ]
tt_sub = within(tt, remove(FPU2015))
plot(st_geometry(tt_sub), col = sf.colors(12, categorical = TRUE), border = 'grey', axes = TRUE)

mapview(tt_sub, col.regions = sf.colors(10))
########
tmap_mode("view")
tm_shape(tt_sub) + tm_fill("USAFPU", palette = sf.colors(5))
##################################################################################################

data_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/IMPACT_FPU_Map/"
layer_name = "fpu2015_polygons_v3_multipart_polygons"
tt <- read_sf(dsn=path.expand(data_dir), layer = layer_name, quiet = TRUE)
tofind <- c("_USA")
tt <- tt[grep(paste(tofind, collapse = "|"), tt$FPU2015), ]

ggplot(tt) + geom_sf(aes(fill = USAFPU), lwd = .1) 


pnts$region <- apply(pnts, 1, function(row) {  
   # transformation to palnar is required, since sf library assumes planar projection 
   tt_pl <- st_transform(tt, 2163)   
   coords <- as.data.frame(matrix(row, nrow = 1, dimnames = list("", c("x", "y"))))   
   pnt_sf <- st_transform(st_sfc(st_point(row),crs = 4326), 2163)
   # st_intersects with sparse = FALSE returns a logical matrix
   # with rows corresponds to argument 1 (points) and 
   # columns to argument 2 (polygons)

   tt_pl[which(st_intersects(pnt_sf, tt_pl, sparse = FALSE)), ]$NAME_1 
})

