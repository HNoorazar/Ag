library(ggmap)
library(rgdal)
library(tmap)
library(magrittr)
library(sf)
library(raster)

ds = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/UScounties"
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

plot(st_geometry(tt_UScounties), col=sf.colors(12, categorical=T), border = 'black', axes=F, lwd= .01)
plot(st_geometry(the_shape), border = 'red', axes=F, add=T, lwd= .01)
##################################################################################################
plot(st_geometry(tt_UScounties), col = sf.colors(12, categorical = TRUE), border='grey', axes=F)
mapview(tt_UScounties, col.regions = sf.colors(10))

##################################################################################################
# https://r-spatial.github.io/sf/articles/sf5.html

plot(st_geometry(the_shape), col = sf.colors(12, categorical = TRUE), border = 'grey', axes = TRUE)
mapview(the_shape, col.regions = sf.colors(10))
##################################################################################################
tmap_mode("view")
tm_shape(the_shape) + tm_fill("USAFPU", palette = sf.colors(5))
##################################################################################################

extractCoords <- function(sp.df)
{
    results <- list()
    for(i in 1:length(sp.df@polygons[[1]]@Polygons))
    {
        results[[i]] <- sp.df@polygons[[1]]@Polygons[[i]]@coords
    }
    results <- Reduce(rbind, results)
    results
}