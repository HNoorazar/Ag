#
# GTWR is not doble in Python. Or it seems so!!!
# Jan 29, 2025. It seems we need to do regression and then
# find trends. We cannot just use time as X and find Trend
# The GWTR package in R is using the following paper:
# Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices.
# and here we are.
#
library(GWmodel)
library(tmap)  # For visualization, optional

library(foreign) # to read .sav files
library(haven) # to read .sav files
library(spdep)
library(data.table)
library(dplyr)
library(rgdal)
library(sf)
library(sp)

# First library("remotes") then the following line
if (!require("rspat")) remotes::install_github('rspatial/rspat')
library(rspat)

library(RColorBrewer)
grps <- 10

dir_base_ = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/"
reOrganized_dir = paste0(dir_base_, "reOrganized/")

# Strange. csv file is lighter than the sav file
# Strange: R cannot read CSV file (memory issue) but reads the sav file
# fid_queen_neib_rowSTD = read_sav(paste0(reOrganized_dir, "fid_contiguity_Queen_neighbors_rowSTD_for_R.sav"))
# col_names <- colnames(fid_queen_neib_rowSTD)
## fid_queen_neib_rowSTD = read.spss(paste0(reOrganized_dir, "fid_contiguity_Queen_neighbors_rowSTD_for_R.sav"), to.data.frame=TRUE)
## fid_queen_neib_rowSTD = data.table(read.csv(paste0(reOrganized_dir, "fid_contiguity_Queen_neighbors_rowSTD_for_R.sav")))
# for ( col in 1:ncol(fid_queen_neib_rowSTD)){
#     colnames(fid_queen_neib_rowSTD)[col] <-  sub("var_", "", colnames(fid_queen_neib_rowSTD)[col])
# }

########################################
fid_queen_neib_rowSTD = data.table(read.csv(paste0(reOrganized_dir, "WA_fid_Queen_neighbors_rowSTD.csv")))
## column names have extra "X" in them!
for ( col in 1:ncol(fid_queen_neib_rowSTD)){
    colnames(fid_queen_neib_rowSTD)[col] <-  sub("X", "", colnames(fid_queen_neib_rowSTD)[col])
}
########################################
bpszone_ANPP_no2012 = data.table(read.csv(paste0(reOrganized_dir, "bpszone_ANPP_no2012_for_R.csv")))
head(bpszone_ANPP_no2012, 2)
#
# Lets just focus and develop on one state
#
# Albers_Rangeland <- read_sf(paste0(reOrganized_dir, 
#                                    "Albers_BioRangeland_Min_Ehsan/Albers_BioRangeland_Min_Ehsan.shp"))
# WA_SF <- Albers_Rangeland %>% filter(SATAE_MAX == "Washington")
# WA_FIDs = WA_SF$MinStatsID

# pick the rows of WA The following would be wrong. 
# As some FIDs in WA will have neighbor in other states. So, I have done WA separately.
# fid_queen_neib_rowSTD_WA <- fid_queen_neib_rowSTD %>% filter(fid %in% WA_FIDs)
## pick the columns of WA
# WA_FIDs_str = copy(WA_FIDs)
# WA_FIDs_str <- as.character(WA_FIDs_str)
# need_ <- c("fid", WA_FIDs_str)
# fid_queen_neib_rowSTD_WA <- fid_queen_neib_rowSTD_WA[need_]

WA_SF <- read_sf(paste0(reOrganized_dir, "Albers_BioRangeland_Min_Ehsan_WA/Albers_BioRangeland_Min_Ehsan_WA.shp"))
WA_FIDs = WA_SF$fid

# I have checked in Python: read_sav_write_4_R.ipynb
# in WA all locations have 39 years; no missing value

WA_ANPP_no2012 <- bpszone_ANPP_no2012 %>%
                  filter(fid %in% WA_FIDs)

head(fid_queen_neib_rowSTD, 2)
head(WA_ANPP_no2012, 2)
head(WA_SF, 2)


 WA_ANPP_no2012$groupveg <- NULL
# Outer join
WA_SF_outer = merge(x = WA_SF, y = WA_ANPP_no2012, by = "fid", all = TRUE)
# WA_SF_outerLeft = merge(x = WA_SF, y = WA_ANPP_no2012, by = "fid", all.x = TRUE)
# all.equal(WA_SF_outer, WA_SF_outerLeft)

# bandwidth selection
npp_yr_formula <- mean_lb_per_acr ~ year

## sort so that distance matrix and SF have the same order
# OR ... do we have to expand the distance matrix so it has the same dimension as SF?
WA_SF_outer <- WA_SF_outer[with(WA_SF_outer, order(fid, year)), ]

# check if the dmatrix is sorted by fid in increasing fashion
dmatrix_fid_list = fid_queen_neib_rowSTD$fid

# everything seems to be positive?
a = dmatrix_fid_list[(2:length(dmatrix_fid_list))] - dmatrix_fid_list[(1:length(dmatrix_fid_list)-1)]


colnames_ = strtoi(colnames(fid_queen_neib_rowSTD)[2:length(fid_queen_neib_rowSTD)])
# everything seems to be positive?
b = colnames_[(2:length(colnames_))] - colnames_[(1:length(colnames_)-1)]
# rows and columns are identical?
a-b
# dmatrix_fid_list[seq(length=n, from=length(dmatrix_fid_list), by=-1)] 
distance_M = copy(fid_queen_neib_rowSTD)
distance_M <- subset(distance_M, select = -c(fid))
distance_M <- as.matrix(distance_M)
distance_M <- unname(distance_M)

library(Matrix)
k = nrow(WA_SF_outer) / nrow(distance_M)
block_diag = bdiag(replicate(k, distance_M, simplify = FALSE))

bw.gwr(formula=npp_yr_formula, data=WA_SF_outer, 
       approach="CV", kernel="bisquare", adaptive=FALSE, p=2, theta=0, longlat=F, dMat=block_diag,
       parallel.method=F, parallel.arg=NULL)

bw.gtwr(formula=npp_yr_formula, data=WA_SF_outer, 
        approach="CV", kernel="bisquare", adaptive=FALSE, p=2, theta=0, longlat=F, st.dMat=???,
        parallel.method=F, parallel.arg=NULL)

bw.gtwr(formula=npp_yr_formula, data=WA_SF_outer, obs.tv=WA_SF_outer$year, approach="CV", kernel="bisquare", adaptive=FALSE, p=2, theta=0, longlat=F)

# different (?) bandwidth from Scotts link:
# https://rspatial.org/analysis/6-local_regression.html
WA_SF_outer_df = data.frame(WA_SF_outer)
WA_SF_outer_df <- subset(WA_SF_outer_df, select = -c(geometry, hucsgree_4, value, bps_code, bps_model, 
                                                     state_1, state_2, pixel_count, area_sqMeter, bps_name))

# repeating whatever they did in the link above
alb <- "+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +datum=WGS84 +units=m"
WA_sp <- vect(WA_SF_outer_df, c("long", "lat"), crs="+proj=longlat +datum=WGS84")
WA_spt <- project(WA_sp, alb)
# The link above says BW is 4340.569 which is in a different projection. Lets try it with lat long
bw_spt <- gwr.sel(mean_lb_per_acr ~ year, data=as.data.frame(WA_spt), coords=geom(WA_spt)[,c("x", "y")]) # bw_spt = 4340.569 for WA.

# band width with lat long: bw_sp_lat_long = 0.04689746 for WA
bw_sp_lat_long <- gwr.sel(mean_lb_per_acr ~ year, data=as.data.frame(WA_sp), coords=geom(WA_sp)[,c("x", "y")])


# https://www.rdocumentation.org/packages/GWmodel/versions/2.2-2/topics/gtwr
a_model <- gtwr(formula=npp_yr_formula, data=WA_SF_outer, obs.tv=WA_SF_outer$year)

# the weight matrix in Geographical and Temporal Weighted Regression (GTWR)
# is sorted by year first, i.e. all locations in a given year all together.
# lets see if that can change things
WA_SF_outer_TimeSort <- copy(WA_SF_outer)
WA_SF_outer_TimeSort <- WA_SF_outer_TimeSort[with(WA_SF_outer_TimeSort, order(year, fid)), ]
a_model_TimeSort <- gtwr(formula=npp_yr_formula, data=WA_SF_outer_TimeSort, obs.tv=WA_SF_outer_TimeSort$year)
