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
library(sf)

library(rgdal)
library(dplyr)
library(sp)
library(sf)

# First library("remotes") then the following line
if (!require("rspat")) remotes::install_github('rspatial/rspat')
library(rspat)

library(RColorBrewer)
grps <- 10

dir_base_ = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/"
reOrganized_dir = paste0(dir_base_, "reOrganized/")

# Strange. csv file is lighter than the sav file
# Strange: R cannot read CSV file (memory issue) but reads the sav file
fid_queen_neib_rowSTD = read_sav(paste0(reOrganized_dir, "fid_contiguity_Queen_neighbors_rowSTD_for_R.sav"))
col_names <- colnames(fid_queen_neib_rowSTD)
# fid_queen_neib_rowSTD = read.spss(paste0(reOrganized_dir, "fid_contiguity_Queen_neighbors_rowSTD_for_R.sav"), to.data.frame=TRUE)
# fid_queen_neib_rowSTD = data.table(read.csv(paste0(reOrganized_dir, "fid_contiguity_Queen_neighbors_rowSTD_for_R.sav")))

for ( col in 1:ncol(fid_queen_neib_rowSTD)){
    colnames(fid_queen_neib_rowSTD)[col] <-  sub("var_", "", colnames(fid_queen_neib_rowSTD)[col])
}

#
# Lets just focus and develop on one state
#
Albers_Rangeland <- read_sf(paste0(reOrganized_dir, 
                                   "Albers_BioRangeland_Min_Ehsan/Albers_BioRangeland_Min_Ehsan.shp"))

WA_SF <- Albers_Rangeland %>% filter(SATAE_MAX == "Washington")
WA_FIDs = WA_SF$MinStatsID

# pick the rows of WA
fid_queen_neib_rowSTD_WA <- fid_queen_neib_rowSTD %>% filter(fid %in% WA_FIDs)

# pick the columns of WA
WA_FIDs_str = copy(WA_FIDs)
WA_FIDs_str <- as.character(WA_FIDs_str)
need_ <- c("fid", WA_FIDs_str)
fid_queen_neib_rowSTD_WA <- fid_queen_neib_rowSTD_WA[need_]



npp_yr_formula <- npp ~ year


bpszone_ANPP_no2012 = data.table(read.csv(paste0(reOrganized_dir, "bpszone_ANPP_no2012_for_R.csv")))
head(bpszone_ANPP_no2012, 2)

# I have checked in Python: read_sav_write_4_R.ipynb
# in WA all locations have 39 years; no missing value

WA_ANPP_no2012 <- bpszone_ANPP_no2012 %>% 
                  filter()


