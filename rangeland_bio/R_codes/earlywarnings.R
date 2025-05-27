install.packages("earlywarnings")

###################################################################
###################################################################
####
#### 1. Load packages ---------------------------------------------
rm(list=ls())

library(data.table)
library(earlywarnings)
library(dplyr)
library(ggmap) # loads ggplot2

###################################################################
###################################################################
####
#### 2. Data Directories ------------------------------------------
database <- "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/"
reOrganized_dir = paste0(database, "reOrganized/") 

out_dir <- paste0(database, "breakpoints/")
if (dir.exists(file.path(out_dir)) == F) {
    dir.create(path = file.path(out_dir), recursive = T)
}

###################################################################
###################################################################
####
#### 3. Read Files ------------------------------------------------

anpp <- data.table(read.csv(paste0(reOrganized_dir, "bpszone_ANPP_no2012.csv")))
anpp[, c("pixel_count", "groupveg", "area_sqMeter") := NULL] # get rid of unwanted columns
setcolorder(anpp, c("fid", "year"))
anpp[, mean_lb_per_acr := as.numeric(mean_lb_per_acr)]
# anpp[, breakpoint_count := NA_integer_]
# anpp[, breakpoint_years := NA_character_]
###################################################################
###################################################################
####
#### 4. Operate ------------------------------------------------------
FIDs = unique(anpp$fid)

setorder(anpp, fid, year)

# anpp_clean <- anpp[, .(value = mean(mean_lb_per_acr, na.rm = TRUE)), by = .(fid, year)]
# setorder(anpp_clean, fid, year)

# Convert to list of ts

ts_list <- anpp[, {
  ts_obj <- ts(mean_lb_per_acr, start = min(year), frequency = 1)  # Annual time series
  list(ts = list(ts_obj))
}, by = fid]


