### Example from https://cran.r-project.org/web/packages/strucchange/strucchange.pdf
data("Nile")
plot(Nile)

## F statistics indicate one breakpoint
fs_nile <- Fstats(Nile ~ 1)
plot(fs_nile)
bp_fstat <- breakpoints(fs_nile)
lines(breakpoints(fs_nile))


## or
bp_nile <- breakpoints(Nile ~ 1)
summary(bp_nile)
## the BIC also chooses one breakpoint
plot(bp_nile)
breakpoints(bp_nile)

## fit null hypothesis model and model with 1 breakpoint
fm0 <- lm(Nile ~ 1)
fm1 <- lm(Nile ~ breakfactor(bp_nile, breaks = 1))

colors <- c("dodgerblue", "firebrick", "darkorange", "seagreen")
plot(Nile, col="black", lwd=2)
lines(ts(fitted(fm0), start = 1871), col = "orange", lwd = 4)
lines(ts(fitted(fm1), start = 1871), col = "dodgerblue", lwd = 4)
lines(bp_nile, col="red", lwd=2)


###################################################################
###################################################################
####
#### 1. Load packages ---------------------------------------------
rm(list=ls())

library(data.table)
library(strucchange)
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

# anpp[, breakpoint_count := NA_integer_]
# anpp[, breakpoint_years := NA_character_]
###################################################################
###################################################################
####
#### 4. Operate ------------------------------------------------------
FIDs = unique(anpp$fid)

a_fid = FIDs[1]
fid_npp = anpp %>%
          filter(fid == a_fid)

# fs_fid <- Fstats(fid_npp$mean_lb_per_acr ~ 1)
# breaks_fid <- breakpoints(fs_fid)
# lines(breaks_fid)

rm(a_fid, fid_npp)

## Add new columns to datatable to count number of 
## breakpoints and which years they occured.

break_points_dt <- data.table(fid = FIDs,                               # integer column
                              breakpoint_count = numeric(length(FIDs)), # numeric column
                              breakpoint_years = character(length(FIDs))# character column
                              )
for (a_fid in FIDs){
  fid_npp = anpp %>%
            filter(fid == a_fid)
  breaks_fid <- breakpoints(fid_npp$mean_lb_per_acr ~ 1)

  breakpoint_positions = breaks_fid$breakpoints

  if (!is.na(breaks_fid$breakpoints[1])){
    breakpoint_years = fid_npp$year[breakpoint_positions]
    breakpoint_years_str = paste0(breakpoint_years, collapse="_")

    break_points_dt[fid == a_fid, breakpoint_count := length(breakpoint_positions)]
    break_points_dt[fid == a_fid, breakpoint_years := breakpoint_years_str]
  } else{
    break_points_dt[fid == a_fid, breakpoint_count := 0]
    break_points_dt[fid == a_fid, breakpoint_years := NA]
  }
 
}


write.csv(break_points_dt, 
          file = paste0(out_dir, "ANPP_break_points.csv"), row.names = FALSE)

