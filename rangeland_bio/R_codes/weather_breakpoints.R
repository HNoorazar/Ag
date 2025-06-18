### Example from https://cran.r-project.org/web/packages/strucchange/strucchange.pdf

library(data.table)
library(strucchange)
library(dplyr)
library(ggmap) # loads ggplot2

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
if (dir.exists(file.path(out_dir)) == F) {dir.create(path = file.path(out_dir), recursive = T)}

###################################################################
###################################################################
####
#### 3. Read Files ------------------------------------------------

weather <- data.table(read.csv(paste0(reOrganized_dir, "bpszone_annual_tempPrecip_byHN.csv")))

# weather[, breakpoint_count := NA_integer_]
# weather[, breakpoint_years := NA_character_]
###################################################################
###################################################################
####
#### 4. Operate ------------------------------------------------------
FIDs = unique(weather$fid)

a_fid = FIDs[1]
fid_data = weather %>%
          filter(fid == a_fid)

# fs_fid <- Fstats(fid_data$precip_mm ~ 1)
# breaks_fid <- breakpoints(fs_fid)
# lines(breaks_fid)

rm(a_fid, fid_data)

## Add new columns to datatable to count number of 
## breakpoints and which years they occured.
##########
##########   precip
##########

break_points_dt_precip <- data.table(fid = FIDs,                                      # integer column
                                     precip_breakpoint_count = numeric(length(FIDs)), # numeric column
                                     precip_breakpoint_years = character(length(FIDs))# character column
                                     )
for (a_fid in FIDs){
  fid_data = weather %>%
            filter(fid == a_fid)
  breaks_fid <- breakpoints(fid_data$precip_mm ~ 1)

  breakpoint_positions = breaks_fid$breakpoints

  if (!is.na(breaks_fid$breakpoints[1])){
    precip_breakpoint_years = fid_data$year[breakpoint_positions]
    precip_breakpoint_years_str = paste0(precip_breakpoint_years, collapse="_")

    break_points_dt_precip[fid == a_fid, precip_breakpoint_count := length(breakpoint_positions)]
    break_points_dt_precip[fid == a_fid, precip_breakpoint_years := precip_breakpoint_years_str]
  } else{
    break_points_dt_precip[fid == a_fid, precip_breakpoint_count := 0]
    break_points_dt_precip[fid == a_fid, precip_breakpoint_years := NA]
  }
 
}




##########
##########   temps
##########
break_points_dt_temp <- data.table(fid = FIDs,                               # integer column
                                   temp_breakpoint_count = numeric(length(FIDs)), # numeric column
                                   temp_breakpoint_years = character(length(FIDs))# character column
                                   )
for (a_fid in FIDs){
  fid_data = weather %>%
            filter(fid == a_fid)
  breaks_fid <- breakpoints(fid_data$avg_of_dailyAvgTemp_C ~ 1)

  breakpoint_positions = breaks_fid$breakpoints

  if (!is.na(breaks_fid$breakpoints[1])){
    temp_breakpoint_years = fid_data$year[breakpoint_positions]
    temp_breakpoint_years_str = paste0(temp_breakpoint_years, collapse="_")

    break_points_dt_temp[fid == a_fid, temp_breakpoint_count := length(breakpoint_positions)]
    break_points_dt_temp[fid == a_fid, temp_breakpoint_years := temp_breakpoint_years_str]
  } else{
    break_points_dt_temp[fid == a_fid, temp_breakpoint_count := 0]
    break_points_dt_temp[fid == a_fid, temp_breakpoint_years := NA]
  }
 
}


break_points_dt <- merge(x=break_points_dt_temp, y=break_points_dt_precip, by=c("fid"), all=TRUE)
write.csv(break_points_dt, file = paste0(out_dir, "weather_break_points.csv"), row.names = FALSE)

