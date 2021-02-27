######################### Detect how many local grids are in historical that are NOT in future

data_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/00_databases/usa/"
analog_param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"

bad_CMPOP <- data.table(readRDS(paste0(data_dir, "CMPOP_location_ddd_accDDinC.rds")))

biofix_param <- data.table(read.csv(paste0(analog_param_dir, "biofix_param_hi.csv"), 
                                    header=T, as.is=T))

Min_fips <- data.table(read.csv(paste0(analog_param_dir, "Min_fips_st_county_location.csv"), 
                       header=T, as.is=T))

good_CMPOP_local_state <- merge(good_CMPOP_local, Min_fips)

local_grids <- data.table(read.csv(paste0(analog_param_dir, "local_county_fips.csv"), 
                          header=T, as.is=T))

local_grids_not_in_future <- good_CMPOP_local_state %>% 
                             filter(!(good_CMPOP_local_state$location %in% local_grids$location))
local_grids_not_in_future <- subset(local_grids_not_in_future, 
	                                 select=c("location", "st_county", "fips"))
local_grids_not_in_future <- unique(local_grids_not_in_future)

write.table(local_grids_not_in_future, 
            file = "/Users/hn/Documents/GitHub/Kirti/local_grids_not_in_future.csv", 
            row.names=FALSE, na="", col.names=FALSE, sep=",")

