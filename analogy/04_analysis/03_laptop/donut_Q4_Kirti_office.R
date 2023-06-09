######################################################################
rm(list=ls())

library(lubridate)
library(purrr)
library(maps)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(gridExtra)
library(grid)
library(gtable)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/analogy/core_analog.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/analogy/core_analog_plots.R"
source(source_path_1)
source(source_path_2)

library(swfscMisc) # has na.count(.) in it

options(digit=9)
options(digits=9)

######################################################################
####
####           global Files
####
######################################################################
param_dir <- "/Users/hn/Documents/GitHub/Kirti/analogy/parameters/"
new_param_dir <- paste0(param_dir, "new_params_for_analog/")

min_grids <- 5
hist_grid_count <- read.csv(paste0(new_param_dir, "cod_moth_hist_grid_count_within_counties.csv"),
                            header=T, as.is=T)  %>%                            
                            filter(grid_count >= min_grids) %>%
                            data.table()

hist_grid_count$st_county <- paste(hist_grid_count$state, hist_grid_count$county, sep="_" )
hist_grid_count <- within(hist_grid_count, remove(state, county))

####################################################################################

f_loc_fips_st_cnty <- "local_county_fips.csv"
h_loc_fips_st_cnty <- "all_us_1300_county_fips_locations.csv"

f_loc_fips_st_cnty <- data.table(read.csv(paste0(param_dir, f_loc_fips_st_cnty), header=T, sep=",", as.is=T))
h_loc_fips_st_cnty <- data.table(read.csv(paste0(param_dir, h_loc_fips_st_cnty), header=T, sep=",", as.is=T))

f_loc_fips_st_cnty <- get_286_locs(f_loc_fips_st_cnty, h_loc_fips_st_cnty)

local_fips <- unique(f_loc_fips_st_cnty$fips)
 
# f_grid_count <- f_loc_fips_st_cnty %>%
#                 group_by(fips, st_county) %>%
#                 transmute(grid_count = n_distinct(location)) %>%
#                 unique() %>%
#                 data.table()

######## Filter h_loc_fips_st_cnty, so, it only includes
######## counties with minimum number of grids in it. 
######## Doing so in this step may prevent slipping 
######## not qualified counties in calculations:
######## Doing so, takes us from 1293 locations to 1175. (118 locations are gone)

h_loc_fips_st_cnty <- h_loc_fips_st_cnty %>% filter(fips %in% hist_grid_count$fips)

######################################################
# 
# To extract names of counties associated with each unique fips
# 

Min_fips_st_county <- data.table(read.csv(paste0(param_dir, "Min_fips_st_county_location.csv"), header=T, sep=",", as.is=T)) 
Min_fips_st_county <- within(Min_fips_st_county, remove(location))
Min_fips_st_county <- unique(Min_fips_st_county)

######################################################################
####
####         Set up data directories
####
######################################################################
county_averages = FALSE

if (county_averages == FALSE){
    main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/"
   } else {
    main_in <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/county_averages/"
}

data_sub_dirs <- c("w_precip_rcp45/", "w_precip_rcp85/")

#___________________________________________________________________________________________

raw_feat_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/analogs/00_databases/"

if (county_averages==FALSE){
  feat_rcp45 <- data.table(readRDS(paste0(raw_feat_dir, "CDD_precip_rcp45.rds")))
  feat_rcp85 <- data.table(readRDS(paste0(raw_feat_dir, "CDD_precip_rcp85.rds")))
  feat_hist <- data.table(readRDS(paste0(raw_feat_dir, "hist_CDD_precip.rds")))

  feat_rcp45 <- merge(feat_rcp45, f_loc_fips_st_cnty, by="location", all.x=T)
  feat_rcp85 <- merge(feat_rcp85, f_loc_fips_st_cnty, by="location", all.x=T)
  feat_hist <- merge(feat_hist, h_loc_fips_st_cnty, by="location", all.x=T)

  # some of the historical counties has less than min_grid = 5 locations in them
  # so, they need to be gone. Also, 8 locations in future data are not present in
  # the hitotical data, we eliminated them as well, which causes generations of NAs
  # in above data. They need to be gone as well.
  feat_rcp45 <- na.omit(feat_rcp45)
  feat_rcp85 <- na.omit(feat_rcp85)
  feat_hist <- na.omit(feat_hist)
  } else {
    feat_rcp45 <- data.table(readRDS(paste0(raw_feat_dir, "cnty_avg_feat_45.rds")))
    feat_rcp85 <- data.table(readRDS(paste0(raw_feat_dir, "cnty_avg_feat_85.rds")))
    feat_hist <- data.table(readRDS(paste0(raw_feat_dir, "cnty_avg_feat_hist.rds")))
}

#___________________________________________________________________________________________
######################################################################

model_names <- c("BNU-ESM", "CanESM2", "GFDL-ESM2G", "CNRM-CM5", "bcc-csm1-1-m", "GFDL-ESM2M")
time_p <- c("2026_2050", "2051_2075", "2076_2095")
emissions <- c("rcp85", "rcp45") # 
sigma_bds <- c(1, 2) # 
# VL_quans = c(.25, .75)

sub_dir <- data_sub_dirs[1]
sigma_bd <- 1
target_fip <- 53021
model_n <- model_names[1]

for (sub_dir in data_sub_dirs){
  # emission <- substr(unlist(strsplit(sub_dir, "_"))[5], 1, 5)
  emission <- substr(unlist(strsplit(sub_dir, "_"))[3], 1, 5)
  # plot_name_extension <- paste0("_", strsplit(sub_dir, "_")[[1]][1], 
  #                               "_", strsplit(sub_dir, "_")[[1]][3], "_", emission)
  plot_name_extension <- paste0("_", 
                                strsplit(sub_dir, "_")[[1]][1], 
                                "_", strsplit(sub_dir, "_")[[1]][2], 
                                "_", emission)
  ######################################################################

  for (sigma_bd in sigma_bds){
    for (target_fip in local_fips){
      
      for(model_n in model_names){
        print(paste0(sigma_bd, ", ", sub_dir))
        data_dir <- paste0(main_in, sigma_bd, "_sigma/", sub_dir)
        #############################################################################
        #
        # read the data of the three time periods
        #
        analog_file_name_F1 <- paste("analog", model_n, emission, time_p[1], sep="_")
        analog_dat_F1 <- data.table(readRDS(paste0(data_dir, analog_file_name_F1, ".rds")))

        analog_file_name_F2 <- paste("analog", model_n, emission, time_p[2], sep="_")
        analog_dat_F2 <- data.table(readRDS(paste0(data_dir, analog_file_name_F2, ".rds")))

        analog_file_name_F3 <- paste("analog", model_n, emission, time_p[3], sep="_")
        analog_dat_F3 <- data.table(readRDS(paste0(data_dir, analog_file_name_F3, ".rds")))

        #############################################################################
        #
        #   Filter out counties with less than "min_grids" grids in them.
        #  *** Let us remove counties with historical #grids < min_grids from the map ***
        
        analog_dat_F1 <- analog_dat_F1 %>% 
                         filter(analog_NNs_county %in% hist_grid_count$fips) %>% data.table()
        analog_dat_F2 <- analog_dat_F2 %>% 
                         filter(analog_NNs_county %in% hist_grid_count$fips) %>% data.table()
        analog_dat_F3 <- analog_dat_F3 %>% 
                         filter(analog_NNs_county %in% hist_grid_count$fips) %>% data.table()
        
        #############################################################################
        #
        # filter the location of interest:
        #
        analog_dat_F1 <- analog_dat_F1 %>% filter(query_county == target_fip)
        analog_dat_F2 <- analog_dat_F2 %>% filter(query_county == target_fip)
        analog_dat_F3 <- analog_dat_F3 %>% filter(query_county == target_fip)
        
        # remove the stats of not_analog pieces
        analog_dat_F1 <- analog_dat_F1 %>% filter(analog_NNs_county != "no_analog")
        analog_dat_F2 <- analog_dat_F2 %>% filter(analog_NNs_county != "no_analog")
        analog_dat_F3 <- analog_dat_F3 %>% filter(analog_NNs_county != "no_analog")

        # Convert the fips to integers so they can be used for plotting:
        analog_dat_F1$analog_NNs_county <- as.integer(analog_dat_F1$analog_NNs_county)
        analog_dat_F1$query_county <- as.integer(analog_dat_F1$query_county)

        analog_dat_F2$analog_NNs_county <- as.integer(analog_dat_F2$analog_NNs_county)
        analog_dat_F2$query_county <- as.integer(analog_dat_F2$query_county)

        analog_dat_F3$analog_NNs_county <- as.integer(analog_dat_F3$analog_NNs_county)
        analog_dat_F3$query_county <- as.integer(analog_dat_F3$query_county)
        
        #############################################################################
        # 
        # Determine how many years are in future data frames, to be used
        # for standardization of maps and data needed for donuts.
        
        f_years_F1 = 1 + as.numeric(unlist(strsplit(time_p[1], "_")))[2] - 
                         as.numeric(unlist(strsplit(time_p[1], "_")))[1]

        f_years_F2 = 1 + as.numeric(unlist(strsplit(time_p[2], "_")))[2] - 
                         as.numeric(unlist(strsplit(time_p[2], "_")))[1]

        f_years_F3 = 1 + as.numeric(unlist(strsplit(time_p[3], "_")))[2] - 
                         as.numeric(unlist(strsplit(time_p[3], "_")))[1]
        #############################################################################
        #
        # *** Standardize the freq. using # of grids in each county (historical data) ***
        # hist_counties_with_more_10 has the fips and count of grids within each county.
        # But then we talked in her office and we changed the denominator to all possible
        # pairwise combinations, so, fractions are between 0 and 1.
        # This should be done just for maps.
        #
        analog_dat_F1_4_map_b <- standardize_by_all_pairs(analog_dt=analog_dat_F1, 
                                                          f_fips_dt=f_loc_fips_st_cnty, 
                                                          h_fips_dt=h_loc_fips_st_cnty, 
                                                          f_years=f_years_F1, h_years=37)
        
        analog_dat_F2_4_map_b <- standardize_by_all_pairs(analog_dt=analog_dat_F2, 
                                                          f_fips_dt=f_loc_fips_st_cnty, 
                                                          h_fips_dt=h_loc_fips_st_cnty, 
                                                          f_years=f_years_F2, h_years=37)
        
        analog_dat_F3_4_map_b <- standardize_by_all_pairs(analog_dt=analog_dat_F3, 
                                                          f_fips_dt=f_loc_fips_st_cnty, 
                                                          h_fips_dt=h_loc_fips_st_cnty, 
                                                          f_years=f_years_F3, h_years=37)
        
        most_similar_cnty_F1 <- analog_dat_F1_4_map_b[[2]]
        most_similar_cnty_F2 <- analog_dat_F2_4_map_b[[2]]
        most_similar_cnty_F3 <- analog_dat_F3_4_map_b[[2]]

        analog_dat_F1_4_map <- analog_dat_F1_4_map_b[[1]]
        analog_dat_F2_4_map <- analog_dat_F2_4_map_b[[1]]
        analog_dat_F3_4_map <- analog_dat_F3_4_map_b[[1]]
        # rm(analog_dat_F1_4_map_b, analog_dat_F2_4_map_b, analog_dat_F3_4_map_b)

        most_similar_cnty_F1_fip <- most_similar_cnty_F1
        most_similar_cnty_F2_fip <- most_similar_cnty_F2
        most_similar_cnty_F3_fip <- most_similar_cnty_F3
        #############################################################################
        #
        # produce data for geographical map
        #
        one_mod_map_info_F1 <- produce_dt_for_map(analog_dat_F1_4_map)
        one_mod_map_info_F2 <- produce_dt_for_map(analog_dat_F2_4_map)
        one_mod_map_info_F3 <- produce_dt_for_map(analog_dat_F3_4_map)

        # produce data for donuts
        one_mod_pie_info_F1 <- produce_dt_for_pie_Q4(analog_dt = analog_dat_F1, 
                                                     tgt_fip = target_fip,
                                                     hist_target_fip = most_similar_cnty_F1_fip,
                                                     f_fips = f_loc_fips_st_cnty, 
                                                     h_fips = h_loc_fips_st_cnty, 
                                                     f_years = f_years_F1, h_years=37)

        one_mod_pie_info_F2 <- produce_dt_for_pie_Q4(analog_dt = analog_dat_F2, 
                                                     tgt_fip = target_fip,
                                                     hist_target_fip = most_similar_cnty_F2_fip,
                                                     f_fips = f_loc_fips_st_cnty, 
                                                     h_fips = h_loc_fips_st_cnty, 
                                                     f_years = f_years_F2, h_years=37)

        one_mod_pie_info_F3 <- produce_dt_for_pie_Q4(analog_dt = analog_dat_F3, 
                                                     tgt_fip = target_fip,
                                                     hist_target_fip = most_similar_cnty_F3_fip,
                                                     f_fips = f_loc_fips_st_cnty, 
                                                     h_fips = h_loc_fips_st_cnty, 
                                                     f_years= f_years_F3, h_years=37)

        # Extract name of county of interest for putting in the plots:
        target_cnty_name <- Min_fips_st_county$st_county[Min_fips_st_county$fips==target_fip]
        target_cnty_name <- paste(unlist(strsplit(target_cnty_name, "_"))[2], 
                            unlist(strsplit(target_cnty_name, "_"))[1], sep= ", ")

        if (emission=="rcp45"){ttl_emiss = "RCP 4.5"} else { ttl_emiss = "RCP 8.5"}

        titlem_F1 <- paste0(target_cnty_name, 
                            " (", 
                            paste(unlist(strsplit(time_p[1], "_"))[1], 
                                  unlist(strsplit(time_p[1], "_"))[2], sep="-"),
                            ", ", model_n, ", ", ttl_emiss, ")" )

        titlem_F2 <- paste0(target_cnty_name, 
                            " (", 
                            paste(unlist(strsplit(time_p[2], "_"))[1], 
                                  unlist(strsplit(time_p[2], "_"))[2], sep="-"),
                            ", ", model_n, ", ", ttl_emiss, ")" )

        titlem_F3 <- paste0(target_cnty_name, 
                            " (", 
                            paste(unlist(strsplit(time_p[3], "_"))[1], 
                                  unlist(strsplit(time_p[3], "_"))[2], sep="-"),
                            ", ", model_n, ", ", ttl_emiss, ")" )
        
        analog_name_F1 <- Min_fips_st_county$st_county[Min_fips_st_county$fips==most_similar_cnty_F1_fip]
        analog_name_F1 <- paste(unlist(strsplit(analog_name_F1, "_"))[2], 
                                           unlist(strsplit(analog_name_F1, "_"))[1], sep= ", ")
        
        analog_name_F2 <- Min_fips_st_county$st_county[Min_fips_st_county$fips==most_similar_cnty_F2_fip]
        analog_name_F2 <- paste(unlist(strsplit(analog_name_F2, "_"))[2], 
                                           unlist(strsplit(analog_name_F2, "_"))[1], sep= ", ")

        analog_name_F3 <- Min_fips_st_county$st_county[Min_fips_st_county$fips==most_similar_cnty_F3_fip]
        analog_name_F3 <- paste(unlist(strsplit(analog_name_F3, "_"))[2], 
                                           unlist(strsplit(analog_name_F3, "_"))[1], sep= ", ")

        # Plot the donuts
        assign(x = paste0("pie_", gsub("-", "_", model_n), "_F1"), 
               value = {plot_the_pie(DT = one_mod_pie_info_F1, 
                                     titl = titlem_F1, 
                                     subtitle = analog_name_F1)})
        
        assign(x = paste0("pie_", gsub("-", "_", model_n), "_F2"), 
               value = {plot_the_pie(DT = one_mod_pie_info_F2, 
                                     titl = titlem_F2, 
                                     subtitle = analog_name_F2)})
        
        assign(x = paste0("pie_", gsub("-", "_", model_n), "_F3"), 
                          value = {plot_the_pie(DT = one_mod_pie_info_F3, 
                                                titl = titlem_F3, 
                                                subtitle = analog_name_F3)})
        #________________________________________________________________________________
        #
        # Set up right data for heat map plots
        #
        if (emission == "rcp45"){ curr_future_feat <- feat_rcp45
          } else { 
          curr_future_feat <- feat_rcp85 
        }
        
        curr_future_feat <- curr_future_feat %>%
                            filter(model == model_n & fips == target_fip) %>% data.table()
        
        curr_future_feat_F1 <- curr_future_feat %>%
                               filter(year >= 2025 & year <= 2050 ) %>% data.table()

        curr_future_feat_F2 <- curr_future_feat %>%
                               filter(year >= 2051 & year <= 2075 ) %>% data.table()
        
        curr_future_feat_F3 <- curr_future_feat %>%
                               filter(year >= 2076) %>% data.table()

        curr_feat_hist_1 <- feat_hist %>% filter(fips == most_similar_cnty_F1_fip) %>% data.table()
        curr_feat_hist_2 <- feat_hist %>% filter(fips == most_similar_cnty_F2_fip) %>% data.table()
        curr_feat_hist_3 <- feat_hist %>% filter(fips == most_similar_cnty_F3_fip) %>% data.table()

        contour_dt_1 <- rbind(curr_feat_hist_1, curr_future_feat_F1)
        contour_dt_2 <- rbind(curr_feat_hist_2, curr_future_feat_F2)
        contour_dt_3 <- rbind(curr_feat_hist_3, curr_future_feat_F3)
        #
        # Plot the contours
        #
        # assign(x = paste0("con_", gsub("-", "_", model_n), "_F1"), 
        #        value = {plot_the_contour_one_filling(data_dt = contour_dt_1, 
        #                                              con_title = titlem_F1, 
        #                                              con_subT = analog_name_F1 # , v_line_quantiles=VL_quans
        #                                              )})
        
        # assign(x = paste0("con_", gsub("-", "_", model_n), "_F2"), 
        #        value = {plot_the_contour_one_filling(data_dt = contour_dt_2, 
        #                                              con_title = titlem_F1, 
        #                                              con_subT = analog_name_F1 # , v_line_quantiles=VL_quans
        #                                              )})

        # assign(x = paste0("con_", gsub("-", "_", model_n), "_F3"), 
        #        value = {plot_the_contour_one_filling(data_dt = contour_dt_3, 
        #                                              con_title = titlem_F1, 
        #                                              con_subT = analog_name_F1 #, v_line_quantiles=VL_quans
        #                                              )})

        assign(x = paste0("con_", gsub("-", "_", model_n), "_F1"), 
               value = {plot_the_contour(data_dt = contour_dt_1, 
                                         con_title = titlem_F1, 
                                         con_subT = analog_name_F1 # , v_line_quantiles=VL_quans
                                         )})
        
        assign(x = paste0("con_", gsub("-", "_", model_n), "_F2"), 
               value = {plot_the_contour(data_dt = contour_dt_2, 
                                         con_title = titlem_F1, 
                                         con_subT = analog_name_F1 # , v_line_quantiles=VL_quans
                                         )})

        assign(x = paste0("con_", gsub("-", "_", model_n), "_F3"), 
               value = {plot_the_contour(data_dt = contour_dt_3, 
                                         con_title = titlem_F1, 
                                         con_subT = analog_name_F1 #, v_line_quantiles=VL_quans
                                         )})
        
        # Plot the 1D densities
        #
        # assign(x = paste0("den_", gsub("-", "_", model_n), "_F1"), 
        #        value = {plot_the_1D_densities(data_dt = contour_dt_1, 
        #                                       dens_T = titlem_F1, 
        #                                       subT = analog_name_F1 
        #                                       # , v_line_quantiles=VL_quans
        #                                       )})
        
        # assign(x = paste0("den_", gsub("-", "_", model_n), "_F2"), 
        #        value = {plot_the_1D_densities(data_dt = contour_dt_2, 
        #                                       dens_T = titlem_F2, 
        #                                       subT = analog_name_F2 
        #                                       # , v_line_quantiles=VL_quans
        #                                       )})

        # assign(x = paste0("den_", gsub("-", "_", model_n), "_F3"), 
        #        value = {plot_the_1D_densities(data_dt = contour_dt_3, 
        #                                       dens_T = titlem_F3, 
        #                                       subT = analog_name_F3 
        #                                       # , v_line_quantiles=VL_quans
        #                                       )})
        assign(x = paste0("con_marg_", gsub("-", "_", model_n), "_F1"), 
               value = {plot_the_margins(data_dt = contour_dt_1,
                                         contour_plot = get(paste0("con_", gsub("-", "_", model_n), "_F1"))
                                        )})
        
        assign(x = paste0("con_marg_", gsub("-", "_", model_n), "_F2"), 
               value = {plot_the_margins(data_dt = contour_dt_2, 
                                         contour_plot = get(paste0("con_", gsub("-", "_", model_n), "_F2"))
                                        )})

        assign(x = paste0("con_marg_", gsub("-", "_", model_n), "_F3"), 
               value = {plot_the_margins(data_dt = contour_dt_3, 
                                         contour_plot = get(paste0("con_", gsub("-", "_", model_n), "_F3"))
                                        )})

        # _______________________________________________________________________________
        # 
        # bind the goddamn donut and contour together
        #
        # assign(x = paste0("pie_con_", gsub("-", "_", model_n), "_F1"),
        #        value = ggarrange(plotlist = list(get(paste0("pie_", gsub("-", "_", model_n), "_F1")), 
        #                                          get(paste0("con_", gsub("-", "_", model_n), "_F1")))))

        # assign(x = paste0("pie_con_", gsub("-", "_", model_n), "_F2"),
        #        value = ggarrange(plotlist = list(get(paste0("pie_", gsub("-", "_", model_n), "_F2")), 
        #                                          get(paste0("con_", gsub("-", "_", model_n), "_F2")))))

        # assign(x = paste0("pie_con_", gsub("-", "_", model_n), "_F3"),
        #        value = ggarrange(plotlist = list(get(paste0("pie_", gsub("-", "_", model_n), "_F3")), 
        #                                          get(paste0("con_", gsub("-", "_", model_n), "_F3")))))
        assign(x = paste0("pie_con_", gsub("-", "_", model_n), "_F1"),
               value = ggarrange(plotlist = list(get(paste0("pie_", gsub("-", "_", model_n), "_F1")), 
                                                 get(paste0("con_marg_", gsub("-", "_", model_n), "_F1")))))

        assign(x = paste0("pie_con_", gsub("-", "_", model_n), "_F2"),
               value = ggarrange(plotlist = list(get(paste0("pie_", gsub("-", "_", model_n), "_F2")), 
                                                 get(paste0("con_marg_", gsub("-", "_", model_n), "_F2")))))

        assign(x = paste0("pie_con_", gsub("-", "_", model_n), "_F3"),
               value = ggarrange(plotlist = list(get(paste0("pie_", gsub("-", "_", model_n), "_F3")), 
                                                 get(paste0("con_marg_", gsub("-", "_", model_n), "_F3")))))
        # _______________________________________________________________________________
        # 
        # bind the goddamn pie_con_ and 1D_dens together
        #
        # assign(x = paste0("pie_con_dens_", gsub("-", "_", model_n), "_F1"),
        #        value = ggarrange(plotlist = list(get(paste0("pie_con_", gsub("-", "_", model_n), "_F1")), 
        #                                          get(paste0("den_", gsub("-", "_", model_n), "_F1"))),
        #                          ncol = 1, nrow = 2))

        # assign(x = paste0("pie_con_dens_", gsub("-", "_", model_n), "_F2"),
        #        value = ggarrange(plotlist = list(get(paste0("pie_con_", gsub("-", "_", model_n), "_F2")), 
        #                                          get(paste0("den_", gsub("-", "_", model_n), "_F2"))),
        #                          ncol = 1, nrow = 2))

        # assign(x = paste0("pie_con_dens_", gsub("-", "_", model_n), "_F3"),
        #        value = ggarrange(plotlist = list(get(paste0("pie_con_", gsub("-", "_", model_n), "_F3")), 
        #                                          get(paste0("den_", gsub("-", "_", model_n), "_F3"))),
        #                         ncol = 1, nrow = 2))
        #________________________________________________________________________________
        # plot geographical maps:
        data(county.fips) # Load the county.fips dataset for plotting
        cnty <- map_data("county") # Load the county data from the maps package
        cnty2 <- cnty %>%
                 mutate(polyname = paste(region, subregion, sep=",")) %>%
                 left_join(county.fips, by="polyname")

        target_county_map_info <- cnty2 %>% filter(fips == target_fip)
        analog_name_F1_map_info <- cnty2 %>% filter(fips == most_similar_cnty_F1_fip)
        analog_name_F2_map_info <- cnty2 %>% filter(fips == most_similar_cnty_F2_fip)
        analog_name_F3_map_info <- cnty2 %>% filter(fips == most_similar_cnty_F3_fip)
        
        assign(x = paste0("map_", gsub("-", "_", model_n), "_F1"), 
               value = {plot_the_map(one_mod_map_info_F1, cnty2, titlem_F1, 
                                     target_county_map_info, 
                                     analog_name_F1_map_info,
                                     analog_name=analog_name_F1)})

        assign(x = paste0("map_", gsub("-", "_", model_n), "_F2"), 
               value = {plot_the_map(one_mod_map_info_F2, cnty2, titlem_F2,
                                     target_county_map_info, 
                                     analog_name_F2_map_info,
                                     analog_name= analog_name_F2)})

        assign(x = paste0("map_", gsub("-", "_", model_n), "_F3"), 
               value = {plot_the_map(a_dt = one_mod_map_info_F3, county2 = cnty2, 
                                     title_p = titlem_F3,
                                     target_county_map_info = target_county_map_info, 
                                     most_similar_cnty_map_info = analog_name_F3_map_info,
                                     analog_name = analog_name_F3)})

        # rm(most_similar_cnty_F1, most_similar_cnty_F2, most_similar_cnty_F3)
        # rm(most_similar_cnty_F1_fip, most_similar_cnty_F2_fip, most_similar_cnty_F3_fip)
        # rm(titlem_F1, titlem_F2, titlem_F3)
        # rm(one_mod_pie_info_F1, one_mod_pie_info_F2, one_mod_pie_info_F3)
      }

      assign(x = paste0("plot_", target_fip) , 
             value={ggarrange(plotlist = list(map_bcc_csm1_1_m_F1, map_bcc_csm1_1_m_F2, map_bcc_csm1_1_m_F3,
                                              pie_con_bcc_csm1_1_m_F1, pie_con_bcc_csm1_1_m_F2, pie_con_bcc_csm1_1_m_F3,
                                              
                                              map_BNU_ESM_F1, map_BNU_ESM_F2, map_BNU_ESM_F3,
                                              pie_con_BNU_ESM_F1, pie_con_BNU_ESM_F2, pie_con_BNU_ESM_F3,

                                              map_CanESM2_F1, map_CanESM2_F2, map_CanESM2_F3,
                                              pie_con_CanESM2_F1, pie_con_CanESM2_F2, pie_con_CanESM2_F3,

                                              map_CNRM_CM5_F1, map_CNRM_CM5_F2, map_CNRM_CM5_F3,
                                              pie_con_CNRM_CM5_F1, pie_con_CNRM_CM5_F2, pie_con_CNRM_CM5_F3,

                                              map_GFDL_ESM2G_F1, map_GFDL_ESM2G_F2, map_GFDL_ESM2G_F3,
                                              pie_con_GFDL_ESM2G_F1, pie_con_GFDL_ESM2G_F2, pie_con_GFDL_ESM2G_F3,

                                              map_GFDL_ESM2M_F1, map_GFDL_ESM2M_F2, map_GFDL_ESM2M_F3,
                                              pie_con_GFDL_ESM2M_F1, pie_con_GFDL_ESM2M_F2, pie_con_GFDL_ESM2M_F3),
                              heights= rep(c(3, 1), 6), 
                              ncol = 3, nrow = 12, common.legend=FALSE)})

      rm(analog_dat_F1, analog_dat_F2, analog_dat_F3, 
         most_similar_cnty_F1, most_similar_cnty_F2, most_similar_cnty_F3,
         f_years_F3, f_years_F2, f_years_F1, 
         analog_dat_F3_4_map, analog_dat_F2_4_map, analog_dat_F1_4_map, 
         most_similar_cnty_F3_fip, most_similar_cnty_F2_fip, most_similar_cnty_F1_fip,
         titlem_F3, titlem_F2, titlem_F1)
      rm(map_bcc_csm1_1_m_F1, map_bcc_csm1_1_m_F2, map_bcc_csm1_1_m_F3,
         pie_con_bcc_csm1_1_m_F1, pie_con_bcc_csm1_1_m_F2, pie_con_bcc_csm1_1_m_F3,  
         map_BNU_ESM_F1, map_BNU_ESM_F2, map_BNU_ESM_F3,
         pie_con_BNU_ESM_F1, pie_con_BNU_ESM_F2, pie_con_BNU_ESM_F3,
         map_CanESM2_F1, map_CanESM2_F2, map_CanESM2_F3,
         pie_con_CanESM2_F1, pie_con_CanESM2_F2, pie_con_CanESM2_F3,
         map_CNRM_CM5_F1, map_CNRM_CM5_F2, map_CNRM_CM5_F3,
         pie_con_CNRM_CM5_F1, pie_con_CNRM_CM5_F2, pie_con_CNRM_CM5_F3,
         map_GFDL_ESM2G_F1, map_GFDL_ESM2G_F2, map_GFDL_ESM2G_F3,
         pie_con_GFDL_ESM2G_F1, pie_con_GFDL_ESM2G_F2, pie_con_GFDL_ESM2G_F3,
         map_GFDL_ESM2M_F1, map_GFDL_ESM2M_F2, map_GFDL_ESM2M_F3,
         pie_con_GFDL_ESM2M_F1, pie_con_GFDL_ESM2M_F2, pie_con_GFDL_ESM2M_F3)
    }
    
    # plot_out_dir <- paste0(data_dir, "/different_axis/geo_maps_", VL_quans[1]*100, "_", VL_quans[2]*100, "/")
    # plot_out_dir <- "/Users/hn/Desktop/"
    plot_out_dir <- paste0(data_dir, "/diff_axis_geo_maps_clear/")
    if (dir.exists(plot_out_dir) == F) { dir.create(path = plot_out_dir, recursive = T)}
    
    image_dpi = 200
    imgage_h = 150
    imgage_w = 75

    ggsave(filename = paste0("WA_Franklin", plot_name_extension, ".png"), 
           plot = plot_53021, 
           path=plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)
    
    print ("WA_Franklin saved")
    ggsave(filename = paste0("WA_Okanogan", plot_name_extension, ".png"), 
           plot = plot_53047,
           path=plot_out_dir, device="png",
           dpi=image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Chelan", plot_name_extension, ".png"), 
           plot = plot_53007, 
           path = plot_out_dir, device = "png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit = "in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Yakima", plot_name_extension, ".png"), 
           plot = plot_53077,
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)
    print ("WA_Yakima_53077 saved")

    ggsave(filename = paste0("ID_Canyon", plot_name_extension, ".png"), 
           plot = plot_16027, 
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit = "in", limitsize = FALSE)

    ggsave(filename = paste0("OR_Gilliam", plot_name_extension, ".png"), 
           plot = plot_41021, 
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit = "in", limitsize = FALSE)
   
    ggsave(filename = paste0("OR_Hood_River", plot_name_extension, ".png"), 
           plot = plot_41027,
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)

    ggsave(filename = paste0("OR_Morrow", plot_name_extension, ".png"), 
           plot = plot_41049,
           path = plot_out_dir, device = "png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit = "in", limitsize = FALSE)

    ggsave(filename = paste0("OR_Umatilla", plot_name_extension, ".png"), 
           plot = plot_41059,
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit = "in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Adams", plot_name_extension, ".png"), 
           plot = plot_53001,
           path = plot_out_dir, device = "png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit = "in", limitsize = FALSE)
    
    ggsave(filename = paste0("WA_Benton", plot_name_extension, ".png"), 
           plot = plot_53005, 
           path = plot_out_dir, device = "png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit = "in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Columbia", plot_name_extension, ".png"), 
           plot = plot_53013, 
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height=imgage_h, unit="in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Douglas", plot_name_extension, ".png"), 
           plot = plot_53017, 
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Grant", plot_name_extension, ".png"), 
           plot = plot_53025,
           path=plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Kittitas", plot_name_extension, ".png"), 
           plot = plot_53037,
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Klickitat", plot_name_extension, ".png"), 
           plot = plot_53039,
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)

    ggsave(filename = paste0("WA_Walla_Walla", plot_name_extension, ".png"), 
           plot = plot_53071,
           path = plot_out_dir, device="png",
           dpi = image_dpi, width = imgage_w, height = imgage_h, unit="in", limitsize = FALSE)

  }
}




