rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
########################################################################
########################################################################
in_dir <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/storm/"
plot_dir <- paste0(in_dir, "new_2_storm/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}

all_storms <- readRDS(paste0(in_dir, "all_storms.rds"))
head(all_storms, 2)
all_storms <- all_storms %>% filter(return_period != "1979-2016") %>% data.table()
all_storms <- all_storms %>% filter(return_period != "2006-2025") %>% data.table()
all_storms <- convert_5_numeric_clusts_to_alphabet(data_tb = all_storms)

all_storms <- within(all_storms, 
                    remove(five_years, ten_years, 
                           fifteen_years, twenty_years))

########################################################################
clusters <- sort(unique(all_storms$cluster))
clust <- clusters[1]
for (clust in clusters){
  curr_dt <- all_storms %>% filter(cluster == clust)
  curr_dt_45 <- curr_dt %>% filter(emission == "RCP 4.5") %>% data.table()
  curr_dt_85 <- curr_dt %>% filter(emission == "RCP 8.5") %>% data.table()

  ##################
  ################## Actual Values
  ################## 
  AV_title <- paste0(clust, ". (25 yr, 24 hr)")
  quans_85 <- storm_25_quantiles(curr_dt_85, tgt_col= "twenty_five_years")
  quans_45 <- storm_25_quantiles(curr_dt_45, tgt_col= "twenty_five_years")

  AV_85 <- box_dt_25(within(curr_dt_85, remove(location, model))) + 
           ggtitle(label = AV_title) +
           coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

  AV_45 <- box_dt_25(within(curr_dt_45, remove(location, model))) + 
           ggtitle(label = AV_title) +
           coord_cartesian(ylim = c(quans_45[1], quans_45[2]))

  ##################
  ################## Biased
  ##################
  # bias_diffs_45 <- storm_diff_obs_or_modeled(dt_dt =curr_dt_45, 
  #                                                  diff_from="1979-2016")
  
  # bias_diffs_box_45 <- storm_diff_box_25yr(data_tb=bias_diffs_45, 
  #                                          tgt_col="storm_diff") + 
  #                      ggtitle(# label="diff. of 25 yr/24 hr. design storm", 
  #                              label="biased differences") #
  
  # bias_diffs_perc_box_45 <- storm_diff_box_25yr(data_tb = bias_diffs_45,
  #                                               tgt_col="perc_diff") + 
  #                           ggtitle(# label="diff. of 25 yr/24 hr. design storm", 
  #                                   label="biased differences") #
  

  # bias_diffs_85 <- storm_diff_obs_or_modeled(dt_dt =curr_dt_85, 
  #                                                  diff_from="1979-2016")
  # bias_diffs_box_85 <- storm_diff_box_25yr(data_tb=bias_diffs_85, 
  #                                          tgt_col="storm_diff") +
  #                      ggtitle(# label="diff. of 25 yr/24 hr. design storm", 
  #                              label ="biased differences") #
  # bias_diffs_perc_box_85 <- storm_diff_box_25yr(bias_diffs_85, 
  #                                               tgt_col="perc_diff") + 
  #                           ggtitle(# label="diff. of 25 yr/24 hr. design storm", 
  #                                   label="biased differences") #
  ##################
  ################## Unbias
  ##################
  ###
  ### 45
  ###
  unbias_diffs_45 <- storm_diff_obs_or_modeled(dt_dt =curr_dt_45, 
                                               diff_from="1950-2005")
  quans_45 <- storm_25_quantiles(unbias_diffs_45, tgt_col= "perc_diff") 
  
  unbias_diffs_perc_box_45 <- storm_diff_box_25yr(unbias_diffs_45, 
                                                  tgt_col="perc_diff") + 
                              ggtitle(# subtitle ="diff. of 25 yr/24 hr. design storm", 
                                      label ="differences") + # 
                              coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
  ###
  ### 85
  ###
  unbias_diffs_85 <- storm_diff_obs_or_modeled(dt_dt=curr_dt_85, 
                                               diff_from="1950-2005")

  quans_85 <- storm_25_quantiles(unbias_diffs_85, tgt_col= "perc_diff") 
  # unbias_diffs_box_85 <- storm_diff_box_25yr(data_tb=unbias_diffs_85, 
  #                                            tgt_col="storm_diff") +
  #                        ggtitle(# subtitle="diff. of 25 yr/24 hr. design storm", 
  #                                label ="unbiased differences") # 

  unbias_diffs_perc_box_85 <- storm_diff_box_25yr(unbias_diffs_85, 
                                                  tgt_col="perc_diff") +
                              ggtitle(# label="diff. of 25 yr/24 hr. design storm", 
                                      label="differences") + #
                              coord_cartesian(ylim = c(quans_85[1], quans_85[2])) 

  plt_45_unbias <- ggarrange(plotlist = list(AV_45, unbias_diffs_perc_box_45),
                             ncol = 2, nrow = 1, widths = c(1.25, 1, 1),
                             common.legend = TRUE, legend="bottom")

  plt_85_unbias <- ggarrange(plotlist = list(AV_85, unbias_diffs_perc_box_85),
                             ncol = 2, nrow = 1, widths = c(1.25, 1, 1),
                             common.legend = TRUE, legend="bottom")

  ggsave(filename = paste0(gsub("\ ", "_", clust), "_45_unbias_storm.png"),
             plot = plt_45_unbias, 
             width = 6, height = 4, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)
  
  ggsave(filename = paste0(gsub("\ ", "_", clust), "_85_unbias_storm.png"),
             plot = plt_85_unbias, 
             width = 6, height = 4, units = "in", 
             dpi=400, device = "png",
             path = plot_dir)
  print (plot_dir)
}




