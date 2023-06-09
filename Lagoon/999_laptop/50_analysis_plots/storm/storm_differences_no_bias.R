rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)
########################################################################
########################################################################

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/storm/"
plot_dir <- paste0(in_dir, "plots/")
           
all_storms <- readRDS(paste0(in_dir, "all_storms.rds"))
head(all_storms, 2)

# all_storms <- cluster_numeric_2_str(all_storms)
# all_storms$return_period <- factor(all_storms$return_period,
#                                    levels = c("1950-2005", "1979-2016", 
#                                                "2006-2025", "2026-2050",
#                                                "2051-2075", "2076-2099"))

all_storms <- all_storms %>%
              filter(return_period != "2006-2025")%>%
              data.table()

storm_diffs <- storm_diff_4_map_obs_or_modeled(dt_dt =all_storms, 
                                               diff_from="1950-2005")

storm_diffs_box <- storm_diff_box_25yr(data_tb=storm_diffs, tgt_col="storm_diff")
storm_diffs_percentage_box <- storm_diff_box_25yr(storm_diffs, tgt_col="perc_diff")

ggsave(filename = "storm_diffs_no_bias.png",
       plot = storm_diffs_box, 
       width = 9, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

ggsave(filename = "storm_diffs_percentage_no_bias.png",
       plot = storm_diffs_percentage_box, 
       width = 9, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)


##########3 Leasts
storm_diffs <- storm_diffs %>% 
               filter(cluster %in% c("least precip", "lesser precip"))%>% 
               data.table()

storm_diffs_box <- storm_diff_box_25yr(storm_diffs, tgt_col="storm_diff")
storm_diffs_percentage_box <- storm_diff_box_25yr(storm_diffs, tgt_col="perc_diff")

ggsave(filename = "least_storm_diffs_no_bias.png",
       plot = storm_diffs_box, 
       width = 6, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)

ggsave(filename = "least_storm_diffs_percentage_no_bias.png",
       plot = storm_diffs_percentage_box, 
       width = 6, height = 3.5, units = "in", 
       dpi=400, device = "png",
       path = plot_dir)



