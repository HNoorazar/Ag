# """
# This driver will comupte the percentage diff between future CPs and observed CPs.
# and generate a map of colors for them.
# 
# """
rm(list=ls())

library(ggmap)
library(ggpubr)
library(lubridate)
library(purrr)
library(scales)
library(tidyverse)
library(maps)
library(data.table)
library(dplyr)

options(digits=9)
options(digit=9)


core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
plot_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(core_path)
source(plot_core_path)


data_dir = "/Users/hn/Documents/01_research_data/chilling/01_data/02/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/chilling/parameters/"

LocationGroups_NoMontana <- read.csv(paste0(param_dir, "LocationGroups_NoMontana.csv"), 
                                     header=T, sep=",", as.is=T)

remove_montana <- function(data_dt, LocationGroups_NoMontana){
  if (!("location" %in% colnames(data_dt))){
    data_dt$location <- paste0(data_dt$lat, "_", data_dt$long)
  }
  data_dt <- data_dt %>% filter(location %in% LocationGroups_NoMontana$location)
  return(data_dt)
}


sept_summary_comp <- readRDS(paste0(data_dir, "sept_summary_comp.rds")) %>%
                     data.table()

head(sept_summary_comp, 2)
dim(sept_summary_comp)

keep_cols <- c("location", "sum_A1", "model", "emission", "time_period")

sept_summary_comp <- subset(sept_summary_comp, select=keep_cols)

sept_summary_comp <- sept_summary_comp %>% 
                     filter(!(time_period %in% c("2006-2025", "1950-2005"))) %>%
                     data.table()

sept_summary_comp <- remove_montana(sept_summary_comp, LocationGroups_NoMontana)


###### Change time period for sake of plotting:
# sept_summary_comp$time_period[sept_summary_comp$time_period== "2025_2050"] = "2025-2050"
# sept_summary_comp$time_period[sept_summary_comp$time_period== "2051_2075"] = "2051-2075"
# sept_summary_comp$time_period[sept_summary_comp$time_period== "2076_2100"] = "2076-2099"

sept_summary_comp$time_period[sept_summary_comp$model == "observed"] <- "Observed"

sept_summary_comp$emission[sept_summary_comp$emission == "rcp45"] = "RCP 4.5"
sept_summary_comp$emission[sept_summary_comp$emission == "rcp85"] = "RCP 8.5"
sept_summary_comp$emission[sept_summary_comp$emission == "historical"] = "Observed"


time_periods = c("Observed", "2026-2050", "2051-2075", "2076-2099")
sept_summary_comp$time_period = factor(sept_summary_comp$time_period, levels = time_periods, order=T)

sept_summary_comp_yearsMedian_perModel <- sept_summary_comp %>%
                                          group_by(location, model, emission, time_period) %>%
                                          summarise(CP_median_A1=median(sum_A1)) %>%
                                          data.table()
 
diffs <- projec_diff_from_hist(sept_summary_comp_yearsMedian_perModel)

diffs_median <- diffs %>% 
                group_by(location, time_period, emission) %>%
                summarise(CP_diff_median = median(perc_diff)) %>%
                data.table()

diffs_median$CP_diff_median[diffs_median$CP_diff_median < -40] = -40

plot_base <- paste0("/Users/hn/Documents/00_GitHub/ag_papers/chill_paper/02_Springer_2/figure_200/cp_diff_map/")
# plot_base <- "/Users/hn/Documents/00_GitHub/Ag_papers/Chill_Paper/MajorRevision/figures_4_revision/"
if (dir.exists(plot_base) == F) {dir.create(path = plot_base, recursive = T)}

core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
plot_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(core_path)
source(plot_core_path)

diffs_median$emission <- factor(diffs_median$emission, 
                                levels=c("RCP 8.5", "RCP 4.5"),
                                order=TRUE)

a_map <- diff_CP_map(data = diffs_median, color_col = "CP_diff_median")

qual = 400
W = 7.5
H = 5.5
ggsave(filename = paste0("CP_diff_perc_Sept_Apr1_centered.pdf"), 
       plot=a_map, 
       width=W, height=H, units="in", 
       dpi=qual, device="pdf", path=plot_base)


diffs_median_85 <- diffs_median %>% filter(emission=="RCP 8.5")

core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_core.R"
plot_core_path = "/Users/hn/Documents/00_GitHub/Ag/chilling/chill_plot_core.R"
source(core_path)
source(plot_core_path)
rcp85_map <- diff_CP_map_one_emission(data = diffs_median_85, color_col = "CP_diff_median")

W = 7.5
H = 3.3

ggsave(filename = paste0("CP_diff_perc_Sept_Apr1st_centered_85.pdf"), 
       plot=rcp85_map, 
       width=W, height=H, units="in", 
       dpi=qual, device="pdf", 
       path=plot_base)



