rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
options(digit=9)
options(digits=9)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/cum_precip/"
plot_dir <- paste0(in_dir, "plots/maps/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
##############################
fileN <- "ann_all_last_days"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)

tgt_col <- "annual_cum_precip"

meds <- median_diff_4_map_obs_or_modeled(dt=dt_tb, 
                                         tgt_col=tgt_col, 
                                         diff_from="1979-2016")
head(meds, 2)

meds <- median_of_diff_of_medians(meds)
dim(meds)

min_diff <- min(meds$med_of_diffs_of_meds)
max_diff <- max(meds$med_of_diffs_of_meds)

min_diff_perc <- min(meds$perc_med_of_diffs_of_meds)
max_diff_perc <- max(meds$perc_med_of_diffs_of_meds)


emissions <- c("RCP 4.5", "RCP 8.5")
future_rn_pr <- c("2026-2050", "2051-2075", "2076-2099")

#######
#######     Difference of medians of annual precip
#######
subtitle <- "Diff. of medians of cum. precip. (annual)"
for (em in emissions){
  for (rp in future_rn_pr){
    curr_dt <- meds %>%
               filter(emission == em & time_period==rp) %>%
               data.table()
    title <- paste0(em, " (", rp, ")")
    
    assign(x = paste0(gsub(pattern = " ", 
                           replacement = "_", 
                           x = em),
                      "_",
                      gsub(pattern = "-", 
                           replacement = "_", 
                           x = rp)),
           value ={geo_map_of_diffs(dt = curr_dt, 
                                    col_col = "diff" , 
                                    minn = min_diff, maxx = max_diff,
                                    ttl = title, 
                                    subttl= subtitle)})

  }
}

diff_figs <- ggarrange(plotlist = list(RCP_8.5_2026_2050,
                                       RCP_8.5_2051_2075,
                                       RCP_8.5_2076_2099,
                                       RCP_4.5_2026_2050,
                                       RCP_4.5_2051_2075,
                                       RCP_4.5_2076_2099),
                       ncol = 3, nrow = 2,
                       common.legend = TRUE)

rm(RCP_4.5_2026_2050, RCP_8.5_2026_2050,
   RCP_4.5_2051_2075, RCP_8.5_2051_2075,
   RCP_4.5_2076_2099, RCP_8.5_2076_2099)

ggsave(filename = "precip_diff_medians_ANNUAL.png", 
       plot = diff_figs, 
       width = 10, height = 7, units = "in", # width = 7, height = 8,
       dpi=300, device = "png",
       path = plot_dir)

#######
#######     Percentage perc_difference of medians of annual precip
#######
subtitle <- "Diff. of medians of cum. precip.\n(in percentage)"
for (em in emissions){
  for (rp in future_rn_pr){
    curr_dt <- meds %>%
               filter(emission == em & time_period==rp) %>%
               data.table()
    title <- paste0(em, " (", rp, ")")
    assign(x = paste0(gsub(pattern = " ", 
                           replacement = "_", 
                           x = em),
                      "_",
                      gsub(pattern = "-", 
                           replacement = "_", 
                           x = rp)),
           value ={geo_map_of_diffs(dt = curr_dt, 
                                    col_col = "perc_diff" , 
                                    minn = min_diff_perc, maxx = max_diff_perc,
                                    ttl = title, 
                                    subttl= subtitle)})

  }
}

perc_diff_figs <- ggarrange(plotlist = list(RCP_8.5_2026_2050,
                                            RCP_8.5_2051_2075,
                                            RCP_8.5_2076_2099,
                                            RCP_4.5_2026_2050,
                                            RCP_4.5_2051_2075,
                                            RCP_4.5_2076_2099),
                           ncol = 3, nrow = 2,
                           common.legend = TRUE)

ggsave(filename = "precip_perc_diff_medians_ANNUAL.png", 
       plot = perc_diff_figs, 
       width = 10, height = 7, units = "in", # width = 7, height = 8
       dpi=300, device = "png",
       path = plot_dir)


