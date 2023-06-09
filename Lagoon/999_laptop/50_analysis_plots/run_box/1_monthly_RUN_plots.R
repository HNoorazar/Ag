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

in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/runoff/"
plot_dir <- paste0(in_dir, "plots/monthly/")

##############################
fileN <- "all_monthly_cum_runoff_LD"
dt_tb <- data.table(readRDS(paste0(in_dir, fileN, ".rds")))
head(dt_tb, 2)

tg_col <- "monthly_cum_runbase"

y_labb <- "runff (mm)"
ttl <- "monthly cum. runoff"
subttl <- " "

box_plt <- box_trend_monthly_cum(dt=dt_tb, p_type="box", 
                                 y_lab=y_labb, tgt_col = tg_col# ,ttl, subttl
                                 )

box_plt <- box_plt + ggtitle(ttl) # , , subtitle=subttl
ggsave(filename = "monthly_box.png", 
       plot = box_plt, 
       width = 14, height = 6, units = "in", 
       dpi=600, device = "png",
       path = plot_dir)

##############################

dt_tb <- dt_tb %>% filter(month %in% c(11, 12)) %>% data.table()
nov_Dec <- Nov_Dec_cum_box(dt=dt_tb, y_lab = y_labb, tgt_col= tg_col)

nov_Dec <- nov_Dec +  ggtitle(ttl) # , , subtitle=subttl

ggsave(filename = "nov_Dec_box.png", 
       plot = nov_Dec, 
       width = 11, height = 6, units = "in", 
       dpi=300, device = "png",
       path = plot_dir)
