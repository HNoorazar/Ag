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
start_time <- Sys.time()
############################################################################

data_base <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/rain_snow_fractions/"
in_dir <- data_base
############################################################################
####
####  boundaries
####

################################################
AV_title <- "monthly precip."
AV_y_lab <- "cum. precip. (mm)"
AV_tg_col <- "monthly_cum_precip"

AVs <- readRDS(paste0(in_dir, "monthly_fracs.rds")) %>% data.table()
AVs <- na.omit(AVs)
AVs <- AVs %>% filter(!(month %in% c(5, 6, 7, 8)))
plot_dir <- paste0(in_dir, "narrowed_rain_snow_fractions/monthly/sept_april_hard_code/")
if (dir.exists(plot_dir) == F) {dir.create(path = plot_dir, recursive = T)}
print (plot_dir)

AVs <- AVs %>% filter(time_period != "2006-2025") %>% data.table()
AVs <- within(AVs, remove(day, rain_portion, monthly_cum_rain, monthly_cum_snow))
AVs$rain_fraction <- AVs$rain_fraction * 100
AVs$snow_fraction <- AVs$snow_fraction * 100

AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
# rm(AVs)

cluster_types <- c("least precip", "lesser precip", "less precip", "most precip")
timeP_ty <- 1
clust_g <- cluster_types[1]

for (clust_g in cluster_types){
  if (clust_g == "least precip"){
    rain_min <- 50; rain_max <- 100
    snow_min <- 0; snow_max <- 50
  } else if (clust_g == "less precip"){
    rain_min <- 0; rain_max <- 100
    snow_min <- 0; snow_max <- 100
  }else if (clust_g == "lesser precip"){
    rain_min <- 0; rain_max <- 100
    snow_min <- 0; snow_max <- 100
  }else if (clust_g == "most precip"){
    rain_min <- 10; rain_max <- 100
    snow_min <- 0; snow_max <- 85
  }
  curr_AVs_45 <- AVs_45 %>% filter(cluster == clust_g) %>% data.table()
  curr_AVs_85 <- AVs_85 %>% filter(cluster == clust_g) %>% data.table()

  #########
  ######### Actual value plots
  #########
  AV_box_85 <- box_trend_monthly_cum(dt = curr_AVs_85, p_type="box",
                                     y_lab = AV_y_lab, tgt_col = AV_tg_col) +
               ggtitle(AV_title)

  AV_box_45 <- box_trend_monthly_cum(dt = curr_AVs_45, p_type="box",
                                     y_lab = AV_y_lab, tgt_col = AV_tg_col) + 
               ggtitle(AV_title)
  #########
  ######### rain fracitons
  #########
  box_title <- paste0("rain fracion")
  rain_85 <- frac_rain(dt = curr_AVs_85,
                       y_lab = "rain fraction (%)",
                       tgt_col = "rain_fraction") + 
             ggtitle(box_title) +
             coord_cartesian(ylim = c(rain_min, rain_max))

  rain_85 <- ggarrange(plotlist = list(AV_box_85, rain_85),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  ggsave(filename = paste0(clust_g, "_rain_85.png"),
         plot = rain_85, width = 6, height = 5, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  ####################################################################
  rain_45 <- frac_rain(dt = curr_AVs_45,
                       y_lab = "rain fraction (%)",
                       tgt_col = "rain_fraction") + 
             ggtitle(box_title) + 
             coord_cartesian(ylim = c(rain_min, rain_max))
  
  rain_45 <- ggarrange(plotlist = list(AV_box_45, rain_45),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  ggsave(filename = paste0(clust_g, "_rain_45.png"),
         plot = rain_45, width = 6, height = 5, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  ####################################################################
  box_title <- paste0("snow fracion")
  snow_85 <- frac_snow(dt = curr_AVs_85,
                       y_lab = "snow fraction (%)",
                       tgt_col = "snow_fraction") + 
             ggtitle(box_title) +
             coord_cartesian(ylim = c(snow_min, snow_max))
  
  snow_85 <- ggarrange(plotlist = list(AV_box_85, snow_85),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  ggsave(filename = paste0(clust_g, "_snow_85.png"),
         plot = snow_85, width = 6, height = 5, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  ####################################################################
  snow_45 <- frac_snow(dt = curr_AVs_45,
                       y_lab = "snow fraction (%)",
                       tgt_col = "snow_fraction") + 
             ggtitle(box_title) +
             coord_cartesian(ylim = c(snow_min, snow_max))
  
  snow_45 <- ggarrange(plotlist = list(AV_box_45, snow_45),
                       ncol = 1, nrow = 2, common.legend = TRUE, legend="bottom")
  ggsave(filename = paste0(clust_g, "_snow_45.png"),
         plot = snow_45, width = 6, height = 5, units = "in", 
         dpi=400, device = "png", path = plot_dir)
  ##################################################################################
  
}

print (Sys.time() - start_time)