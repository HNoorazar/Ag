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

source_path_1 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Ag/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)
############################################################################
start_time <- Sys.time()

data_base <- "/Users/hn/Desktop/Desktop/Ag/check_point/lagoon/"
plot_base <- paste0(data_base, "plots/runoff/")
in_dir_ext <- c("runbase")
unbias_dir_ext <- "/02_med_diff_med_no_bias/"

runoff_AV_fileNs <- c("wtr_yr_cum_runbase") # "ann_cum_runbase",
timeP_ty_middN <- c("wtr_yr") # "ann", 
av_tg_col_pref <- c("annual_cum_") # "annual_cum_",
av_titles <- c("Total annual runoff")
emissions <- c("RCP 4.5", "RCP 8.5")

dt_type <-  in_dir_ext[1]
in_dir <- paste0(data_base, dt_type, "/")
timeP_ty <- 1

for (dt_type in in_dir_ext){
  in_dir <- paste0(data_base, dt_type, "/")
  for (timeP_ty in length(timeP_ty_middN)){
    ###############################################################
    # set up title stuff
    # 
    # if (timeP_ty_middN[timeP_ty]== "ann"){
    #   title_time <- "calendar year"
    #   } else if (timeP_ty_middN[timeP_ty]== "wtr_yr"){
    #     title_time <- "water year"
    # }
    files <- runoff_AV_fileNs
    AV_y_lab <- "runoff (mm)"
    AV_tg_col <- paste0(av_tg_col_pref[timeP_ty], "runbase")
    AV_title <- paste0(av_titles[timeP_ty])
    ###############################################################

    AVs <- readRDS(paste0(in_dir, files[timeP_ty], ".rds")) %>% 
           data.table()
    unbias_diff <- readRDS(paste0(in_dir, unbias_dir_ext, 
                                  "detail_med_diff_med_", 
                                  timeP_ty_middN[timeP_ty], 
                                  "_", dt_type, ".rds")) %>% 
                   data.table()
    AVs <- remove_observed(AVs)
    unbias_diff <- remove_observed(unbias_diff)

    AVs <- remove_current_timeP(AVs) # remove 2006-2025
    unbias_diff <- remove_current_timeP(unbias_diff) # remove 2006-2025
    
    # update clusters labels
    AVs <- convert_5_numeric_clusts_to_alphabet(AVs)
    unbias_diff <- convert_5_numeric_clusts_to_alphabet(unbias_diff)

    AVs_45 <- AVs %>% filter(emission=="RCP 4.5") %>% data.table()
    AVs_85 <- AVs %>% filter(emission=="RCP 8.5") %>% data.table()
    unbias_diff_45 <- unbias_diff %>% filter(emission=="RCP 4.5") %>% data.table()
    unbias_diff_85 <- unbias_diff %>% filter(emission=="RCP 8.5") %>% data.table()
    rm(AVs, unbias_diff)

    av_quans_85 <- find_quantiles(data_table=AVs_85, 
                                  tgt_col= AV_tg_col, 
                                  time_type="annual")
    av_quans_45 <- find_quantiles(data_table=AVs_45, 
                                  tgt_col= AV_tg_col, 
                                  time_type="annual")

    AV_box_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt=AVs_85, 
                                                   y_lab=AV_y_lab, 
                                                   tgt_col=AV_tg_col) + 
                 ggtitle(AV_title) + 
                 coord_cartesian(ylim = c(av_quans_85[1], 
                                          av_quans_85[2]))

    AV_box_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = AVs_45, 
                                                   y_lab = AV_y_lab, 
                                                   tgt_col = AV_tg_col) + 
                 ggtitle(AV_title) + 
                 coord_cartesian(ylim = c(av_quans_45[1], 
                                          av_quans_45[2]))

    ###################################
    #####
    ##### difference plots
    #####
    ###################################
    box_title <- "Difference (%) in annual runoff"

    quans_85 <- find_quantiles(unbias_diff_85, 
                               tgt_col= "perc_diff", 
                               time_type="annual")
    quans_45 <- find_quantiles(unbias_diff_45, 
                               tgt_col= "perc_diff", 
                               time_type="annual")
  
    unbias_perc_diff_85 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = unbias_diff_85,
                                                             y_lab = "differences (%)",
                                                             tgt_col = "perc_diff",
                                                             ttl = box_title, 
                                                             subttl = box_subtitle) + 
                           ggtitle(box_title) +
              coord_cartesian(ylim = c(quans_85[1], quans_85[2]))

    unbias_perc_diff_45 <- ann_wtrYr_chunk_cum_box_cluster_x(dt = unbias_diff_45,
                                                             y_lab = "differences (%)",
                                                             tgt_col = "perc_diff",
                                                             ttl = box_title, 
                                                             subttl = box_subtitle) + 
                           ggtitle(box_title) + 
          coord_cartesian(ylim = c(quans_45[1], quans_45[2]))
    ###################################
    #####
    ##### arrange plots
    #####
    ###################################

    RCP45 <- ggarrange(plotlist = list(AV_box_45, unbias_perc_diff_45),
                       ncol = 1, nrow=2, 
                       common.legend = TRUE, 
                       legend="bottom")

    RCP45 <- annotate_figure(RCP45,
                             top = text_grob("RCP 4.5", 
                                              face = "bold", 
                                              size = 10,
                                              color="red"))

    RCP85 <- ggarrange(plotlist = list(AV_box_85, unbias_perc_diff_85),
                       ncol = 1, nrow = 2, 
                       common.legend = TRUE, 
                       legend="bottom")

    RCP85 <- annotate_figure(RCP85,
                             top = text_grob("RCP 8.5", 
                                              face = "bold", 
                                              size = 10,
                                              color="red"))
      
    run_n_diff <- paste0(plot_base, 
                         timeP_ty_middN[timeP_ty])
    if (dir.exists(run_n_diff) == F) {
      dir.create(path=run_n_diff, recursive = T)}
      
    ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_RCP45.png"),
           plot = RCP45,  units = "in",
           width=5.5, height=3.5,
           dpi=600, device = "png", path = run_n_diff)

    ggsave(filename = paste0(timeP_ty_middN[timeP_ty], "_RCP85.png"),
           plot = RCP85, units = "in",
           width=5.5, height=3.5,
           dpi=600, device = "png", path = run_n_diff)
    print (run_n_diff)
    
  }
}


print (Sys.time() - start_time)