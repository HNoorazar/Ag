#
# This is obtaied by copying and modifying "plot_densities.R"
# To plot densities for specific locations
#

#####################################################
###                                               ###
###             Sept. thru Apr.                   ###
###                                               ###
#####################################################
rm(list=ls())

.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr) # for ggarrange
options(digit=9)
options(digits=9)

in_dir <- "/data/hydro/users/Hossein/chill/7_time_intervals/RDS_files/2019/"


months = c("Sept", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar")

Sept1_March31_85 <- data.table()
Sept1_March31_45 <- data.table()
Sept1_March31_ModHist <- data.table()
Sept1_March31_obs <- data.table()

for (month in months){
  dt <- data.table(readRDS(paste0(in_dir, month, ".rds")))

  dt_85 <- dt %>% filter(scenario == "rcp85") %>% data.table()
  dt_45 <- dt %>% filter(scenario == "rcp45") %>% data.table()
  dt_ModHist <- dt %>% filter(scenario == "historical") %>% data.table()

  Sept1_March31_85 <- rbind(Sept1_March31_85, dt_85)
  Sept1_March31_45 <- rbind(Sept1_March31_45, dt_45)
  Sept1_March31_ModHist <- rbind(Sept1_March31_ModHist, dt_ModHist)

  obs <- data.table(readRDS(paste0(in_dir, "observed_", month, ".rds")))
  Sept1_March31_obs <- rbind(Sept1_March31_obs, obs)
  
}


setnames(Sept1_March31_85, old=c("scenario"), new=c("emission"))
setnames(Sept1_March31_45, old=c("scenario"), new=c("emission"))
setnames(Sept1_March31_ModHist, old=c("scenario"), new=c("emission"))
setnames(Sept1_March31_obs, old=c("scenario"), new=c("emission"))

out_dir <- "/data/hydro/users/Hossein/chill/7_time_intervals/RDS_files/"

saveRDS(Sept1_March31_85, paste0(out_dir, "Sept1_March31_85.rds"))
saveRDS(Sept1_March31_45, paste0(out_dir, "Sept1_March31_45.rds"))
saveRDS(Sept1_March31_ModHist, paste0(out_dir, "Sept1_March31_ModHist.rds"))
saveRDS(Sept1_March31_obs, paste0(out_dir, "Sept1_March31_obs.rds"))


