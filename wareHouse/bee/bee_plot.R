
library(data.table)
library(ggplot2)
library(dplyr)

in_dir <- "/Users/hn/Documents/GitHub/Kirti/wareHouse/bee/"
vp_results_2002 <- read.csv(paste0(in_dir, "2002.csv"), 
                              header = TRUE, as.is=T) %>% data.table()

vp_results_2001 <- read.csv(paste0(in_dir, "2001.csv"), 
                              header = TRUE,  as.is=T) %>% data.table()

vp_results_2002 <- subset(vp_results_2002, select=c(Date, Colony.Size))
vp_results_2001 <- subset(vp_results_2001, select=c(Date, Colony.Size))

vp_results_2002 <- vp_results_2002[2:nrow(vp_results_2002)]
vp_results_2001 <- vp_results_2001[2:nrow(vp_results_2001)]