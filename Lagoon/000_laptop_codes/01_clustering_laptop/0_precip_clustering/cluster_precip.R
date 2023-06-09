rm(list=ls())
library(lubridate)
library(ggpubr)
library(purrr)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(maps)

source_path_1 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_lagoon.R"
source_path_2 = "/Users/hn/Documents/GitHub/Kirti/Lagoon/core_plot_lagoon.R"
source(source_path_1)
source(source_path_2)

options(digit=9)
options(digits=9)

##### read file
in_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"
file_name <- "precip_avgs.rds"

observed_dt <- readRDS(paste0(in_dir, file_name)) %>% data.table()
head(observed_dt, 2)
setnames(observed_dt, old=c("precip_avg"), new=c("annual_cum_precip"))
outputs <- cluster_yr_avging(observed_dt, scale=FALSE, no_clusters=4)

clusters <- outputs[[1]]
cluster_obj <- outputs[[2]]
head(clusters, 2)
head(observed_dt, 2)

################################################
#
# Check whether truely 1 through 4 
# are most to least precip
#
dt <- observed_dt %>% 
      group_by(location) %>% 
      summarise(target_col = mean(annual_cum_precip)) %>% 
      data.table()

dt <- merge(dt, clusters, by="location", all.x=TRUE)
cluster_1 <- dt %>% filter(cluster==1) %>% data.table()
cluster_2 <- dt %>% filter(cluster==2) %>% data.table()
cluster_3 <- dt %>% filter(cluster==3) %>% data.table()
cluster_4 <- dt %>% filter(cluster==4) %>% data.table()
c1_mean <- mean(cluster_1$target_col)
c2_mean <- mean(cluster_2$target_col)
c3_mean <- mean(cluster_3$target_col)
c4_mean <- mean(cluster_4$target_col)
cluster_means <- c(c1_mean, c2_mean, c3_mean, c4_mean)

########################
# Do the following so in the map they are discrete 
# for sake of coloring.
clusters$cluster <- factor(clusters$cluster)

cluster_plt <- geo_map_of_clusters(clusters) +
               ggtitle("clustering by avg. annual. precip. (observed)") 

plot_dir <- "/Users/hn/Desktop/Desktop/Kirti/check_point/lagoon/"
ggsave(filename = "precip_clusters.png", 
       plot = cluster_plt, device = "png",
       width = 6, height = 6, units = "in", dpi=400,
       path=plot_dir)

# round columns to 2 decimal
clusters <- clusters %>% 
            mutate_at(vars(ann_prec_mean, centroid), funs(round(., 2)))

# save in parameter
out_dir <- "/Users/hn/Documents/GitHub/Kirti/Lagoon/parameters/"
write.table(clusters, 
            file = paste0(out_dir, "observed_clusters.csv"),
            row.names = FALSE, na="", 
            col.names=TRUE, sep=",")




