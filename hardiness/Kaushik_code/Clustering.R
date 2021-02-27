################################ Clustering  code ####################################
.libPaths("/data/hydro/R_libs35")
.libPaths()



library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggplot2)
library(reshape2)

# 
# # creating a clustering data frame for highest average predicted HC value
# cluster_data <- data.frame(Location= character(),location_mean = numeric(), lat = numeric(),long = numeric(), stringsAsFactors = FALSE)
# sapply(cluster_data, class)
# 
# # List of files in the loaction
# 
# # file_location <- "/data/hydro/users/kraghavendra/hardiness/output_data/observed/"
# file_location <- "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/observed/"
# 
# files_Total <- list.files(path= file_location)
# files_Total
# 
# for (names in files_Total)
# {
#   if (names != "consolidated_observed_historical.csv"){
#     lat <- NULL
#     long <- NULL
#     print(names)
#     
#     # accessing the second file name
#     # files_Total[2]
#     
#     # Adding it to a varibale
#     file_name <- names  #files_Total[2] 
#     file_name 
#     
#     # seperating the "_" to concatenate lat and long
#     file_name <- unlist(strsplit(file_name, "_"))
#     lat <- file_name[[5]]
#     
#     long <- str_extract(file_name[[6]], '.*(?=\\.csv)')
#     file_name <- paste0(file_name[[5]],file_name[[6]])
#     
#     
#     # removing the .csv from the file name
#     library(stringr)
#     file_name <- str_extract(file_name, '.*(?=\\.csv)')
#     
#     
#     read_file <- paste0(file_location,names)
#     read_file
#     
#     file_read <- read.csv(file = read_file, stringsAsFactors = FALSE)
#     sapply(file_read, class)
#     
#     file_read_dup <- data.table(file_read)
#     file_read_dup <- subset(file_read_dup, file_read_dup$hardiness_year %in% (1979:2005))
#     sapply(file_read_dup, class)
#     head(file_read_dup)
#     
#     file_read_dup <- file_read_dup[, min(predicted_Hc),by = hardiness_year]
#     
#     colnames(file_read_dup)
#     highest_mean <- colMeans(file_read_dup)
#     
#     highest_mean[["V1"]]
#     
#     row_add <- data.frame(file_name, highest_mean[["V1"]], lat, long)
#     names(row_add)<- c("Location","location_mean","lat", "long")
#     
#     cluster_data <- rbind(cluster_data, row_add)
#     
#     
#   }
# }
# write.csv(cluster_data, file = paste0(file_location, "cluster.csv"))
# # cluster_data
# # 
# # ################# k-means #################
# # 
# # cluster_data <- read.csv("C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/cluster.csv", stringsAsFactors = FALSE)
# # summary(cluster_data)
# 
# # cluster_data <- select(cluster_data, c("Location","location_mean"))
# # 
# # 
# # 
# # # ELBOW method to check the cluster
# # 
# # max_clusters = 12
# # for_elbow = data.table(no_clusters = c(1:max_clusters),
# #                        total_within_cluster_ss = rep(-666, max_clusters))
# # 
# # for (k in 1:max_clusters){
# #   # this is to maintain same randomness over times repeated
# #   set.seed(100)
# #   output <- kmeans(cluster_data$location_mean, centers = k)
# #   clusters_obj <- output
# #   for_elbow[k, "total_within_cluster_ss"] <- clusters_obj$tot.withinss
# # }
# # 
# # plot(for_elbow)
# # 
# # 
# # # choosing cluster number as 4 based on elbow method
# # k <- kmeans(cluster_data$location_mean, centers = 4)
# # k
# # order_K <- k$cluster
# # 
# # cluster_data$cluster_no <- k$cluster
# # cluster_data
# # 
# # ######################## MAp plotting for minimum temperature #########################
# library(ggplot2)
# library(maps)
# library(ggmap)
# cluster_data <- read.csv("C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/cluster.csv", stringsAsFactors = FALSE)
# cluster_data <- select(cluster_data,long,lat,location_mean)
# sapply(cluster_data, class)
# 
# 
# 
# cluster_data
# any(is.na(cluster_data))
# cluster_data$region <- "washington" 
# cluster_data$subregion <- "chelan"
# 
# summary(cluster_data)
# summary(Washington_counties)
# 
# 
# # preparing Data before join
# cluster_data <- select(cluster_data, c("location_mean", "lat", "long", "region"))
# cluster_data
# 
# 
# write.csv(join_data, file = paste0(file_location, "jpin_data.csv"))
# 
# Washington_base <- ggplot(data = Washington, mapping = aes(x=long, y=lat, group = group))+
#   coord_fixed(1.3)+
#   geom_polygon(color = "black", fill = "gray" )
# 
# Washington_base+
#   geom_polygon(data = Washington_counties, fill =NA, color= "white")+
#   geom_point(data =cluster_data,aes(x=long, y= lat,color =location_mean,size=1),inherit.aes = FALSE)
  
################### Clustering check for t-min ###############

cluster_data_tmin <- data.frame(Location= character(),location_mean = numeric(), lat = numeric(),long = numeric(), stringsAsFactors = FALSE)
sapply(cluster_data_tmin, class)

# List of files in the loaction

file_location <- "/data/hydro/users/kraghavendra/hardiness/output_data/observed/"
# file_location <- "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/observed/"

files_Total <- list.files(path= file_location)
files_Total

for (names in files_Total)
{
  if (names != "consolidated_observed_historical.csv" & names!= "cluster.csv"){
    lat <- NULL
    long <- NULL
    print(names)
    
    # accessing the second file name
    # files_Total[2]
    
    # Adding it to a varibale
    file_name <- names  #files_Total[2] 
    file_name 
    
    # seperating the "_" to concatenate lat and long
    file_name <- unlist(strsplit(file_name, "_"))
    lat <- file_name[[5]]
    
    long <- str_extract(file_name[[6]], '.*(?=\\.csv)')
    file_name <- paste0(file_name[[5]],file_name[[6]])
    
    
    # removing the .csv from the file name
    library(stringr)
    file_name <- str_extract(file_name, '.*(?=\\.csv)')
    
    
    read_file <- paste0(file_location,names)
    read_file
    
    file_read <- read.csv(file = read_file, stringsAsFactors = FALSE)
    sapply(file_read, class)
    
    file_read_dup <- data.table(file_read)
    file_read_dup <- subset(file_read_dup, file_read_dup$hardiness_year %in% (1979:2005))
    sapply(file_read_dup, class)
    head(file_read_dup)
    
    file_read_dup$Date <- as.Date(file_read_dup$Date, format ="%Y-%m-%d")
    file_read_dup <- subset(file_read_dup, format(file_read_dup$Date, "%m")== "02")
    
    
    file_read_dup <- file_read_dup[, min(predicted_Hc),by = hardiness_year]
    # file_read_dup <- file_read_dup[, min(predicted_Hc),by = hardiness_year]
    
    colnames(file_read_dup)
    highest_mean <- colMeans(file_read_dup)
    
    highest_mean[["V1"]]
    
    row_add <- data.frame(file_name, highest_mean[["V1"]], lat, long)
    names(row_add)<- c("Location","location_mean","lat", "long")
    
    cluster_data_tmin <- rbind(cluster_data_tmin, row_add)
    
    
  }
}
write.csv(cluster_data_tmin, file = paste0(file_location, "cluster_HC_min_feb.csv"))
# cluster_data





# ################ tetsing ####################
# 
# file_location <- "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/observed/output_observed_historical_data_45.84375_-120.84375.csv"
# 
# file_name <- basename(file_location)
# file_name
# 
# file_name <- unlist(strsplit(file_name, "_"))
# lat <- file_name[[5]]
# 
# long <- str_extract(file_name[[6]], '.*(?=\\.csv)')
# file_name <- paste0(file_name[[5]],file_name[[6]])
# 
# 
# # removing the .csv from the file name
# library(stringr)
# file_name <- str_extract(file_name, '.*(?=\\.csv)')
# lat
# long
# file_name
# 
# 
# file_read <- read.csv(file = file_location, stringsAsFactors = FALSE)
# sapply(file_read, class)
# 
# file_read_dup <- data.table(file_read)
# file_read_dup <- subset(file_read_dup, file_read_dup$hardiness_year %in% (1979:2005))
# sapply(file_read_dup, class)
# head(file_read_dup)
# 
# summary(file_read_dup)
# 
# write.csv(file_read_dup,file = paste0(file_location, "read_dup.csv"))
# 
# file_read_dup$Date <- as.Date(file_read_dup$Date, format ="%Y-%m-%d")
# file_read_dup <- subset(file_read_dup, format(file_read_dup$Date, "%m")== "01")
# 
# file_read_dup <- file_read_dup[, mean(t_min),by = hardiness_year]
# 
# colnames(file_read_dup)
# highest_mean <- colMeans(file_read_dup)
# 
# highest_mean[["V1"]]
# 
# row_add <- data.frame(file_name, highest_mean[["V1"]], lat, long)
# names(row_add)<- c("Location","location_mean","lat", "long")
# 
# cluster_data_tmin <- rbind(cluster_data_tmin, row_add)
# 
