.libPaths("/data/hydro/R_libs35")
.libPaths()



library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggplot2)
library(reshape2)


# creating a clustering data frame for highest average predicted HC value
freezing_data <- data.frame(Location= character(),freezing_years = numeric(), lat = numeric(),long = numeric(), stringsAsFactors = FALSE)
sapply(freezing_data, class)

# List of files in the loaction

file_location <- "/data/hydro/users/kraghavendra/hardiness/output_data/observed/"
# file_location <- "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/observed/"

files_Total <- list.files(path= file_location)
files_Total

for (names in files_Total)
{
  if (names != "consolidated_observed_historical.csv"){
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
    file_read_dup <- subset(file_read_dup, file_read_dup$hardiness_year %in% (1979:2014))
    sapply(file_read_dup, class)
    head(file_read_dup)
    
    # Code for probbaility
    # total_years <- length(unique(file_read_dup$hardiness_year))
    # total_years
    
    # selecting  rows with only 1 in CDI
    file_read_dup <- subset(file_read_dup, file_read_dup$CDI == 1)
    file_read_dup
    
    unique_years <- length(unique(file_read_dup$hardiness_year))
    unique_years
    
    # probab_year_obs <- unique_years / total_years
    # probab_year_obs
    
    row_add <- data.frame(file_name, unique_years, lat, long)
    names(row_add)<- c("Location","freezing_years","lat", "long")
    
    freezing_data <- rbind(freezing_data, row_add)
    freezing_data
    
  }
}
write.csv(freezing_data, file = paste0(file_location, "freezing_observed.csv"))
# cluster_d
#

# ## Plotting the map
# map_plot_location<- "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/Plots/maps/"
# 
# probab_map_data <- read.csv("C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/proabability_observed.csv", stringsAsFactors = FALSE)
# probab_map_data <- select(probab_map_data,long,lat,Probability)
# sapply(probab_map_data, class)
# 
# # creating washington map 
# states <- map_data("state")
# Washington <- subset(states, region == "washington")
# Washington
# 
# Washington_base <- ggplot(data = Washington, mapping = aes(x=long, y=lat, group = group))+
#   coord_fixed(1.3)+
#   geom_polygon(color = "black", fill = "gray" )
# Washington_base
# 
# 
# # Adding counties
# counties <- map_data("county")
# Washington_counties <- subset(counties, region == "washington")
# Washington_counties
# 
# #wroking code for map with predicted Hc
# Washington_base+
#   geom_point(data =probab_map_data,aes(x=long, y= lat,color =Probability,size=0.0001),inherit.aes = FALSE)+
#   geom_polygon(data = Washington_counties, fill =NA, color= "black")+
#   ggtitle('Probability Distribution')+
#   scale_color_gradient(low ="white", high = "red")
# ggsave(paste0(map_plot_location, "Probability.png"), height = 10, width =10)
# 

