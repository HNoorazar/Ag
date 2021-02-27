.libPaths("/data/hydro/R_libs35")
.libPaths()


library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggplot2)
library(reshape2)


# creating a clustering data frame for highest average predicted HC value
# probability_data <- data.frame(Location= character(),probability = numeric(), lat = numeric(),long = numeric(), stringsAsFactors = FALSE)
# sapply(probability_data, class)

# List of files in the location

file_location <- "/data/hydro/users/kraghavendra/hardiness/output_data/observed/"
# file_location <- "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/observed/"

files_Total <- list.files(path= file_location)
files_Total <- files_Total[-c(1:5552)]

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
    
    
    # Code for Facet map for 26 year per location
    
    # print(latlong)
    # print(name)
    
    # plot_path <- "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/Plots/facet/observed/"
    plot_path <- "/data/hydro/users/kraghavendra/hardiness/output_data/Plots/facet/observed/"
    
    # out_dates_loc_obs <- read.csv(file = paste0("C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/observed/output_observed_historical_",latlong,".csv"), stringsAsFactors = FALSE)
    # out_dates_loc_obs
    
    file_read_dup$Date <- as.Date(file_read_dup$Date, format = "%Y-%m-%d")
    # print(out_dates_loc_obs)
    
    # file_read_dup <- subset(out_dates_loc_obs, out_dates_loc_obs$hardiness_year %in% (1979:2014)) 
    setDT(file_read_dup)[, counter := seq_len(.N), by=rleid(hardiness_year)]
    
    # write.csv(out_dates_loc_obs, file = paste0(plot_path,"checking.csv"))
    
    facet_plot <- ggplot()+
      geom_line(data = file_read_dup, aes(x = file_read_dup$counter, y = file_read_dup$predicted_Hc, color = "predicted HC"))+
      geom_line(data = file_read_dup, aes(x = file_read_dup$counter, y = file_read_dup$t_min, color = "tmin"))+
      geom_line(data = file_read_dup, aes(x = file_read_dup$counter, y = file_read_dup$t_max, color = "tmax"))+
      xlab('Years')+
      ylab('Temp(celcius)')+
      ggtitle(file_name)+
      facet_wrap(~ hardiness_year)
    ggsave(facet_plot, filename= paste0(plot_path,file_name,".png"), width = 15, height = 10)
    
    # # Code for probbaility
    # total_years <- length(unique(file_read_dup$hardiness_year))
    # total_years
    # 
    # # selecting  rows with only 1 in CDI
    # file_read_dup <- subset(file_read_dup, file_read_dup$CDI == 1)
    # file_read_dup
    # 
    # unique_years <- length(unique(file_read_dup$hardiness_year))
    # unique_years
    # 
    # probab_year_obs <- unique_years / total_years
    # probab_year_obs
    # 
    # row_add <- data.frame(file_name, probab_year_obs, lat, long)
    # names(row_add)<- c("Location","Probability","lat", "long")
    # 
    # probability_data <- rbind(probability_data, row_add)
    # probability_data
    
  }
}
# write.csv(probability_data, file = paste0(file_location, "proabability_observed.csv"))
