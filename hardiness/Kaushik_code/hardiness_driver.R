.libPaths("/data/hydro/R_libs35")
.libPaths()


library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggplot2)

options(digits=9)

args = commandArgs(trailingOnly = TRUE)
# args[1] = 1

source_path = "/home/kraghavendra/hardiness/hardiness_core.R"
# source_path = "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/hardiness_core.R"
source(source_path)


param_dir = "/home/kraghavendra/hardiness/parameters/"
# param_dir = "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/parameters/"


#Reading a .dbf file
library(foreign)
#VIC_file = paste0(param_dir, "VICID_CO",".DBF")
#VIC_file <- read.dbf(VIC_file, as.is = FALSE)



# read parameters

input_params  = data.table(read.csv(paste0(param_dir, "input_parameters", ".csv")))
variety_params = data.table(read.csv(paste0(param_dir, "variety_parameters", ".csv")))

climate_model = data.table(read.csv(paste0(param_dir,"config",".csv")))
#climate_model 
climate_model <- climate_model[climate_model$ARRAYID == args[1],]
if(climate_model$Scenario == "historical"){
	hist = TRUE

}else{
	hist = FALSE
}

# hist
output_dir <- "/data/hydro/users/kraghavendra/hardiness/output_data/"
# output_dir <- "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/output_data/"


#column with state ID's
#VIC_file

# Filtered columns of latitude and longitude for the states ID, WA, OR and MT  
#VIC_file_filtered <- subset(VIC_file, STATE %in% c("ID", "WA", "OR", "MT"))[,5:7]

# count(VIC_file_filtered) # take a count to cross-verify later

#Concatenation of 2 columns to a column using mutate dunction
library(tidyverse)
#VIC_file_filtered_mutate <- VIC_file_filtered %>% mutate(locations = paste('data', VIC_file_filtered$VICCLAT,VIC_file_filtered$VICCLON, sep = "_"))

# count(VIC_file_filtered_mutate) # cross-verifying count here, if the number of rows is same kudos!!! You havent screwed up, YET!!!

# Write it into a csv file - writing only the locations, so that they can be used to extract files from aeolus for running the model
# write.csv(VIC_file_filtered_mutate$locations, file = "lat_long_list.csv",row.names = FALSE)
#VIC_file_filtered_mutate

# df <-(VIC_file_filtered_mutate$locations)[1:10,] - This was written to select only location from the df

#df <- VIC_file_filtered_mutate[1,] %>% select("locations", "COUNTY")
# df <- df[1:10,]
#df
# count(df) - check the number of rows in df
 
input_file_path = "/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/"
# input_file_path = "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/input_data/"
input_file_path <- paste0(input_file_path,climate_model$Model,"/",climate_model$Scenario,"/")
# input_file_path
filename = paste0(climate_model$Location)
#for (filename in df$locations){
  # print(filename)
  # setwd(input_file_path)
  
file_found <- list.files(path= input_file_path,pattern = filename)
  # file_found<-paste0("Yayyy!!!! I found this file",files)
  # print(file_found)
file_found <- paste0(input_file_path,file_found)
  # print(file_found)
  
meta_data <- data.table(read_binary(file_path = file_found, hist = hist , no_vars= 4))
  # print(head(meta_data))
# meta_data
  # mutate the table to have desired columns
  # Really proud of this code, but this can be  better
  # To calculate the jday, we combined the column of day, month and year and then converted it into date format, then into jday format and then to numeric
  # It was converted to numeric because the "%j" format is from 1-366, we needed 0-365, hence subtracting 1 from the number obtained
  # Try a more cleaner way, if you find time and still have interest
  
# meta_data <- meta_data %>% mutate(Date = paste0(meta_data$day,"/",meta_data$month,"/",meta_data$year))
meta_data <- meta_data %>% mutate(Date = paste0(meta_data$month,"/",meta_data$day,"/",meta_data$year))
# meta_data <- meta_data %>% mutate(jday = as.numeric(format(as.Date(meta_data$Date,format = "%d/%m/%y"),"%j"))-1) 
meta_data <- meta_data %>% mutate(jday = as.numeric(format(as.Date(meta_data$Date,format = "%m/%d/%y"),"%j"))-1)   
# print(meta_data)
  
  
  # Add T_mean for the table, calculationg mean from the two rows of tmax and tmin
meta_data <- meta_data %>% mutate(T_mean = (meta_data$tmax + meta_data$tmin)/2)
  # print(count(meta_data))
  
  # Add loaction, adding COUNTY name for now
  # print(filename)
#meta_data <- meta_data %>% mutate(Location = df$COUNTY[df$locations == filename])
  
  
meta_data <- meta_data %>% select("Date", "year", "jday", "T_mean", "tmax", "tmin")
  # print(head(meta_data))

# meta_data$Date[1]
# grepl("1/1",meta_data$Date[1])

  
# print(head(meta_data))
  
subDir <- paste0(climate_model$Model, "/", climate_model$Scenario, "/")
# meta_data$year[1]


start_time <- Sys.time()
output <- hardiness_model(data = meta_data, input_params = input_params, variety_params = variety_params)
end_time <- Sys.time()
time_taken <- end_time - start_time
print(time_taken)


# head(output)
# sapply(output, class)
# test <- output$year[1]-1
# test

# working code for calculation of hardiness year
# output$hardiness_year <- ifelse(output$jday <=134,output$year-1,
# ifelse(output$jday >=243 & output$jday < 367,output$year,0))


output$hardiness_year <- ifelse(leap_year(output$year),ifelse(output$jday <=136,output$year-1,
                                ifelse(output$jday >=244 & output$jday < 367,output$year,0)),
                                ifelse(output$jday <=135,output$year-1,
                                       ifelse(output$jday >=243 & output$jday < 366,output$year,0)))

# write.csv(output, file ="C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/Output_data/check.csv")
# leap_year(output$year[1])
output_CDI = data.table(matrix(NA,nrow =1, ncol =94))
colnames(output_CDI) <- paste(2006:2099,sep =" ")

# find the count anamolies per hardiness year
for (i in colnames(output_CDI))
{
  # print(output_CDI[[i]])
  # print(years)
  # print(output_CDI[[i]])
  output_CDI[[i]] = sum(subset(output, hardiness_year == i)$CDI)
  # print(output_CDI[["i"]])
  # output_CDI$i<- 
  # output_CDI = sum(subset(output, year==i)$CDI)
  # output_CDI$`2006` = sum(subset(output, year==i)$CDI)
  # print(value)
  # print(output_CDI$i)
}

output_CDI <- cbind(Time_elapsed = time_taken, output_CDI )
output_CDI <- cbind(Scenario = climate_model$Scenario,output_CDI)
output_CDI <- cbind(Model = climate_model$Model,output_CDI)
output_CDI <- cbind(Location= climate_model$Location,output_CDI)


# output_CDI
# rm(output_CDI)
# sum(subset(output, year == 2010)$CDI)

dir.create(file.path(output_dir, climate_model$Model))
dir.create(file.path(paste0(output_dir, climate_model$Model),climate_model$Scenario))
write.csv(output, file = paste0(output_dir, subDir, "output_", filename,".csv"), row.names=FALSE)

# write.csv(output_CDI,file = paste0(output_dir,subDir,"consolidated_",climate_model$Model,"_",climate_model$Scenario,".csv"),row.names = FALSE,append = TRUE)
write.table(output_CDI,paste0(output_dir,subDir,"consolidated_",climate_model$Model,"_",climate_model$Scenario,".csv"),sep = ",",col.names = !file.exists(paste0(output_dir,subDir,"consolidated_",climate_model$Model,"_",climate_model$Scenario,".csv")),row.names= FALSE,append = TRUE )  
  
  # This is the code for plotting the map
  # source_path = "C:/Users/Kaushik Acharya/Documents/Research Task - Kirti/4_kaushik/hardiness/plot_core.R"
  # source(source_path)
  # 
  # plot_dir <- output_dir
  # out_name = paste0("model_output",filename)
  # plot_hardiness(output, plot_dir, out_name)
  
#}

