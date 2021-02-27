.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggplot2)

# Richland - row 120
# Omak - 110
# Wenatchee - 163
# Walla walla -31

####################################
# read the lat long comaprer file
####################################

input_dir <- "/home/kraghavendra/hardiness/parameters/"
# input_dir <- "C:/Users/Kaushik Acharya/Documents/R Scripts/i_code_in_R/4_kaushik/hardiness/"

grid_AG_compare <- readRDS(paste0(input_dir, "/grid_AG_compare.rds"))
# grid_AG_compare = readRDS(paste0(input_dir,"Output_data/AG_weather/grid_AG_compare.rds"))

# reading the AG weather file

AG_station_name <- read.csv(paste0(input_dir,"AWN_T_P_DAILY.csv"))
# AG_station_name <- read.csv("C:/Users/Kaushik Acharya/Documents/R Scripts/i_code_in_R/4_kaushik/hardiness/input_data/AWN_T_P_DAILY/AWN_T_P_DAILY.csv")

dim(grid_AG_compare)
head(grid_AG_compare)

####### checking for zeros in file and analysis #######
zero_compare <- subset(grid_AG_compare, grid_AG_compare$AG_lat == 0)
length(zero_compare$Station_ID)


# seperating the 9 location with no info on lat and long
grid_AG_compare <- subset(grid_AG_compare, grid_AG_compare$AG_lat != 0)
dim(grid_AG_compare)

# locations seperated for into vector for the for loop
# grid_AG_5 <- grid_AG_compare[4,]
# grid_AG_5 <- subset (grid_AG_compare, grid_AG_compare$Station_ID %in% c(300253, 330104,
                                                                        # 330166, 300133))

# grid_AG_5 <- grid_AG_5[1]
# grid_AG_5

# place <- grid_AG_5$Station_ID
# place

for (place in grid_AG_compare$Station_ID){
    
  input_AG <- readRDS(paste0(input_dir,"AGweather.rds"))
  # input_AG <- readRDS("C:/Users/Kaushik Acharya/Documents/R Scripts/i_code_in_R/4_kaushik/hardiness/Output_data/AG_weather/AGweather.rds")
  
  
  # seperating one AGweather location 
  input_AG <- subset(input_AG, input_AG$Station_ID == place)
  dim(input_AG) 
  head(input_AG)
  
  # converting Date column into date format
  input_AG$Date <- as.Date(input_AG$Date, format = "%Y-%m-%d")
  sapply(input_AG, class)
  
  # selecting start date and end date for the grid data
  start_date <- head(input_AG$Date, 1)
  start_date
  end_date <- tail(input_AG$Date, 1)
  end_date
  
  check_date <- as.Date("2015-08-01", format = "%Y-%m-%d")
  check_date
  
  # if date is not present skip the location
  if (start_date > "2015-08-01"){
    print("Not Enough Data")
    next
  }
  
  
  # location of grid data
  grid_location <- "/data/hydro/users/kraghavendra/hardiness/output_data/observed/"
  # grid_location <- "C:/Users/Kaushik Acharya/Documents/R Scripts/i_code_in_R/4_kaushik/hardiness/Output_data/observed/"
  
  # nearest lat long from compare table 
  tuple_need <- subset(grid_AG_compare, grid_AG_compare$Station_ID == place)
  lat <-  tuple_need$grid_lat #grid_AG_compare$grid_lat[place]
  lat
  long <- tuple_need$grid_long
  long
  
  # Data concerning the nearest lat long location
  input_grid <- read.csv(paste0(grid_location,"output_observed_historical_data_",
                                lat,"_", long,".csv"))
  
  grid_location <- paste0(lat,"_",long)
  grid_location
  
  Station_name <- subset(AG_station_name, AG_station_name$STATION_ID == place)[1, "STATION_NAME"]
  Station_name
  
  dim(input_grid)
  head(input_grid) 
  
  # converting date format of grid data
  input_grid$Date <- as.Date(input_grid$Date, format = "%Y-%m-%d")
  sapply(input_grid, class)
  
  # check whether AG end date is lesser or greater than grid data
  if (end_date < tail(input_grid$Date,1)){
    print("you are good")
    
    # Date of AGweather is lesser
    # Reduce the grid data to end_date
    input_AG <- input_AG[input_AG$Date <= end_date,]
    head(input_AG$Date,1)
    tail(input_AG$Date,1)
    
    # Grid data with the dates
    input_grid <- input_grid[input_grid$Date >= start_date & input_grid$Date <= end_date,]
    head(input_grid$Date,1)
    tail(input_grid$Date,1) 
    
  }else {
    print("you are fucked")
    
    # Date of grid data lesser
    # therefore change the end to that of grid data
    # reduce the Agweather data
    end_date <- tail(input_grid$Date, 1)
    end_date  
    
    # reducing the Ag weather data
    input_AG <- input_AG[input_AG$Date <= end_date,]
    head(input_AG$Date,1)
    tail(input_AG$Date,1)
    
    # Grid data with the dates
    input_grid <- input_grid[input_grid$Date >= start_date & input_grid$Date <= end_date,]
    head(input_grid$Date,1)
    tail(input_grid$Date,1)
    
  }
  
  
  
  #######################################################
  # Custom Theme
  #######################################################
  
  custom_theme <- function () {
    theme_gray() %+replace%
      theme(plot.title = element_text(size=36, face="bold"),
            # panel.background = element_blank(),
            # panel.border = element_blank(),
            panel.spacing=unit(.25, "cm"),
            legend.title = element_text(face="plain", size=36),
            legend.text = element_text(size=24),
            legend.position = "bottom",
            legend.key.size = unit(.65, "cm"),
            strip.text = element_text(size= 24, face="bold", color="black"),
            axis.text = element_text(face="bold", size=24, color="black"),
            axis.ticks = element_line(color = "black", size = .2),
            axis.title.x = element_text(face="bold", size=36, margin=margin(t=10, r=0, b=0, l=0), color="black"),
            axis.title.y = element_text(face="bold", size=36, margin=margin(t=0, r=10, b=0, l=0), color="black",angle = 90))
    }
  
  # check the dimension of the table
  dim(input_AG)
  dim(input_grid)

  # Plot Location
  plot_location <- "/data/hydro/users/kraghavendra/hardiness/output_data/Plots/Comparision/observed/"
  # plot_location <- "C:/Users/Kaushik Acharya/Documents/R Scripts/i_code_in_R/4_kaushik/hardiness/Output_data/Plots/Comparison/"
  
  # checking if the directory is present , if not create a folder
  ifelse(!dir.exists(file.path(plot_location, grid_location)), 
         dir.create(file.path(plot_location, grid_location)), FALSE)
  
  # comparison between the two dataframes
  dim(input_AG)
  dim(input_grid)
  
  
  ##############################################################
  # Creating the merged data frame for further use
  ##############################################################
  
  # merging the table for difference calculation
  Merge_diff <- merge(input_grid,input_AG, by = "Date")
  dim(Merge_diff)
  names(Merge_diff)
  sapply(Merge_diff, class)
  
  # Calculating differences to plot later
  Merge_diff$Tmax <- Merge_diff$t_max.x - Merge_diff$t_max.y
  Merge_diff$Tmin <- Merge_diff$t_min.x - Merge_diff$t_min.y
  Merge_diff$Tmean <- Merge_diff$t_mean.x - Merge_diff$t_mean.y
  Merge_diff$Hc <- Merge_diff$predicted_Hc.x - Merge_diff$predicted_Hc.y
  
  # check if there are CDI values
  sum(Merge_diff$CDI.x)
  sum(Merge_diff$CDI.y)
  
  ###############################
  # plot for comparison of Tmax
  ###############################
  
  Max_temp_compare <- ggplot()+
    geom_line(data = input_AG, aes(x = input_AG$Date, y = input_AG$t_max,
                                   color = "AG weather"))+
    geom_line(data = input_grid, aes(x = input_grid$Date, y = input_grid$t_max,
                                     color = "grid data"))+
    geom_col(data = Merge_diff, aes(x = Merge_diff$Date, y = Merge_diff$Tmax,
                                    color = "diff"))+
    
    xlab('Year')+
    ylab('Temperature (\u00B0C)')+
    ggtitle(paste0("Tmax Comparision - ", grid_location," VS ",Station_name))
  
  Max_temp_compare <- Max_temp_compare + custom_theme()
  
  # Max_temp_compare
  
  # save the plot
  ggsave(plot = Max_temp_compare, paste0(plot_location,grid_location,"/",
                                         "Tmax_comparison.PNG"), dpi = "print", scale = 10)
  
  ################################
  # plot for comparison of Tmin
  ################################
  
  Min_temp_compare <-ggplot()+
    geom_line(data = input_AG, aes(x = input_AG$Date, y = input_AG$t_min,
                                   color = "AG weather"))+
    geom_line(data = input_grid, aes(x = input_grid$Date, y = input_grid$t_min,
                                     color = "grid data"))+
    geom_col(data = Merge_diff, aes(x = Merge_diff$Date, y = Merge_diff$Tmin,
                                    color = "diff"))+
    xlab('Year')+
    ylab('Temperature (\u00B0C)')+
    ggtitle(paste0("Tmin Comparision - ", grid_location," VS ",Station_name))
  
  Min_temp_compare <- Min_temp_compare + custom_theme()
  
  # save the plot
  ggsave(plot = Min_temp_compare, paste0(plot_location, grid_location,
                                         "/","Tmin_comparison.PNG"), dpi = "print",
         scale = 10)
  
  ################################
  # plot for comaprison of Tmean
  ###############################
  
  
  Mean_temp_compare <- ggplot()+
    geom_line(data = input_AG, aes(x = input_AG$Date, y = input_AG$t_mean,
                                   color = "AG weather"))+
    geom_line(data = input_grid, aes(x = input_grid$Date, y = input_grid$t_mean,
                                     color = "grid data"))+
    geom_col(data = Merge_diff, aes(x = Merge_diff$Date, y = Merge_diff$Tmean,
                                    color = "diff"))+
    xlab('Date')+
    ylab('Temperature (\u00B0C)')+
    ggtitle(paste0("Tmean Comparision - ", grid_location," VS ",Station_name))
  
  Mean_temp_compare <- Mean_temp_compare + custom_theme()
  
  # save the plot
  ggsave(plot = Mean_temp_compare, paste0(plot_location, grid_location,
                                          "/","Tmean_comparison.PNG"), 
         dpi = "print", scale = 10)
  
  ###########################################
  # plot for comparison of Preddicted_Hc
  ###########################################
  
  # Hc_temp_compare <-  ggplot()+
  #   geom_line(data = input_AG, aes(x = input_AG$Date, y = input_AG$predicted_Hc,
  #                                  color = "AG weather"))+
  #   geom_line(data = input_grid, aes(x = input_grid$Date, y = input_grid$predicted_Hc,
  #                                    color = "grid data"))+
  #   geom_col(data = Merge_diff, aes(x = Merge_diff$Date, y = Merge_diff$Hc,
  #                                   color = "diff"))+
  #   
  #   facet_wrap(~ input_AG$hardiness_year)+
  #   xlab('Date')+
  #   ylab('Temperature')+
  #   ggtitle(paste0("Hardiness Comparision - ", place))
  # 
  # Hc_temp_compare <- Hc_temp_compare + custom_theme()
  # 
  # # save the plot
  # ggsave(plot = Hc_temp_compare, paste0(plot_location,place,
  #                                       "/","Hc_comparison.PNG"), 
  #        dpi = "print", scale = 10)
  # 
  
  ############################################################
  # For the purpose of facet wrap
  ############################################################
  
  Merge_diff$Date <- as.Date(Merge_diff$Date, format = "%Y-%m-%d")
  
  head(Merge_diff)
  
  Merge_diff <- subset(Merge_diff, Merge_diff$hardiness_year.x != 0)
  
  # creating a column called counter to faciliatate facet plotting
  # as counter is common for all points
  setDT(Merge_diff)[, counter := seq_len(.N), by=rleid(hardiness_year.x)]
  
  dim(Merge_diff)
  # length(Merge_diff$CDI.y)
  
  # subset of CDI counts greater than 0
  
  # preparing table for CDI count for AG weather
  Merge_count_AG <- subset(Merge_diff, Merge_diff$CDI.y > 0)
  Merge_count_AG$CDI.y <- Merge_count_AG$predicted_Hc.y
  Merge_count_AG <- Merge_count_AG %>% select (counter, CDI.y,hardiness_year.x)
  Merge_count_AG
  
  
  # preparing table for CDI count for grid weather
  Merge_count_grid <- subset(Merge_diff, Merge_diff$CDI.x > 0)
  Merge_count_grid$CDI.x <- Merge_count_grid$predicted_Hc.x
  Merge_count_grid <- Merge_count_grid %>% select (counter, CDI.x,hardiness_year.x)
  Merge_count_grid
  
  # facet plot for hardiness
  Hc_temp_facet <-  ggplot()+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$predicted_Hc.x,
                                   color = "AG weather"))+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$predicted_Hc.y,
                                     color = "grid data"))+
    geom_col(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$Hc,
                                    color = "diff"))+
    geom_point(data = Merge_count_AG, aes(x = Merge_count_AG$counter, y = Merge_count_AG$CDI.y),
                                          shape = 24, size = 3, fill = "yellow")+
    geom_point(data = Merge_count_grid, aes(x = Merge_count_grid$counter, y = Merge_count_grid$CDI.x),
                                          shape = 21, size = 3, fill = "blue")+
    scale_x_discrete(breaks = c(0,100,200),
                     limit = c(0, 100, 200),
                     labels = c("Sep","Dec","Mar"))+
    facet_wrap(~ hardiness_year.x)+
    xlab('Months')+
    ylab('Cold Hariness (\u00B0C)')+
    ggtitle(paste0("Hardiness Comparision - ", grid_location," VS ",Station_name))
  
  Hc_facet <- Hc_temp_facet +  custom_theme()
  
  # save the plot
  ggsave(plot = Hc_facet, paste0(plot_location,grid_location, "/",
                                 "Hc_comparison_facet.PNG"), 
         dpi = "print", scale = 10)
  
  
  
  ##################################
  # facet plot for Tmax 
  ##################################
  
  Max_temp_facet <-  ggplot()+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$t_max.x,
                                     color = "AG weather"))+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$t_max.y,
                                     color = "grid data"))+
    geom_col(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$Tmax,
                                    color = "diff"))+
    scale_x_discrete(breaks = c(0,100,200),
                     limit = c(0, 100, 200),
                     labels = c("Sep","Dec","Mar"))+
    facet_wrap(~ hardiness_year.x)+
    xlab('Months')+
    ylab('Temperature (\u00B0C)')+
    ggtitle(paste0("Tmax Comparision facet - ", grid_location," VS ",Station_name))
  
  Max_temp_facet <- Max_temp_facet + custom_theme()
  
  # save the plot
  ggsave(plot = Max_temp_facet, paste0(plot_location, grid_location,"/",
                                       "Tmax_comparison_facet.PNG"), 
         dpi = "print", scale = 10)
  
  ##########################
  # facet plot for Tmin
  ##########################
  
  Min_temp_facet <-  ggplot()+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$t_min.x,
                                     color = "AG weather"))+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$t_min.y,
                                     color = "grid data"))+
    geom_col(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$Tmin,
                                    color = "diff"))+
    scale_x_discrete(breaks = c(0,100,200),
                     limit = c(0, 100, 200),
                     labels = c("Sep","Dec","Mar"))+
    facet_wrap(~ hardiness_year.x)+
    xlab('Months')+
    ylab('Temperature (\u00B0C)')+
    ggtitle(paste0("Tmin Comparision Facet - ", grid_location," VS ",Station_name))
  
  Min_temp_facet <- Min_temp_facet + custom_theme()
  
  # save the plot
  ggsave(plot = Min_temp_facet, paste0(plot_location, grid_location,"/",
                                       "Tmin_comparison_facet.PNG"), 
         dpi = "print", scale = 10)
  
  ###############################
  # facet plot Tmean
  ###############################
  
  Mean_temp_facet <-  ggplot()+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$t_mean.x,
                                     color = "AG weather"))+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$t_mean.y,
                                     color = "grid data"))+
    geom_col(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$Tmean,
                                    color = "diff"))+
    scale_x_discrete(breaks = c(0,100,200),
                     limit = c(0, 100, 200),
                     labels = c("Sep","Dec","Mar"))+
    facet_wrap(~ hardiness_year.x)+
    xlab('Months')+
    ylab('Temperature (\u00B0C)')+
    ggtitle(paste0("Tmean Comparision Facet - ", grid_location," VS ",Station_name))
  
  Mean_temp_facet <- Mean_temp_facet + custom_theme()
  
  # save the plot
  ggsave(plot = Mean_temp_facet, paste0(plot_location, grid_location, "/",
                                        "Tmean_comparison_facet.PNG"), 
         dpi = "print", scale = 10)
  
  ################################
  # Bar plots with difference
  ################################
  
  # 
  # Tmax_diff<-ggplot()+
  #   geom_bar(data = Merge_diff, aes(x = Merge_diff$Date, y = Merge_diff$max, 
  #                                   color = "Tmax" ), stat = "identity")+
  #   # geom_bar(data = Merge_diff, aes(x = Merge_diff$counter, y = Merge_diff$min,
  #   #                                 color = "Tmin"), stat = "identity", alpha = 0.3, 
  #   #                                       position = "dodge")+
  #   xlab('Date')+
  #   ylab('Temperature')+
  #   ggtitle('Tmax difference')#+
  #   # facet_wrap(~ hardiness_year.x)
  # 
  # Tmax_diff
  # 
  # Tmin_diff <- ggplot()+
  #   geom_bar(data = Merge_diff, aes(x = Merge_diff$Date, y = Merge_diff$min,
  #                                              color = "Tmin"), stat = "identity")+
  #   xlab('Date')+
  #   ylab('Temperature')+
  #   ggtitle('Tmin difference')#+
  #   # facet_wrap(~ hardiness_year.x)
  # 
  # Tmin_diff
  # 
  # library(gridExtra)
  # grid_plot <- grid.arrange(Tmax_diff,Tmin_diff,ncol = 1)
  # 
  # ggplot(data = Merge_diff)+
  #   grid_plot+ facet_wrap(hardiness_year.x)
  # 
  # 
  # # trying per year scheme
  # total_years <- unique(Merge_diff$year.x)
  # total_years
  # 
  
  ###########################################################
  # Facet Data with difference comparision
  ##########################################################
  
  names(Merge_diff)
  comp_Merge <- Merge_diff %>% select(Date, Tmax, Tmin, Tmean, hardiness_year.x, counter)
  
  comp_Merge_melt <- melt(comp_Merge, id = c("Date","hardiness_year.x","counter"))
  head(comp_Merge_melt)
  
  Multi_grid_compare<- ggplot()+
    geom_bar(data = comp_Merge_melt, aes(x = comp_Merge_melt$counter, 
                                         y = comp_Merge_melt$value, fill = factor(variable))
                                        , stat = "identity")+
    scale_x_discrete(breaks = c(0,100,200),
                     limit = c(0, 100, 200),
                     labels = c("Sep","Dec","Mar"))+
    facet_grid( ~ hardiness_year.x ~variable, scales = "free")+
    xlab('Months')+
    ylab('Temperature (\u00B0C)')+
    ggtitle(paste0("Temperature difference Comparision - ", grid_location," VS ",Station_name))
  
  max_mean_mean_facet <- Multi_grid_compare + custom_theme()
  
  
  ggsave(plot = max_mean_mean_facet, paste0(plot_location, grid_location, "/",
                                            "Temperature_difference_facet_long.PNG"), 
           dpi = "print", scale = 10)
    
  
  ################################################
  # Stacked area plot
  ###############################################
  
  comp_area <- ggplot()+
    geom_area(data = comp_Merge_melt, aes(counter, value, fill = factor(variable)),
                                          position = 'stack')+
    facet_wrap(~ hardiness_year.x,scales = "free")+
    scale_fill_manual(values = c("Tmax" = "red", "Tmin" = "blue", "Tmean" = "yellow"))+
    scale_x_discrete(breaks = c(0,100,200),
                     limit = c(0, 100, 200),
                     labels = c("Sep","Dec","Mar"))+
    xlab('Months')+
    ylab('Temperature difference (\u00B0C)')+
    ggtitle(paste0("Area plot for temperaure difference - ", grid_location," VS ",Station_name))
  
  comp_area <- comp_area + custom_theme()
  
  
  ggsave(plot = comp_area, paste0(plot_location, grid_location,"/",
                                  "Area_plot.PNG"), 
         dpi = "print", scale = 10)
  
  
  #######################################################
  # Predicted HC with different colors and CDI only
  #######################################################
  
  names(Merge_diff)
  head(Merge_diff)
  just_Hc_diff <- Merge_diff %>% select (Date, year.x, Hc, counter, hardiness_year.x)
  dim(just_Hc_diff)
  # head(just_Hc_diff, 50)
  
  # adding a column to check for postives and negatives
  just_Hc_diff$sign <- ifelse(just_Hc_diff$Hc >= 0, "positive", "negative")
  
  # checking for critical events and plotting for 
  
  CDI_grid <- Merge_diff %>% select (Date, counter, CDI.x, hardiness_year.x)
  CDI_grid <- subset(CDI_grid, CDI_grid$CDI.x > 0)
  CDI_grid
  
  CDI_AG <- Merge_diff %>% select (Date, counter, CDI.y, hardiness_year.x)
  CDI_AG <- subset(CDI_AG, CDI_AG$CDI.y > 0)
  CDI_AG
  
  # head(just_Hc_diff$Hc,50)
  
  # plot for difference of Hc with critical days
  just_hc <- ggplot()+
    geom_bar(data = just_Hc_diff, aes(x = just_Hc_diff$counter, y = just_Hc_diff$Hc,
                                      fill = sign), stat = "identity")+
    scale_fill_manual(values = c("positive" = "#56B4E9", "negative" = "#E69F00"))+
    geom_point(data = CDI_AG, aes(x = CDI_AG$counter, y = CDI_AG$CDI.y),
               shape = 24, size = 3, fill = "yellow")+
    geom_point(data = CDI_grid, aes(x = CDI_grid$counter, y = CDI_grid$CDI.x),
               shape = 21, size = 3, fill = "blue")+
    scale_x_discrete(breaks = c(0,100,200),
                     limit = c(0, 100, 200),
                     labels = c("Sep","Dec","Mar"))+
    facet_wrap(~ hardiness_year.x,scales = "free")+
    xlab('Month')+
    ylab('Cold Hradiness difference (\u00B0C)')+
    ggtitle(paste0('Predicted HC diferrence with critical days - ', grid_location," VS ",Station_name))
   
  just_hc <- just_hc + custom_theme()
  
  # just_hc
  
  ggsave(plot = just_hc, paste0(plot_location, grid_location,"/",
                                "HC_difference.png"), 
          dpi = "print", scale = 10)
   
  
  ############################################
  # Hardiness facet with critical column plot
  ############################################
  
  names(Merge_diff)
  head(Merge_diff)
  dim(Merge_diff)
  
  # # comp_bar <- Merge_diff %>% select(Date, predicted_Hc.x, predicted_Hc.y,
  #                                   hardiness_year.x, counter)
  # 
  # comp_bar_melt <- melt(comp_bar, id = c("Date","hardiness_year.x","counter"))
  # head(comp_bar_melt)
  
  comp_grid_bar <- Merge_diff %>% select (Date, counter,t_min.x, t_min.y, CDI.x, hardiness_year.x)
  comp_grid_bar <- subset(comp_grid_bar, comp_grid_bar$CDI.x > 0)
  comp_grid_bar
  
  comp_AG_bar <- Merge_diff %>% select (Date, counter,t_min.x, t_min.y, CDI.y, hardiness_year.x)
  comp_AG_bar <- subset(comp_AG_bar, comp_AG_bar$CDI.y > 0)
  comp_AG_bar
  
  # month_label <- c(0 = "Sep",100 = "Dec", 200 = "Mar")
  
  comp_grid_tmin<- ggplot()+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter,y = Merge_diff$predicted_Hc.x, 
                                        color = "Grid_data"))+
    geom_line(data = Merge_diff, aes(x = Merge_diff$counter, y= Merge_diff$predicted_Hc.y,
                                     color = "AG_data"))+
    geom_point(data = comp_grid_bar, aes(x = comp_grid_bar$counter, 
                                       y = comp_grid_bar$t_min.x),
               shape = 21, size = 3, fill = "blue")+
    geom_point(data = comp_grid_bar, aes(x = comp_grid_bar$counter, 
                                         y = comp_grid_bar$t_min.y),
               shape = 24, size = 3, fill = "blue")+
    geom_point(data = comp_AG_bar, aes(x = comp_AG_bar$counter, 
                                     y = comp_AG_bar$t_min.x),
               shape = 21, size = 3, fill = "yellow")+
    geom_point(data = comp_AG_bar, aes(x = comp_AG_bar$counter, 
                                       y = comp_AG_bar$t_min.y),
               shape = 24, size = 3, fill = "yellow")+
    scale_x_discrete(breaks = c(0,100,200),
                     limit = c(0, 100, 200),
                      labels = c("Sep","Dec","Mar"))+
    facet_wrap(~ hardiness_year.x)+
    xlab('Days')+
    ylab('Temperature (\u00B0C)')+
    ggtitle(paste0("Hardiness with Tmin at critical temperature - ", grid_location," VS ",Station_name))
  
  
  comp_grid_tmin<- comp_grid_tmin + custom_theme()
  
  # comp_grid_tmin
  
  ggsave(plot = comp_grid_tmin, paste0(plot_location, grid_location,"/",
                                "Hardiness_with_Tmin.PNG"), 
         dpi = "print", scale = 10)
  
  ###########################################
  # Density
  #########################################
  
  head(Merge_diff)
  sapply(Merge_diff, class)
  comp_Merge_melt <- as.data.frame(comp_Merge_melt)
  head(comp_Merge_melt)
  
  density_plot <- ggplot(comp_Merge_melt, aes(x = value, fill = factor(variable)))+
    geom_density(position = "stack")+
    scale_fill_manual(values = c("Tmax" = "red", "Tmin" = "blue", "Tmean" = "yellow"))+
    facet_wrap( ~ hardiness_year.x, scales = "free")+
    ylab('Density')+
    xlab('Temperature difference (\u00B0C)')+
    ggtitle(paste0('Density Plot of Temperature difference - ', grid_location," VS ",Station_name))
    
  density_plot <- density_plot + custom_theme()  
  
  ggsave(plot = density_plot, paste0(plot_location,grid_location,"/",
                                       "Density_stack.PNG"), dpi = "print", scale = 10)
  
  
  
  # geom_density(as.data.frame(Merge_diff), aes(x = min))
  
  # This is manual work not needed good that you wrote, but there was a better way
   # for (one_year in total_years){
   #   one_year_data <- subset(Merge_diff, Merge_diff$year.x == one_year)
   #   print(dim(one_year_data))
   #   
   #   Tmax_one <- ggplot()+
   #     geom_bar(data = one_year_data, aes(x = one_year_data$counter, y = one_year_data$max,
   #                                     color = "Tmax"), stat = "identity")+
   #     xlab('Date')+
   #     ylim(-15,15)+
   #     ylab('Temperature')
   #     # ggtitle('Tmin difference')
   #   
   #   Tmin_one <- ggplot()+
   #     geom_bar(data = one_year_data, aes(x = one_year_data$counter, y = one_year_data$min,
   #                                        color = "Tmin"), stat = "identity")+
   #     xlab('Date')+
   #     ylim(-15,15)+
   #     ylab('Temperature')
   #     # ggtitle('Tmin difference')
   #   
   #   Tmean_one <- ggplot()+
   #     geom_bar(data = one_year_data, aes(x = one_year_data$counter, y = one_year_data$mean,
   #                                        color = "Tmean"), stat = "identity")+
   #     xlab('Date')+
   #     ylim(-15,15)+
   #     ylab('Temperature')
   #     
   #     grid_one <- grid.arrange(Tmax_one,Tmin_one,Tmean_one, ncol = 1)
   #     # ggsave(output_dir)
   #     
   #     ggsave (grid_one, file = paste0(plot_location, one_year, ".png"),
   #             height = 10, width = 8)
   #    
   #   
   #     
   # }
   # grid_one
   

  }

