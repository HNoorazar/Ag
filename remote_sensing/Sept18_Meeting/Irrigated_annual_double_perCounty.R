
rm(list=ls())
library(data.table)
library(dplyr)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)


################################################################################

data_dir <- "/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/"
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"

################################################################################
years <- seq(from = 2016, to = 2018, by = 1)
annual_list <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"), as.is=TRUE)

for (yr in years){
  WSDA <- read.csv(paste0(data_dir, "WSDA_DataTable_", yr, ".csv"), as.is=TRUE)

  # filter irrigated
  WSDA <- filter_out_non_irrigated_datatable(WSDA)

  # filter annual
  WSDA$CropTyp <- tolower(WSDA$CropTyp)
  annual_list$Crop_Type <- tolower(annual_list$Crop_Type)

  WSDA <- WSDA %>% 
          filter(CropTyp %in% annual_list$Crop_Type) %>%
          data.table()

  # WSDA$Irrigtn <- tolower(WSDA$Irrigtn)
  # WSDA$Irrigtn <- as.character(WSDA$Irrigtn)
  # WSDA$Irrigtn[is.na(WSDA$Irrigtn)] <- "na"
  
  typess <- c("Total_Acreage", "Double_Cropped", "Not Double_Cropped")
  counties <- sort(unique(WSDA$county))

  output_tbl <- data.table(matrix(666, nrow = length(counties), ncol = (1+length(typess)) ))
  setnames(output_tbl, old=colnames(output_tbl), new = c("county", typess) )
  output_tbl$county <- counties

  for (row in c(1:nrow(output_tbl))){
    curr <- WSDA %>% filter(county == output_tbl[row]$county)

    total_acr <- sum(curr$ExctAcr)
    
    curr_dbl <- filter_double_by_Notes(curr)
    double_acr <- sum(curr_dbl$ExctAcr)
    NonDouble_acr <- total_acr - double_acr

    row_vals <- c(total_acr, double_acr, NonDouble_acr)
    
    output_tbl[row, 2] <- row_vals[1]
    output_tbl[row, 3] <- row_vals[2]
    output_tbl[row, 4] <- row_vals[3]
  }

  cols <- names(output_tbl)[2:dim(output_tbl)[2]]
  output_tbl[,(cols) := round(.SD, 0), .SDcols=cols]
  
  output_tbl <- output_tbl[order(-Total_Acreage),]

  main_out <- "/Users/hn/Documents/01_research_data/remote_sensing/Sept_18_Meeting//"
  if (dir.exists(main_out) == F) {dir.create(path = main_out, recursive = T)}
  
  out_name <- paste0("dbl_NonDbl_Acr_perCounty_", yr, ".csv")
  write.csv(output_tbl, paste0(main_out, out_name), row.names = F)
}

