####
#### This module generates #s in the following link ()
#### https://docs.google.com/document/d/18KX24FkL70_Xhxagwx9EBRWeQmz-Ud-iuTXqnf9YXnk/edit?usp=sharing
#### Chapter 3.
####

rm(list=ls())
library(data.table)
library(rgdal)
library(dplyr)
library(sp)
# library(sf)
library(foreign)

source_1 = "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/R/remote_core.R"
source(source_1)
options(digits=9)
options(digit=9)

##############################################################################

data_dir <- paste0("/Users/hn/Documents/01_research_data/remote_sensing/01_Data_part_not_filtered/")
param_dir <- "/Users/hn/Documents/00_GitHub/Ag/remote_sensing/parameters/"
##############################################################################

SF_year = 2017
given_county = "Grant"

WSDA <- read.csv(paste0(data_dir, "WSDA_DataTable_", SF_year, ".csv"))
WSDA$CropType <- tolower(WSDA$CropTyp) 

########
######## pick eastern part only
########

Eastern <- pick_eastern_counties(WSDA)

########
######## Filter out non-irrigated
########

Eastern_irrigated <- filter_out_non_irrigated_datatable(Eastern)
dim(Eastern_irrigated)
sum(Eastern_irrigated$ExctAcr)

########
######## filter out Perennials (i.e. keep annuals)
########
double_crop_pots <- read.csv(paste0(param_dir, "double_crop_potential_plants.csv"))

Eastern_irrigated_annuals <- Eastern_irrigated %>% 
                             filter(CropTyp %in% double_crop_pots$Crop_Type) %>% 
                             data.table()

dim(Eastern_irrigated_annuals)
sum(Eastern_irrigated_annuals$ExctAcr)

########
######## compute double cropped fields by Notes column
########
Eastern_irrigated_annuals_doubleNotes <- filter_double_by_Notes(Eastern_irrigated_annuals)
dim(Eastern_irrigated_annuals_doubleNotes)
sum(Eastern_irrigated_annuals_doubleNotes$ExctAcr)

########
######## compute double cropped fields by Notes column and last survey date
########
Eastern_irrigated_annuals_doubleNotes_lastSurveyed <- filter_lastSrvyDate(Eastern_irrigated_annuals_doubleNotes, year = 2017)
dim(Eastern_irrigated_annuals_doubleNotes_lastSurveyed)
sum(Eastern_irrigated_annuals_doubleNotes_lastSurveyed$ExctAcr)


########
######## Do the same for the given_county
########

given_county_irrigated <- Eastern_irrigated %>% 
                          filter(county == given_county) %>% 
                          data.table()

dim(given_county_irrigated) # 13793
sum(given_county_irrigated$ExctAcr) # 521766

given_county_irrigated_annuals <- given_county_irrigated %>% 
                                  filter(CropTyp %in% double_crop_pots$Crop_Type) %>% 
                                  data.table()

dim(given_county_irrigated_annuals) # 4575
sum(given_county_irrigated_annuals$ExctAcr) # 256314


########
######## compute double cropped fields by Notes column
########

given_county_irrigated_annuals_doubleNotes <- filter_double_by_Notes(given_county_irrigated_annuals)
dim(given_county_irrigated_annuals_doubleNotes) # 241
sum(given_county_irrigated_annuals_doubleNotes$ExctAcr) # 16411

########
######## compute double cropped fields by Notes column and last survey date
########
given_county_irrigated_annuals_doubleNotes_lastSurveyed <- filter_lastSrvyDate(given_county_irrigated_annuals_doubleNotes, year = 2017)
dim(given_county_irrigated_annuals_doubleNotes_lastSurveyed)
sum(given_county_irrigated_annuals_doubleNotes_lastSurveyed$ExctAcr)


