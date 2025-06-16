## damn Kamiak loads intel compiler that has problem with installing cli.
# Or do module load intel/20.2
# module load gcc
# module load gcc/14.2
# module load r/4.1.0
# module load r/4.4.1 # needs intel compilter. and that was not able to solve things.
# install.packages("earlywarnings")
# install.packages("earlywarnings", dependencies=TRUE, repos="http://cran.r-project.org")
# install.packages("forecast", dependencies=TRUE, repos="http://cran.r-project.org")

####################################
# is the size of the rolling window expressed as percentage of the timeseries length
# (must be numeric between 0 and 100). Default is 50%
#
###################################################################
###################################################################
####
#### 1. Load packages ---------------------------------------------
rm(list=ls())
#.libPaths("/data/hydro/R_libs35")
.libPaths("/home/h.noorazar/R/lib")
.libPaths()

library(data.table)
library(earlywarnings)
library(dplyr)
# library(ggmap) # loads ggplot2
library(forecast)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 9)
options(digit=9)

######################################################################
##                                                                  ##
##              Terminal/shell/bash arguments                       ##
##                                                                  ##
######################################################################

args = commandArgs(trailingOnly=TRUE)
ewr_type = args[1]

###################################################################
###################################################################
####
#### 2. Data Directories ------------------------------------------
dir_base <- "/data/project/agaid/h.noorazar/rangeland_bio/"
database <- paste0(dir_base, "/Data/")
reOrganized_dir = paste0(database, "reOrganized/") 

out_dir <- paste0(database, "earlyWarnings_onlyLinear/")
if (dir.exists(file.path(out_dir)) == F) {
    dir.create(path = file.path(out_dir), recursive = T)
}

plot_dir <- paste0(dir_base, "/plots/earlyWarnings_onlyLinear/", ewr_type, "/")

if (dir.exists(file.path(plot_dir)) == F) {
    dir.create(path = file.path(plot_dir), recursive = T)
}
setwd(plot_dir)

# dev.new(file = paste0(plot_dir, "my_plot.pdf"))
###################################################################
###################################################################
####
#### 3. Read Files ------------------------------------------------

anpp <- data.table(read.csv(paste0(reOrganized_dir, "bpszone_ANPP_no2012.csv")))
anpp[, c("pixel_count", "groupveg", "area_sqMeter") := NULL] # get rid of unwanted columns
setcolorder(anpp, c("fid", "year"))
anpp[, mean_lb_per_acr := as.numeric(mean_lb_per_acr)]

# anpp[, breakpoint_count := NA_integer_]
# anpp[, breakpoint_years := NA_character_]
###################################################################
###################################################################
####
#### 4. Operate ------------------------------------------------------
FIDs = unique(anpp$fid)

# Test if it works with one FID on Kamiak:
test = FALSE
if (test){
  anpp = anpp[anpp$fid==1]
  out_dir <- paste0(database, "earlyWarningsTest/")
  if (dir.exists(file.path(out_dir)) == F) {dir.create(path = file.path(out_dir), recursive = T)}
  FIDs = unique(anpp$fid)
  print ("This is a test")
}


setorder(anpp, fid, year)

########################################################
#######
#######              ddjnonparam_ews
#######  This is disaster. It wants to plot stuff.
#######
# Initialize an empty list to store results
detrending_ = "linear"
if (ewr_type == "generic"){
    ########################################################
    #######
    #######  generic_ews
    #######  This one is nice
    #######
    generic_ews_results_list_separate <- list()
    for (a_fid in FIDs) {
      
      wd_dir <- paste(plot_dir, paste0("fid", a_fid), sep="/")
      if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
      setwd(wd_dir)

      # Subset the data for the current fid
      subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)
      
      # Apply the function and store the result
      graphics.off(); while (dev.cur() > 1) {dev.off()}

      wd_dir <- paste(plot_dir, paste0("fid", a_fid), detrending_, sep="/")
      if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
      setwd(wd_dir)

      graphics.off(); while (dev.cur() > 1) {dev.off()}
      # output_fig <- tempfile(fileext = ".pdf")
      # pdf(output_fig)
      output <- generic_ews(subset_data, winsize = 50,
                            detrending = detrending_,
                            bandwidth = NULL, span = NULL, degree = NULL,
                            logtransform = FALSE, interpolate = FALSE, AR_n = FALSE, powerspectrum = FALSE)
      pdf(tempfile(fileext = paste0("fid", a_fid, "_", detrending_, ".pdf")))
      graphics.off(); while (dev.cur() > 1) {dev.off()}
      generic_ews_results_list_separate[[paste(detrending_, as.character(a_fid), sep="_")]] <- output
    
    }
    saveRDS(generic_ews_results_list_separate, file = paste0(out_dir, "generic_ews_results_list_linear", ".rds"))
  } else if (ewr_type == "qda"){
    qda_ews_results_list_separate <- list()
    for (a_fid in FIDs) {
      subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)

      print (paste0("line 272 FID: ", a_fid, ", detrending_:", detrending_))
      wd_dir <- paste(plot_dir, paste0("fid", a_fid), detrending_, sep="/")
      if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
      setwd(wd_dir)

      graphics.off(); while (dev.cur() > 1) {dev.off()}
      try({pdf(tempfile(fileext = ".pdf"))
           graphics.off(); while (dev.cur() > 1) {dev.off()}
           output <- qda_ews(subset_data, param = NULL, winsize = 10,
                             detrending = detrending_,
                             bandwidth = NULL, boots = 100, s_level = 0.05, cutoff = 0.05, 
                             detection.threshold = 0.002, grid.size = 10, logtransform = FALSE, interpolate = FALSE)
           text_out <- list(indicators = output$indicators, trend = output$trends)
           qda_ews_results_list_separate[[paste(detrending_, as.character(a_fid), sep="_")]] <- output
           pdf(tempfile(fileext = paste0("fid", a_fid, "_detrending_" , detrending_, ".pdf")))
           graphics.off(); while (dev.cur() > 1) {dev.off()}
          }, silent = TRUE)
      graphics.off(); while (dev.cur() > 1) {dev.off()}

    }
    saveRDS(qda_ews_results_list_separate, file = paste0(out_dir, "qda_ews_results_list_linear", ".rds"))
  } else if (ewr_type == "sensitivity"){
    sensitivity_ews_separate <- list()
    for (a_fid in FIDs) {
      subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)
      ###
      ### run for each indicator and deterending separately.
      ### 
      for (indicator_ in c("ar1", "sd", "acf1", "sk", "kurt", "cv", "returnrate", "densratio")){
        wd_dir <- paste(plot_dir, paste0("fid", a_fid), paste(indicator_, detrending_, sep="_"), sep="/")
        if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
        setwd(wd_dir)
      
        graphics.off(); while (dev.cur() > 1) {dev.off()}
        try({output <- sensitivity_ews(subset_data, indicator = indicator_, detrending = detrending_, 
                                       winsizerange = c(25, 75), incrwinsize = 25,
                                       bandwidthrange = c(5, 100), spanrange = c(5, 100), 
                                       degree = NULL, incrbandwidth = 20, incrspanrange = 10, 
                                       logtransform = FALSE, interpolate = FALSE)
            pdf(tempfile(fileext = paste0("fid", a_fid, "_indicator_", indicator_,"_detrending_" , detrending_, ".pdf")))
            graphics.off(); while (dev.cur() > 1) {dev.off()}
            sensitivity_ews_separate[[paste("fid", as.character(a_fid), indicator_, detrending_, sep="_")]] <- output
            }, silent = TRUE)
      }
    }
    saveRDS(sensitivity_ews_separate, file = paste0(out_dir, "sensitivity_ews_linear", ".rds"))
  }
