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

out_dir <- paste0(database, "earlyWarnings/")
if (dir.exists(file.path(out_dir)) == F) {
    dir.create(path = file.path(out_dir), recursive = T)
}

plot_dir <- paste0(dir_base, "/plots/earlyWarnings/", ewr_type, "/")
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

# anpp_subset <- as.data.frame(anpp[anpp$fid==1])
# anpp_subset <- subset(anpp_subset, select=mean_lb_per_acr)

## bdstest_ews() only accepts dataframe with one column!
## It may complain there is no column Y in the dataframe, but
## it will complain regardless of existence of Y or not. ONLY
## one column is needed
## it also prints stuff on the screen and makes a plot,
## but does not save the output in a variable to be collected/exported!
########################################################
#######
#######              bdstest_ews
#######  This is disaster. It wants to plot stuff. 
#######  and does not spit out anything to save.
#######

# bdstest_ews(anpp_subset, ARMAoptim=FALSE, ARMAorder=c(1,0),
#             embdim=3, epsilon=0.5, boots=200,
#             logtransform=FALSE, interpolate=FALSE)

# Convert the list into a dataframe
# The following wont work since each item has different length.
# we can do: names(ddjnonparam_ews_results_list) to get fids/names of each entry in ddjnonparam_ews_results_list
# then maybe each of the 10 outputs of ddjnonparam_ews for all fids will have the same lengths!!!
# we are gonna have to wait and see. Do this in Kamiak. The graphic thing is so annoying.
# ddjnonparam_results_df <- bind_rows(ddjnonparam_ews_results_list, .id = "a_fid")

########################################################
#######
#######              ddjnonparam_ews
#######  This is disaster. It wants to plot stuff.
#######
# Initialize an empty list to store results

if (ewr_type == "ddjnonparam"){
  ddjnonparam_ews_results_list <- list()

  for (a_fid in FIDs) {
    wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), sep="/")
    if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
    setwd(wd_dir)
    # Subset the data for the current fid
    anpp_subset <- as.data.frame(anpp[anpp$fid==1])
    anpp_subset <- subset(anpp_subset, select=mean_lb_per_acr)
    
    # Apply the function and store the result
    graphics.off(); while (dev.cur() > 1) {dev.off()}

    # output_fig_1 <- tempfile(fileext = "1.pdf")
    # pdf(output_fig_1)
    # output_fig_2 <- tempfile(fileext = "2.pdf")
    # pdf(output_fig_2)
    # output_fig_3 <- tempfile(fileext = "3.pdf")
    # pdf(output_fig_3)

    output <- ddjnonparam_ews(anpp_subset, bandwidth = 0.6, na = 500, logtransform = TRUE, interpolate = FALSE)
    graphics.off(); while (dev.cur() > 1) {dev.off()}
    
    # Store the results (flatten the list into a named vector)
    ddjnonparam_ews_results_list[[as.character(a_fid)]] <- output
  }
  saveRDS(ddjnonparam_ews_results_list, file = paste0(out_dir, "ddjnonparam_ews_results_list", ".rds"))
  } else if (ewr_type == "generic"){
    ########################################################
    #######
    #######              generic_ews
    #######  This one is nice
    #######
    generic_ews_results_list <- list()
    generic_ews_results_list_separate <- list()
    for (a_fid in FIDs) {
      wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), sep="/")
      if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
      setwd(wd_dir)

      # Subset the data for the current fid
      subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)
      
      # Apply the function and store the result
      graphics.off(); while (dev.cur() > 1) {dev.off()}
      # output_fig <- tempfile(fileext = ".pdf")
      # pdf(output_fig)
      output <- generic_ews(subset_data, winsize = 50,
                            detrending = c("no", "gaussian", "loess", "linear", "first-diff"),
                            bandwidth = NULL, span = NULL, degree = NULL,
                            logtransform = FALSE, interpolate = FALSE, AR_n = FALSE, powerspectrum = FALSE)
      pdf(tempfile(fileext = paste0("fid", a_fid, ".pdf")))
      graphics.off(); while (dev.cur() > 1) {dev.off()}
      generic_ews_results_list[[as.character(a_fid)]] <- output

      for (detrending_ in c("no", "gaussian", "loess", "linear", "first-diff")){
        wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), detrending_, sep="/")
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
    }
    saveRDS(generic_ews_results_list, file = paste0(out_dir, "generic_ews_results_list", ".rds"))
    saveRDS(generic_ews_results_list_separate, file = paste0(out_dir, "generic_ews_results_list_separate", ".rds"))
  } else if (ewr_type == "livpotential"){
    ########################################################
    #######
    #######              livpotential_ews 
    #######      not too bad. and does not plot nothing.
    #######
    livpotential_ews_results_list <- list()

    for (a_fid in FIDs) {
      wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), sep="/")
      if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
      setwd(wd_dir)

      subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)
      
      # Apply the function and store the result
      graphics.off(); while (dev.cur() > 1) {dev.off()}
      # output_fig <- tempfile(fileext = ".pdf")
      # pdf(output_fig)
      output <- livpotential_ews(subset_data, std = 1, bw = "nrd",
                                 weights = c(), # optional weights in ksdensity (used by movpotentials)
                                 grid.size = NULL, detection.threshold = 1,
                                 bw.adjust = 1, density.smoothing = 0, detection.limit = 1)
      pdf(tempfile(fileext = paste0("fid", a_fid, ".pdf")))
      graphics.off(); while (dev.cur() > 1) {dev.off()}
      livpotential_ews_results_list[[as.character(a_fid)]] <- output
      
    }
    saveRDS(livpotential_ews_results_list, file = paste0(out_dir, "livpotential_ews_results_list", ".rds"))
  } else if (ewr_type == "movpotential"){
    ########################################################
    #######
    #######              movpotential_ews 
    #######      
    #######
    movpotential_ews_results_list <- list()

    for (a_fid in FIDs) {
      wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), sep="/")
      if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
      setwd(wd_dir)

      subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)
      graphics.off(); while (dev.cur() > 1) {dev.off()}
      # output_fig <- tempfile(fileext = ".pdf")
      # pdf(output_fig)
      output <- movpotential_ews(subset_data, 
                                 param = NULL, 
                                 # bw = "nrd",
                                 bw.adjust = 1, 
                                 detection.threshold = 0.1, 
                                 std = 1,
                                 grid.size = 50, 
                                 plot.cutoff = 0.5, 
                                 plot.contours = TRUE,
                                 binwidth = 0.2,
                                 bins = NULL)
      pdf(tempfile(fileext = paste0("fid", a_fid, ".pdf"))) 
      graphics.off(); while (dev.cur() > 1) {dev.off()}
      
      plot_ = output["plot"][[1]]
      output = output["res"][[1]]
      movpotential_ews_results_list[[as.character(a_fid)]] <- output
    }
    saveRDS(movpotential_ews_results_list, file = paste0(out_dir, "movpotential_ews_results_list", ".rds"))
  } else if (ewr_type == "qda_separate"){
    qda_ews_results_list_separate <- list()
    for (a_fid in FIDs) {
      subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)
      for (detrending_ in c("no", "gaussian", "linear", "first-diff")){
        print (paste0("line 272 FID: ", a_fid, ", detrending_:", detrending_))
        wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), detrending_, sep="/")
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
    }
    saveRDS(qda_ews_results_list_separate, file = paste0(out_dir, "qda_ews_results_list_separate", ".rds"))
  } else if (ewr_type == "sensitivity_separate"){
    sensitivity_ews_separate <- list()
    for (a_fid in FIDs) {
      # wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), sep="/")
      # if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
      # setwd(wd_dir)
      
      subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)
      ###
      ### run for each indicator and deterending separately.
      ### 
      for (indicator_ in c("ar1", "sd", "acf1", "sk", "kurt", "cv", "returnrate", "densratio")){
        for (detrending_ in c("no", "gaussian", "loess", "linear", "first-diff")){
          wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), paste(indicator_, detrending_, sep="_"), sep="/")
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
    }
    saveRDS(sensitivity_ews_separate, file = paste0(out_dir, "sensitivity_ews_separate", ".rds"))
  }


##
## d==0 is not a guarantee for being stationary!! we need to learn more.
## more over, if d==0, then (assuming that would mean stationary, then detrending argument should be set to "no")
## strange thing detrending = c("no", "gaussian", "linear", "first-diff") works. but detrending = c("gaussian", "linear", "first-diff")
## complains and asks for argument of length 1. So, let us get rid of d==0 and do deterending things one at a time 
## and see if it works.
##

# check if differencing is needed. Determine the Appropriate Order (p, d, q):
# arima_model <- auto.arima(subset_data, seasonal = FALSE)
# d_arimaOrder <- arimaorder(arima_model)[2]
# if (d == 0) { # in this case differencing is not needed. so, detrending must be no.
#   ##### 
#   ##### even if d=0 is not a a guarantee for stationary. So, let us get rid of this 
#   #### here and only do separate!
#   #####
#   # wd_dir <- paste(dir_base, "plots/earlyWarnings", ewr_type, paste0("fid", a_fid), "noDetrending", sep="/")
#   # if (dir.exists(file.path(wd_dir)) == F) {dir.create(path = file.path(wd_dir), recursive = T)}
#   # setwd(wd_dir)
#   # graphics.off(); while (dev.cur() > 1) {dev.off()}
#   # output <- qda_ews(subset_data, param = NULL, winsize = 10,
#   #                   detrending = "no",
#   #                   bandwidth = NULL, boots = 100, s_level = 0.05, cutoff = 0.05, 
#   #                   detection.threshold = 0.002, grid.size = 10, logtransform = FALSE, interpolate = FALSE
#   #                  )
#   # text_out <- list(indicators = output$indicators, trend = output$trends)
#   # qda_ews_results_list_separate[[paste(detrending_, as.character(a_fid), sep="_")]] <- output
#   # pdf(tempfile(fileext = ".pdf")); graphics.off(); while (dev.cur() > 1) {dev.off()}
#   } else {
# differenced_data <- diff(subset_data, differences = d) # we are doing first_diff below.