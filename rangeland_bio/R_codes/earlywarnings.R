# install.packages("earlywarnings")


####################################
# is the size of the rolling window expressed as percentage of the timeseries length
# (must be numeric between 0 and 100). Default is 50%
#
###################################################################
###################################################################
####
#### 1. Load packages ---------------------------------------------
rm(list=ls())

library(tseries) # to use stationary: adf.test()
library(data.table)
library(earlywarnings)
library(dplyr)
library(ggmap) # loads ggplot2
library(forecast)
###################################################################
###################################################################
####
#### 2. Data Directories ------------------------------------------
database <- "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/"
reOrganized_dir = paste0(database, "reOrganized/") 

out_dir <- paste0(database, "earlyWarnings/")
if (dir.exists(file.path(out_dir)) == F) {
    dir.create(path = file.path(out_dir), recursive = T)
}

plot_dir <- "/Users/hn/Documents/01_research_data/RangeLand_bio/plots/earlyWarnings/"
if (dir.exists(file.path(plot_dir)) == F) {
    dir.create(path = file.path(plot_dir), recursive = T)
}

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

setorder(anpp, fid, year)

# anpp_clean <- anpp[, .(value = mean(mean_lb_per_acr, na.rm = TRUE)), by = .(fid, year)]
# setorder(anpp_clean, fid, year)

# Convert to list of ts

ts_list <- anpp[, {
  ts_obj <- ts(mean_lb_per_acr, start = min(year), frequency = 1)  # Annual time series
  list(ts = list(ts_obj))
}, by = fid]


# setnames(anpp_subset, c("mean_lb_per_acr"), c("Y"))
anpp_subset <- as.data.frame(anpp[anpp$fid == 1])
anpp_subset <- subset(anpp_subset, select = mean_lb_per_acr)

subset_data <- matrix(anpp[anpp$fid == a_fid]$mean_lb_per_acr)

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

bdstest_ews(anpp_subset, ARMAoptim=FALSE, ARMAorder=c(1,0),
            embdim=3, epsilon=0.5, boots=200,
            logtransform=FALSE, interpolate=FALSE)


########################################################
#######
#######              ddjnonparam_ews
#######  This is disaster. It wants to plot stuff.
#######
ddjnonparam_ews_ = ddjnonparam_ews(matrix(anpp[anpp$fid == 1]$mean_lb_per_acr), 
                                   bandwidth = 0.6, na = 500, logtransform = TRUE, interpolate = FALSE)

unique_fids <- unique(anpp$fid)

# Initialize an empty list to store results
ddjnonparam_ews_results_list <- list()

# Loop through each fid and apply the function
for(a_fid in unique_fids) {
  # Subset the data for the current fid
  subset_data <- anpp[anpp$fid == a_fid]$mean_lb_per_acr
  
  # Apply the function and store the result
  output <- ddjnonparam_ews(matrix(subset_data), 
                            bandwidth = 0.6, na = 500, logtransform = TRUE, interpolate = FALSE)

  graphics.off()
  
  # Store the results (flatten the list into a named vector)
  ddjnonparam_ews_results_list[[as.character(a_fid)]] <- output
}

# Convert the list into a dataframe
# The following wont work since each item has different length.
# we can do: names(ddjnonparam_ews_results_list) to get fids/names of each entry in ddjnonparam_ews_results_list
# then maybe each of the 10 outputs of ddjnonparam_ews for all fids will have the same lengths!!!
# we are gonna have to wait and see. Do this in Kamiak. The graphic thing is so annoying.
# ddjnonparam_results_df <- bind_rows(ddjnonparam_ews_results_list, .id = "a_fid")



########################################################
#######
#######              generic_ews
#######  This one is nice
#######
generic_ews_out <- generic_ews(matrix(subset_data), winsize = 50,
                               detrending = c("no", "gaussian", "loess", "linear", "first-diff"),
                               bandwidth = NULL, span = NULL, degree = NULL,
                               logtransform = FALSE, interpolate = FALSE, AR_n = FALSE, powerspectrum = FALSE)



generic_ews_results_list <- list()
for(a_fid in unique_fids) {
  # Subset the data for the current fid
  subset_data <- anpp[anpp$fid == a_fid]$mean_lb_per_acr
  
  # Apply the function and store the result
  generic_ews_out <- generic_ews(matrix(subset_data), winsize = 50,
                               detrending = c("no", "gaussian", "loess", "linear", "first-diff"),
                               bandwidth = NULL, span = NULL, degree = NULL,
                               logtransform = FALSE, interpolate = FALSE, AR_n = FALSE, powerspectrum = FALSE)

  graphics.off()
  while (dev.cur() > 1) {dev.off()}
  generic_ews_results_list[[as.character(a_fid)]] <- output
}

########################################################
#######
#######              livpotential_ews 
#######      not too bad. and does not plot nothing.
#######
livpotential_ews_out <- livpotential_ews(matrix(subset_data), std = 1, bw = "nrd",
                                                weights = c(), grid.size = NULL, detection.threshold = 1,
                                                bw.adjust = 1, density.smoothing = 0, detection.limit = 1)



########################################################
#######
#######              movpotential_ews 
#######      
#######
movpotential_ews_ews_out <- movpotential_ews(matrix(subset_data), param = NULL, bw = "nrd",
                                             bw.adjust = 1, detection.threshold = 0.1, std = 1,
                                             grid.size = 50, plot.cutoff = 0.5, plot.contours = TRUE,
                                             binwidth = 0.2, bins = NULL)

res = movpotential_ews_ews_out["res"][[1]]
plot_ = movpotential_ews_ews_out["plot"][[1]]



########################################################
#######
#######   Quick Detection Analysis for Generic Early Warning Signals     
#######  This one throws an error: 
#######  Error in arima(nsmY, order = c(ij, 0, jj), include.mean = FALSE) : non-stationary AR part from CSS



qda_ews_out <- qda_ews(matrix(subset_data), param = NULL, winsize = 10,
                       detrending = c("no", "gaussian", "linear", "first-diff"),
                       bandwidth = NULL, boots = 100, s_level = 0.05, cutoff = 0.05, 
                       detection.threshold = 0.002, grid.size = 10, logtransform = FALSE, interpolate = FALSE
                       )

adf.test(matrix(subset_data))

# Determine the Appropriate Order (p, d, q):
library(forecast)
arima_model <- auto.arima(matrix(subset_data))
arimaorder_ <- arimaorder(arima_model)

# differencing is supposed to kill non-stationaryness
differenced_ <- diff(matrix(subset_data))
adf.test(matrix(differenced_))


# arimaorder_ in the following line came from arimaorder_ <- arimaorder(arima_model)
# which should not be a good one! 
what_is_this <- arima(differenced_, order = arimaorder_, include.mean = FALSE)
fitted_what_is_this <- fitted(fit)


fit <- auto.arima(differenced_)



acf(differenced_)
pacf(differenced_)

plot(fitted_what_is_this)
plot(ts_list[1]$ts[[1]])
plot(matrix(subset_data))

## plot 2 vectors in one figure
plot(differenced_, type = "l", col = "blue", lwd = 2)
lines(fitted_what_is_this, type = "l", col = "red", lwd = 2)


## find best ij and jj
best_aic <- Inf
best_order <- c(0, 0)

for(p in 0:5){
  for(q in 0:5){
    fit <- try(arima(diff_series, order = c(p, 0, q)), silent = TRUE)
    if(class(fit) != "try-error"){
      if(fit$aic < best_aic){
        best_aic <- fit$aic
        best_order <- c(p, q)
      }
    }
  }
}

best_order







graphics.off()
while (dev.cur() > 1) {dev.off()}




# # differencing is supposed to kill non-stationaryness
# differenced_ <- diff(subset_data)
# adf.test(matrix(differenced_))


# arimaorder_ in the following line came from arimaorder_ <- arimaorder(arima_model)
what_is_this <- arima(differenced_data, order = arimaorder_, include.mean = FALSE)

fit <- auto.arima(differenced_)
fitted_what_is_this <- fitted(fit)


acf(differenced_)
pacf(differenced_)


## plot 2 vectors in one figure
plot(differenced_, type = "l", col = "blue", lwd = 2)
lines(fitted_what_is_this, type = "l", col = "red", lwd = 2)



options(device = "quartz")

dev.new()
qda_ews_out <- qda_ews(differenced_)
png(paste0(plot_dir, "page%d.png"))
plot(1)
plot(2)
plot(3)
graphics.off()
while (dev.cur() > 1) {dev.off()}

dev.new()
qda_ews_out <- qda_ews(differenced_)

qda_ews_out <- qda_ews(differenced_)
pdf(paste0(plot_dir, "page%d.pdf"), onefile=FALSE)
plot(1)
graphics.off()
while (dev.cur() > 1) {dev.off()}


qda_ews_out <- qda_ews(differenced_)
pdf(paste0(plot_dir, "page%d.pdf"), onefile=FALSE)
plot(2)
graphics.off()
while (dev.cur() > 1) {dev.off()}


qda_ews_out <- qda_ews(differenced_)
pdf(paste0(plot_dir, "page%d.pdf"), onefile=FALSE)
plot(3)
graphics.off()
while (dev.cur() > 1) {dev.off()}




############
pdf(paste0(plot_dir, "page%d.pdf"), onefile=FALSE)
qda_ews_out <- qda_ews(differenced_)
plot(1)
plot(2)
plot(3)
graphics.off()
while (dev.cur() > 1) {dev.off()}



quartz.save(plot(1), type = "pdf", device = dev.cur(), dpi = 100)