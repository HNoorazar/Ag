library(lubridate)

wtr_yr <- function(dates, start_month = 10) {
  # Convert dates into POSIXlt
  dates.posix <- as.POSIXlt(dates)
  # Year offset
  offset <- ifelse(dates.posix$mon >= start_month - 1, 1, 0)
  # Water year
  adj.year <- dates.posix$year + 1900 + offset
  # Return the water year
  adj.year
}

wtr_week <- function(dates, start_month = 10) {
  offset <- ifelse(month(dates) >= start_month, 0, 1)
  (dates - dmy(paste(01,start_month,year(dates)-offset)))[[1]]%/%7 + 1
  (dates - dmy(paste(01,start_month,year(dates)-offset)))[[1]]%/%7 + 1
}

wtr_doy <- function(dates, start_month = 10) {
  offset <- ifelse(month(dates) >= start_month, 0, 1)
  (dates - dmy(paste(01,start_month,year(dates)-offset)))[[1]] + 1
}