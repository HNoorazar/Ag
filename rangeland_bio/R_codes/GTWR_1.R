#
# GTWR is not doble in Python. Or it seems so!!!
# Jan 29, 2025. It seems we need to do regression and then
# find trends. We cannot just use time as X and find Trend
# The GWTR package in R is using the following paper:
# Geographically and temporally weighted regression for modeling spatio-temporal variation in house prices.
# and here we are.
#
library(spdep)
library(data.table)
library(dplyr)
library(sf)
# First install.packages("remotes") then the following line
if (!require("rspat")) remotes::install_github('rspatial/rspat')
library(rspat)

library(RColorBrewer)
grps <- 10
