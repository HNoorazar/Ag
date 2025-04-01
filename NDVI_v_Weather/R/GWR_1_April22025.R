library(GWmodel)
library(tmap)  # For visualization, optional

library(foreign) # to read .sav files
library(haven) # to read .sav files
library(spdep)
library(data.table)
library(dplyr)
library(rgdal)
library(sf)
library(sp)

# First library("remotes") then the following line
if (!require("rspat")) remotes::install_github('rspatial/rspat')
library(rspat)

library(RColorBrewer)

### The GWR wants a bandwidth and works with Gaussian.
### Perhaps we should find another library that let us choose the weight matrix.
### or we can start by doing the non-numerical way: 
### β(u_i, v_i) = (X^T W(u_i, v_i) X) ^ {-1} X^T W(u_i, v_i) y
###