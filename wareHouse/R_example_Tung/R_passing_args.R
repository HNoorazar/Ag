
### Make sure that we are using the right library folder
.libPaths("/data/hydro/R_libs")
.libPaths()

#install.packages(c("bitops", "chillR", "MESS", "colorspace", "lubridate", "tidyverse"), 
#                 dependencies = TRUE, repos = 'https://cran.rstudio.com')

#install.packages(c("base64enc"),
#                 dependencies = TRUE, repos = 'https://cran.rstudio.com')

# Test if packages can be used
library(chillR)
library(MESS)
library(hydroGOF)
library(xts)
library(tidyverse)
library(rgdal)
devtools::session_info()

### Getting arg passed from qsub
# http://tuxette.nathalievilla.org/?p=1696
# https://swcarpentry.github.io/r-novice-inflammation/05-cmdline/

args <- commandArgs(trailingOnly = TRUE)
str(args)
cat(args, sep = "\n")

# test if there is at least one argument: if not, return an error
if (length(args) == 0) {
  stop("At least one argument must be supplied (input file).\n", call. = FALSE)
} else if (length(args) == 1) {
  # default output file
  args[2] = "out.txt"
}

cat("\n")
print("Hello World !!!")

cat("\n")
print(paste0("nSet = ", as.numeric(args[1])))

cat("\n")
M <- replicate(2, runif(10e5, 0, 1))
d <- data.frame(M)
colnames(d) <- c("y1", "y2")
head(d)

cat("\n")
str(d)

