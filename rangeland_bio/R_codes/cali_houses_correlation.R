# - Original link: https://rspatial.org/analysis/6-local_regression.html
# - New link: https://rspatial.org/analysis/7-spregression.html

# The original link does not do anything about weights or spatial-correlation. So, here we are, doing the New link!

r_data_dir = "/Users/hn/Documents/01_research_data/RangeLand_bio/data_from_R/"
library(sf)

# First install.packages("remotes") then the following line
if (!require("rspat")) remotes::install_github('rspatial/rspat')
library(rspat)
counties <- spat_data("counties")


h <- spat_data('houses2000')

# using a tiny buffer to get a cleaner aggregation
## it does not seem these are doing anything: test with setequal(x, y)
hb <- buffer(h, 1)
values(hb) <- values(h)
hha <- aggregate(hb, "County")



d1 <- as.data.frame(h)[, c("nhousingUn", "recHouses", "nMobileHom", "nBadPlumbi",
                           "nBadKitche", "Population", "Males", "Females", "Under5", "White",
                           "Black", "AmericanIn", "Asian", "Hispanic", "PopInHouse", "nHousehold", "Families")]

d1a <- aggregate(d1, list(County=h$County), sum, na.rm=TRUE)




d2 <- as.data.frame(h)[, c("houseValue", "yearBuilt", "nRooms", "nBedrooms",
                           "medHHinc", "MedianAge", "householdS",  "familySize")]
d2 <- cbind(d2 * h$nHousehold, hh=h$nHousehold)
d2a <- aggregate(d2, list(County=h$County), sum, na.rm=TRUE)
d2a[, 2:ncol(d2a)] <- d2a[, 2:ncol(d2a)] / d2a$hh


d12 <- merge(d1a, d2a, by='County')
hh <- merge(hha[, "County"], d12, by='County')



