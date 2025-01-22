# - Original link: https://rspatial.org/analysis/6-local_regression.html
# - New link: https://rspatial.org/analysis/7-spregression.html

# The original link does not do anything about weights or spatial-correlation. So, here we are, doing the New link!
library(spdep)
library(data.table)
library(dplyr)
library(sf)
# First install.packages("remotes") then the following line
if (!require("rspat")) remotes::install_github('rspatial/rspat')
library(rspat)

library(RColorBrewer)
grps <- 10

r_data_dir = "/Users/hn/Documents/01_research_data/RangeLand_bio/data_from_R/"


counties <- spat_data("counties")
h <- spat_data('houses2000')

# using a tiny buffer to get a cleaner aggregation
## it does not seem these are doing anything: test with setequal(x, y)
hb <- buffer(h, 1)
values(hb) <- values(h)
hha <- aggregate(hb, "County")



# d1 <- as.data.frame(h)[, c("nhousingUn", "recHouses", "nMobileHom", "nBadPlumbi",
#                            "nBadKitche", "Population", "Males", "Females", "Under5", "White",
#                            "Black", "AmericanIn", "Asian", "Hispanic", "PopInHouse", "nHousehold", "Families")]

# d1a <- aggregate(d1, list(County=h$County), sum, na.rm=TRUE)

# I think this is  more intuitive and clean, comparet to the one above
# where h$County comes from another dataframe
d1a <- as.data.frame(h)[, c("nhousingUn", "recHouses", "nMobileHom", "nBadPlumbi",
                     "nBadKitche", "Population", "Males", "Females", "Under5", "White",
                     "Black", "AmericanIn", "Asian", "Hispanic", "PopInHouse", "nHousehold", "Families", "County")] %>% 
             group_by(County) %>% 
             summarise_each(list(sum))%>% 
             data.table()
# check if 2 dataframes are identical:
# setequal(d1a, d1a_clean)
# all.equal(d1a, d1a_clean)


# d2 <- as.data.frame(h)[, c("houseValue", "yearBuilt", "nRooms", "nBedrooms",
#                            "medHHinc", "MedianAge", "householdS",  "familySize")]
# d2 <- cbind(d2 * h$nHousehold, hh=h$nHousehold)
# d2a <- data.table(aggregate(d2, list(County=h$County), sum, na.rm=TRUE))

d2 <- as.data.frame(h)[, c("houseValue", "yearBuilt", "nRooms", "nBedrooms",
                           "medHHinc", "MedianAge", "householdS",  "familySize")]
d2 <- cbind(d2 * h$nHousehold, hh=h$nHousehold, County=h$County)
d2a <- d2  %>% 
       group_by(County) %>% 
       summarise_each(list(sum))%>% 
       data.table()

d2a[, 2:ncol(d2a)] <- d2a[, 2:ncol(d2a)] / d2a$hh


d12 <- merge(d1a, d2a, by='County')
hh <- merge(hha[, "County"], d12, by='County')


hh$fBadP <- pmax(hh$nBadPlumbi, hh$nBadKitche) / hh$nhousingUn
hh$fWhite <- hh$White / hh$Population
hh$age <- 2000 - hh$yearBuilt

f1 <- houseValue ~ age +  nBedrooms
m1 <- lm(f1, data=as.data.frame(hh))
summary(m1)
y <- matrix(hh$houseValue)
X <- cbind(1, hh$age, hh$nBedrooms)
ols <- solve(t(X) %*% X) %*% t(X) %*% y
rownames(ols) <- c('intercept', 'age', 'nBedroom')
ols
hh$res_ols <- residuals(m1)


sfhh <- sf::st_as_sf(hh)
# The following throws error, and sf::sf_use_s2(FALSE), fixes the error
# and then we get "although coordinates are longitude/latitude, st_intersects assumes that they are planar"
sf::sf_use_s2(FALSE)
nb <- poly2nb(sfhh, snap=1/120)

# I donot know why he is ding this:
# shortcoming of R?
nb[[21]] <- sort(as.integer(c(nb[[21]], 38)))
nb[[38]] <- sort(as.integer(c(21, nb[[38]])))
plot(hh)
plot(nb, crds(centroids(hh)), col='red', lwd=2, add=TRUE)

# We can use the neighbour list object to get the average value for the neighbors of each polygon.
resnb <- sapply(nb, function(x) mean(hh$res_ols[x]))

# 32581.04, 22056.04
counter = 0
for (ii in resnb) {
    counter = counter+1
    if (abs(22056.04 - ii) < 1){
        print (counter)
        print (ii)
        print ("----")
        }
}

cor(hh$res_ols, resnb)
plot(hh$res_ols, resnb, xlab="Residuals", ylab="Mean adjacent residuals", pch=20)
abline(lm(resnb ~ hh$res_ols), lwd=2, lty=2)
lw <- nb2listw(nb)
moran.mc(hh$res_ols, lw, 999)

#
# Spatial lag model
#
#### The same thing got made in Python
#
library(spatialreg)
m1s <- lagsarlm(f1, data=as.data.frame(hh), lw, tol.solve=1.0e-30)
summary(m1s)


writeVector(x = hh, 
            filename = paste0(r_data_dir, "/hh_corr/"), 
            filetype="ESRI Shapefile", 
            layer="hh_corr", 
            insert=FALSE,
            overwrite=TRUE, 
            options="ENCODING=UTF-8")


### Export data and re-do in python:
write.csv(as.data.frame(hh), paste0(r_data_dir, "hh_cali_corr_4spatialLagModel.csv"), row.names=FALSE)

# initiate and  populate the weight_matrix_
weight_matrix_ <- data.table(setNames(data.table(matrix(0, nrow = 58, ncol = 58)), as.character(c(1:58))))
for (ii in 1:length(lw$neighbours)){
  neighbors = lw$neighbours[ii][[1]]
  weights_  = lw$weights[ii][[1]]
  nb_counter = 0
  for (a_neighbor in neighbors){
    nb_counter = nb_counter+1
    weight_matrix_[ii, a_neighbor] = weights_[nb_counter]
  }
}

write.csv(as.data.frame(weight_matrix_), paste0(r_data_dir, "weight_matrix_Cali_houses_corr.csv"), row.names=FALSE)




### The following was also created in Python. Successfully
m1e <- errorsarlm(f1, data=as.data.frame(hh), lw, tol.solve=1.0e-30)
summary(m1e)

hh$res_errModel <- residuals(m1e)
moran.mc(hh$res_errModel, lw, 999)
brks <- quantile(hh$res_errModel, 0:(grps-1)/(grps-1), na.rm=TRUE)

plot(hh, "res_errModel", breaks=brks, col=rev(brewer.pal(grps, "RdBu")))
