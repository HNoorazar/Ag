if (!require("rspat")) remotes::install_github('rspatial/rspat')
library(rspat)
counties <- spat_data("counties")


houses <- spat_data("houses1990.csv")
hvect <- vect(houses, c("longitude", "latitude"))
plot(hvect, cex=0.5, pch=1, axes=TRUE)
crs(hvect) <- crs(counties)


cnty <- extract(counties, hvect)
head(cnty)


# Summarize
hd <- cbind(data.frame(houses), cnty)


hd <- cbind(data.frame(houses), cnty)



totpop <- tapply(hd$population, hd$NAME, sum)
totpop

# total income
hd$suminc <- hd$income * hd$households

# now use aggregate (similar to tapply)
csum <- aggregate(hd[, c('suminc', 'households')], list(hd$NAME), sum)

# divide total income by number of housefholds
csum$income <- 10000 * csum$suminc / csum$households

# sort
csum <- csum[order(csum$income), ]
head(csum)
tail(csum)


# Regression
hd$roomhead <- hd$rooms / hd$population
hd$bedroomhead <- hd$bedrooms / hd$population
hd$hhsize <- hd$population / hd$households

m <- glm( houseValue ~ income + houseAge + roomhead + bedroomhead + population, data=hd)
summary(m)

coefficients(m)



# Geographicaly Weighted Regression
# By county
hd2 <- hd[!is.na(hd$NAME), ]

regfun <- function(x)  {
  dat <- hd2[hd2$NAME == x, ]
  m <- glm(houseValue~income+houseAge+roomhead+bedroomhead+population, data=dat)
  coefficients(m)
}


countynames <- unique(hd2$NAME)
res <- sapply(countynames, regfun)

dotchart(sort(res["income", ]), cex=0.65)


resdf <- data.frame(NAME=colnames(res), t(res))
head(resdf)



dim(counties)
## [1] 68  5
dcounties <- aggregate(counties[, "NAME"], "NAME")
dim(dcounties)



cnres <- merge(dcounties, resdf, by="NAME")
plot(cnres, "income")



# a copy of the data
cnres2 <- cnres
# scale all variables, except the first one (county name)
values(cnres2) <- as.data.frame(scale(as.data.frame(cnres)[,-1]))
plot(cnres2, names(cnres2)[1:6], plg=list(x="topright"), mar=c(1,1,1,1))




lw <- adjacent(cnres2, pairs=FALSE)
autocor(cnres$income, lw)
## [1] 0.1565227
autocor(cnres$houseAge, lw)
## [1] -0.02057022




# By grid cell
# An alternative approach would be to compute a model for grid cells. Let’s use the 
# ‘Teale Albers’ projection (often used when mapping the entire state of California
TA <- "+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +datum=WGS84 +units=m"
countiesTA <- project(counties, TA)


r <- rast(countiesTA)
res(r) <- 50000



xy <- xyFromCell(r, 1:ncell(r))


housesTA <- project(hvect, TA)
crds <- geom(housesTA)[, c("x", "y")]


regfun2 <- function(d)  {
 m <- glm(houseValue~income+houseAge+roomhead+bedroomhead+population, data=d)
 coefficients(m)
}


res <- list()
for (i in 1:nrow(xy)) {
    d <- sqrt((xy[i,1]-crds[,1])^2 + (xy[i,2]-crds[,2])^2)
    j <- which(d < 50000)
    if (length(j) > 49) {
        d <- hd[j,]
        res[[i]] <- regfun2(d)
    } else {
        res[[i]] <- NA
    }
}



inc <- sapply(res, function(x) x['income'])


rinc <- setValues(r, inc)
plot(rinc)
plot(countiesTA, add=T)



autocor(rinc)
##     lyr.1
## 0.4452912



# spgwr package
r <- rast(countiesTA)
res(r) <- 25000
ca <- rasterize(countiesTA, r)


fitpoints <- crds(ca)

# Now specify the model
gwr.model <- ______


sp <- gwr.model$SDF
v <- vect(sp)
v


cells <- cellFromXY(r, fitpoints)
dd <- as.matrix(data.frame(sp))
b <- rast(r, nl=ncol(dd))
b[cells] <- dd
names(b) <- colnames(dd)
plot(b)



