####################################################################################
########################################## Educational - For reference
# file %>%
# filter(Year > 2025 & Year <= 2055,
# Chill_season != "chill_2025-2026" &
# Chill_season != "chill_2055-2056") %>% 
# group_by(Chill_season) %>%

# simple simplify

counties <- rmapshaper::ms_simplify(counties)
# strsplit a column into two columns
A[, c("date", "time") := tstrsplit(PriorityDate, " ", fixed=TRUE)]

Extract row corresponding to minimum value of a variable by group

data_dt <- data_dt %>%
           group_by(location, chill_season, model, emission, time_period) %>% 
           slice(which.min(thresh_55)) %>%
           data.table()

data_dt[ , .SD[which.min(chill_dayofyear)], 
           by = list(lat, long, model, chill_season)]


dt$month[dt$month == 1] = 13
dt$month[dt$month == 2] = 14
dt$month[dt$month == 3] = 15
dt$month[dt$month == 4] = 16
dt$month[dt$month == 5] = 17
dt$month[dt$month == 6] = 18
dt$month[dt$month == 7] = 19
dt$month[dt$month == 8] = 20

transform(dt, month=case_when(month == 1 ~ 13,
                              month == 2 ~ 14,
                              month == 3 ~ 15,
                              month == 4 ~ 16,
                              month == 5 ~ 17,
                              month == 6 ~ 18,
                              month == 7 ~ 19,
                              month == 8 ~ 20,
                              month == 9 ~ 9,
                              month == 10 ~ 10,
                              month == 11 ~ 11,
                              month == 12 ~ 12
                              ))



##########################################
main_out <- file.path(lagoon_out, "/01_storm_cumPrecip/cum_precip/")
if (dir.exists(main_out) == F) {dir.create(path = main_out, recursive = T)}

##########################################

# assign(x = paste0(month, "_density_plot_", "rcp45"),
#        value = {plot_dens(data=data_45, month_name=month)})

##########################################
# round decimal 
cols <- names(mydt)[1:2]
mydt[,(cols) := round(.SD,5), .SDcols=cols]
mydt

# df[4, 5] = auc(x=data_gen$CumulativeDDF, y=data_gen$value)

A_filtered <- A %>% filter_all(any_vars(is.na(.)))
A <- summary_comp %>% filter_all(any_vars(is.na(.)))
A <- summary_compy %>% filter_all(any_vars(is.na(.)))
A <- summary_compy %>% filter(any_vars(is.na(.)))
A <- A %>% filter_all(any_vars(is.na(.)))
#
# Change name of a columns
# rename
#
colnames(data)[colnames(data)=="old_name"] <- "new_name"
setnames(data, old=c("old_name"), new=c("new_name"))

# order a data by a/multiple column. Adding a negative 
# would make the ordering reverse
A <- A[order(location), ]


# count number of NA in each column
all_data_dt[, lapply(.SD, function(x) sum(is.na(x))), .SDcols = 1:9]

# sort/order two vectors simultaneously
vector2[order(vector1, decreasing=TRUE)]


# change order of columns of data table, reorder
data.table::setcolorder(x, c("c", "b", "a"))

# reshape a vector into matrix
d <- matrix(NN.dist, nrow = 70, byrow = FALSE)

df.melted <- melt(myDF[, -1], id.vars = NULL)
myNewDF <- t(df.melted[, 2])
colnames(myNewDF) <- paste0("r", rownames(myDF), df.melted[, 1])


# initialize data frame data table dataframe datatable
table = data.frame()

col_names <- c("Return_Period", "five", "ten", 
               "fifteen", "twenty", "twenty five")
data <- setNames(data.table(matrix(nrow = 24776, ncol = length(col_names))), 
                 col_names)

data <- data.table(lat=numeric(), long=numeric(), 
                   distances=numeric(), 
                   sigma=numeric())

data <- setNames(data.table(matrix(nrow = 3, ncol = 6)), 
                 c("future_fip",  "model", "time_period", 
                   "top_1_fip", "top_2_fip", "top_3_fip"))

convert_vect <- c(244, 258, 
                  274, 288, 
                  305, 319, 
                  335, 349)
convert_vect <- convert_vect %% 243

first_half <- c(1, 15, 
                32, 46, 
                60, 74, 
                91, 105, 
                121, 135, 
                152, 166, 
                182, 196, 
                213, 227)
first_half <- first_half + 122

convert_vect <- c(convert_vect, first_half)

letter_vect <- c("Sep. 1", "Sep. 15", "Oct. 1", "Oct. 15",
                 "Nov. 1", "Nov. 15", "Dec. 1", "Dec. 15",
                 "Jan. 1", "Jan. 15", "Feb. 1", "Feb. 15", 
                 "Mar. 1", "Mar 15", "Apr. 1", "Apr 15.", 
                 "May 1", "May 15", "Jun. 1", "Jun. 15", 
                 "Jul. 1", "Jul. 15", "Aug. 1", "Aug. 15")

data = data.table(day_count_since_sept = convert_vect,
                  letter_day = letter_vect
                  )

write.table(data, 
            file = "/Users/hn/Documents/GitHub/Ag/chill_DoY_map.csv", 
            row.names=FALSE, na="", col.names=TRUE, sep=",")

data = data.table(future_fip = c(target_fip, target_fip, target_fip),
                  model = c(model_n, model_n, model_n),
                  time_period = c("F1", "F2", "F3"),
                  emission = c(emission, emission, emission),
                  top_1_fip = c("NA", "NA", "NA"),
                  top_2_fip = c("NA", "NA", "NA"),
                  top_3_fip = c("NA", "NA", "NA")
                  )

data = data.table(model = c("target"),
                  start = c(10),
                  end = c(20)
                  )

# pick up every other column. pick up odd columns. pick up even columns
newdf <- data_1[, c(rep(c(TRUE, FALSE), (ncols(data_1)/2))), with = FALSE]

# combine odd_data with even_data to form data 
data <- data.table(matrix(nrow = nrow(odd_data), ncol = (ncol(odd_data)*2)))
colnames_data <- colnames(data)
setnames(data, old=colnames_data[c(TRUE, FALSE)], new= colnames(odd_data))
setnames(data, old=colnames_data[c(FALSE, TRUE)], new= colnames(even_data))

DT = data.table(row_count = c("b","b","b","a","a","c"),
                a = 1:6,
                b = 7:12,
                c = 13:18)

DT = data.table(row_count = 1:3)

dtr <- structure(list(location = c("NYC", "NYC", "NYC","NYC", "NYC", 
                                   "LA", "LA", "LA", "LA", "LA"), 
                 year = c(2026, 2026, 2026, 2026, 2026,
                          2026, 2026, 2026, 2026, 2026),
                 value = c(1, 2, 3, 4, 5,
                           6, 7, 8, 9, 10)),
                 class = "data.table", 
                 row.names = c(NA, -10L))

dtr <- structure(list(location = c("NYC", "LA"), 
                 year = c(2026, 2026),
                 value = c(2, 7)),
                 class = "data.table", 
                 row.names = c(NA, -10L))


start <- data.table(start=rep("start", nrow(something)))
############################################################
#################### Install packages on aeolus

https://docs.aeolus.wsu.edu/docs_running_applications.html

qsub -I [job script].sh

module load gcc/7.3.0
module load r/3.5.1/gcc/7.3.0
R
install.packages(c("foreign"), lib="~/.local/lib/R3.5.1", repos="https://ftp.osuosl.org/pub/cran")

Ctrl-d
exit

Finally, you’ll need to create a file to tell R where 
your packages live. Create a file called .Renviron 
in your home directory, and specify your library directory:
R_LIBS_USER=~/.local/lib/R3.5.1


#################### Install packages on aeolus ^^^^^^^^^^^^
############################################################
# count number of NA in columns
colSums(is.na(dt)|dt == '')


# strsplit vector 
x <- sapply(all_us_locations_cod_moth, 
            function(x) strsplit(x, "_")[[1]], 
            USE.NAMES=FALSE)
lat = x[1, ]
long = x[2, ]

A <- subset(A, select=col_names)
# not in, opposite of %in%
D2 = subset(local_locs, !(local_locs %in% local_files))

###### Convert a data table vector to a list:
unlist(as.list(t(analogs)))
as.list(as.data.table(t(analogs)))
x <- sapply(analogs, function(x) strsplit(x, "_")[[1]], USE.NAMES=FALSE)
lat = x[1, ]; long = x[2, ];
analogs <- paste0(lat, "_", long)


Drop the word county: https://stackoverflow.com/questions/55599225/drop-a-word-in-a-column-of-data-table-in-r/55599424#55599424
counties[, COUNTY := sub("\\s+County$", "", COUNTY)]
df$county = gsub("county", "", df$county)

######## compute frequency, aggregation

values <- data.frame(query = c("q1", "q1", "q1","q2", "q2"),
                     NN = c("a", "a", "b", "b", "b"),
                     freq=c(53, 20, 10, 3, 10),
                     model = c("m1", "m1", "m1", "m1", "m1"))

nr.of.appearances <- aggregate(x = values, 
                               by = list(unique.values = values$value), 
                               FUN = length)

values[, .(var = sum(freq)), by = c("query", "NN", "model")]


# Cartesian paste

model_namess <- c("bcc_csm1_1_m", "BNU_ESM", "CanESM2", "CNRM_CM5", "GFDL_ESM2G", "GFDL_ESM2M")
local_county_names <- unique(local_cnty_fips$st_county)
local_county_names <- sapply(local_county_names, function(x) strsplit(x, "_")[[1]], USE.NAMES=FALSE)
local_county_names = paste0(tolower(x[2, ]), "_")
plot_names <- do.call(paste, expand.grid(model_namess, local_county_names, sep='_', stringsAsFactors=FALSE))


###### mask


A_sigma
year          location    sigma_NN_1 sigma_NN_2 sigma_NN_3
2076 43.59375_-116.78125  1.4681173   1.664289   1.735974
2077 43.59375_-116.78125  1.3798515   1.550524   1.551269
2078 43.59375_-116.78125  0.7934367   1.064248   1.177981
2079 43.59375_-116.78125  1.8235574   1.991018   2.288402
2080 43.59375_-116.78125  2.5560329   2.578093   2.589334


A_NN
year            location       location_NN_1      location_NN_2      location_NN_3
2076 43.59375_-116.78125  41.15625_-90.65625 41.21875_-90.65625 41.15625_-90.65625
2077 43.59375_-116.78125  43.34375_-78.15625 43.34375_-78.21875 43.28125_-78.15625
2078 43.59375_-116.78125  41.34375_-90.78125 41.21875_-90.65625 41.53125_-73.96875
2079 43.59375_-116.78125 43.53125_-116.78125 41.34375_-90.78125 41.71875_-74.15625
2080 43.59375_-116.78125  41.34375_-90.78125 41.96875_-86.21875 41.21875_-90.65625


output
year            location       location_NN_1      location_NN_2      location_NN_3
2076 43.59375_-116.78125  41.15625_-90.65625 41.21875_-90.65625 41.15625_-90.65625
2077 43.59375_-116.78125  43.34375_-78.15625 43.34375_-78.21875 43.28125_-78.15625
2078 43.59375_-116.78125  41.34375_-90.78125 41.21875_-90.65625 41.53125_-73.96875
2079 43.59375_-116.78125 43.53125_-116.78125 41.34375_-90.78125                 NA
2080 43.59375_-116.78125                  NA                 NA                 NA


nm1 <- grep("sigma", names(A_sigma), value = TRUE)
i1 <- setDT(A_sigma)[, Reduce(`&`, lapply(.SD, <, 2)), .SDcols = nm1]
setDT(A_NN)[i1] 

A_NN A_sigma
A_NN[, -c(1:2)][A_sigma[, -c(1:2)] >= 2] <- NA

dfb[-c(1,2)][!(dfa[-c(1,2)] < 2)] <- NA
dfb[-c(1:2)][dfa[-c(1:2)] >= 2] <- NA

nm1 <- grep("sigma", names(A), value = TRUE)
i1 <- setDT(A)[, Reduce(`&`, lapply(.SD, `<`, 2)), .SDcols = nm1]
setDT(B)[i1] 


nm2 <- grep("sigma", names(A))
B[, (nm2) := Map(function(x, y) replace(x, y >= 2, NA_character_),
        .SD, A[, nm2, with = FALSE]), .SDcols = nm2][]


df
year            location       location_NN_1      location_NN_2      location_NN_3
2076 43.59375_-116.78125  41.15625_-90.65625  41.21875_-90.65625 41.15625_-90.65625
2077 43.59375_-116.78125  43.34375_-78.15625  43.34375_-78.21875 43.28125_-78.15625
2078 43.59375_-116.78125  41.34375_-90.78125  41.21875_-90.65625 41.53125_-73.96875
2079 43.59375_-116.78125  43.53125_-116.78125 41.34375_-90.78125               <NA>
2080 43.59375_-116.78125                <NA>               <NA>                <NA>


counties
fips   location
36073  43.34375_-78.15625
17161  41.34375_-90.78125

output = 
year            location       location_NN_1      location_NN_2      location_NN_3
2076 43.59375_-116.78125               17131             so_on             so_on
2077 43.59375_-116.78125               36073             so_on             so_on
2078 43.59375_-116.78125               17161             so_on             so_on
2079 43.59375_-116.78125      so_on, so_forth            so_on               <NA>
2080 43.59375_-116.78125                <NA>               <NA>              <NA>


dataframe[, seq(3, ncol(dataframe))] <- mapvalues(dataframe[, seq(3, ncol(dataframe))], 
                                        from = counties$location, to = counties$fips)

df_A_NN[3:5] <- lapply(df_A_NN[3:5], function(x) counties$fips[match(x, counties$location)])




df
location   NN_1    NN_2   NN_3
    NYC    17      17      17
    NYC    17      16      1
    LA     1        1      10
    LA     16      10      1

output
location   NNs  freq
    NYC    17      4
    NYC    16      1
    NYC     1      1
    LA      1      3
    LA      16     1
    LA      10     2

df.groupby('location').count()

############################################################
df <- structure(list(location = c("NYC", "NYC", "LA", "LA"), 
                     NN_1 = c(17, 17, 1, 16), 
                     NN_2 = c(17, 16, 1, 10), 
                     NN_3 = c(17, 1, 10, 1)),
                     class = "data.frame", 
                     row.names = c(NA, -4L))

df %>% 
tidyr::gather("key", "NNs", 2:ncol(.)) %>% 
group_by(location, NNs) %>% 
summarize(freq = n()) %>% 
arrange(desc(location), desc(NNs))

as.data.frame(table(cbind(df[1], NNs=unlist(df[-1]))))

################################################################

# count number of unique elements in a column after filtering

a %>% filter(fips == target_fip) %>% summarise(count = n_distinct(location))
a  %>% filter(fips == target_fip) %>% distinct(location) %>% count()

setDT(dt)[year == 2026, .(count = uniqueN(location))]


kth smallest element in group by
https://stackoverflow.com/questions/56084877/k-th-smallest-element-per-group-in-r/56085151#56085151

# Replace values in column ifelse quick, short
https://mgimond.github.io/ES218/Week03a.html
dat.overwrite <- mutate(dat, Country = ifelse(Country == "Canada", "CAN", "USA"))


data = data.table(year = c(2005, 2006, 2006, 2006, 2006),
                  month = c(1, 1, 1, 2, 10),
                  day = c(10, 20, 30, 40, 50))

data = data.table(city = c("NYC", "NYC", "NYC", "LA", "LA", "LA", "LA"),
                  year = c(2000, 2000, 2000, 2000, 2000, 2000, 2000),
                  target = c(0, 1, 1, 0, 0, 1, 1))

data = data.table(city = c("NYC", "NYC", "NYC", "LA", "LA", "LA", "LA"),
                  year = c(2000, 2000, 2000, 2000, 2000, 2000, 2000),
                  target = c(0, 666, 1, 0, 0, 666, 1))

#
# replace the first nonzero with another thing after group_by:
#
data %>%
group_by(city, year) %>%
mutate(target = replace(target, which.max(target != 0), 666))

OR 
i1 <- data[, .I[target != 0][1], .(city, year)]$V1
data[i1, target := 666][]


OR (not tested)
library(tidyverse)
data %>%
   group_by(city, year) %>% 
   mutate(target = replace(target, which(target != 0)[1], 666))

OR (not tested)
data %>% 
   group_by(city, year) %>%
   mutate(target = replace(target, match(1, target), 666))


# pick the row wich max in a column
analog_dat_F1_4_map[which.max(analog_dat_F1_4_map$analog_freq),]

################################################################
#### Group by and find max, group_by find max

df <- read.table(header = TRUE, text = 'Gene   Value
A      12
A      10
B      3
B      5
B      6
C      1
D      3
D      4')
#
# aggregate
#
aggregate(df$Value, by = list(df$Gene), max)
aggregate(Value ~ Gene, data = df, max)

# tapply
tapply(df$Value, df$Gene, max)

# split + lapply
lapply(split(df, df$Gene), function(y) max(y$Value))

# plyr
require(plyr)
ddply(df, .(Gene), summarise, Value = max(Value))

########
######## The same
########
A = A[, .(mean_gdd = mean(CumDDinF)), by = c("location", "year")]

B <- B %>%
     group_by(location, year) %>%
     summarise_at(.funs = funs(mean(., na.rm=TRUE)), vars(CumDDinF)) %>% 
     data.table()

a <- observed_dt %>%
     group_by(location) %>%
     summarise_at(vars(annual_cum_precip), funs(mean(., na.rm=TRUE))) %>%
     data.table()

b <- observed_dt %>% 
     group_by(location)%>% 
     summarise(mean=mean(annual_cum_precip), sd=sd(annual_cum_precip)) %>%
     data.table()

######## 
df %>% group_by(Gene) %>% summarise(Value = max(Value))

dt %>%
group_by(time_period, month, emission, cluster) %>%
summarise(mean=mean(monthly_cum_precip))%>%
data.table()


#
# summary summerize summerise summarize summarise
#
ddply(us_feat, ~ fips, summarise, mean=mean(age), sd=sd(age))
dt[, list(mean=mean(age), sd=sd(age)), by=group]

result <- dataT %>%
          mutate(thresh_range = cut(get(col_name), breaks = bks )) %>%
          group_by(lat, long, climate_type, time_period, 
                   thresh_range, model, scenario) %>%
          summarize(no_years = n_distinct(Chill_season)) %>% 
          data.table()


quan_per_feb <- feb_result %>% 
                group_by(climate_type, time_period, scenario, thresh_range) %>% 
                summarise(quan_25 = quantile(frac_passed, probs = 0.25)) %>% 
                data.table()

# data.table
require(data.table)
dt <- data.table(df)
dt[ , max(Value), by = Gene]

# doBy
require(doBy)
summaryBy(Value~Gene, data = df, FUN = max)

# sqldf
require(sqldf)
sqldf("select Gene, max(Value) as Value from df group by Gene", drv = 'SQLite')

# ave
df[as.logical(ave(df$Value, df$Gene, FUN = function(x) x == max(x))),]
################################################################
# followigs work equal
# OP's input

########## ONE
clusterDT = data.table(old_label = 1:2, centroid = c(9.5, 1.5))

# overwrite labels by sorting and assigning row number
clusterDT[order(centroid), new_label := .I]

# update data
data[, cluster := clusterDT[.SD, on=.(old_label = cluster), x.new_label]]

     city population cluster
1:    NYC         10       2
2:     LA          9       2
3: Hawaii          1       1
4:  Essex          2       1

####### TWO 

dt <- data.table(city = c("NYC", "LA", "Hawaii", "Essex"),
                 population = c(10, 9, 1, 2),
                 cluster = c(1, 1, 2, 2)
) %>% group_by(cluster) %>% #create the centroids variable
  mutate(centroid = mean(population)) %>% ungroup()

# implicitly rank the centroids, assigning increasing integers to decerasing vals
#assign the result as the cluster
dt %>% mutate("cluster" = frankv(centroid, ties.method = "dense"))

# A tibble: 4 x 4
  city   population cluster centroid
  <chr>       <dbl>   <int>    <dbl>
1 NYC            10       2      9.5
2 LA              9       2      9.5
3 Hawaii          1       1      1.5
4 Essex           2       1      1.5

##############################################
# Same as above
  
# 1st method:
A <- clusters %>% 
       mutate("cluster" = frankv(centroid, 
                                 ties.method = "dense"))

# 2nd method

clusterDT = data.table(old_label = 1:4, 
                       centroid = as.vector(clusters_obj$centers))
clusterDT[order(centroid), new_label := .I]

B <- clusters
B[, cluster := clusterDT[.SD, on=.(old_label = cluster_label), x.new_label]]



labels <- c("c1_mean", "c2_mean", "c3_mean", "c4_mean")
labels <- labels[order(cluster_means, decreasing=TRUE)]
order(cluster_means)


# conditional replacement
A[precip < 0, precip := 0]

#########################################
#
# clusters are 4, ..., 1
# we wantedd strip titles be different:
#
str_labels <- c("4" = "most precip.", 
                "3" ="less precip.", 
                "2" = "lesser precip.", 
                "1" = "least precip.")
facet_grid(~ emission ~ cluster,
           labeller=labeller(cluster = str_labels)
                        ) 


A <- ann_all_last_days %>%
     group_by(year, time_period, emission, cluster) %>%
     mutate(quan_90 = findInterval(get("annual_cum_precip"), 
                                    quantile(get("annual_cum_precip"), probs=0.9))) %>%
     data.table()
A <- A %>% filter(quan_90 == 1)

B <- ann_all_last_days %>% 
     group_by(year, time_period, emission, cluster) %>% 
     summarise(quan_90 = quantile(get("annual_cum_precip"), probs = 0.9)) %>% 
     data.table()

B <- merge(ann_all_last_days, B)
B <- B %>% filter(get("annual_cum_precip") >= quan_90)

setcolorder(B, colnames(A))

setorderv(B, colnames(B))
setorderv(A, colnames(A))



sept_summary_comp <- dplyr::left_join(x = sept_summary_comp, y = limited_cities, by = "location")



Choose every other element
start_ID = ID[seq(1, length(ID), 2)]

Every other 14th element
start_ID = ID[seq(1, length(ID), 14)]

Above works also on data tables.



# initiate data table of given size:
data <- setNames(data.table(matrix(nrow = 0, ncol = 3)), c("va", "vb", "vc"))





#######################################################################
#
#   How to add horizontal lines to facet plots and the legend!
#

ThreshCloud_2_rows_forForcing <- function(d1, trigger_dt){
  
  ##"""
  ##
  ## This function is modification of double_cloud_2_rows(.) to 
  ## plot threshold with shadowy background for Springer. Tossing the damn bloom
  ##
  ##"""
  
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xlabels <- sort(unique(d1$chill_season))
  xlabels <- xlabels[seq(1, length(xlabels), 15)]
  xlabels <- c(xlabels) # , "2097-2098"

  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]
  # xbreaks <- c(xbreaks, 2097)
  # xbreaks <- c(xbreaks, 2100)
  ylow = min(d1$value) - 5
  ylow = 90

  ymax = 220 # this is tuned by visual inspection!!
  # ymax = min(ymax, max(d1$value))
  
  ggplot(d1, aes(x=chill_season, y=value, fill=factor(variable))) +
  labs(x = "chill year", y = "day of year", fill = "data type") +
  # guides(fill=guide_legend(title="")) + 

  facet_wrap(. ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", fun =function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0) }, 
                              fun.max=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.1) }, 
                              fun.max=function(z) { quantile(z,0.9) }, 
               alpha=0.4) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.25) }, 
                              fun.max=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 

  stat_summary(geom = "line", fun=function(z) {quantile(z,0.5)}) +
  # 
  # plot horizontal lines for each city.
  #
  geom_hline(data = trigger_dt, 
             aes(yintercept = median_heatTriggerThresh, 
                 linetype = "heat accumulation trigger"), 
             color = "red", 
             size=1,
             show.legend = TRUE
             ) + 
  scale_linetype_manual(name = "Limits", 
                        labels = c("heat accumulation trigger"), 
                        values = c("heat accumulation trigger" = 1)) +
  
  scale_color_manual(values=c("darkgreen"), breaks=c("thresh"), labels=c("CP threshold")) +
  scale_fill_manual (values=c("darkgreen"), breaks=c("thresh"), labels=c("CP threshold")) +
  
  scale_x_continuous(breaks = xbreaks, label = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day) + 

  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing = unit(.5, "cm"),
        legend.text = element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + # theme_bw() + 
  coord_cartesian(ylim = c(ylow, ymax)) + 
   guides(colour = guide_legend(override.aes = list(linetype = 0)),
         fill = guide_legend(override.aes = list(linetype = 0)),
         shape = guide_legend(override.aes = list(linetype = 0)),
         linetype = guide_legend())

}

#######################################################################



