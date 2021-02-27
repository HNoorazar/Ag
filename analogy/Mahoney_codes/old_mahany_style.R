find_NN_info_W4G_STOPWORKING <- function(all_dt_usa, local_dt, n_neighbors){
  # remove extra columns
  if ("treatment" %in% colnames(all_dt_usa)) {all_dt_usa <- within(all_dt_usa, remove(treatment))}
  if ("treatment" %in% colnames(local_dt)) {local_dt <- within(local_dt, remove(treatment))}

  # sort the columns of data tables so they both have the same order, if theydo not.
  columns_ord <- c("year", "location","ClimateScenario",
                   "medianDoY", "NumLarvaGens_Aug", 
                   "mean_escaped_Gen1", "mean_escaped_Gen2",
                   "mean_escaped_Gen3", "mean_escaped_Gen4",
                   "mean_gdd", "mean_precip")

  setcolorder(all_dt_usa, columns_ord)
  setcolorder(local_dt, columns_ord)

  all_us_locs <- unique(all_dt_usa$location)

  # 9 local locations are not in the all_us data!!!
  local_locations <- unique(local_dt$location)
  local_locations <- local_locations[which(local_locations %in% all_us_locs)]
  local_dt <- local_dt %>% filter(location %in% local_locations)

  future_years <- unique(local_dt$year)

  # Principal component truncation rule
  trunc.SDs <- 0.1 # truncation 

  A <- as.data.frame(all_dt_usa)
  B <- as.data.frame(local_dt)
  C <- as.data.frame(all_dt_usa)

  numeric_cols <- c("medianDoY", "NumLarvaGens_Aug", "mean_escaped_Gen1", 
                    "mean_escaped_Gen2", "mean_escaped_Gen3", "mean_escaped_Gen4", 
                    "mean_gdd", "mean_precip")
  non_numeric_cols = c("year", "location", "ClimateScenario")
  
  NN_dist_tb = data.table()
  NNs_loc_year_tb = data.table()
  NN_sigma_tb = data.table()

  for (loc in local_locations){
    # loc = local_locations[1]
    Bj <- B %>% filter(location==loc)
    Cj <- C %>% filter(location==loc)

    # standard deviation of 1951-1990 interannual variability in each climate 
    # variable, ignoring missing years
    Cj.sd <- apply(Cj[, numeric_cols], MARGIN=2, FUN = sd, na.rm = T)
    Cj.sd[Cj.sd<(10^-10)] = 1

    A_prime <- A
    A_prime[, numeric_cols] <- sweep(A_prime[, numeric_cols], MARGIN = 2, STATS = Cj.sd, FUN = `/`) 
    
    # standardize the analog pool
    Bj_prime <- Bj
    Bj_prime[, numeric_cols] <- sweep(Bj[, numeric_cols], MARGIN = 2, STATS = Cj.sd, FUN = `/`)

    # standardize the reference ICV
    Cj_prime <- Cj
    Cj_prime[, numeric_cols] <- sweep(Cj[, numeric_cols], MARGIN = 2, STATS = Cj.sd, FUN = `/`)

    #################################################################################
    #
    # Thigns below were in the following for-loop
    #
    #################################################################################
    ## Step 2: Extract the principal components (PCs) of 
    ##         the reference period ICV and project all data onto these PCs
    ## 
      
    # Principal components analysis. The !is.na(apply(...)) 
    # term is there simply to select all years with complete observations in all variables. 
    PCA <- prcomp(Cj_prime[, numeric_cols][!is.na(apply(Cj_prime[, numeric_cols], 1, mean)) ,])

    # find the number of PCs to retain using the PC truncation
    # rule of eigenvector stdev > the truncation threshold
    PCs <- max(which(unlist(summary(PCA)[1])>trunc.SDs))

    # project the reference ICV onto the PCs which is the same as analog pool:
    X <- as.data.frame(predict(PCA, A_prime))
    X <- cbind(A_prime[, non_numeric_cols], X)

    # project the projected future conditions onto the PCs
    Yj <- as.data.frame(predict(PCA, Bj_prime[, numeric_cols]))
    Yj = cbind(Bj_prime[, non_numeric_cols], Yj)

    Zj <- as.data.frame(predict(PCA, Cj_prime[, numeric_cols]))
    Zj <- cbind(Cj_prime[, non_numeric_cols], Zj)

    ## Step 3a: express PC scores as standardized
    #           anomalies of reference interannual variability
      
    # standard deviation of 1951-1990 interannual 
    # variability in each principal component, ignoring missing years
    Zj_sd <- apply(Zj[, 4:(PCs+3)], MARGIN = 2, FUN = sd, na.rm=T)

    # standardize the analog pool   
    X_prime <- sweep(X[, 4:(PCs+3)], MARGIN=2, Zj_sd, FUN = `/`)
    X_prime <- cbind(X[, 1:3], X_prime)

    # standardize the projected conditions
    Yj_prime <- sweep(Yj[, 4:(PCs+3)], MARGIN=2, Zj_sd, FUN = `/`)
    Yj_prime = cbind(Yj[, 1:3], Yj_prime)
    
    NN_list <- get.knnx(data = X_prime[, 4:(PCs+3)], 
                        query = Yj_prime[, 4:(PCs+3)], k=n_neighbors, algorithm="brute")

    NN_idx <- NN_list$nn.index 
    NN_dist <- NN_list$nn.dist

    # extract the rows corresponding to nearest neighbors indices
    NNs_loc_year <- X_prime[as.vector(NN_idx), c('year', 'location')]

    # reshape the long list to wide data table
    NNs_loc_year <- Reduce(cbind, 
                          split(NNs_loc_year, 
                                rep(1:n_neighbors, each=(nrow(NNs_loc_year)/n_neighbors)))) %>%
                   data.table()

    # rename columns 
    NN_dist <- NN_dist %>% data.table()
    
    ############################################################
    # 
    # This is for one location, and different years. 
    # So, the location column and ClimateScenario can be dropped.
    # We already know what they are from loc and the input file.
    #
    ############################################################
    # percentile of the nearest neighbour distance on the chi distribution with 
    # degrees of freedom equaling the dimensionality of the distance measurement (PCs)
    NN_chi <- pchi(as.vector(NN_list$nn.dist), PCs)

    # values of the chi percentiles on a 
    # standard half-normal distribution (chi distribution with one degree of freedom)
    # NN.sigma[which(proxy==j)] <- qchi(NN.chi, 1)
    NN_sigma <- qchi(NN_chi, 1)
    NN_sigma_df = Reduce(cbind, 
                          split(NN_sigma, 
                                rep(1:n_neighbors, each=(length(NN_sigma)/n_neighbors)))) %>%
                   data.table()
    
    
    NN_dist_tb <- rbind(NN_dist_tb, NN_dist)
    NNs_loc_year_tb <- rbind(NNs_loc_year_tb, NNs_loc_year)
    NN_sigma_tb <-  rbind(NN_sigma_tb, NN_sigma_df)
    rm(NN_sigma_df, NN_sigma, NN_chi, NN_dist, NNs_loc_year)
  }
  names(NN_dist_tb) <- paste0("NN_", c(1:n_neighbors))
  names(NNs_loc_year_tb) <- paste0(names(NNs_loc_year_tb), paste0("_NN_", rep(1:n_neighbors, each=2)))
  names(NN_sigma_tb) <- paste0("sigma_NN_", c(1:n_neighbors))
  return(list(NN_dist_tb, NNs_loc_year_tb, NN_sigma_tb))
}

find_NN_info <- function(all_dt_usa, local_dt, n_neighbors){
  # remove extra columns
  all_dt_usa <- within(all_dt_usa, remove(treatment, mean_escaped_Gen4))
  local_dt <- within(local_dt, remove(treatment, mean_escaped_Gen4))

  # sort the columns of data tables so they both have the same order, if theydo not.
  columns_ord <- c("year", "location","ClimateScenario",
                   "medianDoY", "NumLarvaGens_Aug", 
                   "mean_escaped_Gen1", "mean_escaped_Gen2",
                   "mean_escaped_Gen3",
                   "mean_gdd", "mean_precip")

  setcolorder(all_dt_usa, columns_ord)
  setcolorder(local_dt, columns_ord)

  min_max <- FALSE
  # min_max scaling of historical data and future data
  # This will put the range of the data in [0, 1]
  # which takes care of different scales, however,
  # it changes the correlations. Do we want that?
  # why that happens? What are consequences?
  if (min_max == TRUE){
    minimums <- apply(all_dt_usa[, numeric_cols], MARGIN=2, FUN=min)
    maximums <- apply(all_dt_usa[, numeric_cols], MARGIN=2, FUN=max)
    range_col <- maximums - minimums
    all_dt_usa[, numeric_cols] <- (all_dt_usa[, numeric_cols] - minimums) / range_col
    local_dt[, numeric_cols] <- (local_dt[, numeric_cols]- minimums) / range_col
  }

  all_us_locs <- unique(all_dt_usa$location)

  # 9 local locations are not in the all_us data!!!
  local_locations <- unique(local_dt$location)
  local_locations <- local_locations[which(local_locations %in% all_us_locs)]
  local_dt <- local_dt %>% filter(location %in% local_locations)

  future_years <- unique(local_dt$year)

  # Principal component truncation rule
  trunc.SDs <- 0.1 # truncation 

  A <- as.data.frame(all_dt_usa)
  B <- as.data.frame(local_dt)
  C <- as.data.frame(all_dt_usa)

  numeric_cols <- c("medianDoY", "NumLarvaGens_Aug", 
                    "mean_escaped_Gen1", "mean_escaped_Gen2",
                    "mean_escaped_Gen3",
                    "mean_gdd", "mean_precip")
  NN_dist_tb = data.table()
  NNs_loc_year_tb = data.table()
  NN_sigma_tb = data.table()

  for (loc in local_locations){
    # loc = local_locations[1]
    Bj <- B %>% filter(location==loc)
    Cj <- C %>% filter(location==loc)

    # standard deviation of 1951-1990 interannual variability in each climate 
    # variable, ignoring missing years
    Cj.sd <- apply(Cj[, numeric_cols], MARGIN=2, FUN = sd, na.rm = T)
    Cj.sd[Cj.sd<(10^-10)] = 1

    A_prime <- A
    A_prime[, numeric_cols] <- sweep(A_prime[, numeric_cols], MARGIN = 2, STATS = Cj.sd, FUN = `/`) 
    
    # standardize the analog pool
    Bj_prime <- Bj
    Bj_prime[, numeric_cols] <- sweep(Bj[, numeric_cols], MARGIN = 2, STATS = Cj.sd, FUN = `/`)

    # standardize the reference ICV
    Cj_prime <- Cj
    Cj_prime[, numeric_cols] <- sweep(Cj[, numeric_cols], MARGIN = 2, STATS = Cj.sd, FUN = `/`)

    #################################################################################
    #
    # Thigns below were in the following for-loop
    #
    #################################################################################
    ## Step 2: Extract the principal components (PCs) of 
    ##         the reference period ICV and project all data onto these PCs
    ## 
      
    # Principal components analysis. The !is.na(apply(...)) 
    # term is there simply to select all years with complete observations in all variables. 
    PCA <- prcomp(Cj_prime[, numeric_cols][!is.na(apply(Cj_prime[, numeric_cols], 1, mean)) ,])

    # find the number of PCs to retain using the PC truncation
    # rule of eigenvector stdev > the truncation threshold
    PCs <- max(which(unlist(summary(PCA)[1])>trunc.SDs))

    # project the reference ICV onto the PCs which is the same as analog pool:
    X <- as.data.frame(predict(PCA, A_prime))
    X <- cbind(A_prime[, 1:3], X)

    # project the projected future conditions onto the PCs
    Yj <- as.data.frame(predict(PCA, Bj_prime[, numeric_cols]))
    Yj = cbind(Bj_prime[, 1:3], Yj)

    Zj <- as.data.frame(predict(PCA, Cj_prime[, numeric_cols]))
    Zj <- cbind(Cj_prime[, 1:3], Zj)

    ## Step 3a: express PC scores as standardized
    #           anomalies of reference interannual variability
      
    # standard deviation of 1951-1990 interannual 
    # variability in each principal component, ignoring missing years
    Zj_sd <- apply(Zj[, 4:(PCs+3)], MARGIN = 2, FUN = sd, na.rm=T)

    # standardize the analog pool   
    X_prime <- sweep(X[, 4:(PCs+3)], MARGIN=2, Zj_sd, FUN = `/`)
    X_prime <- cbind(X[, 1:3], X_prime)

    # standardize the projected conditions
    Yj_prime <- sweep(Yj[, 4:(PCs+3)], MARGIN=2, Zj_sd, FUN = `/`)
    Yj_prime = cbind(Yj[, 1:3], Yj_prime)
    
    NN_list <- get.knnx(data = X_prime[, 4:(PCs+3)], 
                        query = Yj_prime[, 4:(PCs+3)], k=n_neighbors, algorithm="brute")

    NN_idx <- NN_list$nn.index 
    NN_dist <- NN_list$nn.dist

    # extract the rows corresponding to nearest neighbors indices
    NNs_loc_year <- X_prime[as.vector(NN_idx), c('year', 'location')]

    # reshape the long list to wide data table
    NNs_loc_year <- Reduce(cbind, 
                           split(NNs_loc_year, 
                                 rep(1:n_neighbors, each=(nrow(NNs_loc_year)/n_neighbors)))) %>%
                   data.table()

    # rename columns 
    NN_dist <- NN_dist %>% data.table()
    names(NN_dist) <- paste0("NN_", c(1:n_neighbors))
    names(NNs_loc_year) <- paste0(names(NNs_loc_year), paste0("_NN_", rep(1:n_neighbors, each=2)))

    ############################################################
    # 
    # This is for one location, and different years. 
    # So, the location column and ClimateScenario can be dropped.
    # We already know what they are from loc and the input file.
    #
    #
    ############################################################
    # percentile of the nearest neighbour distance 
    # on the chi distribution with 
    # degrees of freedom equaling the dimensionality 
    # of the distance measurement (PCs)
    NN_chi <- pchi(as.vector(NN_list$nn.dist), PCs)

    # values of the chi percentiles on a 
    # standard half-normal distribution 
    # (chi distribution with one degree of freedom)
    # NN.sigma[which(proxy==j)] <- qchi(NN.chi, 1)
    NN_sigma <- qchi(NN_chi, 1)
    NN_sigma_df = Reduce(cbind, 
                          split(NN_sigma, 
                                rep(1:n_neighbors, each=(length(NN_sigma)/n_neighbors)))) %>%
                   data.table()
    names(NN_sigma_df) <- paste0("sigma_NN_", c(1:n_neighbors))
    
    NN_dist_tb = rbind(NN_dist_tb, NN_dist)
    NNs_loc_year_tb =  rbind(NNs_loc_year_tb, NNs_loc_year)
    NN_sigma_tb =  rbind(NN_sigma_tb, NN_sigma_df)
  }

  return(list(NN_dist_tb, NNs_loc_year_tb, NN_sigma_tb))
}
