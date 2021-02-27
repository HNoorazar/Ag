
### read binary files ###
read_binary_ <- function(file_name, file_path, hist){
  
  if (hist) {
    start_year <- 1979
    end_year <- 2015
  }
  
  if (!hist) {
    start_year <- 2006
    end_year <- 2099
  }
  
  ## create year month day values ##
  create_ymdvalues <- function(data_start_year, data_end_year) {
    Years <- seq(data_start_year, data_end_year)
    nYears <- length(Years)
    daycount_in_year <- 0
    moncount_in_year <- 0
    yearrep_in_year <- 0
    
    for(i in 1:nYears){
      ly <- leap_year(Years[i])
      
      if(ly == TRUE){
        days_in_mon <- c(31,29,31,30,31,30,31,31,30,31,30,31)
      }
      
      else{
        days_in_mon <- c(31,28,31,30,31,30,31,31,30,31,30,31)
      }
      
      for( j in 1:12){
        daycount_in_year <- c(daycount_in_year,seq(1,days_in_mon[j]))
        moncount_in_year <- c(moncount_in_year,rep(j,days_in_mon[j]))
        yearrep_in_year <- c(yearrep_in_year,rep(Years[i],days_in_mon[j]))
      }
    }
    
    daycount_in_year <- daycount_in_year[-1] #delete the leading 0
    moncount_in_year <- moncount_in_year[-1]
    yearrep_in_year <- yearrep_in_year[-1]
    ymd <- cbind(yearrep_in_year, moncount_in_year, daycount_in_year)
    colnames(ymd) <- c("year","month","day")
    return(ymd)
  }
  
  ## read binary data and add dates ##
  readbinarydata_addmdy <- function(filename, ymd) {
    Nofvariables <- 4 # number of varaibles or column in the forcing data file
    Nrecords <- nrow(ymd)
    ind <- seq(1, Nrecords*Nofvariables, Nofvariables)
    fileCon = file(filename, "rb")
    temp <- readBin(fileCon, integer(), size = 2, n = Nrecords * Nofvariables, endian="little")
    dataM <- matrix(0, Nrecords, 4)
    k <- 1
    dataM[1:Nrecords,1] <- temp[ind]/40.00  #precip data
    dataM[1:Nrecords,2] <- temp[ind+1]/100.00  #Max temperature data
    dataM[1:Nrecords,3] <- temp[ind+2]/100.00  #Min temperature data
    dataM[1:Nrecords,4] <- temp[ind+3]/100.00  #Wind speed data
    AllData <- cbind(ymd, dataM)
    # calculate daily GDD
    colnames(AllData) <- c("year","month","day","precip","tmax","tmin","winspeed")
    close(fileCon)
    return(AllData)
  }
  
  # create date ranges #
  ymd_file <- create_ymdvalues(start_year, end_year)
  
  # read data #
  
  if (hist) {
    df_hist <- readbinarydata_addmdy(file_path, ymd_file) %>% 
      as_tibble() %>%
      mutate(group = "hist")
    return(df_hist)
  }
  
  if (!hist) {
    df_futr <-
      readbinarydata_addmdy(file_path, ymd_file) %>%
      as_tibble()
    
    df_NA <- df_futr %>% filter(year <= 2025 | year >= 2095) %>%
      mutate(group = NA)
    
    df_2040 <- df_futr %>% filter(year >= 2025, year <= 2055) %>%
      mutate(group = "2040s")
    
    df_2060 <- df_futr %>% filter(year >= 2045, year <= 2075) %>%
      mutate(group = "2060s")
    
    df_2080 <- df_futr %>% filter(year >= 2065, year <= 2095) %>%
      mutate(group = "2080s")
    
    bind_rows(df_NA, df_2040, df_2060, df_2080) %>%
      mutate(group = as.factor(group))
  }
}


