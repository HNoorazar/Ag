## Calculate Gumball EV Distrobution

require(dplyr)
require(purrr)
require(tidyr)
require(tibble)

calc <- function(df, percentile) {
  calc_KT <- function(return_period) {
    (-sqrt(6)/pi)*(0.5772 + log(log(return_period/(return_period - 1))))
  }
  
  calc_XT <- function(KT, mean_precip, sd_precip) {
    mean_precip + KT*sd_precip
  }
  
  df %>% as.data.frame() %>% 
    group_by(year, group) %>%
    summarise(max_hourly_precip = quantile(precip, percentile) / 24) %>%
    group_by(group) %>%
    summarise(
      mean_precip = mean(max_hourly_precip),
      sd_precip = sd(max_hourly_precip)) %>%
    mutate(
      return_period = list(seq(5, 25, 1)),
      KT = map(return_period, calc_KT),
      XT = pmap(list(KT, mean_precip, sd_precip), calc_XT)) %>%
    unnest()
}