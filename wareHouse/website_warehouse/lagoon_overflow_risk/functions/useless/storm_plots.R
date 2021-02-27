require(dplyr)
require(ggplot2)

source('functions/multiplot.R') # plots gridded figures
source("functions/storm_calc.R") ## calculates return intensity

storm_plots <- function(d){
  
  # p1 <- d %>% filter(!is.na(group)) %>% 
  #   storm_calc(1) %>%
  #   ggplot() +
  #   geom_line(aes(x = return_period, y = XT, color = group)) +
  #   labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "Max Precip") +
  #   theme_bw()
  
  # p2 <- d %>% filter(!is.na(group)) %>%
  #   storm_calc(.999) %>%
  #   ggplot() +
  #   geom_line(aes(x = return_period, y = XT, color = group)) +
  #   labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "99.9 Percentile Precip") +
  #   theme_bw()
  
  p3 <- d %>% filter(!is.na(group)) %>%
    storm_calc(.99) %>%
    ggplot() +
    geom_line(aes(x = return_period, y = XT, color = group)) +
    labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "A) 99 Percentile Precip") +
    theme_bw()
  
  p4 <- d %>%
    filter(precip >= quantile(precip, .999), !is.na(group)) %>%
    group_by(group) %>% 
    ggplot() +
    stat_ecdf(aes(x = precip, color = group), geom = "line") +
    labs(title = "B) Precip > 99.9 percentile") +
    theme_bw()
  
  p5 <- d %>%
    filter(precip >= quantile(precip, .95), !is.na(group)) %>%
    ggplot() +
    stat_ecdf(aes(x = precip, color = group), geom = "line") +
    labs(title = "C) Precip > 95th percentile") +
    theme_bw()
  
  p6 <- d %>%
    ggplot() +
    geom_line(aes(y = precip, x = time_stamp)) +
    geom_point(aes(y = precip, x = time_stamp)) +
    labs(title = "D)") +
    theme_bw()
  
  # p7 <- d %>% filter(month <= 4 | month >= 9) %>%
  #   group_by(water_year) %>%
  #   filter(precip > 0) %>%
  #   distinct(time_stamp, .keep_all = TRUE) %>%
  #   summarise(rain_days = n()) %>%
  #   ggplot() +
  #   geom_point(aes(x = water_year, y = rain_days)) +
  #   labs(x = "Water Year", y = "No. of days with Precip") +
  #   theme_bw()
  # 
  # p8 <- d %>% distinct(time_stamp, .keep_all = TRUE) %>%
  #   filter(!is.na(group)) %>% 
  #   group_by(month, group) %>%
  #   summarise(month_precip = mean(precip)) %>%
  #   ggplot() +
  #   geom_line(aes(x = month, y = month_precip, color = group)) +
  #   labs(x = "Month", y = "Mean Monthly Precip (mm)") +
  #   theme_bw()
  # 
  # p9 <- d %>% filter(!is.na(group)) %>%
  #   ggplot() +
  #   geom_boxplot(aes(y = precip, x = group)) +
  #   scale_y_log10() +
  #   theme_bw()
  # 
  p10 <- d %>% filter(precip >= quantile(precip, .995), !is.na(group)) %>% 
    group_by(precip, group) %>% 
    summarise(days = n()) %>% 
    arrange(desc(precip)) %>% 
    group_by(group) %>% 
    mutate(days_above = cumsum(days)) %>% 
    ggplot() +
    geom_line(aes(x = precip, y = days_above, color = group)) +
    labs(title = "E) Precip > 99.5%") +
    theme_bw()
  
  p11 <- d %>% filter(precip >= quantile(precip, .95), !is.na(group)) %>% 
    group_by(precip, group) %>% 
    summarise(days = n()) %>% 
    arrange(desc(precip)) %>% 
    group_by(group) %>% 
    mutate(days_above = cumsum(days)) %>% 
    ggplot() +
    geom_line(aes(x = precip, y = days_above, color = group)) +
    labs(title = "F) Precip > 95%") +
    theme_bw()
  
  p12 <- d %>% mutate(precip_sum = stats::filter(precip,rep(1, 5), sides=2)) %>% 
    ggplot() +
    geom_line(aes(x = time_stamp, y = precip_sum)) +
    labs(title = "G) 5 Day window sum precip") +
    theme_bw()
  
  p13 <- d %>% mutate(precip = stats::filter(precip,rep(1, 5), sides=2)) %>% 
    filter(!is.na(group), !is.na(precip)) %>%
    storm_calc(.99) %>%
    ggplot() +
    geom_line(aes(x = return_period, y = XT, color = group)) +
    labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "h) 99 Percentile Precip on 5 day window") +
    theme_bw()
  
  p14 <- d %>% group_by(month, year) %>% 
    mutate(cum_precip = cumsum(precip)) %>% 
    ggplot() +
    geom_line(aes(x = time_stamp, y = cum_precip)) +
    theme_minimal()
  
  multiplot(p3, p4, p5, p6, p10, p11, p12, p13, p14, cols = 3)
}
