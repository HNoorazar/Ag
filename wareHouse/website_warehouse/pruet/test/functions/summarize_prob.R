

summarize_prob <- function(file_path){
  df <- read_rds(file_path) %>% filter(climate_proj %in% c(NA, "rcp85")) %>% 
    mutate(time_stamp = ymd(paste(year, month, day, sep="-")),
           water_year = year(time_stamp %m+% months(3)))
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  octmar_exceedance_val <- df %>%
    filter(month >= 10 | month <= 3, group != "hist") %>% 
    group_by(group, model) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>% 
    gather(exceedance, prob, -model, -group) %>% 
    group_by(group, exceedance) %>% 
    summarise(prob_median = median(prob))
  
  return(octmar_exceedance_val)
  
}

# map_df <- map_df %>% mutate(data = map(file_path, summarize_prob))
