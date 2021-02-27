#### Print Probability Plots ####

print_prob_plots <- function(df){
  # Plot Size
  size <- 16
  
  #### DAILY PROB ####
  df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
    group_by(month) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df, df_monthly_exceedance, by = c("month")) %>% 
    filter(!is.na(group))
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    filter(group != "hist", month >= 10 | month <= 3) %>% 
    group_by(group, month, model) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95))),
              month_name = factor(first(month(time_stamp, label = TRUE)), 
                                  levels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'))) %>%
    gather(exceedance, prob, -month_name, -group, -month, -model) %>% 
    group_by(group, month, exceedance) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05)) %>% 
    plot_monthly_prob("Daily Prob")
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  p_octmar_exceedance_val <- df %>%
    filter(month >= 10 | month <= 3, group != "hist") %>% 
    group_by(group, model) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>% 
    gather(exceedance, prob, -model, -group) %>% 
    group_by(group, exceedance) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05)) %>% 
    plot_octmar_prob()
  
  p_prob_daily <- plot_grid(p_monthly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "vh", rel_widths = c(3.25, 1), axis = 'b')
  
  # save_plot("figures/daily_exceedance_probability_dry.pdf", p_prob, base_aspect_ratio = 2, scale = 1, limitsize = FALSE)
  
  #### 7 DAY PROB ####
  
  df_7 <- df %>% group_by(group, model) %>% mutate(precip = rollsum(precip, 7, align = "right", fill = NA))
  
  df_monthly_exceedance <- df_7 %>% filter(group == 'hist') %>% 
    group_by(month) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df_7, df_monthly_exceedance, by = c("month")) %>% 
    filter(!is.na(group))
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    filter(group != "hist", month >= 10 | month <= 3) %>% 
    group_by(group, month, model) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95))),
              month_name = factor(first(month(time_stamp, label = TRUE)), 
                                  levels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'))) %>%
    gather(exceedance, prob, -month_name, -group, -month, -model) %>% 
    group_by(group, month, exceedance) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05)) %>% 
    plot_monthly_prob("7 Day Prob")
  
  df_octmar_exceedance <- df_7 %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  p_octmar_exceedance_val <- df_7 %>%
    filter(month >= 10 | month <= 3, group != "hist") %>% 
    group_by(group, model) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>% 
    gather(exceedance, prob, -model, -group) %>% 
    group_by(group, exceedance) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05)) %>% 
    plot_octmar_prob()
  
  p_prob_7day <- plot_grid(p_monthly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "vh", rel_widths = c(3.25, 1), axis = 'b')
  
  
  # save_plot("figures/7_day_exceedance_probability_dry.pdf", p_prob, base_aspect_ratio = 2, scale = 1, limitsize = FALSE)
  
  #### MONTH PROB ####
  
  df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
    group_by(month, year) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(month) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df, df_monthly_exceedance, by = c("month")) %>% 
    filter(!is.na(group))
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    filter(group != "hist", month >= 10 | month <= 3) %>% 
    group_by(group, month, model, year) %>%
    summarise(precip = sum(precip),
              precip_80 = first(precip_80),
              precip_90 = first(precip_90),
              precip_95 = first(precip_95),
              time_stamp = first(time_stamp)) %>% 
    group_by(group, month, model) %>% 
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95))),
              month_name = factor(first(month(time_stamp, label = TRUE)), 
                                  levels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'))) %>%
    gather(exceedance, prob, -month_name, -model, -month, -group) %>% 
    group_by(group, month, exceedance) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05)) %>% 
    plot_monthly_prob("Monthly Prob")
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    group_by(group, month, model, year) %>% 
    summarise(precip = sum(precip)) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  p_octmar_exceedance_val <- df %>%
    filter(month >= 10 | month <= 3, group != "hist") %>% 
    group_by(group, month, model, year) %>% 
    summarise(precip = sum(precip)) %>% 
    group_by(group, model) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>%  
    gather(exceedance, prob, -model, -group) %>% 
    group_by(group, exceedance) %>% 
    mutate(prob_median = median(prob),
           hist_prob = case_when(exceedance == "prob_80" ~ 0.20,
                                 exceedance == "prob_90" ~ 0.10,
                                 exceedance == "prob_95" ~ 0.05)) %>% 
    plot_octmar_prob()
  
  p_prob_monthly <- plot_grid(p_monthly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "vh", rel_widths = c(3.25, 1), axis = 'b')
  
  p_all <- plot_grid(p_prob_daily, p_prob_7day, p_prob_monthly, ncol = 1, align = "vh")
  
  return(p_all)

}

