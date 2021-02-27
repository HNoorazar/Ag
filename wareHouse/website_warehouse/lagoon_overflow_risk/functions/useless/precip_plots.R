

precip_plots <- function(df){
  
  #### INIT CALCS ####
  
  df <- filter(df, !is.na(group))
  df <- mutate(df, water_year = year(time_stamp + month(3)),
               month_name = month(time_stamp, label = TRUE))

  # set plot scale
  size <- 14
  
  #### PROBABILITY PLOT DAILY ####
  
  # Calculate monthly exceedance
  df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
    group_by(month) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df, df_monthly_exceedance, by = "month")
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    group_by(group, month) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95))),
              month_name = first(month_name)) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = "Daily Prob") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_yearly_exceedance <- df %>% filter(group == 'hist') %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  
  p_yearly_exceedance_val <- df %>% 
    group_by(group) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))

  p_octmar_exceedance_val <- df %>%
    filter(month >= 10 | month <= 3) %>% 
    group_by(group) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_prob <- plot_grid(p_monthly_exceedance_val, p_yearly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "h", rel_widths = c(6, 1, 1), axis = 'b')
  
  #### PROBABILITY PLOT MONTHLY ####
  
  # Calculate monthly exceedance
  df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
    group_by(month, year) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(month) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df, df_monthly_exceedance, by = "month")
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    group_by(group, month, year) %>% 
    summarise(precip = sum(precip),
              precip_80 = first(precip_80),
              precip_90 = first(precip_90),
              precip_95 = first(precip_95)) %>% 
    group_by(group, month) %>% 
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month, switch = "x") +
    labs(x = " ", y = "Monthly Prob") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_yearly_exceedance <- df %>% filter(group == 'hist') %>% 
    group_by(month, year) %>% 
    summarise(precip = sum(precip)) %>% 
    ungroup() %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  
  p_yearly_exceedance_val <- df %>% 
    group_by(month, group, year) %>% 
    summarise(precip = sum(precip)) %>% 
    group_by(group) %>% 
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_95)))) %>%
    ggplot() +
    # geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    # geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    # geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_octmar_exceedance <- df %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    group_by(month, group, water_year) %>% 
    summarise(precip = sum(precip)) %>% 
    ungroup() %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  p_octmar_exceedance_val <- df %>% filter(month >= 10 | month <= 3) %>%
    group_by(month, group, year) %>% 
    summarise(precip = sum(precip)) %>% 
    group_by(group) %>% 
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_prob_m <- plot_grid(p_monthly_exceedance_val, p_yearly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "h", rel_widths = c(6, 1, 1), axis = 'b')
  
  #### PROBABILITY PLOT 7 DAY WINDOW ####
  
  # Calculate monthly exceedance
  
  df_7 <- df %>% group_by(group) %>% mutate(precip = rollsum(precip, 7, align = "right", fill = NA))
  
  df_monthly_exceedance <- df_7 %>% filter(group == 'hist') %>% 
    group_by(month) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  df_monthly_prob <- full_join(df_7, df_monthly_exceedance, by = "month")
  
  
  p_monthly_exceedance_val <- df_monthly_prob %>% 
    group_by(group, month) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-precip_95))),
              month_name = first(month_name)) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = "7 day Prob") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_yearly_exceedance <- df_7 %>% filter(group == 'hist') %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  
  p_yearly_exceedance_val <- df_7 %>% 
    group_by(group) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_yearly_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(), 
          axis.ticks.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  df_octmar_exceedance <- df_7 %>% filter(group == 'hist', month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))))
  
  p_octmar_exceedance_val <- df_7 %>%
    filter(month >= 10 | month <= 3) %>% 
    group_by(group) %>%
    mutate(prob = rank(-precip)/(n()+1)) %>%
    summarise(prob_80 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_80))),
              prob_90 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_90))),
              prob_95 = nth(prob, which.min(abs(precip-df_octmar_exceedance$precip_95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = prob_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = prob_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_prob_7 <- plot_grid(p_monthly_exceedance_val, p_yearly_exceedance_val,p_octmar_exceedance_val, nrow = 1, align = "h", rel_widths = c(6, 1, 1), axis = 'b')
  
  #### EXCEEDANCE PLOT DAILY ####
  
  p_monthly_exceedance <- df %>% group_by(group, month, year) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    group_by(group, month) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))),
              month_name = month(first(time_stamp), label = TRUE)) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = "Daily Exceed") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()

  p_yearly_exceedance <- df %>% group_by(group) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()

  p_octmar_exceedance <- df %>%
    group_by(group, water_year) %>%
    filter(month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    group_by(group) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()

  p_exceedance <- plot_grid(p_monthly_exceedance, p_yearly_exceedance, p_octmar_exceedance, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")

  
  #### EXCEEDANCE PLOT MONTHLY ####
  
  p_monthly_exceedance <- df %>% group_by(group, month, year) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(group, month) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month, switch = "x") +
    labs(x = " ", y = "Month Exceed") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()

  p_yearly_exceedance <- df %>% group_by(group, month, water_year) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(group) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()

  p_octmar_exceedance <- df %>% 
    filter(month >= 10 | month <= 3) %>%
    group_by(group, water_year, month) %>%
    summarise(precip = sum(precip)) %>% 
    group_by(group) %>% 
    mutate(prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()

  p_exceedance_m <- plot_grid(p_monthly_exceedance, p_yearly_exceedance, p_octmar_exceedance, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")

  
  #### EXCEEDANCE PLOT 7 DAY WINDOW ####
  
  p_monthly_exceedance <- df %>% group_by(group) %>% 
    mutate(precip = rollsum(precip, 7, align = "right", fill = NA)) %>% 
    group_by(group, month, year) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    group_by(group, month) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95))),
              month_name = month(first(time_stamp), label = TRUE)) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = "7 Day Exceed") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_yearly_exceedance <- df %>% group_by(group) %>% 
    mutate(precip = rollsum(precip, 7, align = "right", fill = NA),
           prob = rank(precip)/(n()+1)) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_octmar_exceedance <- df %>% group_by(group) %>% 
    mutate(precip = rollsum(precip, 7, align = "right", fill = NA)) %>% 
    group_by(group, water_year) %>%
    filter(month >= 10 | month <= 3) %>%
    mutate(prob = rank(precip)/(n()+1)) %>%
    group_by(group) %>%
    summarise(precip_80 = nth(precip, which.min(abs(prob-0.8))),
              precip_90 = nth(precip, which.min(abs(prob-0.9))),
              precip_95 = nth(precip, which.min(abs(prob-0.95)))) %>%
    ggplot() +
    geom_col(aes(x = group, y = precip_80, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_90, color = group, fill = group), alpha = 0.33) +
    geom_col(aes(x = group, y = precip_95, color = group, fill = group), alpha = 0.33) +
    # facet_grid(.~month_name, switch = "x") +
    labs(x = " ", y = " ") +
    theme_classic(base_size = size) +
    theme(axis.text.x=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu()
  
  p_exceedance_7 <- plot_grid(p_monthly_exceedance, p_yearly_exceedance, p_octmar_exceedance, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")
  
  
  #### DRY DAYS PLOT ####
  p_monthly_dry <- df %>% group_by(month, year, group) %>%
    filter(precip == 0) %>% 
    summarise(dry_days = n(),
              month_name = month(time_stamp[1], label = TRUE)) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = dry_days, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          strip.text = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = "# Dry Days", x = " ")

  p_yearly_dry <- df %>% group_by(year, group) %>%
    filter(precip == 0) %>% 
    summarise(dry_days = n()) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = dry_days, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    # facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = " ", x = "Annual")

  p_octmar_dry <- df %>% mutate(water_year = year(time_stamp + month(3))) %>%
    group_by(group, water_year) %>%
    filter(month >= 10 | month <= 3, precip == 0) %>%
    summarise(dry_days = n()) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = dry_days, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    # facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = " ", x = "Oct - Mar")
  
  p_dry <- plot_grid(p_monthly_dry, p_yearly_dry, p_octmar_dry, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")

  #### PRECIP PLOT ####
  p_monthly_precip <- df %>% group_by(month, year, group) %>%
    summarise(total_precip = sum(precip),
              month_name = month(time_stamp[1], label = TRUE)) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = total_precip, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "bottom",
          strip.background = element_blank()) +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = "Precip", x = " ")

  p_yearly_precip <- df %>% group_by(year, group) %>%
    summarise(total_precip = sum(precip)) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = total_precip, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    # facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = " ", x = "Annual")

  p_octmar_precip <- df %>% mutate(water_year = year(time_stamp + month(3))) %>%
    group_by(group, water_year) %>%
    filter(month >= 10 | month <= 3) %>%
    summarise(total_precip = sum(precip)) %>%
    ggplot() +
    geom_boxplot(aes(x = group, y = total_precip, color = group, fill = group), alpha = 0.75) +
    # geom_jitter(aes(x = group, y = total_precip, color = group), alpha = 0.5, width = 0.05) +
    theme_classic(base_size = size) +
    # facet_grid(.~month_name, switch = "x") +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title = element_blank(),
          legend.position = "none") +
    scale_color_wsu() +
    scale_fill_wsu() +
    labs(y = " ", x = "Oct - Mar")

  #### FINAL PLOT COMBINE ####
  
  legend <- get_legend(p_monthly_precip)

  p_precip <- plot_grid(p_monthly_precip + theme(legend.position = "none"), p_yearly_precip, p_octmar_precip, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")

  p_grid <- plot_grid(p_prob, p_prob_m, p_prob_7, p_exceedance, p_exceedance_m, p_exceedance_7, p_dry, p_precip, ncol = 1, align = "v", axis = 'l')

  plot_grid(p_grid, legend, ncol = 1, rel_heights = c(1,.05))
}
