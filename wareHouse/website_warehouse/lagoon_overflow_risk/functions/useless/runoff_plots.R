runoff_plots <- function(df){
  
  size <- 16
  
  df <- df %>% filter(!is.na(group)) %>% 
    mutate(combined = runoff+baseflow, 
           water_year = year(time_stamp + month(3)))
    
    # set plot scale
    size <- 14
    
    #### PROBABILITY PLOT MONTHLY ####
    
    # Calculate monthly exceedance
    df_monthly_exceedance <- df %>% filter(group == 'hist') %>% 
      group_by(month, year) %>%
      summarise(combined = sum(combined)) %>% 
      group_by(month) %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95))))
    
    df_monthly_prob <- full_join(df, df_monthly_exceedance, by = "month")
    
    p_monthly_exceedance_val <- df_monthly_prob %>% 
      group_by(group, month, year) %>% 
      summarise(combined = sum(combined),
                combined_80 = first(combined_80),
                combined_90 = first(combined_90),
                combined_95 = first(combined_95)) %>% 
      group_by(group, month) %>% 
      mutate(prob = rank(-combined)/(n()+1)) %>%
      summarise(prob_80 = nth(prob, which.min(abs(combined-combined_80))),
                prob_90 = nth(prob, which.min(abs(combined-combined_90))),
                prob_95 = nth(prob, which.min(abs(combined-combined_95)))) %>%
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
      summarise(combined = sum(combined)) %>% 
      ungroup() %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95))))
    
    
    p_yearly_exceedance_val <- df %>% 
      group_by(month, group, year) %>% 
      summarise(combined = sum(combined)) %>% 
      group_by(group) %>% 
      mutate(prob = rank(-combined)/(n()+1)) %>%
      summarise(prob_80 = nth(prob, which.min(abs(combined-df_yearly_exceedance$combined_80))),
                prob_90 = nth(prob, which.min(abs(combined-df_yearly_exceedance$combined_90))),
                prob_95 = nth(prob, which.min(abs(combined-df_yearly_exceedance$combined_95)))) %>%
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
      summarise(combined = sum(combined)) %>% 
      ungroup() %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95))))
    
    p_octmar_exceedance_val <- df %>% filter(month >= 10 | month <= 3) %>%
      group_by(month, group, year) %>% 
      summarise(combined = sum(combined)) %>% 
      group_by(group) %>% 
      mutate(prob = rank(-combined)/(n()+1)) %>%
      summarise(prob_80 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_80))),
                prob_90 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_90))),
                prob_95 = nth(prob, which.min(abs(combined-df_octmar_exceedance$combined_95)))) %>%
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
    

    
    #### EXCEEDANCE PLOT MONTHLY ####
    
    p_monthly_exceedance <- df %>% group_by(group, month, year) %>%
      summarise(combined = sum(combined)) %>% 
      group_by(group, month) %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95)))) %>%
      ggplot() +
      geom_col(aes(x = group, y = combined_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_95, color = group, fill = group), alpha = 0.33) +
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
      summarise(combined = sum(combined)) %>% 
      group_by(group) %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95)))) %>%
      ggplot() +
      geom_col(aes(x = group, y = combined_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_95, color = group, fill = group), alpha = 0.33) +
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
      summarise(combined = sum(combined)) %>% 
      group_by(group) %>% 
      mutate(prob = rank(combined)/(n()+1)) %>%
      summarise(combined_80 = nth(combined, which.min(abs(prob-0.8))),
                combined_90 = nth(combined, which.min(abs(prob-0.9))),
                combined_95 = nth(combined, which.min(abs(prob-0.95)))) %>%
      ggplot() +
      geom_col(aes(x = group, y = combined_80, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_90, color = group, fill = group), alpha = 0.33) +
      geom_col(aes(x = group, y = combined_95, color = group, fill = group), alpha = 0.33) +
      labs(x = " ", y = " ") +
      theme_classic(base_size = size) +
      theme(axis.text.x=element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none") +
      scale_color_wsu() +
      scale_fill_wsu()
    
    p_exceedance_m <- plot_grid(p_monthly_exceedance, p_yearly_exceedance, p_octmar_exceedance, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")
    
  
    
    #### combined PLOT ####
    p_monthly_combined <- df %>% group_by(month, year, group) %>%
      summarise(total_combined = sum(combined)) %>%
      ggplot() +
      geom_boxplot(aes(x = group, y = total_combined, color = group, fill = group), alpha = 0.75) +
      # geom_jitter(aes(x = group, y = total_combined, color = group), alpha = 0.5, width = 0.05) +
      theme_classic(base_size = size) +
      facet_grid(.~month, switch = "x") +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "bottom",
            strip.background = element_blank()) +
      scale_color_wsu() +
      scale_fill_wsu() +
      labs(y = "combined", x = " ")
    
    p_yearly_combined <- df %>% group_by(year, group) %>%
      summarise(total_combined = sum(combined)) %>%
      ggplot() +
      geom_boxplot(aes(x = group, y = total_combined, color = group, fill = group), alpha = 0.75) +
      # geom_jitter(aes(x = group, y = total_combined, color = group), alpha = 0.5, width = 0.05) +
      theme_classic(base_size = size) +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none") +
      scale_color_wsu() +
      scale_fill_wsu() +
      labs(y = " ", x = "Annual")
    
    p_octmar_combined <- df %>% mutate(water_year = year(time_stamp + month(3))) %>%
      group_by(group, water_year) %>%
      filter(month >= 10 | month <= 3) %>%
      summarise(total_combined = sum(combined)) %>%
      ggplot() +
      geom_boxplot(aes(x = group, y = total_combined, color = group, fill = group), alpha = 0.75) +
      # geom_jitter(aes(x = group, y = total_combined, color = group), alpha = 0.5, width = 0.05) +
      theme_classic(base_size = size) +
      theme(axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            legend.title = element_blank(),
            legend.position = "none") +
      scale_color_wsu() +
      scale_fill_wsu() +
      labs(y = " ", x = "Oct - Mar")
    
    #### FINAL PLOT COMBINE ####
    
    legend <- get_legend(p_monthly_combined)
    
    p_combined <- plot_grid(p_monthly_combined + theme(legend.position = "none"), p_yearly_combined, p_octmar_combined, rel_widths = c(6, 1, 1), align = 'h', nrow = 1, axis = "b")
    
    p_grid <- plot_grid(p_prob_m, p_exceedance_m, p_combined, ncol = 1, align = "v", axis = 'l')
    
    plot_grid(p_grid, legend, ncol = 1, rel_heights = c(1,.05))
}

