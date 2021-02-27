# size <- 20

# scaleFUN <- function(x) sprintf("%.2f", x)

plot_monthly_prob <- function(df, label){
  
  ggplot(df) +
    geom_jitter(aes(x = group, y = prob, color = group, fill = group), width = .2) +
    geom_line(aes(x = group, y = prob_median, group = model, color = "hist")) +
    geom_hline(aes(yintercept = hist_prob), color = "grey40", linetype = "longdash") +
    facet_grid(exceedance~month_name, switch = "x", scales = "free_y") +
    labs(y = label) +
    theme_linedraw() +
    theme(axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(),
          axis.title.x = element_blank(),
          legend.title = element_blank(), 
          legend.position = "none",
          strip.background = element_blank(), 
          strip.text.y = element_blank(),
          strip.placement = "outside",
          panel.spacing.y = unit(.75, "lines"),
          strip.text = element_text(colour = 'black')) +
    scale_color_wsu(palette = 'rev') +
    scale_fill_wsu(palette = 'rev') +
    # scale_color_viridis_d(option = "plasma") +
    # scale_fill_viridis_d(option = "plasma") +
    scale_y_continuous(labels=scales::percent)
}

plot_octmar_prob <- function(df){
  
  ggplot(df) +
    geom_jitter(aes(x = group, y = prob, color = group), width = .2) +
    geom_line(aes(x = group, y = prob_median, group = model, color = "hist")) +
    geom_hline(aes(yintercept = hist_prob), color = "grey40", linetype = "longdash") +
    facet_grid(exceedance~., switch = "x", scales = "free_y") +
    labs(x = "Oct - Mar") +
    theme_linedraw() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.title.y = element_blank(),
          legend.title = element_blank(),
          legend.position = "none",
          strip.background = element_blank(),
          axis.title.x = element_text(size = 8, face = "plain"),
          panel.spacing.y = unit(.75, "lines")) +
    scale_color_wsu(palette = 'rev') +
    scale_fill_wsu(palette = 'rev') +
    # scale_color_viridis_d(option = "plasma") +
    # scale_fill_viridis_d(option = "plasma") +
    scale_y_continuous(labels=scales::percent)
  
}

