library(dplyr)
library(forcats)
library(ggplot2)
library(cowplot)

plot_drydays_boxplot <- function(df, sel_climate_proj){
  
  df %>% 
    mutate(group = fct_relevel(as_factor(group), c("hist", "2040s", "2060s", "2080s")),
           season = case_when(month <= 3 ~ "Jan-Mar",
                              between(month, 4, 6) ~ "Apr-Jun",
                              between(month, 7, 9) ~ "Jul-Sep",
                              month >= 10 ~ "Oct-Dec")) %>% 
    filter(precip <= quantile(precip, 0.05, na.rm = TRUE), !is.na(group)) %>% 
    group_by(climate_proj, model, group, season, year) %>% 
    summarise(dry_days = n()) %>%
    # summarise(dry_days = median(dry_days)) %>%
    filter(climate_proj %in% c(NA, sel_climate_proj)) %>% 
    ggplot() +
    geom_boxplot(aes(x = group, y = dry_days)) +
    geom_jitter(aes(x = group, y = dry_days), width = 0.2, alpha = 0.2) +
    facet_grid(season~., scales = "free_y") +
    labs(y = "Number of Dry Days", title  = "Days under 5 Percentile of Precipitation") +
    theme(axis.title.x = element_blank())
  
}
