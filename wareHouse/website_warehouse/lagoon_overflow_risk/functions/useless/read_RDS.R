### Load RDS Files ###

read_RDS <- function(file_name, climate_proj, model){
  hist <- readRDS(paste0("/data/pruett/RDS/historical/", file_name, ".rds"))
  df_futr <- readRDS(paste0("/data/pruett/RDS/",model, "/", climate_proj, "/", file_name, ".rds")) %>% 
    filter(year >= 2016)
  
  df_NA <- df_futr %>% 
           filter(year <= 2025 || year >= 2095) %>%
           mutate(group = NA)

  df_2040 <- df_futr %>% 
             filter(year >= 2025, year <= 2055) %>%
             mutate(group = "2040s")

  df_2060 <- df_futr %>% 
             filter(year >= 2045, year <= 2075) %>%
             mutate(group = "2060s")

  df_2080 <- df_futr %>% 
             filter(year >= 2065, year <= 2095) %>%
             mutate(group = "2080s")

  futr <- bind_rows(df_NA, df_2040, df_2060, df_2080) %>%
          mutate(group = as.factor(group))

  df <- rbind(hist, futr)
  df <- mutate(df, 
               time_stamp = ymd(paste(year, month, day, sep="-")),
               water_year = year(time_stamp + month(3)))
  return(df)
}
