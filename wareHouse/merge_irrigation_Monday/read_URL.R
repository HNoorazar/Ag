################
library(tidyr)
library(dplyr)
library(readr)
library(tidyverse)

url <- "https://www.nass.usda.gov/Data_and_Statistics/County_Data_Files/Frequently_Asked_Questions/county_list.txt"

url_df <- read_lines(url, skip = 12) %>%
          data.frame(col = .) %>%
          mutate(col = str_replace_all(col, "\\t", " ")) %>%
          separate(col, into = paste0("X", 1:5), sep = "\\s{2,}", extra = "drop") %>%
          na.omit()

colnames(url_df) <- c("state_FIPS", "district_FIPS", "county_FIPS", "state_cnt_name", "flag")
url_df <- data.table(url_df)
url_df <- url_df[url_df$county_FIPS != "000", ]
url_df <- url_df[url_df$county_FIPS != "888", ]
url_df <- url_df[url_df$county_FIPS != "999", ]
url_df <- url_df[url_df$flag == 1]
url_df$FIPS <- with(url_df, paste0(state_FIPS, county_FIPS))
url_df$CDR_FIPS <- with(url_df, paste0(state_FIPS, district_FIPS))

url_dir <- "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/"
saveRDS(url_df, paste0(url_dir, "NASS_county.rds"))
write.csv(url_df, file = paste0(url_dir, "NASS_county_CRD.csv" ), row.names=FALSE)




