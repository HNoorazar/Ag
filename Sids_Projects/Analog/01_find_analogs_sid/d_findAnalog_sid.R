.libPaths("/data/hydro/R_libs35")
.libPaths()

library(data.table)
library(dplyr)
library(FNN)
library(EnvStats, lib.loc = "~/.local/lib/R3.5.1")

source_path = "/Users/hn/Documents/00_GitHub/Ag/Sids_Projects/Analog/core_analog_sid.R"
source_path = "/home/hnoorazar/sid/analog_pairwise_2021/core_analog_sid.R"
source(source_path)
options(digit=9)
options(digits=9)

################################################################################
################################################################################
# 
#                   Terminal arguments and parameters
# 
################################################################################
args = commandArgs(trailingOnly=TRUE)

emission = args[1] # rcp45 rcp85
# time_period = args[2]

print (paste0("emission is: ", emission))
# print (paste0("time_period is: ", time_period))
################################################################################
# 
#                     set up proper directories
# 
################################################################################
in_dir <- "/Users/hn/Documents/01_research_data/Sids_Projects/analog_data/"
in_dir <- "/data/hydro/users/Hossein/sids_projects/analog/00_data_from_sid/"

main_out <- "/data/hydro/users/Hossein/sids_projects/analog/"
out_dir <- paste0(main_out, "01_analogs_perLocModel/")

print (paste0("out_dir is : ",  out_dir))
if (dir.exists(out_dir) == F) { dir.create(path = out_dir, recursive = T) }

################################################################################
# 
#                   Read Data
# 
################################################################################

all_dt <- data.table(read.csv(paste0(in_dir, "Climate_Data_All_Variables.csv"), as.is=TRUE))
colnames(all_dt) <- tolower(colnames(all_dt))

#
#  future_dt includes the locations for which we are looking for an analog for.
#
future_data <- all_dt %>%
               filter(scenario == emission) %>%
               data.table()
# if (time_period == "mid_century"){
#   future_data <- future_data%>%
#                  filter(year >= 2030 & year <= 2070) %>%
#                  data.table()
# } else{
#   future_data <- future_data%>%
#                  filter(year >= 2070) %>%
#                  data.table()
# }

#
#  all_dt_usa will be the historical data that can come from all the USA
#
all_dt_usa <- all_dt %>%
              filter(scenario == "gridmet") %>%
              data.table()

#
#  2020 includes NAs
#
# all_dt_usa <- all_dt_usa %>%
#               filter(year < 2020) %>%
#               data.table()

all_dt_usa <- tidyr::drop_na(all_dt_usa)
n_nghs <- nrow(all_dt_usa)

################################################################################
start_time <- Sys.time()

# run the model for separate time frames
time_frames = c("2030_2070", "2070_2100")

non_numeric_cols <- colnames(all_dt)[1:4]
numeric_col <- colnames(all_dt)[5:length(colnames(all_dt))]
print (non_numeric_cols)
#
# I will add two extra for-loops outside the damn most
# interior for-loop and try if it works out, before making 100 jobs!!!!
#

# it seems this for-loop is already in the function find_NN_info_biofix(.)
# moreover, some locations have missing models! I do not know why! so,
# the second loop picks the model that are present for a given location!
# 
for (a_future_loc in unique(future_data$location)){ 
  a_loc_future <- future_data %>%
                  filter(location == a_future_loc) %>%
                  # filter(model == a_model) %>%
                  data.table()
  
  for (a_model in unique(a_loc_future$model)){
    a_loc_model_future <- a_loc_future %>%
                          filter(location == a_future_loc) %>%
                          filter(model == a_model) %>%
                          data.table()
    
    for (time in time_frames){
      if (time == time_frames[1]){
        local_dt_time <- a_loc_model_future %>% filter(year >= 2030 & year <= 2070) %>% data.table()
        } else if (time == time_frames[2]) {
        local_dt_time <- a_loc_model_future %>% filter(year >= 2070) %>% data.table()
      } 
      
      print (a_future_loc)
      print (a_model)
      print (time)
      print ("_____________________________________________")
      information = find_NN_info_biofix(ICV = all_dt_usa,
                                        historical_dt = all_dt_usa,
                                        future_dt = local_dt_time,
                                        n_neighbors = n_nghs,
                                        numeric_cols = numeric_col,
                                        non_numeric_cols = non_numeric_cols)

      NN_dist_tb = information[[1]]
      NN_loc_year_tb = information[[2]]
      NN_sigma_tb = information[[3]]

      saveRDS(NN_dist_tb,     paste0(out_dir, paste("/NN_dist_tb",     a_future_loc, gsub("-", "_", a_model), time, emission, sep="_"), ".rds"))
      saveRDS(NN_loc_year_tb, paste0(out_dir, paste("/NN_loc_year_tb", a_future_loc, gsub("-", "_", a_model), time, emission, sep="_"), ".rds"))
      saveRDS(NN_sigma_tb,    paste0(out_dir, paste("/NN_sigma_tb",    a_future_loc, gsub("-", "_", a_model), time, emission, sep="_"), ".rds"))
    }
  }
}



end_time <- Sys.time()
print( end_time - start_time)

