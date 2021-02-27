### Load RDS Files ###

read_RDS <- function(file_name, climate_proj){
  hist <- readRDS(paste0("RDS/historical/", file_name, ".rds"))
  futr <- readRDS(paste0("RDS/", climate_proj, "/", file_name, ".rds"))
  rbind(hist, futr)
}