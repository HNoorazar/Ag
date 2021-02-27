### This file reads the combined data, 
### grabs the USA information from the file
### converts the coloumns' data type to a proper form to work with. (e.g. year is saved as '2040' as opposed to just 2040)
### saves the data as RDS file to the disk and carries on to plot the bar plots.
###

rm(list=ls())
library(data.table)
library(dplyr)
data_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/share_website/"
plot_path = data_dir
share <- data.table(read.csv(paste0(data_dir, "giwd_s_fpu_rdx.csv"), header= FALSE))
colnames(share) <- c("model", "location", "year", "FPU")

tofind <- c("_USA")
share_USA <- share[grep(paste(tofind, collapse = "|"), share$location), ]

share_USA$numeric_year <- numeric()
share_USA$loc <- character()

end_count = dim(share_USA)[1]
for (ii in 1:end_count){
	share_USA$numeric_year[ii] = as.numeric(substr(share_USA[ii, year], 2, 5))
}

for (ii in 1:end_count){
	share_USA$loc[ii] = substr(share_USA$location[ii], 2, 4)
}

share_USA$mod <- character()
for (ii in 1:end_count){
	share_USA$mod[ii] = substr(share_USA$model[ii], 2, 24)
}

share_USA = within(share_USA, remove(location))
share_USA = within(share_USA, remove(year))
share_USA = within(share_USA, remove(model))

colnames(share_USA) <- c("FPU", "year", "location", "model")


### just pick up with or without co2
for (ii in 1:end_count){
	share_USA$model[ii] = tolower(unlist(strsplit(share_USA$model[ii], "-"))[3])
}

#saveRDS(share_USA, paste0(data_dir, "USA.rds"))
##################################################################
USA_B1 <- share_USA 
USA_B2 <- share_USA
USA_F1 <- share_USA
USA_F2 <- share_USA

USA_B1 = USA_B1[USA_B1$year <= 2010]
USA_B1$timePeriod = "B1"

USA_B2 = USA_B2[USA_B2$year <= 2018]
USA_B2$timePeriod = "B2"

USA_F1 = USA_F1[USA_F1$year <= 2021 & USA_F1$year <= 2050]
USA_F1$timePeriod = "F1"

USA_F2 = USA_F2[USA_F2$year <= 2041 & USA_F2$year <= 2050]
USA_F2$timePeriod = "F2"
####################
USA_time = rbind(USA_B1, USA_B2, USA_F1, USA_F2)

USA_time_NoCo2 = USA_time[USA_time$model == "noco2"]
USA_time_WCo2 = USA_time[USA_time$model == "wco2"]

USA_time_NoCo2$location = as.factor(USA_time_NoCo2$location)
USA_time_WCo2$location = as.factor(USA_time_WCo2$location)

USA_time_NoCo2$timePeriod = as.factor(USA_time_NoCo2$timePeriod)
USA_time_WCo2$timePeriod = as.factor(USA_time_WCo2$timePeriod)
##################################################################
# Compute Medians
df_NoCo2 <- data.frame(USA_time_NoCo2)
df_NoCo2 <- (df_NoCo2 %>% group_by(location, timePeriod))
medians_NoCo2 <- (df_NoCo2 %>% summarise(med = median(FPU)))

p = ggplot(data=medians_NoCo2, aes(x=location, y=med)) +
	geom_bar(aes(fill = timePeriod), stat="identity", position="dodge") + 
	labs(x="State", y="FPU median (no co2)") +
	theme_bw() + 
	theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
	      legend.position="bottom",
	      legend.title=element_blank(),
	      legend.text=element_text(size=10),
	      legend.key.size = unit(.4, "cm"))

ggsave(filename=paste0("noCo2", ".png"), 
	   plot=p, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=500, 
	   device="png")


df_WCo2 <- data.frame(USA_time_WCo2)
df_WCo2 <- (df_WCo2 %>% group_by(location, timePeriod))
medians_WCo2 <- (df_WCo2 %>% summarise(med = median(FPU)))

p = ggplot(data=medians_WCo2, aes(x=location, y=med)) +
	geom_bar(aes(fill = timePeriod), stat="identity", position="dodge") + 
	labs(x="State", y="FPU median (with co2)") +
	theme_bw() + 
	theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
	      legend.position="bottom",
	      legend.title=element_blank(),
	      legend.text=element_text(size=10),
	      legend.key.size = unit(.4, "cm"))

ggsave(filename=paste0("withCo2", ".png"), 
	   plot=p, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=500, 
	   device="png")



