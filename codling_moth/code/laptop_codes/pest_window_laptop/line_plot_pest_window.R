rm(list=ls())
library(data.table)
library(ggplot2)
data_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/pest_window/"
color_ord = c("grey70", "dodgerblue", "olivedrab4", "darkgoldenrod1")

input_name = "rcp45_50.csv"
rcp45_50 <- data.table(read.csv(paste0(data_dir, input_name)))
rcp45_50[rcp45_50$ClimateGroup == "colder", ]$ClimateGroup = "Cooler Areas"
rcp45_50[rcp45_50$ClimateGroup == "warmer", ]$ClimateGroup = "Warmer Areas"

rcp45_50 = melt(rcp45_50, id = c("ClimateGroup", "CountyGroup"))

rcp45_50$CountyGroup = factor(rcp45_50$CountyGroup, levels = c("Historical",
                                                               "2040's",
                                                               "2060's",
                                                               "2080's"))
br_plt <- ggplot(rcp45_50, aes(x=CountyGroup, y=value, color=variable)) + 
          geom_point() +
          #geom_line() + 
          geom_line(group= rcp45_50$CountyGroup) + 
          facet_grid(. ~ ClimateGroup, scales = "free") +
          labs(y="Julian day (median)") +
          ylim(0, 300) +
          theme_bw() + 
          theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
			    axis.text.x = element_text(size = 9, color="black"),
			    axis.title.x=element_blank(),
	            legend.position="bottom",
		        legend.spacing.x = unit(.1, 'cm'),
		        legend.title=element_blank(),
			    legend.text=element_text(size=10),
			    legend.key.size = unit(.4, "cm"),
	            panel.grid.major = element_line(size = 0.05),
		        panel.grid.minor = element_line(size = 0.2)) + 
		  scale_fill_manual(values=color_ord,
		                    name="Time\nperiod", 
		                    labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4")) +
		  scale_color_manual(values=color_ord,
			                 name="Time\nperiod", 
			                 labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4"))

ggsave("rcp45_50_line.png", br_plt, path=data_dir, dpi=500, device="png", width=5.5, height=3.1, unit="in")
###############################################################################################################
###########################
###########################          rcp85_50
###########################
rm(list=ls())
library(data.table)
library(ggplot2)
data_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/pest_window/"
color_ord = c("grey70", "dodgerblue", "olivedrab4", "darkgoldenrod1")

input_name = "rcp85_50.csv"
rcp85_50 <- data.table(read.csv(paste0(data_dir, input_name)))
rcp85_50[rcp85_50$ClimateGroup == "colder", ]$ClimateGroup = "Cooler Areas"
rcp85_50[rcp85_50$ClimateGroup == "warmer", ]$ClimateGroup = "Warmer Areas"

rcp85_50 = melt(rcp85_50, id = c("ClimateGroup", "CountyGroup"))

rcp85_50$CountyGroup = factor(rcp85_50$CountyGroup, levels = c("Historical",
                                                               "2040's",
                                                               "2060's",
                                                               "2080's"))
br_plt <- ggplot(rcp85_50, aes(x=CountyGroup, y=value, color=variable)) + 
          geom_point() +
          #geom_line() + 
          geom_line(group= rcp85_50$CountyGroup) + 
          facet_grid(. ~ ClimateGroup, scales = "free") +
          labs(y="Julian day (median)") +
          ylim(0, 300) +
          theme_bw() + 
          theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
			    axis.text.x = element_text(size = 9, color="black"),
			    axis.title.x=element_blank(),
	            legend.position="bottom",
		        legend.spacing.x = unit(.1, 'cm'),
		        legend.title=element_blank(),
			    legend.text=element_text(size=10),
			    legend.key.size = unit(.4, "cm"),
	            panel.grid.major = element_line(size = 0.05),
		        panel.grid.minor = element_line(size = 0.2)) + 
		  scale_fill_manual(values=color_ord,
		                    name="Time\nperiod", 
		                    labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4")) +
		  scale_color_manual(values=color_ord,
			                 name="Time\nperiod", 
			                 labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4"))

ggsave("rcp85_50_line.png", br_plt, path=data_dir, dpi=500, device="png", width=5.5, height=3.1, unit="in")

###########################
###########################          25 %
###########################
###############################################################################################################
###############################################################################################################
###########################
###########################          rcp45_25
###########################
rm(list=ls())
library(data.table)
library(ggplot2)
data_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/pest_window/"
color_ord = c("grey70", "dodgerblue", "olivedrab4", "darkgoldenrod1")

input_name = "rcp45_25.csv"
rcp45_25 <- data.table(read.csv(paste0(data_dir, input_name)))
rcp45_25[rcp45_25$ClimateGroup == "colder", ]$ClimateGroup = "Cooler Areas"
rcp45_25[rcp45_25$ClimateGroup == "warmer", ]$ClimateGroup = "Warmer Areas"

rcp45_25 = melt(rcp45_25, id = c("ClimateGroup", "CountyGroup"))
rcp45_25$CountyGroup = factor(rcp45_25$CountyGroup, levels = c("Historical",
                                                               "2040's",
                                                               "2060's",
                                                               "2080's"))
br_plt <- ggplot(rcp45_25, aes(x=CountyGroup, y=value, color=variable)) + 
          geom_point() +
          #geom_line() + 
          geom_line(group= rcp45_25$CountyGroup) + 
          facet_grid(. ~ ClimateGroup, scales = "free") +
          labs(y="Julian day (median)") +
          ylim(0, 300) +
          theme_bw() + 
          theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
			    axis.text.x = element_text(size = 9, color="black"),
			    axis.title.x=element_blank(),
	            legend.position="bottom",
		        legend.spacing.x = unit(.1, 'cm'),
		        legend.title=element_blank(),
			    legend.text=element_text(size=10),
			    legend.key.size = unit(.4, "cm"),
	            panel.grid.major = element_line(size = 0.05),
		        panel.grid.minor = element_line(size = 0.2)) + 
		  scale_fill_manual(values=color_ord,
		                    name="Time\nperiod", 
		                    labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4")) +
		  scale_color_manual(values=color_ord,
			                 name="Time\nperiod", 
			                 labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4")) 

ggsave("rcp45_25_line.png", br_plt, path=data_dir, dpi=500, device="png", width=5.5, height=3.1, unit="in")

rm(rcp45_25) 
###############################################################################################################
###########################
###########################          rcp85_25
###########################
rm(list=ls())
library(data.table)
library(ggplot2)
data_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/pest_window/"
color_ord = c("grey70", "dodgerblue", "olivedrab4", "darkgoldenrod1")

input_name = "rcp85_25.csv"
rcp85_25 <- data.table(read.csv(paste0(data_dir, input_name)))
rcp85_25[rcp85_25$ClimateGroup == "colder", ]$ClimateGroup = "Cooler Areas"
rcp85_25[rcp85_25$ClimateGroup == "warmer", ]$ClimateGroup = "Warmer Areas"

rcp85_25 = melt(rcp85_25, id = c("ClimateGroup", "CountyGroup"))

rcp85_25$CountyGroup = factor(rcp85_25$CountyGroup, levels = c("Historical",
                                                               "2040's",
                                                               "2060's",
                                                               "2080's"))
br_plt <- ggplot(rcp85_25, aes(x=CountyGroup, y=value, color=variable)) + 
          geom_point() +
          #geom_line() + 
          geom_line(group= rcp85_25$CountyGroup) + 
          facet_grid(. ~ ClimateGroup, scales = "free") +
          labs(y="Julian day (median)") +
          ylim(0, 300) +
          theme_bw() + 
          theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
			    axis.text.x = element_text(size = 9, color="black"),
			    axis.title.x=element_blank(),
	            legend.position="bottom",
		        legend.spacing.x = unit(.1, 'cm'),
		        legend.title=element_blank(),
			    legend.text=element_text(size=10),
			    legend.key.size = unit(.4, "cm"),
	            panel.grid.major = element_line(size = 0.05),
		        panel.grid.minor = element_line(size = 0.2)) + 
		  scale_fill_manual(values=color_ord,
		                    name="Time\nperiod", 
		                    labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4")) +
		  scale_color_manual(values=color_ord,
			                 name="Time\nperiod", 
			                 labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4"))
ggsave("rcp85_25_line.png", br_plt, path=data_dir, dpi=500, device="png", width=5.5, height=3.1, unit="in")

###########################
###########################          75 %
###########################
###############################################################################################################
###############################################################################################################
###########################
###########################          rcp45_75
###########################
rm(list=ls())
library(data.table)
library(ggplot2)
data_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/pest_window/"
color_ord = c("grey70", "dodgerblue", "olivedrab4", "darkgoldenrod1")

input_name = "rcp45_75.csv"
rcp45_75 <- data.table(read.csv(paste0(data_dir, input_name)))
rcp45_75[rcp45_75$ClimateGroup == "colder", ]$ClimateGroup = "Cooler Areas"
rcp45_75[rcp45_75$ClimateGroup == "warmer", ]$ClimateGroup = "Warmer Areas"

rcp45_75 = melt(rcp45_75, id = c("ClimateGroup", "CountyGroup"))
rcp45_75$CountyGroup = factor(rcp45_75$CountyGroup, levels = c("Historical",
                                                               "2040's",
                                                               "2060's",
                                                               "2080's"))
br_plt <- ggplot(rcp45_75, aes(x=CountyGroup, y=value, color=variable)) + 
          geom_point() +
          #geom_line() + 
          geom_line(group= rcp45_75$CountyGroup) + 
          facet_grid(. ~ ClimateGroup, scales = "free") +
          labs(y="Julian day (median)") +
          ylim(0, 300) +
          theme_bw() + 
          theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
			    axis.text.x = element_text(size = 9, color="black"),
			    axis.title.x=element_blank(),
	            legend.position="bottom",
		        legend.spacing.x = unit(.1, 'cm'),
		        legend.title=element_blank(),
			    legend.text=element_text(size=10),
			    legend.key.size = unit(.4, "cm"),
	            panel.grid.major = element_line(size = 0.05),
		        panel.grid.minor = element_line(size = 0.2)) + 
		  scale_fill_manual(values=color_ord,
		                    name="Time\nperiod", 
		                    labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4")) +
		  scale_color_manual(values=color_ord,
			                 name="Time\nperiod", 
			                 labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4"))
ggsave("rcp45_75_line.png", br_plt, path=data_dir, dpi=500, device="png", width=5.5, height=3.1, unit="in")

rm(rcp45_75) 
###############################################################################################################
###########################
###########################          rcp85_75
###########################
rm(list=ls())
library(data.table)
library(ggplot2)
data_dir = "/Users/hn/Documents/GitHub/Kirti/codling_moth/to_write_paper/figures/pest_window/"
color_ord = c("grey70", "dodgerblue", "olivedrab4", "darkgoldenrod1")

input_name = "rcp85_75.csv"
rcp85_75 <- data.table(read.csv(paste0(data_dir, input_name)))
rcp85_75[rcp85_75$ClimateGroup == "colder", ]$ClimateGroup = "Cooler Areas"
rcp85_75[rcp85_75$ClimateGroup == "warmer", ]$ClimateGroup = "Warmer Areas"

rcp85_75 = melt(rcp85_75, id = c("ClimateGroup", "CountyGroup"))

rcp85_75$CountyGroup = factor(rcp85_75$CountyGroup, levels = c("Historical",
                                                               "2040's",
                                                               "2060's",
                                                               "2080's"))
br_plt <- ggplot(rcp85_75, aes(x=CountyGroup, y=value, color=variable)) + 
          geom_point() +
          #geom_line() + 
          geom_line(group= rcp85_75$CountyGroup) + 
          facet_grid(. ~ ClimateGroup, scales = "free") +
          labs(y="Julian day (median)") +
          ylim(0, 300) +
          theme_bw() + 
          theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
			    axis.text.x = element_text(size = 9, color="black"),
			    axis.title.x=element_blank(),
	            legend.position="bottom",
		        legend.spacing.x = unit(.1, 'cm'),
		        legend.title=element_blank(),
			    legend.text=element_text(size=10),
			    legend.key.size = unit(.4, "cm"),
	            panel.grid.major = element_line(size = 0.05),
		        panel.grid.minor = element_line(size = 0.2)) + 
		  scale_fill_manual(values=color_ord,
		                    name="Time\nperiod", 
		                    labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4")) +
		  scale_color_manual(values=color_ord,
			                 name="Time\nperiod", 
			                 labels=c("Gen. 1","Gen. 2","Gen. 3","Gen. 4"))

ggsave("rcp85_75_line.png", br_plt, path=data_dir, dpi=500, device="png", width=5.5, height=3.1, unit="in")
rm(rcp85_75)