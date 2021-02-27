library(shiny)
library(shinydashboard)
library(shinyBS)
library(rgdal)    # for readOGR and others
library(maps)
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(data.table)
library(reshape2)
library(RColorBrewer)
#library(plotly)
#library(Hmisc)

data_dir = "/data/codmoth_data"
d = data.table(readRDS(paste0(data_dir,"/combinedData.rds")))
# ordering time frame levels 
d$timeFrame <-as.factor(d$timeFrame)
d$timeFrame <- factor(d$timeFrame, levels = levels(d$timeFrame)[c(4,1,2,3)])

d_rcp45 = data.table(readRDS(paste0(data_dir,"/combinedData_rcp45.rds")))
names(d_rcp45)[names(d_rcp45) == "ClimateGroup"] = "timeFrame"
d_rcp45$location = paste0(d_rcp45$latitude, "_", d_rcp45$longitude)


d1 <- data.table(readRDS(paste0(data_dir, "/subData.rds")))
d1$month = as.factor(d1$month)
levels(d1$month) = c("January", "February",
                       "March", "April", "May",
                       "June", "July", "August",
                       "September", "October",
                       "November", "December")
d1$location = paste0(d1$latitude, "_", d1$longitude)

d1_rcp45 <- data.table(readRDS(paste0(data_dir, "/subData_rcp45.rds")))
d1_rcp45$month = as.factor(d1_rcp45$month)
levels(d1_rcp45$month) = c("January", "February",
                       "March", "April", "May",
                       "June", "July", "August",
                       "September", "October",
                       "November", "December")
d1_rcp45$location = paste0(d1_rcp45$latitude, "_", d1_rcp45$longitude)


RdBu_reverse <- rev(brewer.pal(11, "RdBu"))
head(d1)

diap <- data.table(readRDS(paste0(data_dir, "/diapause_map_data1.rds")))
diap_rcp45 <- data.table(readRDS(paste0(data_dir, "/diapause_map_data1_rcp45.rds")))

bloom <- data.table(readRDS(paste0(data_dir, "/bloom_data.rds")))
bloom_rcp45 <- data.table(readRDS(paste0(data_dir, "/bloom_data_rcp45.rds")))

# sub_d = subset(subset(d, !is.na(d$timeFrame)), select = c(timeFrame, year, location, AGen1_0.25, AGen1_0.5, AGen1_0.75, AGen2_0.25, AGen2_0.5, AGen2_0.75, AGen3_0.25, AGen3_0.5, AGen3_0.75, AGen4_0.25, AGen4_0.5, AGen4_0.75))
# melt_d = melt(sub_d, id.vars = c("timeFrame", "location", "year"), na.rm = TRUE)
# 
# # hack to show same y-scale ranges for each row
# dummy <- data.frame(timeFrame = rep("Historical", 24),
#                     variable = rep(c("AGen1_0.25", "AGen1_0.5", "AGen1_0.75", "AGen2_0.25", "AGen2_0.5", "AGen2_0.75", "AGen3_0.25", "AGen3_0.5", "AGen3_0.75", "AGen4_0.25", "AGen4_0.5", "AGen4_0.75"), each = 2),
#                     value = c(rep(c(55, 205), 3), rep(c(130, 320), 3), rep(c(160, 370), 3), rep(c(195, 370), 3)))
# dummy$timeFrame <- as.factor(dummy$timeFrame)
# dummy$variable <- as.factor(dummy$variable)
# dummy$value <- as.integer(dummy$value)
# 
# #faceted violin plots with bar plot
# vplot <- ggplot(data = melt_d, aes(x = timeFrame, y = value, fill = timeFrame)) +
#   geom_violin() +
#   geom_boxplot(width=0.1, outlier.colour = NA) +
#   # hack for same y-scale range
#   geom_point(data = dummy, aes(x = timeFrame, y = value), colour = "white", alpha = 0) +
#   stat_summary(geom="text", fun.y=median,
#                aes(label=sprintf("%d", ..y..)),
#                position=position_nudge(x=0.225), size=3.5) +
#   facet_wrap(~ variable, scales = "free_y", ncol = 3) +
#   theme_bw() +
#   labs(title = "Generation Quartile, Time Frame Vs Day of Year", x = "Time Frame", y = "Day of Year", fill = "Time Frame") +
#   theme(axis.text.x = element_blank(),
#         axis.ticks.x = element_blank(),
#         axis.title = element_text(face = "bold", size = 12),
#         panel.grid.major.x = element_blank(),
#         panel.grid.minor.x = element_blank(),
#         legend.title = element_text(face = "bold", size = 12),
#         legend.text = element_text(size = 10),
#         legend.background = element_rect(color = "black"),
#         legend.position = "bottom",
#         strip.text = element_text(size = 10, face = "bold"),
#         plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
# 
# 
# freq_data = subset(d, select = c(timeFrame, year, location, AGen3_0.25, AGen3_0.5, AGen3_0.75, AGen4_0.25, AGen4_0.5, AGen4_0.75))
# freq_data_melted = melt(freq_data, id.vars = c("timeFrame", "location", "year"), na.rm = FALSE)
# f1 = freq_data_melted[, .(years_range = (max(year) - min(year) + 1)), by = list(variable, timeFrame, location)][order(variable, timeFrame, location)]
# f2 = freq_data_melted[complete.cases(freq_data_melted$value), .(years_freq = uniqueN(year)), by = list(variable, timeFrame, location)][order(variable, timeFrame, location)]
# # left join - merge both tables
# f = merge(f1, f2, by = c("variable", "timeFrame", "location"), all.x = TRUE)
# # replace na values by 0
# f[is.na(years_freq), years_freq := 0]
# f$percentage = (f$years_freq / f$years_range) * 100
# 
# bplot <- ggplot(data = f, aes(x = timeFrame, y = percentage, fill = timeFrame)) +
#   geom_boxplot(width = 0.25) +
#   stat_summary(geom="text", fun.y=quantile,
#                aes(label=sprintf("%1.1f", ..y..), color = timeFrame),
#                position=position_nudge(x=0.35), size=3.5, show.legend = FALSE) +
#   facet_wrap(~ variable) +
#   theme_bw() +
#   labs(title = "Generation Percentage Years in each Time Frame", x = "Time Frame", y = "Percentage of Years", fill = "Time Frame") +
#   theme(
#     axis.text = element_text(size = 12),
#     axis.title = element_text(face = "bold", size = 14),
#     panel.grid.major.x = element_blank(),
#     panel.grid.minor.x = element_blank(),
#     legend.title = element_text(face = "bold", size = 14),
#     legend.text = element_text(size = 12),
#     legend.background = element_rect(color = "black"),
#     legend.position = "right",
#     strip.text = element_text(size = 14, face = "bold"),
#     plot.title = element_text(face = "bold", size = 16, hjust = 0.5))
