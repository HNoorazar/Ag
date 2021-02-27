# panel.grid.major = element_line(size=0.2), # inside theme
# panel.grid.minor = element_blank(),

ThreshCloud_2_rows_forForcing <- function(d1){
  
  ##"""
  ##
  ## This function is modification of double_cloud_2_rows(.) to 
  ## plot threshold with shadowy background for Springer. Tossing the damn bloom
  ##
  ##"""
  
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xlabels <- sort(unique(d1$chill_season))
  xlabels <- xlabels[seq(1, length(xlabels), 15)]
  xlabels <- c(xlabels) # , "2097-2098"

  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]
  # xbreaks <- c(xbreaks, 2097)
  # xbreaks <- c(xbreaks, 2100)
  ylow = min(d1$value) - 5
  # ylow = 90

  ymax = 200
  ymax = min(ymax, max(d1$value))
  
  ggplot(d1, aes(x=chill_season, y=value, fill=factor(variable))) +
  labs(x = "chill year", y = "day of year", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_wrap(. ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", fun =function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0) }, 
                              fun.max=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.1) }, 
                              fun.max=function(z) { quantile(z,0.9) }, 
               alpha=0.4) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.25) }, 
                              fun.max=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen"), breaks=c("thresh"), labels=c("CP threshold")) +
  scale_fill_manual (values=c("darkgreen"), breaks=c("thresh"), labels=c("CP threshold")) +
  
  scale_x_continuous(breaks = xbreaks, label = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day
                     ) + 
  # limits = c(ylow, ymax) This shit removes data beyod the limits, use coord_cartesian(xlim = c(-5000, 5000)) 
  # shade on the background
  # geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = 110, ymax = Inf), 
  #            fill = 'grey47', alpha = 0.004) + 
  geom_hline(yintercept = 110, linetype = "solid" , color = "red", size = 1) + 

  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + # theme_bw() + 
  coord_cartesian(ylim = c(ylow, ymax)) 

}

cloudy_frost <- function(d1, colname="chill_dayofyear", fil){
  if (colname=="fifty_perc_chill_DoY"){
     # bloom color
     cls <- "orange"
     } else if(colname == "chill_dayofyear"){
      # frost color
      cls <- "deepskyblue"
      } else { # colname == "thresh_DoY")
        # threshold color
        cls <- "darkgreen"
  }
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 10)]
  xbreaks <- c(xbreaks)

  ggplot(d1, aes(x=chill_season, y=get(colname), 
                 fill=fil, group=time_period)) +
  labs(x = "chill year", y = "day of year") + #, fill = "Climate Group"
  # guides(fill=guide_legend(title="Time period")) + 
  facet_grid(. ~ emission ~ location) + # scales = "free"
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0) }, 
                              fun.ymax=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun.y=function(z) {quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.1) }, 
               fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +

  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.25) }, 
                              fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) +

  stat_summary(geom="line", fun.y=function(z) {quantile(z,0.5) }, 
               size = 1) + 
  # scale_x_continuous(breaks=seq(1970, 2100, 10)) +
  scale_x_discrete(breaks = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day) +
  scale_color_manual(values = cls) +
  scale_fill_manual(values = cls) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        )
}

double_cloud <- function(d1){
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xlabels <- sort(unique(d1$chill_season))
  xlabels <- xlabels[seq(1, length(xlabels), 15)]
  xlabels <- c(xlabels) # , "2097-2098"

  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]
  # xbreaks <- c(xbreaks, 2100)
  # xbreaks <- c(xbreaks, 2097)
  ylow = min(d1$value) - 5
  ymax = min(260, max(d1$value))
  ymax = 260
  
  ggplot(d1, aes(x=chill_season, y=value, fill=factor(variable))) +
  labs(x = "chill year", y = "day of year", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_grid(. ~ emission ~ location, scales="free") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0) }, 
                              fun.ymax=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.1) }, 
                              fun.ymax=function(z) { quantile(z,0.9) }, 
               alpha=0.4) +

  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.25) }, 
                              fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 

  stat_summary(geom="line", fun.y=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("thresh", "fifty_perc_DoY"),
                     labels=c("CP threshold", "bloom"))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("thresh", "fifty_perc_chill_DoY"),
                    labels=c("CP threshold", "bloom")) +
  
  scale_x_continuous(breaks = xbreaks, label = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day
                     ) + # limits = c(ylow, ymax) This shit removes data beyod the limits, use coord_cartesian(xlim = c(-5000, 5000)) 
  
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + 
  # theme_bw() + 
  coord_cartesian(ylim = c(ylow, ymax)) 

}
####################################################################################

double_cloud_2_rows <- function(d1){
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  xlabels <- sort(unique(d1$chill_season))
  xlabels <- xlabels[seq(1, length(xlabels), 15)]
  xlabels <- c(xlabels) # , "2097-2098"

  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]
  # xbreaks <- c(xbreaks, 2097)
  # xbreaks <- c(xbreaks, 2100)
  ylow = min(d1$value) - 5
  ymax = min(260, max(d1$value))
  # ylow = 90
  ymax = 260
  
  ggplot(d1, aes(x=chill_season, y=value, fill=factor(variable))) +
  labs(x = "chill year", y = "day of year", fill = "data type") +
  guides(fill=guide_legend(title="")) + 
  facet_wrap(. ~ location, scales="fixed") +
  # geom_line(aes(fill=factor(Timeframe), color=factor(Timeframe) )) +
  stat_summary(geom="ribbon", fun =function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0) }, 
                              fun.max=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.1) }, 
                              fun.max=function(z) { quantile(z,0.9) }, 
               alpha=0.4) +

  stat_summary(geom="ribbon", fun=function(z) { quantile(z,0.5) }, 
                              fun.min=function(z) { quantile(z,0.25) }, 
                              fun.max=function(z) { quantile(z,0.75) }, 
               alpha=0.8) + 

  stat_summary(geom="line", fun=function(z) {quantile(z,0.5)}) +

  scale_color_manual(values=c("darkgreen", "orange"),
                     breaks=c("thresh", "fifty_perc_DoY"),
                     labels=c("CP threshold", "bloom"))+
  
  scale_fill_manual(values=c("darkgreen", "orange"),
                    breaks=c("thresh", "fifty_perc_chill_DoY"),
                    labels=c("CP threshold", "bloom")) +
  
  scale_x_continuous(breaks = xbreaks, label = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day
                     ) + # limits = c(ylow, ymax) This shit removes data beyod the limits, use coord_cartesian(xlim = c(-5000, 5000)) 
  
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) + # theme_bw() + 
  coord_cartesian(ylim = c(ylow, ymax)) 

}
####################################################################################


cloudy_frost_2_rows <- function(d1, colname="chill_dayofyear", fil){
  if (colname=="fifty_perc_chill_DoY"){
     # bloom color
     cls <- "orange"
     } else if(colname == "chill_dayofyear"){
      # frost color
      cls <- "deepskyblue"
      } else { # colname == "thresh_DoY")
        # threshold color
        cls <- "darkgreen"
  }
  d1$chill_season <- gsub("chill_", "", d1$chill_season)
  d1$chill_season <- substr(d1$chill_season, 1, 4)
  d1$chill_season <- as.numeric(d1$chill_season)
  
  xbreaks <- sort(unique(d1$chill_season))
  xbreaks <- xbreaks[seq(1, length(xbreaks), 15)]

  ggplot(d1, aes(x=chill_season, y=get(colname), 
                 fill=fil, group=time_period)) +
  labs(x = "chill year", y = "day of year") + #, fill = "Climate Group"
  # guides(fill=guide_legend(title="Time period")) + 
  facet_wrap(. ~ location) + # scales = "free"
  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0) }, 
                              fun.ymax=function(z) { quantile(z,1) }, 
               alpha=0.2) +

  stat_summary(geom="ribbon", fun.y=function(z) {quantile(z,0.5) }, 
               fun.ymin=function(z) { quantile(z,0.1) }, 
               fun.ymax=function(z) { quantile(z,0.9) }, alpha=0.4) +

  stat_summary(geom="ribbon", fun.y=function(z) { quantile(z,0.5) }, 
                              fun.ymin=function(z) { quantile(z,0.25) }, 
                              fun.ymax=function(z) { quantile(z,0.75) }, 
               alpha=0.8) +

  stat_summary(geom="line", fun.y=function(z) {quantile(z,0.5) }, 
               size = 1) + 
  # scale_x_continuous(breaks=seq(1970, 2100, 10)) +
  scale_x_continuous(breaks = xbreaks) +
  scale_y_continuous(breaks = chill_doy_map$day_count_since_sept, 
                     labels = chill_doy_map$letter_day) +
  scale_color_manual(values = cls) +
  scale_fill_manual(values = cls) +
  theme(panel.grid.major = element_line(size=0.2),
        panel.spacing=unit(.5, "cm"),
        legend.text=element_text(size=18, face="bold"),
        legend.title = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face="bold", size=16, color="black"),
        axis.text = element_text(size=16, color="black"), # face="bold",
        # axis.text.x = element_text(angle=20, hjust = 1),
        axis.ticks = element_line(color = "black", size = .2),
        axis.title.x = element_text(size=18,  face="bold", 
                                    margin=margin(t=10, r=0, b=0, l=0)),
        axis.title.y = element_text(size=18, face="bold",
                                    margin=margin(t=0, r=10, b=0, l=0)),
        plot.title = element_text(lineheight=.8, face="bold", size=20)
        ) 
}