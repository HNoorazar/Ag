
### Seperate box plots by States
bplot_W_CO2 <- ggplot(data = USA_time_WCo2, aes(x=timePeriod, y=FPU), group = location) +
geom_boxplot(outlier.size=-.15, notch=FALSE, width=.2, lwd=.25, aes(fill=timePeriod), position=position_dodge(width=0.5)) + 
facet_wrap(~location, scales="free", ncol=6, dir="v") +
labs(x="", y="FPU") + 
#scale_fill_discrete(labels = c("(1981-2010)", "(2021-2050)", "(2040-2070)")) +
theme_bw() +
theme( legend.position="bottom", 
	   legend.margin=margin(t=-.7, r=0, b=5, l=0, unit = 'cm'),
	   legend.title = element_blank(),
	   legend.text = element_text(size=5, face="plain"),
	   legend.key.size = unit(.3, "cm"), 
	   legend.spacing.x = unit(0.3, 'cm'),
	   panel.grid.major = element_line(size = 0.1),
	   panel.grid.minor = element_line(size = 0.1),
	   strip.text = element_text(size= 7, face = "plain"),
	   axis.text = element_text(face = "plain", size = 4, color="black"),
	   axis.title.x = element_text(face = "plain", size = 10, 
	                               margin = margin(t=10, r=0, b=0, l=0)),
	   axis.text.x = element_blank(), # tick text font size
	   axis.text.y = element_text(size = 6, color="black", angle=90),
	   axis.title.y = element_text(face = "plain", size = 10, 
	                               margin = margin(t=0, r=7, b=0, l=0)),
	   plot.margin = unit(c(t=0.3, r=.7, b=-4.7, l=0.3), "cm")
	  )

input_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/share_website/"
out_name = paste0("bplot_W_CO2.png")
ggsave(out_name, bplot_W_CO2, width=7.5, height=5, unit="in", path=input_dir, dpi=300, device="png")




### Seperate box plots by States NO_CO2
bplot_NO_CO2 <- ggplot(data = USA_time_NoCo2, aes(x=timePeriod, y=FPU), group = location) +
geom_boxplot(outlier.size=-.15, notch=FALSE, width=.2, lwd=.25, aes(fill=timePeriod), position=position_dodge(width=0.5)) + 
facet_wrap(~location, scales="free", ncol=6, dir="v") +
labs(x="", y="FPU") + 
#scale_fill_discrete(labels = c("(1981-2010)", "(2021-2050)", "(2040-2070)")) +
theme_bw() +
theme( legend.position="bottom", 
	   legend.margin=margin(t=-.7, r=0, b=5, l=0, unit = 'cm'),
	   legend.title = element_blank(),
	   legend.text = element_text(size=5, face="plain"),
	   legend.key.size = unit(.3, "cm"), 
	   legend.spacing.x = unit(0.3, 'cm'),
	   panel.grid.major = element_line(size = 0.1),
	   panel.grid.minor = element_line(size = 0.1),
	   strip.text = element_text(size= 7, face = "plain"),
	   axis.text = element_text(face = "plain", size = 4, color="black"),
	   axis.title.x = element_text(face = "plain", size = 10, 
	                               margin = margin(t=10, r=0, b=0, l=0)),
	   axis.text.x = element_blank(), # tick text font size
	   axis.text.y = element_text(size = 6, color="black", angle=90),
	   axis.title.y = element_text(face = "plain", size = 10, 
	                               margin = margin(t=0, r=7, b=0, l=0)),
	   plot.margin = unit(c(t=0.3, r=.7, b=-4.7, l=0.3), "cm")
	  )

input_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/share_website/"
out_name = "bplot_NO_CO2.png"
ggsave(out_name, bplot_NO_CO2, width=7.5, height=5, unit="in", path=input_dir, dpi=300, device="png")



