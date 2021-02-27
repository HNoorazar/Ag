


emissions <- c("RCP 4.5", "RCP 8.5")
time_periods <- c("Historical", "2040's","2060's","2080's")
time_periods_n <- c("Historical", "2026-2050","2051-2075","2076-2095")


geom_boxplot(outlier.size=-.15, notch=FALSE, width=.5, lwd=.25, 
             aes(fill=ClimateGroup), position=position_dodge(width=0.5))

theme(plot.title = element_text(size = 35, face="bold"),
         legend.position="bottom", 
         legend.margin=margin(t=-.3, r=0, b=.3, l=0, unit = 'cm'),
         legend.title = element_blank(),
         legend.text = element_text(size=35, face="plain"),
         legend.key.size = unit(2, "cm"), 
         legend.spacing.x = unit(0.5, 'cm'),
         panel.grid.major.x = element_blank(),
         panel.grid.minor.x = element_blank(),
         panel.grid.major.y = element_line(size = 0.1),
         panel.grid.minor.y = element_line(size = 0.1),
         strip.text = element_text(size= 30, face = "bold"),
         axis.text.x = element_text(size = 25, face="bold", color="black"),
         axis.text.y = element_text(size = 25, face="bold", color="black"),
         axis.title.y= element_text(size = 35, face = "bold", 
                                    margin = margin(t=0, r=20, b=0, l=0))
        )

scale_fill_manual(values=color_ord,
                 name="Time\nPeriod", 
                 labels = time_periods) + 
scale_color_manual(values = color_ord,
                  name="Time\nPeriod", 
                  limits = color_ord,
                  labels = time_periods) 

plot_dpi <- 350
color_ord <- c("grey47", "dodgerblue", "olivedrab4", "red")






