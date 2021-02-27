
library(data.table)
library(ggplot2)
library(dplyr)

in_dir <- "/Users/hn/Documents/GitHub/Kirti/wareHouse/bee/"
plot_dt <- read.csv(paste0(in_dir, "PanelPlot.csv"), 
                    header = TRUE, as.is=T) %>% data.table()

plot_dt$ColonCol.Size <- as.numeric(plot_dt$ColonCol.Size)
plot_dt$y..AdultDrones <- as.numeric(plot_dt$y..AdultDrones)
plot_dt$AdultWkr <- as.numeric(plot_dt$AdultWkr )
plot_dt$Forgr <- as.numeric(plot_dt$Forgr)
plot_dt$Drones.Brood <- as.numeric(plot_dt$Drones.Brood)
plot_dt$WkrBrood <- as.numeric(plot_dt$WkrBrood)
plot_dt$DronLarv <- as.numeric(plot_dt$DronLarv)
plot_dt$e..WkrLarv <- as.numeric(plot_dt$e..WkrLarv)
plot_dt$DroneEggs <- as.numeric(plot_dt$DroneEggs)
plot_dt$WkrEggs <- as.numeric(plot_dt$WkrEggs)
plot_dt$ColonyPollen <- as.numeric(plot_dt$ColonyPollen)
plot_dt$ColonyNectar <- as.numeric(plot_dt$ColonyNectar)

plot_dt <- plot_dt[1:1250, ]
plot_dt <- within(plot_dt, remove(Date))
plot_dt$date <- 1
plot_dt$date <- cumsum(plot_dt$date)

melted <- melt(plot_dt, id.vars = 'date', variable.name = 'series')

bee_plot <- ggplot(data = melted, 
                   aes(x=date, y=value)) + 
            geom_line(aes(colour = series)) + 
            facet_wrap(~series, scales="free") + 
            theme(legend.position = "bottom",
            	  legend.title = element_blank())


ggsave(filename = "bee_plot.png",
       plot = bee_plot, 
       width = 7, height = 7, units = "in", 
       dpi=400, device = "png",
       path = in_dir)


