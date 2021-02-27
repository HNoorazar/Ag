library(map)
library(ggplot2)

us_states <- map_data("state")
base <- ggplot(data = us_states,
               mapping = aes(x = long, y = lat,
               group = group))

us_map_plot <- base + geom_polygon(fill="gray", color = "white", size = 0.3) +
               coord_map(projection = "albers", lat0 = 39, lat1 = 45) +
               guides(fill = FALSE) + 
               theme_light() + 
               theme(panel.grid.major = element_blank(),
                     panel.grid.minor = element_blank(),
                     legend.title = element_blank(),
                     axis.text = element_blank(),
                     axis.text.x = element_blank(),
                     axis.ticks = element_blank(),
                     axis.title.x = element_blank(),
                     axis.title.y = element_blank(),
                     plot.title = element_text(lineheight=.8, face="bold"),
                     plot.margin = margin(t=.2, r=0, b=.2, l=0, "cm"),
                     ) +
               xlim(-125, -65) +
               ylim(25, 50) 

ggsave(plot = us_map_plot,
       filename = paste0("us_map_plot_8_by_4.png"), 
       width=8, height=4, units = "in", 
       dpi=600, device = "png",
       path="/Users/hn/Desktop/")


ggsave(plot = us_map_plot,
       filename = paste0("us_map_plot_8_by_4.pdf"), 
       width=8, height=4, units = "in", 
       dpi=600, device = "pdf",
       path="/Users/hn/Desktop/")


ggsave(plot = us_map_plot,
       filename = paste0("us_map_plot_8_by_4.jpeg"), 
       width=8, height=4, units = "in", 
       dpi=600, device = "jpeg",
       path="/Users/hn/Desktop/")

ggplot(data = states) + 
  geom_polygon(aes(x = long, y = lat, fill = region, group = group), color = "white") + 
  coord_fixed(1.3) +
  guides(fill=FALSE)  # do this to leave off the color legend





