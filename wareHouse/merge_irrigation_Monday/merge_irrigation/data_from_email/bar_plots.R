data_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/"
input_name = "compiled_water_supply.rds"
data <- readRDS(paste0(data_dir, input_name))

################################################################# supply
df <- data.frame(data)
df <- (df %>% group_by(district, model))
medians <- (df %>% summarise(med = median(supply)))
# medians_vec <- medians$med

p <- ggplot(data=medians, aes(x=district, y=med)) +
     geom_bar(aes(fill = model), stat="identity", position="dodge") + 
     labs(x="District", y="Supply median (MGD)") +
     theme_bw() + 
     theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
           legend.position="bottom",
           legend.title=element_blank(),
           legend.text=element_text(size=10),
           legend.key.size = unit(.4, "cm"))
    
plot_path = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/"
ggsave(filename=paste0("supply", ".png"), 
	   plot=p, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=500, 
	   device="png")

p_log <- ggplot(data=medians, aes(x=district, y=med)) +
         geom_bar(aes(fill = model), stat="identity", position="dodge") + 
         labs(x="District", y="Supply median (MGD)") +
	     theme_bw() + 
	     theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
	           legend.position="bottom",
	           legend.title=element_blank(),
	           legend.text=element_text(size=10),
	           legend.key.size = unit(.4, "cm"))+ 
		     scale_y_log10()

ggsave(filename=paste0("supply_log", ".png"), 
	   plot=p_log, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=500, 
	   device="png")


################################################################# Available
df <- data.frame(data)
df <- (df %>% group_by(district, model))
medians <- (df %>% summarise(med = median(available)))
# medians_vec <- medians$med

p <- ggplot(data=medians, aes(x=district, y=med)) +
     geom_bar(aes(fill = model), stat="identity", position="dodge") + 
     labs(x="District", y="Available median (MGD)") +
     theme_bw() + 
     theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
           legend.position="bottom",
           legend.title=element_blank(),
           legend.text=element_text(size=10),
           legend.key.size = unit(.4, "cm"))
    
plot_path = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/"
ggsave(filename=paste0("available", ".png"), 
	   plot=p, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=300, 
	   device="png")

p_log <- ggplot(data=medians, aes(x=district, y=med)) +
         geom_bar(aes(fill = model), stat="identity", position="dodge") + 
         labs(x="District", y="Available median (MGD)") +
	     theme_bw() + 
	     theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
	           legend.position="bottom",
	           legend.title=element_blank(),
	           legend.text=element_text(size=10),
	           legend.key.size = unit(.4, "cm"))+ 
		     scale_y_log10()

ggsave(filename=paste0("available_log", ".png"), 
	   plot=p_log, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=300, 
	   device="png")