################################################################################################
##############
##############              Bar plots of difference of medians for available
##############
################################################################################################
data_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/"
plot_path = data_dir
input_name = "water_supply_No_Arizona_Minnesota.rds"
data <- readRDS(paste0(data_dir, input_name))


############### available
###############
# data = within(data, remove(state, ENVFLW, year))
df <- data.frame(data)
df <- (df %>% group_by(district, model))
medians <- (df %>% summarise(med = median(available)))
# medians_vec <- medians$med
# medians = na.omit(medians)
# compute difference of medians
diff_df <- data.frame(matrix(ncol = 5, nrow = dim(medians)[1]/3))
colnames(diff_df) <- c("district", "F1_Base", "F2_Base", "relative_F1_B", "relative_F2_B")
row_counter = 1
for (ii in seq(1, dim(medians)[1], 3)){
	curr = medians[ii:(ii+2), ]
	ds = curr[1, "district"][[1]]
	diff_df[row_counter, "district"] = as.numeric(levels(ds)[ds])
	diff_df[row_counter, "F1_Base"]  = curr[curr$model == "Future1", ]$med - curr[curr$model == "Baseline", ]$med
	diff_df[row_counter, "F2_Base"]  = curr[curr$model == "Future2", ]$med - curr[curr$model == "Baseline", ]$med
    
    # relative
	diff_df[row_counter, "relative_F1_B"]  = (curr[curr$model == "Future1", ]$med - curr[curr$model == "Baseline", ]$med) * (100/curr[curr$model == "Baseline", ]$med)
	diff_df[row_counter, "relative_F2_B"]  = (curr[curr$model == "Future2", ]$med - curr[curr$model == "Baseline", ]$med) * (100/curr[curr$model == "Baseline", ]$med)

	row_counter = row_counter + 1
}

diff_df = na.omit(diff_df)
diff_df$district = as.factor(diff_df$district)

######
###### difference plot
######
diff_abs = within(diff_df, remove(relative_F1_B, relative_F2_B))
diff_df_melt = melt(diff_abs, id=c("district"))

p <- ggplot(data=diff_df_melt, aes(x=district, y=value)) +
     geom_bar(aes(fill =variable ), stat="identity", position="dodge") + 
     labs(x="District", y="Available median differences (MGD)") +
     theme_bw() + 
     scale_fill_discrete(labels = c("Future1 - Base", "Future2 - Base")) + 
     theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
           legend.position="bottom",
           legend.title=element_blank(),
           legend.text=element_text(size=10),
           legend.key.size = unit(.4, "cm"))
  
ggsave(filename="available_differences.png", 
	   plot=p, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=300, 
	   device="png")
######
###### relative difference plot
######

diff_rel = within(diff_df, remove(F1_Base, F2_Base))
diff_df_melt = melt(diff_rel, id=c("district"))
p <- ggplot(data=diff_df_melt, aes(x=district, y=value)) +
     geom_bar(aes(fill =variable ), stat="identity", position="dodge") + 
     labs(x="District", y="Available median differences (MGD)") +
     theme_bw() + 
     theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
           legend.position="bottom",
           legend.title=element_blank(),
           legend.text=element_text(size=10),
           legend.key.size = unit(.4, "cm"))

ggsave(filename="available_rel_diff.png", 
	   plot=p, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=300, 
	   device="png")



################################################################################################
###################
################### Bar plots of difference of medians for supply
###################
################################################################################################
data_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/"
plot_path = data_dir
input_name = "water_supply_No_Arizona_Minnesota.rds"
data <- readRDS(paste0(data_dir, input_name))


############### supply
###############
# data = within(data, remove(state, ENVFLW, year))
df <- data.frame(data)
df <- (df %>% group_by(district, model))
medians <- (df %>% summarise(med = median(supply)))
# medians_vec <- medians$med
# medians = na.omit(medians)
# compute difference of medians
diff_df <- data.frame(matrix(ncol = 5, nrow = dim(medians)[1]/3))
colnames(diff_df) <- c("district", "F1_Base", "F2_Base", "relative_F1_B", "relative_F2_B")
row_counter = 1
for (ii in seq(1, dim(medians)[1], 3)){
	curr = medians[ii:(ii+2), ]
	ds = curr[1, "district"][[1]]
	diff_df[row_counter, "district"] = as.numeric(levels(ds)[ds])
	diff_df[row_counter, "F1_Base"]  = curr[curr$model == "Future1", ]$med - curr[curr$model == "Baseline", ]$med
	diff_df[row_counter, "F2_Base"]  = curr[curr$model == "Future2", ]$med - curr[curr$model == "Baseline", ]$med
    
    # relative
	diff_df[row_counter, "relative_F1_B"]  = (curr[curr$model == "Future1", ]$med - curr[curr$model == "Baseline", ]$med) * (100/curr[curr$model == "Baseline", ]$med)
	diff_df[row_counter, "relative_F2_B"]  = (curr[curr$model == "Future2", ]$med - curr[curr$model == "Baseline", ]$med) * (100/curr[curr$model == "Baseline", ]$med)

	row_counter = row_counter + 1
}

diff_df = na.omit(diff_df)
diff_df$district = as.factor(diff_df$district)

######
###### difference plot
######
diff_abs = within(diff_df, remove(relative_F1_B, relative_F2_B))
diff_df_melt = melt(diff_abs, id=c("district"))

p <- ggplot(data=diff_df_melt, aes(x=district, y=value)) +
     geom_bar(aes(fill =variable ), stat="identity", position="dodge") + 
     labs(x="District", y="supply median differences (MGD)") +
     theme_bw() + 
     scale_fill_discrete(labels = c("Future1 - Base", "Future2 - Base")) + 
     theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
           legend.position="bottom",
           legend.title=element_blank(),
           legend.text=element_text(size=10),
           legend.key.size = unit(.4, "cm"))
  
ggsave(filename="supply_differences.png", 
	   plot=p, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=300, 
	   device="png")
######
###### relative difference plot
######

diff_rel = within(diff_df, remove(F1_Base, F2_Base))
diff_df_melt = melt(diff_rel, id=c("district"))
p <- ggplot(data=diff_df_melt, aes(x=district, y=value)) +
     geom_bar(aes(fill =variable ), stat="identity", position="dodge") + 
     labs(x="District", y="supply median differences (MGD)") +
     theme_bw() + 
     theme(axis.text.y = element_text(size = 9, angle=90, color="black"),
           legend.position="bottom",
           legend.title=element_blank(),
           legend.text=element_text(size=10),
           legend.key.size = unit(.4, "cm"))

ggsave(filename="supply_rel_diff.png", 
	   plot=p, 
	   path=plot_path, 
	   width=21,
	   height=5, 
	   dpi=300, 
	   device="png")