# I changed the Python code to include only common counties in pairwise years
# so that the $ change in national share adds up to zero.
# New Python notebook is called "inventory_diff_4_MapinR_CommonCountiesInAllYears.ipynb" and
# old one is called "inventory_diff_4_MapinR.ipynb"

rm(list=ls())
library(usmap)
library(ggplot2)
library(colorspace) # for scale_fill_continuous_diverging()
# library(readspss) # for reading pickles .sav
# library(haven) # for reading pickles .sav
library(data.table)
library(dplyr)
library(stringr)

base_dir <- "/Users/hn/Documents/01_research_data/RangeLand/Data/"
diff_dir <- paste0(base_dir, "data_4_plot/")
plot_dir <- paste0(base_dir, "plots/")


SoI_abb <- c('AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 
             'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
             'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 
             'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 
             'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 
             'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 
             'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 
             'DC')

inventory_map <- function(data_, col_, theme_, legend_d_, title_){
  states <- plot_usmap("states", color = "red", fill = alpha(0.01))
  m <- usmap::plot_usmap(data = data_, values = col_,
                         labels = TRUE, label_color = "orange") + 
       geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                   color = "yellow", fill = alpha(0.01), linewidth = 0.5) + 
       scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                            space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                            name=legend_d_) +
       scale_colour_continuous_diverging() + 
       theme_ + 
       ggtitle(title_)
  m$layers[[2]]$aes_params$size <- 8
  return (m)
}

states <- plot_usmap("states", color = "red", fill = alpha(0.01))
the_theme <-  theme(legend.title = element_text(size=25, face="bold"),
                    legend.text = element_text(size=25, face="plain"),
                    legend.key.size = unit(1, 'cm'), #change legend key size
                    legend.key.height = unit(1, 'cm'), #change legend key height
                    legend.key.width = unit(1, 'cm'), #change legend key width
                    axis.text.x = element_blank(),
                    axis.text.y = element_blank(),
                    axis.ticks.x = element_blank(),
                    axis.ticks.y = element_blank(),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    plot.title = element_text(size=25, lineheight=2, face="bold"))

###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
###########
###########        All years. 
###########        Panel is not gonna look good! 
###########        So, one at a time
###########
inventory_AbsChange <- read.csv(paste0(diff_dir, "inventory_AbsChange_4panel_commonCounties.csv"))
inv_PercChangeShare <- read.csv(paste0(diff_dir, "inventory_ShareChange_4panel_commonCounties.csv"))
setnames(inventory_AbsChange, old = c('county_fips'), new = c('fips'))
setnames(inv_PercChangeShare, old = c('county_fips'), new = c('fips'))

Abs_plot_dir <- paste0(plot_dir, "abs_change_map/")
if (dir.exists(Abs_plot_dir) == F) {dir.create(path = Abs_plot_dir, recursive = T)}
share_plot_dir <- paste0(plot_dir, "share_change_map/")
if (dir.exists(share_plot_dir) == F) {dir.create(path = share_plot_dir, recursive = T)}

for (a_col in colnames(inventory_AbsChange)[4:length(colnames(inventory_AbsChange))]){
  if (stringr::str_detect(a_col, pat="change")){
    s_year = substr(a_col, start = 11, stop = 14)
    e_year = substr(a_col, start = 17, stop = 20)
    if (stringr::str_detect(a_col, pat="asPerc")){
       legend_d <- paste0("Percentage change ", s_year, "_", e_year)
      } else {
       legend_d <- paste0("Absolute change ", s_year, "_", e_year)
      }
   } else {
       yr = tail(strsplit(a_col, split = "_")[[1]], 1)
       legend_d <- paste0("Inventory ", yr)

   }

  inventory_AbsChange_copy <- data.table::copy(inventory_AbsChange)
  inventory_AbsChange_copy <- subset(inventory_AbsChange_copy, select = c("fips", a_col))
  inventory_AbsChange_copy <- inventory_AbsChange_copy %>% tidyr::drop_na()
  
  map <- inventory_map(data_=inventory_AbsChange_copy, col_=a_col, 
                       theme_=the_theme, legend_d_=legend_d, title_ =legend_d)

  ggsave(paste0(gsub("\ ", "_", legend_d), ".pdf"), map, 
         path=Abs_plot_dir, device="pdf",
         dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
}

##### Changes as in National Share

for (a_col in colnames(inv_PercChangeShare)[4:length(colnames(inv_PercChangeShare))]){
  if (str_detect(a_col, pat="change")){
    s_year = unlist(strsplit(a_col, split = "_"))[2]
    e_year = unlist(strsplit(a_col, split = "_"))[3]
    legend_d <- paste0("Change of share ", s_year, "_", e_year)
   } else {
       yr = unlist(strsplit(a_col, split = "_"))[2]
       legend_d <- paste0("Inventory ", yr, " as percentage share")
   }

  inv_PercChangeShare_copy <- copy(inv_PercChangeShare)
  inv_PercChangeShare_copy <- subset(inv_PercChangeShare_copy, select = c("fips", a_col))
  inv_PercChangeShare_copy <- inv_PercChangeShare_copy %>% tidyr::drop_na()
  
  map <- inventory_map(data_=inv_PercChangeShare_copy, col_=a_col, 
                       theme_=the_theme, legend_d_=legend_d, title_ =legend_d)

  ggsave(paste0(gsub("\ ", "_", legend_d), ".pdf"), map, 
         path=share_plot_dir, device="pdf",
         dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
}

######## Remove max and min in the hope that there is only one outlier

Abs_plot_dir <- paste0(plot_dir, "abs_change_map_NoOutlier/")
if (dir.exists(Abs_plot_dir) == F) {dir.create(path = Abs_plot_dir, recursive = T)}
share_plot_dir <- paste0(plot_dir, "share_change_map_NoOutlier/")
if (dir.exists(share_plot_dir) == F) {dir.create(path = share_plot_dir, recursive = T)}

for (a_col in colnames(inventory_AbsChange)[4:length(colnames(inventory_AbsChange))]){
  if (str_detect(a_col, pat="change")){
    s_year = substr(a_col, start = 11, stop = 14)
    e_year = substr(a_col, start = 17, stop = 20)
    if (str_detect(a_col, pat="asPerc")){
       legend_d <- paste0("Percentage change ", s_year, "_", e_year)
      } else {
       legend_d <- paste0("Absolute change ", s_year, "_", e_year)
      }
   } else {
       yr = tail(strsplit(a_col, split = "_")[[1]], 1)
       legend_d <- paste0("Inventory ", yr)

   }

  inventory_AbsChange_copy = copy(inventory_AbsChange)
  inventory_AbsChange_copy <- subset(inventory_AbsChange_copy, select = c("fips", a_col))
  inventory_AbsChange_copy <- inventory_AbsChange_copy %>% tidyr::drop_na()

  minn = min(inventory_AbsChange_copy[, 2])
  maxx = max(inventory_AbsChange_copy[, 2])
  inventory_AbsChange_copy <- inventory_AbsChange_copy[inventory_AbsChange_copy[a_col] > minn ,]
  inventory_AbsChange_copy <- inventory_AbsChange_copy[inventory_AbsChange_copy[a_col] < maxx ,]
  
  map <- inventory_map(data_=inventory_AbsChange_copy, col_=a_col, 
                       theme_=the_theme, legend_d_=legend_d, title_ =legend_d)

  ggsave(paste0(gsub("\ ", "_", legend_d), "_noOutlier.pdf"), map, 
         path=Abs_plot_dir, device="pdf",
         dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
}

##### Changes as in National Share

for (a_col in colnames(inv_PercChangeShare)[4:length(colnames(inv_PercChangeShare))]){
  if (str_detect(a_col, pat="change")){
    s_year = unlist(strsplit(a_col, split = "_"))[2]
    e_year = unlist(strsplit(a_col, split = "_"))[3]
    legend_d <- paste0("Change of share ", s_year, "_", e_year)
   } else {
       yr = unlist(strsplit(a_col, split = "_"))[2]
       legend_d <- paste0("Inventory ", yr, " as percentage share")
   }

  inv_PercChangeShare_copy = copy(inv_PercChangeShare)
  inv_PercChangeShare_copy <- subset(inv_PercChangeShare_copy, select = c("fips", a_col))
  inv_PercChangeShare_copy <- inv_PercChangeShare_copy %>% tidyr::drop_na()

  minn = min(inv_PercChangeShare_copy[, 2])
  maxx = max(inv_PercChangeShare_copy[, 2])
  inv_PercChangeShare_copy <- inv_PercChangeShare_copy[inv_PercChangeShare_copy[a_col] > minn ,]
  inv_PercChangeShare_copy <- inv_PercChangeShare_copy[inv_PercChangeShare_copy[a_col] < maxx ,]

  map <- inventory_map(data_=inv_PercChangeShare_copy, col_=a_col, 
                       theme_=the_theme, legend_d_=legend_d, title_ =legend_d)

  ggsave(paste0(gsub("\ ", "_", legend_d), "_noOutlier.pdf"), map, 
         path=share_plot_dir, device="pdf",
         dpi=300, width=15, height=12, unit="in", limitsize = FALSE)
}

########################################################################
########################################################################
########################################################################
########
########    See if you can melt and create a big panel or sth!!!
########

the_theme <-  theme(legend.title = element_text(size=25, face="plain"),
                    legend.text = element_text(size=25, face="plain"),
                    legend.key.size = unit(1, 'cm'), #change legend key size
                    legend.key.height = unit(1, 'cm'), #change legend key height
                    legend.key.width = unit(1, 'cm'), #change legend key width
                    axis.text.x = element_blank(),
                    axis.text.y = element_blank(),
                    axis.ticks.x = element_blank(),
                    axis.ticks.y = element_blank(),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    plot.title = element_text(size=25, lineheight=2, face="bold"),
                    strip.text = element_text(face="bold", size=16, color="black"),
                    plot.background = element_rect(fill = "gray"))


inventory_map_melt <- function(data_, theme_, legend_d_, title_, num_facet_cols, leg_pos="right"){
  states <- plot_usmap("states", color = "red", fill = alpha(0.01))
  m <- usmap::plot_usmap(data = na.omit(data_), values = "value",
                         labels = TRUE, label_color = "yellow") + 
       geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                   color = "black", fill = alpha(0.01), linewidth = 0.5) + 
       # facet_grid(~ variable, scales="fixed") +
       facet_wrap(~ variable, scales="fixed", ncol = num_facet_cols) +
       scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                            space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                            name=legend_d_) +
       scale_colour_continuous_diverging() + 
       theme_ + theme(legend.position = leg_pos) + 
       ggtitle(title_)
  m$layers[[2]]$aes_params$size <- 3
  return (m)
}
#####
#####   Read
##### 
inventory_AbsChange <- read.csv(paste0(diff_dir, "inventory_AbsChange_4panel_commonCounties.csv"))
inv_PercChangeShare <- read.csv(paste0(diff_dir, "inventory_ShareChange_4panel_commonCounties.csv"))
setnames(inventory_AbsChange, old = c('county_fips'), new = c('fips'))
setnames(inv_PercChangeShare, old = c('county_fips'), new = c('fips'))


Abs_plot_dir <- paste0(plot_dir, "abs_change_map/")
if (dir.exists(Abs_plot_dir) == F) {dir.create(path = Abs_plot_dir, recursive = T)}
share_plot_dir <- paste0(plot_dir, "share_change_map/")
if (dir.exists(share_plot_dir) == F) {dir.create(path = share_plot_dir, recursive = T)}

colnames(inv_PercChangeShare)

common_col = c("fips")

perc_change_cols = c("change_1997_2017_asPercShare", "change_1997_2012_asPercShare",
                     "change_1997_2007_asPercShare", "change_1997_2002_asPercShare",
                     "change_2002_2017_asPercShare", "change_2002_2012_asPercShare",
                     "change_2002_2007_asPercShare", "change_2007_2017_asPercShare",
                     "change_2007_2012_asPercShare", "change_2012_2017_asPercShare")

invent_cols = c("inv_2017_asPercShare", "inv_2012_asPercShare", 
                "inv_2007_asPercShare", "inv_2002_asPercShare",
                "inv_1997_asPercShare")

perc_change_cols = c(common_col, perc_change_cols)
invent_cols = c(common_col, invent_cols)

inv_PercChangeShare_df = inv_PercChangeShare[perc_change_cols]
invShare_df = inv_PercChangeShare[invent_cols]

invShare_df_melt <- melt(data.table(invShare_df), id = common_col)
inv_PercChangeShare_df_melt <- melt(data.table(inv_PercChangeShare_df), id = common_col)

invShare_df_melt <- invShare_df_melt %>% tidyr::drop_na()
inv_PercChangeShare_df_melt <- inv_PercChangeShare_df_melt %>% tidyr::drop_na()

########################################
###
###     invetory national share
###
legend_d = "inventory (national %)"
title_ = "inventory as national share"
map <- inventory_map_melt(data_=invShare_df_melt, theme_=the_theme, 
                          legend_d_= legend_d, title_ = title_, num_facet_cols=2,
                          leg_pos=c(0.6, 0.05))
inventory_panel_dir <- paste0(plot_dir, "inventory_panel/")
if (dir.exists(inventory_panel_dir) == F) {dir.create(path = inventory_panel_dir, recursive = T)}

ggsave(paste0(gsub("\ ", "_", title_), "_panel.pdf"), map, 
         path=inventory_panel_dir, device="pdf",
         dpi=300, width=10, height=12, unit="in", limitsize = FALSE)

########################################
###
###     invetory national share change
###
legend_d = "change (national %)"
title_ = "inventory change as national share"
map <- inventory_map_melt(data_=inv_PercChangeShare_df_melt, theme_=the_theme, 
                          legend_d_= legend_d, title_ = title_, num_facet_cols=3,
                          leg_pos=c(0.7, 0.05))

inventory_panel_dir <- paste0(plot_dir, "inventory_panel/")
if (dir.exists(inventory_panel_dir) == F) {dir.create(path = inventory_panel_dir, recursive = T)}

ggsave(paste0(gsub("\ ", "_", title_), "_panel.pdf"), map, 
         path=inventory_panel_dir, device="pdf",
         dpi=300, width=18, height=18, unit="in", limitsize = FALSE)

#########################################################################
###
###     absolute values panel
###
#########################################################################
rm(list=ls())
library(usmap)
library(ggplot2)
library(colorspace) # for scale_fill_continuous_diverging()
library(data.table)
library(dplyr)
library(stringr)

base_dir <- "/Users/hn/Documents/01_research_data/RangeLand/Data/"
diff_dir <- paste0(base_dir, "data_4_plot/")
plot_dir <- paste0(base_dir, "plots/")

inventory_panel_dir <- paste0(plot_dir, "inventory_panel/")
if (dir.exists(inventory_panel_dir) == F) {dir.create(path = inventory_panel_dir, recursive = T)}


################################################################################
SoI_abb <- c('AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 
             'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
             'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 
             'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 
             'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 
             'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 
             'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 
             'DC')
the_theme <-  theme(legend.title = element_text(size=25, face="plain"),
                    legend.text = element_text(size=25, face="plain"),
                    legend.key.size = unit(1, 'cm'), #change legend key size
                    legend.key.height = unit(1, 'cm'), #change legend key height
                    legend.key.width = unit(1, 'cm'), #change legend key width
                    axis.text.x = element_blank(),
                    axis.text.y = element_blank(),
                    axis.ticks.x = element_blank(),
                    axis.ticks.y = element_blank(),
                    axis.title.x = element_blank(),
                    axis.title.y = element_blank(),
                    plot.title = element_text(size=25, lineheight=2, face="bold"),
                    strip.text = element_text(face="bold", size=16, color="black"),
                    plot.background = element_rect(fill = "gray"))

inventory_map_melt <- function(data_, theme_, legend_d_, title_, num_facet_cols, leg_pos="right"){
  states <- plot_usmap("states", color = "red", fill = alpha(0.01))
  m <- usmap::plot_usmap(data = na.omit(data_), values = "value",
                         labels = TRUE, label_color = "yellow") + 
       geom_polygon(data=states[[1]], aes(x=x, y=y, group=group), 
                   color = "black", fill = alpha(0.01), linewidth = 0.5) + 
       # facet_grid(~ variable, scales="fixed") +
       facet_wrap(~ variable, scales="fixed", ncol = num_facet_cols) +
       scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                            space = "Lab", na.value = "grey50", guide = "colourbar", aesthetics = "fill",
                            name=legend_d_) +
       scale_colour_continuous_diverging() + 
       theme_ + theme(legend.position = leg_pos) + 
       ggtitle(title_)
  m$layers[[2]]$aes_params$size <- 3
  return (m)
}
#####
#####   Read
##### 
inventory_AbsChange <- read.csv(paste0(diff_dir, "inventory_AbsChange_4panel_commonCounties.csv"))
inv_PercChangeShare <- read.csv(paste0(diff_dir, "inventory_ShareChange_4panel_commonCounties.csv"))
setnames(inventory_AbsChange, old = c('county_fips'), new = c('fips'))
setnames(inv_PercChangeShare, old = c('county_fips'), new = c('fips'))

##################################
###
###     invetory panel
###

common_col = c("fips")
inv_change_cols = c("inv_change1997to2017", "inv_change1997to2012",                      
                    "inv_change1997to2007", "inv_change1997to2002",                      
                    "inv_change2002to2017", "inv_change2002to2012",
                    "inv_change2002to2007", "inv_change2007to2017",
                    "inv_change2007to2012", "inv_change2012to2017")

inv_change_perc_cols_ <- c("inv_change1997to2017_asPerc", "inv_change1997to2012_asPerc",
                           "inv_change1997to2007_asPerc", "inv_change2012to2017_asPerc",
                           "inv_change2007to2012_asPerc", "inv_change1997to2002_asPerc",
                           "inv_change2007to2017_asPerc", "inv_change2002to2007_asPerc",
                           "inv_change2002to2012_asPerc", "inv_change2002to2017_asPerc")

inv_cols <- c("inven_2017", "inven_2012",
              "inven_2007", "inven_2002",
              "inven_1997")

inv_change_cols <- c(common_col, inv_change_cols)
inv_change_perc_cols_ <- c(common_col, inv_change_perc_cols_)
inv_cols <- c(common_col, inv_cols)

inv_df                   = inventory_AbsChange[inv_cols]
inventory_Change_df      = inventory_AbsChange[inv_change_cols]
inventory_Change_perc_df = inventory_AbsChange[inv_change_perc_cols_]


inv_df_melt <- melt(data.table(inv_df), id = common_col)
inventory_Change_df_melt <- melt(data.table(inventory_Change_df), id = common_col)
inventory_Change_perc_df_melt <- melt(data.table(inventory_Change_perc_df), id = common_col)

inv_df_melt                   <- inv_df_melt %>% tidyr::drop_na()
inventory_Change_df_melt      <- inventory_Change_df_melt %>% tidyr::drop_na()
inventory_Change_perc_df_melt <- inventory_Change_perc_df_melt %>% tidyr::drop_na()

##### Remove a couple of outliers

minn = min(inv_df_melt[, "value"])
maxx = max(inv_df_melt[, "value"])
inv_df_melt <- inv_df_melt[inv_df_melt$value > minn,]
inv_df_melt <- inv_df_melt[inv_df_melt$value < maxx,]
########################################
###
###     invetory 
###
legend_d = "inventory (absolute)"
title_ = "inventory"
map <- inventory_map_melt(data_=inv_df_melt, theme_=the_theme, 
                          legend_d_= legend_d, title_ = title_, num_facet_cols=2, leg_pos=c(.6, .05))
ggsave(paste0(gsub("\ ", "_", title_), "_panel.pdf"), map, 
       path=inventory_panel_dir, device="pdf",
       dpi=300, width=11, height=12, unit="in", limitsize = FALSE)

########################################
###
###     invetory less than 120K
###
inv_df_melt <- inv_df_melt[inv_df_melt$value > minn,]
inv_df_melt <- inv_df_melt[inv_df_melt$value < 120000,]

legend_d = "inventory (absolute)"
title_ = "inventory less than 120K"
map <- inventory_map_melt(data_=inv_df_melt, theme_=the_theme, 
                          legend_d_= legend_d, title_ = title_, num_facet_cols=2, leg_pos=c(0.6, 0.05))
ggsave(paste0(gsub("\ ", "_", title_), "_panel.pdf"), map, 
       path=inventory_panel_dir, device="pdf",
       dpi=300, width=12, height=12, unit="in", limitsize = FALSE)

########################################
###
###     invetory change in absolute counts
###
inventory_panel_dir <- paste0(plot_dir, "inventory_panel/")
if (dir.exists(inventory_panel_dir) == F) {dir.create(path = inventory_panel_dir, recursive = T)}


legend_d = "inventory change (abs.)"
title_ = "inventory change"
map <- inventory_map_melt(data_=inventory_Change_df_melt, theme_=the_theme, 
                          legend_d_= legend_d, title_ = title_, num_facet_cols=4, leg_pos=c(0.8, 0.1))
ggsave(paste0(gsub("\ ", "_", title_), "_panel.pdf"), map, 
       path=inventory_panel_dir, device="pdf",
       dpi=300, width=20, height=15, unit="in", limitsize = FALSE)


########################################
###
###     invetory change in percentage
###
legend_d = "inventory change (%)"
title_ = "inventory change percentage"

minn = min(inventory_Change_perc_df_melt[, "value"])
maxx = max(inventory_Change_perc_df_melt[, "value"])
inventory_Change_perc_df_melt <- inventory_Change_perc_df_melt[inventory_Change_perc_df_melt$value > minn,]
inventory_Change_perc_df_melt <- inventory_Change_perc_df_melt[inventory_Change_perc_df_melt$value < maxx,]

map <- inventory_map_melt(data_=inventory_Change_perc_df_melt, theme_=the_theme, 
                          legend_d_= legend_d, title_ = title_, num_facet_cols=4, leg_pos=c(0.8, 0.1))
ggsave(paste0(gsub("\ ", "_", title_), "_panel.pdf"), map, 
       path=inventory_panel_dir, device="pdf",
       dpi=300, width=20, height=15, unit="in", limitsize = FALSE)


