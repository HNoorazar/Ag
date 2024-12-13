# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# # !pip3 install pymannkendall

# %%
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os, os.path, pickle, sys
import pymannkendall as mk

import statistics
import statsmodels.api as sm

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc


# %%
def plot_SF(SF, ax_, cmap_ = "Pastel1", col="EW_meridian"):
    SF.plot(column=col, ax=ax_, alpha=1, cmap=cmap_, edgecolor='k', legend=False, linewidth=0.1)


# %%
dpi_, map_dpi_=300, 900
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

# %%

# %%
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

yue_plots = bio_plots + "yue/"
os.makedirs(yue_plots, exist_ok=True)

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

# %%
2012 in sorted(ANPP.year.unique())

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan" # laptop
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)

Albers_SF["centroid"] = Albers_SF["geometry"].centroid
Albers_SF.head(2)

# %%
# %%time
## bad 2012
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %%
groupveg = sorted(SF_west["groupveg"].unique())

veg_colors = {"Barren-Rock/Sand/Clay" : "blue",
              "Conifer" : "green",
              "Grassland" : "red",
              "Hardwood" : "cyan",
              "Riparian" : "magenta",
              "Shrubland" : "yellow",
              "Sparse" : "black"}

for a_veg in  groupveg:
    SF_west.loc[SF_west['groupveg'] == a_veg, 'color'] = veg_colors[a_veg]
    Albers_SF.loc[Albers_SF['groupveg'] == a_veg, 'color'] = veg_colors[a_veg]

SF_west.head(2)

# %%
SF_west[["fid", "trend_yue"]].groupby("trend_yue").count()

# %%
sorted(SF_west['state_1'].unique())

# %%
SF_west.rename(columns={"EW_meridia": "EW_meridian",
                        "p_valueSpe" : "p_valueSpearman",
                        "medians_di": "medians_diff_ANPP",
                        "medians__1" : "medians_diff_slope_ANPP",
                        "median_ANP" : "median_ANPP_change_as_perc",
                        "state_majo" : "state_majority_area"}, 
                      inplace=True)

# %% [markdown]
# ## Greening Yue but not Original

# %%
SF_west.columns

# %%
greening_yue_df = ANPP_MK_df[ANPP_MK_df["trend_yue"] == "increasing"].copy()
greening_yue_df = greening_yue_df[["fid", "trend_yue"]]

greening_original_df = ANPP_MK_df[ANPP_MK_df["trend"] == "increasing"].copy()
greening_original_df = greening_original_df[["fid", "trend"]]

# %%
greening_yue_FIDs = list(greening_yue_df["fid"].unique())
greening_original_FIDs = list(greening_original_df["fid"].unique())

# %%
intersection_yue_orig_FIDs = list(set(greening_yue_FIDs).intersection(set(greening_original_FIDs)))
len(intersection_yue_orig_FIDs)

# %%
# s.difference(t)
green_Yue_notoriginal_FIDs = set(greening_yue_FIDs) - set(greening_original_FIDs)
green_original_notYue_FIDs = set(greening_original_FIDs) - set(greening_yue_FIDs)

# %%
print (f"{len(green_original_notYue_FIDs) = }")
print (f"{len(green_Yue_notoriginal_FIDs) = }")

# %%
YueGreen_notOrig_df = ANPP_MK_df[ANPP_MK_df["fid"].isin(list(green_Yue_notoriginal_FIDs))].copy()
OrigGreen_notYue_df = ANPP_MK_df[ANPP_MK_df["fid"].isin(list(green_original_notYue_FIDs))].copy()
Yue_orig_inter_green_df = ANPP_MK_df[ANPP_MK_df["fid"].isin(list(intersection_yue_orig_FIDs))].copy()

# %%

# %%
## Check if our set operations are correct!
union_set = green_original_notYue_FIDs.union(green_Yue_notoriginal_FIDs).union(intersection_yue_orig_FIDs)
len(union_set) == len(YueGreen_notOrig_df) + len(OrigGreen_notYue_df) + len(Yue_orig_inter_green_df)

# %%

# %%

# %% [markdown]
# # Make some plots

# %%

# %%
county_fips_dict = pd.read_pickle(rangeland_reOrganized + "county_fips.sav")

county_fips = county_fips_dict["county_fips"]
full_2_abb = county_fips_dict["full_2_abb"]
abb_2_full_dict = county_fips_dict["abb_2_full_dict"]
abb_full_df = county_fips_dict["abb_full_df"]
filtered_counties_29States = county_fips_dict["filtered_counties_29States"]
SoI = county_fips_dict["SoI"]
state_fips = county_fips_dict["state_fips"]

state_fips = state_fips[state_fips.state != "VI"].copy()
state_fips.head(2)

# %%
from shapely.geometry import Polygon
# gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')
gdf = geopandas.read_file(rangeland_bio_base +'cb_2018_us_state_500k')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%
ANPP.head(2)

# %%
sorted(ANPP.year.unique())

# %%

# %% [markdown]
# ### Plot a couple of examples

# %%
ANPP_west = ANPP.copy()
ANPP_west.head(2)

# %%

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
ANPP_west = pd.merge(ANPP_west, SF_west[cols_], how="left", on = "fid")
ANPP_west.head(2)

# %%
ANPP_west["EW_meridian"].unique()

# %%
mystery_FID_wNoState_in_SF = ANPP_west[ANPP_west["EW_meridian"] != "W"]["fid"].unique()
mystery_FID_wNoState_in_SF

# %%
ANPP_west.head(2)

# %%
SF_west_Yue_notOrig = SF_west[SF_west["fid"].isin(green_Yue_notoriginal_FIDs)].copy()
SF_west_Orig_notYue = SF_west[SF_west["fid"].isin(green_original_notYue_FIDs)].copy()

# %%
from matplotlib.lines import Line2D

# %%
tick_legend_FontSize = 2
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
         'axes.linewidth' : .05}

plt.rcParams.update(params)

# %%
Albers_SF.head(2)

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])
plt.title('Where are FIDs {}?'.format(list(mystery_FID_wNoState_in_SF)), y=.92)

plot_SF(SF=visframe_mainLand, ax_=ax, col="EW_meridian", cmap_ = "Pastel2")

mystery_SF = Albers_SF[Albers_SF["fid"].isin(list(mystery_FID_wNoState_in_SF))]
mystery_SF["geometry"].centroid.plot(ax=ax, markersize=.5)

# plt.rcParams['axes.linewidth'] = .05
plt.tight_layout()
# plt.legend(fontsize=3) # ax.axis('off')
# plt.show();

labels = list(veg_colors.keys())
colors = list(veg_colors.values())
lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='-') for c in colors]
plt.legend(lines, labels, frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.01))

file_name = yue_plots + "Albers_locs_vegType.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
count_per_veg_df = SF_west_Yue_notOrig[["groupveg", "fid"]].groupby("groupveg").count().reset_index()
count_per_veg_df

# %%
veg_colors_for_label = veg_colors.copy()
for a_key in veg_colors.keys():
    count_ = str(count_per_veg_df.loc[count_per_veg_df["groupveg"] == a_key]["fid"].item())
    new_key = a_key + " (" + count_ + ")"
    veg_colors_for_label[new_key] = veg_colors_for_label[a_key]
    del (veg_colors_for_label[a_key])
    
veg_colors_for_label

# %%
tick_legend_FontSize = 4
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1.5,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
#          'axes.linewidth' : .05
         }

plt.rcParams.update(params)

# %%
SF_west_Yue_notOrig["trend_yue"].unique()

# %%
# SF_west.plot(column='EW_meridian', categorical=True, legend=True);
##########################################################################################

fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([]); ax.set_yticks([])
plt.title('greening locations by Yue, missed by original MK', y=0.98)

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_ = "Pastel2")
dots_DF = SF_west_Yue_notOrig.copy()
dots_DF["geometry"].centroid.plot(ax=ax, c=dots_DF['color'], markersize=.1)

# plt.rcParams['axes.linewidth'] = .05
plt.tight_layout()
# plt.legend(fontsize=3) # ax.axis('off')
# plt.show();

labels = list(veg_colors_for_label.keys())
colors = list(veg_colors_for_label.values())
lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]
plt.legend(lines, labels, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02))

file_name = yue_plots + "greenYue_missedOriginal.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
del(dots_DF)

# %%
SF_west_Yue_notOrig.head(2)

# %%
SF_west_Yue_notOrig.describe()

# %%
min_idx = SF_west_Yue_notOrig["sens_slope"].idxmin()
cc = ["sens_slope", "Tau", "Spearman", "medians_diff_ANPP", "median_ANPP_change_as_perc"]
SF_west_Yue_notOrig.loc[min_idx, cc]

# %%
min_idx = SF_west_Yue_notOrig["Tau"].idxmin()
cc = ["sens_slope", "Tau", "Spearman", "medians_diff_ANPP", "median_ANPP_change_as_perc"]
SF_west_Yue_notOrig.loc[min_idx, cc]

# %% [markdown]
#

# %%
SF_west.head(2)

# %%
ANPP_MK_df.head(2)

# %%
ANPP_west.head(2)

# %% [markdown]
# ## Set up ticks for grid lines

# %% [raw]
# # https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# # Major ticks every 20, minor ticks every 5
# major_ticks = np.arange(0, 101, 20)
# minor_ticks = np.arange(0, 101, 5)
#
# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)
#
# # And a corresponding grid
# ax.grid(which='both')
#
# # Or if you want different settings for the grids:
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)
#
# plt.show()
#
#
# [ax2.spines[side].set_visible(False) for side in ax2.spines] # removes frame around the plot

# %%
tick_legend_FontSize = 12
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1.5,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
#          'axes.linewidth' : .05
         }

plt.rcParams.update(params)

# %%
# Major ticks every 5, minor ticks every 1
major_ticks = np.arange(1984, 2024, 5)
minor_ticks = np.arange(1984, 2024, 1)
y_var = "mean_lb_per_acr"

# %%

# %% [markdown]
# ### Plot the FID with minimum and maximum of median_ANPP_change_as_perc

# %%
a_metric = "median_ANPP_change_as_perc"
min_idx = SF_west_Yue_notOrig[a_metric].idxmin()
max_idx = SF_west_Yue_notOrig[a_metric].idxmax()
fid_min = SF_west_Yue_notOrig.loc[min_idx, "fid"]
fid_max = SF_west_Yue_notOrig.loc[max_idx, "fid"]
########################################################################################
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2) = axes

ax1.set_xticks(major_ticks)
ax1.set_xticks(minor_ticks, minor=True)
ax1.grid(which='minor', alpha=0.2, axis="x")
ax1.grid(which='major', alpha=0.5, axis="x")

ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
ax2.grid(which='minor', alpha=0.2, axis="x")
ax2.grid(which='major', alpha=0.5, axis="x")
######
###### subplot 1
######
a_fid = fid_min
df = ANPP_west[ANPP_west["fid"] == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend_yue"].values[0]
a_metric_val = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, a_metric].values[0], 2)
state_ = list(df['state_majority_area'].unique())[0]
ax1.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
ax1.scatter(df.year, df[y_var], color="dodgerblue");
# ax1.legend(loc='best')

text_ = ("Yue trend:   {}\n" + a_metric + ": {}\n{} (FID: {})").format(trend_, a_metric_val, state_, a_fid)

y_txt = df[y_var].max() * .99
ax1.text(1984, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
######
###### subplot 2
######
a_fid = fid_max
df = ANPP_west[ANPP_west["fid"] == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend_yue"].values[0]
a_metric_val = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, a_metric].values[0], 2)
state_ = list(df['state_majority_area'].unique())[0]
ax2.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
ax2.scatter(df.year, df[y_var], color="dodgerblue");

text_ = ("Yue trend:   {}\n" + a_metric + ": {}\n{} (FID: {})").format(trend_, a_metric_val, state_, a_fid)
y_txt = df[y_var].max() * .99
ax2.text(1984, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");

ax1.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
ax2.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
ax2.set_xlabel('year') #, fontsize=14

# plt.subplots_adjust(left=0.9, right=0.92, top=0.92, bottom=0.9)
ax1.set_title("Yue Greening, dismissed by Original", y=1.15, fontsize=14)
plt.suptitle("(extremes of " + a_metric + ")", fontsize=14, y=.95, color="red");
# plt.tight_layout();
# fig.subplots_adjust(top=0.8, bottom=0.08, left=0.082, right=0.981)
file_name = yue_plots + "greenYue_extreme" + a_metric +".pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
del(a_metric, min_idx, max_idx, fid_min, fid_max)

# %%

# %% [markdown]
# ### Plot the FID with minimum Sen's slope

# %%

# %%
a_metric = "sens_slope"
min_idx = SF_west_Yue_notOrig[a_metric].idxmin()
max_idx = SF_west_Yue_notOrig[a_metric].idxmax()
fid_min = SF_west_Yue_notOrig.loc[min_idx, "fid"]
fid_max = SF_west_Yue_notOrig.loc[max_idx, "fid"]
########################################################################################
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2) = axes

ax1.set_xticks(major_ticks)
ax1.set_xticks(minor_ticks, minor=True)
ax1.grid(which='minor', alpha=0.2, axis="x")
ax1.grid(which='major', alpha=0.5, axis="x")

ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
ax2.grid(which='minor', alpha=0.2, axis="x")
ax2.grid(which='major', alpha=0.5, axis="x")
######
###### subplot 1
######
a_fid = fid_min
df = ANPP_west[ANPP_west["fid"] == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend_yue"].values[0]
a_metric_val = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, a_metric].values[0], 2)
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = list(df['state_majority_area'].unique())[0]
ax1.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
ax1.scatter(df.year, df[y_var], color="dodgerblue");
# ax1.legend(loc='best')

text_ = ("Yue trend:   {}\nSen's slope: {}"  + "\nTau: {}" + "\n{} (FID: {})").format(trend_, slope_, Tau_,
                                                                                         state_, a_fid)
y_txt = df[y_var].max() * .99
ax1.text(1984, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
######
###### subplot 2
######
a_fid = fid_max
df = ANPP_west[ANPP_west["fid"] == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend_yue"].values[0]
a_metric_val = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, a_metric].values[0], 2)
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)

state_ = list(df['state_majority_area'].unique())[0]
ax2.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
ax2.scatter(df.year, df[y_var], color="dodgerblue");
# ax2.legend(loc='best')

text_ = ("Yue trend:   {}\nSen's slope: {}"  + "\nTau: {}" + "\n{} (FID: {})").format(trend_, slope_, Tau_,
                                                                                         state_, a_fid)
y_txt = df[y_var].max() * .99
ax2.text(1984, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");

ax1.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
ax2.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
ax2.set_xlabel('year') #, fontsize=14

# plt.subplots_adjust(left=0.9, right=0.92, top=0.92, bottom=0.9)
ax1.set_title("Yue Greening, dismissed by Original", y=1.18, fontsize="14")
plt.suptitle("(extremes of " + a_metric + ")", fontsize=15, y=.95, color="red");
# plt.tight_layout();
# fig.subplots_adjust(top=0.8, bottom=0.08, left=0.082, right=0.981)
file_name = yue_plots + "greenYue_extreme" + a_metric +".pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
del(a_metric, min_idx, max_idx, fid_min, fid_max)

# %%

# %%

# %%
SF_west_Yue_notOrig.columns

# %%
a_metric = "Tau"
min_idx = SF_west_Yue_notOrig[a_metric].idxmin()
max_idx = SF_west_Yue_notOrig[a_metric].idxmax()
fid_min = SF_west_Yue_notOrig.loc[min_idx, "fid"]
fid_max = SF_west_Yue_notOrig.loc[max_idx, "fid"]
########################################################################################
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2) = axes

ax1.set_xticks(major_ticks)
ax1.set_xticks(minor_ticks, minor=True)
ax1.grid(which='minor', alpha=0.2, axis="x")
ax1.grid(which='major', alpha=0.5, axis="x")

ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
ax2.grid(which='minor', alpha=0.2, axis="x")
ax2.grid(which='major', alpha=0.5, axis="x")
######
###### subplot 1
######
a_fid = fid_min
df = ANPP_west[ANPP_west["fid"] == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend_yue"].values[0]
a_metric_val = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, a_metric].values[0], 2)
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)

state_ = list(df['state_majority_area'].unique())[0]
ax1.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
ax1.scatter(df.year, df[y_var], color="dodgerblue");

text_ = ("Yue trend:   {}\nSen's slope: {}"  + "\nTau: {}" + "\n{} (FID: {})").format(trend_, slope_, Tau_,
                                                                                         state_, a_fid)
y_txt = df[y_var].max() * .99
ax1.text(1984, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
######
###### subplot 2
######
a_fid = fid_max
df = ANPP_west[ANPP_west["fid"] == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend_yue"].values[0]
a_metric_val = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, a_metric].values[0], 2)
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)

state_ = list(df['state_majority_area'].unique())[0]
ax2.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
ax2.scatter(df.year, df[y_var], color="dodgerblue");
# ax2.legend(loc='best')

text_ = ("Yue trend:   {}\nSen's slope: {}"  + "\nTau: {}" + "\n{} (FID: {})").format(trend_, slope_, Tau_,
                                                                                      state_, a_fid)
y_txt = df[y_var].max() * .99
ax2.text(1984, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");

ax1.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
ax2.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
ax2.set_xlabel('year') #, fontsize=14

# plt.subplots_adjust(left=0.9, right=0.92, top=0.92, bottom=0.9)
ax1.set_title("Yue Greening, dismissed by Original", fontsize=13, y=1.18)
plt.suptitle("(extremes of " + a_metric + ")", fontsize=15, y=.95, color='red');
# plt.tight_layout();
# fig.subplots_adjust(top=0.8, bottom=0.08, left=0.082, right=0.981)
file_name = yue_plots + "greenYue_extreme" + a_metric +".pdf"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
del(a_metric, min_idx, max_idx, fid_min, fid_max)

# %%

# %%
random.seed(3)
random_idx = random.sample(list(SF_west_Yue_notOrig.index), 2)
min_idx = random_idx[0]
max_idx = random_idx[1]
fid_min = SF_west_Yue_notOrig.loc[min_idx, "fid"]
fid_max = SF_west_Yue_notOrig.loc[max_idx, "fid"]

########################################################################################
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                         gridspec_kw={"hspace": 0.02, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2) = axes

ax1.set_xticks(major_ticks)
ax1.set_xticks(minor_ticks, minor=True)
ax1.grid(which='minor', alpha=0.2, axis="x")
ax1.grid(which='major', alpha=0.5, axis="x")

ax2.set_xticks(major_ticks)
ax2.set_xticks(minor_ticks, minor=True)
ax2.grid(which='minor', alpha=0.2, axis="x")
ax2.grid(which='major', alpha=0.5, axis="x")
######
###### subplot 1
######
a_fid = fid_min
df = ANPP_west[ANPP_west["fid"] == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend_yue"].values[0]
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)

state_ = list(df['state_majority_area'].unique())[0]
ax1.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
ax1.scatter(df.year, df[y_var], color="dodgerblue");
# ax1.legend(loc='best')

text_ = ("Yue trend:   {}\nSen's slope: {}"  + "\nTau: {}" + "\n{} (FID: {})").format(trend_, slope_, Tau_,
                                                                                      state_, a_fid)
y_txt = df[y_var].max() * .99
ax1.text(1984, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
ax1.xaxis.set_ticks_position('none')
######
###### subplot 2
######
a_fid = fid_max
df = ANPP_west[ANPP_west["fid"] == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend_yue"].values[0]
state_ = list(df['state_majority_area'].unique())[0]
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)

ax2.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
ax2.scatter(df.year, df[y_var], color="dodgerblue");

text_ = ("Yue trend:   {}\nSen's slope: {}"  + "\nTau: {}" + "\n{} (FID: {})").format(trend_, slope_, Tau_,
                                                                                         state_, a_fid)
y_txt = df[y_var].max() * .99
ax2.text(1984, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");

ax1.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
ax2.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
ax2.set_xlabel('year') #, fontsize=14

# plt.subplots_adjust(left=0.9, right=0.92, top=0.92, bottom=0.9)
#ax1.set_title("Yue Greening, dismissed by Original", fontsize=13, y=1.18)
plt.suptitle("random FIDs. Green by Yue. Dismissed by original", fontsize=13, y=.95);
# plt.tight_layout();
# fig.subplots_adjust(top=0.8, bottom=0.08, left=0.082, right=0.981)
file_name = yue_plots + "greenYue_ random.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
# drop trend so there is no bug later
print (ANPP_MK_df.shape)
# ANPP_MK_df.drop(columns=["trend"], inplace=True)
print (ANPP_MK_df.shape)

# %%
ANPP_MK_df.columns

# %%
np.sort(ANPP_MK_df["sens_slope"])[:10]

# %%
np.sort(ANPP_MK_df["sens_slope"])[-20:]

# %%

# %%
np.sort(ANPP_MK_df.loc[ANPP_MK_df["trend_yue"] == "increasing", "sens_slope"])[:10]

# %%
np.sort(ANPP_MK_df.loc[ANPP_MK_df["trend_yue"] == "increasing", "sens_slope"])[-10:]

# %% [markdown]
# ### Plot everything and color based on slope

# %%

# %%
tick_legend_FontSize = 15
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True}

plt.rcParams.update(params)

# %%
SF_west.head(2)

# %% [markdown]
# In order to have the center at ```yellow``` we manipulated ```vmin``` and ```vmax```.
# Another way is [TwoSlopeNorm](https://matplotlib.org/stable/users/explain/colors/colormapnorms.html). Not pretty.
#
# Or from AI?
# ```norm = colors.MidpointNormalize(midpoint=midpoint, vmin=data.min(), vmax=data.max())```?

# %% [markdown]
# ### Plot increasing trends and color based on slope
#
# The labels seem tobe based on p-values. increasing means **```p-value < 0.05```**.

# %%
print (ANPP_MK_df[ANPP_MK_df["trend_yue"] == "increasing"]["p"].max())
print (ANPP_MK_df[ANPP_MK_df["trend_yue"] == "increasing"]["p"].min())

# %%
SF_west.columns

# %%
# Update Dec. 3, 2024. Add Yue's new locations to this plot
SF_west_increase = SF_west[SF_west["trend_yue"] == "increasing"]
SF_west_increase.shape

# %%

# %% [markdown]
# ### Plot positive Spearman's with p-value smaller than 0.05

# %%
print (SF_west["Spearman"].min())
SF_west.head(2)

# %%
SF_west_spearmanP5 = SF_west[(SF_west["Spearman"] > 0) & (SF_west["p_Spearman"] < 0.05)]


# %%

# %% [markdown]
# ## Side by side

# %%
tick_legend_FontSize = 15
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True}

plt.rcParams.update(params)

# %%

# %% [markdown]
# # Investigate large change in median diff

# %%
SF_west_median_diff_increase.head(2)

# %%
max_loc = SF_west_median_diff_increase["median_ANPP_change_as_perc"].idxmax()
SF_west_median_diff_increase.loc[max_loc]

# %%
max_percChange_median_fid = SF_west_median_diff_increase.loc[max_loc]["fid"]

# %%
tick_legend_FontSize = 15
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True}

plt.rcParams.update(params)

# %% [markdown]
# # Same plot as above. Just pick the ones with low p-value

# %%

# %%
NPP_TS = pd.read_csv(rangeland_bio_base + ".csv.csv")
NPP_TS.head(2)

# %%
NPP_TS_21519 = NPP_TS[NPP_TS.FID==21519].copy()

# %%
