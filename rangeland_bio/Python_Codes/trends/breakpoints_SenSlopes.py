# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # breakpoints in R.
#
# Lets do the breakpoints in R. Here we can analyze and plot things.

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
import statsmodels.formula.api as smf

import statsmodels.stats.api as sms
import statsmodels.api as sm

from scipy import stats
import scipy.stats as scipy_stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc


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
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

rangeland_bio_base = research_db + "/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

breakpoint_plot_base = bio_plots + "breakpoints/"
os.makedirs(breakpoint_plot_base, exist_ok=True)

breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)


G_breakpoint_TS_dir = breakpoint_TS_dir + "/greening/"
B_breakpoint_TS_dir = breakpoint_TS_dir + "/browning/"
noTrend_breakpoint_TS_dir = breakpoint_TS_dir + "/notrend/"

os.makedirs(G_breakpoint_TS_dir, exist_ok=True)
os.makedirs(B_breakpoint_TS_dir, exist_ok=True)
os.makedirs(noTrend_breakpoint_TS_dir, exist_ok=True)

# %%
breakpoints_dir = rangeland_bio_data + "breakpoints/"

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

# %%

# %%
# import importlib
# importlib.reload(rc)
# importlib.reload(rpc)

# %%
ANPP_breaks = pd.read_csv(breakpoints_dir + "ANPP_break_points.csv")
ANPP_breaks.head(2)

# %%
# %%time
## bad 2012
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %%
print (SF_west.shape)
SF_west = pd.merge(SF_west, ANPP_breaks, on="fid", how="left")
print (SF_west.shape)

# %%

# %%
FIDs = SF_west["fid"]
green_FIDs = list(SF_west[SF_west["trend"] == "increasing"]["fid"])
brown_FIDs = list(SF_west[SF_west["trend"] == "decreasing"]["fid"])
noTrend_FIDs = list(SF_west[SF_west["trend"] == "no trend"]["fid"])
print (f"{len(green_FIDs) = }")
print (f"{len(brown_FIDs) = }")
print (f"{len(noTrend_FIDs) = }")

# %%
SF_west.head(2)

# %%
bp_cols = SF_west['breakpoint_years'].str.split('_', expand=True)
bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
SF_west = pd.concat([SF_west, bp_cols], axis=1)
SF_west.head(5)

# %%

# %%
greening = SF_west[SF_west["trend"] == "increasing"].copy()
browning = SF_west[SF_west["trend"] == "decreasing"].copy()
noTrend = SF_west[SF_west["trend"] == "no trend"].copy()

print (f'{greening["trend"].unique()[0] = }')
print (f'{browning["trend"].unique()[0] = }')
print (f'{noTrend["trend"].unique()[0] = }')

# %%

# %%
fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([]); ax.set_yticks([])
plt.title('greening locations by Yue, missed by original MK', y=0.98)
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan" # laptop
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)

Albers_SF["centroid"] = Albers_SF["geometry"].centroid
Albers_SF.head(2)

# %% [markdown]
# ## Some stats

# %%
df = greening.groupby('breakpoint_count').size().reset_index(name='count')
df

# %%
tick_legend_FontSize = 9
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .6,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 5,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold'}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=sharey_, sharex=False, dpi=dpi_)

df = greening.groupby('breakpoint_count').size().reset_index(name='count')
axes[0].bar(df['breakpoint_count'], df['count'], color='skyblue', edgecolor='black', zorder=3);

axes[0].set_ylabel('count')
axes[0].set_title('greening locations', color="green", fontdict={'family': 'serif', 'weight': 'bold'})
axes[0].set_xticks(df['breakpoint_count']);
############################################################################################################
###
### browning
df = browning.groupby('breakpoint_count').size().reset_index(name='count')
axes[1].bar(df['breakpoint_count'], df['count'], color='skyblue', edgecolor='black', zorder=3);
axes[1].set_title('browning locations', color="brown", fontdict={'family': 'serif', 'weight': 'bold'})
axes[1].set_xticks(df['breakpoint_count']);

############################################################################################################
###
### no trend
df = noTrend.groupby('breakpoint_count').size().reset_index(name='count')
axes[2].bar(df['breakpoint_count'], df['count'], color='skyblue', edgecolor='black', zorder=3);
axes[2].set_title('no-trend locations', fontdict={'family': 'serif', 'weight': 'bold'})
axes[2].set_xticks(df['breakpoint_count']);


for ax in axes:
    ax.tick_params(axis='y', length=0);
    ax.tick_params(axis='x', length=0);
    ax.set_xlabel('number of breakpoints')
    ax.grid(axis='y', alpha=0.7, zorder=0);
        
if sharey_:
    file_name = breakpoint_plot_base + "BP_count_trendBased_identicalScale.pdf"
else:
    file_name = breakpoint_plot_base + "BP_count_trendBased.pdf"

plt.tight_layout()
plt.savefig(file_name, dpi=dpi_)

# %% [markdown]
# ### Histogram of 1 breakpoints based on Year
#
# In the greening locations pick those with at least one breakpoint, and then pick the first year and plot histogram.

# %%

# %%
tick_legend_FontSize = 13
params = {"legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 5,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          "font.family": "Palatino"}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %% [markdown]
# ## Distribution of first breakpoint

# %%
sharex_ = True
sharey_ = False
fig, axes = plt.subplots(3, 1, figsize=(8, 4), sharey=sharey_, sharex=sharex_, dpi=dpi_)
##########################################################################################
#####
##### greening
#####
df = greening[greening["breakpoint_count"]>=1].copy()
df = df.groupby('BP_1').size().reset_index(name='count')
axes[0].bar(df['BP_1'], df['count'], color='skyblue', edgecolor='black', zorder=3);
axes[0].set_title('greening locations', color='green', fontdict={'family': 'serif', 'weight': 'bold'})

##########################################################################################
#####
##### browning
#####
df = browning[browning["breakpoint_count"]>=1].copy()
df = df.groupby('BP_1').size().reset_index(name='count')
axes[1].bar(df['BP_1'], df['count'], color='skyblue', edgecolor='black', zorder=3);
axes[1].set_title('browning locations', color="brown", fontdict={'family': 'serif', 'weight': 'bold'})

##########################################################################################
#####
##### no trend
#####
df = noTrend[noTrend["breakpoint_count"]>=1].copy()
df = df.groupby('BP_1').size().reset_index(name='count')
axes[2].bar(df['BP_1'], df['count'], color='skyblue', edgecolor='black', zorder=3);
axes[2].set_title('no-trend locations', fontdict={'family': 'serif', 'weight': 'bold'})

for ax in axes:
    ax.tick_params(axis='y', length=0);
    ax.tick_params(axis='x', length=0);
    # ax.set_ylabel('count')
    ax.grid(axis='y', alpha=0.7, zorder=0);

fig.text(-.01, 0.55, 'count', va='center', rotation='vertical', fontsize=12)
axes[2].set_xlabel('BP1 year')
axes[2].tick_params(axis='x', rotation=45)


if sharey_:
    file_name = breakpoint_plot_base + "dist_BP1s_trendBased_identicalScale.pdf"
else:
    file_name = breakpoint_plot_base + "dist_BP1s_trendBased.pdf"

plt.tight_layout()
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %% [markdown]
# ## Distribution of 2nd breakpoint

# %%
sharex_ = True
sharey_ = False
fig, axes = plt.subplots(3, 1, figsize=(8, 4), sharey=sharey_, sharex=sharex_, dpi=dpi_)
##########################################################################################
#####
##### greening
#####
df = greening[greening["breakpoint_count"]>=2].copy()
df = df.groupby('BP_2').size().reset_index(name='count')
axes[0].bar(df['BP_2'], df['count'], color='skyblue', edgecolor='black', zorder=3);
axes[0].set_title('greening locations', color='green', fontdict={'family': 'serif', 'weight': 'bold'})

##########################################################################################
#####
##### browning
#####
df = browning[browning["breakpoint_count"]>=2].copy()
df = df.groupby('BP_2').size().reset_index(name='count')
axes[1].bar(df['BP_2'], df['count'], color='skyblue', edgecolor='black', zorder=3);
axes[1].set_title('browning locations', color="brown", fontdict={'family': 'serif', 'weight': 'bold'})

##########################################################################################
#####
##### no trend
#####
df = noTrend[noTrend["breakpoint_count"]>=2].copy()
df = df.groupby('BP_2').size().reset_index(name='count')
axes[2].bar(df['BP_2'], df['count'], color='skyblue', edgecolor='black', zorder=3);
axes[2].set_title('no-trend locations', fontdict={'family': 'serif', 'weight': 'bold'})

for ax in axes:
    ax.tick_params(axis='y', length=0);
    ax.tick_params(axis='x', length=0);
    # ax.set_ylabel('count')
    ax.grid(axis='y', alpha=0.7, zorder=0);

fig.text(-.01, 0.55, 'count', va='center', rotation='vertical', fontsize=12)
axes[2].set_xlabel('BP2 year')
axes[2].tick_params(axis='x', rotation=45)

if sharey_:
    file_name = breakpoint_plot_base + "dist_BP2s_trendBased_identicalScale.pdf"
else:
    file_name = breakpoint_plot_base + "dist_BP2s_trendBased.pdf"

plt.tight_layout()
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %% [markdown]
# ## Maps

# %%
y_var = "mean_lb_per_acr"

# %%
county_fips_dict = pd.read_pickle(common_data + "county_fips.sav")

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
gdf = geopandas.read_file(common_data +'cb_2018_us_state_500k.zip')
# gdf = geopandas.read_file(common_data +'cb_2018_us_state_500k')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%
tick_legend_FontSize = 12
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 10,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05
}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %% [markdown]
# ### Color based on year that 1st breakpoint occurred

# %%
fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([]); ax.set_yticks([])

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_ = custom_cmap_BW)
SF_west_BP1 = SF_west[SF_west["breakpoint_count"]>=1].copy()
cent_plt = SF_west_BP1["centroid"].plot(ax=ax, c=SF_west_BP1['BP_1'], markersize=.1);
plt.tight_layout()

############# color bar
cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
# cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.5)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label(r"BP1 years", labelpad=2)

#############
plt.title('BP1 years', y=0.98);

# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981)
file_name = breakpoint_plot_base + "BP1_years_map.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1)

# %%

# %%
fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([]); ax.set_yticks([])

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_ = custom_cmap_BW)
SF_west_BP1 = SF_west[SF_west["breakpoint_count"]>=1].copy()

min_col_ = SF_west_BP1['BP_1'].min()
max_col_ = SF_west_BP1['BP_1'].max()
norm_col = Normalize(vmin= min_col_, vmax = max_col_);


cent_plt = SF_west_BP1["centroid"].plot(ax=ax, c=SF_west_BP1['BP_1'], markersize=.1, norm=norm_col);
plt.tight_layout()


############# color bar
cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
# cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.5)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label(r"BP1 years", labelpad=2)

#############
plt.title('BP1 years', y=0.98);

# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981)
file_name = breakpoint_plot_base + "BP1_years_map_norm_col.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1)

# %%
tick_legend_FontSize = 10
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 10,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05
}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)


# %%
SF_west_BP1 = SF_west[SF_west["breakpoint_count"]>=1].copy()
SF_west_BP1_green = SF_west_BP1[SF_west_BP1["trend"] == "increasing"].copy()

min_col_ = SF_west_BP1['BP_1'].min()
max_col_ = SF_west_BP1['BP_1'].max()
norm_col = Normalize(vmin= min_col_, vmax = max_col_);

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_,
                         gridspec_kw={"hspace": 0.15, "wspace": -0.11})
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])

#############
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax1, cmap_="Pastel1", col="EW_meridian")
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax2, cmap_="Pastel1", col="EW_meridian")
#############

# p1 = SF_west_BP1.plot(column='BP_1', ax=ax1, cmap=cmap_G, norm=norm_col)
p1 = SF_west_BP1["centroid"].plot(ax=ax1, c=SF_west_BP1['BP_1'], 
                                  markersize=.1, norm=norm_col);
ax1.set_title(r"BP1 years", y=0.98);
#############
p2 = SF_west_BP1_green["centroid"].plot(ax=ax2, c=SF_west_BP1_green['BP_1'], 
                                        markersize=.1, norm=norm_col);
ax2.set_title(r"BP1 years in greening locations", y=0.98);

############# color bar
cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
cbar1=fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical');
# cbar1.set_label(r"BP1 years", labelpad=2)

file_name = breakpoint_plot_base + "BP1_years_all_and_green_map_norm_col.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(p1, p2, cax, cbar1)

# %%

# %%

# %%
