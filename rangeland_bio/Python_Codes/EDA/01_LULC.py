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
# **Mingliang Liu**
#
# **Generated from** ```projects/rap-data-365417/assets/vegetation-cover-v3```
#
#  - MinStatsID is same as FID
#  - Fraction of vegetation cover for each FID.
#
#  - AFG - Annual Forb/Grass
#  - BGR - Bare Ground
#  - LTR - Litter
#  - PFG - Perennial Forb/Grass
#  - SHR - Shrub
#  - TRE - Tree

# %%
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pandas as pd
import numpy as np

import os, os.path, pickle, sys

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
from matplotlib import colormaps
print (list(colormaps)[:4])


# %%
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

best_cmap_ = ListedColormap([(0.9, 0.9, 0.9), 'black'])

fontdict_normal = {'family':'serif', 'weight':'normal'}
fontdict_bold   = {'family':'serif', 'weight':'bold'}
inset_axes_     = [0.1, 0.13, 0.45, 0.03]

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"
rangeland_bio_base = research_db + "RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"
min_bio_dir_v11 = rangeland_bio_data + "Min_Data_v1.1/"

LULC_dir = min_bio_dir + "Rangeland_Landcover/"
rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

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
LULC_dir

# %% [markdown]
# ## Read CSV files

# %%
LULC_file_list = sorted([x for x in os.listdir(LULC_dir) if x.endswith(".csv")])
print (f"{len(LULC_file_list) = }")
LULC_file_list[:4]

# %%
LULC_1986_2023 = pd.read_csv(LULC_dir + "Rangeland_rap_mean_vegcover_allpixels_1986_2023.csv")
LULC_1986_2023.sort_values(["MinStatsID", "year"], inplace=True)
LULC_1986_2023.reset_index(drop=True, inplace=True)

LULC_1986_2023.head(2)

# %%
LULC_1986 = pd.read_csv(LULC_dir + "Rangeland_rap_mean_vegcover_allpixels_1986.csv")

LULC_1986.sort_values("MinStatsID", inplace=True)
LULC_1986.reset_index(drop=True, inplace=True)
LULC_1986.head(2)

# %%
# LULC_1986_2023.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
LULC_1986_2023.rename(columns={"MinStatsID": "fid", 
                               "AFG" : "annual_forb_grass",
                               "BGR" : "bare_ground",
                               "LTR" : "litter",
                               "PFG" : "perennial_forb_grass",
                               "SHR" : "shrub",
                               "TRE" : "tree"}, 
                      inplace=True)
LULC_1986_2023.head(2)

# %%
FIDs_list = sorted(list(LULC_1986_2023['fid'].unique()))

# %%
fid_column = LULC_1986_2023['fid']

LULC_diff = LULC_1986_2023.groupby('fid').diff()
LULC_diff = LULC_diff.dropna(subset=['year'])
LULC_diff['fid'] = fid_column[LULC_diff.index]

LULC_diff['year_diff'] = LULC_1986_2023['year'].astype(str) + '-' + (LULC_1986_2023['year'] - 1).astype(str)
LULC_diff.drop(columns=["year"], inplace=True)

LULC_diff.reset_index(drop=True, inplace=True)
LULC_diff.head(5)

# %%
# cols_to_sum = ['annual_forb_grass', 'bare_ground', 'litter',
#                'perennial_forb_grass', 'shrub', 'tree']

# LULC_diff['total_change'] = LULC_diff[cols_to_sum].sum(axis=1)

# LULC_diff.head(5)

# %%
# after seaborn, things get messed up. Fix them:
matplotlib.rc_file_defaults()
font = {"size": 14}
matplotlib.rc("font", **font)

tick_legend_FontSize = 6
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5, 
          'legend.handlelength': 2,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05}

plt.rcParams.update(params)

tick_legend_FontSize = 10
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1.2,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
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
y_var = "tree"

# %%
a_fid = FIDs_list[0]

fig, axes = plt.subplots(1, 1, figsize=(4, 1), sharey=False, sharex=False, dpi=dpi_)
# fig, axes = plt.subplots(3, 1, figsize=(2, 4.5), sharey=True, sharex=True, dpi=dpi_)
###################################
df = LULC_diff[LULC_diff["fid"] == a_fid]

axes.plot(df['year_diff'], df["tree"], lw=2, c="dodgerblue", zorder=1, label="tree");
# axes.plot(df['year_diff'], df["shrub"], lw=2, c="red", zorder=1, label="{shrub}");
# axes.plot(df['year_diff'], df["annual_forb_grass"], lw=2, c="green", zorder=1, label="{annual grass}");
axes.set_title(f"(FID: {a_fid})",  y=1.15, fontsize=tick_legend_FontSize, pad=-5)

axes.set_ylabel(r'fraction change')
axes.set_xlabel('year') #, fontsize=14
axes.set_xticks(df['year_diff'].iloc[::2]);
axes.tick_params(axis='x', rotation=45)
plt.legend();

# %%
a_fid = FIDs_list[0]

fig, axes = plt.subplots(1, 1, figsize=(4, 1), sharey=False, sharex=False, dpi=dpi_)
# fig, axes = plt.subplots(3, 1, figsize=(2, 4.5), sharey=True, sharex=True, dpi=dpi_)
###################################
df = LULC_1986_2023[LULC_1986_2023["fid"] == a_fid]

axes.plot(df['year'], df["tree"], lw=2, c="dodgerblue", zorder=1, label="tree");
# axes.plot(df['year_diff'], df["shrub"], lw=2, c="red", zorder=1, label="{shrub}");
# axes.plot(df['year_diff'], df["annual_forb_grass"], lw=2, c="green", zorder=1, label="{annual grass}");
axes.set_title(f"(FID: {a_fid})",  y=1.15, fontsize=tick_legend_FontSize, pad=-5)

axes.set_ylabel(r'fraction area')
axes.set_xlabel('year') #, fontsize=14
axes.set_xticks(df['year'].iloc[::2]);
axes.tick_params(axis='x', rotation=45)
plt.legend();

# %%
a_fid = FIDs_list[1]

fig, axes = plt.subplots(1, 1, figsize=(4, 1), sharey=False, sharex=False, dpi=dpi_)
# fig, axes = plt.subplots(3, 1, figsize=(2, 4.5), sharey=True, sharex=True, dpi=dpi_)
###################################
df = LULC_1986_2023[LULC_1986_2023["fid"] == a_fid]

axes.plot(df['year'], df["tree"], lw=2, c="dodgerblue", zorder=1, label="tree");
# axes.plot(df['year_diff'], df["shrub"], lw=2, c="red", zorder=1, label="{shrub}");
# axes.plot(df['year_diff'], df["annual_forb_grass"], lw=2, c="green", zorder=1, label="{annual grass}");
axes.set_title(f"(FID: {a_fid})",  y=1.15, fontsize=tick_legend_FontSize, pad=-5)

axes.set_ylabel(r'fraction area')
axes.set_xlabel('year') #, fontsize=14
axes.set_xticks(df['year'].iloc[::2]);
axes.tick_params(axis='x', rotation=45)
plt.legend();

# %%
LULC_1986_2023.head(2)

# %%
cols = ['annual_forb_grass', 'bare_ground', 'litter', 'perennial_forb_grass', 'tree']

# %%
LULC_1986_2023[cols].sum(axis=1)

# %%
LULC_1986_2023.tail(4)

# %%
