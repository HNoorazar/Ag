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

# %%
# # !pip3 install pymannkendall

# %%
import warnings
warnings.filterwarnings("ignore")
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from datetime import datetime
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

# %%
bpszone_ANPP = pd.read_csv(min_bio_dir + "bpszone_annual_productivity_rpms_MEAN.csv")

bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.sort_values(by=['fid', 'year'], inplace=True)
bpszone_ANPP.head(2)

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
Albers_SF_west = geopandas.read_file(f_name)
Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid
Albers_SF_west.head(2)

# %%
Albers_SF_west.rename(columns={"EW_meridia": "EW_meridian",
                               "p_valueSpe" : "p_valueSpearman",
                               "medians_di": "medians_diff_ANPP",
                               "medians__1" : "medians_diff_slope_ANPP",
                               "median_ANP" : "median_ANPP_change_as_perc",
                               "state_majo" : "state_majority_area"}, 
                      inplace=True)

# %% [markdown]
# ## Greening Yue but not Original

# %%
Albers_SF_west.columns

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
len(greening_original_FIDs)

# %%

# %%

# %% [markdown]
# # Make some plots

# %%
# Albers_SF_west.plot(column='EW_meridian', categorical=True, legend=True);

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
gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')
# gdf = geopandas.read_file(rangeland_bio_base +'cb_2018_us_state_500k')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %% [markdown]
# ### Plot a couple of examples

# %%
bpszone_ANPP_west = bpszone_ANPP.copy()

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
bpszone_ANPP_west = pd.merge(bpszone_ANPP_west, Albers_SF_west[cols_], how="left", on = "fid")
bpszone_ANPP_west.head(2)

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
Albers_SF_west.head(2)

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
Albers_SF_west.columns

# %%
# Update Dec. 3, 2024. Add Yue's new locations to this plot
Albers_SF_west_increase = Albers_SF_west[Albers_SF_west["trend_yue"] == "increasing"]
Albers_SF_west_increase.shape

# %%

# %% [markdown]
# ### Plot positive Spearman's with p-value smaller than 0.05

# %%
print (Albers_SF_west["Spearman"].min())
Albers_SF_west.head(2)

# %%
Albers_SF_west_spearmanP5 = Albers_SF_west[(Albers_SF_west["Spearman"] > 0) & 
                                           (Albers_SF_west["p_Spearman"] < 0.05)]


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
Albers_SF_west_median_diff_increase.head(2)

# %%
max_loc = Albers_SF_west_median_diff_increase["median_ANPP_change_as_perc"].idxmax()
Albers_SF_west_median_diff_increase.loc[max_loc]

# %%
max_percChange_median_fid = Albers_SF_west_median_diff_increase.loc[max_loc]["fid"]

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
