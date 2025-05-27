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
import rangeland_core as rpc


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

yue_plots = bio_plots + "yue/"
os.makedirs(yue_plots, exist_ok=True)

breakpoint_plot_base = bio_plots + "breakpoints/"
os.makedirs(breakpoint_plot_base, exist_ok=True)

breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)


G_breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/greening/"
B_breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/browning/"
noTrend_breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/notrend/"

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
FIDs = SF_west["fid"]
green_FIDs = list(SF_west[SF_west["trend"] == "increasing"]["fid"])
brown_FIDs = list(SF_west[SF_west["trend"] == "decreasing"]["fid"])
noTrend_FIDs = list(SF_west[SF_west["trend"] == "no trend"]["fid"])
print (f"{len(green_FIDs) = }")
print (f"{len(brown_FIDs) = }")
print (f"{len(noTrend_FIDs) = }")

# %%

# %%

# %%
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan" # laptop
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)

Albers_SF["centroid"] = Albers_SF["geometry"].centroid
Albers_SF.head(2)

# %% [markdown]
# ### Plot time series of ANPP and the breakpoints

# %%
tick_legend_FontSize = 6
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5, 
          'legend.handlelength': 2}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
y_var = "mean_lb_per_acr"
a_fid = green_FIDs[0]
a_fid

# %%
for a_fid in green_FIDs:
    fig, axes = plt.subplots(1, 1, figsize=(4, 2), sharey=False, sharex=False, dpi=dpi_)
    # fig, axes = plt.subplots(3, 1, figsize=(2, 4.5), sharey=True, sharex=True, dpi=dpi_)
    ###################################
    df = ANPP[ANPP["fid"] == a_fid]
    axes.plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);
    # axes.scatter(df.year, df[y_var], marker='o', facecolors='r', edgecolors='r', s=5, zorder=2);

    break_yrs = ANPP_breaks[ANPP_breaks["fid"] == a_fid]["breakpoint_years"].iloc[0]
    if not (pd.isna(break_yrs)):
        break_yrs = break_yrs.split("_")
        break_yrs = [int(x) for x in break_yrs]

        for brk_yr in break_yrs:
            plt.axvline(x=brk_yr, color='r', linestyle='--', label=f"{brk_yr}")

    state_ = SF_west[SF_west["fid"]==a_fid]["state_majo"].iloc[0]
    slope_ = round(SF_west.loc[SF_west["fid"] == a_fid, "sens_slope"].item(), 2)
    trend_ = SF_west[SF_west["fid"]==a_fid]["trend"].iloc[0]
    axes.set_title(f"{state_} (FID: {a_fid}). Sens slope: {slope_}. (MK: {trend_}.)", 
                   y=1.15, fontsize=tick_legend_FontSize, pad=-15)
    axes.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
    axes.set_xlabel('year') #, fontsize=14

    plt.legend()
    # file_name = G_breakpoint_TS_dir + str(a_fid) + "_MKG_breaks.pdf"
    # plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

    file_name = G_breakpoint_TS_dir + str(a_fid) + "_MK_breaks.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=230)
    plt.close()

# %%

# %%
for a_fid in brown_FIDs:
    fig, axes = plt.subplots(1, 1, figsize=(4, 2), sharey=False, sharex=False, dpi=dpi_)
    # fig, axes = plt.subplots(3, 1, figsize=(2, 4.5), sharey=True, sharex=True, dpi=dpi_)
    ###################################
    df = ANPP[ANPP["fid"] == a_fid]
    axes.plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);

    break_yrs = ANPP_breaks[ANPP_breaks["fid"] == a_fid]["breakpoint_years"].iloc[0]
    if not (pd.isna(break_yrs)):
        break_yrs = break_yrs.split("_")
        break_yrs = [int(x) for x in break_yrs]

        for brk_yr in break_yrs:
            plt.axvline(x=brk_yr, color='r', linestyle='--', label=f"{brk_yr}")

    state_ = SF_west[SF_west["fid"]==a_fid]["state_majo"].iloc[0]
    slope_ = round(SF_west.loc[SF_west["fid"] == a_fid, "sens_slope"].item(), 2)
    trend_ = SF_west[SF_west["fid"]==a_fid]["trend"].iloc[0]
    axes.set_title(f"{state_} (FID: {a_fid}). Sens slope: {slope_}. (MK: {trend_}.)", 
                   y=1.15, fontsize=tick_legend_FontSize, pad=-15)
    axes.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
    axes.set_xlabel('year') #, fontsize=14

    plt.legend()
    # file_name = G_breakpoint_TS_dir + str(a_fid) + "_MKB_breaks.pdf"
    # plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

    file_name = B_breakpoint_TS_dir + str(a_fid) + "_MK_breaks.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=230)
    plt.close()

# %%

# %%
for a_fid in noTrend_FIDs:
    fig, axes = plt.subplots(1, 1, figsize=(4, 2), sharey=False, sharex=False, dpi=dpi_)
    # fig, axes = plt.subplots(3, 1, figsize=(2, 4.5), sharey=True, sharex=True, dpi=dpi_)
    ###################################
    df = ANPP[ANPP["fid"] == a_fid]
    axes.plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);

    break_yrs = ANPP_breaks[ANPP_breaks["fid"] == a_fid]["breakpoint_years"].iloc[0]
    if not (pd.isna(break_yrs)):
        break_yrs = break_yrs.split("_")
        break_yrs = [int(x) for x in break_yrs]

        for brk_yr in break_yrs:
            plt.axvline(x=brk_yr, color='r', linestyle='--', label=f"{brk_yr}")

    state_ = SF_west[SF_west["fid"]==a_fid]["state_majo"].iloc[0]
    slope_ = round(SF_west.loc[SF_west["fid"] == a_fid, "sens_slope"].item(), 2)
    trend_ = SF_west[SF_west["fid"]==a_fid]["trend"].iloc[0]
    axes.set_title(f"{state_} (FID: {a_fid}). Sens slope: {slope_}. (MK: {trend_}.)", 
                   y=1.15, fontsize=tick_legend_FontSize, pad=-15)
    axes.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
    axes.set_xlabel('year') #, fontsize=14

    plt.legend()
    # file_name = G_breakpoint_TS_dir + str(a_fid) + "_MKG_breaks.pdf"
    # plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

    file_name = noTrend_breakpoint_TS_dir + str(a_fid) + "_MK_breaks.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=230)
    plt.close()

# %%
