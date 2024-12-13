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

import scipy.signal
import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import pymannkendall as mk
from scipy.stats import variation

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
    
    
# Major ticks every 5, minor ticks every 1
major_ticks = np.arange(1984, 2024, 5)
minor_ticks = np.arange(1984, 2024, 1)
y_var = "mean_lb_per_acr"

dpi_, map_dpi_=300, 900
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

# %%
txt_font_dict = {'fontsize':10, 'fontweight':'bold'}

tick_legend_FontSize = 12
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05
         }

plt.rcParams.update(params)

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

# %% [markdown]
# ## Directories

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

# %%
# county_fips_dict = pd.read_pickle(rangeland_reOrganized + "county_fips.sav")

# county_fips = county_fips_dict["county_fips"]
# full_2_abb = county_fips_dict["full_2_abb"]
# abb_2_full_dict = county_fips_dict["abb_2_full_dict"]
# abb_full_df = county_fips_dict["abb_full_df"]
# filtered_counties_29States = county_fips_dict["filtered_counties_29States"]
# SoI = county_fips_dict["SoI"]
# state_fips = county_fips_dict["state_fips"]

# state_fips = state_fips[state_fips.state != "VI"].copy()
# state_fips.head(2)

# %% [markdown]
# ## Read NPP time-series

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
print (2012 in sorted(ANPP.year.unique()))
ANPP.head(2)


# %% [markdown]
# ### Read trend data
#
# **Do we need this? they are in SF.**

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
ANPP_MK_df["EW_meridian"].unique()

# %% [markdown]
# # Read shapefile

# %%
# # %%time
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
# SF_west = geopandas.read_file(f_name)
# SF_west["centroid"] = SF_west["geometry"].centroid


# SF_west.rename(columns={"EW_meridia": "EW_meridian",
#                         "p_valueSpe" : "p_valueSpearman",
#                         "medians_di": "medians_diff_ANPP",
#                         "medians__1" : "medians_diff_slope_ANPP",
#                         "median_ANP" : "median_ANPP_change_as_perc",
#                         "state_majo" : "state_majority_area"}, 
#                       inplace=True)
# SF_west.head(2)

# %%
ANPP_MK_df[["fid", "trend_yue"]].groupby("trend_yue").count()

# %%
ANPP_MK_df[["fid", "trend"]].groupby("trend").count()

# %%
ANPP_MK_df_99confidence_orig = ANPP_MK_df[ANPP_MK_df["p"]<0.01].copy()
ANPP_MK_df_99confidence_orig[["fid", "trend"]].groupby("trend").count()

# %%
ANPP_MK_df_99confidence_orig[ANPP_MK_df_99confidence_orig["sens_slope"]>30].shape

# %%
print (ANPP_MK_df[ANPP_MK_df["sens_slope"]>30].shape)
print (ANPP_MK_df[(ANPP_MK_df["sens_slope"]>30) & (ANPP_MK_df["p"]<0.05)].shape)
print (ANPP_MK_df[(ANPP_MK_df["sens_slope"]>30) & (ANPP_MK_df["p"]<0.01)].shape)

# %%
green_99 = ANPP_MK_df[["fid", "trend", "p", "sens_slope"]].copy()
green_99 = green_99[green_99["sens_slope"] > 30]
green_99 = green_99[green_99["trend"] != "increasing"]
print (list(green_99["fid"]))

# %%
bio_plots = rangeland_bio_base + "plots/"
steepSlope_noTrend_dir = bio_plots + "steepSlope_noTrend/"
os.makedirs(steepSlope_noTrend_dir, exist_ok=True)

# %%
for a_fid in list(green_99["fid"]):
    fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, 
                             gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
    # axes.grid(which='major', alpha=0.5, axis="both")
    axes.set_xticks(major_ticks)
    axes.set_xticks(minor_ticks, minor=True)
    axes.grid(which='minor', alpha=0.2, axis="x")
    axes.grid(which='major', alpha=0.5, axis="x")

    ########## plot 1
    df = ANPP[ANPP.fid == a_fid].copy()
    axes.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
    axes.scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

    ###
    ### Text 
    trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].item()
    slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
    Tau_   = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
    state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

    text_ = "trend: {}\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, slope_, Tau_, state_, a_fid)
    y_txt = df[y_var].max() * .99
    axes.text(1983, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
    ####
    axes.set_title("Sens slope greater than 30 but labeled as no trend by original MK");
    axes.set_ylabel(r'$\mu_{NPP}$ (lb/acr)');

    file_name = steepSlope_noTrend_dir + str(a_fid) + "_SloleGE30_noTrendOrigMK.pdf"
    plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')
    plt.close("all")

# %% [markdown]
# ## 99% confidence level

# %%
ANPP_MK_df_99confidence = ANPP_MK_df[ANPP_MK_df["p_yue"]<0.01].copy()
print (f"{ANPP_MK_df_99confidence.shape[0] = }")
print (f"{ANPP_MK_df.shape[0] = }")
ANPP_MK_df_99confidence.head(2)

# %%
ANPP_MK_df_99confidence_G = ANPP_MK_df_99confidence[ANPP_MK_df_99confidence["trend_yue"] == "increasing"]
print (len(ANPP_MK_df_99confidence_G))

# %%
25187 - 24350

# %%
ANPP_MK_df_99confidence_G[ANPP_MK_df_99confidence_G["trend"] != "increasing"].shape

# %%
ANPP_MK_df_99confidence[["fid", "trend_yue"]].groupby("trend_yue").count()

# %%
ANPP.shape

# %%
greening_99_FIDs_list = list(ANPP_MK_df_99confidence_G["fid"].unique())

ANPP_99CL_G = ANPP[ANPP["fid"].isin(greening_99_FIDs_list)].copy()
ANPP_MK_99CL_G = ANPP_MK_df[ANPP_MK_df["fid"].isin(greening_99_FIDs_list)].copy()

# %%
print (len(ANPP))
print (len(ANPP_99CL_G))
print()
print (len(ANPP_MK_df))
print (len(ANPP_MK_99CL_G))

# %%

# %%
ANPP_MK_99CL_G.head(2)

# %%
ANPP_MK_99CL_G["trend"].unique()

# %%
ANPP_MK_99CL_G[ANPP_MK_99CL_G["trend"] != "increasing"].shape

# %%
ANPP_99CL_G
ANPP_MK_99CL_G

# %% [markdown]
# # Extremes of Sens slope in 99% Yue

# %%
green_FID_min_99CL = ANPP_MK_99CL_G.loc[ANPP_MK_99CL_G.sens_slope.idxmin(), "fid"]
green_FID_max_99CL = ANPP_MK_99CL_G.loc[ANPP_MK_99CL_G.sens_slope.idxmax(), "fid"]

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
# axes[0].grid(which='major', alpha=0.5, axis="both")
# axes[1].grid(which='major', alpha=0.5, axis="y")
axes[0].set_xticks(major_ticks)
axes[0].set_xticks(minor_ticks, minor=True)
axes[0].grid(which='minor', alpha=0.2, axis="x")
axes[0].grid(which='major', alpha=0.5, axis="x")

axes[1].set_xticks(major_ticks)
axes[1].set_xticks(minor_ticks, minor=True)
axes[1].grid(which='minor', alpha=0.2, axis="x")
axes[1].grid(which='major', alpha=0.5, axis="x")

##########
########## plot 1
##########
a_fid = green_FID_min_99CL
df = ANPP[ANPP.fid == a_fid].copy()
axes[0].plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes[0].scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

###
### Text 
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_   = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

text_ = "trend: {} (Yue. 99% CL)\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, slope_, Tau_, \
                                                                                  state_, a_fid)
y_txt = df[y_var].max() * .99
axes[0].text(1983, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
##########
########## plot 2
##########
a_fid = green_FID_max_99CL
df = ANPP[ANPP.fid == a_fid].copy()
axes[1].plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes[1].scatter(df.year, df[y_var], zorder=3, color="dodgerblue");
### Text 
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_   = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

text_ = "trend: {} (Yue. 99% CL)\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, slope_, Tau_, \
                                                                                  state_, a_fid)

y_txt = df[y_var].max() * .99
axes[1].text(1983, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");

####
axes[0].set_title("Sen's slope extremes (Yue greening with 99% CL)");
axes[0].set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
axes[1].set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
# axes.legend(loc="best");

file_name = yue_plots + "Yue99PercCL_SensExtremes.pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %%

# %% [markdown]
# # Extremes of Tau in 99% Yue

# %%
green_FID_min_99CL = ANPP_MK_99CL_G.loc[ANPP_MK_99CL_G.Tau.idxmin(), "fid"]
green_FID_max_99CL = ANPP_MK_99CL_G.loc[ANPP_MK_99CL_G.Tau.idxmax(), "fid"]

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
# axes[0].grid(which='major', alpha=0.5, axis="both")
# axes[1].grid(which='major', alpha=0.5, axis="y")
axes[0].set_xticks(major_ticks)
axes[0].set_xticks(minor_ticks, minor=True)
axes[0].grid(which='minor', alpha=0.2, axis="x")
axes[0].grid(which='major', alpha=0.5, axis="x")

axes[1].set_xticks(major_ticks)
axes[1].set_xticks(minor_ticks, minor=True)
axes[1].grid(which='minor', alpha=0.2, axis="x")
axes[1].grid(which='major', alpha=0.5, axis="x")

##########
########## plot 1
##########
a_fid = green_FID_min_99CL
df = ANPP[ANPP.fid == a_fid].copy()
axes[0].plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes[0].scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

###
### Text 
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

text_ = "trend: {} (Yue. 99% CL)\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, slope_, Tau_, \
                                                                                  state_, a_fid)
y_txt = df[y_var].max() / 1.3
axes[0].text(1983, y_txt, text_, fontsize = 12);
# axes[0].set_ylim(3000, 4500);
##########
########## plot 2
##########
a_fid = green_FID_max_99CL
df = ANPP[ANPP.fid == a_fid].copy()
axes[1].plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes[1].scatter(df.year, df[y_var], zorder=3, color="dodgerblue");
### Text 
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

text_ = "trend: {} (Yue. 99% CL)\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, slope_, Tau_, \
                                                                                  state_, a_fid)
y_txt = df[y_var].max() / 1.3
axes[1].text(1983, y_txt, text_, fontsize = 12);

####
axes[0].set_title("Tau extremes (Yue greening with 99% CL)");
axes[0].set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
axes[1].set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
# axes.legend(loc="best");

file_name = yue_plots + "Yue99PercCL_TauExtremes.pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %% [markdown]
# ### Filter by both Slope and Tau and see:

# %%
df = ANPP_MK_99CL_G.copy()
df = df[df["sens_slope"] > 20]
df[["fid", "trend_yue"]].groupby("trend_yue").count().reset_index()

# %%
df = ANPP_MK_df.copy()
df = df[df["sens_slope"] > 30]
df[["fid", "trend_yue"]].groupby("trend_yue").count().reset_index()

# %%

# %%
df = ANPP_MK_99CL_G.copy()
df = df[df["sens_slope"] > 20]
df[["fid", "trend_yue"]].groupby("trend_yue").count().reset_index()

# %%
df = ANPP_MK_99CL_G.copy()
df = df[df["sens_slope"] > 20]
df[["fid", "trend_yue"]].groupby("trend_yue").count().reset_index()

# %%
df = ANPP_MK_99CL_G.copy()
df = df[df["Tau"] > 0.5]
df = df[df["sens_slope"] > 30]
df[["fid", "trend_yue"]].groupby("trend_yue").count().reset_index()

# %%
df = ANPP_MK_99CL_G.copy()
df = df[df["Tau"] > 0.15]
df = df[df["sens_slope"] > 3]
df[["fid", "trend_yue"]].groupby("trend_yue").count().reset_index()

# %%
(24350+21366)/2 

# %% [markdown]
# ## Smooth effect

# %%
FIDs = [27045,  14926]
two_NPP_TS = ANPP[ANPP["fid"].isin(FIDs)].copy()
two_NPP_TS.reset_index(drop=True, inplace=True)

SG_col = "SG_" + y_var
two_NPP_TS[SG_col] = -666
two_NPP_TS.head(2)

# %%

# %%

# %%
window_s = 7
polyorder_ = 3

for an_ID in two_NPP_TS["fid"].unique():
    print (an_ID)
    df_ = two_NPP_TS[two_NPP_TS["fid"] == an_ID]
    TS = df_[y_var].values
    SG_y = scipy.signal.savgol_filter(TS, window_length=window_s, polyorder=polyorder_)
    
    two_NPP_TS.loc[df_.index, SG_col] = SG_y
    
two_NPP_TS.head(2)

# %%
need_cols = ["fid"]
ANPP_MK_SG = two_NPP_TS[need_cols].copy()
ANPP_MK_SG.drop_duplicates(inplace=True)
ANPP_MK_SG.reset_index(drop=True, inplace=True)

##### z: normalized test statistics
##### Tau: Kendall Tau
MK_test_cols = ["trend", "trend_yue", "p", "p_yue", "Tau", "MK_score", "var_s", "var_s_yue", "sens_slope"]

ANPP_MK_SG = pd.concat([ANPP_MK_SG, pd.DataFrame(columns = MK_test_cols)])
ANPP_MK_SG[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)

# Why data type changed?!
ANPP_MK_SG["fid"] = ANPP_MK_SG["fid"].astype(np.int64)
ANPP_MK_SG.head(2)


# populate the dataframe with MK test result now
for a_FID in ANPP_MK_SG["fid"].unique():
    ANPP_TS = two_NPP_TS.loc[two_NPP_TS.fid==a_FID, SG_col].values
    year_TS = two_NPP_TS.loc[two_NPP_TS.fid==a_FID, "year"].values
    
    # MK test
    trend, _, p, _, Tau, MK_score, var_s, slope, intercept = mk.original_test(ANPP_TS)
    trend_yue, _, p_yue, _, _, _, var_s_yue, _, _ = mk.yue_wang_modification_test(ANPP_TS)

    # Update dataframe by MK result
    L_ = [trend, trend_yue, p, p_yue, Tau, MK_score, var_s, var_s_yue, slope]
    
    ANPP_MK_SG.loc[ANPP_MK_SG["fid"]==a_FID, MK_test_cols] = L_

# Round the columns to 6-decimals
for a_col in list(ANPP_MK_SG.columns[7:]):
    ANPP_MK_SG[a_col] = ANPP_MK_SG[a_col].round(6)

ANPP_MK_SG.head(2)

# %%
ANPP_MK_df.loc[ANPP_MK_df.fid.isin(list(FIDs)), ANPP_MK_SG.columns]

# %%

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
axes[0].set_xticks(major_ticks)
axes[0].set_xticks(minor_ticks, minor=True)
axes[0].grid(which='minor', alpha=0.2, axis="x")
axes[0].grid(which='major', alpha=0.5, axis="x")

axes[1].set_xticks(major_ticks)
axes[1].set_xticks(minor_ticks, minor=True)
axes[1].grid(which='minor', alpha=0.2, axis="x")
axes[1].grid(which='major', alpha=0.5, axis="x")
##########
########## plot 1
##########
a_fid = FIDs[0]
df = two_NPP_TS[two_NPP_TS.fid == a_fid].copy()
axes[0].plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes[0].scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

axes[0].plot(df.year, df[SG_col], linewidth=3, color="green", label="SG");
axes[0].scatter(df.year, df[SG_col], zorder=3, color="green");

###
### Text 
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
p_yue_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "p_yue"].item()
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

text_ = "trend: {} ({:.4f})\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, p_yue_, slope_, Tau_, \
                                                                          state_, a_fid)
# y_txt = df[y_var].max() / 1.5
# axes[0].text(1983, y_txt, text_, fontsize = 12);

y_txt = df[y_var].max() * .99
axes[0].text(1983, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
print (p_yue_)
del(trend_, p_yue_, slope_, Tau_, state_)
### SG Text
trend_ = ANPP_MK_SG.loc[ANPP_MK_SG.fid == a_fid, "trend_yue"].item()
p_yue_ = ANPP_MK_SG.loc[ANPP_MK_SG.fid == a_fid, "p_yue"].item()
slope_ = round(ANPP_MK_SG.loc[ANPP_MK_SG.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_SG.loc[ANPP_MK_SG.fid == a_fid, "Tau"].item(), 2)
text_ = "trend: {} ({:.4f})\nSen's slope: {}, \nTau: {}\n".format(trend_, p_yue_, slope_, Tau_)
axes[0].text(1995, y_txt, text_, color="green", fontsize=tick_legend_FontSize*1.2, va="top");
print (p_yue_)
del(trend_, p_yue_, slope_, Tau_)
##########
########## plot 2
##########
a_fid = FIDs[1]
df = two_NPP_TS[two_NPP_TS.fid == a_fid].copy()
axes[1].plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes[1].scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

axes[1].plot(df.year, df[SG_col], linewidth=3, color="green", label="SG");
axes[1].scatter(df.year, df[SG_col], zorder=3, color="green");
###
### Text 
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
p_yue_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "p_yue"].item()
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

text_ = "trend: {} ({:.4f})\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, p_yue_, slope_, Tau_, \
                                                                          state_, a_fid)
y_txt = df[y_var].max() * .99
axes[1].text(1983, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
print (p_yue_)
del(trend_, p_yue_, slope_, Tau_, state_)
### SG Text
trend_ = ANPP_MK_SG.loc[ANPP_MK_SG.fid == a_fid, "trend_yue"].item()
p_yue_ = ANPP_MK_SG.loc[ANPP_MK_SG.fid == a_fid, "p_yue"].item()
slope_ = round(ANPP_MK_SG.loc[ANPP_MK_SG.fid == a_fid, "sens_slope"].item(), 2)
Tau_ = round(ANPP_MK_SG.loc[ANPP_MK_SG.fid == a_fid, "Tau"].item(), 2)
text_ = "trend: {} ({:.4f})\nSen's slope: {}, \nTau: {}\n".format(trend_, p_yue_, slope_, Tau_)
axes[1].text(1995, y_txt, text_, color="green", fontsize=tick_legend_FontSize*1.2, va="top");

print (p_yue_)
del(trend_, p_yue_, slope_, Tau_)
####################################################################################
axes[0].legend(loc="upper right")
axes[1].legend(loc="upper right")
file_name = yue_plots + "A3.pdf"
# plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %% [markdown]
# ### Annual change sign

# %%
# %%time
all_FID_list = ANPP_MK_df["fid"].unique()
cols_ = ["fid", "num_yrs", "increase_count"]
positive_count_df = pd.DataFrame(index=np.arange(len(all_FID_list)), columns=cols_)
positive_count_df["fid"] = all_FID_list

for an_fid in all_FID_list:
    df = ANPP[ANPP["fid"] == an_fid]
    delta = df.iloc[1:][y_var].values - df.iloc[:-1][y_var].values
    
    positive_count_df.loc[positive_count_df["fid"] == an_fid, "num_yrs"] = len(df)
    positive_count_df.loc[positive_count_df["fid"] == an_fid, "increase_count"] = sum(delta>0)

# %%
positive_count_df["num_yrs"] = positive_count_df["num_yrs"].astype("int")
positive_count_df["increase_count"] = positive_count_df["increase_count"].astype("int")
positive_count_df.head(3)

# %%
print (ANPP_MK_df.shape)
ANPP_MK_df = pd.merge(ANPP_MK_df, positive_count_df, how="left", on="fid")
print (ANPP_MK_df.shape)

# %%
tick_legend_FontSize = 10
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
          "ytick.labelleft": True,
          'axes.linewidth' : .05}
plt.rcParams.update(params)

# %%
FIDs_pos_deltas = positive_count_df[positive_count_df["increase_count"] > 21]
FIDs_pos_deltas = FIDs_pos_deltas["fid"].values
print (len(FIDs_pos_deltas))
FIDs_pos_deltas

# %%
ii = 3
a_fid = FIDs_pos_deltas[ii]

fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
# axes.grid(which='major', alpha=0.5, axis="both")
axes.set_xticks(major_ticks)
axes.set_xticks(minor_ticks, minor=True)
axes.grid(which='minor', alpha=0.2, axis="x")
axes.grid(which='major', alpha=0.5, axis="x")

########## plot 1
df = ANPP[ANPP.fid == a_fid].copy()
axes.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes.scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

###
### Text 
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_   = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

text_ = "trend: {}\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, slope_, Tau_, state_, a_fid)
y_txt = df[y_var].max() * .99
axes.text(1983, y_txt, text_, fontsize=tick_legend_FontSize*1.2, va="top");
####
axes.set_title("a field with positive annual changes");
axes.set_ylabel(r'$\mu_{NPP}$ (lb/acr)');

# file_name = yue_plots + "Yue99PercCL_SensExtremes.pdf"
# plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %%

# %%

# %%
groupveg = sorted(ANPP["groupveg"].unique())
groupveg

# %%
veg_colors = {"Barren-Rock/Sand/Clay" : "blue",
              "Conifer" : "green",
              "Grassland" : "red",
              "Hardwood" : "cyan",
              "Riparian" : "magenta",
              "Shrubland" : "yellow",
              "Sparse" : "black"}
