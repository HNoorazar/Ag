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
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
import statsmodels.api as sm
from datetime import datetime

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

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
# filename = bio_reOrganized + "ANPP_MK_Spearman.sav"
# ANPP_MK_Spearman = pd.read_pickle(filename)
# ANPP_MK_Spearman = ANPP_MK_Spearman["ANPP_MK_df"]
# ANPP_MK_Spearman.head(2)

# %%

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_Spearman_no2012 = pd.read_pickle(filename)
ANPP_MK_Spearman_no2012 = ANPP_MK_Spearman_no2012["ANPP_MK_df"]
ANPP_MK_Spearman_no2012.head(2)

# %%
# trend_col = "trend"
# trend_count_orig = ANPP_MK_Spearman[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
# trend_count_orig.rename(columns={"fid": "fid_original"}, inplace=True)

# trend_col = "trend_yue"
# trend_count_yue = ANPP_MK_Spearman[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
# trend_count_yue.rename(columns={"fid": "fid_yue",
#                                 "trend_yue" : "trend"}, inplace=True)
# trend_col = "trend_rao"
# trend_count_rao = ANPP_MK_Spearman[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
# trend_count_rao.rename(columns={"fid": "fid_rao", 
#                                "trend_rao" : "trend"}, inplace=True)

# trend_counts = pd.merge(trend_count_rao, trend_count_yue, on="trend", how="outer")
# trend_counts = pd.merge(trend_counts, trend_count_orig, on="trend", how="outer")
# trend_counts

# %%
trend_col = "trend"
# trend_count_orig = ANPP_MK_Spearman_no2012[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
# trend_count_orig.rename(columns={"fid": "fid_original"}, inplace=True)
# trend_count_orig

trend_col = "trend_yue"
trend_count_yue = ANPP_MK_Spearman_no2012[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue.rename(columns={"fid": "fid_yue",
                                "trend_yue" : "trend"}, inplace=True)
trend_count_yue

trend_col = "trend_rao"
trend_count_rao = ANPP_MK_Spearman_no2012[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_rao.rename(columns={"fid": "fid_rao", 
                               "trend_rao" : "trend"}, inplace=True)
trend_count_rao

trend_counts_no2012 = pd.merge(trend_count_rao, trend_count_yue, on="trend", how="outer")
# trend_counts_no2012 = pd.merge(trend_counts_no2012, trend_count_orig, on="trend", how="outer")
trend_counts_no2012

# %%

# %%
# orig = (trend_counts_no2012.loc[trend_counts_no2012["trend"] == "increasing", "fid_original"] - \
#         trend_counts.loc[trend_counts["trend"] == "increasing", "fid_original"]).item()

# Yue = (trend_counts_no2012.loc[trend_counts_no2012["trend"] == "increasing", "fid_yue"] - \
#        trend_counts.loc[trend_counts["trend"] == "increasing", "fid_yue"]).item()

# Rao = (trend_counts_no2012.loc[trend_counts_no2012["trend"] == "increasing", "fid_rao"] - \
#        trend_counts.loc[trend_counts["trend"] == "increasing", "fid_rao"]).item()

# print ("diff. as a result of removing 2012 in  MK: {}".format(orig))
# print ("diff. as a result of removing 2012 in Rao: {}".format(Rao))
# print ("diff. as a result of removing 2012 in Yue: {}".format(Yue))

# %%
# bpszone_ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP.sav")
# print (bpszone_ANPP["Date"])
# bpszone_ANPP = bpszone_ANPP["bpszone_ANPP"]
# bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
# bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
#                              "count": "pixel_count",
#                              "mean" : "mean_lb_per_acr"}, inplace=True)

# bpszone_ANPP.sort_values(by=['fid', 'year'], inplace=True)
# bpszone_ANPP.reset_index(drop=True, inplace=True)
# bpszone_ANPP.head(2)

# %% [markdown]
# ### Lets stick to no-2012
#
# out of more than 27,000 ```FID``` on west of meridian we have only 4,332 ```FID``` for 2012 in NPP data.
#
#    - Find FIDs in common between Rao and Yue with increasing trend.
#    - Find FIDs not in common between Rao and Yue with increasing trend.

# %%
ANPP_MK_Spearman_no2012.head(2)

# %%
green_FIDs_Yue = ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["trend_yue"] == "increasing", "fid"].unique()
green_FIDs_Rao = ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["trend_rao"] == "increasing", "fid"].unique()
green_FIDs_Yue_Rao = set(green_FIDs_Yue).intersection(set(green_FIDs_Rao))

print (len(green_FIDs_Yue))
print (len(green_FIDs_Rao))
print (len(green_FIDs_Yue_Rao))

# %%
# %%time 
# Rao_FIDs_in_Yue = [0] * len(green_FIDs_Rao)
Rao_FIDs_in_Yue = set()
for a_fid in green_FIDs_Rao:
    if a_fid in green_FIDs_Yue:
        Rao_FIDs_in_Yue.add(a_fid)

len(Rao_FIDs_in_Yue)

# %%
print (len((Rao_FIDs_in_Yue)))
print (len(set(Rao_FIDs_in_Yue)))

# %%
# %%time 
# Rao_FIDs_in_Yue = [0] * len(green_FIDs_Rao)
Yue_FIDs_missed_by_Rao = set()
for a_fid in green_FIDs_Yue:
    if not(a_fid in green_FIDs_Rao):
        Yue_FIDs_missed_by_Rao.add(a_fid)

len(Yue_FIDs_missed_by_Rao)

# %% [markdown]
# ### Durbin-Watson test: This test is commonly used to detect autocorrelation in the residuals of a regression model
#
# **High positive autocorrelation**
#
#    - Near 2: Low autocorrelation 
#    - Near 0: Strong positive autocorrelation 
#    - Near 4: Strong negative autocorrelation

# %%
# bpszone_ANPP_no2012 = bpszone_ANPP[bpszone_ANPP["year"]!=2012].copy()

# %%
# # %%time
# data = {'fid': bpszone_ANPP["fid"].unique(), 
#         'dw_stat': [-666]*len(bpszone_ANPP["fid"].unique())}

# dw_stats_df = pd.DataFrame(data)

# # Perform the Durbin-Watson test
# for a_fid in bpszone_ANPP["fid"].unique():
#     TS = bpszone_ANPP.loc[bpszone_ANPP["fid"] == a_fid, "mean_lb_per_acr"]
#     dw_stats_df.loc[dw_stats_df["fid"] == a_fid, "dw_stat"] = sm.stats.stattools.durbin_watson(TS)

# dw_stats_df.head(2)

# %%
bpszone_ANPP_no2012 = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
print (bpszone_ANPP_no2012["Date"])
bpszone_ANPP_no2012 = bpszone_ANPP_no2012["bpszone_ANPP"]
bpszone_ANPP_no2012.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP_no2012.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP_no2012.sort_values(by=['fid', 'year'], inplace=True)
bpszone_ANPP_no2012.reset_index(drop=True, inplace=True)
bpszone_ANPP_no2012.head(2)

# %%
# %%time
data = {'fid': bpszone_ANPP_no2012["fid"].unique(), 
        'dw_stat': [-666]*len(bpszone_ANPP_no2012["fid"].unique())}

dw_stats_df_no2012 = pd.DataFrame(data)

# Perform the Durbin-Watson test
for a_fid in bpszone_ANPP_no2012["fid"].unique():
    TS = bpszone_ANPP_no2012.loc[bpszone_ANPP_no2012["fid"] == a_fid, "mean_lb_per_acr"]
    dw_stats_df_no2012.loc[dw_stats_df_no2012["fid"] == a_fid, "dw_stat"] = sm.stats.stattools.durbin_watson(TS)

dw_stats_df_no2012.rename(columns={"dw_stat": "dw_stat_no2012"}, inplace=True)
dw_stats_df_no2012.head(2)

# %%
# dw_stats_df = pd.merge(dw_stats_df, dw_stats_df_no2012, how="left", on="fid")
# dw_stats_df.head(2)

# %%
# dw_stats_df["dw_diff"] = dw_stats_df["dw_stat"] - dw_stats_df["dw_stat_no2012"]

# dw_stats_df.sort_values(by=["dw_diff"], ascending=False, inplace=True)
# dw_stats_df.reset_index(drop=True, inplace=True)
# dw_stats_df.head(6)

# %%
y_var = "mean_lb_per_acr"
dpi_ = 200
save_dpi=400

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
# fig, axes = plt.subplots(1, 1, figsize=(4, 1.5), sharex=True, sharey=False, dpi=dpi_)

# axes.grid(axis="y", which="both");

# ######### Maximum correlation by DW test
# loc_ = 0
# fid_corr = int(dw_stats_df.iloc[loc_]["fid"])
# df = bpszone_ANPP[bpszone_ANPP["fid"] == fid_corr]
# axes.plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);
# axes.scatter(df.year, df[y_var], marker='o', facecolors='r', edgecolors='r', s=5, zorder=2);

# state = ANPP_MK_Spearman.loc[ANPP_MK_Spearman["fid"] == fid_corr, "state_majority_area"].item()
# DW_ = round(dw_stats_df.iloc[loc_]["dw_stat"], 2)
# DW_no2012 = round(dw_stats_df.iloc[loc_]["dw_stat_no2012"], 2)

# title_ = "{}, FID: {}, DW: {}, DW (no 2012): {}".format(state, str(fid_corr), DW_, DW_no2012)
# axes.set_title(title_);

# axes.axvline(x=2012, color = 'k', label = '2012', linestyle="-")
# axes.legend(loc='best');

# axes.set_xlabel("year"); axes.set_ylabel("ANPP (mean lb/acr)");

# # plt.tight_layout()
# fig.subplots_adjust(top=0.85, bottom=0.21, left=0.11, right=0.981)
# file_name = bio_plots + "DW_test_2012_adverse.pdf"
# plt.savefig(file_name, dpi=save_dpi)

# %%
# del(dw_stats_df)

# %% [markdown]
# https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf
#
# https://en.wikipedia.org/wiki/Autoregressive_moving-average_model
#
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html

# %%
dw_stats_df_no2012.head(2)

# %%
print (len(dw_stats_df_no2012["fid"]))
print (dw_stats_df_no2012[dw_stats_df_no2012["dw_stat_no2012"] < 0.3].shape)
print (dw_stats_df_no2012[dw_stats_df_no2012["dw_stat_no2012"] < 0.4].shape)
print (dw_stats_df_no2012[dw_stats_df_no2012["dw_stat_no2012"] < 0.5].shape)

# %%
tick_legend_FontSize = 8
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
fig, axes = plt.subplots(1, 1, figsize=(4, 1.5), sharey=False, sharex=False, dpi=dpi_)
# sns.set_style({'axes.grid' : False})
# axes.grid(axis="y", which="both");
sns.histplot(data=dw_stats_df_no2012["dw_stat_no2012"], ax=axes, bins=100, kde=True); # height=5
# axes.legend(["Durbin–Watson statistic"], loc='upper right');

axes.set_xlabel("Durbin–Watson statistic");

fig.subplots_adjust(top=0.85, bottom=0.23, left=0.12, right=0.981)
file_name = bio_plots + "DW_test_distribution.pdf"
# plt.savefig(file_name, dpi=save_dpi)

# %% [markdown]
# ### Plot 3 ACFs

# %%
dw_stats_df_no2012["dw_stat_no2012"].median()

# %%
dw_stats_df_no2012[dw_stats_df_no2012.fid==6645]["dw_stat_no2012"].item()

# %%
min_DW_fid = dw_stats_df_no2012.loc[dw_stats_df_no2012["dw_stat_no2012"].idxmin(), "fid"]
max_DW_fid = dw_stats_df_no2012.loc[dw_stats_df_no2012["dw_stat_no2012"].idxmax(), "fid"]

# median
median_DF = dw_stats_df_no2012["dw_stat_no2012"].median()
# median_DW_fid = dw_stats_df_no2012.loc[dw_stats_df_no2012["dw_stat_no2012"] == median_DF, "fid"].item()
median_DW_fid = 6645 # Dec. 10. from old plot! how come this does not exist anymore?

# %%
import matplotlib.ticker as plticker

# %%
fig, axes = plt.subplots(3, 1, figsize=(4, 4.5), sharey=True, sharex=True, dpi=dpi_)
###################################
ax_ = 0
a_fid = max_DW_fid
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_], label="max DW stat.")
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)
# axes[ax_].legend(loc='best');

state_= ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["fid"] == max_DW_fid, "state_majority_area"].item()

txt_ = "location with maximum DW stat.\n{} (FID:{})".format(state_, max_DW_fid)
axes[ax_].text(3, .6, txt_, fontsize=6);
###################################
ax_ = 1
a_fid = median_DW_fid
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_])
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)

state_= ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["fid"] == \
                                    median_DW_fid, "state_majority_area"].item()
txt_ = "location with median DW stat.\n{} (FID:{})".format(state_, median_DW_fid)
axes[ax_].text(3, .6, txt_, fontsize=6);
###################################
ax_ = 2
a_fid = min_DW_fid
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_])
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)

state_= ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["fid"] == \
                                    min_DW_fid, "state_majority_area"].item()

txt_ = "location with minimum DW stat.\n{} (FID:{})".format(state_, min_DW_fid)
axes[ax_].text(3, .6, txt_, fontsize=6);

###################################
axes[0].set_title("ACF of ANPP");
axes[1].set_title(None);
axes[2].set_title(None);

axes[2].set_xlabel("lag"); axes[1].set_ylabel("autocorrelation");

fig.subplots_adjust(top=0.95, bottom=0.08, left=0.14, right=0.981)
file_name = bio_plots + "ACF_for_3FIDs.pdf"
plt.savefig(file_name, dpi=save_dpi, bbox_inches="tight")

# %%
tick_legend_FontSize = 8
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
fig, axes = plt.subplots(1, 3, figsize=(4.5, 2), sharey=True, sharex=True, dpi=dpi_)
x_text, y_text = 0.2, 0.6
###################################
ax_ = 0
a_fid = max_DW_fid
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_], label="max DW stat.")
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)
# axes[ax_].legend(loc='best');

state_= ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["fid"] == max_DW_fid, "state_majority_area"].item()

txt_ = "FID w/ max. DW stat.\n{} (FID:{})".format(state_, max_DW_fid)
axes[ax_].text(x_text, y_text, txt_, fontsize=6);
###################################
ax_ = 1
a_fid = median_DW_fid
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_])
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)

state_= ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["fid"] == \
                                    median_DW_fid, "state_majority_area"].item()
txt_ = "FID w/ median DW stat.\n{} (FID:{})".format(state_, median_DW_fid)
axes[ax_].text(x_text, y_text, txt_, fontsize=6);
###################################
ax_ = 2
a_fid = min_DW_fid
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_])
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)

state_= ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["fid"] == \
                                    min_DW_fid, "state_majority_area"].item()

txt_ = "FID w/ min. DW stat.\n{} (FID:{})".format(state_, min_DW_fid)
axes[ax_].text(x_text, y_text, txt_, fontsize=6);

###################################
axes[1].set_title("ACF of ANPP"); axes[0].set_title(None); axes[2].set_title(None);
axes[1].set_xlabel("lag"); axes[0].set_ylabel("autocorrelation");

fig.subplots_adjust(top=0.91, bottom=0.2, left=0.12, right=0.981)
file_name = bio_plots + "ACF_for_3FIDs_wide2.pdf"
plt.savefig(file_name, dpi=save_dpi, bbox_inches="tight")

# %%

# %%

# %%

# %%
fig, axes = plt.subplots(3, 1, figsize=(4, 4.5), sharey=False, sharex=False, dpi=dpi_)
###################################
ax_ = 0
a_fid = max_DW_fid

df = bpszone_ANPP_no2012[bpszone_ANPP_no2012["fid"] == a_fid]
axes[ax_].plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);
axes[ax_].scatter(df.year, df[y_var], marker='o', facecolors='r', edgecolors='r', s=5, zorder=2);


state_= ANPP_MK_Spearman.loc[ANPP_MK_Spearman["fid"] == max_DW_fid, "state_majority_area"].item()

txt_ = "location with maximum DW stat.\n{} (FID:{})".format(state_, max_DW_fid)
y_txt = int(df[y_var].max() /1.3)
axes[ax_].text(2005, y_txt, txt_, fontsize=6);
###################################
ax_ = 1
a_fid = median_DW_fid

df = bpszone_ANPP_no2012[bpszone_ANPP_no2012["fid"] == a_fid]
axes[ax_].plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);
axes[ax_].scatter(df.year, df[y_var], marker='o', facecolors='r', edgecolors='r', s=5, zorder=2);


state_= ANPP_MK_Spearman.loc[ANPP_MK_Spearman["fid"] == min_DW_fid, "state_majority_area"].item()

txt_ = "location with median DW stat.\n{} (FID:{})".format(state_, min_DW_fid)
y_txt = int(df[y_var].max() / 1.2)
axes[ax_].text(1985, y_txt, txt_, fontsize=6);
###################################
ax_ = 2
a_fid = min_DW_fid

df = bpszone_ANPP_no2012[bpszone_ANPP_no2012["fid"] == a_fid]
axes[ax_].plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);
axes[ax_].scatter(df.year, df[y_var], marker='o', facecolors='r', edgecolors='r', s=5, zorder=2);


state_= ANPP_MK_Spearman.loc[ANPP_MK_Spearman["fid"] == min_DW_fid, "state_majority_area"].item()

txt_ = "location with minimum DW stat.\n{} (FID:{})".format(state_, min_DW_fid)
y_txt = int(df[y_var].min())
axes[ax_].text(2005, y_txt, txt_, fontsize=6);
###################################

axes[0].set_title("ANPP for the 3 DW locations");
axes[1].set_title(None);
axes[2].set_title(None);

axes[2].set_xlabel("lag"); axes[1].set_ylabel("autocorrelation");


fig.subplots_adjust(top=0.95, bottom=0.08, left=0.14, right=0.981)
file_name = bio_plots + "NPP_for_3FIDs.pdf"
# plt.savefig(file_name, dpi=save_dpi)

# %%
df[y_var].max() / 1.1

# %% [markdown]
# ### Label of Yue-Geerning in Original MK

# %%
a_fid = max_DW_fid
TS = bpszone_ANPP.loc[bpszone_ANPP["fid"] == a_fid, "mean_lb_per_acr"]
sm.stats.stattools.durbin_watson(TS).round(2)

# %%
a_fid = max_DW_fid
TS = bpszone_ANPP.loc[bpszone_ANPP["fid"] == a_fid].copy()
NPP_outlier_idx  = TS["mean_lb_per_acr"].idxmax()
print (TS.shape)
TS.drop(index=[NPP_outlier_idx], inplace=True)
print (TS.shape)
sm.stats.stattools.durbin_watson(TS["mean_lb_per_acr"]).round(2)

# %%
# [4774, 21470, 23272]

FIDs = list(bpszone_ANPP_no2012["fid"].unique())
random.sample(list(FIDs), 3)

# %%
# random.seed(1)
random.seed(2)
three_random_fids = random.sample(list(FIDs), 3)

fig, axes = plt.subplots(3, 1, figsize=(4, 4.5), sharey=True, sharex=True, dpi=dpi_)
###################################
ax_ = 0
a_fid = three_random_fids[0]
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_], label="max DW stat.")
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)
###################################
ax_ = 1
a_fid = three_random_fids[1]
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_])
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)
###################################
ax_ = 2
a_fid = three_random_fids[2]
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_])
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)
###################################
axes[0].set_title("ACF of ANPP for 3 random locations");
axes[1].set_title(None);
axes[2].set_title(None);

axes[2].set_xlabel("lag"); axes[1].set_ylabel("autocorrelation");

fig.subplots_adjust(top=0.95, bottom=0.08, left=0.14, right=0.981)
file_name = bio_plots + "ACF_for_3random_FIDs.pdf"
plt.savefig(file_name, dpi=save_dpi, bbox_inches="tight")

# %%
tick_legend_FontSize = 8
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
# random.seed(1)
random.seed(2)
three_random_fids = random.sample(list(FIDs), 3)

fig, axes = plt.subplots(1, 3, figsize=(4.5, 2), sharey=True, sharex=True, dpi=dpi_,
                         gridspec_kw={"hspace": 0.5, "wspace": 0.05})
###################################
ax_ = 0
a_fid = three_random_fids[0]
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_], label="max DW stat.")
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)
###################################
ax_ = 1
a_fid = three_random_fids[1]
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_])
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)
###################################
ax_ = 2
a_fid = three_random_fids[2]
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5, ax=axes[ax_])
loc = plticker.MultipleLocator(base=.5)
axes[ax_].yaxis.set_major_locator(loc)
###################################
axes[1].set_title("ACF of ANPP for 3 random locations");
axes[0].set_title(None);
axes[2].set_title(None);

axes[1].tick_params(axis='y', left=False)
axes[2].tick_params(axis='y', left=False)

axes[1].set_xlabel("lag"); axes[0].set_ylabel("autocorrelation");
fig.subplots_adjust(top=0.91, bottom=0.2, left=0.12, right=0.981)
file_name = bio_plots + "ACF_for_3random_FIDs_narow.pdf"
plt.savefig(file_name, dpi=save_dpi, bbox_inches="tight")

# %%
