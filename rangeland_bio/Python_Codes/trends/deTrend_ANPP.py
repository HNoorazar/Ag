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
# Here we detrend ANPP so that we can use that in computing ACF1 and rolling windows.

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
import rangeland_plot_core as rcp

# %%
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

fontdict_normal = {'family':'serif', 'weight':'normal'}
fontdict_bold   = {'family':'serif', 'weight':'bold'}
inset_axes_     = [0.1, 0.13, 0.45, 0.03]

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
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

# %% [markdown]
# ### Get the Sen's slopes for detrending

# %%
# %%time
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
Albers_SF_west = geopandas.read_file(f_name)
Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid

Albers_SF_west.rename(columns={"EW_meridia" : "EW_meridian",
                               "sens_inter" : "sens_intercept",
                               "p_valueSpe" : "p_valueSpearman",
                               "medians_di" : "medians_diff_ANPP",
                               "medians__1" : "medians_diff_slope_ANPP",
                               "median_ANP" : "median_ANPP_change_as_perc",
                               "state_majo" : "state_majority_area"}, 
                      inplace=True)

Albers_SF_west.head(2)

# %%

# %%
# Merge the dataframes on 'fid'
ANPP = ANPP.merge(Albers_SF_west[['fid', 'sens_slope', 'sens_intercept']], on='fid', how='left')
ANPP.head(2)

# %% [markdown]
# ### Sens prediction 
#
# must not be based on year since that test only lookst at y values.
#

# %%
ANPP['row_number_perfid'] = ANPP.groupby('fid').cumcount()
ANPP.head(2)

# %%
ANPP["anpp_senPred"] = ANPP["row_number_perfid"] * ANPP["sens_slope"] + ANPP["sens_intercept"]
ANPP.head(2)

# %%

# %%
ANPP["anpp_detrendSens"] = ANPP["mean_lb_per_acr"] - ANPP["anpp_senPred"]
ANPP.head(2)

# %%
# after seaborn, things get messed up. Fix them:
matplotlib.rc_file_defaults()
font = {"size": 14}
matplotlib.rc("font", **font)
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
ANPP.head(2)

# %%
max_senSlope_fid = Albers_SF_west.loc[Albers_SF_west['sens_slope'].idxmax(), 'fid']
min_senSlope_fid = Albers_SF_west.loc[Albers_SF_west['sens_slope'].idxmin(), 'fid']

# %%
a_fid = max_senSlope_fid
fig, axes = plt.subplots(2, 1, figsize=(10, 3), sharex=True, sharey=False, dpi=dpi_,
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05});
(ax1, ax2) = axes
df = ANPP[ANPP.fid == a_fid]
interc_ = df['sens_intercept'].unique().item()
senSlope_ = slope=df['sens_slope'].unique().item()
ax1.plot(df["year"], df['mean_lb_per_acr'], color="dodgerblue", linewidth=3, label="ANPP");
ax1.plot(df["year"], df['anpp_senPred'], color="red", linewidth=3, label="Sen's prediction");


ax2.plot(df["year"], df['anpp_detrendSens'], color="g", linewidth=3, label="Sen's detrended");

ax1.legend(loc='best');
ax2.legend(loc='best');

# %%

# %% [markdown]
# ## First difference detrending

# %%
ANPP['anpp_detrendDiff'] = ANPP.groupby('fid')['mean_lb_per_acr'].diff()
ANPP.head(2)

# %% [markdown]
# ## detrend using Simple Linear regression

# %%
from sklearn.linear_model import LinearRegression

# %%
# %%time

unique_fids = ANPP['fid'].unique()
# Initialize empty DataFrame with desired columns
regression_df = pd.DataFrame({'fid': unique_fids,
                              'linReg_slope': np.nan,
                              'linReg_intercept': np.nan})
regression_df = regression_df.set_index('fid')


# Prepare a column to store detrended values
ANPP['anpp_detrendLinReg'] = np.nan

# Loop over each fid group
for fid, group in ANPP.groupby('fid'):
    # Reshape year for sklearn
    X = group['year'].values.reshape(-1, 1)
    y = group['mean_lb_per_acr'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    yhat = model.predict(X)
    ANPP.loc[group.index, 'anpp_detrendLinReg'] = y - yhat

    # Optionally store slope/intercept
    regression_df.loc[fid, 'linReg_slope'] = model.coef_[0]
    regression_df.loc[fid, 'linReg_intercept'] = model.intercept_
    
regression_df.reset_index(drop=False, inplace=True)

regression_df.head(2)

# %%
ANPP.head(2)

# %%
a_fid = max_senSlope_fid
fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True, sharey=False, dpi=dpi_,
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05});
(ax1, ax2, ax3, ax4) = axes

df = ANPP[ANPP.fid == a_fid]
interc_ = df['sens_intercept'].unique().item()
senSlope_ = slope=df['sens_slope'].unique().item()
ax1.plot(df["year"], df['mean_lb_per_acr'], color="dodgerblue", linewidth=3, label="ANPP");
ax1.plot(df["year"], df['anpp_senPred'], color="red", linewidth=3, label="Sen's prediction");

###############################################################################################
ax2.plot(df["year"], df['anpp_detrendSens'], color="g", linewidth=3, label="Sen's detrended");
ax2.plot(df["year"], df['anpp_detrendLinReg'], color="k", linewidth=1.5, label="regression detrended");
###############################################################################################
ax3.plot(df["year"], df['anpp_detrendLinReg'], color="k", linewidth=3, label="regression detrended");
###############################################################################################
ax4.plot(df["year"], df['anpp_detrendDiff'], color="y", linewidth=3, label="first-diff");

ax1.legend(loc='best'); ax2.legend(loc='best');
ax3.legend(loc='best'); ax4.legend(loc='best');

# %%
a_fid = max_senSlope_fid
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True, dpi=dpi_,
                        gridspec_kw={"hspace": 0.1, "wspace": 0.05});
(ax1, ax2) = axes

ax1.grid(axis='y', alpha=0.7, zorder=0);
ax2.grid(axis='y', alpha=0.7, zorder=0);

df = ANPP[ANPP.fid == a_fid]
interc_ = df['sens_intercept'].unique().item()
senSlope_ = slope=df['sens_slope'].unique().item()
ax1.plot(df["year"], df['mean_lb_per_acr'], c="dodgerblue", lw=3, label="ANPP");
ax1.plot(df["year"], df['anpp_senPred'], c="red", lw=3, label=r"$\widehat{ANPP}(Sens)$");


linReg_slope = regression_df[regression_df.fid==a_fid]["linReg_slope"].item()
linReg_intercept = regression_df[regression_df.fid==a_fid]["linReg_intercept"].item()

initial_x = df["year"].min()
initial_y = linReg_intercept + df["year"].min() * linReg_slope
ax1.axline(xy1=(initial_x, initial_y), slope=linReg_slope, 
           c='k', ls="--", lw=1.5, label='lin. reg. line') # slategrey
###############################################################################################
ax2.plot(df["year"], df['anpp_detrendSens'], c="red", lw=3, label="Sen's detrended");
ax2.plot(df["year"], df['anpp_detrendLinReg'], c="k", ls="--", lw=1.5, label="Lin. Reg. detrended");
ax2.plot(df["year"], df['anpp_detrendDiff'], c="y", lw=3, label="first-diff");

ax1.set_xlim(df["year"].min()-1, df["year"].max()+1)
ax1.legend(loc='best'); ax2.legend(loc='best');

fig.text(0.04, 0.5, r'$\mu_{NPP}$ (lb/acr)', va='center', rotation='vertical', fontdict=fontdict_normal);
ax2.set_xlabel('year', fontdict=fontdict_normal);
ax2.set_xticks(df['year'].iloc[::2]);
ax2.tick_params(axis='x', rotation=45)

state_ = Albers_SF_west[Albers_SF_west.fid==a_fid]["state_majority_area"].item()
ax1.set_title(f"FID: {a_fid} ({state_})", fontdict={'family': 'serif', 'weight': 'bold'})

file_name = bio_plots + f"fid_{a_fid}_anpp_dtrend.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=dpi_)

# %%
ANPP.head(2)

# %%
unique_FIDs = list(ANPP.fid.unique())

# %%
ANPP[ANPP['fid'] == min_senSlope_fid]['sens_slope'].unique()

# %%
a_fid = unique_FIDs[15200] # 1527 # 25863 1527 282 min_senSlope_fid
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=False, dpi=dpi_);
(ax1, ax2) = axes
ax1.grid(axis='y', alpha=0.7, zorder=0); ax2.grid(axis='y', alpha=0.7, zorder=0);

df = ANPP[ANPP.fid == a_fid]
interc_ = df['sens_intercept'].unique().item()
senSlope_ = slope=df['sens_slope'].unique().item()
ax1.plot(df["year"], df['mean_lb_per_acr'], c="dodgerblue", lw=3, label="ANPP");
ax1.plot(df["year"], df['anpp_senPred'], c="red", lw=3, label=r"$\widehat{ANPP}(Sens)$");

linReg_slope = regression_df[regression_df.fid==a_fid]["linReg_slope"].item()
linReg_intercept = regression_df[regression_df.fid==a_fid]["linReg_intercept"].item()

initial_x = df["year"].min()
initial_y = linReg_intercept + df["year"].min() * linReg_slope
ax1.axline(xy1=(initial_x, initial_y), slope=linReg_slope, c='k', ls="--", lw=1.5, label='lin. reg. line')
###############################################################################################
ax2.plot(df["year"], df['anpp_detrendSens'], c="red", lw=3, label="Sen's detrended");
ax2.plot(df["year"], df['anpp_detrendLinReg'], c="k", ls="--", lw=1.5, label="Lin. Reg. detrended");
ax2.plot(df["year"], df['anpp_detrendDiff'], c="y", lw=3, label="first-diff");

ax1.set_xlim(df["year"].min()-1, df["year"].max()+1)
ax1.legend(loc='best'); ax2.legend(loc='best');

fig.text(0.04, 0.5, r'$\mu_{NPP}$ (lb/acr)', va='center', rotation='vertical', fontdict=fontdict_normal);
ax2.set_xlabel('year', fontdict=fontdict_normal);
ax2.set_xticks(df['year'].iloc[::2]);
ax2.tick_params(axis='x', rotation=45)

state_ = Albers_SF_west[Albers_SF_west.fid==a_fid]["state_majority_area"].item()
ax1.set_title(f"FID: {a_fid} ({state_})", fontdict={'family': 'serif', 'weight': 'bold'})

file_name = bio_plots + f"fid_{a_fid}_anpp_dtrend.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=dpi_)

# %%

# %%

# %%

# %%
filename = bio_reOrganized + f"bpszone_ANPP_no2012_detrended.sav"

export_ = {'ANPP_no2012_detrended': ANPP,
           "source_code": "deTrend_ANPP",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),}

pickle.dump(export_, open(filename, "wb"))

# %%
