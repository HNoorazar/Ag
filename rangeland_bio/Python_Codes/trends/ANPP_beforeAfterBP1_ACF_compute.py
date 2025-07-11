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
import importlib;
importlib.reload(rc);
importlib.reload(rpc);

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

breakpoint_TS_dir = bio_plots + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)


G_breakpoint_TS_dir = breakpoint_TS_dir + "/greening/"
B_breakpoint_TS_dir = breakpoint_TS_dir + "/browning/"
noTrend_breakpoint_TS_dir = breakpoint_TS_dir + "/notrend/"

os.makedirs(G_breakpoint_TS_dir, exist_ok=True)
os.makedirs(B_breakpoint_TS_dir, exist_ok=True)
os.makedirs(noTrend_breakpoint_TS_dir, exist_ok=True)

# %%
breakpoint_TS_sen_dir = breakpoint_plot_base + "breakpoints_TS_sensSlope/"
os.makedirs(breakpoint_TS_sen_dir, exist_ok=True)


# %%
breakpoints_dir = rangeland_bio_data + "breakpoints/"

# %%
# ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
# ANPP = ANPP["bpszone_ANPP"]
# ANPP.head(2)

# %%

# %%
filename = bio_reOrganized + f"bpszone_ANPP_no2012_detrended.sav"

ANPP_no2012_detrended = pd.read_pickle(filename)
ANPP_no2012_detrended = ANPP_no2012_detrended["ANPP_no2012_detrended"]
ANPP_no2012_detrended.head(2)

# %%
ANPP_breaks = pd.read_csv(breakpoints_dir + "ANPP_break_points.csv")
ANPP_breaks = ANPP_breaks[ANPP_breaks["breakpoint_count"]>0]
ANPP_breaks.reset_index(drop=True, inplace=True)
ANPP_breaks.head(5)

# %%
ANPP_breaks.tail(5)

# %%
bp_cols = ANPP_breaks['breakpoint_years'].str.split('_', expand=True)
bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
ANPP_breaks = pd.concat([ANPP_breaks, bp_cols], axis=1)
ANPP_breaks.head(2)

# %%

# %%
print (ANPP_breaks.shape)
ANPP_breaks['BP_1'] = ANPP_breaks['BP_1'].dropna().astype(int)
print (ANPP_breaks.shape)

# %%
lag = 1
ys = ["mean_lb_per_acr", "anpp_detrendLinReg", "anpp_detrendDiff", "anpp_detrendSens"]

# %%

# %%
# %%time

unique_fids = ANPP_breaks['fid'].unique()
# Initialize empty DataFrame with desired columns
ACFs_df = pd.DataFrame({'fid': unique_fids,
                        'BP1' : np.nan,
                        'n_before': np.nan,
                        'n_after': np.nan,
                        'ACF_before_mean_lb_per_acr': np.nan,
                        'ACF_after_mean_lb_per_acr': np.nan,
                       
                        'ACF_before_anpp_detrendLinReg': np.nan,
                        'ACF_after_anpp_detrendLinReg': np.nan,
                       
                        'ACF_before_anpp_detrendDiff': np.nan,
                        'ACF_after_anpp_detrendDiff': np.nan,
                       
                        'ACF_before_anpp_detrendSens': np.nan,
                        'ACF_after_anpp_detrendSens': np.nan})


ACFs_df = ACFs_df.set_index('fid')
ACFs_df.head(2)

# y_var = "mean_lb_per_acr"
for _, row in ANPP_breaks.iterrows():
    fid = row['fid']
    bp_year = row['BP_1']
    
    ACFs_df.loc[fid, 'BP1'] = bp_year
    
    # Filter ANPP by current fid
    subset = ANPP_no2012_detrended[ANPP_no2012_detrended['fid'] == fid]
    for y_var in ys:
        # Separate before and after BP_1
        before = subset[subset['year'] < bp_year][y_var].dropna()
        after = subset[subset['year'] >= bp_year][y_var].dropna()
        
        ACFs_df.loc[fid, 'n_before'] = len(before)
        ACFs_df.loc[fid, 'n_after'] = len(after)

        autocorr = before.autocorr(lag=lag) if before.nunique() > 1 else np.nan
        ACFs_df.loc[fid, 'ACF_before_' + y_var] = autocorr

        autocorr = after.autocorr(lag=lag) if after.nunique() > 1 else np.nan
        ACFs_df.loc[fid, 'ACF_after_' + y_var] = autocorr

ACFs_df.reset_index(drop=False, inplace=True)
ACFs_df.head(3)

# %%

# %%
filename = breakpoints_dir + "ACFs_beforeAfter_BP1.sav"

export_ = {
    "ACFs_beforeAfter_BP1": ACFs_df,
    "source_code": "breakpoints_ACF_compute",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, 'wb'))

# %%
