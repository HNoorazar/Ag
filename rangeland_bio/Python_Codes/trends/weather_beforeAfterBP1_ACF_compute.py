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
custom_cmap_GrayW = ListedColormap(['grey', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

fontdict_normal = {'family':'serif', 'weight':'normal'}
fontdict_bold   = {'family':'serif', 'weight':'bold'}
fontdict_bold_sup= {'family':'serif', 'fontweight':'bold'}
inset_axes_     = [0.1, 0.14, 0.45, 0.03]

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
filename = bio_reOrganized + f"weather_detrended.sav"

weather_detrended = pd.read_pickle(filename)
weather_detrended.keys()
weather_detrended = weather_detrended["weather_detrended"]
weather_detrended.head(2)

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
weather_detrended.rename(columns={"avg_of_dailyAvgTemp_C": "temp",
                                 "precip_mm":"prec"}, 
                         inplace=True)

# %%
lag = 1

temp_ys = ["temp", "temp_detrendLinReg", "temp_detrendDiff", "temp_detrendSens"]
prec_ys = ["prec", "prec_detrendLinReg", "prec_detrendDiff", "prec_detrendSens"]

cols_ = ["fid", "year"] + temp_ys + prec_ys
weather_detrended = weather_detrended[cols_]

# %%

# %%
# %%time

unique_fids = ANPP_breaks['fid'].unique()
# Initialize empty DataFrame with desired columns
ACFs_df_temp = pd.DataFrame({'fid': unique_fids,
                             'BP1' : np.nan,
                             'n_before': np.nan,
                             'n_after': np.nan,
                             'ACF_before_temp': np.nan,
                             'ACF_after_temp': np.nan,
                             
                             'ACF_before_temp_detrendLinReg': np.nan,
                             'ACF_after_temp_detrendLinReg': np.nan,
                       
                             'ACF_before_temp_detrendDiff': np.nan,
                             'ACF_after_temp_detrendDiff': np.nan,
                       
                             'ACF_before_temp_detrendSens': np.nan,
                             'ACF_after_temp_detrendSens': np.nan})


ACFs_df_temp = ACFs_df_temp.set_index('fid')
ACFs_df_temp.head(2)

# y_var = "temp"
for _, row in ANPP_breaks.iterrows():
    fid = row['fid']
    bp_year = row['BP_1']
    
    ACFs_df_temp.loc[fid, 'BP1'] = bp_year
    
    # Filter by current fid
    subset = weather_detrended[weather_detrended['fid'] == fid]
    for y_var in temp_ys:
        # Separate before and after BP_1
        before = subset[subset['year'] < bp_year][y_var].dropna()
        after = subset[subset['year'] >= bp_year][y_var].dropna()
        
        ACFs_df_temp.loc[fid, 'n_before'] = len(before)
        ACFs_df_temp.loc[fid, 'n_after'] = len(after)

        autocorr = before.autocorr(lag=lag) if before.nunique() > 1 else np.nan
        ACFs_df_temp.loc[fid, 'ACF_before_' + y_var] = autocorr

        autocorr = after.autocorr(lag=lag) if after.nunique() > 1 else np.nan
        ACFs_df_temp.loc[fid, 'ACF_after_' + y_var] = autocorr

ACFs_df_temp.reset_index(drop=False, inplace=True)
ACFs_df_temp.head(3)

# %%

# %%
# %%time

unique_fids = ANPP_breaks['fid'].unique()
# Initialize empty DataFrame with desired columns
ACFs_df_prec = pd.DataFrame({'fid': unique_fids,
                             'ACF_before_prec': np.nan,
                             'ACF_after_prec': np.nan,
 
                             'ACF_before_prec_detrendLinReg': np.nan,
                             'ACF_after_prec_detrendLinReg': np.nan,

                             'ACF_before_prec_detrendDiff': np.nan,
                             'ACF_after_prec_detrendDiff': np.nan,
 
                             'ACF_before_prec_detrendSens': np.nan,
                             'ACF_after_prec_detrendSens': np.nan})


ACFs_df_prec = ACFs_df_prec.set_index('fid')
ACFs_df_prec.head(2)

# y_var = "prec"
for _, row in ANPP_breaks.iterrows():
    fid = row['fid']
    bp_year = row['BP_1']
    
    ACFs_df_prec.loc[fid, 'BP1'] = bp_year
    
    # Filter by current fid
    subset = weather_detrended[weather_detrended['fid'] == fid]
    for y_var in prec_ys:
        # Separate before and after BP_1
        before = subset[subset['year'] < bp_year][y_var].dropna()
        after = subset[subset['year'] >= bp_year][y_var].dropna()
        
        ACFs_df_prec.loc[fid, 'n_before'] = len(before)
        ACFs_df_prec.loc[fid, 'n_after'] = len(after)

        autocorr = before.autocorr(lag=lag) if before.nunique() > 1 else np.nan
        ACFs_df_prec.loc[fid, 'ACF_before_' + y_var] = autocorr

        autocorr = after.autocorr(lag=lag) if after.nunique() > 1 else np.nan
        ACFs_df_prec.loc[fid, 'ACF_after_' + y_var] = autocorr

ACFs_df_prec.reset_index(drop=False, inplace=True)
ACFs_df_prec.head(3)

# %%
ACFs_df = pd.merge(ACFs_df_temp, ACFs_df_prec, on='fid', how='left')
ACFs_df.head(2)

# %%
filename = breakpoints_dir + "weather_ACFs_beforeAfter_BP1.sav"

export_ = {
    "weather_ACFs_beforeAfter_BP1": ACFs_df,
    "source_code": "weather_beforeAfterBP1_ACF_compute",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, 'wb'))

# %%
