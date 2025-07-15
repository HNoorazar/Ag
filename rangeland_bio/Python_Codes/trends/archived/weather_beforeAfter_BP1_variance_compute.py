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
weather_detrended.rename(columns={"avg_of_dailyAvgTemp_C": "temp",
                                 "precip_mm":"prec"}, 
                         inplace=True)

# %%
ANPP_breaks = pd.read_csv(breakpoints_dir + "ANPP_break_points.csv")
ANPP_breaks = ANPP_breaks[ANPP_breaks["breakpoint_count"]>0]
ANPP_breaks.reset_index(drop=True, inplace=True)
ANPP_breaks.head(5)

# %%
bp_cols = ANPP_breaks['breakpoint_years'].str.split('_', expand=True)
bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
ANPP_breaks = pd.concat([ANPP_breaks, bp_cols], axis=1)
ANPP_breaks.head(2)

# %%
print (ANPP_breaks.shape)
ANPP_breaks['BP_1'] = ANPP_breaks['BP_1'].dropna().astype(int)
print (ANPP_breaks.shape)

# %%
temp_ys = ["temp", "temp_detrendLinReg", "temp_detrendDiff", "temp_detrendSens"]
prec_ys = ["prec", "prec_detrendLinReg", "prec_detrendDiff", "prec_detrendSens"]

cols_ = ["fid", "year"] + temp_ys + prec_ys
weather_detrended = weather_detrended[cols_]

# %%
ANPP_breaks.head(2)

# %%
unique_fids_w_BP = list(ANPP_breaks['fid'].unique())
weather_detrended = weather_detrended[weather_detrended['fid'].isin(unique_fids_w_BP)]
df_merged = pd.merge(weather_detrended, ANPP_breaks[['fid', 'BP_1']], on='fid', how='left')
df_merged.head(2)


# %%
# %%time
# Define a function to calculate variance before and after the breakpoint year
def calculate_variance(group, y_col):
    breakpoint_year = group['BP_1'].iloc[0]  # Get the BP_1 year for the group
    
    # Split the data into before and after the breakpoint year
    before_bp = group[group['year'] < breakpoint_year]
    after_bp = group[group['year'] >= breakpoint_year]
    
    # Calculate variances
    variance_before = before_bp[y_col].var() if not before_bp.empty else None
    variance_after = after_bp[y_col].var() if not after_bp.empty else None
    
    return pd.Series({f'variance_before_{y_col}': variance_before, 
                      f'variance_after_{y_col}': variance_after})


# %%
# %%time

# Apply the function to each group of 'fid'
y_ = temp_ys[0]
result_temp = df_merged.groupby('fid').apply(calculate_variance, y_col=y_).reset_index()

y_ = temp_ys[1]
result_temp_detrendLinReg = df_merged.groupby('fid').apply(calculate_variance, y_col=y_).reset_index()

y_ = temp_ys[2]
result_temp_detrendDiff = df_merged.groupby('fid').apply(calculate_variance, y_col=y_).reset_index()

y_ = temp_ys[3]
result_temp_detrendSens = df_merged.groupby('fid').apply(calculate_variance, y_col=y_).reset_index()

# %%
# %%time

# Apply the function to each group of 'fid'
y_ = prec_ys[0]
result_prec = df_merged.groupby('fid').apply(calculate_variance, y_col=y_).reset_index()

y_ = prec_ys[1]
result_prec_detrendLinReg = df_merged.groupby('fid').apply(calculate_variance, y_col=y_).reset_index()

y_ = prec_ys[2]
result_prec_detrendDiff = df_merged.groupby('fid').apply(calculate_variance, y_col=y_).reset_index()

y_ = prec_ys[3]
result_aprec_detrendSens = df_merged.groupby('fid').apply(calculate_variance, y_col=y_).reset_index()

# %%
from functools import reduce

# %%
df_list = [result_temp, result_temp_detrendLinReg, result_temp_detrendDiff, result_temp_detrendSens]
merged_var_diffs_temp = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

df_list = [result_prec, result_prec_detrendLinReg, result_prec_detrendDiff, result_aprec_detrendSens]
merged_var_diffs_prec = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

# %%
merged_var_diffs = pd.merge(merged_var_diffs_temp, merged_var_diffs_prec, on='fid', how='left')
merged_var_diffs.head(2)

# %%
filename = breakpoints_dir + "weather_variances_beforeAfter_BP1.sav"

export_ = {
    "weather_variances_beforeAfter_BP1": merged_var_diffs,
    "source_code": "weather_beforeAfter_BP1_variance_compute",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, 'wb'))

# %%
