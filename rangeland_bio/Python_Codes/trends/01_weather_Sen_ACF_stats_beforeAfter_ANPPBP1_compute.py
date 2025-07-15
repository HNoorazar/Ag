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
# In this notebook we use breakpoints of ANPP and we compute Sen's slope before and after that breakpoint for weather variables, as well as mean/median before and after of ANPP-BP1
#
# The reason this notebook's name is starting with ```01_``` is that we need to convert monthly data of Min to annual scale. That is done in the notebook called ```00_weather_monthly2Annual_and_40yearsMK.ipynb```.

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
dpi_, map_dpi_=300, 500
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
weather = pd.read_pickle(bio_reOrganized + "bpszone_annualWeatherByHN_and_deTrended.sav")
print (weather.keys())
print (weather['source_code'])
weather = weather['bpszone_annual_weather_byHN']
# weather.drop(columns=["state_1", "state_2", "state_majority_area", "EW_meridian"], inplace=True)
weather.head(2)

# %%
list(weather.columns)

# %%
ANPP_breaks = pd.read_csv(breakpoints_dir + "ANPP_break_points.csv")
ANPP_breaks = ANPP_breaks[ANPP_breaks["breakpoint_count"]>0]
ANPP_breaks.reset_index(drop=True, inplace=True)
ANPP_breaks.head(2)

# %%
# weather_breaks = pd.read_csv(breakpoints_dir + "weather_break_points.csv")
# temp_breaks = weather_breaks[["fid", "temp_breakpoint_count", "temp_breakpoint_years"]].copy()
# precip_breaks = weather_breaks[["fid", "precip_breakpoint_count", "precip_breakpoint_years"]].copy()

# temp_breaks = temp_breaks[temp_breaks["temp_breakpoint_count"]>0]
# precip_breaks = precip_breaks[precip_breaks["precip_breakpoint_count"]>0]

# temp_breaks.sort_values("fid", inplace=True)
# precip_breaks.sort_values("fid", inplace=True)

# temp_breaks.reset_index(drop=True, inplace=True)
# precip_breaks.reset_index(drop=True, inplace=True)

# temp_breaks.head(2)

# bp_cols = temp_breaks['temp_breakpoint_years'].str.split('_', expand=True)
# bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
# bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
# temp_breaks = pd.concat([temp_breaks, bp_cols], axis=1)
# temp_breaks.head(2)



# bp_cols = precip_breaks['precip_breakpoint_years'].str.split('_', expand=True)
# bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
# bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
# precip_breaks = pd.concat([precip_breaks, bp_cols], axis=1)
# precip_breaks.head(2)

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
weather.head(2)

# %%
static_columns = ['fid', 'year', 'state_majority_area', 'EW_meridian']
y_vars = [x for x in weather.columns if not (x in static_columns)]
y_vars

# %% [markdown]
# # Some FIDs have no breakpoints. Toss them

# %%
lag = 1

# %%
print (len(weather["fid"].unique()))
print (len(ANPP_breaks["fid"].unique()))

# %%
fids_ = list(ANPP_breaks['fid'].unique())
weather = weather[weather['fid'].isin(fids_)]


# %%
# %%time
# Iterate through each row in ANPP_breaks
results = []

for _, row in ANPP_breaks.iterrows():
    fid = row['fid']
    bp_year = row['BP_1']
    subset = weather[weather['fid'] == fid]
    a_fid_results = {}
    
    for y_var in y_vars:
        # Separate before and after BP_1
        before = subset[subset['year'] < bp_year][y_var]
        after  = subset[subset['year'] >= bp_year][y_var]

        # Apply Mann-Kendall test if sufficient data
        result = {'fid': fid, 
                  'BP_1': bp_year, 
                  'n_before': len(before),
                  'n_after': len(after),
                  f'{y_var}_slope_before': None,
                  f'{y_var}_slope_after': None,
                  f'{y_var}_intercept_before': None,
                  f'{y_var}_intercept_after': None,
                  f'{y_var}_trend_before': None,
                  f'{y_var}_trend_after': None,
                  f'{y_var}_mean_before': None,
                  f'{y_var}_mean_after': None,
                  f'{y_var}_median_before': None,
                  f'{y_var}_median_after': None,
                  f'{y_var}_variance_before': None,
                  f'{y_var}_variance_after': None,
                  f'{y_var}_ACF1_before': None,
                  f'{y_var}_ACF1_after': None,
                 }
        # Why 3? is 2 enough?
        # We can count the number of cases that we had 2
        if len(before) >= 3:
            trend, _, _, _, _, _, _, slope, intercept = mk.original_test(before)
            result[f'{y_var}_trend_before'] = trend
            result[f'{y_var}_slope_before'] = slope.round(2)
            result[f'{y_var}_intercept_before'] = intercept.round(2)

        if len(after) >= 3:
            trend, _, _, _, _, _, _, slope, intercept = mk.original_test(after)
            result[f'{y_var}_trend_after'] = trend
            result[f'{y_var}_slope_after'] = slope.round(2)
            result[f'{y_var}_intercept_after'] = intercept.round(2)

        #########  Mean. Median. Variance.
        if len(before) >= 1:
            result[f'{y_var}_mean_before'] = before.mean()
            result[f'{y_var}_median_before'] = before.median()
            result[f'{y_var}_variance_before'] = before.var()

        if len(after) >= 1:
            result[f'{y_var}_mean_after'] = after.mean()
            result[f'{y_var}_median_after'] = after.median()
            result[f'{y_var}_variance_after'] = after.var()
            
        autocorr = before.autocorr(lag=lag) if before.nunique() > 1 else np.nan
        result[f'{y_var}_ACF1_before'] = autocorr
        
        autocorr = after.autocorr(lag=lag) if after.nunique() > 1 else np.nan
        result[f'{y_var}_ACF1_after'] = autocorr
    
        a_fid_results.update(result)

    results.append(a_fid_results)

# Create results DataFrame
slope_results = pd.DataFrame(results)
slope_results.head(3)

# %%
print (slope_results.shape)

# %%
list(slope_results.columns)

# %%

# %% [markdown]
# # Compute variance before and after ANPP-BP1

# %%
# # %%time

# mean_median_variance_dict = []

# for y_var in y_vars:
#     for stat_ in ["mean", "median", "variance"]:
#         result_temp = weather.groupby('fid').apply(rc.calculate_stat_beforeAfterBP, 
#                                                    y_col=y_var, stat=stat_).reset_index()
#         mean_median_variance_dict.append(result_temp)

# %%
slope_results.head(2)

# %% [markdown]
# ## Compute differences and ratios here

# %%
weather_ANPPBP1 = slope_results.copy()

# %%
stats_tuple_ = ("slope", "mean", "median", "variance", "ACF1")
y_cols = [x for x in weather_ANPPBP1.columns if any(k in x for k in stats_tuple_)]
y_cols[:4]

# %%
## remove before and after to get patterns
y_cols_patterns = [x.replace('before', '').replace('after', '') for x in y_cols]
y_cols_patterns = list(set(y_cols_patterns))
y_cols_patterns[:4]

# %%
for a_pattern in y_cols_patterns:
    weather_ANPPBP1[f'{a_pattern}diff'] = weather_ANPPBP1[f"{a_pattern}after"] - \
                                                 weather_ANPPBP1[f"{a_pattern}before"]
    
    weather_ANPPBP1[f"{a_pattern}ratio"] = weather_ANPPBP1[f"{a_pattern}after"] / \
                                                  weather_ANPPBP1[f"{a_pattern}before"]
weather_ANPPBP1.head(2)

# %%
weather_ANPPBP1.shape

# %% [markdown]
# ## Separate the diff/ratio DF from the actual values

# %%
diff_ratio_cols = [x for x in weather_ANPPBP1 if ("diff" in x) or ("ratio" in x)]
keep_cols = ["fid", "BP_1", "n_before", "n_after"] + diff_ratio_cols

# %%
weather_ANPPBP1 = weather_ANPPBP1[keep_cols]
weather_ANPPBP1.shape

# %%

# %%
filename = breakpoints_dir + "01_weather_Sen_ACF_stats_beforeAfter_ANPPBP1.sav"

export_ = {
    "sensSlope_stats_ACF_beforeAfter_ANPPBP1": slope_results,
    "weather_diffsRatios": weather_ANPPBP1,
    "source_code": "01_weather_Sen_ACF_stats_beforeAfter_ANPPBP1_compute",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, 'wb'))

# filename = breakpoints_dir + "weather_sensSlope_beforeAfter_ANPPBP1.csv"
# slope_results.to_csv(filename, index=False)

# %%

# %%

# %%

# %%
# # %%time
# # Iterate through each row in ANPP_breaks
# all_results = []

# for _, row in ANPP_breaks.iterrows():
#     fid = row['fid']
#     bp_year = row['BP_1']
    
#     # Filter ANPP by current fid
#     subset = weather[weather['fid'] == fid]
    
#     # Separate before and after BP_1
#     temp_before = subset[subset['year'] < bp_year]['avg_of_dailyAvgTemp_C']
#     temp_after  = subset[subset['year'] >= bp_year]['avg_of_dailyAvgTemp_C']
    
#     precip_before = subset[subset['year'] < bp_year]['precip_mm']
#     precip_after  = subset[subset['year'] >= bp_year]['precip_mm']
    
#     # Apply Mann-Kendall test if sufficient data
#     result = {'fid': fid, 
#               'BP_1': bp_year, 
#               'n_before': len(temp_before),
#               'n_after': len(temp_after),
#               'temp_slope_before': None,
#               'temp_slope_after': None,
#               'temp_intercept_before': None,
#               'temp_intercept_after': None,
#               'temp_trend_before': None,
#               'temp_trend_after': None,
#               'precip_slope_before': None,
#               'precip_slope_after': None,
#               'precip_intercept_before': None,
#               'precip_intercept_after': None,
#               'precip_trend_before': None,
#               'precip_trend_after': None
#              }
    
#     if len(temp_before) >= 3:
#         trend, _, _, _, _, _, _, slope, intercept = mk.original_test(temp_before)
#         result['temp_slope_before'] = slope.round(2)
#         result['temp_trend_before'] = trend
#         result['temp_intercept_before'] = intercept.round(2)
        
#         trend, _, _, _, _, _, _, slope, intercept = mk.original_test(precip_before)
#         result['precip_slope_before'] = slope.round(2)
#         result['precip_trend_before'] = trend
#         result['precip_intercept_before'] = intercept.round(2)
    
#     if len(temp_after) >= 3:
#         trend, _, _, _, _, _, _, slope, intercept = mk.original_test(temp_after)
#         result['temp_slope_after'] = slope.round(2)
#         result['temp_trend_after'] = trend
#         result['temp_intercept_after'] = intercept.round(2)
        
#         trend, _, _, _, _, _, _, slope, intercept = mk.original_test(precip_after)
#         result['precip_slope_after'] = slope.round(2)
#         result['precip_trend_after'] = trend
#         result['precip_intercept_after'] = intercept.round(2)
    
#     all_results.append(result)

# # Create results DataFrame
# MK_results = pd.DataFrame(all_results)
# MK_results.head(2)
