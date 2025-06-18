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
# In this notebook we use breakpoints of ANPP and we compute Sen's slope before and after that breakpoint for weather variables.

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

breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/"
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
weather = pd.read_csv(bio_reOrganized + "bpszone_annual_tempPrecip_byHN.csv")
weather.head(2)

# %%
ANPP_breaks = pd.read_csv(breakpoints_dir + "ANPP_break_points.csv")
ANPP_breaks = ANPP_breaks[ANPP_breaks["breakpoint_count"]>0]
ANPP_breaks.reset_index(drop=True, inplace=True)
ANPP_breaks.head(2)

# %%
weather_breaks = pd.read_csv(breakpoints_dir + "weather_break_points.csv")
temp_breaks = weather_breaks[["fid", "temp_breakpoint_count", "temp_breakpoint_years"]].copy()
precip_breaks = weather_breaks[["fid", "precip_breakpoint_count", "precip_breakpoint_years"]].copy()

temp_breaks = temp_breaks[temp_breaks["temp_breakpoint_count"]>0]
precip_breaks = precip_breaks[precip_breaks["precip_breakpoint_count"]>0]

temp_breaks.sort_values("fid", inplace=True)
precip_breaks.sort_values("fid", inplace=True)

temp_breaks.reset_index(drop=True, inplace=True)
precip_breaks.reset_index(drop=True, inplace=True)

temp_breaks.head(2)

# %%
precip_breaks.head(2)

# %%
bp_cols = ANPP_breaks['breakpoint_years'].str.split('_', expand=True)
bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
ANPP_breaks = pd.concat([ANPP_breaks, bp_cols], axis=1)
ANPP_breaks.head(2)

# %%
bp_cols = temp_breaks['temp_breakpoint_years'].str.split('_', expand=True)
bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
temp_breaks = pd.concat([temp_breaks, bp_cols], axis=1)
temp_breaks.head(2)

# %%
bp_cols = precip_breaks['precip_breakpoint_years'].str.split('_', expand=True)
bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
precip_breaks = pd.concat([precip_breaks, bp_cols], axis=1)
precip_breaks.head(2)

# %%

# %%
# ANPP['year'] = ANPP['year'].astype(int)

# %%
print (ANPP_breaks.shape)
ANPP_breaks['BP_1'] = ANPP_breaks['BP_1'].dropna().astype(int)
print (ANPP_breaks.shape)

# %%
weather.head(2)

# %%
# %%time
# Iterate through each row in ANPP_breaks
results = []

for _, row in ANPP_breaks.iterrows():
    fid = row['fid']
    bp_year = row['BP_1']
    
    # Filter ANPP by current fid
    subset = weather[weather['fid'] == fid]
    
    # Separate before and after BP_1
    temp_before = subset[subset['year'] < bp_year]['avg_of_dailyAvgTemp_C']
    temp_after  = subset[subset['year'] >= bp_year]['avg_of_dailyAvgTemp_C']
    
    precip_before = subset[subset['year'] < bp_year]['precip_mm']
    precip_after  = subset[subset['year'] >= bp_year]['precip_mm']
    
    # Apply Mann-Kendall test if sufficient data
    result = {'fid': fid, 
              'BP_1': bp_year, 
              'n_before': len(temp_before),
              'n_after': len(temp_after),
              'temp_slope_before': None,
              'temp_slope_after': None,
              'temp_intercept_before': None,
              'temp_intercept_after': None,
              'temp_trend_before': None,
              'temp_trend_after': None,
              'precip_slope_before': None,
              'precip_slope_after': None,
              'precip_intercept_before': None,
              'precip_intercept_after': None,
              'precip_trend_before': None,
              'precip_trend_after': None
             }
    
    if len(temp_before) >= 3:
        trend, _, _, _, _, _, _, slope, intercept = mk.original_test(temp_before)
        result['temp_slope_before'] = slope.round(2)
        result['temp_trend_before'] = trend
        result['temp_intercept_before'] = intercept.round(2)
        
        trend, _, _, _, _, _, _, slope, intercept = mk.original_test(precip_before)
        result['precip_slope_before'] = slope.round(2)
        result['precip_trend_before'] = trend
        result['precip_intercept_before'] = intercept.round(2)
    
    if len(temp_after) >= 3:
        trend, _, _, _, _, _, _, slope, intercept = mk.original_test(temp_after)
        result['temp_slope_after'] = slope.round(2)
        result['temp_trend_after'] = trend
        result['temp_intercept_after'] = intercept.round(2)
        
        trend, _, _, _, _, _, _, slope, intercept = mk.original_test(precip_after)
        result['precip_slope_after'] = slope.round(2)
        result['precip_trend_after'] = trend
        result['precip_intercept_after'] = intercept.round(2)
    
    results.append(result)

# Create results DataFrame
slope_results = pd.DataFrame(results)

# %% [markdown]
# ### plot TS of FID=1

# %%

# %%
tick_legend_FontSize = 6
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 5,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          'axes.linewidth' : .05,
          'xtick.major.width': 0.1,
          'ytick.major.width': 0.1,
          'xtick.major.size': 2,
          'ytick.major.size': 2,
         }

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
# axes.axline((ress_["BP_1"].item(), ress_["intercept_after"].item()), 
#             slope=ress_["slope_after"].item(), 
#             color='k', label='after')


# %%

# %%
slope_results.head(2)

# %%

# %%
filename = breakpoints_dir + "weather_sensSlope_beforeAfter_ANPPBP1.sav"

export_ = {
    "sensSlope_beforeAfter_ANPPBP1": slope_results,
    "source_code": "breakpoints_WeatherSenSlopes_compute",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, 'wb'))

# %%
filename = breakpoints_dir + "sensSlope_beforeAfter_ANPPBP1.csv"
slope_results.to_csv(filename, index=False)

# %%

# %%

# %%
