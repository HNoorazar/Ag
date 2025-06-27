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

breakpoint_TS_dir = bio_plots + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)


G_breakpoint_TS_dir = breakpoint_TS_dir + "/greening/"
B_breakpoint_TS_dir = breakpoint_TS_dir + "/browning/"
noTrend_breakpoint_TS_dir = breakpoint_TS_dir + "/notrend/"

os.makedirs(G_breakpoint_TS_dir, exist_ok=True)
os.makedirs(B_breakpoint_TS_dir, exist_ok=True)
os.makedirs(noTrend_breakpoint_TS_dir, exist_ok=True)

# %%
weather = pd.read_csv(bio_reOrganized + "bpszone_annual_tempPrecip_byHN.csv")
weather.head(2)

# %%
weather.drop(columns = ["state_1", "state_2", "avg_of_dailyAvg_rel_hum", "thi_avg"], inplace=True)

# %% [markdown]
# ### Get the Sen's slopes for detrending

# %%
# %%time
f_name = bio_reOrganized + 'Albers_SF_west_weather_MK_Spearman.shp.zip'
Albers_SF_west = geopandas.read_file(f_name)

Albers_SF_west.rename(columns={"EW_meridia" : "EW_meridian",
                               "state_majo" : "state_majority_area",
                               "intcp_prec" : "sens_intercept_prec",
                               "intcp_temp" : "sens_intercept_temp"},
                      inplace=True)

Albers_SF_west.head(2)

# %%

# %%
sorted(Albers_SF_west.columns)

# %%
# Merge the dataframes on 'fid'
weather = weather.merge(Albers_SF_west[['fid', 'sen_m_prec', 'sen_m_temp',
                                               'sens_intercept_prec', 'sens_intercept_temp']], 
                        on='fid', how='left')
weather.head(2)

# %%

# %% [markdown]
# ### Sens prediction 
#
# must not be based on year since that test only lookst at y values.
#

# %%
weather['row_number_perfid'] = weather.groupby('fid').cumcount()
weather.head(2)

# %%
weather["prec_senPred"] = weather["row_number_perfid"] * weather["sen_m_prec"] + weather["sens_intercept_prec"]
weather["temp_senPred"] = weather["row_number_perfid"] * weather["sen_m_temp"] + weather["sens_intercept_temp"]
weather.head(2)

# %%

# %%
weather["temp_detrendSens"] = weather["avg_of_dailyAvgTemp_C"] - weather["temp_senPred"]
weather["prec_detrendSens"] = weather["precip_mm"] - weather["prec_senPred"]
weather.head(2)

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
weather.head(2)

# %% [markdown]
# ## First difference detrending

# %%
weather['temp_detrendDiff'] = weather.groupby('fid')['avg_of_dailyAvgTemp_C'].diff()
weather['prec_detrendDiff'] = weather.groupby('fid')['precip_mm'].diff()
weather.head(2)

# %% [markdown]
# ## detrend using Simple Linear regression

# %%
from sklearn.linear_model import LinearRegression

# %%
# %%time

unique_fids = weather['fid'].unique()
# Initialize empty DataFrame with desired columns
regression_df = pd.DataFrame({'fid': unique_fids,
                              'prec_linReg_slope': np.nan,
                              'prec_linReg_intercept': np.nan,
                              'temp_linReg_slope': np.nan,
                              'temp_linReg_intercept': np.nan})
regression_df = regression_df.set_index('fid')


# Prepare a column to store detrended values
weather['prec_detrendLinReg'] = np.nan
weather['temp_detrendLinReg'] = np.nan

# Loop over each fid group
for fid, group in weather.groupby('fid'):
    ########
    ########     Temp
    ########
    # Reshape year for sklearn
    X = group['year'].values.reshape(-1, 1)
    y = group['avg_of_dailyAvgTemp_C'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    yhat = model.predict(X)
    weather.loc[group.index, 'temp_detrendLinReg'] = y - yhat

    # Optionally store slope/intercept
    regression_df.loc[fid, 'temp_linReg_slope'] = model.coef_[0]
    regression_df.loc[fid, 'temp_linReg_intercept'] = model.intercept_
    
    ########
    ########     precip
    ########
    # Reshape year for sklearn
    X = group['year'].values.reshape(-1, 1)
    y = group['precip_mm'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    yhat = model.predict(X)
    weather.loc[group.index, 'prec_detrendLinReg'] = y - yhat

    # Optionally store slope/intercept
    regression_df.loc[fid, 'prec_linReg_slope'] = model.coef_[0]
    regression_df.loc[fid, 'prec_linReg_intercept'] = model.intercept_
    
    
    
regression_df.reset_index(drop=False, inplace=True)

regression_df.head(2)

# %%
weather.head(2)

# %%
filename = bio_reOrganized + f"weather_detrended.sav"

export_ = {'weather_detrended': weather,
           "source_code": "deTrend_weather",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),}

pickle.dump(export_, open(filename, "wb"))

# %%
sorted(weather.columns)

# %%
