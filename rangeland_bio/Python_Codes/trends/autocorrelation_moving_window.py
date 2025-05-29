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
#
#
# # <span style="color:red">Moved to Kamiak.</span>

# %%
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os, os.path, pickle, sys
import pymannkendall as mk

from scipy import stats
import scipy.stats as scipy_stats
from statsmodels.tsa.stattools import acf
import geopandas

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc

# %%
dpi_, map_dpi_=300, 900
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

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

# %%

# %%
ACF_data = rangeland_bio_data + "ACF1/"
os.makedirs(ACF_data, exist_ok=True)

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

# %%
print (ANPP.year.min())
print (ANPP.year.max())

# %%
# Example usage:
data = [10, 12, 15, 13, 16, 18, 17]
time_series = pd.Series(data)

lag_1_acf = time_series.autocorr(lag=1)
print(f"Lag-1 Autocorrelation: {lag_1_acf:.2f}")

# %%
# ANPP_ACF1 = ANPP.groupby('fid')['mean_lb_per_acr'].apply(lambda x: x.autocorr(lag=1))
# ANPP_ACF1 = ANPP_ACF1.reset_index(name='mean_lb_per_acr_lag1_autocorr')
# ANPP_ACF1.head(3)

# %%
ANPP_years = list(ANPP["year"].unique())
ANPP_FIDs = list(ANPP["fid"].unique())
window_sizes = np.arange(5, 11)

# %%
window_size = 5

# %%
## check if all FIDs have all data
## that way we can use ANPP.groupby('fid')['mean_lb_per_acr'].apply(lambda x: x.autocorr(lag=1))
## to go faster
for a_fid in ANPP_FIDs:
    curr_fid = ANPP[ANPP["fid"] == a_fid]
    if len(curr_fid["year"]) != len(ANPP_years):
        print (a_fid)

# %%
ANPP.head(2)

# %%

# %%
df = ANPP[ANPP["fid"] == 1]
df.groupby('fid')['mean_lb_per_acr'].apply(lambda x: x.autocorr(lag=1))

# %%
df["mean_lb_per_acr"].autocorr(lag=1)

# %%
acf(df["mean_lb_per_acr"].values, nlags=1, fft=False, adjusted=True)

# %%
acf(df["mean_lb_per_acr"].values, nlags=1, fft=False, adjusted=False)

# %%
acf(df["mean_lb_per_acr"].values, nlags=1, fft=False)

# %% [markdown]
# # Moved to Kamiak

# %%
import importlib;
importlib.reload(rc);
importlib.reload(rpc);

# %%
# %%time
all_ACF1s_2 = {}
for ws in np.arange(5, 11):
    ACF1s_window = rc.rolling_autocorr_df_prealloc(df=ANPP, y_var="mean_lb_per_acr", window_size=ws, lag=1)
    all_ACF1s_2[f"autocorrs_ws{ws}"] = ACF1s_window
    print (ws)

# %%
ACF_data = rangeland_bio_data + "ACF1/"
os.makedirs(ACF_data, exist_ok=True)

# %%
# filename = ACF_data + "rolling_autocorrelations.sav"

# export_ = {
#     "rolling_autocorrelations": all_ACF1s_2,
#     "source_code": "autocorrelation_moving_window",
#     "Author": "HN",
#     "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# }

# pickle.dump(export_, open(filename, 'wb'))

# %%

# %%
# # %%time

# all_ACF1s = {}
# for ws in np.arange(5, 11):
#     ACF1s_window = rc.rolling_autocorr(df=ANPP, y_var="mean_lb_per_acr", window_size=ws, lag=1)
#     all_ACF1s[f"autocorrs_ws{ws}"] = ACF1s_window
