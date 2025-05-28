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
ACF_plot_base = bio_plots + "ACF1/"
os.makedirs(ACF_plot_base, exist_ok=True)

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
ANPP_ACF1 = ANPP.groupby('fid')['mean_lb_per_acr'].apply(lambda x: x.autocorr(lag=1))
ANPP_ACF1 = ANPP_ACF1.reset_index(name='mean_lb_per_acr_lag1_autocorr')
ANPP_ACF1.head(3)

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

# %%

# %%

# %%
# since 2012 is missing. we have to do sth about it: use right pointer!
for a_fid in ANPP_FIDs:
    curr_fid = ANPP[ANPP["fid"] == a_fid]
    curr_yrs = curr_fid["year"].unique()
    len_curr_yrs = len(curr_yrs)
    left_pointer = 0
    
    curr_df_window = curr_fid[curr_fid["year"]]
    while right_pointer

# %%
curr_fid["year"].unique()

# %%

# %%
