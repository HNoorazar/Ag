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
# There is another script called ```autocorr_moving_window_analysis_ANPP.ipynb```.
#
# This is a copy of that with modifications to do weather.
#
# **June 25, 2025**

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


sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc

# %%

# %%
dpi_, map_dpi_=300, 500
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
bio_plots = rangeland_bio_base + "plots/"
ACF_plot_base = bio_plots + "ACF1/"
os.makedirs(ACF_plot_base, exist_ok=True)

# %%
ACF_data = rangeland_bio_data + "ACF1/"

# %%

# %%
# %%time
## bad 2012
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
# SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %% [markdown]
# ## Read all rolling window ACFs

# %%
temp_ys = ["temp", "temp_detrendLinReg", "temp_detrendDiff", "temp_detrendSens"]
prec_ys = ["prec", "prec_detrendLinReg", "prec_detrendDiff", "prec_detrendSens"]

# %%
temp_ACF_dict = {}
for window_size in np.arange(5, 11):
    for y_ in temp_ys:
        key_ = f"rolling_autocorrelations_ws{window_size}_{y_}"
        f_name = ACF_data + key_ + ".sav"
        ACF_df = pd.read_pickle(f_name)
        ACF_df = ACF_df[key_]
        key_ = f"ACF1_ws{window_size}_{y_}"
        temp_ACF_dict[key_] = ACF_df

# %%

# %%
prec_ACF_dict = {}
for window_size in np.arange(5, 11):
    for y_ in prec_ys:
        key_ = f"rolling_autocorrelations_ws{window_size}_{y_}"
        f_name = ACF_data + key_ + ".sav"
        ACF_df = pd.read_pickle(f_name)
        ACF_df = ACF_df[key_]
        key_ = f"ACF1_ws{window_size}_{y_}"
        prec_ACF_dict[key_] = ACF_df

# %%
ACF_df = pd.read_pickle(f_name)
ACF_df.keys()

# %% [markdown]
# ### Find trends of ACF1 time-series via MK again!

# %%
print (len(list(temp_ACF_dict.keys())))
list(temp_ACF_dict.keys())

# %%
print (len(list(prec_ACF_dict.keys())))
list(prec_ACF_dict.keys())

# %%
temp_ACF_dict["ACF1_ws5_temp"].head(2)

# %%
temp_ACF_dict["ACF1_ws5_temp_detrendLinReg"].head(2)

# %%
prec_ACF_dict["ACF1_ws5_prec"].head(2)

# %%
import re

import importlib;
importlib.reload(rc);
importlib.reload(rpc);

# %%
# %%time

temp_ACF_trends_MK_dict = {}

for a_key in temp_ACF_dict.keys():
    ws = re.search(r'ws(\d+)', a_key).group(1)
    curr_df = temp_ACF_dict[a_key]
    curr_col = f'autocorr_lag1_ws{ws}'
    temp_ACF_trends_MK_dict[a_key] = rc.compute_mk_by_fid(df=curr_df, groupby_='fid', value_col=curr_col)
    del(a_key)

# %%
prec_ACF_dict["ACF1_ws5_prec"].head(2)

# %%
# %%time

prec_ACF_trends_MK_dict = {}

for a_key in prec_ACF_dict.keys():
    ws = re.search(r'ws(\d+)', a_key).group(1)
    curr_df = prec_ACF_dict[a_key]
    curr_col = f'autocorr_lag1_ws{ws}'
    prec_ACF_trends_MK_dict[a_key] = rc.compute_mk_by_fid(df=curr_df, groupby_='fid', value_col=curr_col)
    del(a_key)

# %%

# %%
temp_ACF_trends_MK_dict[list(temp_ACF_dict.keys())[0]].head(3)

# %%
prec_ACF_trends_MK_dict[list(prec_ACF_dict.keys())[0]].head(3)

# %% [markdown]
# ### Reduce/merge the data tables in the dictionaries

# %%
for key_ in temp_ACF_trends_MK_dict.keys():
    temp_ACF_trends_MK_dict[key_].rename(columns={"trend": f"trend_{key_}",
                                                  "p_value": f"p_value_{key_}",
                                                  "slope": f"slope_{key_}"}, 
                                                  inplace=True)
    del(key_)

# %%
for key_ in prec_ACF_trends_MK_dict.keys():
    prec_ACF_trends_MK_dict[key_].rename(columns={"trend": f"trend_{key_}",
                                                  "p_value": f"p_value_{key_}",
                                                  "slope": f"slope_{key_}"},
                                                  inplace=True)
    del(key_)

# %%
# for key_ in prec_ACF_trends_MK_dict.keys():
#     prec_ACF_trends_MK_dict[key_].rename(columns={"trend": f"trend_{key_}".replace('precip_mm', 'prec'),
#                                                   "p_value": f"p_value_{key_}".replace('precip_mm', 'prec'),
#                                                   "slope": f"slope_{key_}".replace('precip_mm', 'prec')},
#                                                   inplace=True)
#     del(key_)

# %%

# %%
temp_ACF_trends_MK_dict[list(temp_ACF_dict.keys())[0]].head(3)

# %%
from functools import reduce

# Convert dict values to a list of DataFrames
df_list = list(temp_ACF_trends_MK_dict.values())

# Perform left merges iteratively
temp_ACF_trends_MK_df = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

temp_ACF_trends_MK_df.head(2)

# %%
from functools import reduce

# Convert dict values to a list of DataFrames
df_list = list(prec_ACF_trends_MK_dict.values())

# Perform left merges iteratively
prec_ACF_trends_MK_df = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

prec_ACF_trends_MK_df.head(2)

# %%
weather_ACF_trends_MK_df = prec_ACF_trends_MK_df.merge(temp_ACF_trends_MK_df, on="fid", how="outer")
weather_ACF_trends_MK_df.head(2)

# %%
print(weather_ACF_trends_MK_df.shape)
print(temp_ACF_trends_MK_df.shape)
print(prec_ACF_trends_MK_df.shape)

# %%
filename = bio_reOrganized + "weather_ACFs_rollingWindow_trends.sav"

export_ = {
    "weather_ACF_trends_MK_df": weather_ACF_trends_MK_df,
    "source_code": "ACF_moving_window_compute_trend",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, 'wb'))

# %%
SF_west = pd.merge(SF_west, weather_ACF_trends_MK_df, how="left", on="fid")

# %%
# %%time
f_name = rangeland_bio_data + 'SF_west_weather_movingACF1s.shp.zip'
SF_west.to_file(filename=f_name, driver='ESRI Shapefile')

# %%
