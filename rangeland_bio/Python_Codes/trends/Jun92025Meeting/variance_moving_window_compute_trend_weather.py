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


sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc

# %%

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
rolling_var_data_dir = rangeland_bio_data + "rolling_variances/"

# %%
os.listdir(rolling_var_data_dir)

# %%
# %%time
## bad 2012
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west.head(2)

# %% [markdown]
# ## Read all rolling window variances

# %%
temp_ys = ["temp", "temp_detrendLinReg", "temp_detrendDiff", "temp_detrendSens"]
prec_ys = ["prec", "prec_detrendLinReg", "prec_detrendDiff", "prec_detrendSens"]

# %%
temp_variance_dict = {}
for window_size in np.arange(5, 11):
    for y_ in temp_ys:
        key_ = f"rolling_variance_ws{window_size}_{y_}"
        f_name = rolling_var_data_dir + key_ + ".sav"
        variance_df = pd.read_pickle(f_name)
        variance_df = variance_df[key_]
        key_ = f"variance_ws{window_size}_{y_}"
        temp_variance_dict[key_] = variance_df

# %%
prec_variance_dict = {}
for window_size in np.arange(5, 11):
    for y_ in prec_ys:
        key_ = f"rolling_variance_ws{window_size}_{y_}"
        f_name = rolling_var_data_dir + key_ + ".sav"
        variance_df = pd.read_pickle(f_name)
        variance_df = variance_df[key_]
        key_ = f"variance_ws{window_size}_{y_}"
        prec_variance_dict[key_] = variance_df

# %%

# %% [markdown]
# ### Find trends of variances time-series via MK again!

# %%
(list(temp_variance_dict.keys()))

# %%

# %%
(list(prec_variance_dict.keys()))

# %%
temp_variance_dict["variance_ws5_temp"].head(2)

# %%
temp_variance_dict["variance_ws5_temp_detrendLinReg"].head(2)

# %%
import re
import importlib;
importlib.reload(rc);
importlib.reload(rpc);

# %%
# %%time

temp_variance_trends_MK_dict = {}

for a_key in temp_variance_dict.keys():
    ws = re.search(r'ws(\d+)', a_key).group(1)
    curr_df = temp_variance_dict[a_key]
    curr_col = f'variance_ws{ws}'
    temp_variance_trends_MK_dict[a_key] = rc.compute_mk_by_fid(df=curr_df, groupby_='fid', value_col=curr_col)
    del(a_key, curr_col)

# %%

# %%
# %%time

prec_variance_trends_MK_dict = {}

for a_key in prec_variance_dict.keys():
    ws = re.search(r'ws(\d+)', a_key).group(1)
    curr_df = prec_variance_dict[a_key]
    curr_col = f'variance_ws{ws}'
    prec_variance_trends_MK_dict[a_key] = rc.compute_mk_by_fid(df=curr_df, groupby_='fid', value_col=curr_col)
    del(a_key, curr_col)

# %%
temp_variance_trends_MK_dict[list(temp_variance_dict.keys())[0]].head(3)

# %%
for key_ in temp_variance_trends_MK_dict.keys():
    temp_variance_trends_MK_dict[key_].rename(columns={"trend": f"trend_{key_}",
                                                       "p_value": f"p_value_{key_}",
                                                       "slope": f"slope_{key_}"}, 
                                         inplace=True)
del(key_)

# %%
for key_ in prec_variance_trends_MK_dict.keys():
    prec_variance_trends_MK_dict[key_].rename(columns={"trend": f"trend_{key_}",
                                                       "p_value": f"p_value_{key_}",
                                                       "slope": f"slope_{key_}"}, 
                                         inplace=True)
del(key_)

# %%
temp_variance_trends_MK_dict[list(temp_variance_dict.keys())[0]].head(3)

# %%
from functools import reduce

# %%
# Convert dict values to a list of DataFrames
df_list = list(temp_variance_trends_MK_dict.values())

# Perform left merges iteratively
temp_variance_trends_MK_df = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

temp_variance_trends_MK_df.head(2)

# %%
# Convert dict values to a list of DataFrames
df_list = list(prec_variance_trends_MK_dict.values())

# Perform left merges iteratively
prec_variance_trends_MK_df = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

prec_variance_trends_MK_df.head(2)
# %%
weather_variance_trends_MK_df = prec_variance_trends_MK_df.merge(temp_variance_trends_MK_df, 
                                                                 on="fid", how="outer")
weather_variance_trends_MK_df.head(2)

# %%
filename = bio_reOrganized + "weather_variance_rollingWindow_trends.sav"

export_ = {"weather_variance_trends_MK_df": weather_variance_trends_MK_df,
           "source_code": "variance_moving_window_compute_trend_weather",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),}

pickle.dump(export_, open(filename, 'wb'))

# %%
SF_west = pd.merge(SF_west, weather_variance_trends_MK_df, how="left", on="fid")

# %%
# %%time
f_name = rangeland_bio_data + 'SF_west_weather_movingVariance.shp.zip'
SF_west.to_file(filename=f_name, driver='ESRI Shapefile')

# %%

# %%
