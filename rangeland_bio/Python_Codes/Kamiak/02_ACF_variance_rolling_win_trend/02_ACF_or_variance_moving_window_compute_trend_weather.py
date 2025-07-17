# %% [markdown]
# We are modifying this notebook on July 16, 2025, to have more weather variables in it and eliminate
# first-diff detrending.
#
# The reason for "0_1" in the name of this notebook is that first we need to compute ACF1s for rolling windows, and that is done in "weather_ACF1_rolling.py" on Kamiak.
#
# And, I am adding variance trend computations to this notebook as well;
# I am merging ```variance_moving_window_compute_trend_weather.ipynb``` into this current notebook.
#
# ------------------------------------------------------------
# There is another script called ```autocorr_moving_window_analysis_ANPP.ipynb```.
#
# This is a copy of that with modifications to do weather.
#
# **June 25, 2025**
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


sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc
import rangeland_plot_core as rpc

###############################################################
#######
#######    Terminal Arguments
#######
variable_ = str(sys.argv[1])

###############################################################
#######
#######    Directories
#######
research_data_ = "/data/project/agaid/h.noorazar/"
rangeland_bio_base = research_data_ + "rangeland_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
bio_reOrganized = rangeland_bio_data + "reOrganized/"
common_data = research_data_ + "common_data/"


if variable_ == "ACF":
    rolling_data_dir = rangeland_bio_data + "rolling_ACF1/"
elif variable_ == "variance":
    rolling_data_dir = rangeland_bio_data + "rolling_variances/"


# ## Read all rolling window data

# %%
data_dirlist = [x for x in os.listdir(rolling_data_dir) if x.endswith(".sav")]
# leave anpp out here, for now. It is already done in another notebook.
data_dirlist = [x for x in data_dirlist if not ("anpp" in x)]
print(len(data_dirlist))

# %%
# %%time
data_dict = {}

for f_name in data_dirlist:
    df = pd.read_pickle(rolling_data_dir + f_name)
    df = df[f_name.replace(".sav", "")]
    if variable_ == "ACF":
        key_ = f_name.replace("rolling_ACF1_", "").replace(".sav", "")
    elif variable_ == "variance":
        key_ = f_name.replace("rolling_variance_", "").replace(".sav", "")
    data_dict[key_] = df

# %% [markdown]
# ### Find trends of ACF1/variances time-series via MK

# %%
print(len(list(data_dict.keys())))
list(sorted(data_dict.keys()))[:4]

# %%
data_dict[list(data_dict.keys())[0]].head(4)

# %%
import re
import importlib

# importlib.reload(rc)
# importlib.reload(rpc)

# %%
# %%time

trends_MK_dict = {}

for a_key in sorted(list(data_dict.keys())):
    ws = re.search(r"ws(\d+)", a_key).group(1)
    curr_df = data_dict[a_key]
    if variable_ == "ACF":
        curr_col = f"autocorr_lag1_ws{ws}"
    elif variable_ == "variance":
        curr_col = f"variance_ws{ws}"
    trends_MK_dict[a_key] = rc.compute_mk_by_fid(
        df=curr_df, groupby_="fid", value_col=curr_col
    )
    del a_key

# %%
trends_MK_dict[list(trends_MK_dict.keys())[0]].head(3)

#### Reduce/merge the data tables in the dictionaries
for key_ in trends_MK_dict.keys():
    trends_MK_dict[key_].rename(
        columns={
            "trend": f"trend_{key_}",
            "p_value": f"p_value_{key_}",
            "slope": f"slope_{key_}",
        },
        inplace=True,
    )
    del key_

# %%
trends_MK_dict[list(trends_MK_dict.keys())[0]].head(3)

# %%
from functools import reduce

# Convert dict values to a list of DataFrames
df_list = list(trends_MK_dict.values())

# Perform left merges iteratively
trends_MK_df = reduce(
    lambda left, right: pd.merge(left, right, on="fid", how="left"), df_list
)

trends_MK_df.head(2)

# %%
print(trends_MK_df.shape)

# %%
if variable_ == "ACF":
    filename = bio_reOrganized + "weather_ACFs_rollingWindow_trends.sav"
    export_ = {
        "weather_ACF_trends_MK_df": trends_MK_df,
        "source_code": "02_ACF_or_variance_moving_window_compute_trend_weather",
        "Author": "HN",
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
elif variable_ == "variance":
    filename = bio_reOrganized + "weather_variances_rollingWindow_trends.sav"
    export_ = {
        "weather_variances_trends_MK_df": trends_MK_df,
        "source_code": "01_ACF_or_variance_moving_window_compute_trend_weather",
        "Author": "HN",
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


pickle.dump(export_, open(filename, "wb"))

# %%

# %%
