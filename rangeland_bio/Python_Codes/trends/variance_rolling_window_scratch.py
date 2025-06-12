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

from scipy import stats
import scipy.stats as scipy_stats
from statsmodels.tsa.stattools import acf

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc

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
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012_detrended.sav")
ANPP = ANPP["ANPP_no2012_detrended"]
ANPP.head(3)

# %%
df = ANPP.copy()
y_var="mean_lb_per_acr"
window_size=5

total_windows = 0
for _, group in df.groupby("fid"):
    years = sorted(group["year"].unique())
    total_windows += max(0, len(years) - window_size + 1)

# %%
# Step 2: Preallocate an empty DataFrame
var_col_name = f"variance_ws{window_size}"
columns = ["fid", "years", var_col_name]
dtype_map = {"fid": "Int64", "years": "str", var_col_name: "float"}

# %%
preallocated_df = pd.DataFrame({col: pd.Series(index=range(total_windows), dtype=dtype)
                                    for col, dtype in dtype_map.items()})

# %%
idx = 0

# %%
df_grouped = df.groupby("fid")
print (df_grouped)

# %%
fid, group = next(iter(df_grouped))

# %%
window_years = years[i : i + window_size]
window_data = group[group["year"].isin(window_years)]

# %%
window_data

# %%
values

# %%
values.var()

# %%
ANPP = ANPP[ANPP["fid"].isin([1, 2, 3])]

# %%

# %%
values.dropna().nunique()

# %%

# %%
ANPP_var_win5 = rolling_variance_df_prealloc(df=ANPP, y_var="mean_lb_per_acr", window_size=5)


# %%

# %%
def rolling_variance_df_prealloc_notTruncate(df, y_var="mean_lb_per_acr", window_size=5):
    # Step 1: Count how many total windows we'll need
    total_windows = 0
    for _, group in df.groupby("fid"):
        years = sorted(group["year"].unique())
        total_windows += max(0, len(years) - window_size + 1)

    # Step 2: Preallocate an empty DataFrame
    var_col_name = f"variance_ws{window_size}"
    columns = ["fid", "years", var_col_name]
    dtype_map = {"fid": "Int64", "years": "str", var_col_name: "float"}

    preallocated_df = pd.DataFrame({col: pd.Series(index=range(total_windows), dtype=dtype)
                                    for col, dtype in dtype_map.items()})

    # Step 3: Populate DataFrame by index
    idx = 0
    for fid, group in df.groupby("fid"):
        group = group.sort_values("year").reset_index(drop=True)
        years = group["year"].tolist()

        for i in range(len(years) - window_size + 1):
            window_years = years[i : i + window_size]
            window_data = group[group["year"].isin(window_years)]
            
            # is the following necessary? (can it be violated? and if does, then what?)
            if len(window_data["year"]) == window_size:
                values = window_data[y_var]
                variance_ = (values.var() if len(values.dropna()) > 1 else np.nan)

                preallocated_df.loc[idx] = {
                    "fid": fid,
                    "years": "_".join(map(str, window_years)),
                    var_col_name: variance_,
                }
                idx += 1

    # Step 4: Truncate to actual number of rows (if some skipped)
    return preallocated_df.reset_index(drop=True)


# %%
# %%time

ANPP_var_win5_notTruncate = rolling_variance_df_prealloc_notTruncate(df=ANPP, 
                                                                     y_var="mean_lb_per_acr", window_size=5)

# %%
ANPP_var_win5_notTruncate[ANPP_var_win5_notTruncate["fid"]==3].head(3)

# %%

# %%
df_1 = ANPP[ANPP["fid"]==3]

# %%
df_1_G1 = df_1[df_1["year"].isin([1984, 1985, 1986, 1987, 1988])]
df_1_G1['mean_lb_per_acr'].var()

# %%

# %%
