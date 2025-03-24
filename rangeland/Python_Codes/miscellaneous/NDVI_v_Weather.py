# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import shutup
shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

from datetime import datetime, date
from scipy.linalg import inv

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

from pysal.model import spreg

# %%
dpi_ = 300

plot_dir = "/Users/hn/Documents/01_research_data/RangeLand/Mike_Results/plots/"
os.makedirs(plot_dir, exist_ok=True)

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
common_data = "/Users/hn/Documents/01_research_data/" + "common_data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(common_data + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_name_fips = pd.DataFrame({"state_full" : list(abb_dict["full_2_abb"].keys()),
                                "state" : list(abb_dict["full_2_abb"].values())})

state_name_fips = pd.merge(state_name_fips, 
                           abb_dict["state_fips"][["state_fips", "EW_meridian", "state"]], 
                           on=["state"], how="left")
state_name_fips.head(2)

# %%
state_fips_SoI = state_name_fips[state_name_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
state_fips_SoI.head(2)

# %%
state_fips_west = list(state_fips_SoI[state_fips_SoI["EW_meridian"] == "W"]["state_fips"].values)
state_fips_west[:3]

# %%
county_grid_mean_idx = pd.read_csv(Min_data_base + "county_gridmet_mean_indices.csv")
county_grid_mean_idx.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
county_grid_mean_idx.rename(columns={"county": "county_fips"}, inplace=True)
county_grid_mean_idx.head(2)

# %%
county_grid_mean_idx.columns

# %%
## pick proper columns
wanted_cols = ['county_fips', 'year', 'month',  'tavg_avg', 'ppt']
county_grid_mean_idx = county_grid_mean_idx[wanted_cols]
county_grid_mean_idx.head(2)

# %% [markdown]
# # 2012 
#
# We tossed 2012 in NPP trends. What to do there?

# %%
NDVI_df = pd.read_csv(reOrganized_dir + "county_monthly.csv")
NDVI_df = NDVI_df[["county", "year", "month", "MODIS_NDVI"]]
NDVI_df.rename(columns={"county": "county_fips"}, inplace=True)
# NDVI_df.dropna(subset=["MODIS_NDVI"], inplace=True)
NDVI_df.reset_index(drop=True, inplace=True)
NDVI_df.head(2)

# %%
NDVI_df = rc.correct_Mins_county_6digitFIPS(NDVI_df, col_="county_fips")
NDVI_df.head(2)

# %%
MODIS_NDVI_years = list(NDVI_df["year"].unique())

# %%
county_grid_mean_idx = rc.correct_Mins_county_6digitFIPS(county_grid_mean_idx, col_="county_fips")
print (county_grid_mean_idx.shape)
county_grid_mean_idx = county_grid_mean_idx[county_grid_mean_idx["year"].isin(MODIS_NDVI_years)].copy()
county_grid_mean_idx.reset_index(drop=True, inplace=True)
print (county_grid_mean_idx.shape)
county_grid_mean_idx.head(2)

# %%
NDVI_df[NDVI_df["county_fips"] == "01003"]

# %% [markdown]
# ### First drop the counties for which the whole year is missing

# %%
a = NDVI_df[(NDVI_df["county_fips"] == "01003")& (NDVI_df["year"] == 1982)]

# %%
counties_list = list(NDVI_df["county_fips"].unique())
year_list = list(NDVI_df["year"].unique())

# %%
# no data before 2000 from MODIS
NDVI_df[NDVI_df.year<2001]["MODIS_NDVI"].unique()

# %%
print (NDVI_df.shape)
NDVI_df = NDVI_df[NDVI_df["year"] > 2001].copy()
print (NDVI_df.shape)

# %%
# %%time
counties_with_oneYearMissingNDVI = {}

for county in counties_list:
    for year in year_list:
        a_df = NDVI_df[NDVI_df["year"] == year].copy()
        a_df = a_df[a_df["county_fips"] == county].copy()
        if a_df["MODIS_NDVI"].isna().sum() == 12:
            if county in counties_with_oneYearMissingNDVI:
                counties_with_oneYearMissingNDVI[county] += [year]
            else:
                counties_with_oneYearMissingNDVI[county] = [year]

# %%
counties_with_oneYearMissingNDVI

# %%

# %%
# I am doing outer because NDVI might be missing
# The only one missing. Lets see
NDVI_weather = pd.merge(NDVI_df, county_grid_mean_idx, how="left", on=['county_fips', 'year', 'month'])
NDVI_weather.sort_values(by=["county_fips", "year"], inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)
NDVI_weather.head(5)

# %%
NDVI_miss = NDVI_weather.dropna(subset=["MODIS_NDVI"], inplace=False)
temp_miss = NDVI_weather.dropna(subset=["tavg_avg"], inplace=False)
ppt_miss = NDVI_weather.dropna(subset=["ppt"], inplace=False)

# %%
print (f"{NDVI_weather.shape = }")
print (f"{NDVI_miss.shape = }")
print (f"{temp_miss.shape = }")
print (f"{ppt_miss.shape = }")

# %%
570780-568514

# %% [markdown]
# ### Impute missing data, if any
#
# - First check; just so we know.

# %%
counties_list = list(NDVI_weather["county_fips"].unique())
year_list = list(NDVI_weather["year"].unique())

# %%
# find location of missing values
missing_val_idx = NDVI_weather[NDVI_weather["MODIS_NDVI"].isnull()].index.tolist()
missing_val_idx[:4]

# %% [markdown]
# # Impute Carefully
#
# Impute by taking average of neghboring months. But, since we must do this county by county
# we should do a for-loop.

# %%
NDVI_weather.head(5)

# %%
###
###    All counties have missing values!!!
###
print (f"{len(missing_val_idx) = }")
print (f"{len(NDVI_weather.loc[missing_val_idx]['county_fips'].unique()) = }")

# %%
NDVI_weather.sort_values(by=["county_fips", "year"], inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)

# %% [markdown]
# # Min: Why lots of Feb. NDVIs are missing?

# %%
a = np.array(missing_val_idx[:-1]) - np.array(missing_val_idx[1:])
a

# %%
pd.Series(a).unique()

# %%
NDVI_weather.loc[missing_val_idx].month.unique()

# %%
NDVI_weather.loc[missing_val_idx].groupby("month").count()

# %%
np.where(a == -18)

# %%
np.where(a == -234)

# %%
NDVI_weather.loc[missing_val_idx[2129]-2 : missing_val_idx[2129]+2]

# %%
NDVI_weather.loc[missing_val_idx[2130]-2 : missing_val_idx[2130]+2]

# %%
# # %%time

# for county in counties_list:
#     a_df = NDVI_weather[NDVI_weather["county_fips"] == county]
#     curr_miss_idx = a_df[a_df["MODIS_NDVI"].isnull()].index.tolist()
    
#     if a_df["MODIS_NDVI"].isna().sum() > 0:
#         curr_miss_idx = a_df[a_df["MODIS_NDVI"].isnull()].index.tolist()
        
#         # if missing value is first or last, just drop it!
#         L = len(curr_miss_idx)
#         if curr_miss_idx[0] == a_df.index.min():
#             NDVI_weather.drop(curr_miss_idx[0])
#             curr_miss_idx.pop(0)
        
#         if L > 2:
#             if curr_miss_idx[-1] == a_df.index.max():
#                 NDVI_weather.drop(curr_miss_idx[-1])
#                 curr_miss_idx.pop(-1)
        
#         # now, if anything is left:
#         if len(curr_miss_idx) > 0:
#             for a_miss_idx in curr_miss_idx:
#                 # lets assume no two/three consecutive months are missing!

# %%
## It seems there are no consecutive missing values.
## We can find index of pre- and post-missing values
## and do it quickly. (no for-loop)

# %%
pre_miss_idx = NDVI_weather[NDVI_weather["MODIS_NDVI"].isnull()].index - 1
post_miss_idx = NDVI_weather[NDVI_weather["MODIS_NDVI"].isnull()].index + 1

# %%
NDVI_weather.loc[missing_val_idx, "MODIS_NDVI"] = 0.5 * (NDVI_weather.loc[post_miss_idx, "MODIS_NDVI"].values + 
                                                            NDVI_weather.loc[pre_miss_idx, "MODIS_NDVI"].values)

# %%
NDVI_weather.head(5)

# %%
# # %%time

# county_year_check = {}

# for county in counties_list:
#     for year in year_list:
#         a_df = NDVI_weather[NDVI_weather["year"] == year]
#         a_df = a_df[a_df["county_fips"] == county]
#         if len(a_df) != 12 :
#             if county in county_year_check:
#                 county_year_check[county] += [year]
#             else:
#                 county_year_check[county] = [year]

# %%
print (NDVI_weather.shape)
print (NDVI_weather.dropna().shape)

# %%

# %%

# %%
