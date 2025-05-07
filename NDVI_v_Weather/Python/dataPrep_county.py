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
# import shutup
# shutup.please()
#
# import pandas as pd
# import numpy as np
# import os, os.path, pickle, sys
# import seaborn as sns
#
# import matplotlib
# import matplotlib.pyplot as plt
#
# from sklearn import preprocessing
# from datetime import datetime, date
#
# current_time = datetime.now().strftime("%H:%M:%S")
# print("Today's date:", date.today())
# print("Current Time =", current_time)
#
# sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
# import rangeland_core as rc

# %%
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from keras import losses, optimizers, metrics

# %%

# %%
dpi_ = 300

plot_dir = "/Users/hn/Documents/01_research_data/RangeLand/Mike_Results/plots/"
os.makedirs(plot_dir, exist_ok=True)

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

data_dir_base = research_db + "RangeLand/Data/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(common_data + "county_fips.sav")
county_fips_df = abb_dict["county_fips"]
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
# # pick up western meridian

# %%
WM_counties = county_fips_df[county_fips_df["EW_meridian"] == "W"]
WM_counties = list(WM_counties["county_fips"])
len(WM_counties)

# %%

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
len(NDVI_df["county_fips"].unique())

# %%
NDVI_df = rc.correct_Mins_county_6digitFIPS(NDVI_df, col_="county_fips")
NDVI_df.head(2)

# %%
MODIS_NDVI_years = list(NDVI_df["year"].unique())

# %%
print (NDVI_df.shape)
NDVI_df = NDVI_df[NDVI_df["county_fips"].isin(WM_counties)]
NDVI_df.shape

# %%
county_grid_mean_idx = rc.correct_Mins_county_6digitFIPS(county_grid_mean_idx, col_="county_fips")
print (county_grid_mean_idx.shape)
county_grid_mean_idx = county_grid_mean_idx[county_grid_mean_idx["year"].isin(MODIS_NDVI_years)].copy()
county_grid_mean_idx.reset_index(drop=True, inplace=True)
print (county_grid_mean_idx.shape)
county_grid_mean_idx.head(2)

# %%
county_grid_mean_idx = county_grid_mean_idx[county_grid_mean_idx["county_fips"].isin(WM_counties)]
county_grid_mean_idx.shape

# %%
county_grid_mean_idx.head(2)

# %% [markdown]
# ### First drop the counties for which the whole year is missing

# %%
a = NDVI_df[(NDVI_df["county_fips"] == "04001")& (NDVI_df["year"] == 1982)]
a

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

for county in list(NDVI_df["county_fips"].unique()):
    for year in list(NDVI_df["year"].unique()):
        a_df = NDVI_df[NDVI_df["year"] == year].copy()
        a_df = a_df[a_df["county_fips"] == county].copy()
        if a_df["MODIS_NDVI"].isna().sum() == 12:
            if county in counties_with_oneYearMissingNDVI:
                counties_with_oneYearMissingNDVI[county] += [year]
            else:
                counties_with_oneYearMissingNDVI[county] = [year]
                
counties_with_oneYearMissingNDVI

# %%
county_grid_mean_idx = county_grid_mean_idx[county_grid_mean_idx["year"] > 2001].copy()

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

print (f"{NDVI_weather.shape = }")
print (f"{NDVI_miss.shape = }")
print (f"{temp_miss.shape = }")
print (f"{ppt_miss.shape = }")

# %%
258300-257275

# %% [markdown]
# ### Impute missing data, if any
#
# - First check; just so we know.

# %% [markdown]
# # 2012 
#
# We tossed 2012 in NPP trends. What to do there?

# %%
tick_legend_FontSize = 6
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5, 
          "legend.handlelength": 2}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
cnty = list(NDVI_df["county_fips"].unique())[200]
df = NDVI_weather[NDVI_weather["county_fips"] == cnty].copy()
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')

fig, axes = plt.subplots(1, 1, figsize=(4, 1.5), sharey=False, sharex=False, dpi=dpi_)
axes.plot(df["date"], df['MODIS_NDVI'], linewidth=1.2, color="dodgerblue", zorder=1);

# %%
NDVI_weather.sort_values(by=["county_fips", "year"], inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)

# %% [markdown]
# # Min
# - Why lots of Feb. ```NDVI```s are missing?
# - Why ```NPP``` (state level) was bad in 2012 but not here; ```NDVI``` on county level 

# %%
# %%time

year_2012_check = {}

for county in list(NDVI_weather["county_fips"].unique()):
    a_df = NDVI_weather[NDVI_weather["year"] == 2012].copy()
    a_df = a_df[a_df["county_fips"] == county]
    a_df.dropna(subset=["MODIS_NDVI"], inplace=True)
    if len(a_df) < 11 :
        if county in year_2012_check:
            year_2012_check[county] += [len(a_df)]
        else:
            year_2012_check[county] = [len(a_df)]

# %% [markdown]
# # Impute Carefully
#
# Impute by taking average of neghboring months. But, since we must do this county by county
# we should do a for-loop.

# %%
# find location of missing values
missing_val_idx = NDVI_weather[NDVI_weather["MODIS_NDVI"].isnull()].index.tolist()
missing_val_idx[:4]

# %%
###
###    All counties have missing values!!!
###
print (f"{len(missing_val_idx) = }")
print (f"{len(NDVI_weather.loc[missing_val_idx]['county_fips'].unique()) = }")

# %%
NDVI_weather.head(4)

# %%
a = np.array(missing_val_idx[:-1]) - np.array(missing_val_idx[1:])
a

# %%
pd.Series(a).unique()

# %%
NDVI_weather.loc[missing_val_idx].month.unique()

# %%
NDVI_weather.loc[250:255]

# %%
NDVI_weather.loc[missing_val_idx].groupby("month").count()

# %%
# When we focused on western meridian this cell became irrelevant
# np.where(a == -18)
# np.where(a == -234)

# NDVI_weather.loc[missing_val_idx[2129]-2 : missing_val_idx[2129]+2]
# NDVI_weather.loc[missing_val_idx[2130]-2 : missing_val_idx[2130]+2]

# %%

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

# for county in list(NDVI_weather["county_fips"].unique()):
#     for year in list(NDVI_weather["year"].unique()):
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

# %% [markdown]
# This is the base model. Lets see what we can change
#
# \begin{equation}
# \label{eq:DLNDVI}
# \text{NDVI}_t = f(T_{t-1}, S_{t-1}, P_{t-1}, \Delta \text{NDVI}_{t-1}, t, \text{lat}, \text{long})
# \end{equation}
#
# Let us make a copy of data, keep the original as is. and then manipulate it.

# %%
NDVI_weather_orig = NDVI_weather.copy()

# %%
# NDVI_weather = NDVI_weather_orig.copy()

# %%
NDVI_weather_shifted = NDVI_weather.copy()
NDVI_weather_shifted['month'] = NDVI_weather_shifted['month'] + 1
# change the year for Januaries
NDVI_weather_shifted.head(5)

# %%
idx_13 = NDVI_weather_shifted[NDVI_weather_shifted["month"]==13].index
NDVI_weather_shifted.loc[idx_13, "year"] = NDVI_weather_shifted.loc[idx_13, "year"] + 1
NDVI_weather_shifted.loc[idx_13, "month"] = 1
NDVI_weather_shifted.head(5)

# %%
NDVI_weather_shifted.rename(columns={"MODIS_NDVI": "MODIS_NDVI_lag1",
                                     "tavg_avg" : "tavg_avg_lag1",
                                     "ppt" : "ppt_lag1"}, inplace=True)

# %%
NDVI_weather_orig[8:16]

# %%
NDVI_weather_shifted[8:16]

# %%
NDVI_weather = pd.merge(NDVI_weather, NDVI_weather_shifted, how="left", on=['county_fips', 'year', 'month'])
NDVI_weather.head(2)

# %%
NDVI_weather[8:16]

# %%
NDVI_weather["delta_NDVI"] = np.nan

# %%
# %%time
for a_county in list(NDVI_weather.county_fips.unique()):
    df = NDVI_weather[NDVI_weather["county_fips"] == a_county]
    curr_idx = df.index
    deltas = df.loc[curr_idx[1:], "MODIS_NDVI"].values - df.loc[curr_idx[:-1], "MODIS_NDVI"].values
    
    NDVI_weather.loc[curr_idx[2:], "delta_NDVI"] = deltas[:-1]


# %%
NDVI_weather.head(15)

# %%
NDVI_weather.tail(15)

# %%
NDVI_weather.isna().sum()

# %%
NDVI_weather[NDVI_weather["delta_NDVI"].isna()].head(10)

# %%
####### Do not drop NA here   #######
####### if we decide to train with lag_NDVI rather than delta_NDVI
####### we will have more data

# print (NDVI_weather.shape)
# NDVI_weather.dropna(inplace=True)
# print (NDVI_weather.shape)
# 258300 - 256250

# %%
filename = "/Users/hn/Documents/01_research_data/NDVI_v_Weather/data/" + "NDVI_weather.sav"

export_ = {"NDVI_weather_input": NDVI_weather, 
           "NDVI_weather_orig": NDVI_weather_orig, 
           "source_code" : "NDVI_v_Weather_NB1",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
print (NDVI_weather.shape)
NDVI_weather.dropna(inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)
print (NDVI_weather.shape)
# 258300 - 256250

indp_vars = ['county_fips', 'year', 'month', 'tavg_avg_lag1', 'ppt_lag1', 'delta_NDVI']
y_var = 'MODIS_NDVI'

X = NDVI_weather[indp_vars].copy()
y = NDVI_weather[y_var].copy()
