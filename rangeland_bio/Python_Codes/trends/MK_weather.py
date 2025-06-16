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
# !pip3 install pymannkendall

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import pymannkendall as mk
from scipy.stats import variation
from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from datetime import datetime

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"
rangeland_bio_base = research_db + "RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
# min_bio_dir = rangeland_bio_data + "Min_Data/"
min_bio_dir_v11 = rangeland_bio_data + "Min_Data_v1.1/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

# %%
county_fips_dict = pd.read_pickle(common_data + "county_fips.sav")

county_fips = county_fips_dict["county_fips"]
full_2_abb = county_fips_dict["full_2_abb"]
abb_2_full_dict = county_fips_dict["abb_2_full_dict"]
abb_full_df = county_fips_dict["abb_full_df"]
filtered_counties_29States = county_fips_dict["filtered_counties_29States"]
SoI = county_fips_dict["SoI"]
state_fips = county_fips_dict["state_fips"]

state_fips = state_fips[state_fips.state != "VI"].copy()
state_fips.head(2)

# %% [markdown]
# ## Read the shapefile
# And keep the vegtype in subsequent dataframes

# %%
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)
Albers_SF.head(2)

# %%
len(Albers_SF["fid"].unique())

# %% [markdown]
# # Focus only on West Meridian

# %%
print ((Albers_SF["state_majority_area"] == Albers_SF["state_1"]).sum())
print ((Albers_SF["state_majority_area"] == Albers_SF["state_2"]).sum())
print (Albers_SF.shape)
print (len(Albers_SF) - (Albers_SF["state_1"] == Albers_SF["state_2"]).sum())
print ((Albers_SF["state_1"] == Albers_SF["state_2"]).sum())

# %%
Albers_SF = pd.merge(Albers_SF, state_fips[["EW_meridian", "state_full"]], 
                     how="left", left_on="state_majority_area", right_on="state_full")

Albers_SF.drop(columns=["state_full"], inplace=True)

print (Albers_SF.shape)
Albers_SF.head(2)

# %%
Albers_SF = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF.shape

# %%
print (len(Albers_SF["fid"].unique()))
print (len(Albers_SF["value"].unique()))
print (len(Albers_SF["hucsgree_4"].unique()))

print ((Albers_SF["hucsgree_4"] - Albers_SF["value"]).unique())
print ((list(Albers_SF.index) == Albers_SF.fid).sum())

Albers_SF.drop(columns=["value"], inplace=True)
Albers_SF.head(2)

# %% [markdown]
# ## Read weather Data

# %%
filename = bio_reOrganized + "bps_weather.sav"
bps_weather = pd.read_pickle(filename)
bps_weather = bps_weather["bps_weather"]
bps_weather["fid"].unique()[-8::]

# %%
west_FIDs = list(Albers_SF["fid"])
bps_weather = bps_weather[bps_weather['fid'].isin(west_FIDs)]
bps_weather.reset_index(drop=True, inplace=True)
bps_weather.head(2)

# %%
# filename = bio_reOrganized + "bps_weather_wide.sav"
# bps_weather_wide = pd.read_pickle(filename)
# bps_weather_wide = bps_weather_wide['bps_weather_wide']
# bps_weather_wide = bps_weather_wide[bps_weather_wide['fid'].isin(west_FIDs)]
# bps_weather_wide.reset_index(drop=True, inplace=True)
# bps_weather_wide.head(2)

# %%
# 

# %%
# bps_gridmet_mean = pd.read_csv(rangeland_bio_data + "Min_Data/" + "bps_gridmet_mean_indices.csv")

# bps_gridmet_mean.rename(columns={"bpshuc": "fid"}, inplace=True)
# bps_gridmet_mean = bps_gridmet_mean[bps_gridmet_mean['fid'].isin(west_FIDs)]
# bps_gridmet_mean.head(2)

# bps_gridmet_mean.reset_index(drop=True, inplace=True)
# A = bps_gridmet_mean[["fid", "year", "month", "RAVG_AVG", 'TAVG_AVG', 'THI_AVG', "PPT"]].copy()
# A.columns = list(bps_weather.columns)
# A.equals(bps_weather)

# %%
# print (f"{bps_weather_wide.shape=}")
# print (f"{bps_gridmet_mean.shape=}")
print (f"{bps_weather.shape=     }")
print (f"{Albers_SF.shape=}")

# %%
print (f'{len(Albers_SF["fid"])=}')
print (f'{len(Albers_SF["fid"].unique())=}')
# print (f'{len(bps_weather_wide["fid"].unique())=}')
# print (f'{len(bps_gridmet_mean["fid"].unique())=}')
# print (f"{len(bps_weather_wide["fid"])=}")

# %%
annual_weather = bps_weather.groupby(['fid', 'year']).agg({'avg_of_dailyAvg_rel_hum': 'mean',
                                                           'avg_of_dailyAvgTemp_C': 'mean',
                                                           'thi_avg': 'mean',
                                                           'precip_mm_month': 'sum'}).reset_index()

annual_weather.rename(columns={"precip_mm_month": "precip_mm"}, inplace=True)

annual_weather.head(3)

# %% [markdown]
# ### Check if all locations have all years in it

# %%
len(annual_weather[annual_weather.fid == 1])

# %%
# %%time
unique_number_of_years = {}

for a_fid in annual_weather.fid.unique():
    LL = str(len(annual_weather[annual_weather.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%
print (f'{len(Albers_SF["fid"].unique()) = }')
print (f'{len(annual_weather["fid"].unique())= }')
print (f'{annual_weather["fid"].unique().max()= }')
print (f'{Albers_SF["fid"].unique().max()= }')

# %%
annual_weather.head(2)

# %%
annual_weather.head(2)

# %%
num_locs = len(annual_weather["fid"].unique())
num_locs

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
if not ("EW_meridian" in annual_weather.columns):
    annual_weather = pd.merge(annual_weather, Albers_SF[cols_], how="left", on = "fid")
annual_weather.head(2)

# %% [markdown]
# # MK test and Spearman's rank for Wather

# %%
# %%time
y_var = "precip_mm"
need_cols = ["fid"]
precip_MK_df = annual_weather[need_cols].copy()
print (precip_MK_df.shape)
precip_MK_df.drop_duplicates(inplace=True)
precip_MK_df.reset_index(drop=True, inplace=True)
print (precip_MK_df.shape)
precip_MK_df.head(3)
##### z: normalized test statistics
##### Tau: Kendall Tau
MK_test_cols = ["sens_slope", "sens_intercept", "Tau", "MK_score",
                "trend", "p", "var_s",
                "trend_yue", "p_yue", "var_s_yue",
#                 "trend_yue_lag0", "p_yue_lag0", "var_s_yue_lag0",
#                 "trend_yue_lag1", "p_yue_lag1", "var_s_yue_lag1",
#                 "trend_yue_lag2", "p_yue_lag2", "var_s_yue_lag2",
#                 "trend_yue_lag3", "p_yue_lag3", "var_s_yue_lag3",
                "trend_rao", "p_rao", "var_s_rao",
                "Spearman", "p_Spearman"]

precip_MK_df = pd.concat([precip_MK_df, pd.DataFrame(columns = MK_test_cols)])
precip_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)

# Why data type changed?!
precip_MK_df["fid"] = precip_MK_df["fid"].astype(np.int64)
###############################################################
# populate the dataframe with MK test result now
for a_FID in precip_MK_df["fid"].unique():
    precip_TS = annual_weather.loc[annual_weather.fid==a_FID, y_var].values
    year_TS = annual_weather.loc[annual_weather.fid==a_FID, "year"].values
    
    # MK test
    #### original
    trend, _, p, z, Tau, MK_score, var_s, slope, intercept = mk.original_test(precip_TS)
    
    #### Yue
    trend_u, _, p_u, _, _, _, var_s_u, _, _                = mk.yue_wang_modification_test(precip_TS)
#     trend_u_lag0, _, p_u_lag0, _, _, _, var_s_u_lag0, _, _ = mk.yue_wang_modification_test(precip_TS, lag=0)
#     trend_u_lag1, _, p_u_lag1, _, _, _, var_s_u_lag1, _, _ = mk.yue_wang_modification_test(precip_TS, lag=1)
#     trend_u_lag2, _, p_u_lag2, _, _, _, var_s_u_lag2, _, _ = mk.yue_wang_modification_test(precip_TS, lag=2)
#     trend_u_lag3, _, p_u_lag3, _, _, _, var_s_u_lag3, _, _ = mk.yue_wang_modification_test(precip_TS, lag=3)

    #### Rao
    trend_rao, _, p_rao, _, _, _, var_s_rao, _, _ = mk.hamed_rao_modification_test(precip_TS)
    Spearman, p_Spearman = stats.spearmanr(year_TS, precip_TS) # Spearman's rank
    
    # Update dataframe by MK result
    L_ = [slope, intercept, Tau, MK_score, 
          trend,        p,        var_s,
          trend_u,      p_u,      var_s_u, 
#           trend_u_lag0, p_u_lag0, var_s_u_lag0,
#           trend_u_lag1, p_u_lag1, var_s_u_lag1,
#           trend_u_lag2, p_u_lag2, var_s_u_lag2,
#           trend_u_lag3, p_u_lag3, var_s_u_lag3,
          trend_rao,    p_rao,    var_s_rao,
          Spearman, p_Spearman]
    
    precip_MK_df.loc[precip_MK_df["fid"]==a_FID, MK_test_cols] = L_
    
    del(slope, intercept, Tau, MK_score)
    del(trend, p, var_s)
    del(trend_u, p_u, var_s_u)
#     del(trend_u_lag0, p_u_lag0, var_s_u_lag0)
#     del(trend_u_lag1, p_u_lag1, var_s_u_lag1)
#     del(trend_u_lag2, p_u_lag2, var_s_u_lag2)
#     del(trend_u_lag3, p_u_lag3, var_s_u_lag3)
    del(Spearman, p_Spearman )
    del(L_, precip_TS, year_TS)
    
# Round the columns to 6-decimals
for a_col in ["sens_slope", "sens_slope", "Tau", "MK_score",
              "p", "var_s",
              "p_yue"     , "var_s_yue",
#               "p_yue_lag0", "var_s_yue_lag0",
#               "p_yue_lag1", "var_s_yue_lag1",
#               "p_yue_lag2", "var_s_yue_lag2",
#               "p_yue_lag3", "var_s_yue_lag3"
             ]:
    precip_MK_df[a_col] = precip_MK_df[a_col].astype(float)
    precip_MK_df[a_col] = round(precip_MK_df[a_col], 6)
    
precip_MK_df.head(2)

# %%
print (len(precip_MK_df["var_s"].unique()))
print (len(precip_MK_df["var_s_yue"].unique()))
print (len(precip_MK_df["var_s_rao"].unique()))

precip_MK_df.head(2)

# %%

# %%
# %%time
y_var = "avg_of_dailyAvgTemp_C"
need_cols = ["fid"]
temp_MK_df = annual_weather[need_cols].copy()
print (temp_MK_df.shape)
temp_MK_df.drop_duplicates(inplace=True)
temp_MK_df.reset_index(drop=True, inplace=True)
print (temp_MK_df.shape)
temp_MK_df.head(3)
##### z: normalized test statistics
##### Tau: Kendall Tau
MK_test_cols = ["sens_slope", "sens_intercept", "Tau", "MK_score",
                "trend", "p", "var_s",
                "trend_yue", "p_yue", "var_s_yue",
#                 "trend_yue_lag0", "p_yue_lag0", "var_s_yue_lag0",
#                 "trend_yue_lag1", "p_yue_lag1", "var_s_yue_lag1",
#                 "trend_yue_lag2", "p_yue_lag2", "var_s_yue_lag2",
#                 "trend_yue_lag3", "p_yue_lag3", "var_s_yue_lag3",
                "trend_rao", "p_rao", "var_s_rao",
                "Spearman", "p_Spearman"]

temp_MK_df = pd.concat([temp_MK_df, pd.DataFrame(columns = MK_test_cols)])
temp_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)

# Why data type changed?!
temp_MK_df["fid"] = temp_MK_df["fid"].astype(np.int64)
###############################################################
# populate the dataframe with MK test result now
for a_FID in temp_MK_df["fid"].unique():
    temp_TS = annual_weather.loc[annual_weather.fid==a_FID, y_var].values
    year_TS = annual_weather.loc[annual_weather.fid==a_FID, "year"].values
    
    # MK test
    #### original
    trend, _, p, z, Tau, MK_score, var_s, slope, intercept = mk.original_test(temp_TS)
    
    #### Yue
    trend_u, _, p_u, _, _, _, var_s_u, _, _                = mk.yue_wang_modification_test(temp_TS)
#     trend_u_lag0, _, p_u_lag0, _, _, _, var_s_u_lag0, _, _ = mk.yue_wang_modification_test(temp_TS, lag=0)
#     trend_u_lag1, _, p_u_lag1, _, _, _, var_s_u_lag1, _, _ = mk.yue_wang_modification_test(temp_TS, lag=1)
#     trend_u_lag2, _, p_u_lag2, _, _, _, var_s_u_lag2, _, _ = mk.yue_wang_modification_test(temp_TS, lag=2)
#     trend_u_lag3, _, p_u_lag3, _, _, _, var_s_u_lag3, _, _ = mk.yue_wang_modification_test(temp_TS, lag=3)

    #### Rao
    trend_rao, _, p_rao, _, _, _, var_s_rao, _, _ = mk.hamed_rao_modification_test(temp_TS) 
    Spearman, p_Spearman = stats.spearmanr(year_TS, temp_TS) # Spearman's rank
    
    # Update dataframe by MK result
    L_ = [slope, intercept, Tau, MK_score, 
          trend,        p,        var_s,
          trend_u,      p_u,      var_s_u, 
#           trend_u_lag0, p_u_lag0, var_s_u_lag0,
#           trend_u_lag1, p_u_lag1, var_s_u_lag1,
#           trend_u_lag2, p_u_lag2, var_s_u_lag2,
#           trend_u_lag3, p_u_lag3, var_s_u_lag3,
          trend_rao,    p_rao,    var_s_rao,
          Spearman, p_Spearman]
    
    temp_MK_df.loc[temp_MK_df["fid"]==a_FID, MK_test_cols] = L_
    
    del(slope, intercept, Tau, MK_score)
    del(trend, p, var_s)
    del(trend_u, p_u, var_s_u)
#     del(trend_u_lag0, p_u_lag0, var_s_u_lag0)
#     del(trend_u_lag1, p_u_lag1, var_s_u_lag1)
#     del(trend_u_lag2, p_u_lag2, var_s_u_lag2)
#     del(trend_u_lag3, p_u_lag3, var_s_u_lag3)
    del(Spearman, p_Spearman )
    del(L_, temp_TS, year_TS)
    
# Round the columns to 6-decimals
for a_col in ["sens_slope", "sens_slope", "Tau", "MK_score",
              "p", "var_s",
              "p_yue"     , "var_s_yue",
#               "p_yue_lag0", "var_s_yue_lag0",
#               "p_yue_lag1", "var_s_yue_lag1",
#               "p_yue_lag2", "var_s_yue_lag2",
#               "p_yue_lag3", "var_s_yue_lag3"
             ]:
    temp_MK_df[a_col] = temp_MK_df[a_col].astype(float)
    temp_MK_df[a_col] = round(temp_MK_df[a_col], 6)
    
temp_MK_df.head(2)

# %%

# %%
filename = bio_reOrganized + "temp_MK_Spearman.sav"

export_ = {"temp_MK_df": temp_MK_df, 
           "source_code" : "MK_weather",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
filename = bio_reOrganized + "precip_MK_Spearman.sav"

export_ = {"precip_MK_df": precip_MK_df, 
           "source_code" : "MK_weather",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
some_col = ["fid", "sens_slope", "sens_intercept", "trend", 
            "trend_yue","p_yue", "trend_rao", "p_rao",
            "Tau", "Spearman", "p_Spearman"]

# %%
Albers_SF_temp = Albers_SF.copy()
Albers_SF_temp = pd.merge(Albers_SF_temp, temp_MK_df[some_col], on="fid", how="left")
Albers_SF_temp.head(2)

# %%
Albers_SF_precip = Albers_SF.copy()
Albers_SF_precip = pd.merge(Albers_SF_precip, precip_MK_df[some_col], on="fid", how="left")
Albers_SF_precip.head(2)

# %%

# %%
f_name = bio_reOrganized + 'Albers_SF_west_temp_MK_Spearman.shp.zip'
Albers_SF_temp.to_file(filename=f_name, driver='ESRI Shapefile')


# %%
f_name = bio_reOrganized + 'Albers_SF_west_precip_MK_Spearman.shp.zip'
Albers_SF_precip.to_file(filename=f_name, driver='ESRI Shapefile')
