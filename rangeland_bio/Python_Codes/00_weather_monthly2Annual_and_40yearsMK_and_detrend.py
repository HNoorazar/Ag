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
# # Convert Monthly weather to annual
# and compute MK for them.
#
# This was called ``MK_weather``. We are adding min and max of weather stuff and numbering notebooks
#
# Jul. 11, 2025

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
sorted(bps_weather.columns)

# %%
annual_weather = bps_weather.groupby(['fid', 'year']).agg({'avg_of_dailyAvg_rel_hum': 'mean',
                                                           'avg_of_dailyAvgTemp_C': 'mean',
                                                           'thi_avg': 'mean',
                                                           'avg_of_dailyMaxTemp_C' : 'max',
                                                           'avg_of_dailyMinTemp_C' : 'min',
                                                           'max_of_dailyMaxTemp_C' : 'mean',
                                                           'min_of_dailyMinTemp_C' : 'mean',
                                                           'precip_mm_month': 'sum',
                                                          }).reset_index()

annual_weather.rename(columns={"precip_mm_month": "precip_mm",
                               "avg_of_dailyMaxTemp_C" : "max_of_monthlyAvg_of_dailyMaxTemp_C",
                               'avg_of_dailyMinTemp_C' : 'min_of_monthlyAvg_of_dailyMinTemp_C',
                               'max_of_dailyMaxTemp_C' : 'avg_of_monthlymax_of_dailyMaxTemp_C',
                               'min_of_dailyMinTemp_C' : 'avg_of_monthlymin_of_dailyMinTemp_C',
                              }, inplace=True)


annual_weather.head(3)

# %% [markdown]
# ### Check if all locations have all years in it

# %%
len(annual_weather[annual_weather.fid == 1])

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

# %%
annual_weather.drop(columns=['state_1', 'state_2'], inplace=True)

# %% [markdown]
# # MK test and Spearman's rank for Weather

# %%
sorted(annual_weather.columns)

# %%

# %% [markdown]
# # MK on weather variables

# %%
# %%time

y_vars = ['avg_of_dailyAvgTemp_C',
          'avg_of_dailyAvg_rel_hum',
          'avg_of_monthlymax_of_dailyMaxTemp_C',
          'avg_of_monthlymin_of_dailyMinTemp_C',
          'max_of_monthlyAvg_of_dailyMaxTemp_C',
          'min_of_monthlyAvg_of_dailyMinTemp_C',
          'precip_mm',
          'thi_avg',]
len_y_vars = len(y_vars)
count=1

all_treds_dict = {}
MK_test_cols = ["sens_slope", "sens_intercept", "Tau", "MK_score",
                "trend", "p", "var_s"]

for y_var in y_vars:
    MK_df = annual_weather[["fid"]].copy()
    print (MK_df.shape)
    MK_df.drop_duplicates(inplace=True)
    MK_df.reset_index(drop=True, inplace=True)
    print (MK_df.shape)
    
    ##### z: normalized test statistics
    ##### Tau: Kendall Tau
    MK_df = pd.concat([MK_df, pd.DataFrame(columns = MK_test_cols)])
    MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)

    # Why data type changed?!
    MK_df["fid"] = MK_df["fid"].astype(np.int64)
    ###############################################################
    # populate the dataframe with MK test result now
    for a_FID in MK_df["fid"].unique():
        precip_TS = annual_weather.loc[annual_weather.fid==a_FID, y_var].values
        year_TS = annual_weather.loc[annual_weather.fid==a_FID, "year"].values

        # MK test original
        trend, _, p, z, Tau, MK_score, var_s, slope, intercept = mk.original_test(precip_TS)
        # Spearman, p_Spearman = stats.spearmanr(year_TS, precip_TS) # Spearman's rank

        # Update dataframe by MK result
        L_ = [slope, intercept, Tau, MK_score, trend, p, var_s]

        MK_df.loc[MK_df["fid"]==a_FID, MK_test_cols] = L_

        del(slope, intercept, Tau, MK_score, trend, p, var_s)
        del(L_, precip_TS, year_TS)

    # Round the columns to 4-decimals
    for a_col in ["sens_slope", "sens_intercept", "Tau", "MK_score", "p", "var_s"]:
        MK_df[a_col] = MK_df[a_col].astype(float)
        MK_df[a_col] = round(MK_df[a_col], 4)
        
    MK_df.rename(columns={col: col+'_'+y_var if col != "fid" else col for col in MK_df.columns}, inplace=True)
    key_ = "MK_" + y_var
    all_treds_dict[key_] = MK_df
    print (f"{count} out of {len_y_vars}")
    count += 1
    print ("================================================================")

# %%

# %%
from functools import reduce

# %%
# temp_ACF_trends_MK_dict[list(temp_ACF_dict.keys())[0]].head(3)

# Convert dict values to a list of DataFrames
df_list = list(all_treds_dict.values())

# Perform left merges iteratively
weather_MK_df = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

weather_MK_df.head(2)

# %%
filename = bio_reOrganized + "weather_MK_Spearman.sav"

export_ = {"weather_MK_df": weather_MK_df, 
           "source_code" : "00_weather_monthly2Annual_and_MK",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
print (Albers_SF.shape)
print (weather_MK_df.shape)

# %% [markdown]
# # Detrend
# ### Add detrending to this notebook from ```deTrend_weather.ipynb```


# %%
weather_MK_df.head(2)

# %%
annual_weather.head(2)

# %%
sens_cols = ["fid"] + [x for x in weather_MK_df.columns if ("slope" in x) or ("intercept" in x)]
sens_cols

# %%
annual_weather_detrend = annual_weather.copy()
annual_weather_detrend = pd.merge(annual_weather_detrend, weather_MK_df[sens_cols], how="left", on="fid")
annual_weather_detrend.head(2)

# %% [markdown]
# ### Sens prediction 
#
# must not be based on year since that test only lookst at y values.

# %%
annual_weather_detrend['row_number_perfid'] = annual_weather_detrend.groupby('fid').cumcount()
annual_weather_detrend.head(2)

# %%
sorted(annual_weather.columns)

# %%
for y_var in y_vars:
    annual_weather_detrend[f"{y_var}_senPred"] = annual_weather_detrend["row_number_perfid"] * \
                                                     annual_weather_detrend[f"sen_slope_{y_var}"] + \
                                                       annual_weather_detrend[f"sens_intercept_{y_var}"]
    
    annual_weather_detrend[f"{y_var}_detrendSens"] = annual_weather_detrend[y_var] - \
                                                            annual_weather_detrend[f"{y_var}_senPred"]
    
annual_weather_detrend.head(2)

# %%
## detrend using Simple Linear regression

# %%

# %%

# %%

# %%

# %%
## out_name = bio_reOrganized + "bpszone_annual_tempPrecip_byHN.csv"
# out_name = bio_reOrganized + "bpszone_annual_weather_and_deTrended_byHN.csv"
# annual_weather.to_csv(out_name, index = False)


# filename = bio_reOrganized + "bpszone_annual_weather_and_deTrended_byHN.sav"

# export_ = {"bpszone_annual_weather_byHN": annual_weather, 
#            "source_code" : "00_weather_monthly2Annual_and_MK",
#            "Author": "HN",
#            "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# pickle.dump(export_, open(filename, 'wb'))

# %%
# # out_name = bio_reOrganized + "bpszone_annual_tempPrecip_byHN.csv"
# out_name = bio_reOrganized + "bpszone_annual_weather_byHN.csv"
# annual_weather.to_csv(out_name, index = False)



# filename = bio_reOrganized + "bpszone_annual_weather_byHN.sav"

# export_ = {"bpszone_annual_weather_byHN": annual_weather, 
#            "source_code" : "00_weather_monthly2Annual_and_MK",
#            "Author": "HN",
#            "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# pickle.dump(export_, open(filename, 'wb'))

# %% [markdown]
# # We had these here before, 
# and ```slope``` was changed to ```m``` at some point but names are too long, so, there is no point

# %%
# # %%time
# f_name = bio_reOrganized + 'Albers_SF_west_weather_MK_Spearman.shp.zip'
# Albers_SF.to_file(filename=f_name, driver='ESRI Shapefile')


# # %%time
# f_name = bio_reOrganized + 'Albers_SF_west_weather_MK_Spearman.shp.zip'
# A = geopandas.read_file(f_name)
# A.head(2)

# %%
