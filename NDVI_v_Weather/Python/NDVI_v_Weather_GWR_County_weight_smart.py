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
# The reason the name of this notebook is smart is that it assumes data is complete (all locations have full data for each year) and avoids for-loop in ```NDVI_v_Weather_GWR_County_weight_dumb.ipynb```

# %% [markdown]
# It seems all the libraryes want to do it the bandwidth way; no pre-specified weight matrix!
# Lets just do it outselves 
#
# $$\beta(u_i, v_i) = (X^T W(u_i, v_i) X) ^ {-1} X^T W(u_i, v_i) y$$

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
from datetime import datetime, date

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
# import rangeland_core as rc

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

NDVI_weather_data_dir = research_db + "/NDVI_v_Weather/"

# %%
bio_data_dir_base = research_db + "/RangeLand_bio/Data/"
bio_reOrganized_dir = bio_data_dir_base + "reOrganized/"

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

# %% [markdown]
# **western meridian** saved data is already western side (```NDVI_v_Weather_dataPrep.ipynb```)

# %%
WM_counties = county_fips_df[county_fips_df["EW_meridian"] == "W"]
WM_counties = list(WM_counties["county_fips"])
len(WM_counties)

# %%
# # %%time
# weight_rowSTD = pd.read_csv(bio_reOrganized_dir + "fid_contiguity_Queen_neighbors_rowSTD.csv")
# weight_rowSTD.head(2)

# %%
# # %%time
# weight_rowSTD_sav = pd.read_pickle(bio_reOrganized_dir + "fid_contiguity_Queen_neighbors_rowSTD.sav")
# print (weight_rowSTD_sav["source_code"])

# weight_rowSTD_sav = weight_rowSTD_sav["fid_contiguity_Queen_neighbors_rowSTD"]
# weight_rowSTD_sav.head(3)
# weight_rowSTD_sav.reset_index(inplace=True, drop=True)
# weight_rowSTD_sav.head(3)

# %%
len(WM_counties)

# %%
# # %%time
# a = pd.read_csv(bio_reOrganized_dir + "county_contiguity_Queen_neighbors_rowSTD.csv")
# a.head(3)

# %%
# %%time
weight_rowSTD_sav = pd.read_pickle(bio_reOrganized_dir + "county_contiguity_Queen_neighbors_rowSTD.sav")
print (weight_rowSTD_sav["source_code"])

weight_rowSTD_sav = weight_rowSTD_sav["county_contiguity_Queen_neighbors_rowSTD"]
weight_rowSTD_sav.head(3)

# %%
filename = "/Users/hn/Documents/01_research_data/NDVI_v_Weather/data/NDVI_weather.sav"
NDVI_weather = pd.read_pickle(filename)
print (NDVI_weather["source_code"])
NDVI_weather = NDVI_weather["NDVI_weather_input"]

# %%
NDVI_weather.head(2)

# %%
NDVI_weather[(NDVI_weather["county_fips"] == "04001") & (NDVI_weather["year"] == 2002)]

# %%
indp_vars = ['county_fips', 'year', 'month', 'tavg_avg', 'ppt', 'tavg_avg_lag1', 'ppt_lag1', 'MODIS_NDVI_lag1']
y_var = 'MODIS_NDVI'

NDVI_weather = NDVI_weather[[y_var] + indp_vars]

print (NDVI_weather.shape)
NDVI_weather.dropna(inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)
print (NDVI_weather.shape)
# 258300 - 256250

# X = NDVI_weather[indp_vars].copy()
# y = NDVI_weather[y_var].copy()

# %%
print (NDVI_weather.shape)
NDVI_weather = NDVI_weather[NDVI_weather["county_fips"].isin(WM_counties)]
print (NDVI_weather.shape)

# %%
NDVI_weather[(NDVI_weather["county_fips"] == "04001") & (NDVI_weather["year"] == 2002)]

# %%
NDVI_weather[(NDVI_weather["county_fips"] == "04001") & (NDVI_weather["year"] == 2003)]

# %%
# it was working before, without doing this!!!
# X['county_fips'] = X['county_fips'].astype(np.float64)

# %% [markdown]
# ### Shannon County, SD
#
# Shannon County, SD (```FIPS code = 46113```) was renamed Oglala Lakota County and assigned anew FIPS code (```46102```) effective in 2014.
#
#
# Old county fips for this county is ```46113``` which is what Min has in its dataset.
#
# How can I take care of this? If I get an old county shapefile, then, which year?
#
#
# Lets just figure out how many mismatches are there and exclude them

# %%
NDVI_counties = list(NDVI_weather["county_fips"].unique())

# %%
weight_rowSTD_sav_counties = list(weight_rowSTD_sav.index)

NDVI_missing_from_weights = []

for a_county in NDVI_counties:
    if not(a_county in list(weight_rowSTD_sav.index)):
        NDVI_missing_from_weights = NDVI_missing_from_weights + [a_county]

print (NDVI_missing_from_weights)
weights_missing_from_NDVI = []
for a_county in weight_rowSTD_sav_counties:
    if not(a_county in NDVI_counties):
        weights_missing_from_NDVI = weights_missing_from_NDVI + [a_county]
print (len(weights_missing_from_NDVI))
weights_missing_from_NDVI


# %% [markdown]
# # Toss 46113

# %%
"46113" in list(NDVI_weather["county_fips"].unique())

# %%
NDVI_weather = NDVI_weather[NDVI_weather['county_fips'] != "46113"].copy()

# %%
"46113" in list(NDVI_weather["county_fips"].unique())

# %% [markdown]
# ## Split train and test set

# %%
print (f"{len(sorted(NDVI_weather.year.unique())) = }")
print (sorted(NDVI_weather.year.unique()))

# %%
years_list = sorted(NDVI_weather.year.unique())
train_perc = 70/100
year_count = len(years_list)
train_years = years_list[: int(np.ceil(train_perc * year_count))]

# %%
train_DF = NDVI_weather[NDVI_weather.year.isin(train_years)].copy()
x_train = train_DF[indp_vars].copy()
y_train = train_DF[y_var].copy()

# %%
print (x_train.shape)
x_train.head(2)

# %%
# from scipy.sparse import csr_matrix, save_npz, load_npz
# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))
# sparse_matrix.toarray()

# %%
weight_rowSTD_sav.head(2)

# %% [markdown]
# ### Form weight matrix for x_train. 
#
# Sort ```x_train``` by year, so that from year to year a given county is not its own neighbor! 
#
# Or we can explore the idea of doing this for few years prior to a given year to consider time-neighboring!!!

# %%
NDVI_weather.head(2)

# %%
x_train.head(2)

# %%
y_train.head(2)

# %%
## If the data is complete; for all years and months
## we have all data, then we can compute the neighbors once
## and repeat it on main diagonal of bigger matrix. 
## Let us check that? This is not possible. split of train and test was done
## randomly.

train_unique_years = sorted(x_train["year"].unique())
train_unique_months = sorted(x_train["month"].unique())
train_unique_counties = x_train["county_fips"].unique()

# %%
# # %%time
### This crashes the Kernel. Too big.
### Lets try numpy matrix, if that does not work, go into sparse matrix.

# # the data is monthly. So, This is bad operation. Kernel dies.
w = np.zeros((len(x_train), len(x_train)))
weightMatrix = pd.DataFrame(w)

idx = list(x_train['county_fips'] + "_" + x_train['year'].astype(str) + "_" + x_train["month"].astype(str))
# weightMatrix.index = idx
weightMatrix.columns = idx
weightMatrix.index = idx
weightMatrix.head(3)

#####################################################
# weightMatrix["county_fips"] = x_train["county_fips"]
# weightMatrix["year"] = x_train["year"]
# weightMatrix["month"] = x_train["month"]
# weightMatrix = weightMatrix.set_index(['county_fips', 'year', 'month'])
#####################################################
# idx_arrays = [x_train['county_fips'], x_train['year'], x_train['month']]
# idx_tuples = list(zip(*idx_arrays))
# index_ = pd.MultiIndex.from_tuples(idx_tuples, names=["county_fips", "year", "month"])
# weightMatrix.index = index_
# weightMatrix.head(13)

# %%
weight_rowSTD_sav.head(2)

# %% [markdown]
# ### Check data is complete and form weight matrix. Fast

# %%
test_county = train_unique_counties[0]
default_length = len(x_train[x_train["county_fips"] == test_county])
default_length

# %%
bad_counties = []
for a_county in train_unique_counties:
    curr_df = x_train[x_train["county_fips"]==a_county]
    if len(curr_df) != default_length:
        bad_counties = bad_counties + [a_county]
bad_counties     

# %%
weightMatrix["04001_2002_2"].unique()

# %%
