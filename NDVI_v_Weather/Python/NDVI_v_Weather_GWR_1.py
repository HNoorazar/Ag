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

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

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
print (bio_reOrganized_dir + "fid_Queen_neighbors_rowSTD.csv")


# %%
# %%time
a = pd.read_csv(bio_reOrganized_dir + "fid_contiguity_Queen_neighbors_rowSTD.csv")

# %%
a.head(2)

# %%
filename = "/Users/hn/Documents/01_research_data/NDVI_v_Weather/data/NDVI_weather.sav"
NDVI_weather = pd.read_pickle(filename)
NDVI_weather=NDVI_weather["NDVI_weather_input"]

# %%
NDVI_weather.head(2)

# %%
indp_vars = ['county_fips', 'year', 'month', 'tavg_avg', 'ppt', 'tavg_avg_lag1', 'ppt_lag1', 'MODIS_NDVI_lag1']
y_var = 'MODIS_NDVI'

NDVI_weather = NDVI_weather[[y_var] + indp_vars]

print (NDVI_weather.shape)
NDVI_weather.dropna(inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)
print (NDVI_weather.shape)
# 258300 - 256250

X = NDVI_weather[indp_vars].copy()
y = NDVI_weather[y_var].copy()

# %%
print (NDVI_weather.shape)
NDVI_weather = NDVI_weather[NDVI_weather["county_fips"].isin(WM_counties)]
print (NDVI_weather.shape)

# %%
# it was working before, without doing this!!!
# X['county_fips'] = X['county_fips'].astype(np.float64)

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# %%
from scipy.sparse import csr_matrix, save_npz, load_npz
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))

# %%
sparse_matrix.toarray()

# %%

# %%
