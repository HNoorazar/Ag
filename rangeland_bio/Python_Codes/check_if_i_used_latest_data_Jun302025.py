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
# There was a chain of emails on ```sept, 17, 2024``` that lead Min to do new computations.
#
# if you are looking for that email chain look for
# [https://drive.google.com/drive/folders/1_iYDQfz1BcD0kFZ2XWvAF8qjdeVizWWS?usp=sharing](https://drive.google.com/drive/folders/1_iYDQfz1BcD0kFZ2XWvAF8qjdeVizWWS?usp=sharing)
#  in emails.
#  
#  
# Let me check if I have used latest data!!! dammit

# %%
import os
import pandas as pd

# %%
sept_17 = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/Min_Data/stat_09172024/"
old_dir = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/reOrganized/"

# %%
bpszone_ANPP_no2012 = pd.read_csv(old_dir + "bpszone_ANPP_no2012.csv")
bps_weather = pd.read_pickle(old_dir + "bps_weather.sav")
bps_weather = bps_weather["bps_weather"]

# %%
sept_anpp = pd.read_csv(sept_17+'bpszone_annual_productivity_rpms_MEAN.csv')
sept_weather = pd.read_csv(sept_17 + "bps_gridmet_mean_indices.csv")

# %%
sept_anpp.head(2)

# %%
bpszone_ANPP_no2012.head(2)

# %%
sept_weather.head(2)

# %%
sept_weather.rename(columns={"bpshuc": "fid", 
                             "PPT" : "precip_mm_month",
                             "THI_AVG" : "thi_avg",
                             'TAVG_AVG': 'avg_of_dailyAvgTemp_C',
                             'RAVG_AVG':'avg_of_dailyAvg_rel_hum',
                            },
                    inplace=True)


# %%
sept_weather[list(bps_weather.columns)].head(5)

# %%
bps_weather.head(5)

# %%
sept_weather[list(bps_weather.columns)].tail(5)

# %%
bps_weather.tail(5)

# %%

# %% [markdown]
# # LULC

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"
rangeland_bio_base = research_db + "RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"
min_bio_dir_v11 = rangeland_bio_data + "Min_Data_v1.1/"

LULC_dir = min_bio_dir + "Rangeland_Landcover/"

# %%
anpp_sept_17 = pd.read_csv(sept_17 + "bpszone_annual_productivity_rpms_MEAN.csv")
anpp_v1_1 = pd.read_csv(min_bio_dir_v11 + "bpszone_annual_productivity_rpms_MEAN.csv")

# %%
anpp_sept_17.head(2)

# %%
anpp_v1_1.head(2)

# %%
anpp_sept_17 = anpp_sept_17[anpp_sept_17.year != 2012]
anpp_sept_17 = anpp_sept_17[anpp_sept_17.FID.isin(list(anpp_v1_1.fid.unique()))]

# %%
print (anpp_sept_17.shape)
print (anpp_v1_1.shape)

# %%
anpp_sept_17.sort_values(["FID", "year"], inplace=True)
anpp_sept_17.reset_index(drop=True, inplace=True)

anpp_v1_1.sort_values(["fid", "year"], inplace=True)
anpp_v1_1.reset_index(drop=True, inplace=True)

# %%
(anpp_v1_1['mean_lb_per_acr'] - anpp_sept_17['MEAN']).sum()

# %%
import numpy as np
Y = np.array(([1, 0], [0, 1], [1, 1]))
B = np.array(([1, 4], [0, 2]))
Z = np.array(([1, 0, 0, 0], [0, .5, 1, 0.5]))
np.dot(np.dot(Y, B), Z)

# %%
A = np.array(([2, 5, 1], [1, 3, 5], [3, 1, 1]))

# %%
np.linalg.det(A)

# %% [markdown]
# ### Check GDDs

# %%

# %%
dir_ = "/Users/hn/Documents/01_research_data/RangeLand/Data_large_notUsedYet/Min_data/Min_WeeklyClimateMean/"

research_db = "/Users/hn/Documents/01_research_data/"
rangeland_bio_base = research_db + "/RangeLand_bio/"

common_data = research_db + "common_data/"

rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

# %%
os.listdir(dir_)

# %%
bps_weather = pd.read_csv(min_bio_dir + "bps_gridmet_mean_indices.csv")
bps_weather.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bps_weather.head(2)

# %%
print (len(bps_weather['bpshuc'].unique()))

# %%
df = pd.read_csv(dir_ + "ecozone_gridmet_mean_weekly_ppt_et0_gddetc.csv")
print (len(df['ecozone'].unique()))
df.head(2)

# %%
df = pd.read_csv(dir_ + "prfgrid_gridmet_mean_weekly_ppt_et0_gddetc.csv")
print (len(df['prfgrid'].unique()))
df.head(2)

# %%
