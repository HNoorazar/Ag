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
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/"
min_dir = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
bpszone_ANPP = pd.read_csv(min_dir + "bpszone_annual_productivity_rpms_MEAN.csv")
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP["FID"].unique())

# %%
import geopandas
f_name = "albers_HucsGreeningBpSAtts250_For_Zonal_Stats"
bps_SF = geopandas.read_file(min_dir + f_name + "/" + f_name + ".shp")
bps_SF.head(2)

# %%
print (len(bps_SF["MinStatsID"].unique()))
print (len(bps_SF["Value"].unique()))
print (len(bps_SF["hucsgree_4"].unique()))

# %%
print ((bps_SF["hucsgree_4"] - bps_SF["Value"]).unique())

print ((list(bps_SF.index) == bps_SF.MinStatsID).sum())

# %%
bps_SF.drop(columns=["Value"], inplace=True)
bps_SF.head(2)

# %%
bpszone_ANPP["FID"].unique()[-10::]

# %% [markdown]
# #### rename columns

# %%
bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.head(2)

# %%
print (len(bpszone_ANPP["fid"].unique()))
print (len(bps_SF["hucsgree_4"].unique()))

# %%
print (bpszone_ANPP["fid"].unique().max())
print (bps_SF["BPS_CODE"].unique().max())

# %% [markdown]
# ### Check if all locations have all years in it

# %%
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP[bpszone_ANPP.fid == 1])

# %%
unique_number_of_years = {}
for a_fid in bpszone_ANPP.fid.unique():
    if not (len(bpszone_ANPP[bpszone_ANPP.fid == a_fid]) in unique_number_of_years.keys()):
        unique_number_of_years[len(bpszone_ANPP[bpszone_ANPP.fid == a_fid])] = 1
    else:
        unique_number_of_years[len(bpszone_ANPP[bpszone_ANPP.fid == a_fid])] = \
            unique_number_of_years[len(bpszone_ANPP[bpszone_ANPP.fid == a_fid])] + 1

# %%
unique_number_of_years

# %%
len(bps_SF["MinStatsID"].unique())

# %%
len(bpszone_ANPP["fid"].unique())

# %%
bpszone_ANPP["fid"].unique().max()

# %%
bps_SF["MinStatsID"].unique().max()

# %%
