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

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

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
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)
####### Laptop
# rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
# min_bio_dir = rangeland_bio_base
# rangeland_base = rangeland_bio_base
# rangeland_reOrganized = rangeland_base

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman.sav"
ANPP_MK_Spearman = pd.read_pickle(filename)
ANPP_MK_Spearman = ANPP_MK_Spearman["ANPP_MK_df"]
ANPP_MK_Spearman.head(2)

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_Spearman_no2012 = pd.read_pickle(filename)
ANPP_MK_Spearman_no2012 = ANPP_MK_Spearman_no2012["ANPP_MK_df"]
ANPP_MK_Spearman_no2012.head(2)

# %%
trend_col = "trend"
trend_count_orig = ANPP_MK_Spearman[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_orig.rename(columns={"fid": "fid_original"}, inplace=True)

trend_count_orig

trend_col = "trend_yue"
trend_count_yue = ANPP_MK_Spearman[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue.rename(columns={"fid": "fid_yue",
                                "trend_yue" : "trend"}, inplace=True)
trend_count_yue

trend_col = "trend_rao"
trend_count_rao = ANPP_MK_Spearman[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_rao.rename(columns={"fid": "fid_rao", 
                               "trend_rao" : "trend"}, inplace=True)
trend_count_rao

trend_counts = pd.merge(trend_count_orig, trend_count_yue, on="trend", how="outer")
trend_counts = pd.merge(trend_counts, trend_count_rao, on="trend", how="outer")
trend_counts

# %%
trend_col = "trend"
trend_count_orig = ANPP_MK_Spearman_no2012[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_orig.rename(columns={"fid": "fid_original"}, inplace=True)

trend_count_orig

trend_col = "trend_yue"
trend_count_yue = ANPP_MK_Spearman_no2012[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue.rename(columns={"fid": "fid_yue",
                                "trend_yue" : "trend"}, inplace=True)
trend_count_yue

trend_col = "trend_rao"
trend_count_rao = ANPP_MK_Spearman_no2012[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_rao.rename(columns={"fid": "fid_rao", 
                               "trend_rao" : "trend"}, inplace=True)
trend_count_rao

trend_counts_no2012 = pd.merge(trend_count_orig, trend_count_yue, on="trend", how="outer")
trend_counts_no2012 = pd.merge(trend_counts_no2012, trend_count_rao, on="trend", how="outer")
trend_counts_no2012

# %%

# %%
orig = (trend_counts_no2012.loc[trend_counts_no2012["trend"] == "increasing", "fid_original"] - \
        trend_counts.loc[trend_counts["trend"] == "increasing", "fid_original"]).item()

Yue = (trend_counts_no2012.loc[trend_counts_no2012["trend"] == "increasing", "fid_yue"] - \
       trend_counts.loc[trend_counts["trend"] == "increasing", "fid_yue"]).item()

Rao = (trend_counts_no2012.loc[trend_counts_no2012["trend"] == "increasing", "fid_rao"] - \
       trend_counts.loc[trend_counts["trend"] == "increasing", "fid_rao"]).item()

print ("diff. as a result of removing 2012 in  MK: {}".format(orig))
print ("diff. as a result of removing 2012 in Rao: {}".format(Rao))
print ("diff. as a result of removing 2012 in Yue: {}".format(Yue))

# %%
bpszone_ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP.sav")
print (bpszone_ANPP["Date"])
bpszone_ANPP = bpszone_ANPP["bpszone_ANPP"]
# bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
# bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
#                              "count": "pixel_count",
#                              "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.sort_values(by=['fid', 'year'], inplace=True)
bpszone_ANPP.reset_index(drop=True, inplace=True)
bpszone_ANPP.head(2)

# %% [markdown]
# ### Lets stick to no-2012
#
# out of more than 27,000 ```FID``` on west of meridian we have only 4,332 ```FID``` for 2012 in NPP data.
#
#    - Find FIDs in common between Rao and Yue with increasing trend.
#    - Find FIDs not in common between Rao and Yue with increasing trend.

# %%
ANPP_MK_Spearman_no2012.head(2)

# %%
green_FIDs_Yue = ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["trend_yue"] == "increasing", "fid"].unique()
green_FIDs_Rao = ANPP_MK_Spearman_no2012.loc[ANPP_MK_Spearman_no2012["trend_rao"] == "increasing", "fid"].unique()
green_FIDs_Yue_Rao = set(green_FIDs_Yue).intersection(set(green_FIDs_Rao))

print (len(green_FIDs_Yue))
print (len(green_FIDs_Rao))
print (len(green_FIDs_Yue_Rao))

# %%
# %%time 
# Rao_FIDs_in_Yue = [0] * len(green_FIDs_Rao)
Rao_FIDs_in_Yue = set()
for a_fid in green_FIDs_Rao:
    if a_fid in green_FIDs_Yue:
        Rao_FIDs_in_Yue.add(a_fid)

len(Rao_FIDs_in_Yue)

# %%
print (len((Rao_FIDs_in_Yue)))
print (len(set(Rao_FIDs_in_Yue)))

# %%
# %%time 
# Rao_FIDs_in_Yue = [0] * len(green_FIDs_Rao)
Yue_FIDs_missed_by_Rao = set()
for a_fid in green_FIDs_Yue:
    if not(a_fid in green_FIDs_Rao):
        Yue_FIDs_missed_by_Rao.add(a_fid)

len(Yue_FIDs_missed_by_Rao)

# %%

# %%
