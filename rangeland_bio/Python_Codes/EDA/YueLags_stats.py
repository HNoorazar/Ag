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
from datetime import datetime

import pymannkendall as mk
from scipy.stats import variation
from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
import matplotlib.ticker as plticker

import statsmodels.api as sm
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%

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

# %%
county_fips_dict = pd.read_pickle(rangeland_reOrganized + "county_fips.sav")

county_fips = county_fips_dict["county_fips"]
full_2_abb = county_fips_dict["full_2_abb"]
abb_2_full_dict = county_fips_dict["abb_2_full_dict"]
abb_full_df = county_fips_dict["abb_full_df"]
filtered_counties_29States = county_fips_dict["filtered_counties_29States"]
SoI = county_fips_dict["SoI"]
state_fips = county_fips_dict["state_fips"]

state_fips = state_fips[state_fips.state != "VI"].copy()
state_fips.head(2)

# %%

# %% [markdown]
# ## Read NPP Data

# %%
bpszone_ANPP = pd.read_csv(min_bio_dir + "2012_bpszone_annual_productivity_rpms_MEAN.csv")

bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.sort_values(by= ['fid', 'year'], inplace=True)
bpszone_ANPP.reset_index(drop=True, inplace=True)
bpszone_ANPP.head(2)

# %%
bpszone_ANPP = bpszone_ANPP[bpszone_ANPP.year != 2012]

# %%
2012 in list(bpszone_ANPP.year.unique())

# %%
ANPP_MK_df = pd.read_pickle(bio_reOrganized + "ANPP_MK_Spearman_no2012.sav")
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]
print (ANPP_MK_df.shape)
ANPP_MK_df.head(2)

# %% [markdown]
# # MK test for ANPP and Spearman's rank

# %%
# # %%time

# need_cols = ["fid"]
# ANPP_MK_df = bpszone_ANPP[need_cols].copy()
# print (ANPP_MK_df.shape)

# ANPP_MK_df.drop_duplicates(inplace=True)
# ANPP_MK_df.reset_index(drop=True, inplace=True)

# print (ANPP_MK_df.shape)
# ANPP_MK_df.head(3)
# ##### z: normalized test statistics
# ##### Tau: Kendall Tau
# MK_test_cols = ["trend", "p", "var_s",
#                 "trend_yue", "p_yue", "var_s_yue",
#                 "trend_yue_lag0", "p_yue_lag0", "var_s_yue_lag0",
#                 "trend_yue_lag1", "p_yue_lag1", "var_s_yue_lag1",
#                 "trend_yue_lag2", "p_yue_lag2", "var_s_yue_lag2",
#                 "trend_yue_lag3", "p_yue_lag3", "var_s_yue_lag3"]

# ANPP_MK_df = pd.concat([ANPP_MK_df, pd.DataFrame(columns = MK_test_cols)])
# ANPP_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)

# # Why data type changed?!
# ANPP_MK_df["fid"] = ANPP_MK_df["fid"].astype(np.int64)
# ANPP_MK_df.head(2)
# ###############################################################
# # populate the dataframe with MK test result now
# for a_FID in ANPP_MK_df["fid"].unique():
#     ANPP_TS = bpszone_ANPP.loc[bpszone_ANPP.fid==a_FID, "mean_lb_per_acr"].values
#     year_TS = bpszone_ANPP.loc[bpszone_ANPP.fid==a_FID, "year"].values
    
#     # MK test
#     trend, _, p, _, _, _, var_s, _, _ = mk.original_test(ANPP_TS)
#     trend_u, _, p_u, _, _, _, var_s_u, _, _                =mk.yue_wang_modification_test(ANPP_TS)
#     trend_u_lag0, _, p_u_lag0, _, _, _, var_s_u_lag0, _, _ =mk.yue_wang_modification_test(ANPP_TS, lag=0)
#     trend_u_lag1, _, p_u_lag1, _, _, _, var_s_u_lag1, _, _ =mk.yue_wang_modification_test(ANPP_TS, lag=1)
#     trend_u_lag2, _, p_u_lag2, _, _, _, var_s_u_lag2, _, _ =mk.yue_wang_modification_test(ANPP_TS, lag=2)
#     trend_u_lag3, _, p_u_lag3, _, _, _, var_s_u_lag3, _, _ =mk.yue_wang_modification_test(ANPP_TS, lag=3)
    
#     # Update dataframe by MK result
#     L_ = [trend, p, var_s,
#           trend_u, p_u, var_s_u, 
#           trend_u_lag0, p_u_lag0, var_s_u_lag0,
#           trend_u_lag1, p_u_lag1, var_s_u_lag1,
#           trend_u_lag2, p_u_lag2, var_s_u_lag2,
#           trend_u_lag3, p_u_lag3, var_s_u_lag3]
    
#     ANPP_MK_df.loc[ANPP_MK_df["fid"]==a_FID, MK_test_cols] = L_
    
#     del(trend, p, var_s)
#     del(trend_u, p_u, var_s_u)
#     del(trend_u_lag0, p_u_lag0, var_s_u_lag0)
#     del(trend_u_lag1, p_u_lag1, var_s_u_lag1)
#     del(trend_u_lag2, p_u_lag2, var_s_u_lag2)
#     del(trend_u_lag3, p_u_lag3, var_s_u_lag3)
#     del(L_, ANPP_TS, year_TS)


# # Round the columns to 6-decimals
# for a_col in ["p", "var_s",
#               "p_yue", "var_s_yue",
#               "p_yue_lag0", "var_s_yue_lag0",
#               "p_yue_lag1", "var_s_yue_lag1",
#               "p_yue_lag2", "var_s_yue_lag2",
#               "p_yue_lag3", "var_s_yue_lag3"]:
#     ANPP_MK_df[a_col] = ANPP_MK_df[a_col].round(6)
# ANPP_MK_df.head(2)

# %% [markdown]
# # Original and 0-lag the same?

# %%
print ((ANPP_MK_df["trend"] == ANPP_MK_df["trend_yue_lag0"]).sum() == len(ANPP_MK_df))

# %% [markdown]
# ## Original Counts

# %%
trend_col = "trend"
trend_count_orig = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_orig.rename(columns={"fid": "original"}, inplace=True)

# %% [markdown]
# ## Yue Counts

# %%
trend_col = "trend_yue"
trend_count_yue = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue.rename(columns={"fid": "Yue",
                                "trend_yue" : "trend"}, inplace=True)
##########################################################################################
trend_col = "trend_yue_lag0"
trend_count_yue_lag0 = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue_lag0.rename(columns={"fid": "Yue_lag0", "trend_yue_lag0" : "trend"}, inplace=True)
##########################################################################################
trend_col = "trend_yue_lag1"
trend_count_yue_lag1 = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue_lag1.rename(columns={"fid": "Yue_lag1", "trend_yue_lag1" : "trend"}, inplace=True)
##########################################################################################
trend_col = "trend_yue_lag2"
trend_count_yue_lag2 = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue_lag2.rename(columns={"fid": "Yue_lag2", "trend_yue_lag2" : "trend"}, inplace=True)
##########################################################################################
trend_col = "trend_yue_lag3"
trend_count_yue_lag3 = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue_lag3.rename(columns={"fid": "Yue_lag3", "trend_yue_lag3" : "trend"}, inplace=True)

# %%
trend_count_all = pd.merge(trend_count_orig, trend_count_yue, how="left", on="trend")
trend_count_all = pd.merge(trend_count_all, trend_count_yue_lag0, how="left", on="trend")
trend_count_all = pd.merge(trend_count_all, trend_count_yue_lag1, how="left", on="trend")
trend_count_all = pd.merge(trend_count_all, trend_count_yue_lag2, how="left", on="trend")
trend_count_all = pd.merge(trend_count_all, trend_count_yue_lag3, how="left", on="trend")
trend_count_all

# %%

# %%
CL_99_orig = ANPP_MK_df.copy()
CL_99_orig = CL_99_orig[CL_99_orig.p < 0.01].copy()

CL_99_yue = ANPP_MK_df.copy()
CL_99_yue = CL_99_yue[CL_99_yue.p_yue < 0.01].copy()

CL_99_yue_lag0 = ANPP_MK_df.copy()
CL_99_yue_lag0 = CL_99_yue_lag0[CL_99_yue_lag0.p_yue_lag0 < 0.01].copy()

CL_99_yue_lag1 = ANPP_MK_df.copy()
CL_99_yue_lag1 = CL_99_yue_lag1[CL_99_yue_lag1.p_yue_lag1 < 0.01].copy()

CL_99_yue_lag2 = ANPP_MK_df.copy()
CL_99_yue_lag2 = CL_99_yue_lag2[CL_99_yue_lag2.p_yue_lag2 < 0.01].copy()

CL_99_yue_lag3 = ANPP_MK_df.copy()
CL_99_yue_lag3 = CL_99_yue_lag3[CL_99_yue_lag3.p_yue_lag3 < 0.01].copy()

# %%
trend_col = "trend"
trend_count_orig = CL_99_orig[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_orig.rename(columns={"fid": "Original"}, inplace=True)
#############################################################################
trend_col = "trend_yue"
trend_count_yue = CL_99_yue[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue.rename(columns={"fid": "Yue", "trend_yue" : "trend"}, inplace=True)
#############################################################################
trend_col = "trend_yue_lag0"
trend_count_yue_lag0 = CL_99_yue_lag0[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue_lag0.rename(columns={"fid": "Yue_lag0", "trend_yue_lag0" : "trend"}, inplace=True)
#############################################################################
trend_col = "trend_yue_lag1"
trend_count_yue_lag1 = CL_99_yue_lag1[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue_lag1.rename(columns={"fid": "Yue_lag1", "trend_yue_lag1" : "trend"}, inplace=True)
#############################################################################
trend_col = "trend_yue_lag2"
trend_count_yue_lag2 = CL_99_yue_lag2[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue_lag2.rename(columns={"fid": "Yue_lag2", "trend_yue_lag2" : "trend"}, inplace=True)
#############################################################################
trend_col = "trend_yue_lag3"
trend_count_yue_lag3 = CL_99_yue_lag3[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue_lag3.rename(columns={"fid": "Yue_lag3", "trend_yue_lag3" : "trend"}, inplace=True)
#############################################################################

# %%
trend_count_all_99CL = pd.merge(trend_count_orig, trend_count_yue, how="left", on="trend")
trend_count_all_99CL = pd.merge(trend_count_all_99CL, trend_count_yue_lag0, how="left", on="trend")
trend_count_all_99CL = pd.merge(trend_count_all_99CL, trend_count_yue_lag1, how="left", on="trend")
trend_count_all_99CL = pd.merge(trend_count_all_99CL, trend_count_yue_lag2, how="left", on="trend")
trend_count_all_99CL = pd.merge(trend_count_all_99CL, trend_count_yue_lag3, how="left", on="trend")
trend_count_all_99CL

# %%

# %%

# %%
