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
dpi_ = 300
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

# %%

# %%
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
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
# ## Read NPP Data

# %%
filename = bio_reOrganized + "bpszone_ANPP_no2012.sav"
bpszone_ANPP = pd.read_pickle(filename)
bpszone_ANPP = bpszone_ANPP["bpszone_ANPP"]

bpszone_ANPP["fid"].unique()[-8::]

# %%
print (len(bpszone_ANPP["fid"].unique()))
print (len(Albers_SF["hucsgree_4"].unique()))

print (bpszone_ANPP["fid"].unique().max())
print (Albers_SF["bps_code"].unique().max())

# %% [markdown]
# ### Check if all locations have all years in it

# %%
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP[bpszone_ANPP.fid == 1])

# %%
# %%time
unique_number_of_years = {}

for a_fid in bpszone_ANPP.fid.unique():
    LL = str(len(bpszone_ANPP[bpszone_ANPP.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%
print (f'{len(Albers_SF["fid"].unique()) = }')
print (f'{len(bpszone_ANPP["fid"].unique())= }')
print (f'{bpszone_ANPP["fid"].unique().max()= }')
print (f'{Albers_SF["fid"].unique().max()= }')

# %%
bpszone_ANPP.head(2)

# %% [markdown]
# # Make some plots

# %%
Albers_SF.plot(column='EW_meridian', categorical=True, legend=True);

# %%
from shapely.geometry import Polygon

gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%

# %%
bpszone_ANPP.head(2)

# %%
2012 in list(bpszone_ANPP.year.unique())

# %%
num_locs = len(bpszone_ANPP["fid"].unique())
num_locs

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
if not ("EW_meridian" in bpszone_ANPP.columns):
    bpszone_ANPP = pd.merge(bpszone_ANPP, Albers_SF[cols_], how="left", on = "fid")
bpszone_ANPP.head(2)

# %%
median_diff = bpszone_ANPP[["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]].copy()
print (median_diff.shape)

median_diff.drop_duplicates(inplace=True)
median_diff.reset_index(drop=True, inplace=True)

print (median_diff.shape)
median_diff.head(3)

# %%
# %%time
## Check if each ID has unique state_1, state_2, and state_majority_area
bad_FIDs = []
for a_FID in median_diff["fid"].unique():
    curr_df = median_diff[median_diff.fid == a_FID]
    if len(curr_df) > 1:
        print (a_FID)
        bad_FIDs += bad_FIDs + [a_FID]

# %%
median_diff.head(2)

# %%
len(median_diff["fid"].unique())

# %%
# Not all locations have the same number of data in them
# Lets just assume they do. The missing year
median_diff["first_10_years_median_ANPP"] = -666
median_diff["last_10_years_median_ANPP"]  = -666

# %%
# %%time
# Find median of first decade and last decade of ANPP

for a_FID in median_diff["fid"].unique():
    curr_df = bpszone_ANPP[bpszone_ANPP["fid"] == a_FID]
    
    min_year = curr_df["year"].min()
    max_year = curr_df["year"].max()
    
    first_decade = curr_df[curr_df["year"] < min_year + 10]
    last_decade  = curr_df[curr_df["year"] > max_year - 10]
    
    median_diff.loc[median_diff["fid"] == a_FID, "first_10_years_median_ANPP"] = \
                                                first_decade['mean_lb_per_acr'].median()

    median_diff.loc[median_diff["fid"] == a_FID, "last_10_years_median_ANPP"] = \
                                                    last_decade['mean_lb_per_acr'].median()

# %%
median_diff.head(2)

# %% [markdown]
# # WARNING: year_diff below is hard coded and is wrong
# False warning. Regardless of missing data in between! we are correct

# %%
year_diff = bpszone_ANPP["year"].max() - bpszone_ANPP["year"].min()

median_diff["medians_diff_ANPP"] = median_diff["last_10_years_median_ANPP"] - \
                                      median_diff["first_10_years_median_ANPP"]

median_diff["medians_diff_slope_ANPP"] = median_diff["medians_diff_ANPP"] / year_diff
median_diff.head(2)

# %%
print (median_diff["medians_diff_slope_ANPP"].min())
print (median_diff["medians_diff_slope_ANPP"].max())

# %%
median_diff[median_diff["medians_diff_slope_ANPP"] < -19]

# %% [markdown]
# ### change (as %)

# %%
median_diff["median_ANPP_change_as_perc"] = (100 * median_diff["medians_diff_ANPP"]) / \
                                                  median_diff["first_10_years_median_ANPP"]
median_diff.head(2)

# %%
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP["fid"].unique())

# %% [markdown]
# # MK test for ANPP and Spearman's rank

# %%
# %%time

need_cols = ["fid"]
ANPP_MK_df = bpszone_ANPP[need_cols].copy()
print (ANPP_MK_df.shape)
ANPP_MK_df.drop_duplicates(inplace=True)
ANPP_MK_df.reset_index(drop=True, inplace=True)
print (ANPP_MK_df.shape)
ANPP_MK_df.head(3)
##### z: normalized test statistics
##### Tau: Kendall Tau
MK_test_cols = ["sens_slope", "Tau", "MK_score",
                "trend", "p", "var_s",
                "trend_yue", "p_yue", "var_s_yue",
                "trend_yue_lag0", "p_yue_lag0", "var_s_yue_lag0",
                "trend_yue_lag1", "p_yue_lag1", "var_s_yue_lag1",
                "trend_yue_lag2", "p_yue_lag2", "var_s_yue_lag2",
                "trend_yue_lag3", "p_yue_lag3", "var_s_yue_lag3",
                "trend_rao", "p_rao", "var_s_rao",
                "Spearman", "p_Spearman"]

ANPP_MK_df = pd.concat([ANPP_MK_df, pd.DataFrame(columns = MK_test_cols)])
ANPP_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)

# Why data type changed?!
ANPP_MK_df["fid"] = ANPP_MK_df["fid"].astype(np.int64)
###############################################################
# populate the dataframe with MK test result now
for a_FID in ANPP_MK_df["fid"].unique():
    ANPP_TS = bpszone_ANPP.loc[bpszone_ANPP.fid==a_FID, "mean_lb_per_acr"].values
    year_TS = bpszone_ANPP.loc[bpszone_ANPP.fid==a_FID, "year"].values
    
    # MK test
    #### original
    trend, _, p, z, Tau, MK_score, var_s, slope, intercept = mk.original_test(ANPP_TS)
    
    #### Yue
    trend_u, _, p_u, _, _, _, var_s_u, _, _                = mk.yue_wang_modification_test(ANPP_TS)
    trend_u_lag0, _, p_u_lag0, _, _, _, var_s_u_lag0, _, _ = mk.yue_wang_modification_test(ANPP_TS, lag=0)
    trend_u_lag1, _, p_u_lag1, _, _, _, var_s_u_lag1, _, _ = mk.yue_wang_modification_test(ANPP_TS, lag=1)
    trend_u_lag2, _, p_u_lag2, _, _, _, var_s_u_lag2, _, _ = mk.yue_wang_modification_test(ANPP_TS, lag=2)
    trend_u_lag3, _, p_u_lag3, _, _, _, var_s_u_lag3, _, _ = mk.yue_wang_modification_test(ANPP_TS, lag=3)
    
    trend_rao, _, p_rao, _, _, _, var_s_rao, _, _ = mk.hamed_rao_modification_test(ANPP_TS) #### Rao
    Spearman, p_Spearman = stats.spearmanr(year_TS, ANPP_TS) # Spearman's rank
    
    # Update dataframe by MK result
    L_ = [slope, Tau, MK_score, 
          trend,        p,        var_s,
          trend_u,      p_u,      var_s_u, 
          trend_u_lag0, p_u_lag0, var_s_u_lag0,
          trend_u_lag1, p_u_lag1, var_s_u_lag1,
          trend_u_lag2, p_u_lag2, var_s_u_lag2,
          trend_u_lag3, p_u_lag3, var_s_u_lag3,
          trend_rao,    p_rao,    var_s_rao,
          Spearman, p_Spearman]
    
    ANPP_MK_df.loc[ANPP_MK_df["fid"]==a_FID, MK_test_cols] = L_
    
    del(slope, Tau, MK_score)
    del(trend, p, var_s)
    del(trend_u, p_u, var_s_u)
    del(trend_u_lag0, p_u_lag0, var_s_u_lag0)
    del(trend_u_lag1, p_u_lag1, var_s_u_lag1)
    del(trend_u_lag2, p_u_lag2, var_s_u_lag2)
    del(trend_u_lag3, p_u_lag3, var_s_u_lag3)
    del(Spearman, p_Spearman )
    del(L_, ANPP_TS, year_TS)
    
# Round the columns to 6-decimals
for a_col in ["sens_slope", "Tau", "MK_score",
              "p", "var_s",
              "p_yue"     , "var_s_yue",
              "p_yue_lag0", "var_s_yue_lag0",
              "p_yue_lag1", "var_s_yue_lag1",
              "p_yue_lag2", "var_s_yue_lag2",
              "p_yue_lag3", "var_s_yue_lag3"]:
    ANPP_MK_df[a_col] = ANPP_MK_df[a_col].astype(float)
    ANPP_MK_df[a_col] = round(ANPP_MK_df[a_col], 6)
    
ANPP_MK_df.head(2)

# %%
ANPP_MK_df.head(2)

# %%
print (len(ANPP_MK_df["var_s"].unique()))
print (len(ANPP_MK_df["var_s_yue"].unique()))
print (len(ANPP_MK_df["var_s_rao"].unique()))

# %%
# for a_FID in ANPP_MK_df["fid"].unique():
ii = 3
a_FID = ANPP_MK_df["fid"].unique()[ii]
# a_FID = 27045
ANPP_TS = bpszone_ANPP.loc[bpszone_ANPP.fid==a_FID, "mean_lb_per_acr"].values
year_TS = bpszone_ANPP.loc[bpszone_ANPP.fid==a_FID, "year"].values
mk.original_test(ANPP_TS)
print (variation(ANPP_TS, ddof=1))
print (np.var(ANPP_TS))
mk.original_test(ANPP_TS)[6]

# %%
ANPP_MK_df.shape

# %%
some_col = ["fid", "medians_diff_ANPP", "medians_diff_slope_ANPP", "median_ANPP_change_as_perc", 
            "state_majority_area"]

ANPP_MK_df = pd.merge(ANPP_MK_df, median_diff[some_col], on="fid", how="left")
ANPP_MK_df.head(2)

# %%
spearman_increase_pval5 = ANPP_MK_df[ANPP_MK_df["p_Spearman"] < 0.05]
spearman_increase_pval5 = spearman_increase_pval5[spearman_increase_pval5["Spearman"] > 0]
spearman_increase_pval5.shape

# %%
spearman_decrease_pval5 = ANPP_MK_df[ANPP_MK_df["p_Spearman"] < 0.05]
spearman_decrease_pval5 = spearman_decrease_pval5[spearman_decrease_pval5["Spearman"] < 0]
spearman_decrease_pval5.shape

# %%
len(ANPP_MK_df) - (len(spearman_increase_pval5) + len(spearman_decrease_pval5))

# %%
Albers_SF.head(2)

# %%
some_col = ["fid", "sens_slope", "trend", 'trend_yue','p_yue', 'trend_rao', 'p_rao',
            "Tau", "Spearman", "p_Spearman",
            "medians_diff_ANPP", "medians_diff_slope_ANPP", "median_ANPP_change_as_perc"]

Albers_SF = pd.merge(Albers_SF, ANPP_MK_df[some_col], on="fid", how="left")
Albers_SF.head(2)

Albers_SF["centroid"] = Albers_SF["geometry"].centroid
Albers_SF.head(2)

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"

export_ = {"ANPP_MK_df": ANPP_MK_df, 
           "source_code" : "MK_MedianDiffs_no2012",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
Albers_SF_noCentroid = Albers_SF.copy()
Albers_SF_noCentroid.drop(columns=["centroid"], inplace=True)
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
Albers_SF_noCentroid.to_file(filename=f_name, driver='ESRI Shapefile')


# %%
