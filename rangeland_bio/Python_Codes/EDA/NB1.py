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
    LL = str(len(bpszone_ANPP[bpszone_ANPP.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

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

# %%
import geopandas

Albers_SF_name = min_dir + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)


Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "Min_statID", 
                          "satae_max": "satae_majority_area"}, inplace=True)


Albers_SF.head(2)

# %%
bps_SF.head(2)

# %%
fips_dir = "/Users/hn/Documents/01_research_data/RangeLand/Data/reOrganized/"
county_fips_dict = pd.read_pickle(fips_dir + "county_fips.sav")

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
(Albers_SF["satae_majority_area"] == Albers_SF["state_1"]).sum()

# %%
(Albers_SF["satae_majority_area"] == Albers_SF["state_2"]).sum()

# %%
Albers_SF.shape

# %%
len(Albers_SF) - (Albers_SF["state_1"] == Albers_SF["state_2"]).sum()

# %%
(Albers_SF["state_1"] == Albers_SF["state_2"]).sum()

# %%
Albers_SF = pd.merge(Albers_SF, state_fips[["EW_meridian", "state_full"]], 
                     how="left", left_on="satae_majority_area", right_on="state_full")

Albers_SF.drop(columns=["state_full"], inplace=True)

print (Albers_SF.shape)
Albers_SF.head(3)

# %%
Albers_SF_west = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF_west.shape

# %%
bpszone_ANPP.head(2)

# %%
# I think Min mentioned that FID is the same as Min_statID
# So, let us subset the west metidians
bpszone_ANPP_west = bpszone_ANPP[bpszone_ANPP["fid"].isin(list(Albers_SF_west["Min_statID"]))].copy()

print (bpszone_ANPP.shape)
print (bpszone_ANPP_west.shape)

print (len(bpszone_ANPP) - len(bpszone_ANPP_west))

# %%
unique_number_of_years = {}

for a_fid in bpszone_ANPP_west.fid.unique():
    LL = str(len(bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%
# {'39_years': 22430,
#  '40_years': 4332,
#  '38_years': 447,
#  '37_years': 4,
#  '35_years': 16}

# %%
bpszone_ANPP_west.head(2)

# %%
cols_ = ["Min_statID", "satae_majority_area", "state_1", "state_2", "EW_meridian"]
bpszone_ANPP_west = pd.merge(bpszone_ANPP_west, Albers_SF[cols_], 
                             how="left", left_on = "fid", right_on="Min_statID")
bpszone_ANPP_west.head(2)

# %%
if  (bpszone_ANPP_west["fid"] == bpszone_ANPP_west["Min_statID"]).sum() == len(bpszone_ANPP_west):
    bpszone_ANPP_west.drop(columns=["Min_statID"], inplace=True)

bpszone_ANPP_west.head(2)

# %%
num_locs = len(bpszone_ANPP_west["fid"].unique())
num_locs

# %%
Matt_diff_idea = bpszone_ANPP_west[["fid", "satae_majority_area", "state_1", "state_2", "EW_meridian"]].copy()

print (Matt_diff_idea.shape)

Matt_diff_idea.drop_duplicates(inplace=True)
Matt_diff_idea.reset_index(drop=True, inplace=True)

print (Matt_diff_idea.shape)
Matt_diff_idea.head(3)

# %%
# %%time
## Check if each ID has unique state_1, state_2, and state_majority_area
bad_FIDs = []
for a_FID in Matt_diff_idea["fid"].unique():
    curr_df = Matt_diff_idea[Matt_diff_idea.fid == a_FID]
    if len(curr_df) > 1:
        print (a_FID)
        bad_FIDs += bad_FIDs + [a_FID]

# %%
Matt_diff_idea.head(2)

# %%
# Not all locations have the same number of data in them
# Lets just assume they do. The missing year
Matt_diff_idea["first_10_years_median_ANPP"] = -666
Matt_diff_idea["last_10_years_median_ANPP"]  = -666

# %%
# %%time
# Find median of first decare and last decade of ANPP
for a_FID in Matt_diff_idea["fid"].unique():
    curr_df = bpszone_ANPP_west[bpszone_ANPP_west["fid"] == a_FID]
    
    min_year = curr_df["year"].min()
    max_year = curr_df["year"].max()
    
    first_decade = curr_df[curr_df["year"] < min_year + 10]
    last_decade  = curr_df[curr_df["year"] > max_year - 10]
    
    Matt_diff_idea.loc[Matt_diff_idea["fid"] == a_FID, "first_10_years_median_ANPP"] = \
                                                first_decade['mean_lb_per_acr'].median()

    Matt_diff_idea.loc[Matt_diff_idea["fid"] == a_FID, "last_10_years_median_ANPP"] = \
                                                    last_decade['mean_lb_per_acr'].median()

# %%
Matt_diff_idea.head(4)

# %%
year_diff = bpszone_ANPP_west["year"].max() - bpszone_ANPP_west["year"].min()

Matt_diff_idea["medians_diff_ANPP"] = Matt_diff_idea["last_10_years_median_ANPP"] - \
                                 Matt_diff_idea["first_10_years_median_ANPP"]

Matt_diff_idea["medians_diff_slope_ANPP"] = Matt_diff_idea["medians_diff_ANPP"] / year_diff
Matt_diff_idea.head(2)

# %%
print (Matt_diff_idea["medians_diff_slope_ANPP"].min())
print (Matt_diff_idea["medians_diff_slope_ANPP"].max())

# %%
Matt_diff_idea[Matt_diff_idea["medians_diff_slope_ANPP"] < -19]

# %% [markdown]
# ### change as percenatge of first decade

# %%
Matt_diff_idea["median_ANPP_change_as_perc"] = (100 * Matt_diff_idea["medians_diff_ANPP"]) / \
                                                  Matt_diff_idea["first_10_years_median_ANPP"]

Matt_diff_idea.head(2)

# %%

# %%
