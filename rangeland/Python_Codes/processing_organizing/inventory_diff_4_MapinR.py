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
import shutup
shutup.please()

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

plots_dir = data_dir_base + "plots/"

# %%
SoI = ["Alabama", "Arizona", "Arkansas", "California", 
       "Colorado", "Florida", "Georgia", 
       "Idaho", "Illinois", "Iowa", 
       "Kansas", "Kentucky", "Louisiana", 
       "Mississippi", "Missouri", "Montana", 
       "Nebraska", "Nevada", "New Mexico", "North Dakota", 
       "Oklahoma", "Oregon", "South Dakota", 
       "Tennessee", "Texas", "Utah","Virginia", "Washington",
       "Wyoming"]

abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI_abb = []
for x in SoI:
    SoI_abb = SoI_abb + [abb_dict["full_2_abb"][x]]

# %%
len(SoI)

# %%
USDA_data = pd.read_pickle(reOrganized_dir + "USDA_data.sav")
print ("-----------  reading USDA data  -----------")
cattle_inventory = USDA_data["cattle_inventory"]
cattle_inventory = cattle_inventory[["year", "cattle_cow_beef_inventory", "county_fips"]]
cattle_inventory.head(2)

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]
county_fips = county_fips[["county_fips", "county_name", "state"]]

print (len(county_fips.state.unique()))
county_fips = county_fips[county_fips.state.isin(SoI_abb)]
county_fips.reset_index(drop=True, inplace=True)
print (len(county_fips.state.unique()))
county_fips.head(2)

# %%
cattle_inventory = pd.merge(cattle_inventory, county_fips, on = ["county_fips"], how = "left")
print (cattle_inventory.shape)
cattle_inventory.dropna(how='any', inplace=True)
print (cattle_inventory.shape)
len(cattle_inventory.state.unique())

cattle_inventory.head(2)

# %%
cattle_inventory_2002 = cattle_inventory[cattle_inventory.year == 2002].copy()
cattle_inventory_2017 = cattle_inventory[cattle_inventory.year == 2017].copy()

cattle_inventory_2002.reset_index(drop=True, inplace=True)
cattle_inventory_2017.reset_index(drop=True, inplace=True)

inv_col = "cattle_cow_beef_inventory"
cattle_inventory_2002.rename(columns={inv_col: "cattle_cow_beef_inven_2002"}, inplace=True)
cattle_inventory_2017.rename(columns={inv_col: "cattle_cow_beef_inven_2017"}, inplace=True)
cattle_inventory_2002.head(2)

# %%
invent_2002_2017outer = pd.merge(cattle_inventory_2002[["cattle_cow_beef_inven_2002", "county_fips"]], 
                            cattle_inventory_2017[["cattle_cow_beef_inven_2017", "county_fips"]], 
                            on = ["county_fips"], how = "outer")

print (invent_2002_2017outer.shape)
invent_2002_2017outer.dropna(how='any', inplace=True)
invent_2002_2017outer.reset_index(drop=True, inplace=True)
print (invent_2002_2017outer.shape)

invent_2002_2017outer.head(2)

# %%
invent_2002_2017 = pd.merge(cattle_inventory_2002[["cattle_cow_beef_inven_2002", "county_fips"]], 
                            cattle_inventory_2017[["cattle_cow_beef_inven_2017", "county_fips"]], 
                            on = ["county_fips"], how = "left")

print (invent_2002_2017.shape)
invent_2002_2017.dropna(how='any', inplace=True)
invent_2002_2017.reset_index(drop=True, inplace=True)
print (invent_2002_2017.shape)

invent_2002_2017.head(2)

# %%
invent_2002_2017.equals(invent_2002_2017outer)

# %%
invent_2002_2017["inv_change2002to2017"] = invent_2002_2017["cattle_cow_beef_inven_2017"] - \
                                                           invent_2002_2017["cattle_cow_beef_inven_2002"]

invent_2002_2017["inv_change2002to2017_asPerc"] = 100 * \
             invent_2002_2017["inv_change2002to2017"] / invent_2002_2017["cattle_cow_beef_inven_2002"]

invent_2002_2017["inv_change2002to2017_asPerc"] = invent_2002_2017["inv_change2002to2017_asPerc"].round(2)
invent_2002_2017.head(2)

# %%
out_name = data_dir_base + "data_4_plot/" + "inventory_AbsChange_2002to2017.csv"
invent_2002_2017.to_csv(out_name, index=False)

# %%
cattle_inventory.head(2)

# %% [markdown]
# # Compute as percentage (national share)

# %%
cattle_inventory_2002 = cattle_inventory[cattle_inventory.year == 2002].copy()
cattle_inventory_2017 = cattle_inventory[cattle_inventory.year == 2017].copy()

cattle_inventory_2002.reset_index(drop=True, inplace=True)
cattle_inventory_2017.reset_index(drop=True, inplace=True)

inv_col = "cattle_cow_beef_inventory"
cattle_inventory_2002.rename(columns={inv_col: "cattle_cow_beef_inven_2002"}, inplace=True)
cattle_inventory_2017.rename(columns={inv_col: "cattle_cow_beef_inven_2017"}, inplace=True)
cattle_inventory_2002.head(2)

# %%
cattle_inventory_2002_total = cattle_inventory_2002.cattle_cow_beef_inven_2002.sum()
cattle_inventory_2017_total = cattle_inventory_2017.cattle_cow_beef_inven_2017.sum()

# %%
cattle_inventory_2002["inv_2002_asPercShare"] = 100*\
                          (cattle_inventory_2002["cattle_cow_beef_inven_2002"]/cattle_inventory_2002_total)

cattle_inventory_2017["inv_2017_asPercShare"] = 100*\
                          (cattle_inventory_2017["cattle_cow_beef_inven_2017"]/cattle_inventory_2017_total)

cattle_inventory_2002.head(2)

# %%
invent_2002_2017_asPerc = pd.merge(cattle_inventory_2002[["inv_2002_asPercShare", "county_fips"]], 
                                   cattle_inventory_2017[["inv_2017_asPercShare", "county_fips"]], 
                                   on = ["county_fips"], how = "left")

invent_2002_2017_asPerc.head(2)

# %%
print (invent_2002_2017_asPerc.shape)
invent_2002_2017_asPerc.dropna(how='any', inplace=True)
print (invent_2002_2017_asPerc.shape)

# %%
invent_2002_2017_asPerc.head(2)

# %%
invent_2002_2017_asPerc["change_2002_2017_asPercShare"] = invent_2002_2017_asPerc["inv_2002_asPercShare"] - \
                                                          invent_2002_2017_asPerc["inv_2017_asPercShare"]
invent_2002_2017_asPerc.head(2)

# %%
out_name = data_dir_base + "data_4_plot/" + "inv_PercChangeShare_2002_2017.csv"
invent_2002_2017_asPerc.to_csv(out_name, index=False)

# %%
census_years = [2017, 2012, 2007, 2002]
census_years = [2017, 2012, 2007, 2002]
all_df_ = county_fips.copy()

small_yr = census_years.pop()
yr = 2017

# %%
cattle_inventory_smYr = cattle_inventory[cattle_inventory.year == small_yr].copy()
cattle_inventory_Yr = cattle_inventory[cattle_inventory.year == yr].copy()

cattle_inventory_smYr.reset_index(drop=True, inplace=True)
cattle_inventory_Yr.reset_index(drop=True, inplace=True)

inv_col = "cattle_cow_beef_inventory"
n_smallYr = "cattle_cow_beef_inven_" + str(small_yr)
n_Yr = "cattle_cow_beef_inven_" + str(yr)
cattle_inventory_smYr.rename(columns={inv_col: n_smallYr}, inplace=True)
cattle_inventory_Yr.rename(columns={inv_col: n_Yr}, inplace=True)

# %%
invent_smYr_Yr = pd.merge(cattle_inventory_2002[[n_smallYr, "county_fips"]], 
                                  cattle_inventory_2017[[n_Yr, "county_fips"]], 
                                  on = ["county_fips"], how = "outer")
invent_smYr_Yr.dropna(how='any', inplace=True)
invent_smYr_Yr.reset_index(drop=True, inplace=True)
invent_smYr_Yr.head(2)

# %%
change_colN = "inv_change" + str(small_yr) + "to" + str(yr)
invent_smYr_Yr[change_colN] = invent_smYr_Yr[n_Yr] - invent_smYr_Yr[n_smallYr]

invent_smYr_Yr[change_colN+"_asPerc"] = 100 * invent_smYr_Yr[change_colN] / invent_smYr_Yr[n_smallYr]
invent_smYr_Yr[change_colN+"_asPerc"] = invent_smYr_Yr[change_colN+"_asPerc"].round(2)

# %%
all_df_ = pd.merge(all_df_, invent_smYr_Yr, on=["county_fips"], how="outer")
all_df_.head(2)

# %%
yr=2012

# %%
cattle_inventory_smYr = cattle_inventory[cattle_inventory.year == small_yr].copy()
cattle_inventory_smYr.reset_index(drop=True, inplace=True)
n_smallYr = "cattle_cow_beef_inven_" + str(small_yr)
cattle_inventory_smYr.rename(columns={inv_col: n_smallYr}, inplace=True)
cattle_inventory_smYr.head(2)

# %%
yr

# %%
cattle_inventory_Yr = cattle_inventory[cattle_inventory.year == yr].copy()
cattle_inventory_Yr.reset_index(drop=True, inplace=True)
n_Yr = "cattle_cow_beef_inven_" + str(yr)
cattle_inventory_Yr.rename(columns={inv_col: n_Yr}, inplace=True)
cattle_inventory_Yr.head(2)

# %%

# %% [markdown]
# # For Kirti's panel

# %%
census_years = [2017, 2012, 2007, 2002]
all_df_ = county_fips.copy()
inv_col = "cattle_cow_beef_inventory"
while census_years:
    small_yr = census_years.pop()
    cattle_inventory_smYr = cattle_inventory[cattle_inventory.year == small_yr].copy()
    cattle_inventory_smYr.reset_index(drop=True, inplace=True)
    n_smallYr = "cattle_cow_beef_inven_" + str(small_yr)
    cattle_inventory_smYr.rename(columns={inv_col: n_smallYr}, inplace=True)
    for yr in census_years:
        cattle_inventory_Yr = cattle_inventory[cattle_inventory.year == yr].copy()
        cattle_inventory_Yr.reset_index(drop=True, inplace=True)
        n_Yr = "cattle_cow_beef_inven_" + str(yr)
        cattle_inventory_Yr.rename(columns={inv_col: n_Yr}, inplace=True)
        
        invent_smYr_Yr = pd.merge(cattle_inventory_smYr[[n_smallYr, "county_fips"]], 
                                  cattle_inventory_Yr[[n_Yr, "county_fips"]], 
                                  on = ["county_fips"], how = "outer")
        
        invent_smYr_Yr.dropna(how='any', inplace=True)
        invent_smYr_Yr.reset_index(drop=True, inplace=True)
        
        change_colN = "inv_change" + str(small_yr) + "to" + str(yr)
        invent_smYr_Yr[change_colN] = invent_smYr_Yr[n_Yr] - invent_smYr_Yr[n_smallYr]
        
        change_colN_asPerc = change_colN+"_asPerc"
        invent_smYr_Yr[change_colN_asPerc] = 100 * invent_smYr_Yr[change_colN] / invent_smYr_Yr[n_smallYr]
        invent_smYr_Yr[change_colN_asPerc] = invent_smYr_Yr[change_colN_asPerc].round(2)

        needed_cols = ["county_fips", change_colN, change_colN_asPerc]
        all_df_ = pd.merge(all_df_, invent_smYr_Yr[needed_cols], on=["county_fips"], how="outer")
        
# add individual years inventory.
census_years = [2017, 2012, 2007, 2002]
for yr in census_years:
    cattle_inventory_Yr = cattle_inventory[cattle_inventory.year == yr].copy()
    cattle_inventory_Yr.reset_index(drop=True, inplace=True)
    n_Yr = "cattle_cow_beef_inven_" + str(yr)
    cattle_inventory_Yr.rename(columns={inv_col: n_Yr}, inplace=True)
    all_df_ = pd.merge(all_df_, cattle_inventory_Yr[["county_fips", n_Yr]], on=["county_fips"], how="outer")

# %%
all_df_.head(2)

# %%
A = all_df_[invent_2002_2017.columns].copy()
A.dropna(how='any', inplace=True)
A.reset_index(drop=True, inplace=True)
A.head(2)

# %%
invent_2002_2017.sort_values(by=["county_fips"], inplace=True)
A.sort_values(by=["county_fips"], inplace=True)


# %%