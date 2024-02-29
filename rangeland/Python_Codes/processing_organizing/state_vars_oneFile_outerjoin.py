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
# **<span style='color:red'>Feb 29, 2024 update</span>:** Use Min's state level data (not conversion of county to state-level)
#
# -------------------------------------------------------
# **Feb 8, 2024**
#
# **Forgotten lesson** Keep everything: ***all states, not just 25***

# %% [markdown]
# <font color='red'>red</font>, <span style='color:green'>green</span>, $\color{blue}{\text{blue}}$

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
census_population_dir = data_dir_base + "census/"
# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# USDA_data_dir = data_dir_base + "/NASS_downloads/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
# for bold print
start_b = "\033[1m"
end_b = "\033[0;0m"
print("This is " + start_b + "a_bold_text" + end_b + "!")

# %%
abb_dict = pd.read_pickle(param_dir + "state_abbreviations.sav")
SoI = abb_dict['SoI']
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %% [markdown]
# ## County Fips

# %%
county_fips = pd.read_pickle(reOrganized_dir + "county_fips.sav")
county_fips = county_fips["county_fips"]

print(f"{len(county_fips.state.unique()) = }")
county_fips = county_fips[county_fips.state.isin(SoI_abb)].copy()
county_fips.drop_duplicates(inplace=True)
county_fips.reset_index(drop=True, inplace=True)
print(f"{len(county_fips.state.unique()) = }")

county_fips = county_fips[["county_fips", "county_name", "state_fips", "EW"]]
print(f"{len(county_fips.state_fips.unique()) = }")

# %%
county_fips

# %%

# %% [markdown]
# ## Inventory

# %%
# read USDA data
USDA_data = pd.read_pickle(reOrganized_dir + "state_USDA_ShannonCattle.sav")
sorted(USDA_data.keys())

# %%
AgLand = USDA_data['AgLand']
FarmOperation = USDA_data['FarmOperation']

beef_fromCATINV_csv = USDA_data["beef_fromCATINV_csv"]
Shannon_beef_fromCATINV_deltas = USDA_data["Shannon_beef_fromCATINV_deltas"]
Shannon_cows_tall = USDA_data["Shannon_Beef_Cows_fromCATINV_tall"]

feed_expense = USDA_data['feed_expense']
slaughter = USDA_data['slaughter']
wetLand_area = USDA_data['wetLand_area']

# %%
slaughter.rename(columns={"sale_4_slaughter_head": "slaughter"}, inplace=True)

# %%
beef_fromCATINV_csv.head(2)

# %%
beef_fromCATINV_csv.state.unique()

# %%
Shannon_cows_tall.state.unique()

# %%
Shannon_cows_tall = Shannon_cows_tall[Shannon_cows_tall.state != "US"]
beef_fromCATINV_csv = beef_fromCATINV_csv[beef_fromCATINV_csv.state != "US"]
Shannon_cows_tall.state_fips.unique()

# %%
beef_fromCATINV_csv.state_fips.unique()

# %%
Shannon_cows_tall.reset_index(drop=True, inplace=True)
beef_fromCATINV_csv.reset_index(drop=True, inplace=True)

# %%
AgLand.head(2)

# %%
wetLand_area.head(2)

# %%
feed_expense.head(2)

# %%
FarmOperation.head(2)

# %% [markdown]
# ### Time-invariants
#
# - **Herb ratio** stuff
# - **Rangeland area** stuff
# - **Irr Hay** stuff

# %% [markdown]
# ### Herb

# %%
#####
##### The following 2 lines are old. Supriya, on Feb 8, created a new
##### file in which we have both herb and max NDVI DoY.
#####
# herb = pd.read_pickle(data_dir_base + "Supriya/Nov30_HerbRatio/county_herb_ratio.sav")
# herb = herb["county_herb_ratio"]
herb = pd.read_pickle(reOrganized_dir + "county_state_NDVIDOY_Herb.sav")
herb = herb['State_NDVIDOY_Herb']

herb.dropna(how="any", inplace=True)
herb.head(2)


## Compute total herb area.
herb = rc.compute_herbRatio_totalArea(herb)
herb.reset_index(drop=True, inplace=True)
herb = herb.round(3)

# herb = herb[["county_fips", "herb_avg", "herb_area_acr"]]
herb.drop(columns=["pixel_count"], inplace=True)
herb.drop(['state'], axis=1, inplace=True)
herb.head(2)

# %%
herb2 = pd.read_pickle(reOrganized_dir + "county_state_NDVIDOY_Herb.sav")
herb2.keys()

# %%
County_NDVIDOY_Herb = herb2['County_NDVIDOY_Herb']
County_NDVIDOY_Herb.head(2)

# %%
County_NDVIDOY_Herb["state_fips"] = County_NDVIDOY_Herb.county_fips.str.slice(start=0, stop=2)
County_NDVIDOY_Herb.head(2)
WA_County_NDVIDOY_Herb = County_NDVIDOY_Herb[County_NDVIDOY_Herb.state_fips == "53"]
sorted(WA_County_NDVIDOY_Herb.maxNDVI_DoY_countyMean.unique())

# %% [markdown]
# ### RA
#
# We need RA to convert unit NPP to total NPP.

# %%
RA = pd.read_csv(reOrganized_dir + "county_rangeland_and_totalarea_fraction.csv")
RA.rename(columns={"fips_id": "county_fips"}, inplace=True)
RA = rc.correct_Mins_county_6digitFIPS(df=RA, col_="county_fips")
print(f"{len(RA.county_fips.unique()) = }")
RA.reset_index(drop=True, inplace=True)

RA["state_fips"] = RA.county_fips.str.slice(start=0, stop=2)
RA.head(2)

# %%
Min_state_NPP = pd.read_csv(Min_data_base + "statefips_annual_MODIS_NPP.csv")
Min_state_NPP.rename(columns={"statefips90m": "state_fips", "NPP": "unit_npp"}, inplace=True)
Min_state_NPP["state_fips"] = Min_state_NPP["state_fips"].astype(str)

Min_state_NPP["state_fips"] = Min_state_NPP.state_fips.str.slice(start=1, stop=3)

Min_state_NPP.head(2)

# %%
print (f"{Min_state_NPP.year.min() = }")
print (f"{Min_state_NPP.year.max() = }")

# %%

# %%
cty_yr_npp = pd.read_csv(reOrganized_dir + "county_annual_GPP_NPP_productivity.csv")

cty_yr_npp.rename(columns={"county": "county_fips", 
                           "MODIS_NPP": "unit_npp"}, 
                  inplace=True)

cty_yr_npp = rc.correct_Mins_county_6digitFIPS(df=cty_yr_npp, col_="county_fips")

cty_yr_npp = cty_yr_npp[["year", "county_fips", "unit_npp"]]

# Some counties do not have unit NPPs
cty_yr_npp.dropna(subset=["unit_npp"], inplace=True)
cty_yr_npp.reset_index(drop=True, inplace=True)

cty_yr_npp = pd.merge(cty_yr_npp, RA[["county_fips", "rangeland_acre"]], on=["county_fips"], how="left")

cty_yr_npp.head(2)

# %%
cty_yr_npp = rc.covert_unitNPP_2_total(NPP_df=cty_yr_npp,
                                       npp_unit_col_="unit_npp",
                                       acr_area_col_="rangeland_acre",
                                       npp_area_col_="county_total_npp")

cty_yr_npp.head(2)

# %% [markdown]
# ## State-level NPPs
#
# **We have county-level unit NPP. We need to compute county-level total NPP, then sum them up to get statel-level total NPP, then divide by the area to get state-level unit-NPP**

# %%
state_yr_npp = cty_yr_npp.copy()

state_yr_npp["state_fips"] = state_yr_npp.county_fips.str.slice(start=0, stop=2)
state_yr_npp.head(2)

# %%
state_yr_npp.drop(labels=["county_fips", "unit_npp"], axis=1, inplace=True)
state_yr_npp = state_yr_npp.groupby(["state_fips", "year"]).sum().reset_index()
state_yr_npp.head(2)

# %%
state_yr_npp.rename(columns={"county_total_npp": "state_total_npp"}, inplace=True)
state_yr_npp.head(2)
# del(cty_yr_npp)

# %%
##### Compute state-level unit-NPP
state_yr_npp["state_unit_npp"] = state_yr_npp["state_total_npp"] / state_yr_npp["area_m2"]
state_yr_npp.head(2)

# %%
state_yr_npp[(state_yr_npp.state_fips=="04") & (state_yr_npp.year==2001)]

# %%
Min_state_NPP.head(2)

# %% [markdown]
# # <font color='red'>Warning</font>
#
# Min's RA file (```county_rangeland_and_totalarea_fraction.csv```) includes some counties that 
# are not present in NPP file. Thus, if you want total rangeland acre, filter based on NPP!!!
#
# Do NOT run the following cell... 
#  - inconsistency
#  - Already it is computed above

# %%
## Convert RA to state level
# RA_state = RA.copy()
# RA_state.drop(labels=["county_fips", "rangeland_fraction"], axis=1, inplace=True)
# RA_state = RA_state.groupby(["state_fips"]).sum().reset_index()

# RA_state.rename(columns={"county_area_acre": "state_area_acre"}, 
#                 inplace=True)
# RA_state.head(2)

# %% [markdown]
# ### Weather

# %%
filename = reOrganized_dir + "state_annual_avg_Tavg.sav"
state_yr_avg_Tavg = pd.read_pickle(filename)
state_yr_avg_Tavg = state_yr_avg_Tavg["annual_temp"]

state_yr_avg_Tavg.reset_index(drop=True, inplace=True)
state_yr_avg_Tavg.head(2)

# %%

# %%
filename = reOrganized_dir + "state_seasonal_temp_ppt_weighted.sav"
SW = pd.read_pickle(filename)
SW = SW["seasonal"]
SW.head(2)

# %%
SW.columns

# %%
seasonal_precip_vars = ["s1_statemean_total_precip", "s2_statemean_total_precip",
                        "s3_statemean_total_precip", "s4_statemean_total_precip"]

seasonal_temp_vars = ["s1_statemean_avg_tavg", "s2_statemean_avg_tavg",
                      "s3_statemean_avg_tavg", "s4_statemean_avg_tavg"]

SW_vars = seasonal_precip_vars + seasonal_temp_vars
for a_col in SW_vars:
    SW[a_col] = SW[a_col].astype(float)

SW["annual_statemean_total_precip"] = SW[seasonal_precip_vars].sum(axis=1)
# SW["annual_statemean_avg_tavg"]   = SW[seasonal_temp_vars].sum(axis=1)
# SW["annual_statemean_avg_tavg"]   = SW["annual_statemean_avg_tavg"]/4
SW = pd.merge(SW, state_yr_avg_Tavg, on=["state_fips", "year"], how="outer")
del state_yr_avg_Tavg
SW = SW.round(3)

SW.head(2)

# %%
### We already have done this before and have the data in
### SW.

# state_grid_mean_idx = pd.read_csv(Min_data_base + "statefips_gridmet_mean_indices.csv")
# state_grid_mean_idx.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

# state_grid_mean_idx.rename(columns={"statefips": "state_fips"}, inplace=True)

# state_grid_mean_idx["state_fips"] = state_grid_mean_idx["state_fips"].astype(str)
# state_grid_mean_idx["state_fips"] = state_grid_mean_idx["state_fips"].str.slice(start=1, stop=3)

# state_grid_mean_idx = state_grid_mean_idx[["year", "month", "state_fips", 
#                                            "normal", "alert", "danger", "emergency"]]


# for a_col in ["normal", "alert", "danger", "emergency"]:
#     state_grid_mean_idx[a_col] = state_grid_mean_idx[a_col].astype(float)

# state_grid_mean_idx.reset_index(drop=True, inplace=True)
# state_grid_mean_idx.head(2)

# state_grid_mean_idx["season"] = "s5"
# state_grid_mean_idx['season'] = np.where(state_grid_mean_idx['month'].isin([1, 2, 3]), 's1', 
#                                          state_grid_mean_idx.season)

# state_grid_mean_idx['season'] = np.where(state_grid_mean_idx['month'].isin([4, 5, 6, 7]), 's2', 
#                                          state_grid_mean_idx.season)

# state_grid_mean_idx['season'] = np.where(state_grid_mean_idx['month'].isin([8, 9]), 's3', 
#                                          state_grid_mean_idx.season)

# state_grid_mean_idx['season'] = np.where(state_grid_mean_idx['month'].isin([10, 11, 12]), 's4', 
#                                          state_grid_mean_idx.season)

# state_grid_mean_idx.head(2)

# state_grid_mean_idx = state_grid_mean_idx[["year",  "state_fips", "season", "danger", "emergency"]]
# state_grid_mean_idx = state_grid_mean_idx.groupby(["year", "state_fips", "season"]).sum().reset_index()
# state_grid_mean_idx.head(2)


# state_grid_mean_idx["dangerEncy"] = state_grid_mean_idx["danger"] + state_grid_mean_idx["emergency"]


# state_grid_mean_idx = state_grid_mean_idx.pivot(index=['year', 'state_fips'], 
#                                                 columns='season', values='dangerEncy')
# state_grid_mean_idx.reset_index(drop=False, inplace=True)
# state_grid_mean_idx.head(2)

# state_grid_mean_idx.rename(columns={"s1": "s1_dangerEncy", "s2": "s2_dangerEncy",
#                                     "s3": "s3_dangerEncy", "s4": "s4_dangerEncy"}, 
#                            inplace=True)
# state_grid_mean_idx.head(2)

# %% [markdown]
# # Others (Controls)
#
#  - herb ratio
#  - irrigated hay
#  - feed expense
#  - population
#  - slaughter

# %% [markdown]
# ### irrigated hay
#
# **Need to find 2012 and fill in some of those (D)/missing stuff**

# %%
irr_hay = pd.read_pickle(reOrganized_dir + "irr_hay.sav")
irr_hay = irr_hay["irr_hay_perc"]
irr_hay.head(2)
irr_hay.rename(columns={"value_irr": "irr_hay_area",
                        "value_total" : "total_area_irrHayRelated"}, inplace=True)
irr_hay = irr_hay[["county_fips", "irr_hay_area", "total_area_irrHayRelated"]]
irr_hay["state_fips"] = irr_hay.county_fips.str.slice(start=0, stop=2)

irr_hay.head(2)

# %%
### Check if irr_hay above is filtered by Pallavi or not
RA_Pallavi = pd.read_pickle(param_dir + "filtered_counties_RAsizePallavi.sav")
RA_Pallavi = RA_Pallavi["filtered_counties_29States"]
RA_Pallavi.head(2)

print (len(RA_Pallavi.county_fips))
print (len(irr_hay.county_fips))

# %%
irr_hay = irr_hay[["state_fips", "irr_hay_area", "total_area_irrHayRelated"]]
irr_hay.head(2)

# %%
irr_hay = irr_hay.groupby(["state_fips"]).sum().reset_index()
irr_hay.head(2)

# %%
irr_hay["irr_hay_as_perc"] = irr_hay["irr_hay_area"]*100/irr_hay["total_area_irrHayRelated"]
irr_hay.head(2)

# %%
feed_expense.head(2)

# %%
feed_expense = feed_expense[["year", "state_fips", "feed_expense"]]
feed_expense.head(2)

# %%
slaughter = slaughter[["year", "state_fips", "slaughter"]]
slaughter.head(2)

# %%
# Compute state-level population using county-level data we have!
human_population = pd.read_pickle(reOrganized_dir + "human_population.sav")
human_population = human_population["human_population"]

human_population["state_fips"] = human_population.county_fips.str.slice(start=0, stop=2)

# drop county_fips
human_population = human_population[["state_fips", "year", "population"]]

# state-level population
human_population = human_population.groupby(["state_fips", "year"]).sum().reset_index()
human_population.head(2)

# %%
filename = reOrganized_dir + "state_seasonal_ndvi_V2MinBased.sav"
seasonal_ndvi = pd.read_pickle(filename)
seasonal_ndvi = seasonal_ndvi["seasonal_ndvi"]
seasonal_ndvi.head(2)

# %% [markdown]
# ## State-level RA

# %%
RA.head(2)

# %% [markdown]
# # <font color='red'>Warning</font>
#
# ### FILTER RA based on the counties that are in county-level NPP?

# %%
RA_state = RA.copy()
RA_state = RA_state[["state_fips", "rangeland_acre", "county_area_acre"]]
RA_state = RA_state.groupby(["state_fips"]).sum().reset_index()
RA_state.rename(columns={"county_area_acre": "state_area_acre"}, inplace=True)
RA_state["rangeland_fraction"] = RA_state["rangeland_acre"] / RA_state["state_area_acre"]
RA_state.head(2)

# %%
AgLand.head(2)

# %%
FarmOperation.head(2)

# %%
RA_state.head(2)

# %%
SW.head(2)

# %%
feed_expense.head(2)

# %%
herb.head(2)

# %%
irr_hay.head(2)

# %%
human_population.head(2)

# %%
slaughter.head(2)

# %%
wetLand_area.head(2)

# %%
Shannon_cows_tall = Shannon_cows_tall[["year", "inventory", "state_fips"]]
Shannon_cows_tall.head(2)

# %%
Shannon_beef_fromCATINV_deltas.head(2)

# %%
beef_fromCATINV_csv.head(2)

# %%
seasonal_ndvi.head(2)

# %%
SW.head(2)

# %%
SW['annual_statemean_total_danger'] = SW['s1_statemean_total_danger'] + \
                                      SW['s2_statemean_total_danger'] + \
                                      SW['s3_statemean_total_danger'] + \
                                      SW['s4_statemean_total_danger']

SW['annual_statemean_total_emergency'] = SW['s1_statemean_total_emergency'] + \
                                         SW['s2_statemean_total_emergency'] + \
                                         SW['s3_statemean_total_emergency'] + \
                                         SW['s4_statemean_total_emergency']

# %%

# %% [markdown]
# ### Subset to states of interest at the end!

# %%
filename = reOrganized_dir + "state_data_forOuterJoin.sav"

export_ = {
    "AgLand": AgLand,
    "FarmOperation": FarmOperation,
    "RA": RA_state,
    "SW": SW,
    "SoI": SoI,
    "SoI_abb": SoI_abb,
    "abb_dict": abb_dict,
    "npp": state_yr_npp,
    "Min_state_NPP" : Min_state_NPP,
    "feed_expense": feed_expense,
    "herb": herb,
    "irr_hay": irr_hay,
    "human_population": human_population,
    "slaughter": slaughter,
    "wetLand_area": wetLand_area,
    "beef_fromCATINV_csv": beef_fromCATINV_csv,
    "Shannon_beef_fromCATINV_deltas" : Shannon_beef_fromCATINV_deltas,
    "Shannon_Beef_Cows_fromCATINV_tall" : Shannon_cows_tall,
    "seasonal_ndvi": seasonal_ndvi,
    "source_code": "state_vars_oneFile_outerjoin",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %% [markdown]
# ## Do the outer join and normalize and save in another file
#
# #### First do variables that change over time then constant variables such as rangeland area.

# %%
AgLand.head(2)

# %% [markdown]
# ### Time-invariant Variables

# %%
RA_state.head(2)

# %%
herb.head(2)

# %%
irr_hay.head(2)

# %%
constants = pd.merge(RA_state, herb, on=["state_fips"], how="outer")
constants = pd.merge(constants, irr_hay, on=["state_fips"], how="outer")
constants.head(2)

# %% [markdown]
# ### Annual variables

# %%
## Leave Agland out for now. Since we do not know how we want to use it.
print(len(AgLand.data_item.unique()))
AgLand = AgLand[["state_fips", "year", "AgLand"]]
AgLand.head(2)

# %%
FarmOperation.head(2)

# %%
FarmOperation = FarmOperation[["state_fips", "year", "number_of_farm_operation"]]
FarmOperation.head(2)

# %%
FarmOperation.head(2)

# %%
Min_state_NPP.head(2)

# %%
state_yr_npp = state_yr_npp[["state_fips", "year", "state_unit_npp", "state_total_npp"]]
state_yr_npp.head(2)

# %%
feed_expense.head(2)

# %%
human_population.head(2)

# %%
slaughter.head(2)

# %%
Shannon_cows_tall.head(2)

# %%
beef_fromCATINV_csv.head(2)

# %%
wetLand_area = wetLand_area[["state_fips", "year", "crp_wetland_acr"]]
wetLand_area.head(2)

# %%
annual_outer = pd.merge(Shannon_cows_tall, state_yr_npp, on=["state_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, slaughter, on=["state_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, human_population, on=["state_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, feed_expense, on=["state_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, FarmOperation, on=["state_fips", "year"], how="outer")
annual_outer = pd.merge(annual_outer, wetLand_area, on=["state_fips", "year"], how="outer")
annual_outer.head(2)

# %% [markdown]
# ### Seasonal variables
# #### Seasonal ```NDVI``` to be added.

# %%
SW.head(2)

# %%
all_df = pd.merge(annual_outer, SW, on=["state_fips", "year"], how="outer")
all_df.head(2)

# %%
all_df = pd.merge(all_df, constants, on=["state_fips"], how="outer")
all_df.head(2)

# %%

# %%
all_df = pd.merge(all_df, seasonal_ndvi, on=["state_fips", "year"], how="outer")
print(all_df.shape)
all_df.head(2)

# %% [markdown]
# # normalization, normalize

# %%
print (len(list(all_df.columns)))

# %%
state_fips_ = county_fips.copy()
state_fips_ = state_fips_[["state_fips", "EW"]]
state_fips_.drop_duplicates(inplace=True)
state_fips_.reset_index(drop=True, inplace=True)
print (f"{state_fips_.shape = }")
state_fips_.head(2)

# %%
all_df = pd.merge(all_df, state_fips_, on=["state_fips"], how="left")

# %%
# This is the one came from TIFF file!
all_df[all_df.state_fips == "01"].maxNDVI_DoY_stateMean.unique()

# %%
list(all_df.columns)

# %%
non_normal_cols = [
    'year',
    'state_fips',
    'inventory',
    'EW',
    'max_ndvi_month_avhrr',
    'max_ndvi_season_avhrr',
    'max_ndvi_season_avhrr_s1',
    'max_ndvi_season_avhrr_s2',
    'max_ndvi_season_avhrr_s3',
    'max_ndvi_season_avhrr_s4',
    'max_ndvi_month_gimms',
    'max_ndvi_season_gimms',
    'max_ndvi_season_gimms_s1',
    'max_ndvi_season_gimms_s2',
    'max_ndvi_season_gimms_s3',
    'max_ndvi_season_gimms_s4',
    'max_ndvi_month_modis',
    'max_ndvi_season_modis',
    'max_ndvi_season_modis_s1',
    'max_ndvi_season_modis_s2',
    'max_ndvi_season_modis_s3',
    'max_ndvi_season_modis_s4']

numeric_cols = [x for x in sorted(all_df.columns) if not (x in non_normal_cols)]

# %%
numeric_cols

# %%
all_df_normalized = all_df.copy()

all_df_normalized[numeric_cols] = (all_df_normalized[numeric_cols] - all_df_normalized[numeric_cols].mean()) / \
                                   all_df_normalized[numeric_cols].std(ddof=1)
all_df_normalized[all_df_normalized.year==2002].head(2)

# %%
all_df[all_df.year == 2002].head(2)

# %%
filename = reOrganized_dir + "state_data_and_normalData_OuterJoined.sav"

export_ = {
    "all_df": all_df,
    "all_df_normalized": all_df_normalized,
    "source_code": "state_vars_oneFile_outerjoin",
    "NOTE" : "state NPPs come from HN's computation, not statefips_annual_MODIS_NPP.csv",
    "normal_cols" : numeric_cols,
    "non_normal_cols" : non_normal_cols,

    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%
datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
numeric_cols

# %% [markdown]
# # Compute differences/deltas and then normalize

# %%
# %%
delta_cols = ['year',
              'state_fips',
              'inventory',
              'state_unit_npp',
              'state_total_npp',
              'slaughter',
              'population',
              'feed_expense',
              'number_of_farm_operation',
              'crp_wetland_acr',
              's1_statemean_total_precip',
              's2_statemean_total_precip',
              's3_statemean_total_precip',
              's4_statemean_total_precip',
              's1_statemean_avg_tavg',
              's2_statemean_avg_tavg',
              's3_statemean_avg_tavg',
              's4_statemean_avg_tavg',
              'annual_statemean_total_precip',
              'annual_avg_tavg',

              'annual_statemean_total_danger',
              'annual_statemean_total_emergency',
              
              's1_statemean_total_danger',
              's2_statemean_total_danger',
              's3_statemean_total_danger',
              's4_statemean_total_danger',
              's1_statemean_total_emergency',
              's2_statemean_total_emergency',
              's3_statemean_total_emergency',
              's4_statemean_total_emergency',

              's1_mean_avhrr_ndvi',
              's2_mean_avhrr_ndvi',
              's3_mean_avhrr_ndvi',
              's4_mean_avhrr_ndvi',
              'max_ndvi_month_avhrr',
              'max_ndvi_in_year_avhrr',
              'max_ndvi_season_avhrr',
              'max_ndvi_season_avhrr_s1',
              'max_ndvi_season_avhrr_s2',
              'max_ndvi_season_avhrr_s3',
              'max_ndvi_season_avhrr_s4',
              's1_mean_gimms_ndvi',
              's2_mean_gimms_ndvi',
              's3_mean_gimms_ndvi',
              's4_mean_gimms_ndvi',
              'max_ndvi_month_gimms',
              'max_ndvi_in_year_gimms',
              'max_ndvi_season_gimms',
              'max_ndvi_season_gimms_s1',
              'max_ndvi_season_gimms_s2',
              'max_ndvi_season_gimms_s3',
              'max_ndvi_season_gimms_s4',
              's1_mean_modis_ndvi',
              's2_mean_modis_ndvi',
              's3_mean_modis_ndvi',
              's4_mean_modis_ndvi',
              'max_ndvi_month_modis',
              'max_ndvi_in_year_modis',
              'max_ndvi_season_modis',
              'max_ndvi_season_modis_s1',
              'max_ndvi_season_modis_s2',
              'max_ndvi_season_modis_s3',
              'max_ndvi_season_modis_s4',
              's1_sum_avhrr_ndvi',
              's2_sum_avhrr_ndvi',
              's3_sum_avhrr_ndvi',
              's4_sum_avhrr_ndvi',
              's1_sum_gimms_ndvi',
              's2_sum_gimms_ndvi',
              's3_sum_gimms_ndvi',
              's4_sum_gimms_ndvi',
              's1_sum_modis_ndvi',
              's2_sum_modis_ndvi',
              's3_sum_modis_ndvi',
              's4_sum_modis_ndvi',
              's1_max_avhrr_ndvi',
              's2_max_avhrr_ndvi',
              's3_max_avhrr_ndvi',
              's4_max_avhrr_ndvi',
              's1_max_gimms_ndvi',
              's2_max_gimms_ndvi',
              's3_max_gimms_ndvi',
              's4_max_gimms_ndvi',
              's1_max_modis_ndvi',
              's2_max_modis_ndvi',
              's3_max_modis_ndvi',
              's4_max_modis_ndvi']

# %%
non_delta_cols = ['year', 'state_fips'] + [x for x in all_df.columns if not (x in delta_cols)]
print (len(non_delta_cols))
non_delta_cols

# %%
len(set(non_delta_cols+delta_cols))

# %%
len(all_df.columns)

# %%
[x for x in delta_cols if not (x in all_df.columns)]

# %%
# %%time
ind_deltas_df = pd.DataFrame(columns=delta_cols)
ind_deltas_df.head(2)

for a_state in all_df.state_fips.unique():
    curr_df = all_df[all_df.state_fips == a_state].copy()
    curr_df.sort_values(by=["year"], inplace=True)
    
    curr_diff = curr_df.loc[curr_df.index[1:], delta_cols[2:]].values - \
                curr_df.loc[curr_df.index[:-1], delta_cols[2:]].values
    curr_diff = pd.DataFrame(curr_diff, columns=delta_cols[2:])
    curr_diff["year"] = curr_df.year[1:].values
    curr_diff["state_fips"] = a_state
    ind_deltas_df = pd.concat([ind_deltas_df, curr_diff])
    del(curr_diff)

# %%
nonDelta_df = all_df.copy()

nonDelta_df = nonDelta_df[non_delta_cols]
nonDelta_df.head(3)

# %%
delta_data = pd.merge(ind_deltas_df, nonDelta_df, on=["year", "state_fips"], how="left")
delta_data.head(3)

# %%
print (f"{all_df.shape = }")
print (f"{ind_deltas_df.shape = }")
print (f"{delta_data.shape = }")

# %%
for a_col in numeric_cols:
    delta_data[a_col] = delta_data[a_col].astype(float)
    

# %%

# %%
delta_data_normal = delta_data.copy()
delta_data_normal[numeric_cols] = (delta_data_normal[numeric_cols] - delta_data_normal[numeric_cols].mean()) / \
                                   delta_data_normal[numeric_cols].std(ddof=1)
delta_data_normal.head(2)

# %%
delta_data_normal[delta_data_normal.year == 2002].head(2)

# %%
delta_data[delta_data.year == 2002].head(2)

# %%

# %%
filename = reOrganized_dir + "state_delta_and_normalDelta_OuterJoined.sav"

export_ = {"delta_data": delta_data,
           "delta_data_normal": delta_data_normal,
           "delta_cols": delta_cols,
           "normalized_columns" : numeric_cols,
           "non_normalized_columns": non_normal_cols,
           "source_code": "state_vars_oneFile_outerjoin",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %%
