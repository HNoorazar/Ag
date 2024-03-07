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
# ### State level:
#
# - [state_slaughter](https://quickstats.nass.usda.gov/#79E47847-EA4F-33E4-8665-5DBEC5AB1947)
#
# - [state_feed_cost](https://quickstats.nass.usda.gov/#333604E3-E7AA-3207-A5CA-E7D093D656C5)
#
# - [state_wetLand_area](https://quickstats.nass.usda.gov/#D00341ED-2A09-3F5E-85BC-4F36827B0EDF)
#
# - [state_AgLand](https://quickstats.nass.usda.gov/#6C3CEC1E-7829-336B-B3BD-7486E5A2C92F)
#
# - [state_FarmOperation](https://quickstats.nass.usda.gov/#212EA12D-A220-3650-A70E-C30A2317B1D7)
#
# - [Hay Prices. March 6, 2024.](https://quickstats.nass.usda.gov/#4CF5F365-9FA1-3248-A7E2-DDBAE1E247B2)
#
#
# ### County level:
#
# - [Feed expense by county, 1997-2017:](https://quickstats.nass.usda.gov/#EF899E9D-F162-3655-89D9-5C423132E97F)
#
# - [Acres enrolled in Conservation Reserve, Wetlands Reserve, Farmable Wetlands, or Conservation Reserve Enhancement Programs, 1997-2017](https://quickstats.nass.usda.gov/#3A734A89-9829-3674-AFC6-C4764DF7B728)
#
# - [Number of farm operations 1997-2017](https://quickstats.nass.usda.gov/#7310AC8E-D9CF-3BD9-8DC7-A4EF053FC56E)
#
# - [Irrigated acres and total land in farms by county, 1997-2017](https://quickstats.nass.usda.gov/#B2688D70-61AC-3E14-AA15-11882355E95E)
#
# __________________________________________________________________
#
#  - [Total Beef Cow inventory](https://quickstats.nass.usda.gov/#ADD6AB04-62EF-3DBF-9F83-5C23977C6DC7)
#  - [Inventory of Beef Cows](https://quickstats.nass.usda.gov/#B8E64DC0-E63E-31EA-8058-3E86C0DCB74E)

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
import os

from datetime import datetime, date

import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"

Min_data_dir_base = data_dir_base + "Min_Data/"
Mike_dir = data_dir_base + "Mike/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %% [markdown]
# # First
# do annual_stateLevel_inventoryDeltas. This was a different notebook called ```annual_state_level_inventory_deltas.ipynb```

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

county_id_name_fips = abb_dict["county_fips"]
county_id_name_fips.head(2)

# %%
state_abb_state_fips = county_id_name_fips[["state", "state_fips", "EW_meridian"]]
state_abb_state_fips.drop_duplicates(inplace=True)
state_abb_state_fips.reset_index(drop=True, inplace=True)
print(state_abb_state_fips.shape)
state_abb_state_fips.head(2)

# %% [markdown]
# ### Read inventory
#
# ```Beef_Cows_fromCATINV.csv``` and ```Shannon_Beef_Cows_fromCATINV.csv``` are the same.

# %%
shannon_annual = pd.read_csv(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV.csv")
shannon_annual = shannon_annual[shannon_annual.state.isin(list(state_abb_state_fips.state.unique()))]
print(shannon_annual.state.unique())
shannon_annual.reset_index(drop=True, inplace=True)
shannon_annual.head(2)

# %% [markdown]
# ## Compute deltas

# %%
# form deltas: inventort(t+1) - inventory (t)
inv_deltas = (
    shannon_annual[list(shannon_annual.columns)[2:]].values
    - shannon_annual[list(shannon_annual.columns)[1:-1]].values
)

delta_columns = [(str(x) + "_" + str(x - 1)) for x in np.arange(1921, 2022)]
# form deltas dataframe
inventory_annual_deltas = pd.DataFrame(data=inv_deltas, columns=delta_columns)
inventory_annual_deltas["state"] = shannon_annual["state"]

# re-order columns
inventory_annual_deltas = inventory_annual_deltas[["state"] + delta_columns]
inventory_annual_deltas.head(2)

# %% [markdown]
# ## Compute Ratios

# %%
# form deltas: inventort(t+1) - inventory (t)
inv_ratios = (
    shannon_annual[list(shannon_annual.columns)[2:]].values
    / shannon_annual[list(shannon_annual.columns)[1:-1]].values
)

delta_columns = [(str(x) + "_" + str(x - 1)) for x in np.arange(1921, 2022)]
# form ratios dataframe
inventory_annual_ratios = pd.DataFrame(data=inv_ratios, columns=delta_columns)
inventory_annual_ratios["state"] = shannon_annual["state"]

# re-order columns
inventory_annual_ratios = inventory_annual_ratios[["state"] + delta_columns]
inventory_annual_ratios.head(2)

# %%
shannon_annual.head(2)

# %% [markdown]
# ### convert to tall format

# %%
inventory_deltas_tall = inventory_annual_deltas.melt('state', var_name='year', value_name='inventory_delta')
inventory_ratios_tall = inventory_annual_ratios.melt('state', var_name='year', value_name='inventory_ratio')
inventory_deltas_tall.head(2)

# %%
inventory_ratios_tall.head(2)

# %%
print(state_abb_state_fips.shape)
state_abb_state_fips.head(2)

# %%
inventory_annual_deltas = pd.merge(inventory_annual_deltas, state_abb_state_fips, on=["state"], how="left")
inventory_deltas_tall   = pd.merge(inventory_deltas_tall,   state_abb_state_fips, on=["state"], how="left")
inventory_annual_deltas.head(2)

# %%
inventory_annual_ratios = pd.merge(inventory_annual_ratios, state_abb_state_fips, on=["state"], how="left")
inventory_ratios_tall = pd.merge(inventory_ratios_tall, state_abb_state_fips, on=["state"], how="left")
inventory_annual_ratios.head(2)

# %%
# filename = reOrganized_dir + "Shannon_Beef_Cows_fromCATINV_deltas.sav"

# export_ = {"shannon_annual_inventory_deltas": inventory_annual_deltas,
#            "shannon_annual_inventory_deltas_tall": inventory_deltas_tall,

#            "shannon_annual_inventory_ratios" : inventory_annual_ratios,
#            "shannon_annual_inventory_ratios_tall" : inventory_ratios_tall,

#            "source_code" : "annual_state_level_inventory_deltas",
#            "Author": "HN",
#            "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# pickle.dump(export_, open(filename, 'wb'))

# %%

# %% [markdown]
# # Second: USDA data

# %%
USDA_files = sorted([x for x in os.listdir(NASS_downloads_state) if x.endswith(".csv")])
USDA_files

# %%
AgLand = pd.read_csv(NASS_downloads_state + "state_AgLand.csv")
FarmOperation = pd.read_csv(NASS_downloads_state + "state_FarmOperation.csv")
feed_expense = pd.read_csv(NASS_downloads_state + "state_feed_expense.csv")
slaughter = pd.read_csv(NASS_downloads_state + "state_slaughter.csv")
wetLand_area = pd.read_csv(NASS_downloads_state + "state_wetLand_area.csv")

# %% [markdown]
# # Annual data

# %%
HayPrice_Q1 = pd.read_csv(NASS_downloads_state + "HayPriceMarch62024Q1.csv")
print (HayPrice_Q1.shape)
HayPrice_Q1.head(2)

# %%
print (HayPrice_Q1.State.unique()[:4])
print (HayPrice_Q1.Period.unique())

# %%
HayPrice_Q1 = HayPrice_Q1[HayPrice_Q1.Period == "MARKETING YEAR"].copy()
HayPrice_Q1.reset_index(drop=True, inplace=True)
print (HayPrice_Q1.shape)

# %%

# %% [markdown]
# ### beef price
# beef price is on national scale and it has monthly data in it.
# Pick up ```MARKETING YEAR``` for now.

# %%
beef_price = pd.read_csv(Mike_dir + "Census_BeefPriceMikeMarch62024Email.csv")
print (beef_price.shape)
beef_price.head(2)

# %%
beef_price = beef_price[beef_price.Period == "MARKETING YEAR"].copy()
beef_price.reset_index(drop=True, inplace=True)
print (beef_price.shape)

# %%
feed_expense.head(2)

# %%
feed_expense.Year.unique()

# %%
AgLand.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
wetLand_area.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
FarmOperation.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
feed_expense.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
slaughter.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

beef_price.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
HayPrice_Q1.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

sorted(list(feed_expense.columns))

# %%
print(f"{AgLand.shape = }")
print(f"{wetLand_area.shape = }")
print(f"{FarmOperation.shape = }")
print(f"{feed_expense.shape = }")
print(f"{slaughter.shape = }")
print(f"{HayPrice_Q1.shape = }")

# %%
print((feed_expense.columns == AgLand.columns).all())
print((feed_expense.columns == wetLand_area.columns).all())
print((feed_expense.columns == FarmOperation.columns).all())
print((feed_expense.columns == slaughter.columns).all())

print((feed_expense.columns == beef_price.columns).all())
print((feed_expense.columns == HayPrice_Q1.columns).all())

# %%
print(AgLand.zip_code.unique())
print(wetLand_area.zip_code.unique())
print(feed_expense.zip_code.unique())
print(FarmOperation.zip_code.unique())
print(slaughter.zip_code.unique())
print(beef_price.zip_code.unique())
print(HayPrice_Q1.zip_code.unique())
print()
print(AgLand.week_ending.unique())
print(wetLand_area.week_ending.unique())
print(feed_expense.week_ending.unique())
print(FarmOperation.week_ending.unique())
print(slaughter.week_ending.unique())
print()
print(beef_price.week_ending.unique())
print(HayPrice_Q1.week_ending.unique())

# %%
print(AgLand.watershed.unique())
print(wetLand_area.watershed.unique())
print(feed_expense.watershed.unique())
print(FarmOperation.watershed.unique())
print(slaughter.watershed.unique())

print(beef_price.watershed.unique())
print(HayPrice_Q1.watershed.unique())
print()
print(AgLand.domain_category.unique())
print(wetLand_area.domain_category.unique())
print(feed_expense.domain_category.unique())
print(FarmOperation.domain_category.unique())
print(slaughter.domain_category.unique())
print()
print(beef_price.domain_category.unique())
print(HayPrice_Q1.domain_category.unique())

# %%
FarmOperation.state_ansi.unique()

# %%
print(AgLand.domain.unique())
print(wetLand_area.domain.unique())
print(feed_expense.domain.unique())
print(FarmOperation.domain.unique())
print(slaughter.domain.unique())

print(beef_price.domain.unique())
print(HayPrice_Q1.domain.unique())
print()

print(AgLand.watershed_code.unique())
print(wetLand_area.watershed_code.unique())
print(feed_expense.watershed_code.unique())
print(FarmOperation.watershed_code.unique())
print(slaughter.watershed_code.unique())
print()
print(beef_price.watershed_code.unique())
print(HayPrice_Q1.watershed_code.unique())

# %%

# %%
print(AgLand.watershed.unique())
print(wetLand_area.watershed.unique())
print(feed_expense.watershed.unique())
print(FarmOperation.watershed.unique())
print(slaughter.watershed.unique())
print(beef_price.watershed.unique())
print(HayPrice_Q1.watershed.unique())

# %%
print(AgLand.region.unique())
print(wetLand_area.region.unique())
print(feed_expense.region.unique())
print(FarmOperation.region.unique())
print(slaughter.region.unique())
print(beef_price.region.unique())
print(HayPrice_Q1.region.unique())

# %%
print(AgLand.program.unique())
print(wetLand_area.program.unique())
print(feed_expense.program.unique())
print(FarmOperation.program.unique())
print(slaughter.program.unique())
print(HayPrice_Q1.program.unique())
print(beef_price.program.unique())

# %%
print(AgLand.period.unique())
print(wetLand_area.period.unique())
print(feed_expense.period.unique())
print(FarmOperation.period.unique())
print(slaughter.period.unique())

print(HayPrice_Q1.period.unique())
print(beef_price.period.unique())
print()
print(AgLand.geo_level.unique())
print(wetLand_area.geo_level.unique())
print(feed_expense.geo_level.unique())
print(FarmOperation.geo_level.unique())
print(slaughter.geo_level.unique())

print(HayPrice_Q1.geo_level.unique())
print(beef_price.geo_level.unique())

# %%
print(AgLand.data_item.unique())
print(wetLand_area.data_item.unique())
print(feed_expense.data_item.unique())
print(FarmOperation.data_item.unique())
print(slaughter.data_item.unique())
print ()
print(HayPrice_Q1.data_item.unique())
print(beef_price.data_item.unique())

# %%

# %%
AgLand.columns

# %%
print(AgLand.county.unique())
print(wetLand_area.county.unique())
print(feed_expense.county.unique())
print(FarmOperation.county.unique())
print(slaughter.county.unique())

print(HayPrice_Q1.county.unique())
print(beef_price.county.unique())

# %%
print(AgLand.county_ansi.unique())
print(wetLand_area.county_ansi.unique())
print(feed_expense.county_ansi.unique())
print(FarmOperation.county_ansi.unique())
print(slaughter.county_ansi.unique())

print(HayPrice_Q1.county_ansi.unique())
print(beef_price.county_ansi.unique())

# %%
print(AgLand.ag_district_code.unique())
print(wetLand_area.ag_district_code.unique())
print(feed_expense.ag_district_code.unique())
print(FarmOperation.ag_district_code.unique())
print(slaughter.ag_district_code.unique())
print(HayPrice_Q1.ag_district_code.unique())
print(beef_price.ag_district_code.unique())

print ()
print(AgLand.ag_district.unique())
print(wetLand_area.ag_district.unique())
print(feed_expense.ag_district.unique())
print(FarmOperation.ag_district.unique())
print(slaughter.ag_district.unique())
print(HayPrice_Q1.ag_district.unique())
print(beef_price.ag_district.unique())

# %%

# %%
bad_cols = [
    "watershed",
    "watershed_code",
    "domain",
    "domain_category",
    "ag_district",
    "ag_district_code",
    "region",
    "period",
    "county",
    "county_ansi",
    "week_ending",
    "zip_code",
    "program",
    "geo_level",
]

meta_cols = ["state", "state_ansi"]

# %%
FarmOperation["state_ansi"] = FarmOperation["state_ansi"].astype("int32")
FarmOperation["state_ansi"] = FarmOperation["state_ansi"].astype("str")
FarmOperation.state = FarmOperation.state.str.title()

for idx in FarmOperation.index:
    if len(FarmOperation.loc[idx, "state_ansi"]) == 1:
        FarmOperation.loc[idx, "state_ansi"] = (
            "0" + FarmOperation.loc[idx, "state_ansi"]
        )
        
FarmOperation[["state", "state_ansi"]].head(5)

# %% [markdown]
# # Alaska
# has problem with ansi's

# %%
meta_DF = FarmOperation[meta_cols].copy()
meta_DF.head(2)

# %%
print(f"{meta_DF.shape = }")
print(f"{meta_DF.drop_duplicates().shape = }")

# %%
meta_DF.drop_duplicates(inplace=True)
meta_DF.head(2)

# %%
## We do not need this, do we?
# meta_DF.to_csv(reOrganized_dir + "state_USDA_NASS_Census_metadata.csv", index=False)

# %%
AgLand.head(2)

# %%
AgLand.drop(bad_cols, axis="columns", inplace=True)
wetLand_area.drop(bad_cols, axis="columns", inplace=True)
feed_expense.drop(bad_cols, axis="columns", inplace=True)
FarmOperation.drop(bad_cols, axis="columns", inplace=True)
slaughter.drop(bad_cols, axis="columns", inplace=True)

HayPrice_Q1.drop(bad_cols, axis="columns", inplace=True)
beef_price.drop(bad_cols, axis="columns", inplace=True)

# %%
AgLand.head(2)

# %%
feed_expense[(feed_expense.state == "CALIFORNIA")]

# %%
AgLand["state_ansi"] = AgLand["state_ansi"].astype("int32")
AgLand["state_ansi"] = AgLand["state_ansi"].astype("str")
AgLand.state = AgLand.state.str.title()

for idx in AgLand.index:
    if len(AgLand.loc[idx, "state_ansi"]) == 1:
        AgLand.loc[idx, "state_ansi"] = "0" + AgLand.loc[idx, "state_ansi"]

AgLand[["state", "state_ansi"]].head(5)

# %%
wetLand_area["state_ansi"] = wetLand_area["state_ansi"].astype("int32")
wetLand_area["state_ansi"] = wetLand_area["state_ansi"].astype("str")
wetLand_area.state = wetLand_area.state.str.title()


for idx in wetLand_area.index:
    if len(wetLand_area.loc[idx, "state_ansi"]) == 1:
        wetLand_area.loc[idx, "state_ansi"] = "0" + wetLand_area.loc[idx, "state_ansi"]

wetLand_area[["state", "state_ansi"]].head(5)

# %%
feed_expense["state_ansi"] = feed_expense["state_ansi"].astype("int32")
feed_expense["state_ansi"] = feed_expense["state_ansi"].astype("str")

feed_expense.state = feed_expense.state.str.title()

for idx in feed_expense.index:
    if len(feed_expense.loc[idx, "state_ansi"]) == 1:
        feed_expense.loc[idx, "state_ansi"] = "0" + feed_expense.loc[idx, "state_ansi"]

feed_expense[["state", "state_ansi"]].head(5)

# %%
slaughter["state_ansi"] = slaughter["state_ansi"].astype("int32")
slaughter["state_ansi"] = slaughter["state_ansi"].astype("str")
slaughter.state = slaughter.state.str.title()

for idx in slaughter.index:
    if len(slaughter.loc[idx, "state_ansi"]) == 1:
        slaughter.loc[idx, "state_ansi"] = "0" + slaughter.loc[idx, "state_ansi"]

slaughter[["state", "state_ansi"]].head(5)

# %%
HayPrice_Q1["state_ansi"] = HayPrice_Q1["state_ansi"].astype("int32")
HayPrice_Q1["state_ansi"] = HayPrice_Q1["state_ansi"].astype("str")
HayPrice_Q1.state = HayPrice_Q1.state.str.title()


for idx in HayPrice_Q1.index:
    if len(HayPrice_Q1.loc[idx, "state_ansi"]) == 1:
        HayPrice_Q1.loc[idx, "state_ansi"] = "0" + HayPrice_Q1.loc[idx, "state_ansi"]

HayPrice_Q1[["state", "state_ansi"]].head(5)

HayPrice_Q1.head(2)

# %% [markdown]
# # beef price is national scale
# So, no ```["state_ansi"]```

# %%
print (beef_price.state.unique())
beef_price.head(2)

# %%
slaughter.head(2)

# %%
feed_expense[(feed_expense.state == "Alabama")]

# %%
print(slaughter.data_item.unique())
slaughter.head(2)

# %%

# %%
AgLand.rename(columns={"value": "AgLand", "cv_(%)": "AgLand_cv_(%)"},
                    inplace=True)

feed_expense.rename(columns={"value": "feed_expense", "cv_(%)": "feed_expense_cv_(%)"}, inplace=True)

wetLand_area.rename(columns={"value": "CRP_wetLand_acr", "cv_(%)": "CRP_wetLand_acr_cv_(%)"},
                    inplace=True)

slaughter.rename(columns={"value": "sale_4_slaughter_head",
                          "cv_(%)": "sale_4_slaughter_head_cv_(%)"},
                 inplace=True)

FarmOperation.rename(columns={"state_ansi": "state_fips",
                              "value": "number_of_farm_operation",
                              "cv_(%)": "number_of_farm_operation_cv_(%)"},
                     inplace=True)

beef_price.rename(columns={"state_ansi": "state_fips",
                           "value": "beef_price", "cv_(%)": "beef_price_cv_(%)"}, inplace=True)

HayPrice_Q1.rename(columns={"state_ansi": "state_fips",
                            "value": "hay_price", "cv_(%)": "hay_price_cv_(%)"}, inplace=True)

slaughter.head(2)

# %%
feed_expense.head(2)

# %%
wetLand_area.head(2)

# %%
wetLand_area.head(2)

# %%
wetLand_area.CRP_wetLand_acr.dtype

# %%

# %%
print(AgLand.shape)
AgLand = rc.clean_census(df=AgLand, col_="AgLand")
print(AgLand.shape)

# %%
print(wetLand_area.shape)
wetLand_area = rc.clean_census(df=wetLand_area, col_="crp_wetLand_acr")
print(wetLand_area.shape)

# %%
print(slaughter.shape)
slaughter = rc.clean_census(df=slaughter, col_="sale_4_slaughter_head")
print(slaughter.shape)

# %%
print(feed_expense.shape)
feed_expense = rc.clean_census(df=feed_expense, col_="feed_expense")
print(feed_expense.shape)

# %%
print(FarmOperation.shape)
FarmOperation = rc.clean_census(df=FarmOperation, col_="number_of_farm_operation")
print(FarmOperation.shape)

# %%
print(beef_price.shape)
beef_price = rc.clean_census(df=beef_price, col_="beef_price")
print(beef_price.shape)

# %%
HayPrice_Q1.head(2)

# %%
print(HayPrice_Q1.shape)
HayPrice_Q1 = rc.clean_census(df=HayPrice_Q1, col_="hay_price")
print(HayPrice_Q1.shape)

# %% [markdown]
# ### Beef inventory

# %%
shannon_annual = pd.merge(shannon_annual,
                          state_abb_state_fips[["state", "state_fips"]],
                          on=["state"], how="left")
shannon_annual.head(2)

# %%

# %%
AgLand.rename(columns={"state_ansi": "state_fips"}, inplace=True)

AgLand = AgLand[["year", "state_fips", "data_item", "agland", "agland_cv_(%)"]]
AgLand.head(2)

# %%
wetLand_area.rename(columns={"state_ansi": "state_fips"}, inplace=True)
wetLand_area = wetLand_area[["year", "state_fips", "data_item", "crp_wetland_acr", "crp_wetland_acr_cv_(%)"]]
wetLand_area.head(2)

# %%
print(feed_expense.data_item.unique())
feed_expense.rename(columns={"state_ansi": "state_fips"}, inplace=True)
feed_expense = feed_expense[["year", "state_fips", "data_item", "feed_expense", "feed_expense_cv_(%)"]]
feed_expense.head(2)

# %%
print(FarmOperation.data_item.unique())
FarmOperation = FarmOperation[["year", "state_fips", "data_item",
                               "number_of_farm_operation",
                               "number_of_farm_operation_cv_(%)"]]
FarmOperation.head(2)

# %%
print(slaughter.commodity.unique())
print(slaughter.data_item.unique())
slaughter.rename(columns={"state_ansi": "state_fips"}, inplace=True)
slaughter = slaughter[["year", "state_fips", "data_item",
                       "sale_4_slaughter_head", "sale_4_slaughter_head_cv_(%)"]]

slaughter.head(2)

# %%
shannon_annual.head(2)

# %%
shannon_Beef_Cows_fromCATINV_tall = pd.read_pickle(reOrganized_dir + "Shannon_Beef_Cows_fromCATINV_tall.sav")
print(shannon_Beef_Cows_fromCATINV_tall.keys())
shannon_Beef_Cows_fromCATINV_tall = shannon_Beef_Cows_fromCATINV_tall["CATINV_annual_tall"]
shannon_Beef_Cows_fromCATINV_tall.head(2)

# %%
shannon_Beef_Cows_fromCATINV_tall.head(2)

# %% [markdown]
# # Adjust prices

# %%
PPIACO = pd.read_csv(Mike_dir + "PPIACO.csv")
PPIACO.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
PPIACO['year'] = pd.to_datetime(PPIACO['date']).dt.year
# PPIACO['year'] = PPIACO['date'].str.slice(start=0, stop=4).astype(int)
PPIACO.head(3)

# %%
annual_PPIACO = PPIACO.groupby(["year"])["ppiaco"].mean().reset_index().round(2)
annual_PPIACO.rename(columns={"ppiaco": "annual_mean_ppiaco"}, inplace=True)
annual_PPIACO.head(2)

# %%
beef_price_at_1982  = beef_price.copy()
HayPrice_Q1_at_1982 = HayPrice_Q1.copy()

# %%
beef_price_at_1982 =  pd.merge(beef_price_at_1982, annual_PPIACO, on=["year"], how="left")
HayPrice_Q1_at_1982 = pd.merge(HayPrice_Q1_at_1982, annual_PPIACO, on=["year"], how="left")

HayPrice_Q1_at_1982.head(2)

# %%
HayPrice_Q1_at_1982["hay_price_at_1982"] = 100 * (HayPrice_Q1_at_1982["hay_price"] / 
                                                             HayPrice_Q1_at_1982["annual_mean_ppiaco"])

beef_price_at_1982["beef_price_at_1982"] = 100 * (beef_price_at_1982["beef_price"] / 
                                                             beef_price_at_1982["annual_mean_ppiaco"])

# %%

# %%
filename = reOrganized_dir + "state_USDA_ShannonCattle.sav"

export_ = {"AgLand": AgLand,
           "wetLand_area": wetLand_area,
           "feed_expense": feed_expense,
           "FarmOperation": FarmOperation,
           "slaughter": slaughter,

           "shannon_invt": shannon_annual,
           "shannon_invt_deltas": inventory_annual_deltas,
           "shannon_invt_ratios": inventory_annual_ratios,

           "shannon_invt_tall": shannon_Beef_Cows_fromCATINV_tall,
           "shannon_invt_deltas_tall": inventory_deltas_tall,
           "shannon_invt_ratios_tall": inventory_ratios_tall,
           
           "national_beef_price" : beef_price,
           "HayPrice_Q1" : HayPrice_Q1,
           
           "hay_price_Q1_at_1982" : HayPrice_Q1_at_1982,
           "beef_price_at_1982" : beef_price_at_1982,
           
           
           "source_code": "00_state_clean_USDA_data_addFIPS_and_annual_stateLevel_inventoryDeltas",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          }

pickle.dump(export_, open(filename, "wb"))

# %%
shannon_annual.head(2)

# %%
feed_expense.head(2)

# %%

# %%

# %%

# %%
