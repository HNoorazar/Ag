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
# initiation by copying ```02_01_2024_county_inven_LongAvgNDVI_Cleaner_II.ipynb```
#
# We want to see max NDVI and its variation effect on models. It was not good when independents were averaged over 2001-2017 and ```y``` was 2017 inventory.

# %% [markdown]
# # ATTENTION!!!
#
# Inventory on county level comes from CENSUS. Thus, data for this is not annual.
#
# ```NPP``` and ```SW``` on county level comes from us. Thus, we have annual data for these.
#
# Hence, do not (left) merge inventory with ```NPP``` and ```SW```, otherwise, we miss a lot in
# the same data table! Keep them goddamn separate.

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
import statistics
import statsmodels.api as sm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

from datetime import datetime, date

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

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
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
snap_shot_year = 2017

# %%
df_OuterJoined_all = pd.read_pickle(reOrganized_dir + "county_data_and_normalData_OuterJoined.sav")
df_OuterJoined = df_OuterJoined_all["all_df"]
df_OuterJoined = df_OuterJoined[df_OuterJoined.Pallavi == "Y"]
df_OuterJoined.head(2)

# %% [markdown]
# ## NPP exist only after 2001!
#
# - MODIS covers 2001-2022.
# - GIMMS and AVHRR cover 1982-2013
#
#
# So let us use subset of cattle inventory from census

# %%
# what happened here? Why Pallavi disappeard?
# df_OuterJoined = df_OuterJoined[df_OuterJoined.Pallavi == "Y"] 
df_OuterJoined = df_OuterJoined[df_OuterJoined.year >= 2001]
df_OuterJoined = df_OuterJoined[df_OuterJoined.year <= 2017]

df_OuterJoined.reset_index(drop=True, inplace=True)

df_OuterJoined.head(2)

# %%
list(df_OuterJoined.columns)

# %%
df_OuterJoined["dangerEncy"] = df_OuterJoined["danger"] + df_OuterJoined["emergency"]
df_OuterJoined["s1_dangerEncy"] = (
    df_OuterJoined["s1_danger"] + df_OuterJoined["s1_emergency"]
)
df_OuterJoined["s2_dangerEncy"] = (
    df_OuterJoined["s2_danger"] + df_OuterJoined["s2_emergency"]
)
df_OuterJoined["s3_dangerEncy"] = (
    df_OuterJoined["s3_danger"] + df_OuterJoined["s3_emergency"]
)
df_OuterJoined["s4_dangerEncy"] = (
    df_OuterJoined["s4_danger"] + df_OuterJoined["s4_emergency"]
)

df_OuterJoined.head(2)

# %%
(df_OuterJoined.columns)

# %%
gimms_cols = [x for x in (df_OuterJoined.columns) if "gimms" in x]
avhrr_cols = [x for x in (df_OuterJoined.columns) if "avhrr" in x]

# %%
list(df_OuterJoined.columns)

# %%
LL = (
    gimms_cols
    + avhrr_cols
    + [
        "normal",
        "alert",
        "danger",
        "emergency",
        "s1_normal",
        "s1_alert",
        "s1_danger",
        "s1_emergency",
        "s2_normal",
        "s2_alert",
        "s2_danger",
        "s2_emergency",
        "s3_normal",
        "s3_alert",
        "s3_danger",
        "s3_emergency",
        "s4_normal",
        "s4_alert",
        "s4_danger",
        "s4_emergency",
    ]
)
df_OuterJoined.drop(labels=LL, axis=1, inplace=True)

# %%
(df_OuterJoined.columns)

# %% [markdown]
# ## Inventory

# %%
inventory_snap = df_OuterJoined[df_OuterJoined.year == snap_shot_year].copy()
inventory_snap = inventory_snap[["year", "county_fips", "inventory"]]
print(inventory_snap.shape)
inventory_snap.dropna(how="any", inplace=True)
print(inventory_snap.shape)
inventory_snap.reset_index(drop=True, inplace=True)
inventory_snap.head(2)

SnapInv_Pallavi_cnty_list = list(inventory_snap.county_fips.unique())

# %%
print(inventory_snap.shape)
inventory_snap.head(3)

# %%
#
#   Filter only the counties in the snapshot.
#
df_OuterJoined = df_OuterJoined[
    df_OuterJoined.county_fips.isin(SnapInv_Pallavi_cnty_list)
]

# %% [markdown]
# # WARNING.
#
# **Pallavi's filter shrunk 29 states to 22.**
#
#
# ### Since there are too many incomlete counties, lets just keep them!
#
# Let us keep it simple. inventory as a function of long average ```NPP``` and long average ```SW```.
# And then add heat stress and see what happens.
#
# Then we can add
#  - rangeland acre
#  - herb ratio
#  - irrigated hay %
#  - feed expense
#  - population
#  - slaughter

# %% [markdown]
# # Model Snapshot
# **for snapshot year Inventory and long run avg of independent variables**

# %%
inventory_snap.head(2)

# %% [markdown]
# Since ```NPP``` exist after 2001, we filter ```SW``` from 2001 as well. Otherwise, there is no other reason.

# %%
[x for x in sorted(df_OuterJoined.columns) if "ndvi" in x]

# %%
ndvi_cols = [x for x in sorted(df_OuterJoined.columns) if "ndvi" in x]
mean_ndvi_cols = [x for x in sorted(ndvi_cols) if "mean" in x]

sw_cols = [
    "S1_countyMean_total_precip",
    "S2_countyMean_total_precip",
    "S3_countyMean_total_precip",
    "S4_countyMean_total_precip",
    "S1_countyMean_avg_Tavg",
    "S2_countyMean_avg_Tavg",
    "S3_countyMean_avg_Tavg",
    "S4_countyMean_avg_Tavg",
]
sw_cols = [x.lower() for x in sw_cols]

heat_cols = ["dangerEncy"]

# %%
common_cols = ["county_fips", "year"]

# %%
ndvi_SW_heat = df_OuterJoined[common_cols + ndvi_cols + sw_cols + heat_cols].copy()
# do not drop NA as it will not affect taking mean
# NPP_SW_heat2.dropna(how="any", inplace=True)
ndvi_SW_heat.sort_values(by=["year", "county_fips"], inplace=True)
ndvi_SW_heat.reset_index(drop=True, inplace=True)
ndvi_SW_heat.head(2)

# %%
print(len(ndvi_SW_heat.county_fips.unique()))

# %%
# ndvi_SW_heat_avg = ndvi_SW_heat.groupby("county_fips").mean()
# ndvi_SW_heat_avg.reset_index(drop=False, inplace=True)
# ndvi_SW_heat_avg.drop(labels=["year"], axis=1, inplace=True)
# ndvi_SW_heat_avg = ndvi_SW_heat_avg.round(3)
# ndvi_SW_heat_avg.head(3)

# %% [markdown]
# # Model (time, finally)

# %%
ndvi_SW_heat.sort_values(by=["county_fips", "year"], inplace=True)
# ndvi_SW_heat_avg.sort_values(by=["county_fips"], inplace=True)
inventory_snap.sort_values(by=["county_fips", "year"], inplace=True)
inventory_snap.head(2)

# %%
# print(ndvi_SW_heat_avg.shape)
print(inventory_snap.shape)

# %% [markdown]
# # CAREFUL
#
# Let us merge average data and inventory data. Keep in mind that the year will be 2017 but the data are averaged over the years (except for the inventory).
#
# Merge them so that counties are in the same order. My sick mind!

# %%
inventory_snap.head(2)

# %%
inv_snap_ndvi_SW_heat = pd.merge(
    inventory_snap, ndvi_SW_heat, on=["county_fips", "year"], how="left"
)
inv_snap_ndvi_SW_heat.head(2)

# %% [markdown]
# ### Normalize

# %%
all_df_normalized_ = df_OuterJoined_all["all_df_normalized"]
all_df_normalized_ = all_df_normalized_[all_df_normalized_["Pallavi"]=="Y"].copy()

all_df_normalized_ = all_df_normalized_[(all_df_normalized_.year>= 2001) & (all_df_normalized_.year<= 2017)].copy()
all_df_normalized_.shape

# %%

# %%
HS_var = ["dangerEncy"]
# AW_vars = ['yr_countyMean_total_precip', 'annual_avg_Tavg']

all_indp_vars = list(set(HS_var + sw_cols + ndvi_cols))  # AW_vars
all_indp_vars = sorted(all_indp_vars)
all_indp_vars

# %%
inv_snap_ndvi_SW_heat_normal = inv_snap_ndvi_SW_heat.copy()
inv_snap_ndvi_SW_heat_normal[all_indp_vars] = (
    inv_snap_ndvi_SW_heat_normal[all_indp_vars]
    - inv_snap_ndvi_SW_heat_normal[all_indp_vars].mean()
) / inv_snap_ndvi_SW_heat_normal[all_indp_vars].std(ddof=1)
inv_snap_ndvi_SW_heat_normal.head(2)

# %% [markdown]
# # Model
#
# ### Inventory vs normal ```NDVI``` averaged over 2001-2017

# %%
inv_snap_ndvi_SW_heat_normal.head(2)

# %%
sorted(inv_snap_ndvi_SW_heat_normal.columns)

# %%
mean_ndvi_cols

# %%
inv_snap_ndvi_SW_heat_normal.dropna(subset=["max_ndvi_in_year_modis"], axis=0, inplace=True)

# %%
indp_vars = mean_ndvi_cols
y_var = "inventory"

#################################################################
X = inv_snap_ndvi_SW_heat_normal[indp_vars]
X = sm.add_constant(X)
Y = np.log(inv_snap_ndvi_SW_heat_normal[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
del (X, model, model_result)

# %%
max_ndvi_cols = ["max_ndvi_in_year_modis", "max_ndvi_month_modis", "ndvi_std_modis"]

# %%
indp_vars = max_ndvi_cols
y_var = "inventory"

#################################################################
X = inv_snap_ndvi_SW_heat_normal[indp_vars]
X = sm.add_constant(X)
Y = np.log(inv_snap_ndvi_SW_heat_normal[y_var].astype(float))
model = sm.OLS(Y, X)
model_result = model.fit()
model_result.summary()

# %%
del (X, model, model_result)

# %% [markdown]
# # Others (Controls)
#
#  - rangeland acre
#  - herb ratio
#  - irrigated hay %
#  - feed expense
#  - population
#  - slaughter

# %%
control_cols = (
    ["county_fips"]
    + ["year"]
    + ["rangeland_fraction", "rangeland_acre"]
    + ["herb_avg", "herb_area_acr"]
    + ["irr_hay_area", "irr_hay_as_perc"]
    + ["population"]
    + ["feed_expense", "slaughter"]
)

# %%
controls = df_OuterJoined[control_cols].copy()
controls.head(2)

# %% [markdown]
# ## RA is already satisfies Pallavi condition

# %% [markdown]
# ## One more layer of filter according to 2017 inventory

# %%
print(controls[~(controls.population.isna())].year.unique())
print(controls.dropna(how="any", inplace=False).shape)
print(len(controls.dropna(how="any", inplace=False).county_fips.unique()))

a = controls.dropna(how="any", inplace=False).county_fips.unique()
main_cnties = inv_snap_ndvi_SW_heat_normal.county_fips.unique()
len([x for x in main_cnties if x in a])

# %%
inv_snap_ndvi_SW_heat_normal.head(2)

# %% [markdown]
# # Normalize

# %%
controls["irr_hay_as_perc_categ"] = controls["irr_hay_as_perc"]

controls.loc[(controls.irr_hay_as_perc <= 6), "irr_hay_as_perc_categ"] = 0

controls.loc[
    (controls.irr_hay_as_perc > 6) & (controls.irr_hay_as_perc <= 96),
    "irr_hay_as_perc_categ",
] = 1

controls.loc[(controls.irr_hay_as_perc > 96), "irr_hay_as_perc_categ"] = 2

controls.head(2)

# %%
normalize_cols = [
    "population",
    "slaughter",
    "feed_expense",
    "herb_avg",
    "herb_area_acr",
    "rangeland_fraction",
    "rangeland_acre",
    "irr_hay_area",
    "irr_hay_as_perc",
]

controls_normal = controls.copy()

controls_normal[normalize_cols] = (
    controls_normal[normalize_cols] - controls_normal[normalize_cols].mean()
) / controls_normal[normalize_cols].std(ddof=1)
controls_normal.head(3)

# %%
inv_snap_ndvi_SW_heat_normal.head(2)

# %%
controls_normal_NATossed = controls_normal.dropna(how="any", inplace=False)
control_counties = sorted(list(controls_normal_NATossed.county_fips.unique()))
main_counties = sorted(list(controls_normal.county_fips.unique()))
control_counties == main_counties

A = [x for x in main_counties if not (x in control_counties)]
B = [x for x in control_counties if not (x in main_counties)]
print(f"{len(A)=}, {len(B)=}")

# %% [markdown]
# - NPP vs. log(inventory) (NPP is representative of RA and herb_ratio)
# - NPP and human population vs. log(inventory)
#
# - SW vs. log(inventory)
# - SW and RA and herb_ratio vs. log(inventory)
#
#
# Extras.
#
# - NPP and human population and slaughter vs. log(inventory)
# - NPP and human population and slaughter and irr_hay vs. log(inventory)
# - SW and RA and herb_ratio and irr_hay vs. log(inventory)
#
#
# ## NPP and Control

# %%
inv_snap_ndvi_SW_heat_normal.head(2)

# %%
controls_normal.head(2)

# %%
inv_snap_ndvi_SW_heat_normal.shape

# %%
controls_normal.head(2)

# %%
print(inv_snap_ndvi_SW_heat_normal.shape)
all_df = pd.merge(
    inv_snap_ndvi_SW_heat_normal,
    controls_normal,
    on=["county_fips", "year"],
    how="outer",
)
print(all_df.shape)
all_df.head(2)

# %%
len(all_df.columns)

# %%
all_df_normalized_.columns

# %%
aaa = [x for x in list(all_df.columns) if x in list(all_df_normalized_.columns)]
bbb = [x for x in list(all_df.columns) if not (x in aaa)]
len(aaa)

# %%
all_df_normalized_[aaa].shape

# %%
all_df.shape

# %%
print (len(all_df_normalized_["county_fips"].unique()))
print (len(all_df["county_fips"].unique()))

# %%
indp_vars = mean_ndvi_cols + ["rangeland_acre"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = max_ndvi_cols + ["rangeland_acre"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = mean_ndvi_cols + ["rangeland_acre", "herb_area_acr"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = max_ndvi_cols + ["rangeland_acre", "herb_area_acr"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = mean_ndvi_cols + ["rangeland_acre", "herb_area_acr", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = max_ndvi_cols + ["rangeland_acre", "herb_area_acr", "irr_hay_area"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
curr_all.dropna(how="any", inplace=True)

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = mean_ndvi_cols + ["rangeland_acre", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = max_ndvi_cols + ["rangeland_acre", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = mean_ndvi_cols + ["rangeland_acre", "herb_area_acr", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()


# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = max_ndvi_cols + ["rangeland_acre", "herb_area_acr", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = mean_ndvi_cols + ["rangeland_acre", "irr_hay_area", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = max_ndvi_cols + ["rangeland_acre", "irr_hay_area", "population"]
y_var = "inventory"
#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = mean_ndvi_cols + [
    "rangeland_acre",
    "herb_area_acr",
    "irr_hay_area",
    "population",
]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = max_ndvi_cols + [
    "rangeland_acre",
    "herb_area_acr",
    "irr_hay_area",
    "population",
]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = mean_ndvi_cols + [
    "rangeland_acre",
    "herb_avg",
    "irr_hay_area",
    "population",
]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%
indp_vars = max_ndvi_cols + ["rangeland_acre", "herb_avg", "irr_hay_area", "population"]
y_var = "inventory"

#################################################################
curr_all = all_df[indp_vars + [y_var] + ["county_fips"]].copy()
print(len(curr_all.county_fips.unique()))
curr_all.dropna(how="any", inplace=True)
print(len(curr_all.county_fips.unique()))

X = curr_all[indp_vars]
X = sm.add_constant(X)
Y = np.log(curr_all[y_var].astype(float))

model_ = sm.OLS(Y, X)
model_result = model_.fit()
model_result.summary()

# %%
del (indp_vars, X, Y, model_, model_result, curr_all)

# %%

import pandas as pd

dir_ = "/Users/hn/Documents/01_research_data/RangeLand/Data/reOrganized/"
A = pd.read_pickle(dir_ + "state_data_and_normalData_OuterJoined.sav")
A.keys()

# %%
all_df = A["all_df"]
all_df.head(2)

# %%
list(all_df.columns)

# %%
NDVI = all_df[["year", "s1_max_modis_ndvi"]].copy()
NDVI.dropna(inplace=True, how="any")

# %%
NDVI.year.min()

# %%
B = pd.read_csv(
    "/Users/hn/Documents/01_research_data/RangeLand/Data/Min_Data/statefips_monthly_MODIS_NDVI.csv"
)
print(B.year.min())
print(B.year.max())

# %%
B = pd.read_csv(
    "/Users/hn/Documents/01_research_data/RangeLand/Data/Min_Data/statefips_monthly_GIMMS_NDVI.csv"
)
print(B.year.min())
print(B.year.max())

# %%
B = pd.read_csv(
    "/Users/hn/Documents/01_research_data/RangeLand/Data/Min_Data/statefips_monthly_AVHRR_NDVI.csv"
)
print(B.year.min())
print(B.year.max())

# %%
