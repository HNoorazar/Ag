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
# # May 2
#
# Mike wants to run stuff with some software.
#
#
# Mail said:
#
# inventories by state by year and annual state rangeland productivity (NDVI I guess) in a csv file

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
from scipy.linalg import inv

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_name_fips = pd.DataFrame({"state_full" : list(abb_dict["full_2_abb"].keys()),
                                "state" : list(abb_dict["full_2_abb"].values())})


state_name_fips.head(2)

# %%
abb_dict["state_fips"].head(2)

# %%

# %%

# %%
state_name_fips = pd.merge(state_name_fips, 
                           abb_dict["state_fips"][["state_fips", "EW_meridian", "state"]], 
                           on=["state"], how="left")
state_name_fips.head(2)

# %%
state_fips_SoI = state_name_fips[state_name_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
state_fips_SoI.head(2)

# %%

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_df = all_data_dict["all_df_outerjoined"]
print (all_df.shape)
all_df.head(2)

# %%
test_inventory_yr = all_df[["year", "unit_matt_npp"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
test_inventory_yr = all_df[["year", "inventory"]].copy()
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
test_inventory_yr = all_df[["year", "max_ndvi_in_year_modis"]].copy()
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%
print (all_df.shape)

dummy_cols = [x for x in all_df.columns if "dumm" in x]
all_df.drop(columns = dummy_cols, inplace=True)

# %%
[x for x in all_df.columns if "npp" in x]

# %%
keep_cols = ['year', 'inventory', 'state_fips',
             'unit_matt_npp', 'total_matt_npp',
             'unit_matt_npp_std', 'total_matt_npp_std',
             'hay_price_at_1982', 'beef_price_at_1982',
             'rangeland_acre', 'max_ndvi_in_year_modis',
             'EW_meridian', 'herb_avg', 'herb_std']

print (all_df.year.min())
print (all_df.year.max())

all_df = all_df[keep_cols]
# all_df.dropna(subset = ["total_matt_npp"], inplace=True)

print (all_df.year.min())
print (all_df.year.max())

all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))]
all_df.reset_index(drop=True, inplace=True)
all_df.head(2)

# %%
# all_df.dropna(subset=["unit_matt_npp"], inplace=True)
print (all_df.shape)
all_df.reset_index(drop=True, inplace=True)

# %%
all_df.head(2)

# %%
(all_df["unit_matt_npp"] - (all_df["total_matt_npp"] / all_df["rangeland_acre"] )).unique()

# %%
sorted(list(all_df.columns))

# %%
all_df.rename(columns={"unit_matt_npp": "unit_matt_npp_lb_per_acr",
                       "total_matt_npp" : "total_matt_npp_lb"}, inplace=True)

sorted(list(all_df.columns))

# %%
all_df["rangeland_sq_kilometer"] = all_df["rangeland_acre"] * 0.0040468564
all_df["total_matt_npp_kg"] = all_df["total_matt_npp_lb"] * 0.45359237

all_df.head(2)

# %%
all_df["unit_matt_npp_kg_per_sq_kilometer"] = all_df["total_matt_npp_kg"] / all_df["rangeland_sq_kilometer"]
all_df.head(2)

# %%
print ((all_df["unit_matt_npp_kg_per_sq_kilometer"] - all_df["total_matt_npp_kg"] / 
       all_df["rangeland_sq_kilometer"]).unique())

# %%
all_df = pd.merge(all_df, abb_dict["state_fips"], how="left", on = "state_fips")
all_df.head(2)

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs
ax1.grid(axis="both", which="both")
ax2.grid(axis="both", which="both")


a_state = "Tennessee"
df = all_df[all_df.state_full == a_state].copy()
ax1.plot(df.year, df["unit_matt_npp_kg_per_sq_kilometer"]*80, linewidth=3, color="red");
ax1.plot(df.year, df["total_matt_npp_kg"], linewidth=3);

ax2.plot(df.year, df["total_matt_npp_kg"], linewidth=3);

# %%
all_df.shape

# %%
# converting to CSV file
all_df.to_csv(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.csv", index=False)

# %%
all_df.head(2)

# %%
filename = reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.sav"

export_ = {"NPP_NDVI_Invent_Mike_2May2024": all_df, 
           "source_code" : "subset_data_4_Mike",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%

# %%

# %%
print (all_df.year.min())
print (all_df.year.max())

# %%
test_inventory_yr = all_df[["year", "inventory"]]
test_inventory_yr.dropna(how="any", inplace=True)
print (test_inventory_yr.year.min())
print (test_inventory_yr.year.max())

# %%

# %%
# all_df = rc.convert_lb_2_kg(df=all_df, 
#                             matt_total_npp_col = "total_matt_npp", 
#                             new_col_name = "metricKg_total_matt_npp")

# [x for x in all_df.columns if "npp" in x]
# all_df.head(2)

# %%
# # all_df = rc.convert_lb_2_kg(df=all_df, 
# #                             matt_total_npp_col="unit_matt_npp", 
# #                             new_col_name="metric_unit_matt_npp")

# all_df = rc.convert_lbperAcr_2_kg_in_sqM(all_df, 
#                                          matt_unit_npp_col="unit_matt_npp", 
#                                          new_col_name="metricKg_sqMeter_unit_matt_npp")
# [x for x in all_df.columns if "npp" in x]

# %%

# %%

# %%

# %%

# %%
# 1 acre is 4046.85642199999983859016 m2
all_df["rangeland_area_sqMeter"] = all_df["rangeland_acre"] * 4046.85642199999983859016

# %%
all_df.head(2)

# %%
all_df2 = all_df.copy()
all_df2.dropna(subset = ["metricKg_sqMeter_unit_matt_npp"], inplace=True)
metric_npp_vect = (all_df2["metricKg_sqMeter_unit_matt_npp"] * all_df2["rangeland_area_sqMeter"]).values
metric_npp_vect

# %%
all_df = pd.merge(all_df, state_name_fips, on=["state_fips"], how="left")
all_df.head(2)

# %%
print (all_df[["year", "unit_matt_npp"]].dropna(how="any", inplace=False).year.max())
all_df[["year", "max_ndvi_in_year_modis"]].dropna(how="any", inplace=False).head(2)

# %%
print (format(15979574020, ",d"))

# %%
all_df.tail(2)

# %%
[x for x in all_df.columns if "npp" in x]

# %%
[x for x in list(all_df.columns) if "rangeland" in x]

# %%

# %%

# %%
reOrganized_dir

# %%
list(all_df.columns)

# %%

# %%

# %%

# %%
all_df.year.max()

# %%
all_df.head(2)

# %%
Mike_Dell_df = all_df[["year", "inventory", "rangeland_acre", "unit_matt_npp", "state"]].copy()
Mike_Dell_df.dropna(subset=["inventory", "unit_matt_npp"], inplace=True)
Mike_Dell_df.reset_index(drop=True, inplace=True)

Mike_Dell_df["inventory_div_RA"] = Mike_Dell_df['inventory'] / Mike_Dell_df['rangeland_acre']
Mike_Dell_df.head(2)

# %%
Mike_Dell_df_min = Mike_Dell_df.groupby(["state"])["unit_matt_npp"].min().reset_index().round(4)
Mike_Dell_df_min.rename(columns={"unit_matt_npp": "min_unit_matt_npp"}, inplace=True)

Mike_Dell_df_mean = Mike_Dell_df.groupby(["state"])["unit_matt_npp"].mean().reset_index().round(4)
Mike_Dell_df_mean.rename(columns={"unit_matt_npp": "mean_unit_matt_npp"}, inplace=True)

Mike_Dell_df_max = Mike_Dell_df.groupby(["state"])["unit_matt_npp"].max().reset_index().round(4)
Mike_Dell_df_max.rename(columns={"unit_matt_npp": "max_unit_matt_npp"}, inplace=True)

Mike_Dell_df_range = pd.merge(Mike_Dell_df_min, Mike_Dell_df_mean, on=["state"], how="left")
Mike_Dell_df_range = pd.merge(Mike_Dell_df_range, Mike_Dell_df_max, on=["state"], how="left")
Mike_Dell_df_range.head(2)

# %%
Mike_Dell_df_min = Mike_Dell_df.groupby(["state"])["inventory_div_RA"].min().reset_index().round(4)
Mike_Dell_df_min.rename(columns={"inventory_div_RA": "min_inventory_div_RA"}, inplace=True)
Mike_Dell_df_min

Mike_Dell_df_mean = Mike_Dell_df.groupby(["state"])["inventory_div_RA"].mean().reset_index().round(4)
Mike_Dell_df_mean.rename(columns={"inventory_div_RA": "mean_inventory_div_RA"}, inplace=True)
Mike_Dell_df_mean

Mike_Dell_df_max = Mike_Dell_df.groupby(["state"])["inventory_div_RA"].max().reset_index().round(4)
Mike_Dell_df_max.rename(columns={"inventory_div_RA": "max_inventory_div_RA"}, inplace=True)

Mike_Dell_df_range = pd.merge(Mike_Dell_df_min, Mike_Dell_df_mean, on=["state"], how="left")
Mike_Dell_df_range = pd.merge(Mike_Dell_df_range, Mike_Dell_df_max, on=["state"], how="left")
Mike_Dell_df_range.head(2)

# %%
tick_legend_FontSize = 10

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
all_df.head(2)

# %%
all_df.dropna(subset=["max_ndvi_in_year_modis"])["year"].max()

# %%


# %%

# %%

# %%

# %%

# %%

# %%
