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
import gc
from shapely.geometry import Polygon
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
import rangeland_plot_core as rcp

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"
rangeland_bio_base = research_db + "RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
# min_bio_dir = rangeland_bio_data + "Min_Data/"
min_bio_dir_v11 = rangeland_bio_data + "Min_Data_v1.1/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

# %%
plot_what = "stats"  # "stats" or "ACF1" or "trends"
variable_set = "drought"

# %%
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(["lightcoral", "black"])
custom_cmap_BW = ListedColormap(["white", "black"])
cmap_G = cm.get_cmap("Greens")  # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap("Reds")

best_cmap_ = ListedColormap([(0.9, 0.9, 0.9), "black"])
best_cmap_ = ListedColormap(["grey", "white"])
fontdict_normal = {"family": "serif", "weight": "normal"}
fontdict_bold = {"family": "serif", "weight": "bold"}
inset_axes_ = [0.1, 0.13, 0.45, 0.03]
inset_axes_ = [0.1, 0.18, 0.45, 0.03]  # tight layout

# %%
county_fips_dict = pd.read_pickle(common_data + "county_fips.sav")
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
gdf = geopandas.read_file(common_data + "cb_2018_us_state_500k.zip")
gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

visframe = gdf.to_crs({"init": "epsg:5070"})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[
    ~visframe_mainLand_west.state.isin(["AK", "HI"])
].copy()

# %%
# a = ["et0_Apr", 
# "et0_Apr_detrendLinReg", 
# "et0_Apr_detrendSens", 
# "et0_Aug", 
# "et0_Aug_detrendLinReg", 
# "et0_Aug_detrendSens", 
# "et0_Dec", 
# "et0_Dec_detrendLinReg", 
# "et0_Dec_detrendSens", 
# "et0_Feb", 
# "et0_Feb_detrendLinReg", 
# "et0_Feb_detrendSens", 
# "et0_Jan", 
# "et0_Jan_detrendLinReg", 
# "et0_Jan_detrendSens", 
# "et0_Jul", 
# "et0_Jul_detrendLinReg", 
# "et0_Jul_detrendSens", 
# "et0_Jun", 
# "et0_Jun_detrendLinReg", 
# "et0_Jun_detrendSens", 
# "et0_Mar", 
# "et0_Mar_detrendLinReg", 
# "et0_Mar_detrendSens", 
# "et0_May", 
# "et0_May_detrendLinReg", 
# "et0_May_detrendSens", 
# "et0_Nov", 
# "et0_Nov_detrendLinReg", 
# "et0_Nov_detrendSens", 
# "et0_Oct", 
# "et0_Oct_detrendLinReg", 
# "et0_Oct_detrendSens", 
# "et0_Sep", 
# "et0_Sep_detrendLinReg", 
# "et0_Sep_detrendSens", 
# "et0_annualSum", 
# "et0_annualSum_detrendLinReg", 
# "et0_annualSum_detrendSens", 
# "etr_Apr", 
# "etr_Apr_detrendLinReg", 
# "etr_Apr_detrendSens", 
# "etr_Aug", 
# "etr_Aug_detrendLinReg", 
# "etr_Aug_detrendSens", 
# "etr_Dec", 
# "etr_Dec_detrendLinReg", 
# "etr_Dec_detrendSens", 
# "etr_Feb", 
# "etr_Feb_detrendLinReg", 
# "etr_Feb_detrendSens", 
# "etr_Jan", 
# "etr_Jan_detrendLinReg", 
# "etr_Jan_detrendSens", 
# "etr_Jul", 
# "etr_Jul_detrendLinReg", 
# "etr_Jul_detrendSens", 
# "etr_Jun", 
# "etr_Jun_detrendLinReg", 
# "etr_Jun_detrendSens", 
# "etr_Mar", 
# "etr_Mar_detrendLinReg", 
# "etr_Mar_detrendSens", 
# "etr_May", 
# "etr_May_detrendLinReg", 
# "etr_May_detrendSens", 
# "etr_Nov", 
# "etr_Nov_detrendLinReg", 
# "etr_Nov_detrendSens", 
# "etr_Oct", 
# "etr_Oct_detrendLinReg", 
# "etr_Oct_detrendSens", 
# "etr_Sep", 
# "etr_Sep_detrendLinReg", 
# "etr_Sep_detrendSens", 
# "etr_annualSum", 
# "etr_annualSum_detrendLinReg", 
# "etr_annualSum_detrendSens", 
# "spei_12mon_Apr", 
# "spei_12mon_Apr_detrendLinReg", 
# "spei_12mon_Apr_detrendSens", 
# "spei_12mon_Aug", 
# "spei_12mon_Aug_detrendLinReg", 
# "spei_12mon_Aug_detrendSens", 
# "spei_12mon_Dec", 
# "spei_12mon_Dec_detrendLinReg", 
# "spei_12mon_Dec_detrendSens", 
# "spei_12mon_Feb", 
# "spei_12mon_Feb_detrendLinReg", 
# "spei_12mon_Feb_detrendSens", 
# "spei_12mon_Jan", 
# "spei_12mon_Jan_detrendLinReg", 
# "spei_12mon_Jan_detrendSens", 
# "spei_12mon_Jul", 
# "spei_12mon_Jul_detrendLinReg", 
# "spei_12mon_Jul_detrendSens", 
# "spei_12mon_Jun", 
# "spei_12mon_Jun_detrendLinReg", 
# "spei_12mon_Jun_detrendSens", 
# "spei_12mon_Mar", 
# "spei_12mon_Mar_detrendLinReg", 
# "spei_12mon_Mar_detrendSens", 
# "spei_12mon_May", 
# "spei_12mon_May_detrendLinReg", 
# "spei_12mon_May_detrendSens", 
# "spei_12mon_Nov", 
# "spei_12mon_Nov_detrendLinReg", 
# "spei_12mon_Nov_detrendSens", 
# "spei_12mon_Oct", 
# "spei_12mon_Oct_detrendLinReg", 
# "spei_12mon_Oct_detrendSens", 
# "spei_12mon_Sep", 
# "spei_12mon_Sep_detrendLinReg", 
# "spei_12mon_Sep_detrendSens", 
# "spei_1mon_Apr", 
# "spei_1mon_Apr_detrendLinReg", 
# "spei_1mon_Apr_detrendSens", 
# "spei_1mon_Aug", 
# "spei_1mon_Aug_detrendLinReg", 
# "spei_1mon_Aug_detrendSens", 
# "spei_1mon_Dec", 
# "spei_1mon_Dec_detrendLinReg", 
# "spei_1mon_Dec_detrendSens", 
# "spei_1mon_Feb", 
# "spei_1mon_Feb_detrendLinReg", 
# "spei_1mon_Feb_detrendSens", 
# "spei_1mon_Jan", 
# "spei_1mon_Jan_detrendLinReg", 
# "spei_1mon_Jan_detrendSens", 
# "spei_1mon_Jul", 
# "spei_1mon_Jul_detrendLinReg", 
# "spei_1mon_Jul_detrendSens", 
# "spei_1mon_Jun", 
# "spei_1mon_Jun_detrendLinReg", 
# "spei_1mon_Jun_detrendSens", 
# "spei_1mon_Mar", 
# "spei_1mon_Mar_detrendLinReg", 
# "spei_1mon_Mar_detrendSens", 
# "spei_1mon_May", 
# "spei_1mon_May_detrendLinReg", 
# "spei_1mon_May_detrendSens", 
# "spei_1mon_Nov", 
# "spei_1mon_Nov_detrendLinReg", 
# "spei_1mon_Nov_detrendSens", 
# "spei_1mon_Oct", 
# "spei_1mon_Oct_detrendLinReg", 
# "spei_1mon_Oct_detrendSens", 
# "spei_1mon_Sep", 
# "spei_1mon_Sep_detrendLinReg", 
# "spei_1mon_Sep_detrendSens", 
# "spei_24mon_Apr", 
# "spei_24mon_Apr_detrendLinReg", 
# "spei_24mon_Apr_detrendSens", 
# "spei_24mon_Aug", 
# "spei_24mon_Aug_detrendLinReg", 
# "spei_24mon_Aug_detrendSens", 
# "spei_24mon_Dec", 
# "spei_24mon_Dec_detrendLinReg", 
# "spei_24mon_Dec_detrendSens", 
# "spei_24mon_Feb", 
# "spei_24mon_Feb_detrendLinReg", 
# "spei_24mon_Feb_detrendSens", 
# "spei_24mon_Jan", 
# "spei_24mon_Jan_detrendLinReg", 
# "spei_24mon_Jan_detrendSens", 
# "spei_24mon_Jul", 
# "spei_24mon_Jul_detrendLinReg", 
# "spei_24mon_Jul_detrendSens", 
# "spei_24mon_Jun", 
# "spei_24mon_Jun_detrendLinReg", 
# "spei_24mon_Jun_detrendSens", 
# "spei_24mon_Mar", 
# "spei_24mon_Mar_detrendLinReg", 
# "spei_24mon_Mar_detrendSens", 
# "spei_24mon_May", 
# "spei_24mon_May_detrendLinReg", 
# "spei_24mon_May_detrendSens", 
# "spei_24mon_Nov", 
# "spei_24mon_Nov_detrendLinReg", 
# "spei_24mon_Nov_detrendSens", 
# "spei_24mon_Oct", 
# "spei_24mon_Oct_detrendLinReg", 
# "spei_24mon_Oct_detrendSens", 
# "spei_24mon_Sep", 
# "spei_24mon_Sep_detrendLinReg", 
# "spei_24mon_Sep_detrendSens", 
# "spei_2mon_Apr", 
# "spei_2mon_Apr_detrendLinReg", 
# "spei_2mon_Apr_detrendSens", 
# "spei_2mon_Aug", 
# "spei_2mon_Aug_detrendLinReg", 
# "spei_2mon_Aug_detrendSens", 
# "spei_2mon_Dec", 
# "spei_2mon_Dec_detrendLinReg", 
# "spei_2mon_Dec_detrendSens", 
# "spei_2mon_Feb", 
# "spei_2mon_Feb_detrendLinReg", 
# "spei_2mon_Feb_detrendSens", 
# "spei_2mon_Jan", 
# "spei_2mon_Jan_detrendLinReg", 
# "spei_2mon_Jan_detrendSens", 
# "spei_2mon_Jul", 
# "spei_2mon_Jul_detrendLinReg", 
# "spei_2mon_Jul_detrendSens", 
# "spei_2mon_Jun", 
# "spei_2mon_Jun_detrendLinReg", 
# "spei_2mon_Jun_detrendSens", 
# "spei_2mon_Mar", 
# "spei_2mon_Mar_detrendLinReg", 
# "spei_2mon_Mar_detrendSens", 
# "spei_2mon_May", 
# "spei_2mon_May_detrendLinReg", 
# "spei_2mon_May_detrendSens", 
# "spei_2mon_Nov", 
# "spei_2mon_Nov_detrendLinReg", 
# "spei_2mon_Nov_detrendSens", 
# "spei_2mon_Oct", 
# "spei_2mon_Oct_detrendLinReg", 
# "spei_2mon_Oct_detrendSens", 
# "spei_2mon_Sep", 
# "spei_2mon_Sep_detrendLinReg", 
# "spei_2mon_Sep_detrendSens", 
# "spei_3mon_Apr", 
# "spei_3mon_Apr_detrendLinReg", 
# "spei_3mon_Apr_detrendSens", 
# "spei_3mon_Aug", 
# "spei_3mon_Aug_detrendLinReg", 
# "spei_3mon_Aug_detrendSens", 
# "spei_3mon_Dec", 
# "spei_3mon_Dec_detrendLinReg", 
# "spei_3mon_Dec_detrendSens", 
# "spei_3mon_Feb", 
# "spei_3mon_Feb_detrendLinReg", 
# "spei_3mon_Feb_detrendSens", 
# "spei_3mon_Jan", 
# "spei_3mon_Jan_detrendLinReg", 
# "spei_3mon_Jan_detrendSens", 
# "spei_3mon_Jul", 
# "spei_3mon_Jul_detrendLinReg", 
# "spei_3mon_Jul_detrendSens", 
# "spei_3mon_Jun", 
# "spei_3mon_Jun_detrendLinReg", 
# "spei_3mon_Jun_detrendSens", 
# "spei_3mon_Mar", 
# "spei_3mon_Mar_detrendLinReg", 
# "spei_3mon_Mar_detrendSens", 
# "spei_3mon_May", 
# "spei_3mon_May_detrendLinReg", 
# "spei_3mon_May_detrendSens", 
# "spei_3mon_Nov", 
# "spei_3mon_Nov_detrendLinReg", 
# "spei_3mon_Nov_detrendSens", 
# "spei_3mon_Oct", 
# "spei_3mon_Oct_detrendLinReg", 
# "spei_3mon_Oct_detrendSens", 
# "spei_3mon_Sep", 
# "spei_3mon_Sep_detrendLinReg", 
# "spei_3mon_Sep_detrendSens", 
# "spei_6mon_Apr", 
# "spei_6mon_Apr_detrendLinReg", 
# "spei_6mon_Apr_detrendSens", 
# "spei_6mon_Aug", 
# "spei_6mon_Aug_detrendLinReg", 
# "spei_6mon_Aug_detrendSens", 
# "spei_6mon_Dec", 
# "spei_6mon_Dec_detrendLinReg", 
# "spei_6mon_Dec_detrendSens", 
# "spei_6mon_Feb", 
# "spei_6mon_Feb_detrendLinReg", 
# "spei_6mon_Feb_detrendSens", 
# "spei_6mon_Jan", 
# "spei_6mon_Jan_detrendLinReg", 
# "spei_6mon_Jan_detrendSens", 
# "spei_6mon_Jul", 
# "spei_6mon_Jul_detrendLinReg", 
# "spei_6mon_Jul_detrendSens", 
# "spei_6mon_Jun", 
# "spei_6mon_Jun_detrendLinReg", 
# "spei_6mon_Jun_detrendSens", 
# "spei_6mon_Mar", 
# "spei_6mon_Mar_detrendLinReg", 
# "spei_6mon_Mar_detrendSens", 
# "spei_6mon_May", 
# "spei_6mon_May_detrendLinReg", 
# "spei_6mon_May_detrendSens", 
# "spei_6mon_Nov", 
# "spei_6mon_Nov_detrendLinReg", 
# "spei_6mon_Nov_detrendSens", 
# "spei_6mon_Oct", 
# "spei_6mon_Oct_detrendLinReg", 
# "spei_6mon_Oct_detrendSens", 
# "spei_6mon_Sep", 
# "spei_6mon_Sep_detrendLinReg", 
# "spei_6mon_Sep_detrendSens", 
# "spei_9mon_Apr", 
# "spei_9mon_Apr_detrendLinReg", 
# "spei_9mon_Apr_detrendSens", 
# "spei_9mon_Aug", 
# "spei_9mon_Aug_detrendLinReg", 
# "spei_9mon_Aug_detrendSens", 
# "spei_9mon_Dec", 
# "spei_9mon_Dec_detrendLinReg", 
# "spei_9mon_Dec_detrendSens", 
# "spei_9mon_Feb", 
# "spei_9mon_Feb_detrendLinReg", 
# "spei_9mon_Feb_detrendSens", 
# "spei_9mon_Jan", 
# "spei_9mon_Jan_detrendLinReg", 
# "spei_9mon_Jan_detrendSens", 
# "spei_9mon_Jul", 
# "spei_9mon_Jul_detrendLinReg", 
# "spei_9mon_Jul_detrendSens", 
# "spei_9mon_Jun", 
# "spei_9mon_Jun_detrendLinReg", 
# "spei_9mon_Jun_detrendSens", 
# "spei_9mon_Mar", 
# "spei_9mon_Mar_detrendLinReg", 
# "spei_9mon_Mar_detrendSens", 
# "spei_9mon_May", 
# "spei_9mon_May_detrendLinReg", 
# "spei_9mon_May_detrendSens", 
# "spei_9mon_Nov", 
# "spei_9mon_Nov_detrendLinReg", 
# "spei_9mon_Nov_detrendSens", 
# "spei_9mon_Oct", 
# "spei_9mon_Oct_detrendLinReg", 
# "spei_9mon_Oct_detrendSens", 
# "spei_9mon_Sep", 
# "spei_9mon_Sep_detrendLinReg", 
# "spei_9mon_Sep_detrendSens"]

# %%
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
Albers_SF.rename(
    columns={"minstatsid": "fid", "satae_max": "state_majority_area"}, inplace=True
)
Albers_SF.head(2)

## Focus only on West Meridian
print((Albers_SF["state_majority_area"] == Albers_SF["state_1"]).sum())
print((Albers_SF["state_majority_area"] == Albers_SF["state_2"]).sum())
print(Albers_SF.shape)
print(len(Albers_SF) - (Albers_SF["state_1"] == Albers_SF["state_2"]).sum())
print((Albers_SF["state_1"] == Albers_SF["state_2"]).sum())

# %%
Albers_SF = pd.merge(
    Albers_SF,
    state_fips[["EW_meridian", "state_full"]],
    how="left",
    left_on="state_majority_area",
    right_on="state_full",
)
Albers_SF.drop(columns=["state_full"], inplace=True)

print(Albers_SF.shape)
Albers_SF = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
print(Albers_SF.shape)
Albers_SF.head(2)


# %%
print(len(Albers_SF["fid"].unique()))
print(len(Albers_SF["value"].unique()))
print(len(Albers_SF["hucsgree_4"].unique()))

print((Albers_SF["hucsgree_4"] - Albers_SF["value"]).unique())
print((list(Albers_SF.index) == Albers_SF.fid).sum())

Albers_SF.drop(columns=["value"], inplace=True)
Albers_SF.head(2)
Albers_SF.drop(
    columns=[
        "hucsgree_4",
        "bps_code",
        "bps_model",
        "bps_name",
        "groupveg",
        "state_1",
        "state_2",
    ],
    inplace=True,
)

west_FIDs = list(Albers_SF["fid"])

# %%
filename = bio_reOrganized + "bpszone_annualWeatherByHN_and_deTrended.sav"
A = pd.read_pickle(filename)
bps_weather = A["bpszone_annual_weather_byHN"]
slopes_interceps = A["slopes_interceps"]

# %%
drought_indices_slopes_interceps = [
    x
    for x in slopes_interceps.columns
    if (("spei_" in x) or ("et0_" in x) or ("etr_" in x))
]
weather_indices_slopes_interceps = [
    x for x in slopes_interceps.columns if not (x in drought_indices_slopes_interceps)
]
drought_indices_bps_weather = [
    x for x in bps_weather.columns if (("spei_" in x) or ("et0_" in x) or ("etr_" in x))
]
weather_indices_bps_weather = [
    x for x in bps_weather.columns if not (x in drought_indices_bps_weather)
]
weather_indices_slopes_interceps = list(weather_indices_slopes_interceps)
weather_indices_bps_weather = list(weather_indices_bps_weather)
drought_indices_slopes_interceps = ["fid"] + list(drought_indices_slopes_interceps)
drought_indices_bps_weather = [
    "fid",
    "year",
    "state_majority_area",
    "EW_meridian",
] + list(drought_indices_bps_weather)

# %%
if variable_set == "drought":
    slopes_interceps = slopes_interceps[drought_indices_slopes_interceps]
    bps_weather = bps_weather[drought_indices_bps_weather]
elif variable_set == "weather":
    slopes_interceps = slopes_interceps[weather_indices_slopes_interceps]
    bps_weather = bps_weather[weather_indices_bps_weather]
else:
    print("Bad Condition! Exiting the script.")
    sys.exit()  # This will stop the script

# %%
intercept_cols = [x for x in slopes_interceps.columns if "intercep" in x]
slopes_interceps.drop(columns=intercept_cols, inplace=True)
slopes_interceps.head(2)

# %%
# %%time

if plot_what == "stats":
    bio_plots = bio_plots + "weather_longterm_stats/"
    os.makedirs(bio_plots, exist_ok=True)

    plotting_df = bps_weather.copy()
    print(plotting_df.shape)
    plotting_df = plotting_df[plotting_df["fid"].isin(west_FIDs)]
    plotting_df.reset_index(drop=True, inplace=True)
    print(plotting_df.shape)
    plotting_df.head(2)

    first_cols = ["fid", "year", "state_majority_area", "EW_meridian"]
    new_cols_order = first_cols + [
        col for col in plotting_df.columns if col not in first_cols
    ]
    plotting_df = plotting_df[new_cols_order]
    plotting_df.head(2)
    ####################################################
    ####
    ####     Compute statistics
    ####
    # stats = ['var', 'mean', 'std', 'min', 'max', 'median']
    df = plotting_df.copy()
    df.drop(columns=["year", "state_majority_area", "EW_meridian"], inplace=True)

    stats = ["min", "mean", "median", "max", "var", "std"]
    grouped_stats = df.groupby("fid").agg(stats).reset_index()

    grouped_stats.columns = [
        f"40year{stat}_{col}" for col, stat in grouped_stats.columns
    ]
    grouped_stats.rename(columns={"40year_fid": "fid"}, inplace=True)
    # # Flatten column MultiIndex
    # grouped_stats.columns = ['fid',
    #               'temp_min', 'temp_mean', 'temp_median', 'temp_max', 'temp_variance', 'temp_std',
    #               'precip_variance', 'precip_mean', 'precip_std', 'precip_min', 'precip_max', 'precip_median']

    # # Calculate coefficient of variation
    # grouped_stats['temp_cv_times100']   = 100 * (grouped_stats['temp_std'] / grouped_stats['temp_mean'])
    # grouped_stats['precip_cv_times100'] = 100 * (grouped_stats['precip_std'] / grouped_stats['precip_mean'])
    del df
    grouped_stats.head(3)

    mean_cols = [col for col in grouped_stats.columns if "40yearmean_" in col]
    std_cols = [col for col in grouped_stats.columns if "40yearstd_" in col]

    # Ensure alignment
    mean_cols_sorted = sorted(mean_cols, key=lambda x: x.split("_", 1)[1])
    std_cols_sorted = sorted(std_cols, key=lambda x: x.split("_", 1)[1])

    # Calculate CV
    cv_df = pd.DataFrame(index=grouped_stats.index)
    for mean_col, std_col in zip(mean_cols_sorted, std_cols_sorted):
        varname = mean_col.split("_", 1)[1]
        cv_col_name = f"40yearcv_{varname}_times100"
        cv_df[cv_col_name] = 100 * grouped_stats[std_col] / grouped_stats[mean_col]

    # Combine stats and CVs
    grouped_stats = pd.concat(
        [grouped_stats.reset_index(drop=True), cv_df.reset_index(drop=True)], axis=1
    )
    grouped_stats.head(3)

# %%
grouped_stats.head(3)

# %%
Albers_SF = pd.merge(Albers_SF, grouped_stats, how="left", on="fid")
Albers_SF.head(2)
###########################################################################################
#######
#######    Plot histograms
#######
font = {"size": 14}
matplotlib.rc("font", **font)
font_base = 10
params = {
    # "font.family": "Palatino",
    "legend.fontsize": font_base * 1,
    "axes.labelsize": font_base * 1.2,
    "axes.titlesize": font_base * 1.2,
    "xtick.labelsize": font_base * 1.1,
    "ytick.labelsize": font_base * 1.1,
    "axes.titlepad": 10,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.labelbottom": True,
    "ytick.labelleft": True,
    "axes.linewidth": 0.05,
}

plt.rcParams.update(params)
sharey_ = False
print(
    "number of variables to plot is below (done for histograms and maps and maps with no outliers): "
)
print(len(list(grouped_stats.columns)[1:]))

# %%
variable_ = "40yearmin_spei_12mon_Apr"

# %%
fig, axes = plt.subplots(
    1, 1, figsize=(10, 2), sharey=sharey_, sharex=True, dpi=dpi_
)
axes.grid(axis="y", alpha=0.7, zorder=0)

cleaned = Albers_SF[variable_].replace([np.inf, -np.inf], np.nan).dropna()
axes.hist(cleaned, zorder=3, bins=100, color="skyblue", edgecolor="black")

axes.set_title(f"{variable_}", color="k", fontdict=fontdict_bold)
axes.set_xlabel(f"{variable_}", fontdict=fontdict_normal)
axes.set_ylabel("count", fontdict=fontdict_normal)

# %%
a_percent = 10
perc_ = a_percent / 100
lower_bound = Albers_SF[variable_].quantile(perc_)
upper_bound = Albers_SF[variable_].quantile(1 - perc_)

# Filter rows between and outside outlier boundaries
filtered_between = Albers_SF[
    (Albers_SF[variable_] >= lower_bound)
    & (Albers_SF[variable_] <= upper_bound)
]
cleaned_between = (
    filtered_between[variable_].replace([np.inf, -np.inf], np.nan).dropna()
)

fig, axes = plt.subplots(
    1, 1, figsize=(10, 2), sharey=sharey_, sharex=True, dpi=dpi_
)
axes.grid(axis="y", alpha=0.7, zorder=0)

axes.hist(
    cleaned_between, zorder=3, bins=100, color="skyblue", edgecolor="black"
)

axes.set_title(
    rf"{variable_}. {a_percent}% tails removed",
    color="k",
    fontdict=fontdict_bold,
)
axes.set_xlabel(rf"{variable_}", fontdict=fontdict_normal)
axes.set_ylabel("count", fontdict=fontdict_normal);

# %%
font_base = 8
params = {
    "legend.fontsize": font_base * 0.7,
    "axes.labelsize": font_base * 0.71,
    "axes.titlesize": font_base * 1,
    "xtick.labelsize": font_base * 0.7,
    "ytick.labelsize": font_base * 0.7,
    "axes.titlepad": 5,
    "legend.handlelength": 2,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.labelbottom": False,
    "ytick.labelleft": False,
    "axes.linewidth": 0.05,
}
plt.rcParams.update(params)
fontdict_normal = {"family": "serif", "weight": "normal"}
fontdict_bold = {"family": "serif", "weight": "bold"}

# %%
y_var = variable_

# %%
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([])
ax.set_yticks([])

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=best_cmap_)

Albers_SF[y_var].replace([np.inf, -np.inf], np.nan, inplace=True)
cleaned = Albers_SF.dropna(subset=[y_var]).copy()

min_max = max(np.abs(cleaned[y_var].min()), np.abs(cleaned[y_var].max()))
norm = Normalize(vmin=-min_max, vmax=min_max, clip=True)

cent_plt = cleaned.plot(
    column=y_var, ax=ax, legend=False, cmap="seismic", norm=norm
)

cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(
    cent_plt.collections[1],
    ax=ax,
    orientation="horizontal",
    shrink=0.3,
    norm=norm,
    cax=cax,
)
cbar1.set_label(f"{y_var}", labelpad=1, fontdict=fontdict_normal, fontsize=7)
plt.title(f"{y_var}", fontdict=fontdict_bold, fontsize=7)

ax.set_aspect("equal", adjustable="box")
plt.tight_layout()

# %%
font_base = 10
params = {
    # "font.family": "Palatino",
    "legend.fontsize": font_base * 0.7,
    "axes.labelsize": font_base * 0.71,
    "axes.titlesize": font_base * 1,
    "xtick.labelsize": font_base * 0.7,
    "ytick.labelsize": font_base * 0.7,
    "axes.titlepad": 5,
    "legend.handlelength": 2,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.labelbottom": False,
    "ytick.labelleft": False,
    "axes.linewidth": 0.05,
}
plt.rcParams.update(params)
fontdict_normal = {"family": "serif", "weight": "normal"}
fontdict_bold = {"family": "serif", "weight": "bold"}
print("line 511")
font_c = 0.8
inset_axes_ = [0.15, 0.12, 0.45, 0.03] 

# %%
perc_ = a_percent / 100
# The following line is already done above.
# clean_series = Albers_SF[y_var].replace([np.inf, -np.inf], np.nan).dropna()
df_cleaned = Albers_SF.dropna(subset=[y_var])
clean_series = df_cleaned[y_var]
lower_bound = clean_series.quantile(perc_)
upper_bound = clean_series.quantile(1 - perc_)

# Filter rows between and outside outlier boundaries
lower_df = df_cleaned[df_cleaned[y_var] <= lower_bound]
between_df = df_cleaned[
    (df_cleaned[y_var] > lower_bound) & (df_cleaned[y_var] < upper_bound)
]
upper_df = df_cleaned[df_cleaned[y_var] >= upper_bound]

fig, axes = plt.subplots(
    1, 3, figsize=(12, 4), sharex=True, sharey=True, dpi=map_dpi_
)
(ax1, ax2, ax3) = axes
fig.subplots_adjust(
    top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.005, wspace=-0.3
)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
###############################################################
rcp.plot_SF(
    SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=best_cmap_
)
rcp.plot_SF(
    SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=best_cmap_
)
rcp.plot_SF(
    SF=visframe_mainLand_west, ax_=ax3, col="EW_meridian", cmap_=best_cmap_
)


cbar_label = rf"{y_var}"
###############################################################
###
### plot lower
###
min_max = max(np.abs(lower_df[y_var].min()), np.abs(lower_df[y_var].max()))
norm = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt1 = lower_df.plot(
    column=y_var, ax=ax1, legend=False, cmap="seismic", norm=norm
)
###############################################################
###
### plot between
###
min_max = np.abs(between_df[y_var]).max()
norm = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt2 = between_df.plot(
    column=y_var, ax=ax2, legend=False, cmap="seismic", norm=norm
)
###############################################################
###
### plot upper
###
min_max = np.abs(upper_df[y_var]).max()
norm = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt3 = upper_df.plot(
    column=y_var, ax=ax3, legend=False, cmap="seismic", norm=norm
)

cax = ax1.inset_axes(inset_axes_)
print("---------------------- 591 -----------------------")
print(len(cent_plt1.collections))
print(cent_plt1.collections)

try:
    cbar1 = fig.colorbar(
        cent_plt1.collections[1],
        ax=ax1,
        orientation="horizontal",
        shrink=0.3,
        cax=cax,
    )
    cbar1.ax.tick_params(labelsize=font_base * font_c)
    cbar1.set_label(
        cbar_label,
        labelpad=1,
        fontdict=fontdict_normal,
        fontsize=font_base * font_c,
    )
except IndexError:
    print(f"index error, {y_var = }")

ax1.set_title(rf"lower {a_percent}%", fontdict=fontdict_bold)



cax = ax2.inset_axes(inset_axes_)
try:
    cbar2 = fig.colorbar(
        cent_plt2.collections[1],
        ax=ax2,
        orientation="horizontal",
        shrink=0.3,
        cax=cax,
    )
    cbar2.ax.tick_params(labelsize=font_base * font_c)
    cbar2.set_label(
        cbar_label,
        labelpad=1,
        fontdict=fontdict_normal,
        fontsize=font_base * font_c,
    )
except IndexError:
    print(f"index error, {y_var = }")
ax2.set_title(
    r"between [{}%, {}%]".format(a_percent, 100 - a_percent),
    fontdict=fontdict_bold,
);


try:
    cax = ax3.inset_axes(inset_axes_)
    cbar3 = fig.colorbar(
        cent_plt3.collections[1],
        ax=ax3,
        orientation="horizontal",
        shrink=0.3,
        cax=cax,
    )
    cbar3.ax.tick_params(labelsize=font_base * font_c)
    cbar3.set_label(
        cbar_label,
        labelpad=1,
        fontdict=fontdict_normal,
        fontsize=font_base * font_c,
    )
except IndexError:
    print(f"index error, {y_var = }")
ax3.set_title(rf"upper {a_percent}%", fontdict=fontdict_bold)

# %%
perc_ = a_percent / 100
# The following line is already done above.
# clean_series = Albers_SF[y_var].replace([np.inf, -np.inf], np.nan).dropna()
df_cleaned = Albers_SF.dropna(subset=[y_var])
clean_series = df_cleaned[y_var]
lower_bound = clean_series.quantile(perc_)
upper_bound = clean_series.quantile(1 - perc_)

# Filter rows between and outside outlier boundaries
lower_df = df_cleaned[df_cleaned[y_var] <= lower_bound]
between_df = df_cleaned[
    (df_cleaned[y_var] > lower_bound) & (df_cleaned[y_var] < upper_bound)
]
upper_df = df_cleaned[df_cleaned[y_var] >= upper_bound]

fig, axes = plt.subplots(
    1, 3, figsize=(12, 4), sharex=True, sharey=True, dpi=map_dpi_
)
(ax1, ax2, ax3) = axes
fig.subplots_adjust(
    top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.005, wspace=-0.3
)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
###############################################################
rcp.plot_SF(
    SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=best_cmap_
)
rcp.plot_SF(
    SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=best_cmap_
)
rcp.plot_SF(
    SF=visframe_mainLand_west, ax_=ax3, col="EW_meridian", cmap_=best_cmap_
)

cbar_label = rf"{y_var}"
###############################################################
###
### plot lower
###
min_max = max(np.abs(lower_df[y_var].min()), np.abs(lower_df[y_var].max()))
norm = Normalize(vmin=-min_max, vmax=min_max, clip=True)
lower_plot = lower_df.plot(
    column=y_var, ax=ax1, legend=False, cmap="seismic", norm=norm
)
###############################################################
###
### plot between
###
min_max = np.abs(between_df[y_var]).max()
norm = Normalize(vmin=-min_max, vmax=min_max, clip=True)
between_plot = between_df.plot(
    column=y_var, ax=ax2, legend=False, cmap="seismic", norm=norm
)
###############################################################
###
### plot upper
###
min_max = np.abs(upper_df[y_var]).max()
norm = Normalize(vmin=-min_max, vmax=min_max, clip=True)
upper_plot = upper_df.plot(
    column=y_var, ax=ax3, legend=False, cmap="seismic", norm=norm
)
######################################################
cax = ax1.inset_axes(inset_axes_)
print("---------------------- 591 -----------------------")
print(len(lower_plot.collections))
print(lower_plot.collections)
try:
    lower_bar = fig.colorbar(
        lower_plot.collections[1],
        ax=ax1,
        orientation="horizontal",
        shrink=0.3,
        cax=cax,
    )
    lower_bar.ax.tick_params(labelsize=font_base * font_c)
    lower_bar.set_label(
        cbar_label,
        labelpad=1,
        fontdict=fontdict_normal,
        fontsize=font_base * font_c,
    )
except IndexError:
    print(f"index error, {y_var = }")

ax1.set_title(rf"lower {a_percent}%", fontdict=fontdict_bold)
######################################################
cax = ax2.inset_axes(inset_axes_)
try:
    between_bar = fig.colorbar(
        between_plot.collections[1],
        ax=ax2,
        orientation="horizontal",
        shrink=0.3,
        cax=cax,
    )
    between_bar.ax.tick_params(labelsize=font_base * font_c)
    between_bar.set_label(
        cbar_label,
        labelpad=1,
        fontdict=fontdict_normal,
        fontsize=font_base * font_c,
    )
except IndexError:
    print(f"index error, {y_var = }")
ax2.set_title(
    r"between [{}%, {}%]".format(a_percent, 100 - a_percent),
    fontdict=fontdict_bold,
)
######################################################
try:
    cax = ax3.inset_axes(inset_axes_)
    upper_bar = fig.colorbar(
        upper_plot.collections[1],
        ax=ax3,
        orientation="horizontal",
        shrink=0.3,
        cax=cax,
    )
    upper_bar.ax.tick_params(labelsize=font_base * font_c)
    upper_bar.set_label(
        cbar_label,
        labelpad=1,
        fontdict=fontdict_normal,
        fontsize=font_base * font_c,
    )
except IndexError:
    print(f"index error, {y_var = }")
ax3.set_title(rf"upper {a_percent}%", fontdict=fontdict_bold)

######################################################
ax.set_aspect("equal", adjustable="box")
file_name = bio_plots + f"{y_var}_{a_percent}percentOutlier.png"
# plt.savefig(file_name, bbox_inches="tight", dpi=map_dpi_)
#plt.close(fig)
#gc.collect()
try:
    del (cax)
    del (lower_bar)
    del (between_bar)
    del (upper_bar)
    del (min_max)
    del (norm)
except:
    pass

# %%

# %%

# %%
