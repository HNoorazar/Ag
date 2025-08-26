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
import re
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
acf_or_variance = "ACF1" # options: ACF1 or variance
variable_set = "drought" # options: weather or drought

# %%
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(["lightcoral", "black"])
custom_cmap_BW = ListedColormap(["white", "black"])
custom_cmap_GrayW = ListedColormap(["grey", "black"])
cmap_G = cm.get_cmap("Greens")  # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap("Reds")

fontdict_normal = {"family": "serif", "weight": "normal"}
fontdict_bold = {"family": "serif", "weight": "bold"}
fontdict_bold_sup = {"family": "serif", "fontweight": "bold", "fontsize": 6}
inset_axes_ = [0.1, 0.14, 0.45, 0.03]


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

gdf = geopandas.read_file(common_data + "cb_2018_us_state_500k.zip")
# gdf = geopandas.read_file(common_data +'cb_2018_us_state_500k')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({"init": "epsg:5070"})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[
    ~visframe_mainLand_west.state.isin(["AK", "HI"])
].copy()

# %%
f_name = bio_reOrganized + "Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip"
SF_west = geopandas.read_file(f_name)

SF_west.rename(
    columns={
        "EW_meridia": "EW_meridian",
        "p_valueSpe": "p_valueSpearman",
        "medians_di": "medians_diff_ANPP",
        "medians__1": "medians_diff_slope_ANPP",
        "median_ANP": "median_ANPP_change_as_perc",
        "state_majo": "state_majority_area",
    },
    inplace=True,
)
SF_west.head(2)

# %%
SF_west.drop(
    columns=[
        "medians_diff_ANPP",
        "medians_diff_slope_ANPP",
        "median_ANPP_change_as_perc",
        "state_majority_area",
        "p_Spearman",
        "trend_yue",
        "p_yue",
        "trend_rao",
        "p_rao",
        "Tau",
        "state_1",
        "state_2",
        "hucsgree_4",
        "bps_code",
        "bps_model",
        "bps_name",
        "groupveg",
        "Spearman",
        "sens_inter",
        "sens_slope",
        "trend",
    ],
    inplace=True,
)

# %%
### Read all rolling window ACFs
if acf_or_variance == "ACF1":
    filename = bio_reOrganized + "weather_ACFs_rollingWindow_trends.sav"
    trends_MK_df = pd.read_pickle(filename)
    trends_MK_df = trends_MK_df["weather_ACF_trends_MK_df"]
    trends_MK_df.head(2)
elif acf_or_variance == "variance":
    filename = bio_reOrganized + "weather_variances_rollingWindow_trends.sav"
    trends_MK_df = pd.read_pickle(filename)
    trends_MK_df = trends_MK_df["weather_variances_trends_MK_df"]

# %%
filename = bio_reOrganized + "weather_ACFs_rollingWindow_trends.sav"
filename = "/Users/hn/Desktop/" + "weather_ACFs_rollingWindow_trends.sav"
trends_MK_df = pd.read_pickle(filename)
trends_MK_df["Date"]

# %%
trends_MK_df = trends_MK_df["weather_ACF_trends_MK_df"]


# drop trend and p-value columns
bad_cols = [
    col
    for col in trends_MK_df.columns
    if any(key in col for key in ["trend_ws", "p_value"])
]
trends_MK_df.drop(columns=bad_cols, inplace=True)
len(trends_MK_df.columns)

# %%
drought_indices = [
    x
    for x in trends_MK_df.columns
    if (("spei_" in x) or ("et0_" in x) or ("etr_" in x))
]
weather_indices = [x for x in trends_MK_df.columns if not (x in drought_indices)]
drought_indices = ["fid"] + drought_indices

# %%
weather_indices

# %%

# %%
if variable_set == "drought":
    print(len(trends_MK_df.columns))
    trends_MK_df = trends_MK_df[drought_indices]
    print(len(trends_MK_df.columns))
elif variable_set == "weather":
    print(len(trends_MK_df.columns))
    trends_MK_df = trends_MK_df[weather_indices]
    print(len(trends_MK_df.columns))

# %%

# %%
SF_west = pd.merge(SF_west, trends_MK_df, how="left", on="fid")
SF_west.head(2)

# %%
print(len(SF_west.columns))
print()
sorted(SF_west.columns[:5])

##### Fix the color bar so that for numerical columns, plots are comparable
# sens_slope is slope of ANPP itself that was saved in SW_west to begin with.
slope_cols = [x for x in SF_west.columns if "slope" in x]
slope_cols[:5]

# %%
slope_cols

# %%
# # weather variables
# weather_variables = [re.sub(r"^slope_ws\d+_", "", item) for item in slope_cols]
# weather_variables = [
#     x.replace("_detrendLinReg", "").replace("_detrendSens", "")
#     for x in weather_variables
# ]
# weather_variables = list(set(weather_variables))
# print(len(weather_variables))
# weather_variables[:5]

# %%
tick_legend_FontSize = 8
params = {
    # "font.family": "Palatino",
    "legend.fontsize": tick_legend_FontSize * 0.2,  # this does not work below
    "axes.labelsize": tick_legend_FontSize * 1,
    "axes.titlesize": tick_legend_FontSize * 1.1,
    "xtick.labelsize": tick_legend_FontSize * 0.8,
    "ytick.labelsize": tick_legend_FontSize * 0.8,
    "axes.titlepad": 10,
    "legend.handlelength": 2,
    "axes.titleweight": "bold",
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.labelbottom": True,
    "ytick.labelleft": True,
    "axes.linewidth": 0.05,
}

plt.rcParams.update(params)

if acf_or_variance == "ACF1":
    acf_var_title = acf_or_variance
elif acf_or_variance == "variance":
    acf_var_title = rf"$\sigma^2$"

# %%
a_col = slope_cols[1]
a_variable = re.sub(r"^slope_ws\d+_", "", a_col)
a_variable = a_variable.replace("_detrendLinReg", "").replace("_detrendSens", "")


ws = re.search(r"ws(\d+)", a_col).group(1)
last_part = re.sub(r"^slope_ws\d+_", "", a_col)

# %%
fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([])
ax.set_yticks([])
rcp.plot_SF(
    SF=visframe_mainLand_west,
    ax_=ax,
    col="EW_meridian",
    cmap_=custom_cmap_GrayW,
)

min_max0 = max(np.abs(SF_west[a_col].min()), np.abs(SF_west[a_col].max()))
norm0 = Normalize(vmin=-min_max0, vmax=min_max0, clip=True)
cent_plt = SF_west.plot(column=a_col, ax=ax, legend=False, cmap="seismic", norm=norm0)
############# color bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(
    cent_plt.collections[1],
    ax=ax,
    orientation="horizontal",
    shrink=0.3,
    cax=cax,
)

L_ = f"slope of {acf_var_title}$_{{ws={ws}}}$"
cbar1.set_label(L_, labelpad=2, fontdict=fontdict_normal)
plt.tight_layout()
plt.title(L_ + f" ({last_part})", y=0.98, fontdict=fontdict_bold_sup)
ax.set_aspect("equal", adjustable="box")

# %%
