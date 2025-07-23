# There is another script called ```ACF_moving_window_plot_trend_weather.ipynb``` in the archive folder.
#
# This is new version of that one to include more variables, less hard coding, variances, and possibgly moving to Kamiak.
#
# **July 21, 2025**

###########################################################################################
#######
#######    Libraries
#######
import warnings

warnings.filterwarnings("ignore")
import gc
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os, os.path, pickle, sys
import pymannkendall as mk

import seaborn as sns
from scipy import stats
import scipy.stats as scipy_stats
from statsmodels.tsa.stattools import acf
import geopandas

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc
import rangeland_plot_core as rpc


###########################################################################################
#######
#######    Terminal arguments
#######
acf_or_variance = str(sys.argv[1])


###########################################################################################
#######
#######    Some plotting parameters
#######
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

###########################################################################################
#######
#######    Directories
#######
research_db = "/data/project/agaid/h.noorazar/"
common_data = research_db + "common_data/"
rangeland_bio_base = research_db + "/rangeland_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)


###########################################################################################
#######
#######    read data
#######
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
from shapely.geometry import Polygon

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
# %%time
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
len(list(trends_MK_df.columns))

# %%
# In the new version of the code that computed trends, ACF or variance was removed from column names.
# Here I add them in there for clarity: Is it necessary? Maybe not.

# trends_MK_df.rename(columns={col: f"ACF1_{col}" if any(key in col for key in ['slope', 'trend', 'p_value'])
#                                  else col for col in trends_MK_df.columns},
#                         inplace=True)

# trends_MK_df.head(2)

# %%
# drop trend and p-value columns

bad_cols = [
    col
    for col in trends_MK_df.columns
    if any(key in col for key in ["trend_ws", "p_value"])
]
trends_MK_df.drop(columns=bad_cols, inplace=True)

# %%
len(trends_MK_df.columns)

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
import re

# weather variables
weather_variables = [re.sub(r"^slope_ws\d+_", "", item) for item in slope_cols]
weather_variables = [
    x.replace("_detrendLinReg", "").replace("_detrendSens", "")
    for x in weather_variables
]
weather_variables = list(set(weather_variables))
print(len(weather_variables))
weather_variables[:5]


# %%
tick_legend_FontSize = 8
params = {
    "font.family": "Palatino",
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

###########################################################################################
#######
#######    Plot common color bar between all-window sizes  first
#######
""" Do we need this?
for a_variable in weather_variables:
    for trendType in ["", "_detrendLinReg", "_detrendSens"]:
        all_win_sizes_cols = [
            f"slope_ws{ws}_{a_variable}{trendType}" for ws in range(5, 11)
        ]

        min_ = np.inf
        max_ = -np.inf
        for col_ in all_win_sizes_cols:
            if SF_west[col_].min() < min_:
                min_ = SF_west[col_].min()

            if SF_west[col_].max() > max_:
                max_ = SF_west[col_].max()

        cc_ = max(np.abs(min_), np.max(max_))
        norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True)
        print(round(min_, 2), round(max_, 2), round(cc_, 2))

        outdir = (
            bio_plots
            + acf_or_variance
            + "_"
            + a_variable
            + "/slope/identical_colorbar/"
        )
        os.makedirs(outdir, exist_ok=True)
        for col in all_win_sizes_cols:
            print("numerical:  ", col)
            ws = re.search(r"ws(\d+)", col).group(1)
            last_part = re.sub(r"^slope_ws\d+_", "", col)
            ###################################
            #######
            ####### plot
            #######
            fig, ax = plt.subplots(1, 1, dpi=map_dpi_)  # figsize=(2, 2)
            ax.set_xticks([])
            ax.set_yticks([])
            rpc.plot_SF(
                SF=visframe_mainLand_west,
                ax_=ax,
                col="EW_meridian",
                cmap_=custom_cmap_GrayW,
            )

            cent_plt = SF_west.plot(
                column=col, ax=ax, legend=False, cmap="seismic", norm=norm_col
            )
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
            file_name = outdir + f"{col}.png"
            ax.set_aspect("equal", adjustable="box")
            plt.savefig(file_name, bbox_inches="tight", dpi=map_dpi_)
            plt.close(fig)
            try:
                del (fig, cent_plt, cax, cbar1, ws, last_part, file_name)
                gc.collect()
            except:
                pass

"""

###########################################################################################
#######
#######    Different color bars for each plot
#######
try:
    del (norm_col, min_, max_, cc_)
    gc.collect()
except:
    pass

for a_variable in weather_variables:
    outdir = (
        bio_plots + acf_or_variance + "_" + a_variable + "/slope/individual_colorbar/"
    )
    os.makedirs(outdir, exist_ok=True)

    for trendType in ["", "_detrendLinReg", "_detrendSens"]:
        all_win_sizes_cols = [
            f"slope_ws{ws}_{a_variable}{trendType}" for ws in range(5, 11)
        ]
        for col in all_win_sizes_cols:
            print("numerical:  ", col)
            ws = re.search(r"ws(\d+)", col).group(1)
            last_part = re.sub(r"^slope_ws\d+_", "", col)
            ###################################
            #######
            ####### plot
            #######
            fig, ax = plt.subplots(1, 1, dpi=map_dpi_)  # figsize=(2, 2)
            ax.set_xticks([])
            ax.set_yticks([])
            rpc.plot_SF(
                SF=visframe_mainLand_west,
                ax_=ax,
                col="EW_meridian",
                cmap_=custom_cmap_GrayW,
            )

            min_max0 = max(np.abs(SF_west[col].min()), np.abs(SF_west[col].max()))
            norm0 = Normalize(vmin=-min_max0, vmax=min_max0, clip=True)
            cent_plt = SF_west.plot(
                column=col, ax=ax, legend=False, cmap="seismic", norm=norm0
            )
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
            file_name = outdir + f"{col}.png"
            ax.set_aspect("equal", adjustable="box")
            plt.savefig(file_name, bbox_inches="tight", dpi=map_dpi_)
            plt.close(fig)
            try:
                del (fig, cent_plt, cax, cbar1, ws, last_part, file_name)
                gc.collect()
            except:
                pass

    #############
