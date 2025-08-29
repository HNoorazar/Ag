import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import gc
import random
import os, os.path, pickle, sys
import pymannkendall as mk

import statistics
import statsmodels.formula.api as smf

import statsmodels.stats.api as sms
import statsmodels.api as sm

from scipy import stats
import scipy.stats as scipy_stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc
import rangeland_plot_core as rcp

# %%
import importlib

importlib.reload(rc)
importlib.reload(rcp)

from datetime import datetime
from datetime import date
import time

start_time = time.time()
###########################################################################################
#######
#######    Terminal arguments
#######
diff_or_ratio = str(sys.argv[1])  # must be either "ratio" or "diff"

###########################################################################################
#######
#######    Some plotting parameters
#######
# %%
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(["lightcoral", "black"])
custom_cmap_BW = ListedColormap(["white", "black"])
cmap_G = cm.get_cmap("Greens")  # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap("Reds")

fontdict_normal = {"family": "serif", "weight": "normal"}
fontdict_bold = {"family": "serif", "weight": "normal"}
inset_axes_ = [0.1, 0.13, 0.45, 0.03]

# %%
from matplotlib import colormaps

print(list(colormaps)[:4])

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

breakpoint_plot_base = bio_plots + "breakpoints/"
os.makedirs(breakpoint_plot_base, exist_ok=True)

breakpoint_TS_dir = bio_plots + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)

breakpoints_dir = rangeland_bio_data + "breakpoints/"
###########################################################################################
#######
#######
#######
# filename = breakpoints_dir + "weather_sensSlope_beforeAfter_ANPPBP1.sav"
filename = breakpoints_dir + "01_weather_Sen_ACF_stats_beforeAfter_ANPPBP1.sav"
weather_ANPPBP1 = pd.read_pickle(filename)
print(list(weather_ANPPBP1.keys()))
print()
print(f"{weather_ANPPBP1['Date'] = }")
print(f"{weather_ANPPBP1['source_code'] = }")

weather_ANPPBP1 = weather_ANPPBP1["weather_diffsRatios"]
weather_ANPPBP1.head(2)


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
# %%time
from shapely.geometry import Polygon

gdf = geopandas.read_file(common_data + "cb_2018_us_state_500k.zip")

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
Albers_SF_west = geopandas.read_file(f_name)
Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid

Albers_SF_west.rename(
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

Albers_SF_west.head(2)

# %%
### There might be a confusion using the names I have used
### medians_diff_ANPP refers to median of ANPP in the first decare and last decade to measure
### growth in ANPP over 40 years. But, later we did mean and median differences
### for weather variables before and after breakpoints. So, I will drop these here.
Albers_SF_west.drop(
    columns=[
        "medians_diff_ANPP",
        "medians_diff_slope_ANPP",
        "median_ANPP_change_as_perc",
        "state_majority_area",
        "p_Spearman",
        "centroid",
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
weather_ANPPBP1.head(2)

# %%
## Some FIDs did not have breakpoint in their ANPP time series.
# S subset Albers_SF_west to those that did:
Albers_SF_west = Albers_SF_west[
    Albers_SF_west["fid"].isin(list(weather_ANPPBP1["fid"].unique()))
]
Albers_SF_west.reset_index(drop=True, inplace=True)


non_weather_indices = [
    x
    for x in weather_ANPPBP1.columns
    if (("spei_" in x) or ("et0_" in x) or ("etr_" in x))
]
weather_indices = [x for x in weather_ANPPBP1.columns if not (x in non_weather_indices)]
weather_ANPPBP1 = weather_ANPPBP1[weather_indices]

diff_ratio_cols = [x for x in weather_ANPPBP1 if (diff_or_ratio in x)]
cols_ = ["fid"] + diff_ratio_cols

Albers_SF_west = pd.merge(Albers_SF_west, weather_ANPPBP1[cols_], how="left", on="fid")
Albers_SF_west.head(2)

# %%
tick_legend_FontSize = 9
params = {
    # "font.family": "Palatino",
    "legend.fontsize": tick_legend_FontSize,
    "axes.labelsize": tick_legend_FontSize * 0.71,
    "axes.titlesize": tick_legend_FontSize * 1,
    "xtick.labelsize": tick_legend_FontSize * 0.7,
    "ytick.labelsize": tick_legend_FontSize * 0.7,
    "axes.titlepad": 5,
    "legend.handlelength": 2,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.labelbottom": False,
    "ytick.labelleft": False,
    "axes.linewidth": 0.05,
}

plt.rcParams.update(params)

print(len(diff_ratio_cols))
diff_ratio_cols[:4]
print("line 222 len(diff_ratio_cols)")
print(len(diff_ratio_cols))
for y_var in diff_ratio_cols:
    fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
    ax.set_xticks([])
    ax.set_yticks([])
    rcp.plot_SF(
        SF=visframe_mainLand_west,
        ax_=ax,
        col="EW_meridian",
        cmap_=ListedColormap(["grey", "white"]),
    )

    min_max = max(
        np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max())
    )
    norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
    cent_plt = Albers_SF_west.plot(
        column=y_var, ax=ax, legend=False, cmap="seismic", norm=norm1
    )

    cax = ax.inset_axes(inset_axes_)
    cbar1 = fig.colorbar(
        cent_plt.collections[1],
        ax=ax,
        orientation="horizontal",
        shrink=0.3,
        norm=norm1,
        cax=cax,
    )
    cbar1.set_label(f"{y_var}", labelpad=1, fontdict=fontdict_normal)
    plt.title(f"{y_var} (ANPP-BP1)", fontdict=fontdict_bold)

    plt.tight_layout()
    ax.set_aspect("equal", adjustable="box")
    file_name = breakpoint_plot_base + f"{y_var}_ANPPBP1.png"
    plt.savefig(file_name, bbox_inches="tight", dpi=map_dpi_)
    plt.close(fig)
    del (fig, cent_plt, cax, cbar1, norm1, min_max)
    gc.collect()

###########################################################################################
#######
####### Remove 5% and 10% from either side; outlier
#######
percents = [5, 10]

tick_legend_FontSize = 7
params = {
    # "font.family": "Palatino",
    "legend.fontsize": tick_legend_FontSize,
    "axes.labelsize": tick_legend_FontSize * 0.71,
    "axes.titlesize": tick_legend_FontSize * 1,
    "xtick.labelsize": tick_legend_FontSize * 0.7,
    "ytick.labelsize": tick_legend_FontSize * 0.7,
    "axes.titlepad": 5,
    "legend.handlelength": 2,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.labelbottom": False,
    "ytick.labelleft": False,
    "axes.linewidth": 0.05,
}
plt.rcParams.update(params)

inset_axes_ = [0.15, 0.13, 0.45, 0.03]

for a_percent in [5, 10]:
    for y_var in diff_ratio_cols:
        # print (a_percent, y_var)
        fig, ax = plt.subplots(
            1, 2, dpi=map_dpi_, gridspec_kw={"hspace": 0.02, "wspace": 0.05}
        )
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[0].set_aspect("equal", adjustable="box")
        ax[1].set_aspect("equal", adjustable="box")

        rcp.plot_SF(
            SF=visframe_mainLand_west,
            ax_=ax[0],
            col="EW_meridian",
            cmap_=ListedColormap(["grey", "white"]),
        )
        rcp.plot_SF(
            SF=visframe_mainLand_west,
            ax_=ax[1],
            col="EW_meridian",
            cmap_=ListedColormap(["grey", "white"]),
        )

        df = Albers_SF_west.copy()
        df.dropna(subset=[y_var], inplace=True)

        perc_ = a_percent / 100
        lower_bound = df[y_var].quantile(perc_)
        upper_bound = df[y_var].quantile(1 - perc_)

        # Filter rows between and outside outlier boundaries
        filtered_between = df[(df[y_var] >= lower_bound) & (df[y_var] <= upper_bound)]
        filtered_outside = df[(df[y_var] < lower_bound) | (df[y_var] > upper_bound)]
        """
        avg_of_dailyAvg_rel_hum_slope_ratio has all NA in it
        """
        try:
            ############
            min_max_between = max(
                np.abs(filtered_between[y_var].min()),
                np.abs(filtered_between[y_var].max()),
            )
            min_max_outside = max(
                np.abs(filtered_outside[y_var].min()),
                np.abs(filtered_outside[y_var].max()),
            )

            norm_between = Normalize(
                vmin=-min_max_between, vmax=min_max_between, clip=True
            )
            norm_outside = Normalize(
                vmin=-min_max_outside, vmax=min_max_outside, clip=True
            )

            cent_plt_between = filtered_between.plot(
                ax=ax[0], column=y_var, legend=False, cmap="seismic", norm=norm_between
            )
            cent_plt_outside = filtered_outside.plot(
                ax=ax[1], column=y_var, legend=False, cmap="seismic", norm=norm_outside
            )

            cax0 = ax[0].inset_axes(inset_axes_)
            cax1 = ax[1].inset_axes(inset_axes_)

            cbar_between = fig.colorbar(
                cent_plt_between.collections[1],
                ax=ax[0],
                norm=norm_between,
                cax=cax0,
                shrink=0.3,
                orientation="horizontal",
            )

            cbar_outside = fig.colorbar(
                cent_plt_outside.collections[1],
                ax=ax[1],
                norm=norm_outside,
                cax=cax1,
                shrink=0.3,
                orientation="horizontal",
            )
            ####################################################################################
            cbar_between.set_label(f"{y_var}", labelpad=1, fontdict=fontdict_normal)
            cbar_outside.set_label(f"{y_var}", labelpad=1, fontdict=fontdict_normal)

            fig.suptitle(f"{y_var} (ANPP-BP1)", y=0.82, fontdict={"family": "serif"})
            plt.tight_layout()

            file_name = (
                breakpoint_plot_base + f"{y_var}_ANPPBP1_{a_percent}percent_outlier.png"
            )
            plt.savefig(file_name, bbox_inches="tight", dpi=map_dpi_)
            plt.close(fig)
            del (
                fig,
                cent_plt_between,
                cent_plt_outside,
                cax0,
                cax1,
                cbar_between,
                cbar_outside,
                norm_between,
                norm_outside,
                min_max_between,
                min_max_outside,
            )
            gc.collect()
        except:
            plt.close(fig)
            del fig
            gc.collect()
            pass

end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
print("code is finished")
