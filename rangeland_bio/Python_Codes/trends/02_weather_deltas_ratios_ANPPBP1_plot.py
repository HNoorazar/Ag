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

# %%
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import numpy as np
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

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc

# %%
import importlib;
importlib.reload(rc);
importlib.reload(rpc);

# %%
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

fontdict_normal = {'family':'serif', 'weight':'normal'}
fontdict_bold   = {'family':'serif', 'weight':'normal'}
inset_axes_     = [0.1, 0.13, 0.45, 0.03]

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

rangeland_bio_base = research_db + "/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

breakpoint_plot_base = bio_plots + "breakpoints/"
os.makedirs(breakpoint_plot_base, exist_ok=True)

breakpoint_TS_dir = bio_plots + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)


G_breakpoint_TS_dir = breakpoint_TS_dir + "/greening/"
B_breakpoint_TS_dir = breakpoint_TS_dir + "/browning/"
noTrend_breakpoint_TS_dir = breakpoint_TS_dir + "/notrend/"

os.makedirs(G_breakpoint_TS_dir, exist_ok=True)
os.makedirs(B_breakpoint_TS_dir, exist_ok=True)
os.makedirs(noTrend_breakpoint_TS_dir, exist_ok=True)

# %%
breakpoint_TS_sen_dir = breakpoint_plot_base + "breakpoints_TS_sensSlope/"
# os.makedirs(breakpoint_TS_sen_dir, exist_ok=True)
breakpoints_dir = rangeland_bio_data + "breakpoints/"

# %%
# weather = pd.read_csv(bio_reOrganized + "bpszone_annual_weather_byHN.csv")
# weather = pd.read_csv(bio_reOrganized + "bpszone_annualWeatherByHN_and_deTrended.csv")
# weather.head(2)

# %%
# filename = breakpoints_dir + "weather_sensSlope_beforeAfter_ANPPBP1.sav"
filename = breakpoints_dir + "01_weather_Sen_ACF_stats_beforeAfter_ANPPBP1.sav"
weather_ANPPBP1 = pd.read_pickle(filename)
print (list(weather_ANPPBP1.keys()))
print ()
print (f"{weather_ANPPBP1['Date'] = }")
print (f"{weather_ANPPBP1['source_code'] = }")

weather_ANPPBP1 = weather_ANPPBP1['weather_diffsRatios']
weather_ANPPBP1.head(2)

# %%

# %%
### Moved to 01_weather_Sen_ACF_stats_beforeAfter_ANPPBP1_compute.ipynb


# y_cols = [x for x in weather_ANPPBP1.columns if ("slope" in x) or ("mean" in x) or ("median" in x)]

# ## remove before and after to get patterns
# y_cols_patterns = [x.replace('before', '').replace('after', '') for x in y_cols]
# y_cols_patterns = list(set(y_cols_patterns))
# y_cols_patterns

# for a_pattern in y_cols_patterns:
#     weather_ANPPBP1[f'{a_pattern}diff'] = weather_ANPPBP1[f"{a_pattern}after"] - \
#                                                  weather_ANPPBP1[f"{a_pattern}before"]
    
#     weather_ANPPBP1[f"{a_pattern}ratio"] = weather_ANPPBP1[f"{a_pattern}after"] / \
#                                                   weather_ANPPBP1[f"{a_pattern}before"]
# weather_ANPPBP1.head(2)

# %%
# weather_ANPPBP1["temp_slope_diff"] = weather_ANPPBP1["temp_slope_after"] - \
#                                                 weather_ANPPBP1["temp_slope_before"]

# weather_ANPPBP1["temp_slope_ratio"] = weather_ANPPBP1["temp_slope_after"] / \
#                                                 weather_ANPPBP1["temp_slope_before"]
####################################################################################################
# weather_ANPPBP1["precip_slope_diff"] = weather_ANPPBP1["precip_slope_after"] - \
#                                                 weather_ANPPBP1["precip_slope_before"]

# weather_ANPPBP1["precip_slope_ratio"] = weather_ANPPBP1["precip_slope_after"] / \
#                                                 weather_ANPPBP1["precip_slope_before"]

# weather_ANPPBP1.head(2)

# %%

# %%

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
# %%time
from shapely.geometry import Polygon
gdf = geopandas.read_file(common_data +'cb_2018_us_state_500k.zip')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%
# %%time
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
Albers_SF_west = geopandas.read_file(f_name)
Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid

Albers_SF_west.rename(columns={"EW_meridia": "EW_meridian",
                               "p_valueSpe" : "p_valueSpearman",
                               "medians_di": "medians_diff_ANPP",
                               "medians__1" : "medians_diff_slope_ANPP",
                               "median_ANP" : "median_ANPP_change_as_perc",
                               "state_majo" : "state_majority_area"}, 
                      inplace=True)

Albers_SF_west.head(2)

# %%
### There might be a confusion using the names I have used
### medians_diff_ANPP refers to median of ANPP in the first decare and last decade to measure
### growth in ANPP over 40 years. But, later we did mean and median differences
### for weather variables before and after breakpoints. So, I will drop these here.
Albers_SF_west.drop(columns=[
                             "medians_diff_ANPP", "medians_diff_slope_ANPP",
                             "median_ANPP_change_as_perc", "state_majority_area", 
                             "p_Spearman", "centroid", 'trend_yue', 'p_yue',
                             'trend_rao', 'p_rao', 'Tau', 'state_1', 'state_2',
                             'hucsgree_4', 'bps_code', 'bps_model', 'bps_name', 'groupveg',
                             'Spearman', 'sens_inter', 'sens_slope', 'trend'
                             ],
                    inplace=True)

# %%
weather_ANPPBP1.head(2)

# %%
## Some FIDs did not have breakpoint in their ANPP time series. 
#S subset Albers_SF_west to those that did:
Albers_SF_west = Albers_SF_west[Albers_SF_west["fid"].isin(list(weather_ANPPBP1["fid"].unique()))]
Albers_SF_west.reset_index(drop=True, inplace=True)

# %%
# cols_ = ["fid", "temp_slope_diff", "temp_slope_ratio", "precip_slope_diff", "precip_slope_ratio"]
diff_ratio_cols = [x for x in weather_ANPPBP1 if (("diff" in x) or ("ratio" in x))]
cols_ = ["fid"] + diff_ratio_cols

Albers_SF_west = pd.merge(Albers_SF_west, weather_ANPPBP1[cols_], how="left", on="fid")
Albers_SF_west.head(2)

# %%
tick_legend_FontSize = 9
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * .71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .7,
          "ytick.labelsize": tick_legend_FontSize * .7,
          "axes.titlepad": 5,
          "legend.handlelength": 2,
          "xtick.bottom": False,
          "ytick.left": False,
          "xtick.labelbottom": False,
          "ytick.labelleft": False,
          'axes.linewidth' : .05}

plt.rcParams.update(params)

# %%
print (len(diff_ratio_cols))
diff_ratio_cols[:4]

# %%

# %%
y_var = diff_ratio_cols[0]

fig, ax = plt.subplots(1, 1, dpi=100)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'{y_var}', labelpad=1, fontdict=fontdict_normal);

plt.title(f"{y_var} (ANPP-BP1)", fontdict=fontdict_bold);
file_name = breakpoint_plot_base + f"{y_var}_ANPPBP1.png"
    
plt.tight_layout()
ax.set_aspect('equal', adjustable='box')
del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%

# %%
for y_var in diff_ratio_cols:
    fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
    ax.set_xticks([]); ax.set_yticks([])
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

    min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
    norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
    cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

    cax = ax.inset_axes(inset_axes_)
    cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
    # pre = r'$\Delta$'
    cbar1.set_label(f'{y_var}', labelpad=1, fontdict=fontdict_normal);
    plt.title(f"{y_var} (ANPP-BP1)", fontdict=fontdict_bold);
    file_name = breakpoint_plot_base + f"{y_var}_ANPPBP1.png"
    
    plt.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()
    del(cent_plt, cax, cbar1, norm1, min_max)

# %% [markdown]
# ## Remove 5% and 10% from either side

# %%
percents = [5, 10]

# %%
tick_legend_FontSize = 7
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * .71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .7,
          "ytick.labelsize": tick_legend_FontSize * .7,
          "axes.titlepad": 5,
          "legend.handlelength": 2,
          "xtick.bottom": False,
          "ytick.left": False,
          "xtick.labelbottom": False,
          "ytick.labelleft": False,
          'axes.linewidth' : .05}

plt.rcParams.update(params)

inset_axes_ = [0.15, 0.13, 0.45, 0.03]

# %%

# %%
y_var = diff_ratio_cols[0]
a_percent = 5

# %%
fig, ax = plt.subplots(1, 2, dpi=100, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})
ax[0].set_xticks([]); ax[0].set_yticks([]);
ax[1].set_xticks([]); ax[1].set_yticks([]);

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

df = Albers_SF_west.copy()
df.dropna(subset=[y_var], inplace=True)

perc_ = a_percent / 100
lower_bound = df[y_var].quantile(perc_)
upper_bound = df[y_var].quantile(1 - perc_)

# Filter rows between 10th and 90th percentile (inclusive)
filtered_between = df[(df[y_var] >= lower_bound) & (df[y_var] <= upper_bound)]
filtered_outside = df[(df[y_var] < lower_bound) | (df[y_var] > upper_bound)]

############
min_max0 = max(np.abs(filtered_between[y_var].min()), np.abs(filtered_between[y_var].max()))
min_max1 = max(np.abs(filtered_outside[y_var].min()), np.abs(filtered_outside[y_var].max()))

norm0 = Normalize(vmin= -min_max0, vmax=min_max0, clip=True)
norm1 = Normalize(vmin= -min_max1, vmax=min_max1, clip=True)

cent_plt0 = filtered_between.plot(ax=ax[0], column=y_var, legend=False, cmap='seismic', norm=norm0)
cent_plt1 = filtered_outside.plot(ax=ax[1], column=y_var, legend=False, cmap='seismic', norm=norm1)

cax0 = ax[0].inset_axes(inset_axes_); 
cax1 = ax[1].inset_axes(inset_axes_)

cbar0 = fig.colorbar(cent_plt0.collections[1], ax=ax[0], norm=norm0, cax=cax0, 
                     shrink=0.3, orientation='horizontal')

cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax[1], norm=norm1, cax=cax1,
                     shrink=0.3, orientation='horizontal')
####################################################################################
cbar0.set_label(f'{y_var}', labelpad=1, fontdict=fontdict_normal);
cbar1.set_label(f'{y_var}', labelpad=1, fontdict=fontdict_normal);

fig.suptitle(f"{y_var} (ANPP-BP1)", y=0.82, fontdict={'family':'serif'});

plt.tight_layout()
ax[0].set_aspect('equal', adjustable='box'); ax[1].set_aspect('equal', adjustable='box')

file_name = breakpoint_plot_base + f"{y_var}_ANPPBP1_{a_percent}percent_outlier.png"

# %%

# %%

# %%
for a_percent in [5, 10]:
    for y_var in diff_ratio_cols:
#         print (a_percent, y_var)
        fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})
        ax[0].set_xticks([]); ax[0].set_yticks([]);
        ax[1].set_xticks([]); ax[1].set_yticks([]);
        ax[0].set_aspect('equal', adjustable='box'); ax[1].set_aspect('equal', adjustable='box')

        rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", 
                    cmap_=ListedColormap(['grey', 'white']))
        rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", 
                    cmap_=ListedColormap(['grey', 'white']))

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
            min_max_between = max(np.abs(filtered_between[y_var].min()), np.abs(filtered_between[y_var].max()))
            min_max_outside = max(np.abs(filtered_outside[y_var].min()), np.abs(filtered_outside[y_var].max()))

            norm_between = Normalize(vmin= -min_max_between, vmax=min_max_between, clip=True)
            norm_outside = Normalize(vmin= -min_max_outside, vmax=min_max_outside, clip=True)

            cent_plt_between = filtered_between.plot(ax=ax[0], column=y_var, 
                                                     legend=False, cmap='seismic', norm=norm_between)
            cent_plt_outside = filtered_outside.plot(ax=ax[1], column=y_var, 
                                                     legend=False, cmap='seismic', norm=norm_outside)

            cax0 = ax[0].inset_axes(inset_axes_); 
            cax1 = ax[1].inset_axes(inset_axes_)

            cbar_between = fig.colorbar(cent_plt_between.collections[1], ax=ax[0], norm=norm_between,
                                        cax=cax0, shrink=0.3, orientation='horizontal')

            cbar_outside = fig.colorbar(cent_plt_outside.collections[1], ax=ax[1], norm=norm_outside, 
                                        cax=cax1, shrink=0.3, orientation='horizontal')
            ####################################################################################
            cbar_between.set_label(f'{y_var}', labelpad=1, fontdict=fontdict_normal);
            cbar_outside.set_label(f'{y_var}', labelpad=1, fontdict=fontdict_normal);

            fig.suptitle(f"{y_var} (ANPP-BP1)", y=0.82, fontdict={'family':'serif'});
            plt.tight_layout()

            file_name = breakpoint_plot_base + f"{y_var}_ANPPBP1_{a_percent}percent_outlier.png"
            plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
            plt.close()
            del(cent_plt_between, cent_plt_outside, cax0, cax1, cbar_between, 
                cbar_outside, norm_between, norm_outside, min_max_between, min_max_outside)
        except:
            plt.close()
            pass

# %%

# %%

# %%

# %%

# %% [markdown]
# # Below is Old: 
#
# These are correct but we added more columns and plotted them above.

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)

y_var = 'temp_slope_diff'
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(r'$\Delta(TempSs_{ANPP-BP1})$', labelpad=1, fontdict=fontdict_normal);
plt.title("temp. slope diff. (ANPP-BP1)", fontdict=fontdict_bold);

plt.tight_layout()
# on overleaf, a sublot looked slightly higher than another. lets see if this fixes it
ax.set_aspect('equal', adjustable='box')

# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = breakpoint_plot_base + "temp_senSlopeDelta_ANPPBP1_divergeRB.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)

y_var = 'precip_slope_diff'
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(r'$\Delta(precipSs_{ANPP-BP1})$', labelpad=1, fontdict=fontdict_normal);
plt.title("precip. slope diff. (ANPP-BP1)", fontdict=fontdict_bold);

plt.tight_layout()
# on overleaf, a sublot looked slightly higher than another. lets see if this fixes it
ax.set_aspect('equal', adjustable='box')

# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = breakpoint_plot_base + "precip_senSlopeDelta_ANPPBP1_divergeRB.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)

y_var = 'temp_slope_ratio'
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(r'$ratio(TempSs_{ANPP-BP1})$', labelpad=1, fontdict=fontdict_normal);
plt.title("temp. slope ratio (ANPP-BP1)", fontdict=fontdict_bold);

plt.tight_layout()
# on overleaf, a sublot looked slightly higher than another. lets see if this fixes it
ax.set_aspect('equal', adjustable='box')

# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = breakpoint_plot_base + "temp_senSlopeRatio_ANPPBP1_divergeRB.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)

y_var = 'precip_slope_ratio'
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(r'$ratio(precipSs_{ANPP-BP1})$', labelpad=1, fontdict=fontdict_normal);
plt.title("precip. slope ratio. (ANPP-BP1)", fontdict=fontdict_bold);

plt.tight_layout()
# on overleaf, a sublot looked slightly higher than another. lets see if this fixes it
ax.set_aspect('equal', adjustable='box')

# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = breakpoint_plot_base + "precip_senSlopeRatio_ANPPBP1_divergeRB.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%
