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
fontdict_bold   = {'family':'serif', 'weight':'bold'}
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
weather = pd.read_csv(bio_reOrganized + "bpszone_annual_tempPrecip_byHN.csv")
weather.head(2)

# %%
filename = breakpoints_dir + "weather_sensSlope_beforeAfter_ANPPBP1.sav"
weather_slope_beforeAfter_ANPPBP1 = pd.read_pickle(filename)

print (weather_slope_beforeAfter_ANPPBP1.keys())

# %%

# %%
weather_slope_beforeAfter_ANPPBP1 = weather_slope_beforeAfter_ANPPBP1["sensSlope_beforeAfter_ANPPBP1"]
weather_slope_beforeAfter_ANPPBP1.head(2)

# %%
weather_slope_beforeAfter_ANPPBP1["temp_slope_diff"] = \
                                                weather_slope_beforeAfter_ANPPBP1["temp_slope_after"] - \
                                                weather_slope_beforeAfter_ANPPBP1["temp_slope_before"]

weather_slope_beforeAfter_ANPPBP1["temp_slope_ratio"] = \
                                                weather_slope_beforeAfter_ANPPBP1["temp_slope_after"] / \
                                                weather_slope_beforeAfter_ANPPBP1["temp_slope_before"]


weather_slope_beforeAfter_ANPPBP1["precip_slope_diff"] = \
                                                weather_slope_beforeAfter_ANPPBP1["precip_slope_after"] - \
                                                weather_slope_beforeAfter_ANPPBP1["precip_slope_before"]

weather_slope_beforeAfter_ANPPBP1["precip_slope_ratio"] = \
                                                weather_slope_beforeAfter_ANPPBP1["precip_slope_after"] / \
                                                weather_slope_beforeAfter_ANPPBP1["precip_slope_before"]

weather_slope_beforeAfter_ANPPBP1.head(2)

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
weather_slope_beforeAfter_ANPPBP1.head(2)

# %%
cols_ = ["fid", "temp_slope_diff", "temp_slope_ratio", "precip_slope_diff", "precip_slope_ratio"]

Albers_SF_west = pd.merge(Albers_SF_west, weather_slope_beforeAfter_ANPPBP1[cols_], how="left", on="fid")
Albers_SF_west.head(2)

# %%
tick_legend_FontSize = 12
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
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

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
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

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
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

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
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%
