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
# !pip3 install pymannkendall

# %%
import warnings
warnings.filterwarnings("ignore")

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
from matplotlib import colormaps
print (list(colormaps)[:4])

# %%
dpi_, map_dpi_= 300, 500
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

best_cmap_ = ListedColormap([(0.9, 0.9, 0.9), 'black'])

fontdict_normal ={'family':'serif', 'weight':'normal'}
fontdict_bold   ={'family':'serif', 'weight':'bold'}
inset_axes_     = [0.1, 0.13, 0.45, 0.03]
inset_axes_     = [0.1, 0.18, 0.45, 0.03] # tight layout

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
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

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

# %% [markdown]
# ## Read the shapefile
# And keep the vegtype in subsequent dataframes

# %%
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)
Albers_SF.head(2)

# %%
len(Albers_SF["fid"].unique())

# %% [markdown]
# # Focus only on West Meridian

# %%
print ((Albers_SF["state_majority_area"] == Albers_SF["state_1"]).sum())
print ((Albers_SF["state_majority_area"] == Albers_SF["state_2"]).sum())
print (Albers_SF.shape)
print (len(Albers_SF) - (Albers_SF["state_1"] == Albers_SF["state_2"]).sum())
print ((Albers_SF["state_1"] == Albers_SF["state_2"]).sum())

# %%
Albers_SF = pd.merge(Albers_SF, state_fips[["EW_meridian", "state_full"]], 
                     how="left", left_on="state_majority_area", right_on="state_full")

Albers_SF.drop(columns=["state_full"], inplace=True)

print (Albers_SF.shape)
Albers_SF.head(2)

# %%
Albers_SF = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF.shape

# %%
print (len(Albers_SF["fid"].unique()))
print (len(Albers_SF["value"].unique()))
print (len(Albers_SF["hucsgree_4"].unique()))

print ((Albers_SF["hucsgree_4"] - Albers_SF["value"]).unique())
print ((list(Albers_SF.index) == Albers_SF.fid).sum())

Albers_SF.drop(columns=["value"], inplace=True)
Albers_SF.head(2)

# %% [markdown]
# ## Read weather Data

# %%
filename = bio_reOrganized + "bps_weather.sav"
bps_weather = pd.read_pickle(filename)
bps_weather = bps_weather["bps_weather"]
bps_weather["fid"].unique()[-8::]

# %%
west_FIDs = list(Albers_SF["fid"])
bps_weather = bps_weather[bps_weather['fid'].isin(west_FIDs)]
bps_weather.reset_index(drop=True, inplace=True)
bps_weather.head(2)

# %%
# filename = bio_reOrganized + "bps_weather_wide.sav"
# bps_weather_wide = pd.read_pickle(filename)
# bps_weather_wide = bps_weather_wide['bps_weather_wide']
# bps_weather_wide = bps_weather_wide[bps_weather_wide['fid'].isin(west_FIDs)]
# bps_weather_wide.reset_index(drop=True, inplace=True)
# bps_weather_wide.head(2)

# %%
# 

# %%
# bps_gridmet_mean = pd.read_csv(rangeland_bio_data + "Min_Data/" + "bps_gridmet_mean_indices.csv")

# bps_gridmet_mean.rename(columns={"bpshuc": "fid"}, inplace=True)
# bps_gridmet_mean = bps_gridmet_mean[bps_gridmet_mean['fid'].isin(west_FIDs)]
# bps_gridmet_mean.head(2)

# bps_gridmet_mean.reset_index(drop=True, inplace=True)
# A = bps_gridmet_mean[["fid", "year", "month", "RAVG_AVG", 'TAVG_AVG', 'THI_AVG', "PPT"]].copy()
# A.columns = list(bps_weather.columns)
# A.equals(bps_weather)

# %%
# print (f"{bps_weather_wide.shape=}")
# print (f"{bps_gridmet_mean.shape=}")
print (f"{bps_weather.shape = }")
print (f"{Albers_SF.shape   = }")

# %%
print (f'{len(Albers_SF["fid"])=}')
print (f'{len(Albers_SF["fid"].unique())=}')
# print (f'{len(bps_weather_wide["fid"].unique())=}')
# print (f'{len(bps_gridmet_mean["fid"].unique())=}')
# print (f"{len(bps_weather_wide["fid"])=}")

# %%
annual_weather = bps_weather.groupby(['fid', 'year']).agg({'avg_of_dailyAvg_rel_hum': 'mean',
                                                           'avg_of_dailyAvgTemp_C': 'mean',
                                                           'thi_avg': 'mean',
                                                           'precip_mm_month': 'sum'}).reset_index()

annual_weather.rename(columns={"precip_mm_month": "precip_mm"}, inplace=True)

annual_weather.head(3)

# %% [markdown]
# # Compute variances

# %%
cv_df = annual_weather.groupby('fid').agg({'avg_of_dailyAvgTemp_C': ['var', 'mean', 'std'],
                                           'precip_mm': ['var', 'mean', 'std']}).reset_index()

# Flatten column MultiIndex
cv_df.columns = ['fid', 
                 'temp_variance', 'temp_mean', 'temp_std', 
                 'precip_variance', 'precip_mean', 'precip_std']

# Calculate coefficient of variation
cv_df['temp_cv_times100'] = 100 *  (cv_df['temp_std'] / cv_df['temp_mean'])
cv_df['precip_cv_times100'] = 100 * (cv_df['precip_std'] / cv_df['precip_mean'])

cv_df.head(3)

# %%
Albers_SF = pd.merge(Albers_SF, cv_df, how="left", on="fid")
Albers_SF.head(2)

# %%
annual_weather.head(2)

# %%
num_locs = len(annual_weather["fid"].unique())
num_locs

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
if not ("EW_meridian" in annual_weather.columns):
    annual_weather = pd.merge(annual_weather, Albers_SF[cols_], how="left", on = "fid")
annual_weather.head(2)

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
font = {"size": 14}
matplotlib.rc("font", **font)
font_base = 10
params = {"font.family": "Palatino",
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
          'axes.linewidth' : .05}

plt.rcParams.update(params)

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

axes.hist(Albers_SF["temp_variance"].dropna(), zorder=3,
          bins=100, color='skyblue', edgecolor='black')

axes.set_title('temperature variance distribution', color="k", fontdict=fontdict_bold);
axes.set_xlabel('Variance of temperature (annual mean)', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);
# axes.set_aspect('equal', adjustable='box')
file_name = bio_plots + "temp_40Yr_variance_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
print (Albers_SF["temp_variance"].max())
print (Albers_SF["temp_variance"].min())

# %%
print (Albers_SF["temp_cv_times100"].max())
print (Albers_SF["temp_cv_times100"].min())

# %% [markdown]
# # Distribution of CV of temp is bad! so many outliers.

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

df = Albers_SF[Albers_SF["temp_cv_times100"] < 30]
df = df[df["temp_cv_times100"] > -20]
axes.hist(df["temp_cv_times100"].dropna(), zorder=3, bins=100, color='skyblue', edgecolor='black');

axes.set_title(r'CV(temp.) $\times$ 100 distribution ($(-20, 30)$)', color="k", fontdict=fontdict_bold);
axes.set_xlabel(r'CV(temp.) $\times$ 100 (annual mean)', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);

# axes.set_aspect('equal', adjustable='box')

file_name = bio_plots + "temp_40Yr_CV_histogram_noOutlier.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=False, sharey=False, dpi=dpi_,
                         gridspec_kw={"hspace": 0.3, "wspace": 0.05});
(ax1, ax2) = axes
ax1.grid(axis='y', alpha=0.7, zorder=0); ax2.grid(axis='y', alpha=0.7, zorder=0);

df = Albers_SF[(Albers_SF["temp_cv_times100"] <= -20)]
ax1.hist(df["temp_cv_times100"].dropna(), zorder=3, bins=100, color='skyblue', edgecolor='black');
L_ = r'CV(temp)$\times$ 100 distribution (Top: <= -20, bottom >=50)'
ax1.set_title(L_, color="k", fontdict=fontdict_bold);

ax1.set_xlabel('', fontdict=fontdict_normal);
ax1.set_ylabel('count', fontdict=fontdict_normal);

df = Albers_SF[(Albers_SF["temp_cv_times100"] >= 50)]
ax2.hist(df["temp_cv_times100"].dropna(), zorder=3, bins=100, color='skyblue', edgecolor='black');
ax2.hist(df["temp_cv_times100"].dropna(), zorder=3, bins=100, color='skyblue', edgecolor='black');
ax2.set_title('', color="k", fontdict=fontdict_bold);
ax2.set_xlabel(r'CV(temp) $\times$ 100 (annual mean)', fontdict=fontdict_normal);
ax2.set_ylabel('count', fontdict=fontdict_normal);

# ax1.set_aspect('equal', adjustable='box')
# ax2.set_aspect('equal', adjustable='box')

file_name = bio_plots + "temp_40Yr_CV_histogram_outliers.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
bio_plots

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=False, sharey=False, dpi=dpi_,
                         gridspec_kw={"hspace": 0.3, "wspace": 0.05});
(ax1, ax2) = axes
ax1.grid(axis='y', alpha=0.7, zorder=0); ax2.grid(axis='y', alpha=0.7, zorder=0);

##########################################################################################
ax1.hist(Albers_SF["precip_variance"].dropna(), zorder=3, bins=100, color='skyblue', edgecolor='black')

ax1.set_title('precip. variance distribution', color="k", fontdict=fontdict_bold);
ax1.set_xlabel('', fontdict=fontdict_normal);
ax1.set_ylabel('count', fontdict=fontdict_normal);
##########################################################################################
df = Albers_SF[Albers_SF['precip_variance']<30000]
ax2.hist(df["precip_variance"].dropna(), zorder=3, bins=100, color='skyblue', edgecolor='black')
ax2.set_xlabel('Variance of precip. (annual)', fontdict=fontdict_normal);
ax2.set_ylabel('count', fontdict=fontdict_normal);

# ax1.set_aspect('equal', adjustable='box')
# ax2.set_aspect('equal', adjustable='box')

file_name = bio_plots + "precip_40Yr_variance_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%

# %%
Albers_SF.head(2)

# %%

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

axes.hist(Albers_SF["precip_cv_times100"].dropna(), zorder=3, bins=100, color='skyblue', edgecolor='black')

axes.set_title(r'CV(precip.) $\times$ 100 distribution', color="k", fontdict=fontdict_bold);
axes.set_xlabel(r'CV(precip.) $\times$ 100', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);

# axes.set_aspect('equal', adjustable='box')
file_name = bio_plots + "precip_40Yr_CV_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
font_base = 12
params = {"font.family": "Palatino",
          "legend.fontsize": font_base,
          "axes.labelsize": font_base * .71,
          "axes.titlesize": font_base * 1,
          "xtick.labelsize": font_base * .7,
          "ytick.labelsize": font_base * .7,
          "axes.titlepad": 5,
          "legend.handlelength": 2,
          "xtick.bottom": False,
          "ytick.left": False,
          "xtick.labelbottom": False,
          "ytick.labelleft": False,
          'axes.linewidth' : .05}

plt.rcParams.update(params)

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])
# ListedColormap(['grey', 'white'])
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=best_cmap_)

min_max = max(np.abs(Albers_SF['temp_variance'].min()), np.abs(Albers_SF['temp_variance'].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt = Albers_SF.plot(column='temp_variance', ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(f'$\sigma^2$(temp.)', labelpad=1, fontdict=fontdict_normal);
plt.title("variance of temperature", fontdict=fontdict_bold);

ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "temp_40Yr_variance_divergeRB_GreyBG.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF['precip_variance'].min()), np.abs(Albers_SF['precip_variance'].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF.plot(column='precip_variance', ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(f'$\sigma^2$(precip.)', labelpad=1, fontdict=fontdict_normal);
plt.title("variance of precipitation", fontdict=fontdict_bold);
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "precip_40Yr_variance_divergeRB_GreyBG.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF['precip_cv_times100'].min()), np.abs(Albers_SF['precip_cv_times100'].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF.plot(column='precip_cv_times100', ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(r'CV(precip.) $\times$ 100', labelpad=1, fontdict=fontdict_normal);
plt.title(r"CV(precip.) $\times$ 100", fontdict=fontdict_bold);
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "precip_40Yr_CV_divergeRB_GreyBG.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF['temp_cv_times100'].min()), np.abs(Albers_SF['temp_cv_times100'].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF.plot(column='temp_cv_times100', ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(r'CV(temp.) $\times$ 100', labelpad=1, fontdict=fontdict_normal);
plt.title(r"CV(temp.) $\times$ 100", fontdict=fontdict_bold);
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "temp_40Yr_CV_divergeRB_GreyBG.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2, ax3) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
ax3.set_xticks([]); ax3.set_yticks([])
fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.01, wspace=-.2)
###############################################################
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax3, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

######
cut_1 = -20
cut_2 = 300
cbar_label = r'CV(temp) $\times$ 100'
font_c = 0.8
###############################################################
df = Albers_SF[Albers_SF['temp_cv_times100'] < cut_1].copy()
min_max = max(np.abs(df['temp_cv_times100'].min()), np.abs(df['temp_cv_times100'].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt1 = df.plot(column='temp_cv_times100', ax=ax1, legend=False, cmap='seismic', norm=norm1)

print (df['temp_cv_times100'].min())
print (df['temp_cv_times100'].max())
print ()
###############################################################
df = Albers_SF[(Albers_SF['temp_cv_times100'] >= cut_1) & 
                    (Albers_SF['temp_cv_times100'] < cut_2)].copy()

min_max = max(np.abs(df['temp_cv_times100'].min()), np.abs(df['temp_cv_times100'].max()))
norm2 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt2 = df.plot(column='temp_cv_times100', ax=ax2, legend=False, cmap='seismic', norm=norm2)

print (df['temp_cv_times100'].min())
print (df['temp_cv_times100'].max())
print ()
###############################################################
df = Albers_SF[Albers_SF['temp_cv_times100'] >= cut_2].copy()
min_max = max(np.abs(df['temp_cv_times100'].min()), np.abs(df['temp_cv_times100'].max()))
norm3 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt3 = df.plot(column='temp_cv_times100', ax=ax3, legend=False, cmap='seismic', norm=norm3)

print (df['temp_cv_times100'].min())
print (df['temp_cv_times100'].max())
######################################################
cax = ax1.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.ax.tick_params(labelsize=font_base*font_c)
cbar1.set_label(cbar_label, labelpad=1, fontdict=fontdict_normal, fontsize=font_base*font_c);
ax1.set_title(f"CV(temp) < {cut_1}", fontdict=fontdict_bold);
######################################################

cax = ax2.inset_axes(inset_axes_)
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, cax=cax)
cbar2.ax.tick_params(labelsize=font_base*font_c)
cbar2.set_label(cbar_label, labelpad=1, fontdict=fontdict_normal,fontsize=font_base*font_c);
# ax2.set_title(f"CV(temp) in [{cut_1}, {cut_2}]", fontdict=fontdict_bold);
ax2.set_title(r"CV(temp) $\in$ [{}, {}]".format(cut_1, cut_2), fontdict=fontdict_bold);
######################################################
cax = ax3.inset_axes(inset_axes_)
cbar3 = fig.colorbar(cent_plt2.collections[1], ax=ax3, orientation='horizontal', shrink=0.3, cax=cax)
cbar3.ax.tick_params(labelsize=font_base*font_c)
cbar3.set_label(cbar_label, labelpad=1, fontdict=fontdict_normal, fontsize=font_base*font_c);
ax3.set_title(f"CV(temp) > {cut_2}", fontdict=fontdict_bold);

######################################################
ax.set_aspect('equal', adjustable='box')
file_name = bio_plots + "temp_40Yr_CV_divergeRB_3Categ_SeparNormal.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
df = Albers_SF[(Albers_SF['temp_cv_times100'] < cut_1)].copy()

df.head(2)

# %%
print (df['temp_mean'].min())
print (df['temp_std'].min())
print ()
print (df['temp_mean'].max())
print (df['temp_std'].max())

# %%
print (df['temp_cv_times100'].min())
print (df['temp_std'].max())

# %%
print (f'{df.loc[df["temp_cv_times100"].idxmin()]["temp_mean"] = }')
print (f'{df.loc[df["temp_cv_times100"].idxmin()]["temp_std"] = }')

# %%
df.loc[df['temp_cv_times100'].idxmin()]["temp_std"] / df.loc[df['temp_cv_times100'].idxmin()]["temp_mean"] 

# %%

# %%
