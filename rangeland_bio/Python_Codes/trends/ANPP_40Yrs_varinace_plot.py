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
# # !pip3 install pymannkendall

# %%
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rcp

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

# %%
research_data_ = "/Users/hn/Documents/01_research_data/"
common_data = research_data_ + "common_data/"

rangeland_bio_base = research_data_ + "/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = research_data_ + "/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

# %%
bpszone_ANPP = pd.read_csv(min_bio_dir + "bpszone_annual_productivity_rpms_MEAN.csv")

bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.sort_values(by=['fid', 'year'], inplace=True)
bpszone_ANPP.head(2)

# %%
# ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
# ANPP = ANPP["bpszone_ANPP"]
# ANPP.rename(columns={"area_sqMeter": "area_sqmeter", 
#                      "count": "pixel_count",
#                      "mean" : "mean_lb_per_acr"}, inplace=True)

# ANPP.sort_values(by=['fid', 'year'], inplace=True)
# ANPP.head(2)

# %%
filename = bio_reOrganized + f"bpszone_ANPP_no2012_detrended.sav"
ANPP = pd.read_pickle(filename)
ANPP = ANPP["ANPP_no2012_detrended"]
ANPP.head(2)

# %%
col = "mean_lb_per_acr"
print (sum(ANPP[col] - bpszone_ANPP[col]))
print (sum(ANPP["area_sqMeter"] == bpszone_ANPP["area_sqmeter"]) == len(ANPP))

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
Albers_SF_west = geopandas.read_file(f_name)
# Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid
Albers_SF_west.head(2)

# %%
Albers_SF_west.rename(columns={"EW_meridia": "EW_meridian",
                               "p_valueSpe" : "p_valueSpearman",
                               "medians_di": "medians_diff_ANPP",
                               "medians__1" : "medians_diff_slope_ANPP",
                               "median_ANP" : "median_ANPP_change_as_perc",
                               "state_majo" : "state_majority_area"}, 
                      inplace=True)

# %%

# %%
# NPP_variance_df = ANPP.groupby('fid')['mean_lb_per_acr'].var().reset_index()
# NPP_variance_df.columns = ['fid', 'anpp_variance']
# NPP_variance_df.head(2)

# %%
cv_df = ANPP.groupby('fid').agg({'mean_lb_per_acr': ['var', 'mean', 'std',
                                                     'median', 'min', 'max']}).reset_index()
cv_df.head(2)

# Flatten column MultiIndex
cv_df.columns = ['fid', 'anpp_variance', 'anpp_mean', 'anpp_std', 'anpp_median', 'anpp_min', 'anpp_max']

# Calculate coefficient of variation
cv_df['anpp_cv'] = (cv_df['anpp_std'] / cv_df['anpp_mean'])*100
cv_df.head(3)

# %%
detrended_anpp_cols = [x for x in ANPP.columns if "anpp_detrend" in x]
detrended_anpp_cols

# %%
cv_df_detrendSens = ANPP.groupby('fid').agg({'anpp_detrendSens': ['var', 'mean', 'std',
                                                                  'median', 'min', 'max']}).reset_index()
cv_df_detrendSens.head(2)

# Flatten column MultiIndex
cv_df_detrendSens.columns = ['fid', 'anpp_detrendSens_variance', 'anpp_detrendSens_mean', 'anpp_detrendSens_std', 'anpp_detrendSens_median', 'anpp_detrendSens_min', 'anpp_detrendSens_max']

# Calculate coefficient of variation
cv_df_detrendSens['anpp_detrendSens_cv'] = (cv_df_detrendSens['anpp_detrendSens_std'] / cv_df_detrendSens['anpp_detrendSens_mean'])*100
cv_df_detrendSens.head(3)

# %%
cv_df_detrendLinReg = ANPP.groupby('fid').agg({'anpp_detrendLinReg': ['var', 'mean', 'std',
                                                                  'median', 'min', 'max']}).reset_index()
cv_df_detrendLinReg.head(2)

# Flatten column MultiIndex
cv_df_detrendLinReg.columns = ['fid', 'anpp_detrendLinReg_variance', 
                               'anpp_detrendLinReg_mean', 'anpp_detrendLinReg_std', 
                               'anpp_detrendLinReg_median', 'anpp_detrendLinReg_min', 
                               'anpp_detrendLinReg_max']

# Calculate coefficient of variation
cv_df_detrendLinReg['anpp_detrendLinReg_cv'] = (cv_df_detrendLinReg['anpp_detrendLinReg_std'] /
                                                cv_df_detrendLinReg['anpp_detrendLinReg_mean'])*100
cv_df_detrendLinReg.head(3)

# %%
cv_df = pd.merge(cv_df, cv_df_detrendSens, how="left", on="fid")
cv_df = pd.merge(cv_df, cv_df_detrendLinReg, how="left", on="fid")
cv_df = cv_df.round(3)
cv_df.head(2)

# %% [markdown]
# # Make some plots

# %%
# Albers_SF_west.plot(column='EW_meridian', categorical=True, legend=True);

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

# %%
bpszone_ANPP_west = bpszone_ANPP.copy()

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
bpszone_ANPP_west = pd.merge(bpszone_ANPP_west, Albers_SF_west[cols_], how="left", on = "fid")
bpszone_ANPP_west.head(2)

# %%

# %%
# drop trend so there is no bug later
print (ANPP_MK_df.shape)
ANPP_MK_df.drop(columns=["trend"], inplace=True)
Albers_SF_west.drop(columns=["trend"], inplace=True)
print (ANPP_MK_df.shape)

# %%
Albers_SF_west = pd.merge(Albers_SF_west, cv_df, how="left", on="fid")
Albers_SF_west.head(2)

# %%

# %%
font = {"size": 14}
matplotlib.rc("font", **font)
tick_legend_FontSize = 10
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05}

plt.rcParams.update(params)

# %%

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(7, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

axes.hist(Albers_SF_west["anpp_variance"].dropna(), zorder=3, bins=100, color='skyblue', edgecolor='black')

axes.set_title('ANPP variance distribution', color="k", fontdict=fontdict_bold);
axes.set_xlabel(r'Variance ($\sigma^2$) of mean ANPP (lb/acre)', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);

file_name = bio_plots + "ANPP_40Yr_variance_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(7, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

axes.hist(Albers_SF_west["anpp_detrendLinReg_variance"].dropna(), 
          zorder=3, bins=100, color='skyblue', edgecolor='black')

axes.set_title('ANPP detrendLinReg variance distribution', color="k", fontdict=fontdict_bold);
axes.set_xlabel(r'Variance of mean ANPP (lb/acre)', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);

file_name = bio_plots + "ANPP_detrendLinReg_40Yr_variance_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(7, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

axes.hist(Albers_SF_west["anpp_detrendSens_variance"].dropna(), 
          zorder=3, bins=100, color='skyblue', edgecolor='black')

axes.set_title('ANPP detrendSens variance distribution', color="k", fontdict=fontdict_bold);
axes.set_xlabel(r'Variance of mean ANPP (lb/acre)', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);

file_name = bio_plots + "ANPP_detrendSens_40Yr_variance_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(7, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

axes.hist(Albers_SF_west['anpp_cv'].dropna(), zorder=3,
          bins=100, color='skyblue', edgecolor='black')

axes.set_title(r'ANPP CV distribution', color="k", fontdict=fontdict_bold);
axes.set_xlabel(r'ANPP CV ($\sigma/\mu \times 100$)', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);

file_name = bio_plots + "ANPP_40Yr_CV_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(7, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

axes.hist(Albers_SF_west['anpp_detrendSens_cv'].dropna(), zorder=3,
          bins=100, color='skyblue', edgecolor='black')

axes.set_title(r'ANPP detrendSens CV distribution', color="k", fontdict=fontdict_bold);
axes.set_xlabel(r'ANPP CV ($\sigma/\mu \times 100$)', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);

file_name = bio_plots + "ANPP_detrendSens_40Yr_CV_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(7, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

data = Albers_SF_west.copy()
data = data[np.isfinite(data['anpp_detrendLinReg_cv'])]
axes.hist(data['anpp_detrendLinReg_cv'].dropna(), zorder=3,
          bins=100, color='skyblue', edgecolor='black')

axes.set_title(r'ANPP detrendLinReg CV distribution', color="k", fontdict=fontdict_bold);
axes.set_xlabel(r'ANPP CV ($\sigma/\mu \times 100$)', fontdict=fontdict_normal);
axes.set_ylabel('count', fontdict=fontdict_normal);

file_name = bio_plots + "ANPP_detrendLinReg_40Yr_CV_histogram.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%

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
print (np.abs(Albers_SF_west['anpp_variance'].min()))
print (np.abs(Albers_SF_west['anpp_variance'].max()))
print ()
print (np.abs(Albers_SF_west['anpp_cv'].min()))
print (np.abs(Albers_SF_west['anpp_cv'].max()))


# %%

# %%

# %%
y_var = "anpp_min"
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'min(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP minimum", fontdict=fontdict_bold);

# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "ANPP_40Yr_min.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%
y_var = "anpp_detrendSens_min"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'min(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP (detrendSens) minimum", fontdict=fontdict_bold);

file_name = bio_plots + "ANPP_detrendSens_40Yr_min.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_detrendLinReg_min"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])
min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'min(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP (detrendLinReg) minimum", fontdict=fontdict_bold);

file_name = bio_plots + "ANPP_detrendLinReg_40Yr_min.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%

# %%
y_var = "anpp_max"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'max(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP maximum", fontdict=fontdict_bold);

file_name = bio_plots + "ANPP_40Yr_max.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_detrendSens_max"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'max(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP (detrendSens) maximum", fontdict=fontdict_bold);

file_name = bio_plots + "ANPP_detrendSens_40Yr_max.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_detrendLinReg_max"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'max(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP (detrendLinReg) maximum", fontdict=fontdict_bold);

file_name = bio_plots + "ANPP_detrendLinReg_40Yr_max.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_median"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'median(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP median", fontdict=fontdict_bold);

file_name = bio_plots + "ANPP_40Yr_median.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_detrendLinReg_median"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'median(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP detrendLinReg median", fontdict=fontdict_bold);

file_name = bio_plots + "anpp_detrendLinReg_40Yr_median.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_detrendSens_median"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'median(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP detrendSens median", fontdict=fontdict_bold);

file_name = bio_plots + "anpp_detrendSens_40Yr_median.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_variance"
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP variance", fontdict=fontdict_bold);


# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "ANPP_40Yr_variance_divergeRB_GreyBG.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_detrendSens_variance"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP detrendSens variance", fontdict=fontdict_bold);

file_name = bio_plots + "anpp_detrendSens_40Yr_variance.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
y_var = "anpp_detrendLinReg_variance"
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal);
plt.title("ANPP detrendLinReg variance", fontdict=fontdict_bold);

file_name = bio_plots + "anpp_detrendLinReg_40Yr_variance.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%

# %%
tick_legend_FontSize = 10
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
fig, axes = plt.subplots(1, 2, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

min_max1 = max(np.abs(Albers_SF_west['anpp_cv'].min()), np.abs(Albers_SF_west['anpp_cv'].max()))
min_max2 = max(np.abs(Albers_SF_west['anpp_variance'].min()), np.abs(Albers_SF_west['anpp_variance'].max()))

norm1 = Normalize(vmin=-min_max1, vmax=min_max1, clip=True)
norm2 = Normalize(vmin=-min_max2, vmax=min_max2, clip=True)

cent_plt1 = Albers_SF_west.plot(column='anpp_cv',       ax=ax1, legend=False, cmap='seismic', norm=norm1)
cent_plt2 = Albers_SF_west.plot(column='anpp_variance', ax=ax2, legend=False, cmap='seismic', norm=norm2)

cax1 = ax1.inset_axes(inset_axes_)
cax2 = ax2.inset_axes(inset_axes_)

cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax1)
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, norm=norm2, cax=cax2)

cbar1.set_label(r'CV(ANPP) $\times 100$', labelpad=1, fontdict=fontdict_normal);
cbar2.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal);

ax1.set_title("CV(ANPP)", fontdict=fontdict_bold);
ax2.set_title("variance of ANPP", fontdict=fontdict_bold);

plt.tight_layout()

# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "ANPP_40Yr_variance_and_CV_divergeRB_GreyBG.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt1, cax1, cbar1, norm1, min_max1, 
    cent_plt2, cax2, cbar2, norm2, min_max2)

# %%

# %%
fig, axes = plt.subplots(1, 2, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

min_max1 = max(np.abs(Albers_SF_west['anpp_detrendSens_cv'].min()), 
               np.abs(Albers_SF_west['anpp_detrendSens_cv'].max()))
min_max2 = max(np.abs(Albers_SF_west['anpp_detrendSens_variance'].min()), 
               np.abs(Albers_SF_west['anpp_detrendSens_variance'].max()))

norm1 = Normalize(vmin=-min_max1, vmax=min_max1, clip=True)
norm2 = Normalize(vmin=-min_max2, vmax=min_max2, clip=True)

cent_plt1 = Albers_SF_west.plot(column='anpp_detrendSens_cv',       ax=ax1, legend=False, 
                                cmap='seismic', norm=norm1)
cent_plt2 = Albers_SF_west.plot(column='anpp_detrendSens_variance', ax=ax2, legend=False, 
                                cmap='seismic', norm=norm2)

cax1 = ax1.inset_axes(inset_axes_)
cax2 = ax2.inset_axes(inset_axes_)

cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax1)
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, norm=norm2, cax=cax2)

cbar1.set_label(r'CV(ANPP) $\times 100$', labelpad=1, fontdict=fontdict_normal);
cbar2.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal);

ax1.set_title("CV(ANPP detrendSens)", fontdict=fontdict_bold);
ax2.set_title("variance of ANPP detrendSens", fontdict=fontdict_bold);
plt.tight_layout()

file_name = bio_plots + "anpp_detrendSens_40Yr_variance_and_CV.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt1, cax1, cbar1, norm1, min_max1, 
    cent_plt2, cax2, cbar2, norm2, min_max2)

# %%

# %%
fig, axes = plt.subplots(1, 2, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

min_max1 = max(np.abs(Albers_SF_west['anpp_detrendLinReg_cv'].min()), 
               np.abs(Albers_SF_west['anpp_detrendLinReg_cv'].max()))
min_max2 = max(np.abs(Albers_SF_west['anpp_detrendLinReg_variance'].min()), 
               np.abs(Albers_SF_west['anpp_detrendLinReg_variance'].max()))

norm1 = Normalize(vmin=-min_max1, vmax=min_max1, clip=True)
norm2 = Normalize(vmin=-min_max2, vmax=min_max2, clip=True)

cent_plt1 = Albers_SF_west.plot(column='anpp_detrendLinReg_cv',       ax=ax1, legend=False, 
                                cmap='seismic', norm=norm1)
cent_plt2 = Albers_SF_west.plot(column='anpp_detrendLinReg_variance', ax=ax2, legend=False, 
                                cmap='seismic', norm=norm2)

cax1 = ax1.inset_axes(inset_axes_)
cax2 = ax2.inset_axes(inset_axes_)

cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax1)
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, norm=norm2, cax=cax2)

cbar1.set_label(r'CV(ANPP) $\times 100$', labelpad=1, fontdict=fontdict_normal);
cbar2.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal);

ax1.set_title("CV(ANPP detrendLinReg)", fontdict=fontdict_bold);
ax2.set_title("variance of ANPP detrendLinReg", fontdict=fontdict_bold);
plt.tight_layout()

file_name = bio_plots + "anpp_detrendLinReg_40Yr_variance_and_CV.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt1, cax1, cbar1, norm1, min_max1, 
    cent_plt2, cax2, cbar2, norm2, min_max2)

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
cut_1 = 0.5
cut_2 = 0.75
###############################################################
df = Albers_SF_west[(Albers_SF_west['anpp_variance'] < cut_1 * (10**6))].copy()
min_max = max(np.abs(df['anpp_variance'].min()), np.abs(df['anpp_variance'].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt1 = df.plot(column='anpp_variance', ax=ax1, legend=False, cmap='seismic', norm=norm1)

print (df['anpp_variance'].min())
print (df['anpp_variance'].max())
print ()
###############################################################
df = Albers_SF_west[(Albers_SF_west['anpp_variance'] >= cut_1 * (10**6)) & 
                    (Albers_SF_west['anpp_variance'] < cut_2 * (10**6))].copy()

min_max = max(np.abs(df['anpp_variance'].min()), np.abs(df['anpp_variance'].max()))
norm2 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt2 = df.plot(column='anpp_variance', ax=ax2, legend=False, cmap='seismic', norm=norm2)

print (df['anpp_variance'].min())
print (df['anpp_variance'].max())
print ()
###############################################################
df = Albers_SF_west[Albers_SF_west['anpp_variance'] >= cut_2 * (10**6)].copy()
min_max = max(np.abs(df['anpp_variance'].min()), np.abs(df['anpp_variance'].max()))
norm3 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt3 = df.plot(column='anpp_variance', ax=ax3, legend=False, cmap='seismic', norm=norm3)

print (df['anpp_variance'].min())
print (df['anpp_variance'].max())
######################################################
cax = ax1.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal,
               fontsize=tick_legend_FontSize * .6);
ax1.set_title(r"ANPP variance (< 0.5 $\times$ 1e6)", fontdict=fontdict_bold);
######################################################

cax = ax2.inset_axes(inset_axes_)
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, cax=cax)
cbar2.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar2.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal,
               fontsize=tick_legend_FontSize * .6);
ax2.set_title(r"ANPP variance (in [0.5, 0.75] $\times$ 1e6)", fontdict=fontdict_bold);
######################################################
cax = ax3.inset_axes(inset_axes_)
cbar3 = fig.colorbar(cent_plt2.collections[1], ax=ax3, orientation='horizontal', shrink=0.3, cax=cax)
cbar3.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar3.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal,
               fontsize=tick_legend_FontSize * .6);
ax3.set_title(r"ANPP variance (> 0.75 $\times$ 1e6)", fontdict=fontdict_bold);

######################################################
file_name = bio_plots + "ANPP_40Yr_variance_divergeRB_3Categ_SeparNormal.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

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
cut_1 = 0.5
cut_2 = 0.75

min_max = max(np.abs(Albers_SF_west['anpp_variance'].min()), np.abs(Albers_SF_west['anpp_variance'].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

###############################################################
df = Albers_SF_west[(Albers_SF_west['anpp_variance'] < cut_1 * (10**6))].copy()
cent_plt1 = df.plot(column='anpp_variance', ax=ax1, legend=False, cmap='seismic', norm=norm1)
###############################################################
df = Albers_SF_west[(Albers_SF_west['anpp_variance'] >= cut_1 * (10**6)) & 
                    (Albers_SF_west['anpp_variance'] < cut_2 * (10**6))].copy()
cent_plt2 = df.plot(column='anpp_variance', ax=ax2, legend=False, cmap='seismic', norm=norm1)
###############################################################
df = Albers_SF_west[Albers_SF_west['anpp_variance'] >= cut_2 * (10**6)].copy()
cent_plt3 = df.plot(column='anpp_variance', ax=ax3, legend=False, cmap='seismic', norm=norm1)

######################################################
cax = ax1.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal,
               fontsize=tick_legend_FontSize * .6);
ax1.set_title(r"ANPP variance (< 0.5 $\times$ 1e6)", fontdict=fontdict_bold);
######################################################

cax = ax2.inset_axes(inset_axes_)
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, cax=cax)
cbar2.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar2.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal,
               fontsize=tick_legend_FontSize * .6);
ax2.set_title(r"ANPP variance (in [0.5, 0.75] $\times$ 1e6)", fontdict=fontdict_bold);
######################################################
cax = ax3.inset_axes(inset_axes_)
cbar3 = fig.colorbar(cent_plt2.collections[1], ax=ax3, orientation='horizontal', shrink=0.3, cax=cax)
cbar3.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar3.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict=fontdict_normal,
               fontsize=tick_legend_FontSize * .6);
ax3.set_title(r"ANPP variance (> 0.75 $\times$ 1e6)", fontdict=fontdict_bold);

######################################################
file_name = bio_plots + "ANPP_40Yr_variance_divergeRB_3Categ_identicalNormal.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, dpi=map_dpi_)

(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
plt.tight_layout()
fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.01, wspace=.1)
###############################################################
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

######
cut_1 = 30
###############################################################
df = Albers_SF_west[Albers_SF_west['anpp_cv'] < cut_1].copy()
min_max = max(np.abs(df['anpp_cv'].min()), np.abs(df['anpp_cv'].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt1 = df.plot(column='anpp_cv', ax=ax1, legend=False, cmap='seismic', norm=norm1)

print (df['anpp_cv'].min())
print (df['anpp_cv'].max())
print ()
###############################################################
df = Albers_SF_west[Albers_SF_west['anpp_cv'] >= cut_1].copy()

min_max = max(np.abs(df['anpp_cv'].min()), np.abs(df['anpp_cv'].max()))
norm2 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt2 = df.plot(column='anpp_cv', ax=ax2, legend=False, cmap='seismic', norm=norm2)

print (df['anpp_cv'].min())
print (df['anpp_cv'].max())
print ()
###############################################################
######################################################
cax1 = ax1.inset_axes(inset_axes_)
cax2 = ax2.inset_axes(inset_axes_)

cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, cax=cax1)
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, cax=cax2)

cbar1.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar2.ax.tick_params(labelsize=tick_legend_FontSize*0.6)

lab_ = r'CV(ANPP) $\times 100$'
cbar1.set_label(lab_, labelpad=1, fontdict=fontdict_normal, fontsize=tick_legend_FontSize * .6);
cbar2.set_label(lab_, labelpad=1, fontdict=fontdict_normal, fontsize=tick_legend_FontSize * .6);

ax1.set_title(f"CV(ANPP) < {cut_1}", fontdict=fontdict_bold);
ax2.set_title(f"CV(ANPP) > {cut_1}", fontdict=fontdict_bold);
######################################################
######################################################
file_name = bio_plots + "ANPP_40Yr_CV_divergeRB_2Categ_SeparNormal.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%

# %%
