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
import pymannkendall as mk

import statistics
import statsmodels.api as sm
from scipy import stats

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
dpi_, map_dpi_=300, 900
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

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
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.rename(columns={"area_sqMeter": "area_sqmeter", 
                     "count": "pixel_count",
                     "mean" : "mean_lb_per_acr"}, inplace=True)

ANPP.sort_values(by=['fid', 'year'], inplace=True)
ANPP.head(2)

# %%
col = "mean_lb_per_acr"
sum(ANPP[col] - bpszone_ANPP[col])

# %%
col = "area_sqmeter"
sum(ANPP[col] == bpszone_ANPP[col]) == len(ANPP)

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
Albers_SF_west = geopandas.read_file(f_name)
Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid
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
NPP_variance_df = ANPP.groupby('fid')['mean_lb_per_acr'].var().reset_index()
NPP_variance_df.columns = ['fid', 'mean_lb_per_acr_variance']
NPP_variance_df.head(2)

# %%

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
Albers_SF_west = pd.merge(Albers_SF_west, NPP_variance_df, how="left", on="fid")
Albers_SF_west.head(2)

# %%
y_var = "mean_lb_per_acr_variance"

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
sharey_ = False ### set axis limits to be identical or not

fig, axes = plt.subplots(1, 1, figsize=(7, 2), sharey=sharey_, sharex=True, dpi=dpi_)
axes.grid(axis='y', alpha=0.7, zorder=0);

axes.hist(Albers_SF_west[y_var].dropna(), zorder=3,
          bins=100, color='skyblue', edgecolor='black')

axes.set_ylabel('count')
axes.set_title('ANPP variance distribution', color="k", fontdict={'family':'serif', 'weight':'bold'});
axes.set_xlabel('Variance of mean ANPP (lb/acre)', fontdict={'family':'serif', 'weight':'normal'});
axes.set_ylabel('Frequency', fontdict={'family':'serif', 'weight':'normal'});

file_name = bio_plots + "ANPP_40Yr_variance_histogram.pdf"
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
y_var

# %%
print (np.abs(Albers_SF_west[y_var].min()))
print (np.abs(Albers_SF_west[y_var].max()))


# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['green', 'grey']))

# cent_plt = Albers_SF_west["centroid"].plot(ax=ax, c=Albers_SF_west[y_var], cmap='seismic',
#                                            norm=norm1, markersize=0.1)
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes([0.08, 0.18, 0.45, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict={'family':'serif', 'weight':'normal'});
plt.title("ANPP variance", fontdict={'family':'serif', 'weight':'bold'});

# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "ANPP_40Yr_variance_divergeRB_GrnBG.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
fig, ax = plt.subplots(1, 1, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)
cent_plt = Albers_SF_west.plot(column=y_var, ax=ax, legend=False, cmap=cm.get_cmap('RdYlGn'), norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width 
# of the bar
cax = ax.inset_axes([0.08, 0.18, 0.45, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict={'family':'serif', 'weight':'normal'});
plt.title("ANPP variance", fontdict={'family':'serif', 'weight':'bold'});

# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "ANPP_40Yr_variance.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.01, wspace=-0.2)
###############################################################
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))

###############################################################
df = Albers_SF_west[Albers_SF_west[y_var] < 0.75 * (10**6)].copy()
min_max = max(np.abs(df[y_var].min()), np.abs(df[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt = df.plot(column=y_var, ax=ax1, legend=False, cmap=cm.get_cmap('RdYlGn'), norm=norm1)

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.01, wspace=-0.2)

###############################################################
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))

######
cut_ = 0.75
###############################################################
df = Albers_SF_west[Albers_SF_west[y_var] < cut_ * (10**6)].copy()
print (df[y_var].min())
print (df[y_var].max())
print ()
min_max = max(np.abs(df[y_var].min()), np.abs(df[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt1 = df.plot(column=y_var, ax=ax1, legend=False, cmap='seismic', norm=norm1)

###############################################################
df = Albers_SF_west[Albers_SF_west[y_var] >= cut_ * (10**6)].copy()
print (df[y_var].min())
print (df[y_var].max())
min_max = max(np.abs(df[y_var].min()), np.abs(df[y_var].max()))
norm2 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt2 = df.plot(column=y_var, ax=ax2, legend=False, cmap='seismic', norm=norm1)

# %%

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2, ax3) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
ax3.set_xticks([]); ax3.set_yticks([])
fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.01, wspace=-.2)
###############################################################
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax3, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))

######
cut_1 = 0.5
cut_2 = 0.75
###############################################################
df = Albers_SF_west[(Albers_SF_west[y_var] < cut_1 * (10**6))].copy()
min_max = max(np.abs(df[y_var].min()), np.abs(df[y_var].max()))
norm1 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt1 = df.plot(column=y_var, ax=ax1, legend=False, cmap='seismic', norm=norm1)

print (df[y_var].min())
print (df[y_var].max())
print ()
###############################################################
df = Albers_SF_west[(Albers_SF_west[y_var] >= cut_1 * (10**6)) & (Albers_SF_west[y_var] < cut_2 * (10**6))].copy()

min_max = max(np.abs(df[y_var].min()), np.abs(df[y_var].max()))
norm2 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt2 = df.plot(column=y_var, ax=ax2, legend=False, cmap='seismic', norm=norm2)

print (df[y_var].min())
print (df[y_var].max())
print ()
###############################################################
df = Albers_SF_west[Albers_SF_west[y_var] >= cut_2 * (10**6)].copy()
min_max = max(np.abs(df[y_var].min()), np.abs(df[y_var].max()))
norm3 = Normalize(vmin=-min_max, vmax=min_max, clip=True)
cent_plt3 = df.plot(column=y_var, ax=ax3, legend=False, cmap='seismic', norm=norm3)

print (df[y_var].min())
print (df[y_var].max())

######################################################
cax = ax1.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict={'family':'serif', 'weight':'normal'},
               fontsize=tick_legend_FontSize * .6);
ax1.set_title(r"ANPP variance (< 0.5 $\times$ 1e6)", fontdict={'family':'serif', 'weight':'bold'});
######################################################

cax = ax2.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, cax=cax)
cbar2.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar2.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict={'family':'serif', 'weight':'normal'},
               fontsize=tick_legend_FontSize * .6);
ax2.set_title(r"ANPP variance (in [0.5, 0.75] $\times$ 1e6)", fontdict={'family':'serif', 'weight':'bold'});
######################################################
cax = ax3.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar3 = fig.colorbar(cent_plt2.collections[1], ax=ax3, orientation='horizontal', shrink=0.3, cax=cax)
cbar3.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar3.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict={'family':'serif', 'weight':'normal'},
               fontsize=tick_legend_FontSize * .6);
ax3.set_title(r"ANPP variance (> 0.75 $\times$ 1e6)", fontdict={'family':'serif', 'weight':'bold'});

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
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))
rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax3, col="EW_meridian", cmap_=ListedColormap(['green', 'white']))

######
cut_1 = 0.5
cut_2 = 0.75

min_max = max(np.abs(Albers_SF_west[y_var].min()), np.abs(Albers_SF_west[y_var].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

###############################################################
df = Albers_SF_west[(Albers_SF_west[y_var] < cut_1 * (10**6))].copy()
cent_plt1 = df.plot(column=y_var, ax=ax1, legend=False, cmap='seismic', norm=norm1)
###############################################################
df = Albers_SF_west[(Albers_SF_west[y_var] >= cut_1 * (10**6)) & (Albers_SF_west[y_var] < cut_2 * (10**6))].copy()
cent_plt2 = df.plot(column=y_var, ax=ax2, legend=False, cmap='seismic', norm=norm1)
###############################################################
df = Albers_SF_west[Albers_SF_west[y_var] >= cut_2 * (10**6)].copy()
cent_plt3 = df.plot(column=y_var, ax=ax3, legend=False, cmap='seismic', norm=norm1)

######################################################
cax = ax1.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar1.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict={'family':'serif', 'weight':'normal'},
               fontsize=tick_legend_FontSize * .6);
ax1.set_title(r"ANPP variance (< 0.5 $\times$ 1e6)", fontdict={'family':'serif', 'weight':'bold'});
######################################################

cax = ax2.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar2 = fig.colorbar(cent_plt2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, cax=cax)
cbar2.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar2.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict={'family':'serif', 'weight':'normal'},
               fontsize=tick_legend_FontSize * .6);
ax2.set_title(r"ANPP variance (in [0.5, 0.75] $\times$ 1e6)", fontdict={'family':'serif', 'weight':'bold'});
######################################################
cax = ax3.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar3 = fig.colorbar(cent_plt2.collections[1], ax=ax3, orientation='horizontal', shrink=0.3, cax=cax)
cbar3.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar3.set_label(f'$\sigma^2$(ANPP)', labelpad=1, fontdict={'family':'serif', 'weight':'normal'},
               fontsize=tick_legend_FontSize * .6);
ax3.set_title(r"ANPP variance (> 0.75 $\times$ 1e6)", fontdict={'family':'serif', 'weight':'bold'});

######################################################
file_name = bio_plots + "ANPP_40Yr_variance_divergeRB_3Categ_identicalNormal.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# %%
