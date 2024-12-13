# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
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


# %%
def plot_SF(SF, ax_, cmap_ = "Pastel1", col="EW_meridian"):
    SF.plot(column=col, ax=ax_, alpha=1, cmap=cmap_, edgecolor='k', legend=False, linewidth=0.1)


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
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
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

# %% [markdown]
# # Make some plots

# %%
# Albers_SF_west.plot(column='EW_meridian', categorical=True, legend=True);

# %%
county_fips_dict = pd.read_pickle(rangeland_reOrganized + "county_fips.sav")

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
# gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')
gdf = geopandas.read_file(rangeland_bio_data +'cb_2018_us_state_500k')

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
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, ax = plt.subplots(1, 1, figsize=(2, 3), sharex=True, sharey=True, dpi=dpi_)
# plt.title('rangeland polygons on western meridian')
# # divider = make_axes_locatable(ax)
# # cax = divider.append_axes("right", size="1%", pad=0, alpha=1)
# plot_SF(SF=visframe_mainLand_west, ax_=ax, cmap_ = "Pastel1", col="EW_meridian")
# Albers_SF_west["geometry"].centroid.plot(ax=ax, color='dodgerblue', markersize=0.051)

# plt.rcParams['axes.linewidth'] = .051
# # plt.legend(fontsize=10) # ax.axis('off')
# plt.show();

# %% [markdown]
# ### Plot a couple of examples

# %%
bpszone_ANPP_west = bpszone_ANPP.copy()

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
bpszone_ANPP_west = pd.merge(bpszone_ANPP_west, Albers_SF_west[cols_], how="left", on = "fid")
bpszone_ANPP_west.head(2)

# %%
font = {"size": 14}
matplotlib.rc("font", **font)
tick_legend_FontSize = 15
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
# drop trend so there is no bug later
print (ANPP_MK_df.shape)
ANPP_MK_df.drop(columns=["trend"], inplace=True)
Albers_SF_west.drop(columns=["trend"], inplace=True)
print (ANPP_MK_df.shape)

# %%
ANPP_MK_df.columns

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2, ax3) = axes
# ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both"); 
# ax3.grid(axis="both", which="both")
y_var = "mean_lb_per_acr"
######
###### subplot 1
######
target_idx = ANPP_MK_df["sens_slope"].max()
a_fid = ANPP_MK_df.loc[ANPP_MK_df["sens_slope"] == target_idx, "fid"].values[0]

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
ax1.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax1.plot(X["year"], y_pred, color="red", linewidth=3, label="regression fit");
ax1.legend(loc='best')

text_ = "trend: {}\nSen's slope {}, reg. slope {}\n{} (fid:{})".format(trend_, slope_, reg_slope, state_, a_fid)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/1.2)
ax1.text(2012, y_txt, text_, fontsize = 12);
# ax1.set_ylim(3000, 4500);
######
###### subplot 2
######
target_idx = ANPP_MK_df["sens_slope"].min()
a_fid = ANPP_MK_df.loc[ANPP_MK_df["sens_slope"] == target_idx, "fid"].values[0]

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
ax2.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax2.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, reg. slope {}\n{} (fid:{})".format(trend_, slope_, reg_slope, state_, a_fid)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/2.3)
ax2.text(2012, y_txt, text_, fontsize = 12);
# ax2.set_ylim(3000, 4500);
######
###### subplot 3
###### a location with no trend
a_fid = ANPP_MK_df.loc[ANPP_MK_df["trend_yue"] == "no trend", "fid"].values[0]
df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
ax3.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax3.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, reg. slope {}\n{} (fid:{})".format(trend_, slope_, reg_slope, state_, a_fid)
y_txt = int(df[y_var].max()/1.3)
ax3.text(2012, y_txt, text_, fontsize = 12);
# ax3.set_ylim(3000, 4500);
# plt.subplots_adjust(left=0.9, right=0.92, top=0.92, bottom=0.9)
ax1.set_title("three trend examples")
# plt.tight_layout();
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981)
file_name = bio_plots + "three_trends.pdf"
# This is redone in Weather_EDA.ipynb to add temp and precipitation Spearman's
# plt.savefig(file_name) 

# %%
np.sort(ANPP_MK_df["sens_slope"])[:10]

# %%
np.sort(ANPP_MK_df["sens_slope"])[-20:]

# %%

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2) = axes
# ax1.grid(axis='y', which='both')
a_fid = 100
df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
ax1.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax1.plot(X["year"], y_pred, color="red", linewidth=3, label="regression fit");

text_ = "trend: {}\nSen's slope {}, \nregression slope: {}\nstate: {}".format(trend_, slope_, reg_slope, state_)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/1.2)
ax1.text(1984, y_txt, text_, fontsize = 12);
ax1.legend(loc="lower right")

######### Remove outlier and plot
df = bpszone_ANPP_west[bpszone_ANPP_west["fid"] == 100].copy()

# remove outlier
outlier_index = df[['mean_lb_per_acr']].idxmin()
outlier_index = list(outlier_index)
df.drop(index=outlier_index, inplace=True)

# MK test
ANPP_TS = df["mean_lb_per_acr"].values
trend, _, p, z, Tau, s, var_s, slope, intercept = mk.original_test(ANPP_TS)
ax2.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax2.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, \nregression slope: {}\nstate: {}".format(trend_, slope_, reg_slope, state_)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/2.5)
ax2.text(2010, y_txt, text_, fontsize = 12);

ax1.set_title("Sen's slope is robust to outliers")
# plt.tight_layout();
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981)
file_name = bio_plots + "robustSensSlope.pdf"
plt.savefig(file_name)

# %%
np.sort(ANPP_MK_df.loc[ANPP_MK_df["trend_yue"] == "increasing", "sens_slope"])[:10]

# %%
np.sort(ANPP_MK_df.loc[ANPP_MK_df["trend_yue"] == "increasing", "sens_slope"])[-10:]

# %% [markdown]
# ### Plot everything and color based on slope

# %%

# %%
tick_legend_FontSize = 5
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
Albers_SF_west.head(2)

# %%
# fig, ax = plt.subplots(1,1, figsize=(3, 3), sharex=True, sharey=True, dpi=map_dpi_)
# ax.set_xticks([]); ax.set_yticks([])

# plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))
# cent_plt = Albers_SF_west.plot(column='sens_slope', legend=False, ax=ax, cmap = cm.get_cmap('RdYlGn'))
# cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.3, 
#                      cmap = cm.get_cmap('RdYlGn'))
# cbar1.set_label(r"Sen's slope")
# plt.title(r"rangeland trends (Sen's slope)")

# # plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981)
# file_name = bio_plots + "sensSlopes.png"
# plt.savefig(file_name)

# %% [markdown]
# In order to have the center at ```yellow``` we manipulated ```vmin``` and ```vmax```.
# Another way is [TwoSlopeNorm](https://matplotlib.org/stable/users/explain/colors/colormapnorms.html). Not pretty.
#
# Or from AI?
# ```norm = colors.MidpointNormalize(midpoint=midpoint, vmin=data.min(), vmax=data.max())```?

# %%
# fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
# ax.set_xticks([]); ax.set_yticks([])

# min_max = max(np.abs(Albers_SF_west['sens_slope'].min()), np.abs(Albers_SF_west['sens_slope'].max()))
# norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

# plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))

# cent_plt = Albers_SF_west.plot(column='sens_slope', ax=ax, legend=False,
#                                cmap = cm.get_cmap('RdYlGn'), norm=norm1)
# cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.3, 
#                      cmap = cm.get_cmap('RdYlGn'), norm=norm1)
# cbar1.set_label(r"Sen's slope")
# plt.title("rangeland trends (Sen's slope) - all locations")

# # plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981)
# file_name = bio_plots + "sensSlopes_centerColorBar.png"
# plt.savefig(file_name, dpi=450)

# %%
Albers_SF_west["trend_yue"].unique()

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west['sens_slope'].min()), np.abs(Albers_SF_west['sens_slope'].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))

cent_plt = Albers_SF_west.plot(column='sens_slope', ax=ax, legend=False,
                               cmap = cm.get_cmap('RdYlGn'), norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width 
# of the bar
cax = ax.inset_axes([0.08, 0.18, 0.45, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(r"Sen's slope", labelpad=1)
plt.title("Sen's slope - all locations")

# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "sensSlopes_centerColorBar.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
# Dec 22, 2024. Had a conversation with Matt.
Matt_df = Albers_SF_west.copy()
sens_thresh = 15
Matt_df = Matt_df[Matt_df["sens_slope"] > sens_thresh]

fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Matt_df['sens_slope'].min()), np.abs(Matt_df['sens_slope'].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))

cent_plt = Matt_df.plot(column='sens_slope', ax=ax, legend=False,
                               cmap = cm.get_cmap('RdYlGn'), norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width 
# of the bar
cax = ax.inset_axes([0.08, 0.18, 0.45, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(r"Sen's slope", labelpad=1)
plt.title("Sen's slope > " + str(sens_thresh), y=.97)

# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "/for_Matt/" + "sensSlopes_GE" + str(sens_thresh) + "_centerColorBar.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %% [markdown]
# ### Plot increasing trends and color based on slope
#
# The labels seem tobe based on p-values. increasing means **```p-value < 0.05```**.

# %%
print (ANPP_MK_df[ANPP_MK_df["trend_yue"] == "increasing"]["p"].max())
print (ANPP_MK_df[ANPP_MK_df["trend_yue"] == "increasing"]["p"].min())

# %%
Albers_SF_west.columns

# %%
# Update Dec. 3, 2024. Add Yue's new locations to this plot
Albers_SF_west_increase = Albers_SF_west[Albers_SF_west["trend_yue"] == "increasing"]
Albers_SF_west_increase.shape

# %%

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_="Pastel1")
cent_plt = Albers_SF_west_increase.plot(column='sens_slope', ax=ax)

cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
# cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.5)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label(r"Sen's slope", labelpad=1)
plt.title("Sen's slope - greening locations")

# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981)
file_name = bio_plots + "greening_sensSlope.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1)

# %%

# %%
fig, ax = plt.subplots(1,1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['white', 'black']))
cent_plt = Albers_SF_west_increase.plot(column='Tau', ax=ax, cmap=cm.get_cmap('RdYlGn'))

cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
# cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.5)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label(r"Kendall's $\tau$", labelpad=1)
plt.title(r"Kendall's $\tau$ - greening locations")

plt.tight_layout()
plt.show();
del(cent_plt, cax, cbar1)

# %%

# %% [markdown]
# ### Plot positive Spearman's with p-value smaller than 0.05

# %%
print (Albers_SF_west["Spearman"].min())
Albers_SF_west.head(2)

# %%
Albers_SF_west_spearmanP5 = Albers_SF_west[(Albers_SF_west["Spearman"] > 0) & 
                                           (Albers_SF_west["p_Spearman"] < 0.05)]


# %%
fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
axes.set_xticks([]); axes.set_yticks([])

plot_SF(SF=visframe_mainLand_west, ax_=axes, cmap_ = "Pastel1", col="EW_meridian")
spearman_plot_s = Albers_SF_west_spearmanP5.plot(column='Spearman', ax=axes)

# Add colorbar for Spearman's plot
# cbar1 = fig.colorbar(spearman_plot_s.collections[1], ax=axes, orientation='vertical', shrink=0.6)
cax = axes.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(spearman_plot_s.collections[1], ax=axes, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label('Spearman\'s rank', labelpad=1)

axes.set_title("Spearman's rank - greening locations", y=0.97)
plt.tight_layout()
plt.show();

del(spearman_plot_s, cax, cbar1)

# %%
Albers_SF_west.head(2)

# %%

# %%
# Creating the figure and axes
fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
axes.set_xticks([]); axes.set_yticks([])

plot_SF(SF=visframe_mainLand_west, ax_=axes, cmap_ = "Pastel1", col="EW_meridian")

Albers_SF_west_median_diff_increase = Albers_SF_west[Albers_SF_west["median_ANPP_change_as_perc"] > 0].copy()
plot_s = Albers_SF_west_median_diff_increase.plot(column='median_ANPP_change_as_perc', ax=axes)

# Add colorbar for Spearman's plot
# cbar1 = fig.colorbar(plot_s.collections[1], ax=axes, orientation='vertical', shrink=0.6)
cax = axes.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(plot_s.collections[1], ax=axes, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label('median ANPP change %', labelpad=1)

axes.set_title(r"greening trends (median_ANPP_change_as_perc > 0)", y=0.97)

fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.05, wspace=0.05)
file_name = bio_plots + "medianNPP_percChange.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(plot_s, cax, cbar1)

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
axes.set_xticks([]); axes.set_yticks([])

plot_SF(SF=visframe_mainLand_west, ax_=axes, cmap_ = "Pastel1", col="EW_meridian")

Albers_SF_west_median_diff_increase = Albers_SF_west[Albers_SF_west["median_ANPP_change_as_perc"] > 0].copy()
plot_s = Albers_SF_west_median_diff_increase.plot(column='median_ANPP_change_as_perc', ax=axes, cmap=cmap_G)

# Add colorbar for Spearman's plot
# cbar1 = fig.colorbar(plot_s.collections[1], ax=axes, orientation='vertical', shrink=0.6)
# cbar1.set_label('median ANPP change %')
####
cax = axes.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(plot_s.collections[1], ax=axes, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label('median ANPP change %', labelpad=1)
#####
axes.set_title(r"greening trends (median_ANPP_change_as_perc > 0)", y=0.97)

fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.05, wspace=0.05)
file_name = bio_plots + "medianNPP_percChange_GreenCmap.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
del(plot_s, cax, cbar1)

# %%

# %% [markdown]
# ## Side by side

# %%
# Parameters for font sizes
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * 0.71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.7,
          "ytick.labelsize": tick_legend_FontSize * 0.7,
          "axes.titlepad": 5, 
          "legend.handlelength": 2,
         'axes.linewidth' : .05}
plt.rcParams.update(params)

# %%

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
###############################################################
plot_SF(SF=visframe_mainLand_west, ax_=ax1, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))
plot_SF(SF=visframe_mainLand_west, ax_=ax2, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))
###############################################################
p1 = Albers_SF_west.plot(column='sens_slope', legend=False, ax=ax1, cmap = cm.get_cmap('RdYlGn'))

cax1 = ax1.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(p1.collections[1], ax=ax1, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), cax=cax1)
cbar1.set_label(r"Sen's slope", labelpad=1)
###############################################################

min_max = max(np.abs(Albers_SF_west['sens_slope'].min()), np.abs(Albers_SF_west['sens_slope'].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)
p2 = Albers_SF_west.plot(column='sens_slope', ax=ax2, legend=False,
                               cmap = cm.get_cmap('RdYlGn'), norm=norm1)

cax2 = ax2.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar2 = fig.colorbar(p2.collections[1], ax=ax2, orientation='horizontal', shrink=0.3, 
                     cmap = cm.get_cmap('RdYlGn'), norm=norm1, cax=cax2)
cbar2.set_label(r"Sen's slope", labelpad=1)

# plt.tight_layout();
# plt.show();
fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.01, wspace=-0.2)
file_name = bio_plots + "sensSlopes_2colorbars.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(p1, p2, cax2, cax1, cbar1, cbar2, norm1, min_max)

# %%

# %%
# Creating the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_,
                         gridspec_kw={"hspace": 0.15, "wspace": 0.01})
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])

min_col_ = min(Albers_SF_west_spearmanP5['Spearman'].min(), Albers_SF_west_increase['Tau'].min())
max_col_ = max(Albers_SF_west_increase['Tau'].max(), Albers_SF_west_spearmanP5['Spearman'].max())
norm_col = Normalize(vmin= min_col_, vmax = max_col_);

plot_SF(SF = visframe_mainLand_west, cmap_=custom_cmap_coral, ax_ = ax1, col="EW_meridian")
plot_SF(SF = visframe_mainLand_west, cmap_=custom_cmap_coral, ax_ = ax2, col="EW_meridian")

####### Spearman's rank plot
p1 = Albers_SF_west_spearmanP5.plot(column='Spearman', ax=ax1, cmap=cmap_G, norm=norm_col)
ax1.set_title("greening trends (Spearman's rank based)")
# Create a continuous colorbar for Spearman's plot
# cbar1 = fig.colorbar(cm.ScalarMappable(norm=norm1, cmap=cmap), ax=ax1, orientation='vertical', shrink=0.8)
# cbar1.set_label("Spearman's rank")
####### Kendall's tau plot
p2 = Albers_SF_west_increase.plot(column='Tau', ax=ax2, cmap=cmap_G, norm=norm_col)
ax2.set_title(r"greening trends (Kendall's $\tau$ - MK based)")

# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.01, wspace=0.01)
cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical')
# fig.colorbar(p1.get_children()[1], ax=axes, fraction=0.02, location='bottom', orientation='horizontal')
# plt.tight_layout()
# plt.show();

file_name = bio_plots + "Spearman_tau.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_);

del(p1, p2, cax, norm_col, min_col_, max_col_)

# %%

# %%
# Albers_SF_west_median_diff_increase = Albers_SF_west[Albers_SF_west["medians_diff_slope_ANPP"] > 0].copy()

# min_col_ = min(Albers_SF_west_increase['sens_slope'].min(), 
#                Albers_SF_west_median_diff_increase['medians_diff_slope_ANPP'].min())
# max_col_ = max(Albers_SF_west_increase['sens_slope'].max(), 
#                Albers_SF_west_median_diff_increase['medians_diff_slope_ANPP'].max())
# norm_col = Normalize(vmin= min_col_, vmax = max_col_);

# fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
# (ax1, ax2) = axes
# #############
# plot_SF(SF=visframe_mainLand_west, ax_=ax1, cmap_="Pastel1", col="EW_meridian")
# plot_SF(SF=visframe_mainLand_west, ax_=ax2, cmap_="Pastel1", col="EW_meridian")
# #############
# p1 = Albers_SF_west_increase.plot(column='sens_slope', ax=ax1, norm=norm_col)
# ax1.set_title(r"greening trends (Sen's slope)")
# #############
# p2 = Albers_SF_west_median_diff_increase.plot(column='medians_diff_slope_ANPP', ax=ax2, norm=norm_col)
# ax2.set_title(r"greening trends (ANPP medians diff slope)")

# cbar = fig.colorbar(p2.get_children()[1], ax=axes, fraction=0.02,
#                     location='bottom', orientation='horizontal')

# ax1.set_xticks([]); ax1.set_yticks([])
# ax2.set_xticks([]); ax2.set_yticks([])
# # plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981, hspace=0.01, wspace=0.01)
# # plt.show();
# del(norm_col)
# # file_name = bio_plots + "Sens_MedianDiffSlope.pdf"
# # plt.savefig(file_name);

# %%

# %%
Albers_SF_west_median_diff_increase = Albers_SF_west[Albers_SF_west["medians_diff_slope_ANPP"] > 0].copy()

min_col_ = min(Albers_SF_west_increase['sens_slope'].min(), 
               Albers_SF_west_median_diff_increase['medians_diff_slope_ANPP'].min())
max_col_ = max(Albers_SF_west_increase['sens_slope'].max(), 
               Albers_SF_west_median_diff_increase['medians_diff_slope_ANPP'].max())
norm_col = Normalize(vmin= min_col_, vmax = max_col_);

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_,
                         gridspec_kw={"hspace": 0.15, "wspace": -0.11})
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])

#############
plot_SF(SF=visframe_mainLand_west, ax_=ax1, cmap_="Pastel1", col="EW_meridian")
plot_SF(SF=visframe_mainLand_west, ax_=ax2, cmap_="Pastel1", col="EW_meridian")
#############
p1 = Albers_SF_west_increase.plot(column='sens_slope', ax=ax1, cmap=cmap_G, norm=norm_col)
ax1.set_title(r"greening trends (Sen's slope)")
#############
p2 = Albers_SF_west_median_diff_increase.plot(column='medians_diff_slope_ANPP', ax=ax2, cmap=cmap_G, norm=norm_col)
ax2.set_title(r"greening trends (ANPP medians diff slope)")

# fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-1, hspace=0)
cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical')
#fig.colorbar(p1.get_children()[1], ax=axes, fraction=0.02, location='bottom', orientation='horizontal')
# plt.tight_layout()
# plt.show();
file_name = bio_plots + "Sens_MedianDiffSlope.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_);

del(p1, p2, cax, norm_col, min_col_, max_col_)

# %% [markdown]
# # Investigate large change in median diff

# %%
Albers_SF_west_median_diff_increase.head(2)

# %%
max_loc = Albers_SF_west_median_diff_increase["median_ANPP_change_as_perc"].idxmax()
Albers_SF_west_median_diff_increase.loc[max_loc]

# %%
max_percChange_median_fid = Albers_SF_west_median_diff_increase.loc[max_loc]["fid"]

# %%
# after seaborn, things get messed up. Fix them:
matplotlib.rc_file_defaults()
font = {"size": 14}
matplotlib.rc("font", **font)
tick_legend_FontSize = 10
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1.2,
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
a_fid = max_percChange_median_fid
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
axes.grid(axis='y', which='both')

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
axes.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
axes.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, \nregression slope: {}\nstate: {},\nFID: {}".format(trend_, 
                                                                                        slope_, 
                                                                                        reg_slope, 
                                                                                        state_,
                                                                                        a_fid)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/2)
axes.text(1984, y_txt, text_, fontsize = 12);
axes.set_title("Maximum change (%) in ANPP median")
# plt.tight_layout()
# plt.show();
file_name = bio_plots + "maxPercChangeANPPmediands.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_);


# %%

# %%
tick_legend_FontSize = 10
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1.2,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "ytick.left":True,
          "ytick.labelleft":True,
          "xtick.bottom":False,
          "xtick.labelbottom":True,
         'axes.linewidth' : .05}
plt.rcParams.update(params)

# %%

# %%
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : True})

sns.histplot(data=Albers_SF_west["Tau"], ax=axes[0], bins=100, kde=True); # height=5
axes[0].legend(["Kendal's Tau"], loc='upper left');

sns.histplot(data=Albers_SF_west["Spearman"], ax=axes[1], bins=100, kde=True); # height=5
axes[1].legend(["Spearman's rank"], loc='upper left');

sns.histplot(data=Albers_SF_west["sens_slope"], ax=axes[2], bins=100, kde=True); # height=5
axes[2].legend(["Sen's slope"], loc='upper left');

sns.histplot(data=Albers_SF_west["medians_diff_slope_ANPP"], ax=axes[3], bins=100, kde=True); # height=5
axes[3].legend(["medians_diff_slope_ANPP"], loc='upper right');

sns.histplot(data=Albers_SF_west["median_ANPP_change_as_perc"], ax=axes[4], bins=100, kde=True); # height=5
axes[4].legend(["median_ANPP_change (%)"], loc='upper right');

axes[0].set_xlabel(""); axes[1].set_xlabel("");
axes[2].set_xlabel(""); axes[3].set_xlabel(""); axes[4].set_xlabel("");

# plt.suptitle(title_, fontsize=15, y=.94);
# plt.tight_layout()
# plt.show();
file_name = bio_plots + "trend_distributions.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_);

# %% [markdown]
# # Same plot as above. Just pick the ones with low p-value

# %%

# %%
significant_sens = Albers_SF_west[Albers_SF_west["trend_yue"].isin(["increasing", "decreasing"])].copy()
significant_spearman = Albers_SF_west[Albers_SF_west["p_Spearman"] < 0.05].copy()

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : True})

sns.histplot(data=significant_spearman["Spearman"], ax=axes[0], bins=100, kde=True); # height=5
axes[0].legend(["Spearman's rank"], loc='upper left');

sns.histplot(data=significant_sens["Tau"], ax=axes[1], bins=100, kde=True); # height=5
axes[1].legend(["Kendal's Tau"], loc='upper left');

sns.histplot(data=significant_sens["sens_slope"], ax=axes[2], bins=100, kde=True); # height=5
axes[2].legend(["Sen's slope"], loc='upper left');

# sns.histplot(data=Albers_SF_west["medians_diff_slope_ANPP"], ax=axes[3], bins=100, kde=True); # height=5
# axes[3].legend(["medians_diff_slope_ANPP"], loc='upper right');

# sns.histplot(data=Albers_SF_west["median_ANPP_change_as_perc"], ax=axes[4], bins=100, kde=True); # height=5
# axes[4].legend(["median_ANPP_change_as_perc"], loc='upper right');

axes[0].set_xlabel(""); axes[1].set_xlabel(""); axes[2].set_xlabel("");
# axes[3].set_xlabel("");
# axes[4].set_xlabel("");

# plt.suptitle(title_, fontsize=15, y=.94);
# plt.tight_layout()
# plt.show();
file_name = bio_plots + "trend_distributions_significant.pdf"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_);

# %% [markdown]
# # Outliers of Spearman's?

# %%
significant_spearman.head(2)

# %%
max_loc = significant_spearman["Spearman"].idxmax()
max_spearman_fid = significant_spearman.loc[max_loc]["fid"]

# %%
significant_spearman.loc[max_spearman_fid]

# %%
# after seaborn, things get messed up. Fix them:
matplotlib.rc_file_defaults()
tick_legend_FontSize = 10
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1.2,
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
a_fid = max_spearman_fid
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
axes.grid(axis='y', which='both')

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
axes.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
axes.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope: {}, \nregression slope: {}\nstate: {},\nFID: {}".format(trend_, 
                                                                                         slope_, 
                                                                                         reg_slope, 
                                                                                         state_,
                                                                                         a_fid)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/3.4)
axes.text(1984, y_txt, text_, fontsize = 12);

# %% [markdown]
# ## Find FIDs that are in the intersection of Spearman and MK test

# %%
intersection_FIDs = set(significant_spearman["fid"]).intersection(set(significant_sens["fid"]))
intersection_FIDs = list(intersection_FIDs)
Albers_SF_west_intersec = Albers_SF_west[Albers_SF_west["fid"].isin(intersection_FIDs)]
Albers_SF_west_intersec = Albers_SF_west_intersec[Albers_SF_west_intersec["sens_slope"] > 0]

# %%

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : True})

axes[0].set_title('Intersection of sigfinicant MK test and Spearman (increasing trend)');

sns.histplot(data=Albers_SF_west_intersec["Tau"], ax=axes[0], bins=100, kde=True);
axes[0].legend(["Kendal's Tau"], loc='upper right');

sns.histplot(data=Albers_SF_west_intersec["Spearman"], ax=axes[1], bins=100, kde=True);
axes[1].legend(["Spearman's rank"], loc='upper right');

sns.histplot(data=Albers_SF_west_intersec["sens_slope"], ax=axes[2], bins=100, kde=True); 
axes[2].legend(["Sen's slope"], loc='upper right');

axes[0].set_xlabel(""); axes[1].set_xlabel(""); axes[2].set_xlabel("");

# %% [markdown]
# # no trend locations

# %%
# Parameters for font sizes
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * 0.71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.7,
          "ytick.labelsize": tick_legend_FontSize * 0.7,
          "axes.titlepad": 5,
          "legend.handlelength": 2}
plt.rcParams.update(params)

# %%
print (Albers_SF_west["trend_yue"].unique())
Albers_SF_west.head(2)

# %%
no_trend_df = Albers_SF_west[Albers_SF_west["trend_yue"] == "no trend"].copy()
green_df_MK_based = Albers_SF_west[Albers_SF_west["trend_yue"] == "increasing"].copy()

# %%

# %%
min_color = min(no_trend_df['Spearman'].min(), no_trend_df['Tau'].min())
max_color = max(no_trend_df['Spearman'].max(), no_trend_df['Tau'].max())
norm1 = Normalize(vmin = min_color, vmax = max_color)

# Creating the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
####### Spearman's rank plot
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_=ax1, col="EW_meridian")
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_=ax2, col="EW_meridian")

p1 = no_trend_df.plot(column='Spearman', ax=ax1, cmap=cmap_G, norm=norm1)
ax1.set_title("no trend locations (Spearman's rank - MK based)")

####### Kendall's tau plot
p2 = no_trend_df.plot(column='Tau', ax=ax2, cmap=cmap_G, norm=norm1)
ax2.set_title(r"no trend locations (Kendall's $\tau$ - MK based)")

cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical')
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-0.1, hspace=0)
# fig.colorbar(p1.get_children()[1], ax=axes, fraction=0.02, orientation='horizontal', location="bottom")
# plt.tight_layout();

plt.show();
del(p1, p2, cax, norm1, min_color, max_color)

# %%

# %%
min_color = min(no_trend_df['sens_slope'].min(), green_df_MK_based['sens_slope'].min())
max_color = max(no_trend_df['sens_slope'].max(), green_df_MK_based['sens_slope'].max())
norm_colorB = Normalize(vmin = min_color, vmax = max_color)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])

####### States
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_ = ax1, col="EW_meridian")
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_ = ax2, col="EW_meridian")
####################################################################################
p1 = no_trend_df.plot(column='sens_slope', ax = ax1, cmap=cmap_G, norm=norm_colorB, legend=False)
ax1.set_title(r"no trend locations (Sen's slope)")

p2 = green_df_MK_based.plot(column='sens_slope', ax=ax2, cmap=cmap_G, norm=norm_colorB, legend=False)
ax2.set_title(r"greening locations (Sen's slope)")

fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-0.1, hspace=0)
cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical')
# fig.colorbar(p1.get_children()[1], ax=axes, fraction=0.02, orientation='horizontal', location="bottom")
# plt.tight_layout()

plt.show();
del(p1, p2, cax, norm_colorB, min_color, max_color)

# %%

# %%
min_color = min(no_trend_df['Spearman'].min(), green_df_MK_based['Spearman'].min())
max_color = max(no_trend_df['Spearman'].max(), green_df_MK_based['Spearman'].max())
norm_colorB = Normalize(vmin =min_color, vmax=max_color)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
####### States
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_ = ax1, col="EW_meridian")
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_ = ax2, col="EW_meridian")

####################################################################################
p1 = no_trend_df.plot(column='Spearman', ax=ax1, cmap=cmap_G, norm=norm_colorB, legend=False)
ax1.set_title("no trend locations (Spearman's rank - MK based)")

p2 = green_df_MK_based.plot(column='Spearman', ax = ax2, cmap=cmap_G, norm=norm_colorB, legend=False)
ax2.set_title("greening locations (Spearman's rank - MK based)")
####################################################################################
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-0.1, hspace=0)
cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical')
# fig.colorbar(p1.get_children()[1], ax=axes,  fraction=0.02, orientation='horizontal', location="bottom")
# plt.tight_layout()
del(norm_colorB)
plt.show();
del(p1, p2, cax)

# %%

# %%
min_color = min(no_trend_df['Tau'].min(), green_df_MK_based['Tau'].min())
max_color = max(no_trend_df['Tau'].max(), green_df_MK_based['Tau'].max())
norm_colorB = Normalize(vmin =min_color, vmax = max_color)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])

####### States
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_ = ax1, col="EW_meridian")
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_ = ax2, col="EW_meridian")

####################################################################################
p1 = no_trend_df.plot(column='Tau', ax=ax1, cmap=cmap_G, norm=norm_colorB, legend=False)
ax1.set_title(r"no trend locations (Kenda's $\tau$ - MK based)")

p2 = green_df_MK_based.plot(column='Tau', ax = ax2, cmap=cmap_G, norm=norm_colorB, legend=False)
ax2.set_title(r"greening locations (Kenda's $\tau$ - MK based)")
####################################################################################
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-0.1, hspace=0)
cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical')
# fig.colorbar(p1.get_children()[1], ax=axes,  fraction=0.02, orientation='horizontal', location="bottom")
# plt.tight_layout()
del(norm_colorB)
plt.show();

del(p1, p2, cax)

# %%
tick_legend_FontSize = 5
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * .71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .7,
          "ytick.labelsize": tick_legend_FontSize * .7,
          "axes.titlepad": 5,
          "legend.handlelength": 2,
          "xtick.bottom" : False,
          "ytick.left" : False,
          "xtick.labelbottom": False,
          "ytick.labelleft": False,
         'axes.linewidth' : .2}
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
axes.set_xticks([]); axes.set_yticks([])
####### States
plot_SF(SF = visframe_mainLand_west, cmap_="Pastel1", ax_=axes, col="EW_meridian")
####################################################################################
p1 = no_trend_df.plot(column='Tau', ax=axes, cmap=cmap_R, legend=False)
p2 = green_df_MK_based.plot(column='Tau', ax = axes, cmap=cmap_G, legend=False)

cax1 = axes.inset_axes([1, 0.00, 0.03, 0.45])
cax2 = axes.inset_axes([1, 0.55, 0.03, 0.45])

cbar1 = fig.colorbar(p1.collections[1], ax=axes, orientation='vertical', shrink=0.3, cmap=cmap_R, cax=cax1)
cbar2 = fig.colorbar(p2.collections[2], ax=axes, orientation='vertical', shrink=0.3, cmap=cmap_G, cax=cax2)

cbar1.set_label(r"$Kendal's~\tau$: (no trend locations)", labelpad=1)
cbar2.set_label(r"$Kendal's~\tau$: (greening locations)", labelpad=1)
####################################################################################
axes.set_title(r"Kendal's $\tau$: no trends in red, greening in green")
plt.tight_layout()
# plt.show();
file_name = bio_plots + "noTrend_green_locs.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(p1, p2, cax1, cax2, cbar1, cbar2)

# %%

# %% [markdown]
# # 400 difference in size between slope and Spearmans rank! 
# We need to check if the smaller set is subset of the larger set

# %%
print (Albers_SF_west_spearmanP5.shape)
Albers_SF_west_increase.shape

# %%
