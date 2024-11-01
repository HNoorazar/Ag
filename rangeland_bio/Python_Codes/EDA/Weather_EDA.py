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
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import pymannkendall as mk

import statistics
import statsmodels.api as sm

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rc("font", family="Palatino")

# font = {"size": 10}
# matplotlib.rc("font", **font)

import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from sklearn.model_selection import train_test_split
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

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
# ####### Laptop
# rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
# min_bio_dir = rangeland_bio_base

# rangeland_base = rangeland_bio_base
# rangeland_reOrganized = rangeland_base

# %%
def plot_SF(SF, ax_, cmap_ = "Pastel1", col="EW_meridian"):
    SF.plot(column=col, ax=ax_, alpha=1, cmap=cmap_, edgecolor='k', legend=False, linewidth=0.1)


# %%
dpi_ = 300
map_dpi_ = 200
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')
cmap_RYG = cm.get_cmap('RdYlGn')

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

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


from shapely.geometry import Polygon
gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')
# gdf = geopandas.read_file(rangeland_bio_base +'cb_2018_us_state_500k')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")


visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)
# ANPP.sort_values(by= ['fid', 'year'], inplace=True)
# ANPP.head(2)

# %%

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
# %%time
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %%
SF_west.rename(columns={"EW_meridia": "EW_meridian",
                        "p_valueSpe" : "p_valueSpearman",
                        "medians_di": "medians_diff_ANPP",
                        "medians__1" : "medians_diff_slope_ANPP",
                        "median_ANP" : "median_ANPP_change_as_perc",
                        "state_majo" : "state_majority_area"}, 
               inplace=True)

# %% [markdown]
# # Read Weather Data

# %%
filename = bio_reOrganized + "bps_weather.sav"
monthly_weather = pd.read_pickle(filename)
monthly_weather = monthly_weather["bps_weather"]
# change the order of columns!
monthly_weather.head(2)

# %%
print (f'{len(monthly_weather["fid"].unique())=}')
print (f'{len(ANPP["fid"].unique())=}')

# %%
print (len(set(monthly_weather["fid"].unique()).intersection(ANPP["fid"].unique())))
print (len(monthly_weather[monthly_weather["fid"].isin(list(ANPP["fid"].unique()))]["fid"].unique()))

# %%
FIDs_weather_ANPP_common = list(set(monthly_weather["fid"].unique()).intersection(ANPP["fid"].unique()))

# Lets pick the ones are on the west
print (len(FIDs_weather_ANPP_common))
FIDs_weather_ANPP_common = list(set(FIDs_weather_ANPP_common).intersection(SF_west["fid"].unique()))
print (len(FIDs_weather_ANPP_common))

# %% [markdown]
# ### Subset to common FIDs:

# %%
ANPP    = ANPP[ANPP["fid"].isin(FIDs_weather_ANPP_common)]
monthly_weather = monthly_weather[monthly_weather["fid"].isin(FIDs_weather_ANPP_common)]

SF_west = SF_west[SF_west["fid"].isin(FIDs_weather_ANPP_common)]

# %%
monthly_weather.head(2)

# %%
# # %%time
# unique_number_of_years = {}
# for a_fid in FIDs_weather_ANPP_common:
#     LL = str(len(monthly_weather[monthly_weather.fid == a_fid])) + "_months"    
#     if not (LL in unique_number_of_years.keys()):
#         unique_number_of_years[LL] = 1
#     else:
#         unique_number_of_years[LL] = unique_number_of_years[LL] + 1
# unique_number_of_years

# %%
528 / 12

# %%
# # %%time
# unique_number_of_years = {}
# for a_fid in ANPP.fid.unique():
#     LL = str(len(ANPP[ANPP.fid == a_fid])) + "_years"
#     if not (LL in unique_number_of_years.keys()):
#         unique_number_of_years[LL] = 1
#     else:
#         unique_number_of_years[LL] = unique_number_of_years[LL] + 1
# unique_number_of_years

# %%
ANPP.head(2)

# %%
monthly_weather.head(2)

# %% [markdown]
# # Compute annual 
# ```precipitation``` and ```temp``` for Sen's slope and Spearmans

# %%
cc = ["fid", "year", "precip_mm_month"]
annual_precip = monthly_weather[cc].groupby(["fid", "year"]).sum().reset_index()

cc = ["fid", "year", "avg_of_dailyAvgTemp_C"]
annual_temp = monthly_weather[cc].groupby(["fid", "year"]).mean().reset_index()

cc = ["fid", "year", "avg_of_dailyAvg_rel_hum"]
annual_rel_hum = monthly_weather[cc].groupby(["fid", "year"]).mean().reset_index()

# %%
annual_WA_ANPP = pd.merge(annual_precip, annual_temp, how="left", on=["fid", "year"])
annual_WA_ANPP = pd.merge(annual_WA_ANPP, annual_rel_hum, how="left", on=["fid", "year"])

annual_WA_ANPP = pd.merge(annual_WA_ANPP, ANPP[["fid", "year", "mean_lb_per_acr"]], 
                          how="left", on=["fid", "year"])

annual_WA_ANPP.rename(columns={"precip_mm_month": "precip_mm_yr", 
                               "avg_of_dailyAvgTemp_C": "avg_of_dailyAvgTemp_C_AvgOverMonths",
                               "avg_of_dailyAvg_rel_hum": "avg_of_dailyAvg_rel_hum_AvgOverMonths"}, 
                      inplace=True)
annual_WA_ANPP.head(2)

# %%
# some years are missing in ANPP as we know
print (annual_WA_ANPP.shape)
annual_WA_ANPP.dropna(subset=["mean_lb_per_acr"], inplace=True)

annual_WA_ANPP.sort_values(by= ['fid', 'year'], inplace=True)
annual_WA_ANPP.reset_index(drop=True, inplace=True)

print (annual_WA_ANPP.shape)

annual_WA_ANPP.head(2)

# %%
SF_west.head(2)

# %%
annual_WA_ANPP = pd.merge(annual_WA_ANPP, SF_west[["fid", "groupveg"]], on="fid", how="left")
annual_WA_ANPP.head(2)

# Lets just forget abuot Sparse, Riparian, Barren-Rock/Sand/Clay, and Conifer?
annual_WA_ANPP[["groupveg", "fid"]].drop_duplicates().groupby(["groupveg"]).count().reset_index()

# %%
groupveg = sorted(annual_WA_ANPP["groupveg"].unique())
groupveg

# %%
veg_colors = {"Barren-Rock/Sand/Clay" : "blue",
              "Conifer" : "green",
              "Grassland" : "red",
              "Hardwood" : "cyan",
              "Riparian" : "magenta",
              "Shrubland" : "yellow",
              "Sparse" : "black"}

for a_veg in  groupveg:
    SF_west.loc[SF_west['groupveg'] == a_veg, 'color'] = veg_colors[a_veg]

SF_west.head(2)

# %%

# %%
tick_legend_FontSize = 14
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5,    'legend.handlelength': 2}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : True})

# axes.set_title('Intersection of sigfinicant MK test and Spearman (increasing trend)');
sns.histplot(data=annual_WA_ANPP["mean_lb_per_acr"], ax=axes, bins=100, kde=True);
# axes.legend(["ANPP (mean lb/acr)"], loc='upper right');
axes.set_xlabel("ANPP (mean lb/acr)");

# %%

# %%
plt.rc("font", family="Palatino")
tick_legend_FontSize = 20
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5,    'legend.handlelength': 2}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(2, 2, figsize=(20, 5), sharey=True, sharex=True, 
                         gridspec_kw={"hspace": 0.25, "wspace": 0.01}, dpi=dpi_)
sns.set_style({'axes.grid' : True})

veg_type = groupveg[0]
df = annual_WA_ANPP[annual_WA_ANPP["groupveg"] == veg_type]
sns.histplot(data=df["mean_lb_per_acr"], ax=axes[0][0], bins=100, kde=True, color=veg_colors[veg_type]);
axes[0][0].legend([veg_type], loc='upper right');
################################################################################
veg_type = groupveg[1]
df = annual_WA_ANPP[annual_WA_ANPP["groupveg"] == veg_type]
sns.histplot(data=df["mean_lb_per_acr"], ax=axes[0][1], bins=100, kde=True, color=veg_colors[veg_type]);
axes[0][1].legend([veg_type], loc='upper right');
################################################################################
veg_type = groupveg[2]
df = annual_WA_ANPP[annual_WA_ANPP["groupveg"] == veg_type]
sns.histplot(data=df["mean_lb_per_acr"], ax=axes[1][0], bins=100, kde=True, color=veg_colors[veg_type]);
axes[1][0].legend([veg_type], loc='upper right');
################################################################################
veg_type = groupveg[3]
df = annual_WA_ANPP[annual_WA_ANPP["groupveg"] == veg_type]
sns.histplot(data=df["mean_lb_per_acr"], ax=axes[1][1], bins=100, kde=True, color=veg_colors[veg_type]);
axes[1][1].legend([veg_type], loc='upper right');
################################################################################

axes[0][0].set_xlabel(""); axes[0][1].set_xlabel("");
axes[1][0].set_xlabel("ANPP (mean lb/acr)"); axes[1][1].set_xlabel("ANPP (mean lb/acr)");

fig.suptitle('ANPP distribution', y=0.95, fontsize=18)
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.052, right=0.981, wspace=-0.2, hspace=0)
file_name = bio_plots + "vegType_ANPPDist.pdf"
plt.savefig(file_name)

# %%
print (annual_WA_ANPP[annual_WA_ANPP["fid"].isin([7627])]["groupveg"].unique())
print (annual_WA_ANPP[annual_WA_ANPP["fid"].isin([18778])]["groupveg"].unique())
print (annual_WA_ANPP[annual_WA_ANPP["fid"].isin([1])]["groupveg"].unique())

annual_WA_ANPP.describe()

# %%
tick_legend_FontSize = 6
params = {"legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * .71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .7,
          "ytick.labelsize": tick_legend_FontSize * .7,
          "axes.titlepad": 5,
          'legend.handlelength': 2}

plt.rcParams["xtick.bottom"] = False
plt.rcParams["ytick.left"] = False
plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["ytick.labelleft"] = False
plt.rcParams.update(params)

# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True, dpi=dpi_)
ax.set_xticks([]); ax.set_yticks([])
plt.title('rangeland polygons in Albers shapefile')

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_ = "Pastel1")
SF_west["geometry"].centroid.plot(ax=ax, c=SF_west['color'], markersize=0.2)

plt.rcParams['axes.linewidth'] = .051
plt.tight_layout()
# plt.legend(fontsize=3) # ax.axis('off')
# plt.show();
from matplotlib.lines import Line2D

labels = list(veg_colors.keys())
colors = list(veg_colors.values())
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
plt.legend(lines, labels, fontsize=4, frameon=False)

file_name = bio_plots + "Albers_SF_locs_vegType.png"
plt.savefig(file_name)

# %% [markdown]
# ## Compute Spearman for ANPP and precip

# %%
# # %%time
# need_cols = ["fid", "centroid"]
# precip_MK_df = SF_west[need_cols].copy()
# print (ANPP_MK_df.shape)

# precip_MK_df.drop_duplicates(inplace=True)
# precip_MK_df.reset_index(drop=True, inplace=True)
# print (precip_MK_df.shape)

# MK_test_cols = ["precip_Spearman", "precip_p_valSpearman"]
# precip_MK_df = pd.concat([precip_MK_df, pd.DataFrame(columns = MK_test_cols)])
# precip_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)
# # Why data type changed?!
# precip_MK_df["fid"] = precip_MK_df["fid"].astype(np.int64)
# precip_MK_df.drop(columns="centroid", inplace=True)
# precip_MK_df.head(2)


# # populate the dataframe with MK test result now
# for a_FID in precip_MK_df["fid"].unique():
#     ANPP_TS = annual_WA_ANPP.loc[annual_WA_ANPP["fid"]==a_FID, "mean_lb_per_acr"].values
#     precip_TS = annual_WA_ANPP.loc[annual_WA_ANPP["fid"]==a_FID, "precip_mm_yr"].values
#     Spearman, p_valueSpearman = stats.spearmanr(precip_TS, ANPP_TS)
#     L_ = [Spearman, p_valueSpearman]
#     precip_MK_df.loc[precip_MK_df["fid"]==a_FID, MK_test_cols] = L_

# precip_MK_df.head(2)

# %% [markdown]
# ## Compute Spearman for ANPP and Temp

# %%
# # %%time

# need_cols = ["fid", "centroid"]
# temp_MK_df = SF_west[need_cols].copy()
# print (ANPP_MK_df.shape)

# temp_MK_df.drop_duplicates(inplace=True)
# temp_MK_df.reset_index(drop=True, inplace=True)
# print (temp_MK_df.shape)

# MK_test_cols = ["temp_Spearman", "temp_p_valSpearman"]
# temp_MK_df = pd.concat([temp_MK_df, pd.DataFrame(columns = MK_test_cols)])
# temp_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)
# temp_MK_df.drop(columns="centroid", inplace=True)
# # Why data type changed?!
# temp_MK_df["fid"] = temp_MK_df["fid"].astype(np.int64)
# temp_MK_df.head(2)


# # populate the dataframe with MK test result now
# for a_FID in temp_MK_df["fid"].unique():
#     ANPP_TS = annual_WA_ANPP.loc[annual_WA_ANPP["fid"]==a_FID, "mean_lb_per_acr"].values
#     temp_TS = annual_WA_ANPP.loc[annual_WA_ANPP["fid"]==a_FID, "avg_of_dailyAvgTemp_C_AvgOverMonths"].values
#     Spearman, p_valueSpearman = stats.spearmanr(temp_TS, ANPP_TS)
#     L_ = [Spearman, p_valueSpearman]
#     temp_MK_df.loc[temp_MK_df["fid"]==a_FID, MK_test_cols] = L_

# temp_MK_df.head(2)

# %%
# temp_precip_spear = pd.merge(temp_MK_df[["fid", "temp_Spearman", "temp_p_valSpearman"]], 
#                              precip_MK_df, how="left", on=["fid"])

filename = bio_reOrganized + "temp_precip_spearman.sav"
# export_ = {"temp_precip_spear": temp_precip_spear, 
#            "source_code" : "Weather_EDA",
#            "Author": "HN",
#            "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# pickle.dump(export_, open(filename, 'wb'))


temp_precip_spear = pd.read_pickle(filename)
temp_precip_spear = temp_precip_spear["temp_precip_spear"]
temp_precip_spear.head(2)

# %%
temp   = temp_precip_spear[["fid", "temp_Spearman", "temp_p_valSpearman"]].copy()
precip = temp_precip_spear[["fid", "precip_Spearman", "precip_p_valSpearman"]].copy()

temp = temp[temp["temp_p_valSpearman"] < 0.05].copy()
precip = precip[precip["precip_p_valSpearman"] < 0.05].copy()

print (temp.shape)
print (precip.shape)

temp_precip_spear_sig_95 = pd.merge(temp, precip, how="outer", on=["fid"])

# temp_precip_spear_sig_95 = pd.merge(temp_precip_spear_sig_95, 
#                                      SF_west[["fid", "trend"]], how="left", on=["fid"])

temp_precip_spear_sig_95.head(2)

# %%

# %%
print (f"{temp_precip_spear.shape = }")
print (f"{temp_precip_spear_sig_95.shape = }")

# %%
temp   = temp_precip_spear[["fid", "temp_Spearman", "temp_p_valSpearman"]].copy()
precip = temp_precip_spear[["fid", "precip_Spearman", "precip_p_valSpearman"]].copy()

temp = temp[temp["temp_p_valSpearman"] < 0.1].copy()
precip = precip[precip["precip_p_valSpearman"] < 0.1].copy()

temp_precip_spear_sig_90 = pd.merge(temp, precip, how="outer", on=["fid"])
temp_precip_spear_sig_90.head(2)

# %%
print (f"{temp_precip_spear.shape = }")
print (f"{temp_precip_spear_sig_95.shape = }")
print (f"{temp_precip_spear_sig_90.shape = }")

# %%
temp_precip_spear_sig_95.head(2)

# %%
print (temp_precip_spear_sig_95.shape)
print (temp_precip_spear_sig_95.dropna().shape)

# %%
SF_west_Spearman_95 = pd.merge(SF_west, temp_precip_spear_sig_95, on="fid", how="left")

# %% [markdown]
# ### Some plots

# %%
# Parameters for font sizes
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * 0.71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.7,
          "ytick.labelsize": tick_legend_FontSize * 0.7,
          "axes.titlepad": 5,    'legend.handlelength': 2}
plt.rcParams.update(params)

# %%
min_color = min(SF_west_Spearman_95['temp_Spearman'].min(), SF_west_Spearman_95['precip_Spearman'].min())
max_color = max(SF_west_Spearman_95['temp_Spearman'].max(), SF_west_Spearman_95['precip_Spearman'].max())
norm_colorB = Normalize(vmin=min_color, vmax=max_color)

min_max = max(abs(min_color),abs(max_color))
norm_colorB = Normalize(vmin=-min_max, vmax=min_max)
####################################################################################
####################################################################################
####################################################################################
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
####################################################################### States
plot_SF(SF=visframe_mainLand_west, cmap_="Pastel1", ax_=ax1, col="EW_meridian")
plot_SF(SF=visframe_mainLand_west, cmap_="Pastel1", ax_=ax2, col="EW_meridian")
####################################################################################
p1 = SF_west_Spearman_95.plot(column='temp_Spearman', ax=ax1, cmap=cmap_RYG, norm=norm_colorB, legend=False)
ax1.set_title("temperature (Spearman's rank; 95% significant)")

p2 = SF_west_Spearman_95.plot(column='precip_Spearman', ax=ax2, cmap=cmap_RYG, norm=norm_colorB, legend=False)
ax2.set_title("precipitation (Spearman's rank; 95% significant)")
####################################################################################
# fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=0.1, hspace=0.01)
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-0.2, hspace=0)
fig.colorbar(p1.get_children()[1], ax=ax2, fraction=0.02, orientation='vertical', location="right")
# ax1.inset_axes([0.3, 0.07, 0.4, 0.04])
fig.suptitle('All locations (not just greening)', y=1.01)
plt.show();

# %%
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %% [markdown]
# ### Do Focus on greening locations only

# %%
SF_west_Spearman_95_green = SF_west_Spearman_95[SF_west_Spearman_95["trend"] == "increasing"].copy()
SF_west_Spearman_95_green.head(2)

# %%

# %%
min_color = min(SF_west_Spearman_95_green['temp_Spearman'].min(), 
                SF_west_Spearman_95_green['precip_Spearman'].min())
max_color = max(SF_west_Spearman_95_green['temp_Spearman'].max(), 
                SF_west_Spearman_95_green['precip_Spearman'].max())
norm_colorB = Normalize(vmin=min_color, vmax=max_color)

min_max = max(abs(min_color),abs(max_color))
norm_colorB = Normalize(vmin=-min_max, vmax=min_max)
####################################################################################
####################################################################################
####################################################################################
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
####################################################################### States
plot_SF(SF=visframe_mainLand_west, cmap_="Pastel1", ax_=ax1, col="EW_meridian")
plot_SF(SF=visframe_mainLand_west, cmap_="Pastel1", ax_=ax2, col="EW_meridian")
####################################################################################
p1 = SF_west_Spearman_95_green.plot(column='temp_Spearman', ax=ax1, cmap=cmap_RYG, norm=norm_colorB)
ax1.set_title("temperature (Spearman's rank; 95% significant)")

p2 = SF_west_Spearman_95_green.plot(column='precip_Spearman', ax=ax2, cmap=cmap_RYG, norm=norm_colorB)
ax2.set_title("precipitation (Spearman's rank; 95% significant)")
####################################################################################
cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical')
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-0.1, hspace=0)
fig.suptitle('Greening locations', y=1.01)
plt.show();

# %%
temp_precip_spear.head(2)

# %%

# %%
tick_legend_FontSize = 12
params = {"legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          'axes.grid' : False}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)


# %%
def lin_reg(df):
    X = df[["year", y_var]].copy()
    X.dropna(how="any", inplace=True)
    X = sm.add_constant(X)
    Y = X[y_var].astype(float)
    X = X.drop(y_var, axis=1)
    ks = sm.OLS(Y, X)
    ks_result = ks.fit()
    y_pred = ks_result.predict(X)
    reg_slope = int(ks_result.params["year"].round())
    return(reg_slope, y_pred)


# %%
## re-plot this from EDA_Sens_Plots.ipynb and add
## precip and temp spearmans to the plots

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2, ax3) = axes

# for ax in axes.flat: ax.grid(False)

# ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both"); 
# ax3.grid(axis="both", which="both")
y_var = "mean_lb_per_acr"
######
###### subplot 1
######
target_idx = SF_west["sens_slope"].max()
a_fid = SF_west.loc[SF_west["sens_slope"] == target_idx, "fid"].values[0]

df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)
ax1.plot(df.year, df[y_var], linewidth=3);

## regression line
reg_slope, y_pred = lin_reg(df)
ax1.plot(df["year"], y_pred, color="red", linewidth=3, label="regression fit");
ax1.legend(loc='best')

text_1 = "trend: {}\nSen's slope {}, reg. slope {}\n{}".format(trend_, slope_, reg_slope, state_)
text_2 = " (FID: {})\nPrecip. Spearman {}\nTemp. Spearman {}".format(a_fid, prec_sprear, temp_sprear)
text_ = text_1 + text_2
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/1.2)
ax1.text(2012, y_txt, text_, fontsize = 12);
# ax1.set_ylim(3000, 4500);
######
###### subplot 2
target_idx = SF_west["sens_slope"].min()
a_fid = SF_west.loc[SF_west["sens_slope"] == target_idx, "fid"].values[0]

df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)
ax2.plot(df.year, df[y_var], linewidth=3);

## regression line
reg_slope, y_pred = lin_reg(df)
ax2.plot(df["year"], y_pred, color="red", linewidth=3, label="regression fit");
ax2.legend(loc='lower left')

text_1 = "trend: {}\nSen's slope {}, reg. slope {}\n{}".format(trend_, slope_, reg_slope, state_)
text_2 = " (FID: {})\nPrecip. Spearman {}\nTemp. Spearman {}".format(a_fid, prec_sprear, temp_sprear)
text_ = text_1 + text_2
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/2)
ax2.text(2012, y_txt, text_, fontsize = 12);

######
###### subplot 3
######
a_fid = ANPP_MK_df.loc[ANPP_MK_df["trend"] == "no trend", "fid"].values[0]
df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)
ax3.plot(df.year, df[y_var], linewidth=3);

## regression line
reg_slope, y_pred = lin_reg(df)
ax3.plot(df["year"], y_pred, color="red", linewidth=3); 

text_1 = "trend: {}\nSen's slope {}, reg. slope {}\n{}".format(trend_, slope_, reg_slope, state_)
text_2 = " (FID: {})\nPrecip. Spearman {}\nTemp. Spearman {}".format(a_fid, prec_sprear, temp_sprear)
text_ = text_1 + text_2
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/3)
ax3.text(2012, y_txt, text_, fontsize = 12);

# plt.subplots_adjust(left=0.9, right=0.92, top=0.92, bottom=0.9)
ax2.set_ylabel("ANPP (mean lb/acr)")
ax1.set_title("three trend examples")
# plt.tight_layout();
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981)
file_name = bio_plots + "three_trends.pdf"
plt.savefig(file_name)

# %%
annual_WA_ANPP.head(2)

# %%
tick_legend_FontSize = 12
params = {"legend.fontsize": tick_legend_FontSize * .81,
          "axes.labelsize": tick_legend_FontSize * 1,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 2,
          "axes.titlesize": tick_legend_FontSize}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
annual_WA_ANPP['precip_mm_yr'] = annual_WA_ANPP['precip_mm_yr'] / 10
annual_WA_ANPP.rename(columns={"precip_mm_yr": "precip_cm_yr"}, inplace=True)
annual_WA_ANPP.head(2)

# %%

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2, ax3) = axes
    
y_var = "mean_lb_per_acr"
pre_title =  "{} (FID: {}), Spearmans': Precip. {}, Temp. {}"
######
###### subplot 1
######
target_idx = SF_west["sens_slope"].max()
a_fid = SF_west.loc[SF_west["sens_slope"] == target_idx, "fid"].values[0]

df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)
ax1.plot(df["year"], df["mean_lb_per_acr"], linewidth=3, c="dodgerblue", label="ANPP");
ax1.plot(df["year"], df["precip_cm_yr"]*20, linewidth=3, c="red", label= "precip (cm/yr) \u00D7 20");
var = "avg_of_dailyAvgTemp_C_AvgOverMonths"
ax1.plot(df["year"], df[var]*100, linewidth=3, c="k", label= "Temp \u00D7 100");
ax1.legend(loc='best')

title_ = pre_title.format(state_, a_fid, prec_sprear, temp_sprear)
ax1.set_title(title_);
######
###### subplot 2
######
target_idx = SF_west["sens_slope"].min()
a_fid = SF_west.loc[SF_west["sens_slope"] == target_idx, "fid"].values[0]

df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)
ax2.plot(df["year"], df["mean_lb_per_acr"], linewidth=3, c="dodgerblue", label="ANPP");
ax2.plot(df["year"], df["precip_cm_yr"]*20, linewidth=3, c="red", label= "precip (cm/yr) \u00D7 20");
var = "avg_of_dailyAvgTemp_C_AvgOverMonths"
ax2.plot(df["year"], df[var]*100, linewidth=3, c="k", label= "Temp \u00D7 100");
ax2.legend(loc='best')
title_ = pre_title.format(state_, a_fid, prec_sprear, temp_sprear)
ax2.set_title(title_);
######
###### subplot 3
######
a_fid = ANPP_MK_df.loc[ANPP_MK_df["trend"] == "no trend", "fid"].values[0]
df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)
ax3.plot(df["year"], df["mean_lb_per_acr"], linewidth=3, c="dodgerblue", label="ANPP");
ax3.plot(df["year"], df["precip_cm_yr"]*20, linewidth=3, c="red", label= "precip (cm/yr) \u00D7 20");
var = "avg_of_dailyAvgTemp_C_AvgOverMonths"
ax3.plot(df["year"], df[var]*100, linewidth=3, c="k", label= "Temp \u00D7 100");
ax3.legend(loc='best')

title_ = pre_title.format(state_, a_fid, prec_sprear, temp_sprear)
ax3.set_title(title_);

fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981)
file_name = bio_plots + "time_ANPP_Temp_Prec_three_examples.pdf"
plt.savefig(file_name)

# %% [markdown]
# ### Temp/Precipitation V ANPP plots

# %%
LW_ = 3

# %%
fig, axes = plt.subplots(3, 2, figsize=(10, 6), sharey=False, 
                        gridspec_kw={"hspace": 0.5, "wspace": 0.03}, dpi=dpi_)
(ax1, ax2), (ax3, ax4), (ax5, ax6) = axes

y_var = "mean_lb_per_acr"
pre_title =  "{} (FID: {}), Spearmans': P: {}, T: {}"
######
###### subplot 1
######
target_idx = SF_west["sens_slope"].max()
a_fid = SF_west.loc[SF_west["sens_slope"] == target_idx, "fid"].values[0]

df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)

var = "avg_of_dailyAvgTemp_C_AvgOverMonths"
df.sort_values(by=[var], inplace=True)
ax1.plot(df[var], df[y_var], linewidth=LW_, c="r");

var = "precip_cm_yr"
df.sort_values(by=[var], inplace=True)
ax2.plot(df[var], df[y_var], linewidth=LW_, c="dodgerblue");

title_ = pre_title.format(state_, a_fid, prec_sprear, temp_sprear)
ax1.set_title(title_); ax2.set_title(title_);
#####################################################################################
###### subplot 2
target_idx = SF_west["sens_slope"].min()
a_fid = SF_west.loc[SF_west["sens_slope"] == target_idx, "fid"].values[0]

df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)

var = "avg_of_dailyAvgTemp_C_AvgOverMonths"
df.sort_values(by=[var], inplace=True)
ax3.plot(df[var], df[y_var], linewidth=LW_, c="r");

var = "precip_cm_yr"
df.sort_values(by=[var], inplace=True)
ax4.plot(df[var], df[y_var], linewidth=LW_, c="dodgerblue");

title_ = pre_title.format(state_, a_fid, prec_sprear, temp_sprear)
ax3.set_title(title_); ax4.set_title(title_);
#####################################################################################
###### subplot 3
a_fid = ANPP_MK_df.loc[ANPP_MK_df["trend"] == "no trend", "fid"].values[0]

df = annual_WA_ANPP[annual_WA_ANPP.fid == a_fid]
trend_ = SF_west.loc[SF_west.fid == a_fid, "trend"].values[0]
slope_ = int(SF_west.loc[SF_west.fid == a_fid, "sens_slope"].values[0])
state_ = SF_west.loc[SF_west.fid == a_fid, "state_majority_area"].values[0]
prec_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["precip_Spearman"].values[0], 2)
temp_sprear = round(temp_precip_spear[temp_precip_spear["fid"] == a_fid]["temp_Spearman"].values[0], 2)

var = "avg_of_dailyAvgTemp_C_AvgOverMonths"
df.sort_values(by=[var], inplace=True)
ax5.plot(df[var], df[y_var], linewidth=LW_, c="r");
# ax5.scatter(df[var], df[y_var], marker='s', facecolors='none', edgecolors='r', s=25);

var = "precip_cm_yr"
df.sort_values(by=[var], inplace=True)
ax6.plot(df[var], df[y_var], linewidth=LW_, c="dodgerblue");
# ax6.scatter(df[var], df[y_var], marker='s', facecolors='none', edgecolors='dodgerblue', s=25);
# ax6.scatter(df[var], df[y_var], marker='s', c='dodgerblue', s=25);

title_ = pre_title.format(state_, a_fid, prec_sprear, temp_sprear)
ax5.set_title(title_); ax6.set_title(title_);
#####################################################################################
ax2.set_yticks([]), ax4.set_yticks([]), ax6.set_yticks([])
ax3.set_ylabel("ANPP (mean lb/acr)");
ax5.set_xlabel("Temp (°C)");
ax6.set_xlabel("Precip (cm/yr)");

fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981)
file_name = bio_plots + "ANPP_versus_T_P_3_examples.pdf"
plt.savefig(file_name)

# %%

# %% [markdown]
# # Regression
#
# origin of spreg.OLS_ in ```04_02_2024_NonNormalModelsInterpret.ipynb```.

# %%
from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda
import geopandas, contextily

from scipy.stats import ttest_ind

# %%
annual_WA_ANPP.head(2)

# %%

# %%
annual_WA_ANPP["temp_X_precip"] = annual_WA_ANPP["precip_cm_yr"] * \
                                  annual_WA_ANPP["avg_of_dailyAvgTemp_C_AvgOverMonths"]
    
annual_WA_ANPP.head(2)

# %%
print (len(annual_WA_ANPP.fid.unique()))

# %%
print (SF_west.shape)
SF_west.head(2)

# %%

# %% [markdown]
# # Drop bad Vegs
# ```Sparse```, ```Riparian```, ```Barren-Rock/Sand/Clay```, and ```Conifer```?

# %%
print (f'{round(annual_WA_ANPP["precip_cm_yr"].min(), 2) = }')
print (f'{round(annual_WA_ANPP["precip_cm_yr"].max(), 2) = }')

print (f'{round(annual_WA_ANPP["avg_of_dailyAvgTemp_C_AvgOverMonths"].min()) = }')
print (f'{round(annual_WA_ANPP["avg_of_dailyAvgTemp_C_AvgOverMonths"].max()) = }')

# %%
good_vegs = ['Conifer', 'Grassland', 'Hardwood', 'Shrubland']
annual_WA_ANPP = annual_WA_ANPP[annual_WA_ANPP["groupveg"].isin(good_vegs)].copy()
annual_WA_ANPP.reset_index(drop=True, inplace=True)
groupveg = sorted(annual_WA_ANPP["groupveg"].unique())
groupveg

# %%
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.4,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "axes.titlepad": 10}
plt.rcParams.update(params)
cols_ = ["mean_lb_per_acr", "precip_cm_yr", 
         "avg_of_dailyAvgTemp_C_AvgOverMonths", "avg_of_dailyAvg_rel_hum_AvgOverMonths"]

A = annual_WA_ANPP.copy()
A.rename(columns={"mean_lb_per_acr": "NPP",
                  "precip_cm_yr": "precip",
                  "avg_of_dailyAvgTemp_C_AvgOverMonths": "Temp",
                  "avg_of_dailyAvg_rel_hum_AvgOverMonths": "RH",
                 }, inplace=True)
cols_ = ["NPP", "precip", "Temp", "RH"]

my_scatter = sns.pairplot(A[cols_], size=1.5, diag_kind="None", plot_kws={"s": 4}, corner=True)

# %%

# %%
veg_ = groupveg[0]
my_scatter = sns.pairplot(A[A["groupveg"] == veg_][cols_],  size=1.5, 
                          diag_kind="None", plot_kws={"s": 4}, corner=True)
my_scatter.fig.suptitle(veg_, y=1.08);

# %%
veg_ = groupveg[1]
my_scatter = sns.pairplot(A[A["groupveg"] == veg_][cols_],  size=1.5, 
                          diag_kind="None", plot_kws={"s": 4}, corner=True)
my_scatter.fig.suptitle(veg_, y=1.08);

# %%
veg_ = groupveg[2]
my_scatter = sns.pairplot(A[A["groupveg"] == veg_][cols_],  size=1.5, 
                          diag_kind="None", plot_kws={"s": 4}, corner=True)
my_scatter.fig.suptitle(veg_, y=1.08);

# %%
veg_ = groupveg[3]
my_scatter = sns.pairplot(A[A["groupveg"] == veg_][cols_],  size=1.5, 
                          diag_kind="None", plot_kws={"s": 4}, corner=True)
my_scatter.fig.suptitle(veg_, y=1.08);

# %%

# %%

# %%
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.4,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "axes.titlepad": 10}
plt.rcParams.update(params)


my_scatter = sns.pairplot(A[A["groupveg"] == groupveg[0]], size=2, diag_kind="None", plot_kws={"s": 4})

# %%

# %%

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5 = spreg.OLS_Regimes(y = annual_WA_ANPP[depen_var].values,
                       x = annual_WA_ANPP[indp_vars].values, 

                       # Variable specifying neighborhood membership
                       regimes = annual_WA_ANPP["groupveg"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var, # Dependent variable name
                       name_x=indp_vars)

print (f"{m5.r2.round(2) = }")

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

# %%
groupveg

# %%
# West regime
## Extract variables for the west side 
# Barren_m = [i for i in m5_results.index if "Barren" in i]
Conifer_m = [i for i in m5_results.index if "Conifer" in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m = [i for i in m5_results.index if "Hardwood" in i]
# Riparian_m = [i for i in m5_results.index if "Riparian" in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]
# Sparse_m = [i for i in m5_results.index if "Sparse" in i]

## Subset results to Barren
# veg_ = "Barren"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Barren = m5_results.loc[Barren_m, :].rename(lambda i: i.replace(rep_, ""))
# Barren.columns = pd.MultiIndex.from_product([[veg_], Barren.columns])

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Riparian
# veg_ = "Riparian"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Riparian = m5_results.loc[Riparian_m, :].rename(lambda i: i.replace(rep_, ""))
# Riparian.columns = pd.MultiIndex.from_product([[veg_], Riparian.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

## Subset results to Sparse
# veg_ = "Sparse"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Sparse = m5_results.loc[Sparse_m, :].rename(lambda i: i.replace(rep_, ""))
# Sparse.columns = pd.MultiIndex.from_product([[veg_], Sparse.columns])

# Concat both models
# table_ = pd.concat([Barren, Conifer, Grassland, Hardwood, Riparian, Shrubland, Sparse], axis=1).round(5)
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %% [markdown]
# # Split and normalize

# %%
annual_WA_ANPP.head(2)

# %%
# reorder the dataframe
col_order = ['fid', 'year', 'mean_lb_per_acr',
             "groupveg",
             'precip_cm_yr',
             'avg_of_dailyAvgTemp_C_AvgOverMonths',
             'avg_of_dailyAvg_rel_hum_AvgOverMonths',
             'temp_X_precip',
             ]
annual_WA_ANPP = annual_WA_ANPP[col_order]
annual_WA_ANPP.head(2)

# %%

# %%
depen_var = "mean_lb_per_acr"
indp_vars = list(annual_WA_ANPP.columns[3:])
numeric_indp_vars = list(annual_WA_ANPP.columns[4:])

y_df = annual_WA_ANPP[depen_var].copy()
indp_df = annual_WA_ANPP[indp_vars].copy()

# %%

# %%
X_train, X_test, y_train, y_test = train_test_split(indp_df, y_df, test_size=0.3, random_state=42)
X_train.head(2)

# %%
train_idx = list(X_train.index)
test_idx = list(X_test.index)

# %%
# standard_indp = preprocessing.scale(all_df[explain_vars_herb]) # this is biased
means = X_train[numeric_indp_vars].mean()
stds = X_train[numeric_indp_vars].std(ddof=1)

X_train_normal = X_train.copy()
X_test_normal = X_test.copy()

X_train_normal[numeric_indp_vars] = (X_train_normal[numeric_indp_vars] - means) / stds
X_test_normal[numeric_indp_vars]  = (X_test_normal[numeric_indp_vars]  - means) / stds
X_train_normal.head(2)

# %% [markdown]
# # Model normalized data

# %%
annual_WA_ANPP.head(2)

# %%
SF_west.head(2)

# %%
tick_legend_FontSize = 12
params = {"legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          'axes.grid' : False}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %% [markdown]
#  - ToDo
#     - train a model with a given veg and check if spreg is doing what you think it is doing.
#        - Done. except SEs change!
#
#     - Look at R2 for each model separately. what does the general R2 mean that spreg spits out?
#
#     - Do modeling with interaction terms
#
#     - plot residual plots
#
#     - try model with log(y).

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

# pick a veg indices
idx_ = X_train_normal[X_train_normal["groupveg"] == "Grassland"].index
X_tr_ = X_train_normal.loc[idx_].copy()
y_tr_ = y_train.loc[idx_].copy()

m5_solo = spreg.OLS(y = y_tr_.values, x = X_tr_[indp_vars].values, 
                    name_y=depen_var, name_x=indp_vars)

print (f"{m5_solo.r2.round(2) = }")

m5_solo_results = pd.DataFrame({"Coeff.": m5_solo.betas.flatten(), 
                                "Std. Error": m5_solo.std_err.flatten(), 
                                "P-Value": [i[1] for i in m5_solo.t_stat]
                               }, index=m5_solo.name_x)

# %%
m5_solo_results

# %%
# m5_solo.predy

# %%
X = X_tr_.copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = y_train.loc[idx_].copy().astype(float)
ks = sm.OLS(Y, X[["const", "precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]])
ks_result = ks.fit()
y_pred = ks_result.predict(X[["const", "precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]])
ks_result.params

# %% [markdown]
# # Model based on Temp

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5_T_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                      regimes = X_train_normal["groupveg"].tolist(),
                                      constant_regi="many", regime_err_sep=False,
                                      name_y=depen_var, name_x=indp_vars)

print (f"{m5_T_normal.r2.round(2) = }")

m5_T_normal_results = pd.DataFrame({# Pull out regression coefficients and
                                          "Coeff.": m5_T_normal.betas.flatten(), 
                                          # Pull out and flatten standard errors
                                          "Std. Error": m5_T_normal.std_err.flatten(), 
                                          # Pull out P-values from t-stat object
                                          "P-Value": [i[1] for i in m5_T_normal.t_stat], 
                                           }, index=m5_T_normal.name_x)

## Extract variables for each veg type
# Barren_m    = [i for i in m5_T_normal_results.index if "Barren"    in i]
Conifer_m   = [i for i in m5_T_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_T_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_T_normal_results.index if "Hardwood"  in i]
# Riparian_m  = [i for i in m5_T_normal_results.index if "Riparian"  in i]
Shrubland_m = [i for i in m5_T_normal_results.index if "Shrubland" in i]
# Sparse_m    = [i for i in m5_T_normal_results.index if "Sparse"    in i]

## Subset results to Barren
# veg_ = "Barren"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Barren = m5_T_normal_results.loc[Barren_m, :].rename(lambda i: i.replace(rep_, ""))
# Barren.columns = pd.MultiIndex.from_product([[veg_], Barren.columns])

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_T_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_T_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_T_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Riparian
# veg_ = "Riparian"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Riparian = m5_T_normal_results.loc[Riparian_m, :].rename(lambda i: i.replace(rep_, ""))
# Riparian.columns = pd.MultiIndex.from_product([[veg_], Riparian.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_T_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

## Subset results to Sparse
# veg_ = "Sparse"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Sparse = m5_T_normal_results.loc[Sparse_m, :].rename(lambda i: i.replace(rep_, ""))
# Sparse.columns = pd.MultiIndex.from_product([[veg_], Sparse.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_T_normal.predy, m5_T_normal.u, c="dodgerblue", s=2);

title_ = "model baed on temperature"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %%

# %%

# %% [markdown]
# # Model based on Precip

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr"]

m5_P_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                regimes = X_train_normal["groupveg"].tolist(),
                                constant_regi="many", regime_err_sep=False,
                                name_y=depen_var, name_x=indp_vars)
print (f"{m5_P_normal.r2.round(2) = }")

m5_P_normal_results = pd.DataFrame({"Coeff.": m5_P_normal.betas.flatten(), 
                                    "Std. Error": m5_P_normal.std_err.flatten(), 
                                    "P-Value": [i[1] for i in m5_P_normal.t_stat], 
                                    }, index=m5_P_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_P_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_P_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_P_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_P_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_P_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_P_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_P_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_P_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_P_normal.predy, m5_P_normal.u, c="dodgerblue", s=2);

title_ = "model baed on precipitation"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Model by Temp and Precip

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5_TP_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                 regimes = X_train_normal["groupveg"].tolist(),
                                 constant_regi="many", regime_err_sep=False,
                                 name_y=depen_var, name_x=indp_vars)

print (f"{m5_TP_normal.r2.round(2) = }")

m5_TP_normal_results = pd.DataFrame({"Coeff.": m5_TP_normal.betas.flatten(), 
                                     "Std. Error": m5_TP_normal.std_err.flatten(), 
                                     "P-Value": [i[1] for i in m5_TP_normal.t_stat],
                                    }, index=m5_TP_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TP_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TP_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TP_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TP_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TP_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TP_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TP_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TP_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TP_normal.predy, m5_TP_normal.u, c="dodgerblue", s=2);

title_ = "model baed on temp. and precipitation"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Model with interaction terms

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths", "temp_X_precip"]

m5_TPinter_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                      regimes = X_train_normal["groupveg"].tolist(),
                                      constant_regi="many", regime_err_sep=False,
                                      name_y=depen_var, name_x=indp_vars)
print (f"{m5_TPinter_normal.r2.round(2) = }")

m5_TPinter_normal_results = pd.DataFrame({"Coeff.": m5_TPinter_normal.betas.flatten(), 
                                          "Std. Error": m5_TPinter_normal.std_err.flatten(), 
                                          "P-Value": [i[1] for i in m5_TPinter_normal.t_stat]}, 
                                         index=m5_TPinter_normal.name_x)
## Extract variables for each veg type
Conifer_m   = [i for i in m5_TPinter_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TPinter_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TPinter_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TPinter_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TPinter_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TPinter_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TPinter_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TPinter_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TPinter_normal.predy, m5_TPinter_normal.u, c="dodgerblue", s=2);

title_ = "model baed on temp. and precipitation and their interaction"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Temp, precipitation, humidity

# %%
X_train_normal.head(2)

# %%

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths", "avg_of_dailyAvg_rel_hum_AvgOverMonths"]

m5_TPH_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                 regimes = X_train_normal["groupveg"].tolist(),
                                 constant_regi="many", regime_err_sep=False,
                                 name_y=depen_var, name_x=indp_vars)

print (f"{m5_TPH_normal.r2.round(2) = }")

m5_TPH_normal_results = pd.DataFrame({"Coeff.": m5_TPH_normal.betas.flatten(), 
                                      "Std. Error": m5_TPH_normal.std_err.flatten(), 
                                      "P-Value": [i[1] for i in m5_TPH_normal.t_stat]}, 
                                     index=m5_TPH_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TPH_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TPH_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TPH_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TPH_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TPH_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TPH_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TPH_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TPH_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp", 
                       "avg_of_dailyAvg_rel_hum_AvgOverMonths" : "RH"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TPH_normal.predy, m5_TPH_normal.u, c="dodgerblue", s=2);

title_ = "model baed on temp. and precip. and RH"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Add square terms

# %%
print (X_train_normal.shape)
X_train_normal.head(2)

# %%
X_train_normal["temp_sq"] = X_train_normal["avg_of_dailyAvgTemp_C_AvgOverMonths"] ** 2
X_train_normal["precip_sq"] = X_train_normal["precip_cm_yr"] ** 2
X_train_normal.head(2)

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths", "temp_sq", "precip_sq", "temp_X_precip"]

m5_TP_sq_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                  regimes = X_train_normal["groupveg"].tolist(),
                                  constant_regi="many", regime_err_sep=False,
                                  name_y=depen_var, name_x=indp_vars)

print (f"{m5_TP_sq_normal.r2.round(2) = }")

m5_TP_sq_normal_results = pd.DataFrame({"Coeff.": m5_TP_sq_normal.betas.flatten(), 
                                      "Std. Error": m5_TP_sq_normal.std_err.flatten(), 
                                      "P-Value": [i[1] for i in m5_TP_sq_normal.t_stat]}, 
                                     index=m5_TP_sq_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TP_sq_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TP_sq_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TP_sq_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TP_sq_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TP_sq_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TP_sq_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TP_sq_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TP_sq_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp", 
                       "avg_of_dailyAvg_rel_hum_AvgOverMonths" : "RH"}, inplace=True)
table_

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TP_sq_normal.predy, m5_TP_sq_normal.u, c="dodgerblue", s=2);

title_ = f"model baed on $T$ and $P$ and $T^2$, $P^2$, and $T \u00D7 P$ "
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %% [markdown]
# # log of Y based on Temp, Precip 

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5_TP_normal_logy = spreg.OLS_Regimes(y = np.log10(y_train.values), x = X_train_normal[indp_vars].values, 
                                 regimes = X_train_normal["groupveg"].tolist(),
                                 constant_regi="many", regime_err_sep=False,
                                 name_y=depen_var, name_x=indp_vars)

print (f"{m5_TP_normal_logy.r2.round(2) = }")

m5_TP_normal_logy_results = pd.DataFrame({"Coeff.": m5_TP_normal_logy.betas.flatten(), 
                                     "Std. Error": m5_TP_normal_logy.std_err.flatten(), 
                                     "P-Value": [i[1] for i in m5_TP_normal_logy.t_stat],
                                    }, index=m5_TP_normal_logy.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TP_normal_logy_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TP_normal_logy_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TP_normal_logy_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TP_normal_logy_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TP_normal_logy_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TP_normal_logy_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TP_normal_logy_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TP_normal_logy_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TP_normal_logy.predy, m5_TP_normal_logy.u, c="dodgerblue", s=2);

title_ = f"$log(y) = f(T, P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
X_train_normal.head(2)

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(X_train["precip_cm_yr"], X_train["avg_of_dailyAvgTemp_C_AvgOverMonths"], 
             c="dodgerblue", s=2);

# title_ = f"$log(y) = f(T, P)$"
# axes.set_title(title_);
axes.set_xlabel("precip"); axes.set_ylabel("temp.");

# %%

# %%
