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


# %%
def plot_SF(SF, ax_, cmap_ = "Pastel1", col="EW_meridian"):
    SF.plot(column=col, ax=ax_, alpha=1, cmap=cmap_, edgecolor='k', legend=False, linewidth=0.1)
    
dpi_, map_dpi_=300, 900
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])


dpi_ = 300
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

# %%

# %%
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir_v11 = rangeland_bio_data + "Min_Data_v1.1/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
bio_reOrganized_temp = rangeland_bio_data + "temp_reOrganized/"

bio_plots = rangeland_bio_base + "plots/vegAreaChange/"
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
# #### Read shapefile

# %%
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)
Albers_SF.head(2)

# %%
Albers_SF = pd.merge(Albers_SF, state_fips[["EW_meridian", "state_full"]], 
                     how="left", left_on="state_majority_area", right_on="state_full")

Albers_SF.drop(columns=["state_full"], inplace=True)
Albers_SF.head(2)

# %%
print (Albers_SF.shape)
Albers_SF = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF.reset_index(drop=True, inplace=True)
print (Albers_SF.shape)

# %%
filename = bio_reOrganized + "rangeland_rap.sav"

rangeland_rap = pd.read_pickle(filename)
rangeland_rap = rangeland_rap["rangeland_rap"]
print (len(rangeland_rap['fid'].unique()))
rangeland_rap.head(2)

# %% [markdown]
# ### Check how many years data is there per FID

# %%
# %%time
unique_number_of_years = {}

for a_fid in rangeland_rap.fid.unique():
    LL = str(len(rangeland_rap[rangeland_rap.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %% [markdown]
# ### Check the growth of tree area between first and last decades

# %%
rangeland_rap.head(2)

# %%
median_diff = rangeland_rap[["fid"]].copy()
median_diff["first_decade_median_tree"] = -666.0
median_diff["last_decade_median_tree"]  = -666.0

print (median_diff.shape)

median_diff.drop_duplicates(inplace=True)
median_diff.reset_index(drop=True, inplace=True)

print (median_diff.shape)
median_diff.head(3)

# %%
# %%time
# Find median of first decade and last decade of tree

for a_FID in median_diff["fid"].unique():
    curr_df = rangeland_rap[rangeland_rap["fid"] == a_FID]
    
    min_year = curr_df["year"].min()
    max_year = curr_df["year"].max()
    
    first_decade = curr_df[curr_df["year"] < min_year + 10]
    last_decade  = curr_df[curr_df["year"] > max_year - 10]
    
    median_diff.loc[median_diff["fid"] == a_FID, "first_decade_median_tree"] = \
                                                first_decade['tree'].median()

    median_diff.loc[median_diff["fid"] == a_FID, "last_decade_median_tree"] = \
                                                    last_decade['tree'].median()

# %%
### Hard code. 38 yars worth of data there is
median_diff["medians_diff_tree"] = median_diff["last_decade_median_tree"] - \
                                      median_diff["first_decade_median_tree"]

median_diff["medians_diff_slope_tree"] = median_diff["medians_diff_tree"] / 38
median_diff.head(2)

# %%
median_diff["median_tree_change_as_perc"] = (100 * median_diff["medians_diff_tree"]) / \
                                                  median_diff["first_decade_median_tree"]
median_diff.head(2)

# %%
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5, 
          'legend.handlelength': 2,
          'axes.linewidth' : .51}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(1, 1, figsize=(4, 1.5), sharey=False, sharex=False, dpi=dpi_)
# sns.set_style({'axes.grid' : False})
# axes.grid(axis="y", which="both");
sns.histplot(data=median_diff["median_tree_change_as_perc"], ax=axes, bins=500, kde=True); # height=5

# %%
print (median_diff["median_tree_change_as_perc"].max())
print (median_diff["first_decade_median_tree"].min())

# %%
an_fid = median_diff.loc[median_diff["first_decade_median_tree"].idxmin(), "fid"]
rangeland_rap[rangeland_rap["fid"] == an_fid].head(5)

# %% [markdown]
# ### Lets subset to finite change!

# %%
median_diff.head(2)

# %%
sorted(median_diff["median_tree_change_as_perc"].unique())[-10:]

# %%
print (median_diff.shape)
median_diff_finite = median_diff[np.isfinite(median_diff).all(axis=1)]
print (median_diff_finite.shape)

# %%
median_diff_finite[median_diff_finite["median_tree_change_as_perc"] >= 2000].shape

# %%
median_diff_finite[median_diff_finite["median_tree_change_as_perc"] >= 250].shape

# %%
thresh = 400
median_diff_leThresh = median_diff_finite[median_diff_finite["median_tree_change_as_perc"] < thresh].copy()
########################################################################################################

fig, axes = plt.subplots(1, 1, figsize=(4, 1.5), sharey=False, sharex=False, dpi=dpi_)
# sns.set_style({'axes.grid' : False})
# axes.grid(axis="y", which="both");

var_ = "median_tree_change_as_perc"
sns.histplot(data=median_diff_leThresh[var_], ax=axes, bins=100, kde=True); # height=5

axes.set_title(f"$\Delta$tree area from 1$^s$$^t$ decade to last (as %) capped at {thresh}");
axes.set_xlabel(var_.replace("_", " "));

file_name = bio_plots + "tree_area_change_capped_at" + str(thresh) + ".pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches="tight")

# %%

# %%
from shapely.geometry import Polygon
gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%
median_diff.head(2)

# %%
Albers_SF.head(2)

# %%
Albers_SF = pd.merge(Albers_SF, median_diff_finite[["fid", "median_tree_change_as_perc"]], how="left", on="fid")

# %%
Albers_SF_finite = Albers_SF[Albers_SF["fid"].isin(list(median_diff_finite["fid"].unique()))]

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
Albers_SF_finite.median_tree_change_as_perc

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

var_ = "median_tree_change_as_perc"
min_max = max(np.abs(Albers_SF_finite[var_].min()), np.abs(Albers_SF_finite[var_].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))

cent_plt = Albers_SF_finite.plot(column=var_, ax=ax, legend=False, cmap = cm.get_cmap('RdYlGn'), norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width 
# of the bar
cax = ax.inset_axes([0.08, 0.18, 0.45, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(r"median change (as %)", labelpad=1)
plt.title("tree area change (as % decade median)")

# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "tree_area_change_as_percMap.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])


var_ = "median_tree_change_as_perc"
thresh = 400
df = Albers_SF_finite.copy()
df = df[df[var_] < thresh].copy()

min_max = max(np.abs(df[var_].min()), np.abs(df[var_].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))

cent_plt = df.plot(column=var_, ax=ax, legend=False, cmap = cm.get_cmap('RdYlGn'), norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width 
# of the bar
cax = ax.inset_axes([0.08, 0.18, 0.45, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(f"$\Delta$median tree area (as %)", labelpad=1)
plt.title(f"$\Delta$tree area (as % decade median) capped at {thresh}")

# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "tree_area_change_as_percMap_capped" + str(thresh) +  ".png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(df, cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])


var_ = "median_tree_change_as_perc"
thresh = 400
df = Albers_SF_finite.copy()
df = df[df[var_] > thresh].copy()

min_max = max(np.abs(df[var_].min()), np.abs(df[var_].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))

cent_plt = df.plot(column=var_, ax=ax, legend=False, cmap = cm.get_cmap('RdYlGn'), norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width 
# of the bar
cax = ax.inset_axes([0.08, 0.18, 0.45, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, 
                     cmap=cm.get_cmap('RdYlGn'), norm=norm1, cax=cax)
cbar1.set_label(f"$\Delta$median tree area (as %)", labelpad=1)
plt.title(f"$\Delta$tree area (as % decade median) more than {thresh}")

# plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "tree_area_change_as_percMap_moreThan" + str(thresh) +  ".png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(df, cent_plt, cax, cbar1, norm1, min_max)

# %% [markdown]
# ### Time-series of changes in some FIDs

# %%
rangeland_rap.head(2)

# %%
Albers_SF_finite.head(2)

# %%
A = Albers_SF_finite[Albers_SF_finite[var_] <= 400].copy()
fid_max = A.loc[A[var_].idxmax(), "fid"]
fid_min = A.loc[A[var_].idxmin(), "fid"]
print (fid_min)
print (fid_max)

# %%
fid_max

# %%
tick_legend_FontSize = 15
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1,
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
vegTypes = ["annual_forb_grass", "bare_ground", "litter", "perennial_forb_grass", "shrub", "tree"]

veg_abbr = {"annual_forb_grass" : "AFG",
            "bare_ground" : "BGR",
            "litter" : "LTR",
            "perennial_forb_grass" : "PFG",
            "shrub" : "SHR",
            "tree": "TRE"}


# %%
y_var = "tree"
a_veg = y_var
lw_=2

# %%
veg_abbr

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                        gridspec_kw={"hspace": .5, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2) = axes
ax1.grid(which='major', alpha=0.5, axis="x")
ax2.grid(which='major', alpha=0.5, axis="x")

######## ax1
a_fid = fid_max
df = rangeland_rap[rangeland_rap["fid"] == a_fid].copy()
# for a_veg in vegTypes:
#     ax1.plot(df.year, df[a_veg], lw=lw_, label=veg_abbr[a_veg]);
    
ax1.plot(df.year, df[y_var], lw=lw_, label=veg_abbr[a_veg]);
ax1.set_title(f"FID ({a_fid}) w. maximum tree area change")
######## ax2
a_fid = fid_min
df = rangeland_rap[rangeland_rap["fid"] == a_fid].copy()
# for a_veg in vegTypes:
#     ax2.plot(df.year, df[a_veg], lw=lw_, label=veg_abbr[a_veg]);
ax2.plot(df.year, df[y_var], lw=lw_, label=veg_abbr[a_veg]);
ax2.set_title(f"FID ({a_fid}) w. minimum tree area change")
####################################################
ax1.legend(loc="best"); ax2.legend(loc="best");

# ax2.set_xlabel("year")
fig.supylabel('coverage (%)', fontsize=16, x=0.05);

# %%

# %%

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                        gridspec_kw={"hspace": 0.2, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2) = axes
ax1.grid(which='major', alpha=0.5, axis="x")
ax2.grid(which='major', alpha=0.5, axis="x")

######## ax1
a_fid = fid_max
df = rangeland_rap[rangeland_rap["fid"] == a_fid].copy()
for a_veg in vegTypes:
    ax1.plot(df.year, df[a_veg], lw=lw_, label=veg_abbr[a_veg]);
ax1.set_title(f"FID ({a_fid}) w. maximum tree area change")
######## ax2
a_fid = fid_min
df = rangeland_rap[rangeland_rap["fid"] == a_fid].copy()
for a_veg in vegTypes:
    ax2.plot(df.year, df[a_veg], lw=lw_, label=veg_abbr[a_veg]);
ax2.set_title(f"FID ({a_fid}) w. minimum tree area change")
####################################################
ax1.legend(loc="best"); ax2.legend(loc="best");
fig.supylabel('coverage (%)', fontsize=16, x=0.05);

# %%
