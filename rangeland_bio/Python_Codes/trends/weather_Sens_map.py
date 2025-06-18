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
dpi_, map_dpi_=300, 500
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

best_cmap_ = ListedColormap([(0.9, 0.9, 0.9), 'black'])
fontdict_normal = fontdict={'family':'serif', 'weight':'normal'}
fontdict_bold   = fontdict={'family':'serif', 'weight':'bold'}
inset_axes_     = [0.1, 0.13, 0.45, 0.03]
inset_axes_     = [0.1, 0.18, 0.45, 0.03] # for tight layout we need

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

# %%
f_name = bio_reOrganized + 'Albers_SF_west_weather_MK_Spearman.shp.zip'
Albers_SF_west = geopandas.read_file(f_name)
# Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid
Albers_SF_west.head(2)

# %%
sorted(Albers_SF_west.columns)

# %%
Albers_SF_west.rename(columns={"EW_meridia" : "EW_meridian",
                               'Sprmn_prec' : 'Spearrman_precip',
                               'Sprmn_temp' : 'Spearman_temp',
                               'intcp_prec' : 'intercept_precip',
                               'intcp_temp' : 'intercept_temp',
                               'sen_m_prec' : 'sens_slope_precip',
                               'sen_m_temp' : 'sens_slope_temp',
                               'state_majo' : 'state_majority',
                               'trnd_preci' : 'trend_precip',
                               'trnd_temp'  : 'trend_temp'}, 
                      inplace=True)

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
# cc_ = [, plt.cm.Pastel2.colors[0]]


# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %% [markdown]
# ### Plot a couple of examples

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

# %% [markdown]
# In order to have the center at ```yellow``` we manipulated ```vmin``` and ```vmax```.
# Another way is [TwoSlopeNorm](https://matplotlib.org/stable/users/explain/colors/colormapnorms.html). Not pretty.
#
# Or from AI?
# ```norm = colors.MidpointNormalize(midpoint=midpoint, vmin=data.min(), vmax=data.max())```?

# %%
Albers_SF_west.head(2)

# %%
[x for x in Albers_SF_west.columns if "slope" in x]

# %%
y_ = 'sens_slope_temp'
fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_].min()), np.abs(Albers_SF_west[y_].max()))
norm1 = Normalize(vmin = -min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)

cbar1.set_label(r"Sen's slope", labelpad=1, fontdict=fontdict_normal)
plt.title("temperature Sen's slope", fontdict=fontdict_bold)

plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "temp_sensSlopes_centerColorBar.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%

# %%

# %%
y_ = 'sens_slope_precip'
fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=map_dpi_)
ax.set_xticks([]); ax.set_yticks([])

min_max = max(np.abs(Albers_SF_west[y_].min()), np.abs(Albers_SF_west[y_].max()))
norm1 = Normalize(vmin = -min_max, vmax=min_max, clip=True)

rcp.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

cent_plt = Albers_SF_west.plot(column=y_, ax=ax, legend=False, cmap='seismic', norm=norm1)

# first two arguments are x and y of the legend 
# on the left side of it. The last two are length and width of the bar
cax = ax.inset_axes(inset_axes_)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, norm=norm1, cax=cax)

cbar1.set_label(r"Sen's slope", labelpad=1, fontdict=fontdict_normal)
plt.title("precipitation Sen's slope", fontdict=fontdict_bold)

plt.tight_layout()
# fig.subplots_adjust(top=0.91, bottom=0.01, left=-0.1, right=1)
file_name = bio_plots + "precip_sensSlopes_centerColorBar.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

del(cent_plt, cax, cbar1, norm1, min_max)

# %%
file_name

# %%
