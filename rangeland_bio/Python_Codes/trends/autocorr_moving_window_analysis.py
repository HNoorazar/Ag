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

import seaborn as sns
from scipy import stats
import scipy.stats as scipy_stats
from statsmodels.tsa.stattools import acf
import geopandas

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc

# %%
dpi_, map_dpi_=300, 900
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

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

# %%
bio_plots = rangeland_bio_base + "plots/"
ACF_plot_base = bio_plots + "ACF1/"
os.makedirs(ACF_plot_base, exist_ok=True)

# %%
ACF_data = rangeland_bio_data + "ACF1/"

# %%

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

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
# gdf = geopandas.read_file(common_data +'cb_2018_us_state_500k')

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
## bad 2012
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %%
print (ANPP.year.min())
print (ANPP.year.max())

# %% [markdown]
# ## Read all rolling window ACFs

# %%
window_size = 5
key_ = f"rolling_autocorrelations_ws{window_size}"
f_name = ACF_data + key_ + ".sav"
ACF_df = pd.read_pickle(f_name)
ACF_df

# %%
ACF_dict = {}
for window_size in np.arange(5, 11):
    key_ = f"rolling_autocorrelations_ws{window_size}"
    f_name = ACF_data + key_ + ".sav"
    ACF_df = pd.read_pickle(f_name)
    ACF_df = ACF_df[key_]
    key_ = f"ACF1_ws{window_size}"
    ACF_dict[key_] = ACF_df

# %% [markdown]
# ### Compute all variances and plot them 

# %%
ACF_variances_dict = {}
for a_key in ACF_dict.keys():
    ws_com = a_key.split("_")[-1]
    ACF_df = ACF_dict[a_key]
    ACF_variance_df = ACF_df.groupby("fid")[f"autocorr_lag1_{ws_com}"].var()
    ACF_variance_df = ACF_variance_df.reset_index(name=f'autocorr_lag1_{ws_com}_variance')
    ACF_variances_dict[a_key + "_variances"] = ACF_variance_df
    
    SF_west = pd.merge(SF_west, ACF_variance_df, on="fid", how="left")

# %%
ACF_variances_dict["ACF1_ws5_variances"].head(2)

# %%
ACF_variances_dict["ACF1_ws10_variances"].head(2)

# %%
SF_west.head(2)

# %%

# %%
ACF_variances_dict.keys()

# %%
tick_legend_FontSize = 12
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1.1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 10,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05
}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %% [markdown]
# ## Normalize so plots are comparable

# %%
min_ = np.inf
max_ = -np.inf
for ws in np.arange(5, 11):
    y_var = f"autocorr_lag1_ws{ws}_variance"
    if SF_west[y_var].min() < min_:
        min_ = SF_west[y_var].min()
        
    if SF_west[y_var].max() > max_:
        max_ = SF_west[y_var].max()

cc_ = max(np.abs(min_), np.max(max_))
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);
print (min_, max_, cc_)

# %%
for ws in np.arange(5, 11):
    y_var = f"autocorr_lag1_ws{ws}_variance"
    
    fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
    ax.set_xticks([]); ax.set_yticks([])
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_BW)

    ############# plot centroids with color of certain column
    # cmap = bwr or 'seismic'
    cent_plt = SF_west["centroid"].plot(ax=ax, c=SF_west[y_var], markersize=0.1, norm=norm_col)
    plt.tight_layout()

    ##### color bar
    cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
    cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
    cbar1.set_label(f'$\sigma^2$(ACF1$_{{ws={ws}}}$)', labelpad=2)
    #############

    plt.title(f'variance of ACF1 time series w. window size {ws}', y=0.98);

    file_name = ACF_plot_base + f"variance_of_ACF1_ws{ws}.png" # ANPP_ACF1_zeroWhite or ANPP_ACF1
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    del(cent_plt, cax, cbar1)

# %% [markdown]
# ### Find trends of ACF1 time-series via MK again!

# %%
ACF_dict.keys()

# %%
ACF_dict["ACF1_ws5"].head(2)

# %%
import importlib;
importlib.reload(rc);
importlib.reload(rpc);

# %%
# %%time

ACF_trends_MK_dict = {}

for ws in np.arange(5, 11):
    curr_df = ACF_dict[f'ACF1_ws{ws}']
    curr_col = f'autocorr_lag1_ws{ws}'
    ACF_trends_MK_dict[f'ACF1_ws{ws}'] = rc.compute_mk_by_fid(df=curr_df, groupby_='fid', value_col=curr_col)

# %%
ACF_trends_MK_dict["ACF1_ws5"].head(3)

# %%
for ws_ in np.arange(5, 11):
    key_ = f"ACF1_ws{ws_}"
    ACF_trends_MK_dict[key_].rename(columns={"trend": f"trend_ws{ws_}",
                                             "p_value": f"p_value_ws{ws_}",
                                             "slope": f"slope_ws{ws_}"}, 
                                    inplace=True)

# %%
ACF_trends_MK_dict["ACF1_ws5"].head(2)

# %%
from functools import reduce

# Convert dict values to a list of DataFrames
df_list = list(ACF_trends_MK_dict.values())

# Perform left merges iteratively
ACF_trends_MK_df = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

ACF_trends_MK_df.head(2)

# %%
SF_west = pd.merge(SF_west, ACF_trends_MK_df, how="left", on="fid")

# %%
# %%time
f_name = rangeland_bio_data + 'SF_west_movingACF1s.shp.zip'

SF_west_2write = SF_west.copy()
SF_west_2write["centroid"] = SF_west_2write["centroid"].astype(str)

# SF_west_2write.drop(columns=["centroid"], inplace=True) # it does not like 2 geometries!
SF_west_2write.to_file(filename=f_name, driver='ESRI Shapefile')
del(SF_west_2write)

# %%

# %%
y_var = "trend_ws5"

# %%
SF_west.head(2)

# %%
# SF_west[y_var] = SF_west[y_var].astype("category")
SF_west[y_var] = SF_west[y_var].astype(str)

# %%
# fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
# ax.set_xticks([]); ax.set_yticks([])
# rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_BW)
# cent_plt=SF_west.plot(ax=ax, column="trend_ws7", cmap='viridis', legend=True,
#                      legend_kwds={'bbox_to_anchor': (.05, 0), # Position legend outside plot area
#                                   'loc': 'lower left',        # Location of the legend
#                                   'fontsize': 8,              # Shrink the font size
#                                   'borderaxespad': 0.5,       # Padding between legend and plot
#                                  });

# file_name = ACF_plot_base + f"test.png"
# plt.savefig(file_name, bbox_inches='tight', dpi=300)

# %%
# SF_west["centroid"].plot(c="trend_ws9", cmap='viridis', figsize=(10, 8));

# %%
# SF_west.plot(column="slope_ws5", cmap='viridis', legend=True, figsize=(10, 8))
# SF_west.plot(column="slope_ws9", legend=True, figsize=(10, 8));

# %%
SF_west.head(2)

# %% [markdown]
# ```GeoDataFrame``` and ```GeoSeries``` mage the ```.plot()``` work differetly:
#
#    - When doing ```GeoDataFrame```, it will plot ploygons.
#    
#    - When doing ```GeoSeries``` (e.g. ```df["centroid"]```), it will accept ```c = 'df['numeric column name]'``` for color and that should be numeric. I could not make it categorical. 
#    
#

# %%
print(SF_west[y_var].dtype)
print(SF_west[y_var].unique())

print ()
print(SF_west[y_var].min())
print(SF_west[y_var].max())

# %% [markdown]
# ### Fix the color bar so that for numerical columns, plots are comparable

# %%
del(min_, max_, cc_)

# %%
min_ = np.inf
max_ = -np.inf
for ws in np.arange(5, 11):
    y_var = f"slope_ws{ws}"
    if SF_west[y_var].min() < min_:
        min_ = SF_west[y_var].min()
        
    if SF_west[y_var].max() > max_:
        max_ = SF_west[y_var].max()

cc_ = max(np.abs(min_), np.max(max_))
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);
print (min_, max_, cc_)

# %%

# %%
for ws in np.arange(5, 11):
    for type_ in ['categ', 'slope']:
        fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
        ax.set_xticks([]); ax.set_yticks([])
        rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_BW)

        if type_== 'categ':
            y_var = f"trend_ws{ws}"
            print ("categorical:", y_var)
            cent_plt=SF_west.plot(ax=ax, column=y_var, cmap='viridis', legend=True,
                                  legend_kwds={'bbox_to_anchor': (0, 0), # Position legend outside plot area
                                               'loc': 'lower left',      # Location of the legend
                                               'fontsize': 8,            # Shrink the font size
                                               'borderaxespad': 0.5,     # Padding between legend and plot
                                              });
            file_name = ACF_plot_base + f"Categorical_MKTrend_of_ACF1_ws{ws}.png"

        else:
            y_var = f"slope_ws{ws}"
            print ("numerical:  ", y_var)
            cent_plt = SF_west["centroid"].plot(ax=ax, c=SF_west[y_var], markersize=0.1, norm=norm_col)
            
            ############# color bar
            cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
            cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
            cbar1.set_label(f'trend of ACF1$_{{ws={ws}}}$', labelpad=2)
            
            file_name = ACF_plot_base + f"sensSlope_trend_of_ACF1_ws{ws}.png"
            del(cax, cbar1)

        plt.tight_layout()

        #############

        plt.title(f'(MK) trend of ACF1 time-series w. window size {ws}', y=0.98);
        
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
        plt.close()
        try:
            del(cent_plt, cax, cbar1)
        except:
            pass

# %%

# %%

# %%
tick_legend_FontSize = 12
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .2, # this does not work below
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1.1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 10,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05
}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
y_var = "slope_ws7"

fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})

ax[0].set_xticks([]); ax[0].set_yticks([]);
ax[1].set_xticks([]); ax[1].set_yticks([]);
# ax[1][0].set_xticks([]); ax[1][0].set_yticks([]);
# ax[1][1].set_xticks([]); ax[1][1].set_yticks([]);

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=custom_cmap_BW)
cent_plt00 = SF_west["centroid"].plot(ax=ax[0], c=SF_west[y_var], markersize=0.1, cmap='viridis');
# SF_west["centroid"].plot(ax=ax[0][1], c=SF_west[y_var], markersize=0.1, cmap="plasma");
# SF_west["centroid"].plot(ax=ax[1][1], c=SF_west[y_var], markersize=0.1, cmap="inferno");
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=custom_cmap_BW)
cent_plt10 = SF_west["centroid"].plot(ax=ax[1], c=SF_west[y_var], markersize=0.1);

cax = ax[0].inset_axes([0.03, 0.18, 0.5, 0.03])
cbar00 = fig.colorbar(cent_plt00.collections[1], ax=ax[0], orientation='horizontal', shrink=0.3, cax=cax)
cbar00.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar00.set_label(f"Sen's slope (virdis cmap)", labelpad=2, fontsize=tick_legend_FontSize * .6)

cax = ax[1].inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(cent_plt10.collections[1], ax=ax[1], orientation='horizontal', shrink=0.3, cax=cax)
cbar1.ax.tick_params(labelsize=tick_legend_FontSize*0.6)
cbar1.set_label(f"Sen's slope (default color)", labelpad=2, fontsize=tick_legend_FontSize * .6)


plt.tight_layout()
# plt.subplots_adjust(hspace=0, wspace=0)
fig.suptitle(f'(MK) trend of ACF1 time-series w. window size 7', y=0.82);
file_name = ACF_plot_base + f"sensSlope_ACF1_{y_var}_colorMaps.png"
plt.savefig(file_name, bbox_inches='tight', dpi=300)

# %%

# %%
# Create the figure and axes
fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})

# Removing ticks from both subplots
ax[0].set_xticks([]); ax[0].set_yticks([]);
ax[1].set_xticks([]); ax[1].set_yticks([]);

# Plotting the data with original colormap (don't change the color normalization)
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=custom_cmap_BW)
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=custom_cmap_BW)

cent_plt00 = SF_west["centroid"].plot(ax=ax[0], c=SF_west[y_var], markersize=0.1, cmap='viridis')
cent_plt10 = SF_west["centroid"].plot(ax=ax[1], c=SF_west[y_var], markersize=0.1)
##########################################################################################
###############
############### default map setting
###############
cax10 = ax[1].inset_axes([0.03, 0.18, 0.5, 0.03])  # Define a new inset axis for the second color bar
cbar10 = fig.colorbar(cent_plt10.collections[1], ax=ax[1], orientation='horizontal', shrink=0.3, cax=cax10)

tick_labels = cbar10.get_ticks()
cbar10.set_ticklabels(tick_labels)

cbar10.set_label(f"Sen's slope (default cmap)", labelpad=2, fontsize=tick_legend_FontSize * 0.6)
cbar10.ax.tick_params(labelsize=tick_legend_FontSize * .6)

##########################################################################################
###############
############### virdis map setting
###############
# Adding colorbar for the first plot (ax[0]) with custom normalization for legend only
cax_00 = ax[0].inset_axes([0.03, 0.18, 0.5, 0.03])
cbar00 = fig.colorbar(cent_plt00.collections[1], ax=ax[0], orientation='horizontal', shrink=0.3, cax=cax_00)

# Keep color bar the same but adjust tick labels to reflect the original data range
# Get the default tick locations (in normalized space)
# ticks = cbar00.get_ticks()
# min_ = SF_west[y_var].min()
# max_ = SF_west[y_var].max()
# tick_labels = [round(min_ + (tick * (max_ - min_)), 2) for tick in ticks]

# Set the tick labels to the original data values
# cbar00.set_ticks(ticks)  # Set the locations of the ticks
cbar00.set_ticklabels(tick_labels)

cbar00.set_label(f"Sen's slope (viridis cmap)", labelpad=2, fontsize=tick_legend_FontSize * .6)
cbar00.ax.tick_params(labelsize=tick_legend_FontSize * .6)

##########################################################################################
############### 
############### # Adjust layout and title
plt.tight_layout()
fig.suptitle(f'(MK) trend of ACF1 time-series w. window size 7', y=0.82)

# Optionally save the plot
file_name = ACF_plot_base + f"sensSlope_ACF1_{y_var}_colorMaps_tickIdent.png"
plt.savefig(file_name, bbox_inches='tight', dpi=300)
plt.show()

# %%
