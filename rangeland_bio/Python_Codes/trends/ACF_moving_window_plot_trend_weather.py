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
# There is another script called ```ACF_moving_window_plot_trend_weather.ipynb```.
#
# **June 26, 2025**

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
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
custom_cmap_GrayW = ListedColormap(['grey', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

best_cmap_ = ListedColormap([(0.9, 0.9, 0.9), 'black'])

fontdict_normal = {'family':'serif', 'weight':'normal'}
fontdict_bold   = {'family':'serif', 'weight':'bold'}
inset_axes_     = [0.1, 0.14, 0.45, 0.03]

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
ACF_plot_base_temp = bio_plots + "ACF1_temp/"
ACF_plot_base_prec = bio_plots + "ACF1_prec/"
os.makedirs(ACF_plot_base_temp, exist_ok=True)
os.makedirs(ACF_plot_base_prec, exist_ok=True)

# %%
ACF_data = rangeland_bio_data + "ACF1/"

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
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west.head(2)

# %% [markdown]
# ## Read all rolling window ACFs

# %%
filename = bio_reOrganized + "weather_ACFs_rollingWindow_trends.sav"

ACF_trends_MK_df = pd.read_pickle(filename)
ACF_trends_MK_df = ACF_trends_MK_df["weather_ACF_trends_MK_df"]
ACF_trends_MK_df.head(2)

# %%
SF_west = pd.merge(SF_west, ACF_trends_MK_df, how="left", on="fid")
SF_west.head(2)

# %% [markdown]
# ```GeoDataFrame``` and ```GeoSeries``` make the ```.plot()``` work differetly:
#
#    - When doing ```GeoDataFrame```, it will plot ploygons.
#    
#    - When doing ```GeoSeries``` (e.g. ```df["centroid"]```), it will accept ```c = 'df['numeric column name]'``` for color and that should be numeric. I could not make it categorical. 
#    
#

# %% [markdown]
# ### Fix the color bar so that for numerical columns, plots are comparable

# %%
# sens_slope is slope of ANPP itself that was saved in SW_west to begin with.
slope_cols = [x for x in SF_west.columns if "slope" in x]
slope_cols = [x for x in slope_cols if "ACF" in x]

temp_slope_cols = [x for x in slope_cols if "temp" in x]
prec_slope_cols = [x for x in slope_cols if "prec" in x]
del(slope_cols)
temp_slope_cols[:3]

# %%
prec_slope_cols[:3]

# %%
trend_cols = [x for x in SF_west if x.startswith("trend")]
temp_trend_cols = [x for x in trend_cols if "temp" in x]
prec_trend_cols = [x for x in trend_cols if "prec" in x]

del(trend_cols)
temp_trend_cols[:4]

# %%
prec_trend_cols[:4]

# %%

# %%
import re

# %%
tick_legend_FontSize = 8
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

plt.rcParams.update(params)

# %% [markdown]
# # Plot Temp. first

# %%
min_ = np.inf
max_ = -np.inf
for col_ in temp_slope_cols:
    if SF_west[col_].min() < min_:
        min_ = SF_west[col_].min()
        
    if SF_west[col_].max() > max_:
        max_ = SF_west[col_].max()

cc_ = max(np.abs(min_), np.max(max_))
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);
print (min_, max_, cc_)

# %%

# %%
# %%time
for type_ in ['slope']: # 'categ', 
    if type_== 'categ':
        for col in temp_trend_cols: # not fully developed this section
            fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
            ax.set_xticks([]); ax.set_yticks([])
            rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_GrayW)
            print ("categorical:", col)
            cent_plt=SF_west.plot(ax=ax, column=col, cmap='viridis', legend=True,
                                  legend_kwds={'bbox_to_anchor': (0, 0), # Position legend outside plot area
                                               'loc': 'lower left',      # Location of the legend
                                               'fontsize': 8,
                                               'borderaxespad': 0.5,     # Padding between legend and plot
                                              });
            ws = re.search(r'ws(\d+)', col).group(1)
            last_part = re.search(r'temp_.*', col).group(0)
            last_part = last_part.replace("temp", "temp.").replace("_", " ")

            file_name = ACF_plot_base_temp + f"Categorical_MK{col}.png"
            ax.set_aspect('equal', adjustable='box')
            
            title_ = f'(MK) trend of ACF1 time-series (window size {ws}, {last_part})'
            plt.title(title_, y=0.98, fontdict=fontdict_bold);
            plt.close()
            try:
                del(cent_plt, cax, cbar1, ws, last_part)
            except:
                pass

    else:
        outdir = ACF_plot_base_temp + "slope/identical_colorbar/"
        os.makedirs(outdir, exist_ok=True)
        for col in temp_slope_cols:
            print ("numerical:  ", col)
            ws = re.search(r'ws(\d+)', col).group(1)
            last_part = re.search(r'temp.*', col).group(0)
            last_part = last_part.replace("temp", "temp.").replace("_", " ")
            
            fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
            ax.set_xticks([]); ax.set_yticks([])
            rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_GrayW)

            cent_plt = SF_west.plot(column=col, ax=ax, legend=False, cmap='seismic', norm=norm_col)
            ############# color bar
            cax = ax.inset_axes(inset_axes_)
            cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
            cbar1.set_label(f'slope of ACF1$_{{ws={ws}}}$', labelpad=2, fontdict=fontdict_normal);
            plt.tight_layout()

            title_ = f"slope of ACF1 time-series (window size {ws}, {last_part})"
            plt.title(title_, y=0.98, fontdict=fontdict_bold);
            file_name = outdir + f"{col}.png"
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
            plt.close();

            try:
                del(cent_plt, cax, cbar1, ws, last_part, file_name)
            except:
                pass
            
    #############

# %%

# %% [markdown]
# ## Different color bars for each plot

# %%
# %%time
del(norm_col, min_, max_, cc_)
outdir = ACF_plot_base_temp + "slope/individual_colorbar/"
os.makedirs(outdir, exist_ok=True)

for col in temp_slope_cols:
    print ("numerical:  ", col)
    ws = re.search(r'ws(\d+)', col).group(1)
    last_part = re.search(r'temp.*', col).group(0)
    last_part = last_part.replace("temp", "temp.").replace("_", " ")

    fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
    ax.set_xticks([]); ax.set_yticks([])
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_GrayW)
    
    min_max0 = max(np.abs(SF_west[col].min()), np.abs(SF_west[col].max()))
    norm0 = Normalize(vmin= -min_max0, vmax=min_max0, clip=True)
    cent_plt = SF_west.plot(column=col, ax=ax, legend=False, cmap='seismic', norm=norm0)
    ############# color bar
    #cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
    cax = ax.inset_axes(inset_axes_)
    cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
    cbar1.set_label(f'slope of ACF1$_{{ws={ws}}}$', labelpad=2, fontdict=fontdict_normal);
    plt.tight_layout()

    title_ = f"slope of ACF1 time-series (window size {ws}, {last_part})"
    plt.title(title_, y=0.98, fontdict=fontdict_bold);
    file_name = outdir + f"indiv_cbar_{col}.png"
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()

    try:
        del(cent_plt, cax, cbar1, ws, last_part, file_name)
    except:
        pass

#############

# %%

# %% [markdown]
# ### <span style="color:red">colormap virdis does not work for polygons!</span>

# %% [markdown]
# # Repeat for Prec.

# %%
del(temp_slope_cols, temp_trend_cols, ACF_plot_base_temp)

# %%
min_ = np.inf
max_ = -np.inf
for col_ in prec_slope_cols:
    if SF_west[col_].min() < min_:
        min_ = SF_west[col_].min()
        
    if SF_west[col_].max() > max_:
        max_ = SF_west[col_].max()

cc_ = max(np.abs(min_), np.max(max_))
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);
print (min_.round(2), max_.round(2), cc_.round(2))

# %%

# %%
# %%time
for type_ in ['slope']: # 'categ', 
    if type_== 'categ':
        for col in prec_trend_cols: # not fully developed this section
            fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
            ax.set_xticks([]); ax.set_yticks([])
            rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_GrayW)
            print ("categorical:", col)
            cent_plt=SF_west.plot(ax=ax, column=col, cmap='viridis', legend=True,
                                  legend_kwds={'bbox_to_anchor': (0, 0), # Position legend outside plot area
                                               'loc': 'lower left',      # Location of the legend
                                               'fontsize': 8,
                                               'borderaxespad': 0.5,     # Padding between legend and plot
                                              });
            ws = re.search(r'ws(\d+)', col).group(1)
            last_part = re.search(r'prec_.*', col).group(0)
            last_part = last_part.replace("prec", "prec.").replace("_", " ")

            file_name = ACF_plot_base_prec + f"Categorical_MK{col}.png"
            ax.set_aspect('equal', adjustable='box')
            
            title_ = f'(MK) trend of ACF1 time-series (window size {ws}, {last_part})'
            plt.title(title_, y=0.98, fontdict=fontdict_bold);
            plt.close()
            try:
                del(cent_plt, cax, cbar1, ws, last_part)
            except:
                pass

    else:
        outdir = ACF_plot_base_prec + "slope/identical_colorbar/"
        os.makedirs(outdir, exist_ok=True)
        for col in prec_slope_cols:
            print ("numerical:  ", col)
            ws = re.search(r'ws(\d+)', col).group(1)
            last_part = re.search(r'prec.*', col).group(0)
            last_part = last_part.replace("prec", "prec.").replace("_", " ")
            
            fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
            ax.set_xticks([]); ax.set_yticks([])
            rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_GrayW)

            cent_plt = SF_west.plot(column=col, ax=ax, legend=False, cmap='seismic', norm=norm_col)
            ############# color bar
            cax = ax.inset_axes(inset_axes_)
            cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
            cbar1.set_label(f'slope of ACF1$_{{ws={ws}}}$', labelpad=2, fontdict=fontdict_normal);
            plt.tight_layout()

            title_ = f"slope of ACF1 time-series (window size {ws}, {last_part})"
            plt.title(title_, y=0.98, fontdict=fontdict_bold);
            file_name = outdir + f"{col}.png"
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
            plt.close();

            try:
                del(cent_plt, cax, cbar1, ws, last_part, file_name)
            except:
                pass
            
    #############

# %%

# %%
# %%time
del(norm_col, min_, max_, cc_)
outdir = ACF_plot_base_prec + "slope/individual_colorbar/"
os.makedirs(outdir, exist_ok=True)

for col in prec_slope_cols:
    print ("numerical:  ", col)
    ws = re.search(r'ws(\d+)', col).group(1)
    last_part = re.search(r'prec.*', col).group(0)
    last_part = last_part.replace("prec", "prec.").replace("_", " ")

    fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
    ax.set_xticks([]); ax.set_yticks([])
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_GrayW)
    
    min_max0 = max(np.abs(SF_west[col].min()), np.abs(SF_west[col].max()))
    norm0 = Normalize(vmin= -min_max0, vmax=min_max0, clip=True)
    cent_plt = SF_west.plot(column=col, ax=ax, legend=False, cmap='seismic', norm=norm0)
    ############# color bar
    #cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
    cax = ax.inset_axes(inset_axes_)
    cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
    cbar1.set_label(f'slope of ACF1$_{{ws={ws}}}$', labelpad=2, fontdict=fontdict_normal);
    plt.tight_layout()

    title_ = f"slope of ACF1 time-series (window size {ws}, {last_part})"
    plt.title(title_, y=0.98, fontdict=fontdict_bold);
    file_name = outdir + f"indiv_cbar_{col}.png"
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()

    try:
        del(cent_plt, cax, cbar1, ws, last_part, file_name)
    except:
        pass

#############

# %%
