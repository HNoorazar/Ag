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
# There is another script called ```autocorr_moving_window_analysis_archived.ipynb```.
#
# This is a copy of that with modifications. That notebook was just fine. Here I am editting it so that we have more stuff from detrended ANPP in it. AND, variance of ACF1's are removed here. We may need them again.
#
# **June 12, 2025**

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
dpi_, map_dpi_=300, 500
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
custom_cmap_GrayW = ListedColormap(['gray', 'black'])
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
ys = ["anpp", "anpp_detrendLinReg", "anpp_detrendDiff", "anpp_detrendSens"]

# %%
# %%
filename = bio_reOrganized + "ACFs_rollingWindow_trends.sav"

ACF_trends_MK_df = pd.read_pickle(filename)
ACF_trends_MK_df = ACF_trends_MK_df["ACF_trends_MK_df"]
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
slope_cols

# %%
trend_cols = [x for x in SF_west if x.startswith("trend")]
trend_cols = [x for x in trend_cols if "anpp" in x]
trend_cols

# %%
import re

# %%

# %%
min_ = np.inf
max_ = -np.inf
for col_ in slope_cols:
    if SF_west[col_].min() < min_:
        min_ = SF_west[col_].min()
        
    if SF_west[col_].max() > max_:
        max_ = SF_west[col_].max()

cc_ = max(np.abs(min_), np.max(max_))
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);
print (min_, max_, cc_)

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

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
ACF_plot_base

# %%

# %%
# %%time
for type_ in ['slope']: # 'categ', 
    if type_== 'categ':
        for col in trend_cols: # not fully developed this section
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
            last_part = re.search(r'anpp_.*', col).group(0)
            last_part = last_part.replace("anpp", "ANPP").replace("_", " ")

            file_name = ACF_plot_base + f"Categorical_MK{col}.png"
            plt.title(f'(MK) trend of ACF1 time-series (window size {ws}, {last_part})', y=0.98);
            plt.close()
            try:
                del(cent_plt, cax, cbar1, ws, last_part)
            except:
                pass

    else:
        outdir = ACF_plot_base + "slope/identical_colorbar/"
        os.makedirs(outdir, exist_ok=True)
        for col in slope_cols:
            print ("numerical:  ", col)
            ws = re.search(r'ws(\d+)', col).group(1)
            last_part = re.search(r'anpp.*', col).group(0)
            last_part = last_part.replace("anpp", "ANPP").replace("_", " ")
            
            fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
            ax.set_xticks([]); ax.set_yticks([])
            rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_GrayW)

            cent_plt = SF_west.plot(column=col, ax=ax, legend=False, cmap='seismic', norm=norm_col)
            ############# color bar
            cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
            cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
            cbar1.set_label(f'slope of ACF1$_{{ws={ws}}}$', labelpad=2)
            plt.tight_layout()
            plt.title(f"slope of ACF1 time-series (window size {ws}, {last_part})", y=0.98);
            file_name = outdir + f"{col}.png"
            plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
            plt.close()

            try:
                del(cent_plt, cax, cbar1, ws, last_part, file_name)
            except:
                pass
            
    #############

# %%

# %% [markdown]
# ## Different color bars for each plot

# %%
del(norm_col, min_, max_, cc_)
outdir = ACF_plot_base + "slope/individual_colorbar/"
os.makedirs(outdir, exist_ok=True)

# %%
# %%time
for col in slope_cols:
    print ("numerical:  ", col)
    ws = re.search(r'ws(\d+)', col).group(1)
    last_part = re.search(r'anpp.*', col).group(0)
    last_part = last_part.replace("anpp", "ANPP").replace("_", " ")

    fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
    ax.set_xticks([]); ax.set_yticks([])
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_GrayW)
    
    min_max0 = max(np.abs(SF_west[col].min()), np.abs(SF_west[col].max()))
    norm0 = Normalize(vmin= -min_max0, vmax=min_max0, clip=True)
    cent_plt = SF_west.plot(column=col, ax=ax, legend=False, cmap='seismic', norm=norm0)
    ############# color bar
    cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
    cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
    cbar1.set_label(f'slope of ACF1$_{{ws={ws}}}$', labelpad=2)
    plt.tight_layout()
    plt.title(f"slope of ACF1 time-series (window size {ws}, {last_part})", y=0.98);
    file_name = outdir + f"indiv_cbar_{col}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()

    try:
        del(cent_plt, cax, cbar1, ws, last_part, file_name)
    except:
        pass

#############

# %%

# %% [markdown]
# ## colormap virdis does not work for polygons!
