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
#

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

fontdict_normal = {'family':'serif', 'weight':'normal'}
fontdict_bold   = {'family':'serif', 'weight':'bold'}
fontdict_bold_sup   = {'family':'serif', 'fontweight':'bold'}
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
variance_plot_base_temp = bio_plots + "variance_rolling_window_temp/"
variance_plot_base_prec = bio_plots + "variance_rolling_window_prec/"
os.makedirs(variance_plot_base_temp, exist_ok=True)
os.makedirs(variance_plot_base_prec, exist_ok=True)

# %%
rolling_variances_data_dir = rangeland_bio_data + "rolling_variances/"

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
# ## Read all rolling window variances
#
# advantage of doing it this way as oppsed to reading
#
# - ```f_name = rangeland_bio_data + 'SF_west_weather_movingVariance.shp.zip'```
# - ```SF_west.to_file(filename=f_name, driver='ESRI Shapefile')```
#
# is that columns have full better names!

# %%
filename = bio_reOrganized + "weather_variance_rollingWindow_trends.sav"

variance_trends_MK_df = pd.read_pickle(filename)
variance_trends_MK_df = variance_trends_MK_df["weather_variance_trends_MK_df"]
variance_trends_MK_df.head(2)

# %%
SF_west = pd.merge(SF_west, variance_trends_MK_df, how="left", on="fid")
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
# sens_slope is slope of ANPP that was saved in SW_west to begin with.
slope_cols = [x for x in SF_west.columns if "slope" in x]
slope_cols = [x for x in slope_cols if "variance" in x]

temp_slope_cols = [x for x in slope_cols if "temp" in x]
prec_slope_cols = [x for x in slope_cols if "prec" in x]

print (f"{len(temp_slope_cols)=}")
print (f"{len(prec_slope_cols)=}")
temp_slope_cols[:4]

# %%
prec_slope_cols[:4]

# %%
trend_cols = [x for x in SF_west if x.startswith("trend")]
temp_trend_cols = [x for x in trend_cols if "temp" in x]
prec_trend_cols = [x for x in trend_cols if "prec" in x]

print (len(trend_cols))
temp_trend_cols[:4]

# %%
prec_trend_cols[:4]

# %%
import re
del(slope_cols, trend_cols)

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
          'axes.linewidth' : .05}
plt.rcParams.update(params)

# %% [markdown]
# ### Temperature plots first

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
print (round(min_, 2), round(max_, 2), round(cc_, 2))

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

            file_name = ddd + f"Categorical_MK{col}.png"
            ax.set_aspect('equal', adjustable='box')
            title_ = f'(MK) trend of variance time-series (window size {ws}, {last_part})'
            plt.title(title_, y=0.98, fontdict=fontdict_bold);
            plt.close()
            try:
                del(cent_plt, cax, cbar1, ws, last_part)
            except:
                pass

    else:
        outdir = variance_plot_base_temp + "slope/identical_colorbar/"
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
            # cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
            cax = ax.inset_axes(inset_axes_)
            cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
            
            L_ = fr'slope of $\sigma^2_{{ws={ws}}}$'
            cbar1.set_label(L_, labelpad=2, fontdict=fontdict_normal)
            plt.tight_layout()
            # on overleaf, a sublot looked slightly higher than another. lets see if this fixes it
            # title_ = f"slope of variance time-series (window size {ws}, {last_part})"
            plt.title(L_ + f' ({last_part})', y=0.98, fontdict=fontdict_bold);
            
            file_name = outdir + f"{col}.png"
            ax.set_aspect('equal', adjustable='box')
            
            plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
            plt.close()

            try:
                del(cent_plt, cax, cbar1, ws, last_part, file_name)
            except:
                pass
            
    #############

# %% [markdown]
# ## Different color bars for each plot

# %%
del(norm_col, min_, max_, cc_)
outdir = variance_plot_base_temp + "slope/individual_colorbar/"
os.makedirs(outdir, exist_ok=True)
print (outdir)

# %%
# %%time
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
    # cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
    cax = ax.inset_axes(inset_axes_)
    cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
    
    L_ = fr'slope of $\sigma^2_{{ws={ws}}}$'
    cbar1.set_label(L_, labelpad=2, fontdict=fontdict_normal)
    plt.tight_layout()
    # on overleaf, a sublot looked slightly higher than another. lets see if this fixes it
    # title_ = f"slope of variance time-series (window size {ws}, {last_part})"
    plt.title(L_ + f' ({last_part})', y=0.98, fontdict=fontdict_bold);
    file_name = outdir + f"indiv_cbar_{col}.png"
    ax.set_aspect('equal', adjustable='box')
    
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()
    
    try:
        del(cent_plt, cax, cbar1, ws, last_part, file_name)
    except:
        pass

#############

# %% [markdown]
# ### <span style="color:red">colormap virdis does not work for polygons!</span>

# %% [markdown]
# ## one variable at a time. outliers separated.

# %%
outdir = variance_plot_base_temp + "slope/individual_colorbar_outliers/"
os.makedirs(outdir, exist_ok=True)
print (outdir)

# %%

# %%
# %%time
for y_var in temp_slope_cols:
    print (y_var)
    fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})
    ax[0].set_xticks([]); ax[0].set_yticks([]);
    ax[1].set_xticks([]); ax[1].set_yticks([]);

    # Plotting the data with original colormap (don't change the color normalization)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=custom_cmap_GrayW)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=custom_cmap_GrayW)

    df = SF_west.copy()
    df.dropna(subset=[y_var], inplace=True)

    perc_ = 10 / 100
    lower_bound = df[y_var].quantile(perc_)
    upper_bound = df[y_var].quantile(1 - perc_)

    # Filter rows between 10th and 90th percentile (inclusive)
    filtered_between = df[(df[y_var] >= lower_bound) & (df[y_var] <= upper_bound)]
    filtered_outside = df[(df[y_var] < lower_bound) | (df[y_var] > upper_bound)]

    ############
    min_max0 = max(np.abs(filtered_between[y_var].min()), np.abs(filtered_between[y_var].max()))
    min_max1 = max(np.abs(filtered_outside[y_var].min()), np.abs(filtered_outside[y_var].max()))

    norm0 = Normalize(vmin= -min_max0, vmax=min_max0, clip=True)
    norm1 = Normalize(vmin= -min_max1, vmax=min_max1, clip=True)

    cent_plt0 = filtered_between.plot(ax=ax[0], column=y_var, legend=False, cmap='seismic', norm=norm0)
    cent_plt1 = filtered_outside.plot(ax=ax[1], column=y_var, legend=False, cmap='seismic', norm=norm1)

    cax0 = ax[0].inset_axes(inset_axes_)
    cax1 = ax[1].inset_axes(inset_axes_)

    ws = re.search(r'ws(\d+)', y_var).group(1)
    last_part = re.search(r'temp.*', y_var).group(0)
    last_part = last_part.replace("temp", "temp.").replace("_", " ")

    cbar0 = fig.colorbar(cent_plt0.collections[1], ax=ax[0], norm=norm0, cax=cax0,
                         shrink=0.3, orientation='horizontal')

    cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax[1], norm=norm1, cax=cax1,
                         shrink=0.3, orientation='horizontal')

    L_ = fr'slope of $\sigma^2_{{ws={ws}}}$'
    cbar0.set_label(L_, labelpad=2, fontdict=fontdict_normal)
    cbar1.set_label(L_, labelpad=2, fontdict=fontdict_normal)
    
    # L_ = f"slope of variance time-series (window size {ws}, {last_part})"
    fig.suptitle(L_ + f' ({last_part})', y=0.82, fontdict=fontdict_bold_sup);
    plt.tight_layout()
    
    file_name = outdir + y_var + "_divergeRB_greyBG.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()
    del(cent_plt0, cent_plt1, cax0, cax1, cbar0, cbar1, norm0, norm1, min_max0, min_max1,
        filtered_between, filtered_outside)

# %% [markdown]
# ## Repeat for Precip

# %%
del(temp_slope_cols, variance_plot_base_temp, temp_trend_cols)

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
print (round(min_, 2), round(max_, 2), round(cc_, 2))

# %%
# cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])

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

            file_name = ddd + f"Categorical_MK{col}.png"
            ax.set_aspect('equal', adjustable='box')
            
            title_ = f'(MK) trend of variance time-series (window size {ws}, {last_part})'
            plt.title(title_, y=0.98, fontdict=fontdict_bold);
            plt.close()
            try:
                del(cent_plt, cax, cbar1, ws, last_part)
            except:
                pass

    else:
        outdir = variance_plot_base_prec + "slope/identical_colorbar/"
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
            L_ = fr'slope of $\sigma^2_{{ws={ws}}}$'
            cbar1.set_label(L_, labelpad=2, fontdict=fontdict_normal)
            plt.tight_layout()

            # title_ = f"slope of variance time-series (window size {ws}, {last_part})"
            plt.title(L_ + f' ({last_part})', y=0.98, fontdict=fontdict_bold);
            file_name = outdir + f"{col}.png"
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
            plt.close()

            try:
                del(cent_plt, cax, cbar1, ws, last_part, file_name)
            except:
                pass
            
    #############
print ()   
print (outdir)

# %%
del(norm_col, min_, max_, cc_)

# %%

# %%
# %%time

outdir = variance_plot_base_prec + "slope/individual_colorbar/"
os.makedirs(outdir, exist_ok=True)
print (outdir)
print ()

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
    cax = ax.inset_axes(inset_axes_)
    cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
    
    L_ = fr'slope of $\sigma^2_{{ws={ws}}}$'
    cbar1.set_label(L_, labelpad=2, fontdict=fontdict_normal)
    plt.tight_layout()
    
    # title_ = f"slope of variance time-series (window size {ws}, {last_part})"
    plt.title(L_ + f' ({last_part})', y=0.98, fontdict=fontdict_bold);
    file_name = outdir + f"indiv_cbar_{col}.png"
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()

    try:
        del(cent_plt, cax, cbar1, ws, last_part, file_name)
    except:
        pass

# %%
outdir = variance_plot_base_prec + "slope/individual_colorbar_outliers/"
os.makedirs(outdir, exist_ok=True)
print (outdir)

# %%
# %%time
for y_var in prec_slope_cols:
    print (y_var)
    fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})
    ax[0].set_xticks([]); ax[0].set_yticks([]);
    ax[1].set_xticks([]); ax[1].set_yticks([]);

    # Plotting the data with original colormap (don't change the color normalization)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=custom_cmap_GrayW)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=custom_cmap_GrayW)

    df = SF_west.copy()
    df.dropna(subset=[y_var], inplace=True)

    perc_ = 10 / 100
    lower_bound = df[y_var].quantile(perc_)
    upper_bound = df[y_var].quantile(1 - perc_)

    # Filter rows between 10th and 90th percentile (inclusive)
    filtered_between = df[(df[y_var] >= lower_bound) & (df[y_var] <= upper_bound)]
    filtered_outside = df[(df[y_var] < lower_bound) | (df[y_var] > upper_bound)]

    ############
    min_max0 = max(np.abs(filtered_between[y_var].min()), np.abs(filtered_between[y_var].max()))
    min_max1 = max(np.abs(filtered_outside[y_var].min()), np.abs(filtered_outside[y_var].max()))

    norm0 = Normalize(vmin= -min_max0, vmax=min_max0, clip=True)
    norm1 = Normalize(vmin= -min_max1, vmax=min_max1, clip=True)

    cent_plt0 = filtered_between.plot(ax=ax[0], column=y_var, legend=False, cmap='seismic', norm=norm0)
    cent_plt1 = filtered_outside.plot(ax=ax[1], column=y_var, legend=False, cmap='seismic', norm=norm1)

    cax0 = ax[0].inset_axes(inset_axes_)
    cax1 = ax[1].inset_axes(inset_axes_)

    ws = re.search(r'ws(\d+)', y_var).group(1)
    last_part = re.search(r'prec.*', y_var).group(0)
    last_part = last_part.replace("prec", "prec.").replace("_", " ")

    cbar0 = fig.colorbar(cent_plt0.collections[1], ax=ax[0], norm=norm0, cax=cax0,
                         shrink=0.3, orientation='horizontal')

    cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax[1], norm=norm1, cax=cax1,
                         shrink=0.3, orientation='horizontal')

    L_ = fr'slope of $\sigma^2_{{ws={ws}}}$'
    cbar0.set_label(L_, labelpad=2, fontdict=fontdict_normal)
    cbar1.set_label(L_, labelpad=2, fontdict=fontdict_normal)

    # f"slope of variance time-series (window size {ws}, {last_part})"
    fig.suptitle(L_ + f' ({last_part})', y=0.82, fontdict=fontdict_bold_sup)
    plt.tight_layout()
    
    file_name = outdir + y_var + "_divergeRB_greyBG.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()
    del(cent_plt0, cent_plt1, cax0, cax1, cbar0, cbar1, norm0, norm1, min_max0, min_max1,
        filtered_between, filtered_outside)

print ()    
print (outdir)

# %%

# %%
