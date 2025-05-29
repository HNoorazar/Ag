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
          "axes.titlesize":  tick_legend_FontSize * 1,
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

# %%

# %%
print(SF_west[y_var].min())
print(SF_west[y_var].max())

# %%
y_var = "autocorr_lag1_ws5_variance"

# %%
for ws in np.arange(5, 11):
    y_var = f"autocorr_lag1_ws{ws}_variance"
    
    fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
    ax.set_xticks([]); ax.set_yticks([])
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=custom_cmap_BW)

    # cmap = bwr or 'seismic'
    cent_plt = SF_west["centroid"].plot(ax=ax, c=SF_west[y_var], markersize=0.1) 
    plt.tight_layout()

    ############# color bar
    cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
    cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
    cbar1.set_label(f'$\sigma$(ACF1$_{{ws={ws}}}$)', labelpad=2)

    #############

    plt.title(f'variance of ACF1 time series w. window size {ws}', y=0.98);

    file_name = ACF_plot_base + f"variance_of_ACF1_ws{ws}.png" # ANPP_ACF1_zeroWhite or ANPP_ACF1
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    del(cent_plt, cax, cbar1)

# %%

# %%

# %%
