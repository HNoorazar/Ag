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
# # A draft 
# to do sign switch plots (before and after ANPP-BP1) for both ANPP and weather

# %%
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os, os.path, pickle, sys
import pymannkendall as mk

import statistics
import statsmodels.formula.api as smf

import statsmodels.stats.api as sms
import statsmodels.api as sm

from scipy import stats
import scipy.stats as scipy_stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc

# %%
import importlib;
importlib.reload(rc);
importlib.reload(rpc);

# %%
dpi_, map_dpi_ = 300, 300
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

fontdict_normal = {'family':'serif', 'weight':'normal'}
fontdict_bold   = {'family':'serif', 'weight':'bold'}
inset_axes_     = [0.1, 0.13, 0.45, 0.03]

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

rangeland_bio_base = research_db + "/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

breakpoint_plot_base = bio_plots + "breakpoints/"
os.makedirs(breakpoint_plot_base, exist_ok=True)

sign_switch_plot_dir = breakpoint_plot_base + "sign_switch/"
os.makedirs(sign_switch_plot_dir, exist_ok=True)

# %%
breakpoints_dir = rangeland_bio_data + "breakpoints/"

# %%
anpp_or_weather = "anpp"

# %%
if anpp_or_weather == "weather":
    filename = breakpoints_dir + "01_weather_Sen_ACF_stats_beforeAfter_ANPPBP1.sav"
    weather_ANPPBP1 = pd.read_pickle(filename)
    print (list(weather_ANPPBP1.keys()))
    print ()
    print (f"{weather_ANPPBP1['source_code'] = }")
    print (f"{weather_ANPPBP1['Date'] = }")
    ####
    #### Remember these are Sen's slopes
    ####
    beforeAfterSlopes_df = weather_ANPPBP1["sensSlope_stats_ACF_beforeAfter_ANPPBP1"]
elif anpp_or_weather == "anpp":
    filename = breakpoints_dir + "sensSlope_beforeAfter_BP1.sav"
    ANPP_slopes_beforeAfter_BP1 = pd.read_pickle(filename)
    print (ANPP_slopes_beforeAfter_BP1.keys())
    
    beforeAfterSlopes_df = ANPP_slopes_beforeAfter_BP1["sensSlope_beforeAfter_BP1"]
    
beforeAfterSlopes_df.head(2)

# %%
# Step 1: Get all _before and _after columns
before_cols = [col for col in beforeAfterSlopes_df.columns if col.endswith('slope_before')]
after_cols_set = set([col for col in beforeAfterSlopes_df.columns if col.endswith('slope_after')])

# Step 2: Keep only pairs where both _before and _after columns exist
paired_bases = []
for before_col in before_cols:
    base = before_col.rsplit('_before', 1)[0]
    after_col = base + '_after'
    if after_col in after_cols_set:
        paired_bases.append(base)
        
# Step 3: Compute sign categories for each matched pair
sign_map = {1: 'P', -1: 'N', 0: 'Z'}

for base in paired_bases:
    before_col = base + '_before'
    after_col = base + '_after'

    # Calculate sign-based category for each row
    beforeAfterSlopes_df[base + '_sign_category'] = [
        f"{sign_map.get(b)}_{sign_map.get(a)}"
        for b, a in zip(np.sign(beforeAfterSlopes_df[before_col]), np.sign(beforeAfterSlopes_df[after_col]))
    ]

beforeAfterSlopes_df.head(2)

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
# %%time
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
# %%time
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
Albers_SF_west = geopandas.read_file(f_name)
Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid

Albers_SF_west.rename(columns={"EW_meridia": "EW_meridian",
                               "p_valueSpe" : "p_valueSpearman",
                               "medians_di": "medians_diff_ANPP",
                               "medians__1" : "medians_diff_slope_ANPP",
                               "median_ANP" : "median_ANPP_change_as_perc",
                               "state_majo" : "state_majority_area"}, 
                      inplace=True)

### There might be a confusion using the names I have used
### medians_diff_ANPP refers to median of ANPP in the first decare and last decade to measure
### growth in ANPP over 40 years. But, later we did mean and median differences
### for weather variables before and after breakpoints. So, I will drop these here.
Albers_SF_west.drop(columns=[
                             "medians_diff_ANPP", "medians_diff_slope_ANPP",
                             "median_ANPP_change_as_perc", "state_majority_area", 
                             "p_Spearman", "centroid", 'trend_yue', 'p_yue',
                             'trend_rao', 'p_rao', 'Tau', 'state_1', 'state_2',
                             'hucsgree_4', 'bps_code', 'bps_model', 'bps_name', 'groupveg',
                             'Spearman', 'sens_inter', 'sens_slope', 'trend'
                             ],
                    inplace=True)

Albers_SF_west.head(2)

# %%
beforeAfterSlopes_df.head(2)

# %% [markdown]
# #### Subset to the shapefile to include only locations that have had break points in them

# %%
## Some FIDs did not have breakpoint in their ANPP time series. 
#S subset Albers_SF_west to those that did:
print (Albers_SF_west.shape)
Albers_SF_west = Albers_SF_west[Albers_SF_west["fid"].isin(list(beforeAfterSlopes_df["fid"].unique()))]
Albers_SF_west.reset_index(drop=True, inplace=True)
print (Albers_SF_west.shape)

Albers_SF_west.head(2)

# %%
# cols_ = ["fid", "temp_slope_diff", "temp_slope_ratio", "precip_slope_diff", "precip_slope_ratio"]
_sign_category_cols = [x for x in beforeAfterSlopes_df if ("_sign_category" in x)]
cols_ = ["fid"] + _sign_category_cols

Albers_SF_west = pd.merge(Albers_SF_west, beforeAfterSlopes_df[cols_], how="left", on="fid")
Albers_SF_west.head(2)

# %%
tick_legend_FontSize = 6
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
print (len(_sign_category_cols))
_sign_category_cols[:4]

# %%
# define colors for consistency

category_colors = {'P_P': '#b2182b',
                   'P_N': '#ef8a62',
                   'P_Z': '#fddbc7',
                   'Z_N': '#c7e9c0',
                   'N_N': '#2166ac',
                   'Z_P': '#b2e2e2',
                   'N_P': '#762a83',
                   'N_Z': '#e7d4e8',
                   'Z_Z': '#f0f0f0'}

import matplotlib.patches as mpatches

# %%
y_var = _sign_category_cols[0]

fig, ax = plt.subplots(1, 1)
ax.set_xticks([]); ax.set_yticks([])

rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

# Map colors
Albers_SF_west['color'] = Albers_SF_west[y_var].map(category_colors)
cent_plt = Albers_SF_west.plot(color=Albers_SF_west['color'], ax=ax, legend=True)

# Add the legend and position it
legend_patches = [mpatches.Patch(color=color, label=cat) for cat, color in category_colors.items()]
leg = ax.legend(handles=legend_patches, title=None, loc='center left', bbox_to_anchor=(1, 0.5))

plt.title(f"{y_var}", fontdict=fontdict_bold);
plt.tight_layout()
ax.set_aspect('equal', adjustable='box')

file_name = "/Users/hn/Desktop/" + f"{y_var}_ANPPBP1.png"
plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)

# del(cent_plt, cax, cbar1)

# %%

# %%
for y_var in _sign_category_cols:
    fig, ax = plt.subplots(1, 1)
    ax.set_xticks([]); ax.set_yticks([])

    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['grey', 'white']))

    # Map colors
    Albers_SF_west['color'] = Albers_SF_west[y_var].map(category_colors)
    cent_plt = Albers_SF_west.plot(color=Albers_SF_west['color'], ax=ax, legend=True)

    # Add the legend and position it
    legend_patches = [mpatches.Patch(color=color, label=cat) for cat, color in category_colors.items()]
    leg = ax.legend(handles=legend_patches, title=None, loc='center left', bbox_to_anchor=(1, 0.5))    
    plt.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    
    if anpp_or_weather=="anpp":
        plt.title(f"ANPP {y_var}", fontdict=fontdict_bold);
        file_name = sign_switch_plot_dir + f"ANPP_{y_var}_ANPPBP1.png"
    else:
        plt.title(f"{y_var.replace('_sign_category', ' sign switch')}", fontdict=fontdict_bold);
        file_name = sign_switch_plot_dir + f"{y_var}_ANPPBP1.png"
    
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close(fig)
    del(cent_plt)

# %%

# %%
