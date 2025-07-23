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
import importlib;
importlib.reload(rc);
importlib.reload(rpc);

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

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

breakpoint_plot_base = bio_plots + "breakpoints/"
os.makedirs(breakpoint_plot_base, exist_ok=True)

breakpoint_TS_dir = bio_plots + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)


G_breakpoint_TS_dir = breakpoint_TS_dir + "/greening/"
B_breakpoint_TS_dir = breakpoint_TS_dir + "/browning/"
noTrend_breakpoint_TS_dir = breakpoint_TS_dir + "/notrend/"

os.makedirs(G_breakpoint_TS_dir, exist_ok=True)
os.makedirs(B_breakpoint_TS_dir, exist_ok=True)
os.makedirs(noTrend_breakpoint_TS_dir, exist_ok=True)

# %%
ACF_plot_base = bio_plots + "ACF1/"
os.makedirs(ACF_plot_base, exist_ok=True)

# %%
filename = bio_reOrganized + f"bpszone_ANPP_no2012_detrended.sav"
ANPP = pd.read_pickle(filename)
ANPP = ANPP["ANPP_no2012_detrended"]
ANPP.head(2)

# %%
print (ANPP.year.min())
print (ANPP.year.max())

# %%
# Example usage:
data = [10, 12, 15, 13, 16, 18, 17]
time_series = pd.Series(data)

lag_1_acf = time_series.autocorr(lag=1)
print(f"Lag-1 Autocorrelation: {lag_1_acf:.2f}")

# %%
[x for x in ANPP.columns if "anpp" in x]

# %%
ANPP_ACF1 = ANPP.groupby('fid')['mean_lb_per_acr'].apply(lambda x: x.autocorr(lag=1))
ANPP_ACF1 = ANPP_ACF1.reset_index(name='mean_lb_per_acr_ACF1')
ANPP_ACF1.head(5)

# %%
ANPP_ACF1_detrendLinReg = ANPP.groupby('fid')['anpp_detrendLinReg'].apply(lambda x: x.autocorr(lag=1))
ANPP_ACF1_detrendLinReg = ANPP_ACF1_detrendLinReg.reset_index(name='anpp_detrendLinReg_ACF1')
ANPP_ACF1_detrendLinReg.head(2)

# %%
ANPP_ACF1_detrendSens = ANPP.groupby('fid')['anpp_detrendSens'].apply(lambda x: x.autocorr(lag=1))
ANPP_ACF1_detrendSens = ANPP_ACF1_detrendSens.reset_index(name='anpp_detrendSens_ACF1')
ANPP_ACF1_detrendSens.head(2)

# %%
ANPP_ACF1 = pd.merge(ANPP_ACF1, ANPP_ACF1_detrendSens, how="left", on="fid")
ANPP_ACF1 = pd.merge(ANPP_ACF1, ANPP_ACF1_detrendLinReg, how="left", on="fid")
ANPP_ACF1.head(2)

# %%

# %%
ANPP_ACF1_statLib = ANPP.groupby('fid')['mean_lb_per_acr'].apply(lambda x: 
                                        acf(x.values, nlags=1, fft=False)[1] if len(x.dropna()) > 1 else np.nan)

ANPP_ACF1_statLib = ANPP_ACF1_statLib.reset_index(name='mean_lb_per_acr_ACF1_statLib')
ANPP_ACF1_statLib.head(5)

# %% [markdown]
# ### Make a spatial Map

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
# SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %%
SF_west = pd.merge(SF_west, ANPP_ACF1, on="fid", how="left")
SF_west.head(2)

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
y_var = "mean_lb_per_acr_ACF1"

fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([]); ax.set_yticks([])
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['white', 'green']))

min_col_ = np.abs(SF_west[y_var].min())
max_col_ = np.abs(SF_west[y_var].max())
cc_ = max(min_col_, max_col_)
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);

# cmap = bwr or use 'seismic'
# cent_plt = SF_west["centroid"].plot(ax=ax, c=SF_west[y_var], #cmap='seismic',
#                                     norm=norm_col, markersize=0.1) 
cent_plt = SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm_col)

plt.tight_layout()

############# color bar
cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
# cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.5)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label(r"ACF1", labelpad=2)

#############
plt.title('ACF1 for ANPP (1984-2023, no 2012)', y=0.98);

# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981)
file_name = ACF_plot_base + "ANPP_ACF1_zeroWhite_poly.png" # ANPP_ACF1_zeroWhite or ANPP_ACF1
plt.savefig(file_name, bbox_inches='tight', dpi=300)

del(cent_plt, cax, cbar1)

# %%
[x for x in sorted(SF_west.columns) if "anpp" in x]

# %%
y_var = "anpp_detrendLinReg_ACF1"

fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([]); ax.set_yticks([])
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['white', 'green']))

min_col_ = np.abs(SF_west[y_var].min())
max_col_ = np.abs(SF_west[y_var].max())
cc_ = max(min_col_, max_col_)
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);
cent_plt = SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm_col)
plt.tight_layout()

############# color bar
cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label(r"ACF1", labelpad=2)

#############
plt.title('ACF1 for ANPP detrendLinReg (1984-2023, no 2012)', y=0.98);

file_name = ACF_plot_base + "ANPP_detrendLinReg_ACF1.png"
plt.savefig(file_name, bbox_inches='tight', dpi=300)

del(cent_plt, cax, cbar1)

# %%

# %%
y_var = "anpp_detrendSens_ACF1"

fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([]); ax.set_yticks([])
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['white', 'green']))

min_col_ = np.abs(SF_west[y_var].min())
max_col_ = np.abs(SF_west[y_var].max())
cc_ = max(min_col_, max_col_)
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);
cent_plt = SF_west.plot(column=y_var, ax=ax, legend=False, cmap='seismic', norm=norm_col)
plt.tight_layout()

############# color bar
cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label(r"ACF1", labelpad=2)

#############
plt.title('ACF1 for ANPP detrendSens (1984-2023, no 2012)', y=0.98);

file_name = ACF_plot_base + "ANPP_detrendSens_ACF1.png"
plt.savefig(file_name, bbox_inches='tight', dpi=300)

del(cent_plt, cax, cbar1)

# %%

# %%
fig, ax = plt.subplots(1, 1, dpi=map_dpi_) # figsize=(2, 2)
ax.set_xticks([]); ax.set_yticks([])
rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['white', 'green']))

min_col_ = np.abs(SF_west[y_var].min())
max_col_ = np.abs(SF_west[y_var].max())
cc_ = max(min_col_, max_col_)
norm_col = Normalize(vmin=-cc_, vmax=cc_, clip=True);

# cmap = bwr or use 'seismic'
# cent_plt = SF_west["centroid"].plot(ax=ax, c=SF_west[y_var], #cmap='seismic',
#                                     norm=norm_col, markersize=0.1) 
cent_plt = SF_west.plot(column=y_var, ax=ax, legend=False, norm=norm_col)

plt.tight_layout()

############# color bar
cax = ax.inset_axes([0.03, 0.18, 0.5, 0.03])
# cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.5)
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='horizontal', shrink=0.3, cax=cax)
cbar1.set_label(r"ACF1", labelpad=2)

#############
plt.title('ACF1 for ANPP (1984-2023, no 2012)', y=0.98);

# fig.subplots_adjust(top=0.91, bottom=0.01, left=0.01, right=0.981)
file_name = ACF_plot_base + "ANPP_ACF1.png" # ANPP_ACF1_zeroWhite or ANPP_ACF1
plt.savefig(file_name, bbox_inches='tight', dpi=300)

del(cent_plt, cax, cbar1)

# %%
SF_west.head(2)

# %%
SF_west[SF_west["mean_lb_per_acr_lag1_autocorr"] < 0]["mean_lb_per_acr_lag1_autocorr"].max()

# %%
SF_west[SF_west["mean_lb_per_acr_lag1_autocorr"] < 0]["mean_lb_per_acr_lag1_autocorr"].min()

# %%
# Sample data for a single fid
df = pd.DataFrame({'mean_lb_per_acr': [100, 1, 3, 4, 10, 0]})

# Rolling autocorrelation function using pandas' autocorr
def rolling_autocorr(series, lag=1):
    if len(series.dropna()) ## series.nunique() > 1:  # autocorr undefined for constant values. but what if it is 1 2 1 1 1
        return series.autocorr(lag=lag)
    else:
        return np.nan

# Apply rolling autocorrelation with window=3 and lag=1
df['autocorr_lag1'] = df['mean_lb_per_acr'].rolling(window=3).apply(lambda x: rolling_autocorr(x, lag=1), 
                                                                    raw=False)

# %%
df

# %%
df['mean_lb_per_acr'][1:4].autocorr(lag=1)

# %%
df['mean_lb_per_acr'][1:4]

# %%

# %%

# %%

# %%
