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
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
custom_cmap_GrayW = ListedColormap(['grey', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')

fontdict_normal = {'family':'serif', 'weight':'normal'}
fontdict_bold   = {'family':'serif', 'weight':'bold'}
fontdict_bold_sup= {'family':'serif', 'fontweight':'bold'}
inset_axes_     = [0.1, 0.14, 0.45, 0.03]

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

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
breakpoint_TS_sen_dir = breakpoint_plot_base + "breakpoints_TS_sensSlope/"
# os.makedirs(breakpoint_TS_sen_dir, exist_ok=True)
breakpoints_dir = rangeland_bio_data + "breakpoints/"

# %%
filename = bio_reOrganized + f"weather_detrended.sav"

weather_detrended = pd.read_pickle(filename)
weather_detrended.keys()
weather_detrended = weather_detrended["weather_detrended"]
weather_detrended.head(2)

# %%
filename = breakpoints_dir + "weather_variances_beforeAfter_BP1.sav"
variances_beforeAfter_BP1 = pd.read_pickle(filename)
variances_beforeAfter_BP1 = variances_beforeAfter_BP1['weather_variances_beforeAfter_BP1']
variances_beforeAfter_BP1.head(2)

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
temp_ys = ["temp", "temp_detrendLinReg", "temp_detrendDiff", "temp_detrendSens"]
prec_ys = ["prec", "prec_detrendLinReg", "prec_detrendDiff", "prec_detrendSens"]

for y_ in temp_ys:
    v_after = variances_beforeAfter_BP1['variance_after_' + y_].values
    v_before = variances_beforeAfter_BP1['variance_before_' + y_].values
    
    variances_beforeAfter_BP1["delta_variance_" + y_] = v_after - v_before
    variances_beforeAfter_BP1["ratio_variance_" + y_] = v_after / v_before

for y_ in prec_ys:
    v_after = variances_beforeAfter_BP1['variance_after_' + y_].values
    v_before = variances_beforeAfter_BP1['variance_before_' + y_].values
    
    variances_beforeAfter_BP1["delta_variance_" + y_] = v_after - v_before
    variances_beforeAfter_BP1["ratio_variance_" + y_] = v_after / v_before

variances_beforeAfter_BP1.head(2)


# %%
delta_ratio_cols = [x for x in variances_beforeAfter_BP1.columns if (("delta" in x) or ('ratio' in x) )]
delta_ratio_cols

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

Albers_SF_west.head(2)

# %%
L_ = list(variances_beforeAfter_BP1["fid"].unique())
Albers_SF_west = Albers_SF_west[Albers_SF_west["fid"].isin(L_)]

# %%
cols_ = ['fid'] + delta_ratio_cols

Albers_SF_west = pd.merge(Albers_SF_west, variances_beforeAfter_BP1[cols_], how="left", on="fid")
Albers_SF_west.head(2)

# %%
tick_legend_FontSize = 12
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
delta_ratio_cols

# %%
y_var = delta_ratio_cols[1]
y_var

# %% [markdown]
# ## Ratio and Delta side by side

# %%
temp_patterns = ['temp', 'detrendLinReg', 'detrendDiff', 'detrendSens']
prec_patterns = ['prec', 'detrendLinReg', 'detrendDiff', 'detrendSens']
a_pattern = temp_patterns[0]

# %% [markdown]
# ## First Temp

# %%
# %%time

out_dir = bio_plots + "variances_BABP1_temp/"
os.makedirs(out_dir, exist_ok=True)

curr_out_dir = out_dir + "ratio_and_delta_sideByside/"
os.makedirs(curr_out_dir, exist_ok=True)
curr_out_dir

for a_pattern in temp_patterns:
    y_vars = sorted([x for x in delta_ratio_cols if a_pattern in x])

    fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})
    ax[0].set_xticks([]); ax[0].set_yticks([]);
    ax[1].set_xticks([]); ax[1].set_yticks([]);

    # Plotting the data with original colormap (don't change the color normalization)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=custom_cmap_GrayW)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=custom_cmap_GrayW)

    df0 = Albers_SF_west.copy()
    df0.dropna(subset=[y_vars[0]], inplace=True)

    df1 = Albers_SF_west.copy()
    df1.dropna(subset=[y_vars[1]], inplace=True)

    ############
    min_max0 = max(np.abs(df0[y_vars[0]].min()), np.abs(df0[y_vars[0]].max()))
    min_max1 = max(np.abs(df1[y_vars[1]].min()), np.abs(df1[y_vars[1]].max()))

    norm0 = Normalize(vmin= -min_max0, vmax=min_max0, clip=True)
    norm1 = Normalize(vmin= -min_max1, vmax=min_max1, clip=True)

    cent_plt0 = df0.plot(ax=ax[0], column=y_vars[0], legend=False, cmap='seismic', norm=norm0)
    cent_plt1 = df1.plot(ax=ax[1], column=y_vars[1], legend=False, cmap='seismic', norm=norm1)

    cax0 = ax[0].inset_axes([0.08, 0.18, 0.45, 0.03])
    cax1 = ax[1].inset_axes([0.08, 0.18, 0.45, 0.03])

    cbar0 = fig.colorbar(cent_plt0.collections[1], ax=ax[0], norm=norm0, cax=cax0, 
                         orientation='horizontal', shrink=0.3)

    cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax[1], norm=norm1, cax=cax1,
                         orientation='horizontal', shrink=0.3)

    cbar0.set_label(r'$\Delta(\sigma^2_{BP1})$', labelpad=1, fontdict=fontdict_normal);
    cbar1.set_label(r'ratio$(\sigma^2_{BP1})$' , labelpad=1, fontdict=fontdict_normal);

    L = fr"$\sigma^2$-delta and -ratio after and before ANPP-BP1 ({a_pattern})"
    fig.suptitle(L, y=0.82,  fontdict=fontdict_bold_sup);

    file_name = curr_out_dir + "variance_" + a_pattern + "_BP1.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()
    del(cent_plt0, cent_plt1, cax0, cax1, cbar0, cbar1, norm0, norm1, min_max0, min_max1, df0, df1)

# %% [markdown]
# #### one variable at a time. side by side. outliers separated.

# %%
# %%time

curr_out_dir = out_dir + "ratio_or_delta_outliers/"
os.makedirs(curr_out_dir, exist_ok=True)
curr_out_dir

for y_var in delta_ratio_cols:
    print (y_var)
    fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})
    ax[0].set_xticks([]); ax[0].set_yticks([]);
    ax[1].set_xticks([]); ax[1].set_yticks([]);

    # Plotting the data with original colormap (don't change the color normalization)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=custom_cmap_GrayW)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=custom_cmap_GrayW)

    df = Albers_SF_west.copy()
    df.dropna(subset=[y_var], inplace=True)

    perc_ = 5 / 100
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

    cax0 = ax[0].inset_axes([0.08, 0.18, 0.45, 0.03])
    cax1 = ax[1].inset_axes([0.08, 0.18, 0.45, 0.03])

    cbar0 = fig.colorbar(cent_plt0.collections[1], ax=ax[0], norm=norm0, cax=cax0, 
                         orientation='horizontal', shrink=0.3)

    cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax[1], norm=norm1, cax=cax1, 
                         orientation='horizontal', shrink=0.3)

    if "delta" in y_var:
        L_ = r'$\Delta(\sigma^2_{BP1})$'
        cbar0.set_label(L_, labelpad=1, fontdict=fontdict_normal);
        cbar1.set_label(L_, labelpad=1, fontdict=fontdict_normal);
        pre_title = "diff."
    elif "ratio" in y_var:
        L_ = r'$ratio(\sigma^2_{BP1})$'
        cbar0.set_label(L_, labelpad=1, fontdict=fontdict_normal);
        cbar1.set_label(L_, labelpad=1, fontdict=fontdict_normal);
        pre_title = "ratio"

    t_ = y_var.split("_")[-1]
    # plt.title(f"ACF1 {pre_title} after and before BP1", fontdict={'family':'serif', 'weight':'bold'});    
    # fig.suptitle(f"\sigma^2-{pre_title} after and before BP1 ({t_})", y=0.82, fontdict={'family':'serif'});
    L_ = fr"$\sigma^2$-{pre_title} after and before ANPP-BP1 ({t_})"
    fig.suptitle(L_, y=0.82, fontdict=fontdict_bold_sup);
    
    file_name = curr_out_dir + y_var + "_BP1.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()
    del(cent_plt0, cent_plt1, cax0, cax1, cbar0, cbar1, norm0, norm1, min_max0, min_max1,
        filtered_outside, filtered_between)

# %%

# %% [markdown]
# ## Repeat for Prec

# %%
# %%time

out_dir = bio_plots + "variances_BABP1_prec/"
os.makedirs(out_dir, exist_ok=True)

curr_out_dir = out_dir + "ratio_and_delta_sideByside/"
os.makedirs(curr_out_dir, exist_ok=True)
curr_out_dir

for a_pattern in prec_patterns:
    y_vars = sorted([x for x in delta_ratio_cols if a_pattern in x])

    fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})
    ax[0].set_xticks([]); ax[0].set_yticks([]);
    ax[1].set_xticks([]); ax[1].set_yticks([]);

    # Plotting the data with original colormap (don't change the color normalization)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=custom_cmap_GrayW)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=custom_cmap_GrayW)

    df0 = Albers_SF_west.copy()
    df0.dropna(subset=[y_vars[0]], inplace=True)

    df1 = Albers_SF_west.copy()
    df1.dropna(subset=[y_vars[1]], inplace=True)

    ############
    min_max0 = max(np.abs(df0[y_vars[0]].min()), np.abs(df0[y_vars[0]].max()))
    min_max1 = max(np.abs(df1[y_vars[1]].min()), np.abs(df1[y_vars[1]].max()))

    norm0 = Normalize(vmin= -min_max0, vmax=min_max0, clip=True)
    norm1 = Normalize(vmin= -min_max1, vmax=min_max1, clip=True)

    cent_plt0 = df0.plot(ax=ax[0], column=y_vars[0], legend=False, cmap='seismic', norm=norm0)
    cent_plt1 = df1.plot(ax=ax[1], column=y_vars[1], legend=False, cmap='seismic', norm=norm1)

    cax0 = ax[0].inset_axes([0.08, 0.18, 0.45, 0.03])
    cax1 = ax[1].inset_axes([0.08, 0.18, 0.45, 0.03])

    cbar0 = fig.colorbar(cent_plt0.collections[1], ax=ax[0], norm=norm0, cax=cax0, 
                         orientation='horizontal', shrink=0.3)

    cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax[1], norm=norm1, cax=cax1,
                         orientation='horizontal', shrink=0.3)

    cbar0.set_label(r'$\Delta(\sigma^2_{BP1})$', labelpad=1, fontdict=fontdict_normal);
    cbar1.set_label(r'ratio$(\sigma^2_{BP1})$' , labelpad=1, fontdict=fontdict_normal);

    L = fr"$\sigma^2$-delta and -ratio after and before ANPP-BP1 ({a_pattern})"
    fig.suptitle(L, y=0.82,  fontdict=fontdict_bold_sup);

    file_name = curr_out_dir + "variance_" + a_pattern + "_BP1.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()
    del(cent_plt0, cent_plt1, cax0, cax1, cbar0, cbar1, norm0, norm1, min_max0, min_max1, df0, df1)

# %%

# %%
# %%time

curr_out_dir = out_dir + "ratio_or_delta_outliers/"
os.makedirs(curr_out_dir, exist_ok=True)
curr_out_dir

for y_var in delta_ratio_cols:
    print (y_var)
    fig, ax = plt.subplots(1, 2, dpi=map_dpi_, gridspec_kw={'hspace': 0.02, 'wspace': 0.05})
    ax[0].set_xticks([]); ax[0].set_yticks([]);
    ax[1].set_xticks([]); ax[1].set_yticks([]);

    # Plotting the data with original colormap (don't change the color normalization)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[0], col="EW_meridian", cmap_=custom_cmap_GrayW)
    rpc.plot_SF(SF=visframe_mainLand_west, ax_=ax[1], col="EW_meridian", cmap_=custom_cmap_GrayW)

    df = Albers_SF_west.copy()
    df.dropna(subset=[y_var], inplace=True)

    perc_ = 5 / 100
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

    cax0 = ax[0].inset_axes([0.08, 0.18, 0.45, 0.03])
    cax1 = ax[1].inset_axes([0.08, 0.18, 0.45, 0.03])

    cbar0 = fig.colorbar(cent_plt0.collections[1], ax=ax[0], norm=norm0, cax=cax0, 
                         orientation='horizontal', shrink=0.3)

    cbar1 = fig.colorbar(cent_plt1.collections[1], ax=ax[1], norm=norm1, cax=cax1, 
                         orientation='horizontal', shrink=0.3)

    if "delta" in y_var:
        L_ = r'$\Delta(\sigma^2_{BP1})$'
        cbar0.set_label(L_, labelpad=1, fontdict=fontdict_normal);
        cbar1.set_label(L_, labelpad=1, fontdict=fontdict_normal);
        pre_title = "diff."
    elif "ratio" in y_var:
        L_ = r'$ratio(\sigma^2_{BP1})$'
        cbar0.set_label(L_, labelpad=1, fontdict=fontdict_normal);
        cbar1.set_label(L_, labelpad=1, fontdict=fontdict_normal);
        pre_title = "ratio"

    t_ = y_var.split("_")[-1]
    L_ = fr"$\sigma^2$-{pre_title} after and before ANPP-BP1 ({t_})"
    fig.suptitle(L_, y=0.82, fontdict=fontdict_bold_sup);
    
    file_name = curr_out_dir + y_var + "_BP1.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=map_dpi_)
    plt.close()
    del(cent_plt0, cent_plt1, cax0, cax1, cbar0, cbar1, norm0, norm1, min_max0, min_max1,
        filtered_outside, filtered_between)
