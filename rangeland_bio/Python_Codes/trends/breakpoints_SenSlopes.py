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
# # breakpoints in R.
#
# Lets do the breakpoints in R. Here we can analyze and plot things.

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
dpi_, map_dpi_=300, 900
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

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

breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)


G_breakpoint_TS_dir = breakpoint_TS_dir + "/greening/"
B_breakpoint_TS_dir = breakpoint_TS_dir + "/browning/"
noTrend_breakpoint_TS_dir = breakpoint_TS_dir + "/notrend/"

os.makedirs(G_breakpoint_TS_dir, exist_ok=True)
os.makedirs(B_breakpoint_TS_dir, exist_ok=True)
os.makedirs(noTrend_breakpoint_TS_dir, exist_ok=True)

# %%
breakpoint_TS_sen_dir = breakpoint_plot_base + "breakpoints_TS_sensSlope/"
os.makedirs(breakpoint_TS_sen_dir, exist_ok=True)


# %%
breakpoints_dir = rangeland_bio_data + "breakpoints/"

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

# %%
ANPP_breaks = pd.read_csv(breakpoints_dir + "ANPP_break_points.csv")
ANPP_breaks = ANPP_breaks[ANPP_breaks["breakpoint_count"]>0]
ANPP_breaks.reset_index(drop=True, inplace=True)
ANPP_breaks.head(2)

# %%
bp_cols = ANPP_breaks['breakpoint_years'].str.split('_', expand=True)
bp_cols.columns = [f'BP_{i+1}' for i in range(bp_cols.shape[1])]
bp_cols = bp_cols.apply(pd.to_numeric, errors='coerce')
ANPP_breaks = pd.concat([ANPP_breaks, bp_cols], axis=1)
ANPP_breaks.head(2)

# %%
ANPP['year'] = ANPP['year'].astype(int)

# %%
print (ANPP_breaks.shape)
ANPP_breaks['BP_1'] = ANPP_breaks['BP_1'].dropna().astype(int)
print (ANPP_breaks.shape)

# %%
# %%time
# Iterate through each row in ANPP_breaks
results = []

for _, row in ANPP_breaks.iterrows():
    fid = row['fid']
    bp_year = row['BP_1']
    
    # Filter ANPP by current fid
    subset = ANPP[ANPP['fid'] == fid]
    
    # Separate before and after BP_1
    before = subset[subset['year'] < bp_year]['mean_lb_per_acr']
    after = subset[subset['year'] >= bp_year]['mean_lb_per_acr']
    
    # Apply Mann-Kendall test if sufficient data
    result = {
        'fid': fid,
        'BP_1': bp_year,
        'n_before': len(before),
        'n_after': len(after),
        'slope_before': None,
        'slope_after': None,
        'intercept_before': None,
        'intercept_after': None,
        'trend_before': None,
        'trend_after': None
    }
    
    if len(before) >= 3:
        trend, _, _, _, _, _, _, slope, intercept = mk.original_test(before)
        result['slope_before'] = slope.round(2)
        result['trend_before'] = trend
        result['intercept_before'] = intercept.round(2)
    
    if len(after) >= 3:
        trend, _, _, _, _, _, _, slope, intercept = mk.original_test(after)
        result['slope_after'] = slope.round(2)
        result['trend_after'] = trend
        result['intercept_after'] = intercept.round(2)
    
    results.append(result)

# Create results DataFrame
slope_results = pd.DataFrame(results)

# %%
fid_1_npp = ANPP[ANPP["fid"] == 1]
fid_1_npp.head(2)

# %%
fid_1_npp_before = fid_1_npp[fid_1_npp["year"] < 1990]
fid_1_npp_after  = fid_1_npp[fid_1_npp["year"] >= 1990]

# %%
mk.original_test(fid_1_npp_before["mean_lb_per_acr"].to_numpy())

# %%
trend, _, _, _, _, _, _, slope, intercept = mk.original_test(fid_1_npp_before["mean_lb_per_acr"].to_numpy())
print (f"{trend = }, {slope = }")

# %%
trend, _, _, _, _, _, _, slope, intercept = mk.original_test(fid_1_npp_after["mean_lb_per_acr"].to_numpy())

# %% [markdown]
# ### plot TS of FID=1

# %%
y_var = "mean_lb_per_acr"

# %%
tick_legend_FontSize = 6
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 5,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          'axes.linewidth' : .05,
          'xtick.major.width': 0.1,
          'ytick.major.width': 0.1,
          'xtick.major.size': 2,
          'ytick.major.size': 2,
         }

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
# axes.axline((ress_["BP_1"].item(), ress_["intercept_after"].item()), 
#             slope=ress_["slope_after"].item(), 
#             color='k', label='after')


# %%
a_fid = 1
fig, axes = plt.subplots(1, 1, figsize=(4, 2), sharey=False, sharex=False, dpi=dpi_)
axes.grid(axis='y', alpha=0.2, zorder=0);
###################################
df = ANPP[ANPP["fid"] == a_fid]
ress_ = slope_results[slope_results["fid"] == a_fid]

ymin, ymax = df[y_var].min(), df[y_var].max()
y_ave = 0.5 * (ymin+ymax)

axes.plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=3);

break_yrs = ANPP_breaks[ANPP_breaks["fid"] == a_fid]["breakpoint_years"].iloc[0]
if not (pd.isna(break_yrs)):
    break_yrs = break_yrs.split("_")
    break_yrs = [int(x) for x in break_yrs]
    for brk_yr in break_yrs:
        plt.axvline(x=brk_yr, color='r', linestyle='--', linewidth=1) # , label=f"{brk_yr}"

x_break = break_yrs[0]
x_start, x_end = 1984, 2023

# Split data before and after breakpoint
df_before = df[df["year"] < x_break]
df_after = df[df["year"] >= x_break]

# Compute slope values
slope_before = ress_["slope_before"].item()
slope_after = ress_["slope_after"].item()

# Compute (mean_x, mean_y) for each segment
x_mean_before = df_before["year"].mean()
y_mean_before = df_before[y_var].mean()

x_mean_after = df_after["year"].mean()
y_mean_after = df_after[y_var].mean()

# Compute intercepts so the lines pass through the mean point
intercept_before = y_mean_before - slope_before * x_mean_before
intercept_after = y_mean_after - slope_after * x_mean_after

# Generate x/y for plotting
x_before = [x_start, x_break]
y_before = [slope_before * x + intercept_before for x in x_before]

x_after = [x_break, x_end]
y_after = [slope_after * x + intercept_after for x in x_after]

# Plot lines
axes.plot(x_before, y_before, color='k', linestyle='--', linewidth=1,
          label=f'slope = {slope_before} ({ress_["trend_before"].item()})')

axes.plot(x_after, y_after, color='green', linestyle='--', linewidth=1,
          label=f'slope = {slope_after} ({ress_["trend_after"].item()})')

axes.set_ylim(ymin, ymax)

axes.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
axes.set_xlabel('year') #, fontsize=14
axes.set_xticks(df['year'].iloc[::2]);
axes.tick_params(axis='x', rotation=45)
plt.legend();

plt.tight_layout()

file_name = breakpoint_TS_sen_dir + f"fid_{a_fid}_BP_and_SensSlopes.pdf"
plt.savefig(file_name, dpi=300)

# %%
slope_results.head(2)

# %%
a_fid = 1
y_var = "mean_lb_per_acr"  # Define the y-variable name

fig, axes = plt.subplots(1, 1, figsize=(4, 2), sharey=False, sharex=False, dpi=dpi_)
axes.grid(axis='y', alpha=0.2, zorder=0)

# Filter data for the current fid
df = ANPP[ANPP["fid"] == a_fid]
ress_ = slope_results[slope_results["fid"] == a_fid]

# Get y-axis limits
ymin, ymax = df[y_var].min(), df[y_var].max()
axes.set_ylim(ymin, ymax)

# Plot actual data
axes.plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=3)

# Plot vertical lines for breakpoints (if available)
break_yrs = ANPP_breaks[ANPP_breaks["fid"] == a_fid]["breakpoint_years"].iloc[0]
if not pd.isna(break_yrs):
    break_yrs = [int(x) for x in break_yrs.split("_")]
    for brk_yr in break_yrs:
        axes.axvline(x=brk_yr, color='r', linestyle='--', linewidth=1)

# Define time range
x_start, x_end = 1984, 2023
x_break = break_yrs[0]

# Split data before and after breakpoint
df_before = df[df["year"] < x_break]
df_after = df[df["year"] >= x_break]

# Get slope values
slope_before = ress_["slope_before"].item()
slope_after = ress_["slope_after"].item()

# Get y-value at the breakpoint year
y_break = df[df["year"] == x_break][y_var].mean()

# Compute intercepts so both lines pass through the breakpoint
intercept_before = y_break - slope_before * x_break
intercept_after = y_break - slope_after * x_break

# Generate x/y for plotting
x_before = [x_start, x_break]
y_before = [slope_before * x + intercept_before for x in x_before]

x_after = [x_break, x_end]
y_after = [slope_after * x + intercept_after for x in x_after]

# Plot trend lines
axes.plot(x_before, y_before, color='k', linestyle='--', linewidth=1,
          label=f'slope = {slope_before:.2f} ({ress_["trend_before"].item()})')
axes.plot(x_after, y_after, color='green', linestyle='--', linewidth=1,
          label=f'slope = {slope_after:.2f} ({ress_["trend_after"].item()})')

# Formatting
axes.set_ylabel(r'$\mu_{NPP}$ (lb/acr)')
axes.set_xlabel('year')
axes.set_xticks(df['year'].iloc[::2])
axes.tick_params(axis='x', rotation=45)
axes.legend()
plt.tight_layout()

# Save the plot
file_name = breakpoint_TS_sen_dir + f"fid_{a_fid}_BP_and_SensSlopes.pdf"
# plt.savefig(file_name, dpi=dpi_)
# plt.close()


# %%
a_fid = 1
df = ANPP[ANPP["fid"] == a_fid]
x_break = ress_["BP_1"].item()

# Split data before and after breakpoint
df_before = df[df["year"] < x_break]
df_after = df[df["year"] >= x_break]

# %%
mk.original_test(df_before[y_var])

# %%
mk.original_test(df_after[y_var])

# %%
print (f"{slope_before = }, {intercept_before = }")
print (f"{slope_after = }, {intercept_after = }")

# %%

# %%
filename = breakpoints_dir + "sensSlope_beforeAfter_BP1.sav"

export_ = {
    "sensSlope_beforeAfter_BP1": slope_results,
    "source_code": "breakpoints_SenSlopes",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, 'wb'))

# %%
filename = breakpoints_dir + "sensSlope_beforeAfter_BP1.csv"
slope_results.to_csv(filename, index=False)

# %%
# %%time
## bad 2012
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %%
slope_results.head(2)

# %%
backup = slope_results.copy()

# %%
slope_results = backup.copy()

# %%
slope_results.loc[slope_results['trend_before'] =="increasing", 'trend_before'] = "G"
slope_results.loc[slope_results['trend_before'] =="decreasing", 'trend_before'] = "B"
slope_results.loc[slope_results['trend_before'] =="no trend", 'trend_before'] = "NT"

slope_results.loc[slope_results['trend_after'] =="increasing", 'trend_after'] = "G"
slope_results.loc[slope_results['trend_after'] =="decreasing", 'trend_after'] = "B"
slope_results.loc[slope_results['trend_after'] =="no trend", 'trend_after'] = "NT"

# %%

# %%
slope_results["trendShift"] = slope_results["trend_before"] + "_" + slope_results["trend_after"]
slope_results.head(2)

# %%
df = slope_results.groupby('trendShift').size().reset_index(name='count')
print (df.sort_values("trendShift",))

# %% [markdown]
# ## Do these stats only for greening locations

# %%
green_fids = list(SF_west[SF_west["trend"] == "increasing"]["fid"])

# %%
slope_results_g = slope_results[slope_results["fid"].isin(green_fids)].copy()

# %%
df = slope_results_g.groupby('trendShift').size().reset_index(name='count')
print (df.sort_values("trendShift",))

# %%
