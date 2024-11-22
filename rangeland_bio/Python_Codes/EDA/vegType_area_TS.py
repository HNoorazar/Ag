# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import warnings
warnings.filterwarnings("ignore")

import pickle
from datetime import datetime

import pandas as pd
from datetime import datetime
import os, os.path, pickle, sys

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas
# font = {"size": 10}
# matplotlib.rc("font", **font)

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from sklearn.model_selection import train_test_split
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

# %%
plt.rc("font", family="Helvetica")

dpi_ = 300
map_dpi_ = 200
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')
cmap_RYG = cm.get_cmap('RdYlGn')

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)
# ANPP.sort_values(by= ['fid', 'year'], inplace=True)
# ANPP.head(2)

# %% [markdown]
# # We need common FIDs on the west

# %%
FID_veg = pd.read_pickle(bio_reOrganized + "FID_veg.sav")
FIDs_weather_ANPP_common = FID_veg["FIDs_weather_ANPP_common"]
FIDs_weather_ANPP_common.head(2)

# %%
print (len(ANPP.fid.unique()))
ANPP = ANPP[ANPP["fid"].isin(list(FIDs_weather_ANPP_common.fid))]
print (len(ANPP.fid.unique()))

# %%
ANPP.head(2)

# %%
ANPP = pd.merge(ANPP, FIDs_weather_ANPP_common, on="fid", how="left")
ANPP["area_Km2"] = ANPP["area_sqMeter"] / 1000000
ANPP = ANPP[["year", "area_Km2", "groupveg"]]

ANPP.head(2)

# %%
ANPP_area_TS = ANPP.groupby(["year", "groupveg"]).sum().reset_index()
ANPP_area_TS.head(2)

# %%
veg_types = list(ANPP_area_TS["groupveg"].unique())
veg_types

# %%
tick_legend_FontSize = 15
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5,    'legend.handlelength': 2}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(len(veg_types), 1, figsize=(8, 8), sharex=True, sharey=False, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=100)
for ii in range(len(veg_types)):
    df = ANPP_area_TS[ANPP_area_TS["groupveg"] == veg_types[ii]]
    axes[ii].scatter(df.year, df["area_Km2"], marker='o', facecolors='r', edgecolors='r', s=15, zorder=2);
    axes[ii].plot(df.year, df["area_Km2"], linewidth=3, zorder=1, color="dodgerblue", label=veg_types[ii]);
    axes[ii].legend(loc='lower left');

fig.supylabel(f"area (Km$^2$)", x=0, fontsize=tick_legend_FontSize)
fig.suptitle(f'areas of different land covers', y=.94, fontsize=tick_legend_FontSize)
fig.subplots_adjust(top=0.91, bottom=0.04)
file_name = bio_plots + "area_TS.pdf"
plt.savefig(file_name, dpi=400)

# %%

# %%
fig, axes = plt.subplots(len(veg_types), 1, figsize=(8, 8), sharex=True, sharey=False, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=100)
for ii in range(len(veg_types)):
    df = ANPP_area_TS[ANPP_area_TS["groupveg"] == veg_types[ii]]
    df = df[df["year"] != 2012]
    axes[ii].scatter(df.year, df["area_Km2"], marker='o', facecolors='r', edgecolors='r', s=15, zorder=2);
    axes[ii].plot(df.year, df["area_Km2"], linewidth=3, zorder=1, color="dodgerblue");

    axes[ii].legend([veg_types[ii]], loc='lower right');
    
fig.supylabel(f"area (Km$^2$)", x=0);
fig.suptitle(f'area. 2012 removed', y=.94, fontsize=tick_legend_FontSize)
fig.subplots_adjust(top=0.91, bottom=0.04)
file_name = bio_plots + "area_TS_no_2012.pdf"
plt.savefig(file_name, dpi=400)

# %%

# %%
fig, axes = plt.subplots(len(veg_types), 1, figsize=(8, 8), sharex=True, sharey=False, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)
for ii in range(len(veg_types)):
    df = ANPP_area_TS[ANPP_area_TS["groupveg"] == veg_types[ii]]
    df = df[df["year"] != 2012]
    df = df[df["year"] > 1985]
    axes[ii].scatter(df.year, df["area_Km2"], marker='o', facecolors='r', edgecolors='r', s=15, zorder=2);
    axes[ii].plot(df.year, df["area_Km2"], linewidth=3, zorder=1, color="dodgerblue");
    axes[ii].legend([veg_types[ii]], loc='lower right');

fig.supylabel(f"area (Km$^2$)", x=0)
fig.suptitle(f'area. 2012 and 1984 removed', y=.94)
fig.subplots_adjust(top=0.91, bottom=0.04)
file_name = bio_plots + "area_TS_no_1984_2012.pdf"
plt.savefig(file_name, dpi=400)

# %%
ANPP_area_TS[ANPP_area_TS.year==1984]

# %%
ANPP_area_TS[ANPP_area_TS.year==2012]

# %%
ANPP_area_TS[ANPP_area_TS.year==1989]

# %%
ANPP_area_TS[ANPP_area_TS.year==2000]

# %%
ANPP.head(2)

# %%
## for each variety drop the lowest 3
fig, axes = plt.subplots(len(veg_types), 1, figsize=(8, 8), sharex=True, sharey=False, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=100)
for ii in range(len(veg_types)):
    df = ANPP_area_TS[ANPP_area_TS["groupveg"] == veg_types[ii]]
    df.drop(df["area_Km2"].idxmin(), inplace=True)
    df.drop(df["area_Km2"].idxmin(), inplace=True)
    df.drop(df["area_Km2"].idxmin(), inplace=True)
    axes[ii].scatter(df.year, df["area_Km2"], marker='o', facecolors='r', edgecolors='r', s=15, zorder=2);
    axes[ii].plot(df.year, df["area_Km2"], linewidth=3, zorder=1, color="dodgerblue");

    axes[ii].legend([veg_types[ii]], loc='lower right');
    
fig.supylabel(f"area (Km$^2$)", x=0);
fig.suptitle(f'area. lowest three years removed', y=.94, fontsize=tick_legend_FontSize)
fig.subplots_adjust(top=0.91, bottom=0.04)
file_name = bio_plots + "area_TS_veriety_droppedLowest3.pdf"
plt.savefig(file_name, dpi=400)

# %%

# %%
tick_legend_FontSize = 12
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5,    'legend.handlelength': 2}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

fig, axes = plt.subplots(1, 1, figsize=(8, 2), dpi=dpi_)

df = ANPP[['year', 'area_Km2']].groupby(["year"]).sum().reset_index()
axes.scatter(df.year, df["area_Km2"], marker='o', facecolors='r', edgecolors='r', s=15, zorder=2);
axes.plot(df.year, df["area_Km2"], linewidth=3, zorder=1, color="dodgerblue");

axes.set_ylabel(f"area (Km$^2$)");
plt.title("rangelands area", fontsize=tick_legend_FontSize*1.2);

file_name = bio_plots + "area_TS_allVegs.pdf"
plt.savefig(file_name, dpi=400)

# %%

# %%
