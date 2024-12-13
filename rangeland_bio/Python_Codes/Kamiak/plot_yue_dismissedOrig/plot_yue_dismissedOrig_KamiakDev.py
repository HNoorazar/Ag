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
import os, os.path, pickle, sys
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

# %%
# Major ticks every 5, minor ticks every 1
major_ticks = np.arange(1984, 2024, 5)
minor_ticks = np.arange(1984, 2024, 1)
y_var = "mean_lb_per_acr"

dpi_ = 300

# %%
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
bio_reOrganized = rangeland_bio_data + "reOrganized/"

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

yue_plots = bio_plots + "yue/"
os.makedirs(yue_plots, exist_ok=True)

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
len(ANPP_MK_df["EW_meridian"].unique())

# %%

# %%
Yue_green = ANPP_MK_df[["fid", "trend", "trend_yue", "p_yue"]].copy()
print (len(Yue_green))
Yue_green = Yue_green[Yue_green["trend_yue"] == "increasing"]
print (len(Yue_green))
Yue_green = Yue_green[Yue_green["p_yue"]<0.01].copy()
print (len(Yue_green))
Yue_green = Yue_green[Yue_green["trend"] != "increasing"]
print (len(Yue_green))

print (Yue_green["trend"].unique())
print (Yue_green["trend_yue"].unique())
Yue_green.head(2)

# %%
Yue_green = list(Yue_green["fid"].unique())
len(Yue_green)

# %%

# %%
tick_legend_FontSize = 12
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1.5,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
#          'axes.linewidth' : .05
         }

plt.rcParams.update(params)

# %%
a_fid = Yue_green[0]

fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
# axes.grid(which='major', alpha=0.5, axis="both")
axes.set_xticks(major_ticks)
axes.set_xticks(minor_ticks, minor=True)
axes.grid(which='minor', alpha=0.2, axis="x")
axes.grid(which='major', alpha=0.5, axis="x")

########## plot 1
df = ANPP[ANPP.fid == a_fid].copy()
axes.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes.scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

###
### Text 
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
Tau_   = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
state_ = ANPP_MK_df.loc[ANPP_MK_df.fid==a_fid, "state_majority_area"].item()

text_ = "trend: {}\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(trend_, slope_, Tau_, state_, a_fid)
y_txt = df[y_var].max()*.8
axes.text(1983, y_txt, text_, fontsize=tick_legend_FontSize*1.2);
####
axes.set_title("99% Yue Greening. Dismissed by Original MK.");
axes.set_ylabel(r'$\mu_{NPP}$ (lb/acr)');

# file_name = yue_plots + "Yue99PercCL_SensExtremes.pdf"
# plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %%

# %%
