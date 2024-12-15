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
# # !pip3 install pymannkendall

# %%
import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import pymannkendall as mk

import statistics
import statsmodels.api as sm

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc


# %%
def plot_SF(SF, ax_, cmap_ = "Pastel1", col="EW_meridian"):
    SF.plot(column=col, ax=ax_, alpha=1, cmap=cmap_, edgecolor='k', legend=False, linewidth=0.1)


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

yue_plots = bio_plots + "yue/"
os.makedirs(yue_plots, exist_ok=True)

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
print (2012 in sorted(ANPP.year.unique()))
ANPP.head(2)


# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
len(ANPP_MK_df[ANPP_MK_df["trend"] == "increasing"])

# %%
df = ANPP_MK_df[(ANPP_MK_df["trend"] == "increasing") & (ANPP_MK_df["sens_slope"] < 20)]
df = df[df["p"] < 0.01]
print ("s < 20:    {:,}".format(len(df)))

df = ANPP_MK_df[(ANPP_MK_df["trend"] == "increasing") & 
               (ANPP_MK_df["sens_slope"] > 20) & 
               ((ANPP_MK_df["sens_slope"] < 30))]
df = df[df["p"] < 0.01]
print ("s in 20-30: {:,}".format(len(df)))

df = ANPP_MK_df[(ANPP_MK_df["trend"] == "increasing") & (ANPP_MK_df["sens_slope"] > 30)]
df = df[df["p"] < 0.01]
print ("s > 30:     {:,}".format(len(df)))

# %%
sorted(ANPP_MK_df["EW_meridian"].unique())

# %%

# %%
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid
# SF_west.head(2)

# %%
SF_west[["fid", "trend_yue"]].groupby("trend_yue").count()

# %%
# sorted(SF_west['state_1'].unique())

# %%
SF_west.rename(columns={"EW_meridia": "EW_meridian",
                        "p_valueSpe" : "p_valueSpearman",
                        "medians_di": "medians_diff_ANPP",
                        "medians__1" : "medians_diff_slope_ANPP",
                        "median_ANP" : "median_ANPP_change_as_perc",
                        "state_majo" : "state_majority_area"}, 
                      inplace=True)

# %%
county_fips_dict = pd.read_pickle(rangeland_reOrganized + "county_fips.sav")

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
groupveg = sorted(SF_west["groupveg"].unique())
groupveg

# %%
veg_colors = {"Barren-Rock/Sand/Clay" : "blue",
              "Conifer" : "green",
              "Grassland" : "red",
              "Hardwood" : "cyan",
              "Riparian" : "magenta",
              "Shrubland" : "yellow",
              "Sparse" : "black"}

# %% [markdown]
# ### Plot a couple of examples

# %%
from matplotlib.lines import Line2D

# %%
# Major ticks every 5, minor ticks every 1
major_ticks = np.arange(1984, 2024, 5)
minor_ticks = np.arange(1984, 2024, 1)
y_var = "mean_lb_per_acr"

# %%
txt_font_dict = {'fontsize':10, 'fontweight':'bold'}

tick_legend_FontSize = 10
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05}

plt.rcParams.update(params)

# %%
a_fid = 21519

a_metric = "median_ANPP_change_as_perc"
df = ANPP[ANPP.fid==a_fid].copy()

# %%
years = sorted(list(df.year.unique()))
first_decade = years[:10]
last_decade = years[-10:]
print (len(first_decade))
print (len(last_decade))

print (first_decade)
print (last_decade)

# %%
df_first_decade = df[df.year.isin(first_decade)].copy()
df_last_decade = df[df.year.isin(last_decade)].copy()
df_last_decade.head(2)

# %%
first_decade_median = df_first_decade["mean_lb_per_acr"].median().round(2)
last_decade_median = df_last_decade["mean_lb_per_acr"].median().round(2)
print (f"{first_decade_median = }")
print (f"{last_decade_median = }")
print ()
perc_change = 100 * ( (last_decade_median-first_decade_median)/first_decade_median ).round(4)
print (f"{perc_change = }")

# %%
first_decade_mean = df_first_decade["mean_lb_per_acr"].mean().round(2)
last_decade_mean = df_last_decade["mean_lb_per_acr"].mean().round(2)
print (f"{first_decade_mean = }")
print (f"{last_decade_mean = }")
print ()
perc_change = 100 * ( (last_decade_mean-first_decade_mean)/first_decade_mean ).round(4)
print (f"{perc_change = }")

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
axes.grid(which='major', alpha=0.5, axis="y")

########## Plot Time Series
axes.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes.scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

#################
### Text for Median of first decade
x1, x2 = first_decade[0], first_decade[-1]
axes.axvspan(x1, x2, color="dodgerblue", alpha=0.2); #facecolor='.01'
y_text = int(df[y_var].max()) / 1.2
txt_ = "First Decade\n median is {}".format(first_decade_median)
axes.scatter((x1+x2)/2, first_decade_median, color="red", s=50, zorder=3, label="median first decade")
axes.text(x=(x1+x2)/2, y=y_text, s=txt_, fontdict=txt_font_dict, ha='center');

#################
### Text for Median of last decade
x1, x2 = last_decade[0], last_decade[-1]
axes.axvspan(x1, x2, color="dodgerblue", alpha=0.2); #facecolor='.01'

txt_ = "Last Decade\n median is {}".format(last_decade_median)
axes.scatter((x1+x2)/2, last_decade_median, color="black", s=50, zorder=3, label="median last decade")
axes.text(x=(x1+x2)/2, y=y_text, s=txt_, fontdict=txt_font_dict, ha='center');

# axes.axhline(y=last_decade_median, color='black', linestyle='--')
# axes.axhline(y=first_decade_median, color='r', linestyle='--')
#################
a_metric_val = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, a_metric].values[0], 2)
axes.set_title("FID: " + str(a_fid) + " - " + a_metric + ": " +  str(a_metric_val) , y=0.95, fontsize=14);
####
axes.legend(loc="best");

# %%

# %% [markdown]
# ### Side by Side

# %%
a_fid = 24167
df = ANPP[ANPP.fid==a_fid].copy()

fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, 
                         gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
axes.grid(which='major', alpha=0.5, axis="y")

########## Plot Time Series
axes.plot(df.year, df[y_var], linewidth=3, color="dodgerblue");
axes.scatter(df.year, df[y_var], zorder=3, color="dodgerblue");

#################
a_metric_val = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, a_metric].values[0], 2)
axes.set_title("FID: " + str(a_fid) , y=0.95, fontsize=14);
axes.yaxis.set_major_locator(plt.MultipleLocator(300));
# axes.legend(loc="best");

# %%

# %% [markdown]
# ## Find bad MK labels

# %%
ANPP_MK_df.head(2)

# %%
ANPP_MK_df_99CL = ANPP_MK_df[ANPP_MK_df["p"] < 0.01].copy()
ANPP_MK_df_99CL = ANPP_MK_df_99CL[ANPP_MK_df_99CL["trend"] == "increasing"].copy()
ANPP_MK_df_99CL.shape

# %%
ANPP_MK_df_99CL[ANPP_MK_df_99CL["sens_slope"] < 10].shape

# %%
ANPP_MK_df_99CL[ANPP_MK_df_99CL["sens_slope"] < 10]

# %%
