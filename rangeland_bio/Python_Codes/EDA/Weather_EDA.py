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
import pickle
from datetime import datetime

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
# ####### Laptop
# rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
# min_bio_dir = rangeland_bio_base

# rangeland_base = rangeland_bio_base
# rangeland_reOrganized = rangeland_base

# %%
def plot_SF(SF, ax_, cmap_ = "Pastel1", col="EW_meridian"):
    SF.plot(column=col, ax=ax_, alpha=1, cmap=cmap_, edgecolor='k', legend=False, linewidth=0.1)


# %%
dpi_ = 300
map_dpi_ = 200
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')
cmap_RYG = cm.get_cmap('RdYlGn')

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

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


from shapely.geometry import Polygon
gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')
# gdf = geopandas.read_file(rangeland_bio_base +'cb_2018_us_state_500k')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")


visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)
# ANPP.sort_values(by= ['fid', 'year'], inplace=True)
# ANPP.head(2)

# %%

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %%
SF_west.rename(columns={"EW_meridia": "EW_meridian",
                        "p_valueSpe" : "p_valueSpearman",
                        "medians_di": "medians_diff_ANPP",
                        "medians__1" : "medians_diff_slope_ANPP",
                        "median_ANP" : "median_ANPP_change_as_perc",
                        "state_majo" : "state_majority_area"}, 
               inplace=True)

# %% [markdown]
# # Read Weather Data

# %%
filename = bio_reOrganized + "bps_weather.sav"
monthly_weather = pd.read_pickle(filename)
monthly_weather = monthly_weather["bps_weather"]
# change the order of columns!
monthly_weather.head(2)

# %%
print (f'{len(monthly_weather["fid"].unique())=}')
print (f'{len(ANPP["fid"].unique())=}')

# %%
len(set(monthly_weather["fid"].unique()).intersection(ANPP["fid"].unique()))

# %%
len(monthly_weather[monthly_weather["fid"].isin(list(ANPP["fid"].unique()))]["fid"].unique())

# %%
FIDs_weather_ANPP_common = list(set(monthly_weather["fid"].unique()).intersection(ANPP["fid"].unique()))

# Lets pick the ones are on the west
print (len(FIDs_weather_ANPP_common))
FIDs_weather_ANPP_common = list(set(FIDs_weather_ANPP_common).intersection(SF_west["fid"].unique()))
print (len(FIDs_weather_ANPP_common))

# %% [markdown]
# ### Subset to common FIDs:

# %%
ANPP    = ANPP[ANPP["fid"].isin(FIDs_weather_ANPP_common)]
monthly_weather = monthly_weather[monthly_weather["fid"].isin(FIDs_weather_ANPP_common)]

SF_west = SF_west[SF_west["fid"].isin(FIDs_weather_ANPP_common)]

# %%
monthly_weather.head(2)

# %%
# %%time
unique_number_of_years = {}

for a_fid in FIDs_weather_ANPP_common:
    LL = str(len(monthly_weather[monthly_weather.fid == a_fid])) + "_months"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%
528 / 12

# %%
# %%time
unique_number_of_years = {}

for a_fid in ANPP.fid.unique():
    LL = str(len(ANPP[ANPP.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%
ANPP.head(2)

# %%
monthly_weather.head(2)

# %% [markdown]
# # Compute annual 
# ```precipitation``` and ```temp``` for Sen's slope and Spearmans

# %%
cc = ["fid", "year", "precip_mm_month"]
annual_precip = monthly_weather[cc].groupby(["fid", "year"]).sum().reset_index()

cc = ["fid", "year", "avg_of_dailyAvgTemp_C"]
annual_temp = monthly_weather[cc].groupby(["fid", "year"]).mean().reset_index()

cc = ["fid", "year", "avg_of_dailyAvg_rel_hum"]
annual_rel_hum = monthly_weather[cc].groupby(["fid", "year"]).mean().reset_index()

# %%
annual_WA_ANPP = pd.merge(annual_precip, annual_temp, how="left", on=["fid", "year"])
annual_WA_ANPP = pd.merge(annual_WA_ANPP, annual_rel_hum, how="left", on=["fid", "year"])

annual_WA_ANPP = pd.merge(annual_WA_ANPP, ANPP[["fid", "year", "mean_lb_per_acr"]], 
                          how="left", on=["fid", "year"])

annual_WA_ANPP.rename(columns={"precip_mm_month": "precip_mm_yr", 
                          "avg_of_dailyAvgTemp_C": "avg_of_dailyAvgTemp_C_AvgOverMonths",
                          "avg_of_dailyAvg_rel_hum": "avg_of_dailyAvg_rel_hum_AvgOverMonths"}, 
                 inplace=True)
annual_WA_ANPP.head(2)

# %%
# some years are missing in ANPP as we know
print (annual_WA_ANPP.shape)
annual_WA_ANPP.dropna(subset=["mean_lb_per_acr"], inplace=True)

annual_WA_ANPP.sort_values(by= ['fid', 'year'], inplace=True)
annual_WA_ANPP.reset_index(drop=True, inplace=True)

print (annual_WA_ANPP.shape)

annual_WA_ANPP.head(2)

# %%
SF_west.head(2)

# %% [markdown]
# ## Compute Spearman for ANPP and precip

# %%
need_cols = ["fid", "centroid"]
precip_MK_df = SF_west[need_cols].copy()
print (ANPP_MK_df.shape)

precip_MK_df.drop_duplicates(inplace=True)
precip_MK_df.reset_index(drop=True, inplace=True)

print (precip_MK_df.shape)

MK_test_cols = ["precip_Spearman", "precip_p_valSpearman"]

precip_MK_df = pd.concat([precip_MK_df, pd.DataFrame(columns = MK_test_cols)])
precip_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)

# Why data type changed?!
precip_MK_df["fid"] = precip_MK_df["fid"].astype(np.int64)
precip_MK_df.drop(columns="centroid", inplace=True)
precip_MK_df.head(2)

# %%
# %%time
# populate the dataframe with MK test result now
for a_FID in precip_MK_df["fid"].unique():
    ANPP_TS = annual_WA_ANPP.loc[annual_WA_ANPP["fid"]==a_FID, "mean_lb_per_acr"].values
    precip_TS = annual_WA_ANPP.loc[annual_WA_ANPP["fid"]==a_FID, "precip_mm_yr"].values
    
    # MK test
    # trend, _, p, z, Tau, s, var_s, slope, intercept = mk.original_test(ANPP_TS)

    # Spearman's rank
    Spearman, p_valueSpearman = stats.spearmanr(precip_TS, ANPP_TS)

    # Update dataframe by MK result
    # L_ = [trend, p, z, Tau, s, var_s, slope, intercept, Spearman, p_valueSpearman]
    L_ = [Spearman, p_valueSpearman]
    precip_MK_df.loc[precip_MK_df["fid"]==a_FID, MK_test_cols] = L_

precip_MK_df.head(2)

# %% [markdown]
# ## Compute Spearman for ANPP and Temp

# %%
need_cols = ["fid", "centroid"]
temp_MK_df = SF_west[need_cols].copy()
print (ANPP_MK_df.shape)

temp_MK_df.drop_duplicates(inplace=True)
temp_MK_df.reset_index(drop=True, inplace=True)

print (temp_MK_df.shape)

MK_test_cols = ["temp_Spearman", "temp_p_valSpearman"]

temp_MK_df = pd.concat([temp_MK_df, pd.DataFrame(columns = MK_test_cols)])
temp_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)
temp_MK_df.drop(columns="centroid", inplace=True)
# Why data type changed?!
temp_MK_df["fid"] = temp_MK_df["fid"].astype(np.int64)
temp_MK_df.head(2)

# %%
# %%time
# populate the dataframe with MK test result now
for a_FID in temp_MK_df["fid"].unique():
    ANPP_TS = annual_WA_ANPP.loc[annual_WA_ANPP["fid"]==a_FID, "mean_lb_per_acr"].values
    temp_TS = annual_WA_ANPP.loc[annual_WA_ANPP["fid"]==a_FID, "avg_of_dailyAvgTemp_C_AvgOverMonths"].values

    # Spearman's rank
    Spearman, p_valueSpearman = stats.spearmanr(temp_TS, ANPP_TS)

    # Update dataframe
    L_ = [Spearman, p_valueSpearman]
    temp_MK_df.loc[temp_MK_df["fid"]==a_FID, MK_test_cols] = L_

temp_MK_df.head(2)

# %%
temp_precip_spear = pd.merge(temp_MK_df[["fid", "temp_Spearman", "temp_p_valSpearman"]], 
                             precip_MK_df, how="left", on=["fid"])

# temp_precip_spear = pd.merge(temp_precip_spear, 
#                              SF_west[["fid", "trend"]], how="left", on=["fid"])

temp_precip_spear.head(2)

# %%
temp = temp_MK_df[["fid", "temp_Spearman", "temp_p_valSpearman"]].copy()
precip = precip_MK_df.copy()

temp = temp[temp["temp_p_valSpearman"] < 0.05].copy()
precip = precip[precip["precip_p_valSpearman"] < 0.05].copy()

print (temp.shape)
print (precip.shape)

temp_precip_spear_sig_95 = pd.merge(temp, precip, how="outer", on=["fid"])

# temp_precip_spear_sig_95 = pd.merge(temp_precip_spear_sig_95, 
#                                      SF_west[["fid", "trend"]], how="left", on=["fid"])

temp_precip_spear_sig_95.head(2)

# %%

# %%
print (f"{temp_precip_spear.shape = }")
print (f"{temp_precip_spear_sig_95.shape = }")

# %%
temp = temp_MK_df[["fid", "temp_Spearman", "temp_p_valSpearman"]].copy()
precip = precip_MK_df.copy()

temp = temp[temp["temp_p_valSpearman"] < 0.1].copy()
precip = precip[precip["precip_p_valSpearman"] < 0.1].copy()

temp_precip_spear_sig_90 = pd.merge(temp, precip, how="outer", on=["fid"])
temp_precip_spear_sig_90.head(2)

# %%
print (f"{temp_precip_spear.shape = }")
print (f"{temp_precip_spear_sig_95.shape = }")
print (f"{temp_precip_spear_sig_90.shape = }")

# %% [markdown]
#

# %%
temp_precip_spear_sig_95.head(2)

# %%
print (temp_precip_spear_sig_95.shape)
print (temp_precip_spear_sig_95.dropna().shape)

# %%
SF_west_Spearman_95 = pd.merge(SF_west, temp_precip_spear_sig_95, on="fid", how="left")

# %% [markdown]
# ### Some plots

# %%
# Parameters for font sizes
plt.rc("font", family="Palatino")
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * 0.71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.7,
          "ytick.labelsize": tick_legend_FontSize * 0.7,
          "axes.titlepad": 5,    'legend.handlelength': 2}
plt.rcParams.update(params)

# %%
min_color = min(SF_west_Spearman_95['temp_Spearman'].min(), SF_west_Spearman_95['precip_Spearman'].min())
max_color = max(SF_west_Spearman_95['temp_Spearman'].max(), SF_west_Spearman_95['precip_Spearman'].max())
norm_colorB = Normalize(vmin=min_color, vmax=max_color)

min_max = max(abs(min_color),abs(max_color))
norm_colorB = Normalize(vmin=-min_max, vmax=min_max)
####################################################################################
####################################################################################
####################################################################################
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
####### States
plot_SF(SF=visframe_mainLand_west, cmap_="Pastel1", ax_=ax1, col="EW_meridian")
plot_SF(SF=visframe_mainLand_west, cmap_="Pastel1", ax_=ax2, col="EW_meridian")

####################################################################################
p1 = SF_west_Spearman_95.plot(column='temp_Spearman', ax=ax1, cmap=cmap_RYG, norm=norm_colorB, legend=False)
ax1.set_title("temperature (Spearman's rank; 95% significant)")

p2 = SF_west_Spearman_95.plot(column='precip_Spearman', ax=ax2, cmap=cmap_RYG, norm=norm_colorB, legend=False)

ax2.set_title("precipitation (Spearman's rank; 95% significant)")
####################################################################################
# fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=0.1, hspace=0.01)
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-0.2, hspace=0)
fig.colorbar(p1.get_children()[1], ax=ax2, fraction=0.02, orientation='vertical', location="right")
# ax1.inset_axes([0.3, 0.07, 0.4, 0.04])
fig.suptitle('All locations (not just greening)', y=1.01)
plt.show();

# %%
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %% [markdown]
# ### Do Focus on greening locations only

# %%
SF_west_Spearman_95_green = SF_west_Spearman_95[SF_west_Spearman_95["trend"] == "increasing"].copy()
SF_west_Spearman_95_green.head(2)

# %%
min_color = min(SF_west_Spearman_95_green['temp_Spearman'].min(), 
                SF_west_Spearman_95_green['precip_Spearman'].min())
max_color = max(SF_west_Spearman_95_green['temp_Spearman'].max(), 
                SF_west_Spearman_95_green['precip_Spearman'].max())
norm_colorB = Normalize(vmin=min_color, vmax=max_color)

min_max = max(abs(min_color),abs(max_color))
norm_colorB = Normalize(vmin=-min_max, vmax=min_max)
####################################################################################
####################################################################################
####################################################################################
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True, dpi=map_dpi_)
(ax1, ax2) = axes
ax1.set_xticks([]); ax1.set_yticks([])
ax2.set_xticks([]); ax2.set_yticks([])
####### States
plot_SF(SF=visframe_mainLand_west, cmap_="Pastel1", ax_=ax1, col="EW_meridian")
plot_SF(SF=visframe_mainLand_west, cmap_="Pastel1", ax_=ax2, col="EW_meridian")
####################################################################################
p1 = SF_west_Spearman_95_green.plot(column='temp_Spearman', ax=ax1, cmap=cmap_RYG, norm=norm_colorB)
ax1.set_title("temperature (Spearman's rank; 95% significant)")

p2 = SF_west_Spearman_95_green.plot(column='precip_Spearman', ax=ax2, cmap=cmap_RYG, norm=norm_colorB)
ax2.set_title("precipitation (Spearman's rank; 95% significant)")
####################################################################################
cax = ax2.inset_axes([1.05, 0.3, 0.04, 0.4])
fig.colorbar(p1.get_children()[1], cax=cax, orientation='vertical')
fig.subplots_adjust(top=0.91, bottom=0.08, left=0.082, right=0.981, wspace=-0.1, hspace=0)
fig.suptitle('Greening locations', y=1.01)
plt.show();

# %% [markdown]
# # Regression
#
# origin of spreg.OLS_ in ```04_02_2024_NonNormalModelsInterpret.ipynb```.

# %%
from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda
import geopandas, contextily

from scipy.stats import ttest_ind

# %%
annual_WA_ANPP.head(2)

# %%
annual_WA_ANPP['precip_mm_yr'] = annual_WA_ANPP['precip_mm_yr'] / 10
annual_WA_ANPP.rename(columns={"precip_mm_yr": "precip_cm_yr"}, inplace=True)
annual_WA_ANPP.head(2)

# %%
annual_WA_ANPP["temp_X_precip"] = annual_WA_ANPP["precip_cm_yr"] * \
                                  annual_WA_ANPP["avg_of_dailyAvgTemp_C_AvgOverMonths"]
    
annual_WA_ANPP.head(2)

# %%
print (len(annual_WA_ANPP.fid.unique()))

# %%
print (SF_west.shape)
SF_west.head(2)

# %%
annual_WA_ANPP = pd.merge(annual_WA_ANPP, SF_west[["fid", "groupveg"]], on="fid", how="left")
annual_WA_ANPP.head(2)

# %%
print (f'{round(annual_WA_ANPP["precip_cm_yr"].min(), 2) = }')
print (f'{round(annual_WA_ANPP["precip_cm_yr"].max(), 2) = }')

print (f'{round(annual_WA_ANPP["avg_of_dailyAvgTemp_C_AvgOverMonths"].min()) = }')
print (f'{round(annual_WA_ANPP["avg_of_dailyAvgTemp_C_AvgOverMonths"].max()) = }')

# %%
groupveg = sorted(annual_WA_ANPP["groupveg"].unique())
groupveg

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5 = spreg.OLS_Regimes(y = annual_WA_ANPP[depen_var].values,
                       x = annual_WA_ANPP[indp_vars].values, 

                       # Variable specifying neighborhood membership
                       regimes = annual_WA_ANPP["groupveg"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var, # Dependent variable name
                       name_x=indp_vars)

print (f"{m5.r2.round(2) = }")

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

# %%
groupveg

# %%
# West regime
## Extract variables for the west side 
Barren_m = [i for i in m5_results.index if "Barren" in i]
Conifer_m = [i for i in m5_results.index if "Conifer" in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m = [i for i in m5_results.index if "Hardwood" in i]
Riparian_m = [i for i in m5_results.index if "Riparian" in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]
Sparse_m = [i for i in m5_results.index if "Sparse" in i]


## Subset results to Barren
veg_ = "Barren"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Barren = m5_results.loc[Barren_m, :].rename(lambda i: i.replace(rep_, ""))
Barren.columns = pd.MultiIndex.from_product([[veg_], Barren.columns])

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Riparian
veg_ = "Riparian"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Riparian = m5_results.loc[Riparian_m, :].rename(lambda i: i.replace(rep_, ""))
Riparian.columns = pd.MultiIndex.from_product([[veg_], Riparian.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

## Subset results to Sparse
veg_ = "Sparse"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Sparse = m5_results.loc[Sparse_m, :].rename(lambda i: i.replace(rep_, ""))
Sparse.columns = pd.MultiIndex.from_product([[veg_], Sparse.columns])


# Concat both models
table_ = pd.concat([Barren, Conifer, Grassland, Hardwood, Riparian, Shrubland, Sparse], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %% [markdown]
# # Split and normalize

# %%
annual_WA_ANPP.head(2)

# %%
# reorder the dataframe
col_order = ['fid', 'year', 'mean_lb_per_acr',
             "groupveg",
             'precip_cm_yr',
             'avg_of_dailyAvgTemp_C_AvgOverMonths',
             'avg_of_dailyAvg_rel_hum_AvgOverMonths',
             'temp_X_precip',
             ]
annual_WA_ANPP = annual_WA_ANPP[col_order]
annual_WA_ANPP.head(2)

# %%
depen_var = "mean_lb_per_acr"
indp_vars = list(annual_WA_ANPP.columns[3:])
numeric_indp_vars = list(annual_WA_ANPP.columns[4:])

y_df = annual_WA_ANPP[depen_var].copy()
indp_df = annual_WA_ANPP[indp_vars].copy()

# %%

# %%
X_train, X_test, y_train, y_test = train_test_split(indp_df, y_df, test_size=0.3, random_state=42)
X_train.head(2)

# %%
train_idx = list(X_train.index)
test_idx = list(X_test.index)

# %%
# standard_indp = preprocessing.scale(all_df[explain_vars_herb]) # this is biased
means = X_train[numeric_indp_vars].mean()
stds = X_train[numeric_indp_vars].std(ddof=1)

X_train_normal = X_train.copy()
X_test_normal = X_test.copy()

X_train_normal[numeric_indp_vars] = (X_train_normal[numeric_indp_vars] - means) / stds
X_test_normal[numeric_indp_vars]  = (X_test_normal[numeric_indp_vars]  - means) / stds
X_train_normal.head(2)

# %% [markdown]
# # Model normalized data

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5 = spreg.OLS_Regimes(y = y_train.values,
                       x = X_train_normal[indp_vars].values, 

                       # Variable specifying neighborhood membership
                       regimes = X_train_normal["groupveg"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var, # Dependent variable name
                       name_x=indp_vars)

print (f"{m5.r2.round(2) = }")

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

# %%
# West regime
## Extract variables for the west side 
Barren_m = [i for i in m5_results.index if "Barren" in i]
Conifer_m = [i for i in m5_results.index if "Conifer" in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m = [i for i in m5_results.index if "Hardwood" in i]
Riparian_m = [i for i in m5_results.index if "Riparian" in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]
Sparse_m = [i for i in m5_results.index if "Sparse" in i]


## Subset results to Barren
veg_ = "Barren"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Barren = m5_results.loc[Barren_m, :].rename(lambda i: i.replace(rep_, ""))
Barren.columns = pd.MultiIndex.from_product([[veg_], Barren.columns])

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Riparian
veg_ = "Riparian"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Riparian = m5_results.loc[Riparian_m, :].rename(lambda i: i.replace(rep_, ""))
Riparian.columns = pd.MultiIndex.from_product([[veg_], Riparian.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

## Subset results to Sparse
veg_ = "Sparse"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Sparse = m5_results.loc[Sparse_m, :].rename(lambda i: i.replace(rep_, ""))
Sparse.columns = pd.MultiIndex.from_product([[veg_], Sparse.columns])


# Concat both models
table_ = pd.concat([Barren, Conifer, Grassland, Hardwood, Riparian, Shrubland, Sparse], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%

# %%

# %%
