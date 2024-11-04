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
plt.rc("font", family="Palatino")

# font = {"size": 10}
# matplotlib.rc("font", **font)

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
# county_fips_dict = pd.read_pickle(rangeland_reOrganized + "county_fips.sav")

# county_fips = county_fips_dict["county_fips"]
# full_2_abb = county_fips_dict["full_2_abb"]
# abb_2_full_dict = county_fips_dict["abb_2_full_dict"]
# abb_full_df = county_fips_dict["abb_full_df"]
# filtered_counties_29States = county_fips_dict["filtered_counties_29States"]
# SoI = county_fips_dict["SoI"]
# state_fips = county_fips_dict["state_fips"]

# state_fips = state_fips[state_fips.state != "VI"].copy()
# state_fips.head(2)


# from shapely.geometry import Polygon
# gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')
# # gdf = geopandas.read_file(rangeland_bio_base +'cb_2018_us_state_500k')

# gdf.rename(columns={"STUSPS": "state"}, inplace=True)
# gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
# gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")


# visframe = gdf.to_crs({'init':'epsg:5070'})
# visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

# visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
# visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

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
# %%time
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid

SF_west.rename(columns={"EW_meridia": "EW_meridian",
                        "p_valueSpe" : "p_valueSpearman",
                        "medians_di": "medians_diff_ANPP",
                        "medians__1" : "medians_diff_slope_ANPP",
                        "median_ANP" : "median_ANPP_change_as_perc",
                        "state_majo" : "state_majority_area"}, 
               inplace=True)
SF_west.head(2)

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
print (len(set(monthly_weather["fid"].unique()).intersection(ANPP["fid"].unique())))
print (len(monthly_weather[monthly_weather["fid"].isin(list(ANPP["fid"].unique()))]["fid"].unique()))

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
# # %%time
# unique_number_of_years = {}
# for a_fid in FIDs_weather_ANPP_common:
#     LL = str(len(monthly_weather[monthly_weather.fid == a_fid])) + "_months"    
#     if not (LL in unique_number_of_years.keys()):
#         unique_number_of_years[LL] = 1
#     else:
#         unique_number_of_years[LL] = unique_number_of_years[LL] + 1
# print (unique_number_of_years)

528 / 12

# # %%time
# unique_number_of_years = {}
# for a_fid in ANPP.fid.unique():
#     LL = str(len(ANPP[ANPP.fid == a_fid])) + "_years"
#     if not (LL in unique_number_of_years.keys()):
#         unique_number_of_years[LL] = 1
#     else:
#         unique_number_of_years[LL] = unique_number_of_years[LL] + 1
# unique_number_of_years

# %%
ANPP.head(2)

# %%
monthly_weather.head(2)

# %%
monthly_weather.isna().sum()

# %%
# # %%time 
# monthly_weather_wide = monthly_weather.copy()
# monthly_weather_wide.sort_values(by= ['fid', 'year', "month"], inplace=True)
# monthly_weather_wide["month"] = monthly_weather_wide["month"].astype(str)
# df1 = monthly_weather_wide[['fid', 'year', "month", 'avg_of_dailyAvg_rel_hum']].copy()
# df2 = monthly_weather_wide[['fid', 'year', "month", 'avg_of_dailyAvgTemp_C']].copy()
# df3 = monthly_weather_wide[['fid', 'year', "month", 'thi_avg']].copy()
# df4 = monthly_weather_wide[['fid', 'year', "month", 'precip_mm_month']].copy()
# ########################################################################
# df1 = df1.pivot(index=['fid', 'year'], columns=['month'])
# df2 = df2.pivot(index=['fid', 'year'], columns=['month'])
# df3 = df3.pivot(index=['fid', 'year'], columns=['month'])
# df4 = df4.pivot(index=['fid', 'year'], columns=['month'])
# ########################################################################
# df1.reset_index(drop=False, inplace=True)
# df2.reset_index(drop=False, inplace=True)
# df3.reset_index(drop=False, inplace=True)
# df4.reset_index(drop=False, inplace=True)
# ########################################################################
# df1.columns = ["_".join(tup) for tup in df1.columns.to_flat_index()]
# df2.columns = ["_".join(tup) for tup in df2.columns.to_flat_index()]
# df3.columns = ["_".join(tup) for tup in df3.columns.to_flat_index()]
# df4.columns = ["_".join(tup) for tup in df4.columns.to_flat_index()]
# ########################################################################
# df1.rename(columns={"fid_": "fid", "year_":"year"}, inplace=True)
# df2.rename(columns={"fid_": "fid", "year_": "year"}, inplace=True)
# df3.rename(columns={"fid_": "fid", "year_": "year"}, inplace=True)
# df4.rename(columns={"fid_": "fid", "year_": "year"}, inplace=True)

# df1.head(2)

# wide_WA = pd.merge(df1, df2, how="left", on=["fid", "year"])
# wide_WA = pd.merge(wide_WA, df3, how="left", on=["fid", "year"])
# wide_WA = pd.merge(wide_WA, df4, how="left", on=["fid", "year"])

# filename = bio_reOrganized + "bps_weather_wide.sav"

# export_ = {"bps_weather_wide": wide_WA, 
#            "source_code" : "Weather_monthly_models",
#            "Author": "HN",
#            "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# pickle.dump(export_, open(filename, 'wb'))
# wide_WA.head(2)

# %%
filename = bio_reOrganized + "bps_weather_wide.sav"
monthly_weather_wide = pd.read_pickle(filename)
monthly_weather_wide = monthly_weather_wide["bps_weather_wide"]
monthly_weather_wide.head(2)

# %%
print (monthly_weather_wide.year.min())
print (monthly_weather_wide.year.max())
print ()
print (ANPP.year.min())
print (ANPP.year.max())

# %%
# pick up overlapping years
overlap_years = list(set(ANPP.year.unique()).intersection(set(monthly_weather_wide.year.unique())))
ANPP = ANPP[ANPP.year.isin(overlap_years)].copy()
monthly_weather_wide = monthly_weather_wide[monthly_weather_wide.year.isin(overlap_years)].copy()

# %%
ANPP.head(2)

# %%
ANPP_weather_wide = pd.merge(ANPP[["fid", "year", "mean_lb_per_acr"]], monthly_weather_wide, 
                             how="left", on=["fid", "year"])
ANPP_weather_wide.head(2)

# %%
ANPP_weather_wide.isna().sum().sum()

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

for a_veg in  groupveg:
    SF_west.loc[SF_west['groupveg'] == a_veg, 'color'] = veg_colors[a_veg]

SF_west.head(2)

# %%

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
print (len(ANPP_weather_wide.fid.unique()))

# %% [markdown]
# # Drop bad Vegs
# ```Sparse```, ```Riparian```, ```Barren-Rock/Sand/Clay```, and ```Conifer```?

# %%
ANPP_weather_wide = pd.merge(ANPP_weather_wide, SF_west[["fid", "groupveg"]], how="left", on=["fid"])
ANPP_weather_wide.head(2)

# %%

# %%
good_vegs = ['Conifer', 'Grassland', 'Hardwood', 'Shrubland']
ANPP_weather_wide = ANPP_weather_wide[ANPP_weather_wide["groupveg"].isin(good_vegs)].copy()
ANPP_weather_wide.reset_index(drop=True, inplace=True)
groupveg = sorted(ANPP_weather_wide["groupveg"].unique())
groupveg

# %%
ANPP_weather_wide.head(2)

# %% [markdown]
# # Focus on Greening areas only

# %%
print (ANPP_MK_df["trend"].unique())
greening_FIDs = list(ANPP_MK_df[ANPP_MK_df["trend"] == "increasing"]["fid"].unique())
greening_FIDs[1:3]

# %%
ANPP_weather_wide_G = ANPP_weather_wide[ANPP_weather_wide["fid"].isin(greening_FIDs)].copy()
ANPP_weather_wide_G.reset_index(drop=True, inplace=True)
ANPP_weather_wide_G.head(2)

# %%
groupveg

# %%
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.4,
    "axes.titlesize": tick_legend_FontSize * 2,
    "xtick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1,  #  * 0.75
    "axes.titlepad": 10}
plt.rcParams.update(params)

# %%
veg_ = groupveg[1]
A = ANPP_weather_wide_G.copy()
A = A[A["groupveg"] == veg_]

# Let us just do NPP with precips.
y_var = "mean_lb_per_acr"

fig, axes = plt.subplots(12, 1, figsize=(10, 36), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

for ii in range(1, 13):
    col_name = "avg_of_dailyAvgTemp_C_" + str(ii)
    axes[ii-1].scatter(ANPP_weather_wide_G[col_name], ANPP_weather_wide_G[y_var], marker='s', s=2);
    axes[ii-1].legend(["month " + str(ii)], loc='upper right');

axes[11].set_xlabel("temperature (" + veg_ + ")");

# %%

# %%
# Let us just do NPP with precips.
veg_ = groupveg[1]
y_var = "mean_lb_per_acr"

fig, axes = plt.subplots(12, 1, figsize=(10, 36), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

for ii in range(1, 13):
    col_name = "precip_mm_month_" + str(ii)
    axes[ii-1].scatter(ANPP_weather_wide_G[col_name], ANPP_weather_wide_G[y_var], marker='s', s=2);
    axes[ii-1].legend(["month " + str(ii)], loc='upper right');

axes[11].set_xlabel("precipitation (" + veg_ + ")");

# %%

# %%
# Let us just do NPP with precips.
veg_ = groupveg[1]

fig, axes = plt.subplots(12, 1, figsize=(10, 36), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

for ii in range(1, 13):
    precip_col = "precip_mm_month_" + str(ii)
    temp_col = "avg_of_dailyAvgTemp_C_" + str(ii)
    axes[ii-1].scatter(ANPP_weather_wide_G[precip_col], ANPP_weather_wide_G[temp_col], marker='s', s=2);
    axes[ii-1].legend(["month " + str(ii)], loc='upper right');

axes[11].set_xlabel("precipitation (" + veg_ + ")");

# %%

# %%
# precip_cols = [x for x in ANPP_weather_wide_G.columns if "precip" in x]
# temp_cols = [x for x in ANPP_weather_wide_G.columns if "Temp" in x]
# precip_cols[:2]
# precip_cols = ["mean_lb_per_acr"] + precip_cols
# temp_cols = ["mean_lb_per_acr"] + temp_cols

# veg_ = groupveg[0]
# my_scatter = sns.pairplot(ANPP_weather_wide_G[ANPP_weather_wide_G["groupveg"] == veg_][temp_cols],  
#                           size=1.5, diag_kind="None", plot_kws={"s": 4}, corner=True)
# my_scatter.fig.suptitle(veg_, y=1);

# %%

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
# Barren_m = [i for i in m5_results.index if "Barren" in i]
Conifer_m = [i for i in m5_results.index if "Conifer" in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m = [i for i in m5_results.index if "Hardwood" in i]
# Riparian_m = [i for i in m5_results.index if "Riparian" in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]
# Sparse_m = [i for i in m5_results.index if "Sparse" in i]

## Subset results to Barren
# veg_ = "Barren"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Barren = m5_results.loc[Barren_m, :].rename(lambda i: i.replace(rep_, ""))
# Barren.columns = pd.MultiIndex.from_product([[veg_], Barren.columns])

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

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Riparian
# veg_ = "Riparian"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Riparian = m5_results.loc[Riparian_m, :].rename(lambda i: i.replace(rep_, ""))
# Riparian.columns = pd.MultiIndex.from_product([[veg_], Riparian.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

## Subset results to Sparse
# veg_ = "Sparse"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Sparse = m5_results.loc[Sparse_m, :].rename(lambda i: i.replace(rep_, ""))
# Sparse.columns = pd.MultiIndex.from_product([[veg_], Sparse.columns])

# Concat both models
# table_ = pd.concat([Barren, Conifer, Grassland, Hardwood, Riparian, Shrubland, Sparse], axis=1).round(5)
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
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
annual_WA_ANPP.head(2)

# %%
SF_west.head(2)

# %%
tick_legend_FontSize = 12
params = {"legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          'axes.grid' : False}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %% [markdown]
#  - ToDo
#     - train a model with a given veg and check if spreg is doing what you think it is doing.
#        - Done. except SEs change!
#
#     - Look at R2 for each model separately. what does the general R2 mean that spreg spits out?
#
#     - Do modeling with interaction terms
#
#     - plot residual plots
#
#     - try model with log(y).

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

# pick a veg indices
idx_ = X_train_normal[X_train_normal["groupveg"] == "Grassland"].index
X_tr_ = X_train_normal.loc[idx_].copy()
y_tr_ = y_train.loc[idx_].copy()

m5_solo = spreg.OLS(y = y_tr_.values, x = X_tr_[indp_vars].values, 
                    name_y=depen_var, name_x=indp_vars)

print (f"{m5_solo.r2.round(2) = }")

m5_solo_results = pd.DataFrame({"Coeff.": m5_solo.betas.flatten(), 
                                "Std. Error": m5_solo.std_err.flatten(), 
                                "P-Value": [i[1] for i in m5_solo.t_stat]
                               }, index=m5_solo.name_x)

# %%
m5_solo_results

# %%
# m5_solo.predy

# %%
X = X_tr_.copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = y_train.loc[idx_].copy().astype(float)
ks = sm.OLS(Y, X[["const", "precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]])
ks_result = ks.fit()
y_pred = ks_result.predict(X[["const", "precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]])
ks_result.params

# %% [markdown]
# # Model based on Temp

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5_T_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                      regimes = X_train_normal["groupveg"].tolist(),
                                      constant_regi="many", regime_err_sep=False,
                                      name_y=depen_var, name_x=indp_vars)

print (f"{m5_T_normal.r2.round(2) = }")

m5_T_normal_results = pd.DataFrame({# Pull out regression coefficients and
                                          "Coeff.": m5_T_normal.betas.flatten(), 
                                          # Pull out and flatten standard errors
                                          "Std. Error": m5_T_normal.std_err.flatten(), 
                                          # Pull out P-values from t-stat object
                                          "P-Value": [i[1] for i in m5_T_normal.t_stat], 
                                           }, index=m5_T_normal.name_x)

## Extract variables for each veg type
# Barren_m    = [i for i in m5_T_normal_results.index if "Barren"    in i]
Conifer_m   = [i for i in m5_T_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_T_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_T_normal_results.index if "Hardwood"  in i]
# Riparian_m  = [i for i in m5_T_normal_results.index if "Riparian"  in i]
Shrubland_m = [i for i in m5_T_normal_results.index if "Shrubland" in i]
# Sparse_m    = [i for i in m5_T_normal_results.index if "Sparse"    in i]

## Subset results to Barren
# veg_ = "Barren"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Barren = m5_T_normal_results.loc[Barren_m, :].rename(lambda i: i.replace(rep_, ""))
# Barren.columns = pd.MultiIndex.from_product([[veg_], Barren.columns])

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_T_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_T_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_T_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Riparian
# veg_ = "Riparian"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Riparian = m5_T_normal_results.loc[Riparian_m, :].rename(lambda i: i.replace(rep_, ""))
# Riparian.columns = pd.MultiIndex.from_product([[veg_], Riparian.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_T_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

## Subset results to Sparse
# veg_ = "Sparse"
# rep_ = [x for x in groupveg if veg_ in x][0] + "_"
# Sparse = m5_T_normal_results.loc[Sparse_m, :].rename(lambda i: i.replace(rep_, ""))
# Sparse.columns = pd.MultiIndex.from_product([[veg_], Sparse.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_T_normal.predy, m5_T_normal.u, c="dodgerblue", s=2);

title_ = f"NPP = $f(T)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Model based on Precip

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr"]

m5_P_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                regimes = X_train_normal["groupveg"].tolist(),
                                constant_regi="many", regime_err_sep=False,
                                name_y=depen_var, name_x=indp_vars)
print (f"{m5_P_normal.r2.round(2) = }")

m5_P_normal_results = pd.DataFrame({"Coeff.": m5_P_normal.betas.flatten(), 
                                    "Std. Error": m5_P_normal.std_err.flatten(), 
                                    "P-Value": [i[1] for i in m5_P_normal.t_stat], 
                                    }, index=m5_P_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_P_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_P_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_P_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_P_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_P_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_P_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_P_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_P_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_P_normal.predy, m5_P_normal.u, c="dodgerblue", s=2);

title_ = f"NPP = $f(P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Model by Temp and Precip

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5_TP_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                 regimes = X_train_normal["groupveg"].tolist(),
                                 constant_regi="many", regime_err_sep=False,
                                 name_y=depen_var, name_x=indp_vars)

print (f"{m5_TP_normal.r2.round(2) = }")

m5_TP_normal_results = pd.DataFrame({"Coeff.": m5_TP_normal.betas.flatten(), 
                                     "Std. Error": m5_TP_normal.std_err.flatten(), 
                                     "P-Value": [i[1] for i in m5_TP_normal.t_stat],
                                    }, index=m5_TP_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TP_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TP_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TP_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TP_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TP_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TP_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TP_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TP_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TP_normal.predy, m5_TP_normal.u, c="dodgerblue", s=2);

title_ = f"NPP = $f(T, P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Model with interaction terms

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths", "temp_X_precip"]

m5_TPinter_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                      regimes = X_train_normal["groupveg"].tolist(),
                                      constant_regi="many", regime_err_sep=False,
                                      name_y=depen_var, name_x=indp_vars)
print (f"{m5_TPinter_normal.r2.round(2) = }")

m5_TPinter_normal_results = pd.DataFrame({"Coeff.": m5_TPinter_normal.betas.flatten(), 
                                          "Std. Error": m5_TPinter_normal.std_err.flatten(), 
                                          "P-Value": [i[1] for i in m5_TPinter_normal.t_stat]}, 
                                         index=m5_TPinter_normal.name_x)
## Extract variables for each veg type
Conifer_m   = [i for i in m5_TPinter_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TPinter_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TPinter_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TPinter_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TPinter_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TPinter_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TPinter_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TPinter_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TPinter_normal.predy, m5_TPinter_normal.u, c="dodgerblue", s=2);

title_ = f"NPP = $f(T, P, T \u00D7 P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Temp, precipitation, humidity

# %%
X_train_normal.head(2)

# %%

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths", "avg_of_dailyAvg_rel_hum_AvgOverMonths"]

m5_TPH_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                 regimes = X_train_normal["groupveg"].tolist(),
                                 constant_regi="many", regime_err_sep=False,
                                 name_y=depen_var, name_x=indp_vars)

print (f"{m5_TPH_normal.r2.round(2) = }")

m5_TPH_normal_results = pd.DataFrame({"Coeff.": m5_TPH_normal.betas.flatten(), 
                                      "Std. Error": m5_TPH_normal.std_err.flatten(), 
                                      "P-Value": [i[1] for i in m5_TPH_normal.t_stat]}, 
                                     index=m5_TPH_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TPH_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TPH_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TPH_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TPH_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TPH_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TPH_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TPH_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TPH_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp", 
                       "avg_of_dailyAvg_rel_hum_AvgOverMonths" : "RH"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TPH_normal.predy, m5_TPH_normal.u, c="dodgerblue", s=2);

title_ = f"NPP = $f(T, P, RH)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %% [markdown]
# # Add square terms

# %%
print (X_train_normal.shape)
X_train_normal.head(2)

# %%
X_train_normal["temp_sq"] = X_train_normal["avg_of_dailyAvgTemp_C_AvgOverMonths"] ** 2
X_train_normal["precip_sq"] = X_train_normal["precip_cm_yr"] ** 2
X_train_normal.head(2)

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths", "temp_sq", "precip_sq", "temp_X_precip"]

m5_TP_sq_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                  regimes = X_train_normal["groupveg"].tolist(),
                                  constant_regi="many", regime_err_sep=False,
                                  name_y=depen_var, name_x=indp_vars)

print (f"{m5_TP_sq_normal.r2.round(2) = }")

m5_TP_sq_normal_results = pd.DataFrame({"Coeff.": m5_TP_sq_normal.betas.flatten(), 
                                      "Std. Error": m5_TP_sq_normal.std_err.flatten(), 
                                      "P-Value": [i[1] for i in m5_TP_sq_normal.t_stat]}, 
                                     index=m5_TP_sq_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TP_sq_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TP_sq_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TP_sq_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TP_sq_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TP_sq_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TP_sq_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TP_sq_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TP_sq_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp", 
                       "avg_of_dailyAvg_rel_hum_AvgOverMonths" : "RH"}, inplace=True)
table_

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TP_sq_normal.predy, m5_TP_sq_normal.u, c="dodgerblue", s=2);

title_ = f"NPP = $f(T, P, T^2, P^2, T \u00D7 P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %% [markdown]
# # log of Y based on Temp, Precip 

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths"]

m5_TP_normal_logy = spreg.OLS_Regimes(y = np.log10(y_train.values), x = X_train_normal[indp_vars].values, 
                                 regimes = X_train_normal["groupveg"].tolist(),
                                 constant_regi="many", regime_err_sep=False,
                                 name_y=depen_var, name_x=indp_vars)

print (f"{m5_TP_normal_logy.r2.round(2) = }")

m5_TP_normal_logy_results = pd.DataFrame({"Coeff.": m5_TP_normal_logy.betas.flatten(), 
                                     "Std. Error": m5_TP_normal_logy.std_err.flatten(), 
                                     "P-Value": [i[1] for i in m5_TP_normal_logy.t_stat],
                                    }, index=m5_TP_normal_logy.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TP_normal_logy_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TP_normal_logy_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TP_normal_logy_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TP_normal_logy_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TP_normal_logy_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TP_normal_logy_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TP_normal_logy_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TP_normal_logy_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TP_normal_logy.predy, m5_TP_normal_logy.u, c="dodgerblue", s=2);

title_ = f"$log(y) = f(T, P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
X_train_normal.head(2)

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(X_train["precip_cm_yr"], X_train["avg_of_dailyAvgTemp_C_AvgOverMonths"], 
             c="dodgerblue", s=2);

# title_ = f"$log(y) = f(T, P)$"
# axes.set_title(title_);
axes.set_xlabel("precip"); axes.set_ylabel("temp.");

# %%

# %% [markdown]
# # Add Cubic terms

# %%
X_train_normal["temp_cube"] = X_train_normal["avg_of_dailyAvgTemp_C_AvgOverMonths"] ** 3
X_train_normal["precip_cube"] = X_train_normal["precip_cm_yr"] ** 3
X_train_normal.head(2)

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["precip_cm_yr", "avg_of_dailyAvgTemp_C_AvgOverMonths", 
             "temp_sq", "precip_sq", "temp_X_precip",
             "temp_cube", "precip_cube"]

m5_TP_cube_normal = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                                  regimes = X_train_normal["groupveg"].tolist(),
                                  constant_regi="many", regime_err_sep=False,
                                  name_y=depen_var, name_x=indp_vars)

print (f"{m5_TP_cube_normal.r2.round(2) = }")

m5_TP_cube_normal_results = pd.DataFrame({"Coeff.": m5_TP_cube_normal.betas.flatten(), 
                                          "Std. Error": m5_TP_cube_normal.std_err.flatten(), 
                                          "P-Value": [i[1] for i in m5_TP_cube_normal.t_stat]}, 
                                          index=m5_TP_cube_normal.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_TP_cube_normal_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_TP_cube_normal_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_TP_cube_normal_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_TP_cube_normal_results.index if "Shrubland" in i]

## Subset results to Conifer
veg_ = "Conifer"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_TP_cube_normal_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

## Subset results to Grassland
veg_ = "Grassland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_TP_cube_normal_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

## Subset results to Hardwood
veg_ = "Hardwood"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_TP_cube_normal_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

## Subset results to Shrubland
veg_ = "Shrubland"
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_TP_cube_normal_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat both models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5)
table_ = table_.transpose()
table_.rename(columns={"avg_of_dailyAvgTemp_C_AvgOverMonths": "temp", 
                       "avg_of_dailyAvg_rel_hum_AvgOverMonths" : "RH"}, inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5_TP_cube_normal.predy, m5_TP_cube_normal.u, c="dodgerblue", s=2);

title_ = f"NPP = $f(T, P, T^2, P^2, T^3, P^3, T \u00D7 P)$ "
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%

# %%

# %%

# %%
