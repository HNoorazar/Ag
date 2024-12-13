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

# %% [markdown]
# copy of Weather_monthly_models.ipynb 
#
# where I deleted the trained model on all data and trained on splited data only. and
# added some $R^2$ for test set

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
from sklearn.metrics import r2_score
import statsmodels.api as sm

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rc("font", family="Palatino")

from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda
import geopandas, contextily
from scipy.stats import ttest_ind

# font = {"size": 10}
# matplotlib.rc("font", **font)

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from sklearn.model_selection import train_test_split
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

from sklearn.metrics import mean_squared_error as MSE

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
FIDs_weather_ANPP_common = pd.read_pickle(bio_reOrganized + "common_FID_NPP_weather_west.sav")
FIDs_weather_ANPP_common = FIDs_weather_ANPP_common["common_FID_NPP_weather_west"]
FIDs_weather_ANPP_common = list(FIDs_weather_ANPP_common["common_FIDs_NPP_WA_west"])

# %% [markdown]
# ### Subset to common FIDs:

# %%
ANPP.head(2)

# %%
monthly_weather.head(2)

# %%
monthly_weather.isna().sum()

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

# %% [markdown]
# # Regression
#
# origin of spreg.OLS_ in ```04_02_2024_NonNormalModelsInterpret.ipynb```.

# %%
print (len(ANPP_weather_wide.fid.unique()))

# %% [markdown]
# # Drop bad Vegs
# ```Sparse```, ```Riparian```, ```Barren-Rock/Sand/Clay```, and ```Conifer```?

# %%
ANPP_weather_wide = pd.merge(ANPP_weather_wide, SF_west[["fid", "groupveg"]], how="left", on=["fid"])
ANPP_weather_wide.head(2)

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
# veg_ = groupveg[1]
# A = ANPP_weather_wide_G.copy()
# A = A[A["groupveg"] == veg_]

# # Let us just do NPP with precips.
# y_var = "mean_lb_per_acr"

# fig, axes = plt.subplots(12, 1, figsize=(10, 36), sharex=True, 
#                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

# for ii in range(1, 13):
#     col_name = "avg_of_dailyAvgTemp_C_" + str(ii)
#     axes[ii-1].scatter(ANPP_weather_wide_G[col_name], ANPP_weather_wide_G[y_var], marker='s', s=2);
#     axes[ii-1].legend(["month " + str(ii)], loc='upper right');

# axes[11].set_xlabel("temperature (" + veg_ + ")");

# fig.subplots_adjust(top=0.96, bottom=0.02, left=0.082, right=0.981)
# file_name = bio_plots + veg_ + "_mothly_temp_NPP.pdf"
# plt.savefig(file_name)

# %%
# # Let us just do NPP with precips.
# veg_ = groupveg[1]
# y_var = "mean_lb_per_acr"

# fig, axes = plt.subplots(12, 1, figsize=(10, 36), sharex=True, 
#                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

# for ii in range(1, 13):
#     col_name = "precip_mm_month_" + str(ii)
#     axes[ii-1].scatter(ANPP_weather_wide_G[col_name], ANPP_weather_wide_G[y_var], marker='s', s=2);
#     axes[ii-1].legend(["month " + str(ii)], loc='upper right');

# axes[11].set_xlabel("precipitation (" + veg_ + ")");
# fig.subplots_adjust(top=0.96, bottom=0.02, left=0.082, right=0.981)
# file_name = bio_plots + veg_ + "_mothly_temp_precip.pdf"
# plt.savefig(file_name)

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
ANPP_weather_wide_G.head(2)

# %%
TP_cols = [x for x in ANPP_weather_wide_G.columns if ("Temp" in x) or ("precip" in x)]

# %%
precip_cols = [x for x in ANPP_weather_wide_G.columns if ("precip" in x)]
temp_cols = [x for x in ANPP_weather_wide_G.columns if ("Temp" in x) ]

# %%
ANPP_weather_wide_G.head(2)

# %%
ANPP_weather_wide_G.reset_index(drop=True, inplace=True)

# %% [markdown]
# #### Model only the years where precipitation is less than 600 mm.

# %%
bad_years_idx = set()
for a_col in precip_cols:
    bad_yr = list(ANPP_weather_wide_G[ANPP_weather_wide_G[a_col] > 600].index)
    bad_years_idx.update(bad_yr)

print (f"{len(bad_years_idx) = }")
bad_years_idx = list(bad_years_idx)

# %%
ANPP_weather_wide_G_less600Prec = ANPP_weather_wide_G.loc[~ANPP_weather_wide_G.index.isin(bad_years_idx)].copy()
len(ANPP_weather_wide_G) - len(ANPP_weather_wide_G_less600Prec)

# %% [markdown]
# # Split and normalize

# %%
ANPP_weather_wide_G.head(2)

# %% [markdown]
# ### Lets add interactions and then split

# %%
for ii in range(0, 12):
    new_col = "temp_X_precip_month_" + str(ii+1)
    ANPP_weather_wide_G[new_col] = ANPP_weather_wide_G[temp_cols[ii]] * ANPP_weather_wide_G[precip_cols[ii]]

# %%
print (ANPP_weather_wide_G.shape)
ANPP_weather_wide_G.head(2)

# %%
numeric_cols = [x for x in ANPP_weather_wide_G.columns if ("Temp" in x) or ("precip" in x) 
                or ("rel_hum" in x) or ("thi" in x)]

numeric_cols = sorted(numeric_cols)
non_numeric_cols = [x for x in ANPP_weather_wide_G.columns if (not x in numeric_cols)]

# %%
# re-order columns
new_order = non_numeric_cols + sorted(numeric_cols)
ANPP_weather_wide_G = ANPP_weather_wide_G[new_order]
ANPP_weather_wide_G.head(2)

# %%

# %%
depen_var = "mean_lb_per_acr"
indp_vars = ["groupveg"] + numeric_cols

y_df = ANPP_weather_wide_G[depen_var].copy()
indp_df = ANPP_weather_wide_G[indp_vars].copy()

# %%

# %%
X_train, X_test, y_train, y_test = train_test_split(indp_df, y_df, test_size=0.3, random_state=42)
X_train.head(2)

# %%
train_idx = list(X_train.index)
test_idx = list(X_test.index)

# %%
# standard_indp = preprocessing.scale(all_df[explain_vars_herb]) # this is biased
means = X_train[numeric_cols].mean()
stds = X_train[numeric_cols].std(ddof=1)

X_train_normal = X_train.copy()
X_test_normal = X_test.copy()

X_train_normal[numeric_cols] = (X_train_normal[numeric_cols] - means) / stds
X_test_normal[numeric_cols]  = (X_test_normal[numeric_cols]  - means) / stds
X_train_normal.head(2)

# %% [markdown]
# # Model normalized data

# %%
ANPP_weather_wide_G.head(2)

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

# %% [markdown]
# # Model based on Temp

# %%

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", temp_cols

m5 = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                       regimes = X_train_normal["groupveg"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(),
                           "Std. Error": m5.std_err.flatten(),
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x)

Conifer_m   = [i for i in m5_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]

veg_ = "Conifer" ## Subset results to Conifer
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

veg_ = "Grassland" ## Subset results to Grassland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

veg_ = "Hardwood" ## Subset results to Hardwood
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

veg_ = "Shrubland" ## Subset results to Shrubland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5).transpose()
table_.rename(columns=lambda x: x.replace('avg_of_dailyAvgTemp_C_', 'temp_'), inplace=True)
table_

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);

title_ = f"train: NPP = $f(T)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
y_test_pred  = rc.pred_via_spreg_regime(regime_col="groupveg", a_model=m5, data_df=X_test_normal)
y_train_pred = m5.predy

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"]   = m5.r2.round(4)
df_metric.loc["test",  "R2"]   = r2_score(y_test, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train, y_train_pred)).round(4)
df_metric.loc["test",  "RMSE"] = np.sqrt(MSE(y_test, y_test_pred)).round(4)
df_metric

# %% [markdown]
# # Model based on Precip

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", precip_cols

m5 = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                       regimes = X_train_normal["groupveg"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), 
                           "Std. Error": m5.std_err.flatten(), 
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]

veg_ = "Conifer" ## Subset results to Conifer
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

veg_ = "Grassland" ## Subset results to Grassland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

veg_ = "Hardwood" ## Subset results to Hardwood
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

veg_ = "Shrubland" ## Subset results to Shrubland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5).transpose()
table_.rename(columns=lambda x: x.replace('precip_mm_month_', 'precip_'), inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);

title_ = f"train: NPP = $f(P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
del(y_test_pred)
y_test_pred  = rc.pred_via_spreg_regime(regime_col="groupveg", a_model=m5, data_df=X_test_normal)
y_train_pred = m5.predy

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"]   = m5.r2.round(4)
df_metric.loc["test",  "R2"]   = r2_score(y_test, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train, y_train_pred)).round(4)
df_metric.loc["test",  "RMSE"] = np.sqrt(MSE(y_test, y_test_pred)).round(4)
df_metric

# %% [markdown]
# # Model by Temp and Precip

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", TP_cols

m5 = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                       regimes = X_train_normal["groupveg"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), 
                           "Std. Error": m5.std_err.flatten(), 
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x)

## Extract variables for each veg type
Conifer_m   = [i for i in m5_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]

veg_ = "Conifer" ## Subset results to Conifer
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

veg_ = "Grassland" ## Subset results to Grassland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

veg_ = "Hardwood" ## Subset results to Hardwood
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

veg_ = "Shrubland" ## Subset results to Shrubland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5).transpose()
table_.rename(columns=lambda x: x.replace('precip_mm_month_', 'precip_'), inplace=True)
table_.rename(columns=lambda x: x.replace('avg_of_dailyAvgTemp_C_', 'temp_'), inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);

title_ = f"train: NPP = $f(T, P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
del(y_test_pred)
y_test_pred  = rc.pred_via_spreg_regime(regime_col="groupveg", a_model=m5, data_df=X_test_normal)
y_train_pred = m5.predy

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"]   = m5.r2.round(4)
df_metric.loc["test",  "R2"]   = r2_score(y_test, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train, y_train_pred)).round(4)
df_metric.loc["test",  "RMSE"] = np.sqrt(MSE(y_test, y_test_pred)).round(4)
df_metric

# %%

# %%
coeffs = table_.loc["Grassland"].loc["Coeff."]
temp_idx = [x for x in temp_coeffs.index if "temp" in x]
precip_idx = [x for x in temp_coeffs.index if "precip" in x]
# RH_idx = [x for x in temp_coeffs.index if "RH" in x]

# coeffs[temp_idx], coeffs[precip_idx], coeffs[RH_idx]
colnames = ["month_" + str(ii) for ii in range(1, 13)]
coeff_df = pd.DataFrame(columns=colnames, index=["temp", "precip"])
for ii in range(1, 13):
    coeff_df.loc["temp", "month_" + str(ii)] = coeffs[temp_idx].loc["temp_"+str(ii)].round(0)
    coeff_df.loc["precip", "month_" + str(ii)] = coeffs[precip_idx].loc["precip_"+str(ii)].round(0)
    # coeff_df.loc["RH", "month_" + str(ii)] = coeffs[RH_idx].loc["RH_"+str(ii)].round(2)
    
coeff_df

# %% [markdown]
# # Model with interaction terms

# %%
# %%time
# takes 3 minutes
depen_var = "mean_lb_per_acr"
indp_vars = [x for x in numeric_cols if ("Temp" in x) or ("precip" in x) ]

m5 = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                       regimes = X_train_normal["groupveg"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), 
                           "Std. Error": m5.std_err.flatten(), 
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x)
## Extract variables for each veg type
Conifer_m   = [i for i in m5_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]

veg_ = "Conifer" ## Subset results to Conifer
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

veg_ = "Grassland" ## Subset results to Grassland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

veg_ = "Hardwood" ## Subset results to Hardwood
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

veg_ = "Shrubland" ## Subset results to Shrubland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5).transpose()
table_.rename(columns=lambda x: x.replace('precip_mm_month_', 'precip_'), inplace=True)
table_.rename(columns=lambda x: x.replace('avg_of_dailyAvgTemp_C_', 'temp_'), inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                        gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);

title_ = f"train: NPP = $f(T, P, T \u00D7 P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
del(y_test_pred)
y_test_pred  = rc.pred_via_spreg_regime(regime_col="groupveg", a_model=m5, data_df=X_test_normal)
y_train_pred = m5.predy

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"]   = m5.r2.round(4)
df_metric.loc["test",  "R2"]   = r2_score(y_test, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train, y_train_pred)).round(4)
df_metric.loc["test",  "RMSE"] = np.sqrt(MSE(y_test, y_test_pred)).round(4)
df_metric

# %% [markdown]
# # Temp, precipitation, humidity

# %%
# %%time
# takes 3 minutes
TPH_cols = TP_cols + [x for x in numeric_cols if "rel_hum" in x]
depen_var, indp_vars = "mean_lb_per_acr", TPH_cols

m5 = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                       regimes = X_train_normal["groupveg"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), 
                           "Std. Error": m5.std_err.flatten(), 
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x)
## Extract variables for each veg type
Conifer_m   = [i for i in m5_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]

veg_ = "Conifer" ## Subset results to Conifer
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

veg_ = "Grassland" ## Subset results to Grassland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

veg_ = "Hardwood" ## Subset results to Hardwood
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

veg_ = "Shrubland" ## Subset results to Shrubland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5).transpose()
table_.rename(columns=lambda x: x.replace('precip_mm_month_', 'precip_'), inplace=True)
table_.rename(columns=lambda x: x.replace('avg_of_dailyAvgTemp_C_', 'temp_'), inplace=True)
table_.rename(columns=lambda x: x.replace('avg_of_dailyAvg_rel_hum_', 'RH_'), inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)
axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);
title_ = f"train: NPP = $f(T, P, RH)$"
axes.set_title(title_); axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %% [markdown]
# # Best model?

# %%
del(y_test_pred)
y_test_pred  = rc.pred_via_spreg_regime(regime_col="groupveg", a_model=m5, data_df=X_test_normal)
y_train_pred = m5.predy

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"]   = m5.r2.round(4)
df_metric.loc["test",  "R2"]   = r2_score(y_test, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train, y_train_pred)).round(4)
df_metric.loc["test",  "RMSE"] = np.sqrt(MSE(y_test, y_test_pred)).round(4)
df_metric

# %%
coeffs = table_.loc["Grassland"].loc["Coeff."]
temp_idx = [x for x in temp_coeffs.index if "temp" in x]
precip_idx = [x for x in temp_coeffs.index if "precip" in x]
RH_idx = [x for x in temp_coeffs.index if "RH" in x]

# coeffs[temp_idx], coeffs[precip_idx], coeffs[RH_idx]
colnames = ["month_" + str(ii) for ii in range(1, 13)]
coeff_df = pd.DataFrame(columns=colnames, index=["temp", "precip", "RH"])
for ii in range(1, 13):
    coeff_df.loc["temp", "month_" + str(ii)] = coeffs[temp_idx].loc["temp_"+str(ii)].round(0)
    coeff_df.loc["precip", "month_" + str(ii)] = coeffs[precip_idx].loc["precip_"+str(ii)].round(0)
    coeff_df.loc["RH", "month_" + str(ii)] = coeffs[RH_idx].loc["RH_"+str(ii)].round(0)
    
coeff_df

# %%
coeff_df.loc["RH"].values

# %%

# %%

# %% [markdown]
# # Add square terms

# %%
print (X_train_normal.shape)
X_train_normal.head(2)

# %%
for ii in range(0, 12):
    temp_col = temp_cols[ii]
    month = temp_col.split("_")[-1]
    new_temp = "temp_" + month + "_sq"
    X_train_normal[new_temp] = X_train_normal[temp_col] ** 2
    
    precip_col = precip_cols[ii]
    month = precip_col.split("_")[-1]
    new_P = "precip_" + month + "_sq"
    X_train_normal[new_P] = X_train_normal[precip_col] ** 2

X_train_normal.head(2)

# %%
for ii in range(0, 12):
    temp_col = temp_cols[ii]
    month = temp_col.split("_")[-1]
    new_temp = "temp_" + month + "_sq"
    X_test_normal[new_temp] = X_test_normal[temp_col] ** 2
    
    precip_col = precip_cols[ii]
    month = precip_col.split("_")[-1]
    new_P = "precip_" + month + "_sq"
    X_test_normal[new_P] = X_test_normal[precip_col] ** 2

X_test_normal.head(2)

# %% [markdown]
# ### Takes too long

# %%
# %%time
depen_var = "mean_lb_per_acr"
indp_vars = [x for x in X_train_normal.columns if ("precip" in x) or ("temp" in x) or ("Temp" in x)]

m5 = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                       regimes = X_train_normal["groupveg"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), 
                           "Std. Error": m5.std_err.flatten(), 
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x)
## Extract variables for each veg type
Conifer_m   = [i for i in m5_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]

veg_ = "Conifer" ## Subset results to Conifer
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

veg_ = "Grassland" ## Subset results to Grassland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

veg_ = "Hardwood" ## Subset results to Hardwood
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

veg_ = "Shrubland" ## Subset results to Shrubland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5).transpose()
table_.rename(columns=lambda x: x.replace('precip_mm_month_', 'precip_'), inplace=True)
table_.rename(columns=lambda x: x.replace('avg_of_dailyAvgTemp_C_', 'temp_'), inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);

title_ = f"train: NPP = $f(T, P, T^2, P^2, T \u00D7 P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
del(y_test_pred)
y_test_pred  = rc.pred_via_spreg_regime(regime_col="groupveg", a_model=m5, data_df=X_test_normal)
y_train_pred = m5.predy

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"]   = m5.r2.round(4)
df_metric.loc["test",  "R2"]   = r2_score(y_test, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train, y_train_pred)).round(4)
df_metric.loc["test",  "RMSE"] = np.sqrt(MSE(y_test, y_test_pred)).round(4)
df_metric

# %%

# %% [markdown]
# # log of Y based on Temp, Precip 

# %%
# %%time
# takes 1 minute
depen_var, indp_vars = "mean_lb_per_acr", TP_cols

m5 = spreg.OLS_Regimes(y=y_train.values ** (1. / 3), x=X_train_normal[indp_vars].values, 
                       regimes = X_train_normal["groupveg"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), 
                           "Std. Error": m5.std_err.flatten(), 
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x)
## Extract variables for each veg type
Conifer_m   = [i for i in m5_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]

veg_ = "Conifer" ## Subset results to Conifer
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

veg_ = "Grassland" ## Subset results to Grassland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

veg_ = "Hardwood" ## Subset results to Hardwood
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

veg_ = "Shrubland" ## Subset results to Shrubland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5).transpose()
table_.rename(columns=lambda x: x.replace('precip_mm_month_', 'precip_'), inplace=True)
table_.rename(columns=lambda x: x.replace('avg_of_dailyAvgTemp_C_', 'temp_'), inplace=True)
table_

# %%
fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)
axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);
title_ = f"train: cubic root$(y) = f(T, P)$"
axes.set_title(title_); 
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
del(y_test_pred)
y_test_pred  = rc.pred_via_spreg_regime(regime_col="groupveg", a_model=m5, data_df=X_test_normal)
y_train_pred = m5.predy

y_test_transformed = (y_test.values).reshape(-1, 1)

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"]   = m5.r2.round(4)
df_metric.loc["test",  "R2"]   = r2_score(y_test_transformed, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train.values, y_train_pred**3)).round(4)
df_metric.loc["test",  "RMSE"] = np.sqrt(MSE(y_test_transformed, y_test_pred**3)).round(4)
df_metric

# %%
test_u = y_test_transformed - y_test_pred

fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
                         gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

axes.scatter(y_test_pred, test_u, c="dodgerblue", s=2);

title_ = f"test: cubic root$(y) = f(T, P)$"
axes.set_title(title_);
axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %% [markdown]
# # Model only Grassland
#
# with ```Temp``` and ```precip```.

# %%
X_train_normal.head(2)

# %%
X_train_normal_grass = X_train_normal[X_train_normal["groupveg"] == "Grassland"]
y_train_grass = y_train.loc[X_train_normal_grass.index]
X_train_normal_grass.head(2)

# %%
depen_var, indp_vars = "mean_lb_per_acr", TP_cols

m5 = spreg.OLS(y=y_train_grass.values, x=X_train_normal_grass[indp_vars].values, 
               name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), 
                           "Std. Error": m5.std_err.flatten(), 
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x).transpose()
m5_results

# %%
X = X_train_normal_grass[indp_vars]
X = sm.add_constant(X)
Y = y_train_grass.values.astype(float)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
ks_result.summary()

# %%
X_test_normal_grass = X_test_normal[X_test_normal["groupveg"] == "Grassland"]
y_test_grass = y_test.loc[X_test_normal_grass.index]
X_test_normal_grass.head(2)

# %%
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, 
                         gridspec_kw={"hspace": 0.45, "wspace": 0.05}, dpi=dpi_)

axes[0].scatter(ks_result.fittedvalues, ks_result.resid, c="dodgerblue", s=2);

title_ = f"train LS: $(y) = f(T, P)$"
axes[0].set_title(title_); axes[0].set_ylabel("residual");
##############################################################################
##############################################################################
Xnew = X_test_normal_grass[indp_vars]
Xnew = sm.add_constant(Xnew)

y_test_grass_pred = ks_result.predict(Xnew)
u = y_test_grass - y_test_grass_pred
###########################################
axes[1].scatter(y_test_grass_pred, u, c="dodgerblue", s=2);

title_ = f"test LS: $(y) = f(T, P)$"
axes[1].set_title(title_);
axes[1].set_xlabel("prediction"); axes[1].set_ylabel("residual");
plt.xlim(-1000, 7500)
fig.subplots_adjust(top=0.9, bottom=0.2, left=0.12, right=0.981, wspace=-0.2, hspace=0.5)
file_name = bio_plots + "grassland_OLS.png"
plt.savefig(file_name, dpi=400)

# %%
Xnew = X_test_normal_grass[indp_vars]
Xnew = sm.add_constant(Xnew)
y_test_pred = ks_result.predict(Xnew)
y_train_pred = ks_result.predict(X)

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"] = ks_result.rsquared.round(5)
df_metric.loc["test", "R2"]  = r2_score(y_test_grass, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train_grass, y_train_pred)).round(4)
df_metric.loc["test", "RMSE"]  = np.sqrt(MSE(y_test_grass, y_test_pred)).round(4)
df_metric

# %% [markdown]
# ## Weighted least square

# %%
from statsmodels.formula.api import ols

# %%
d = {"resid_": ks_result.resid.abs(), "preds_" : ks_result.fittedvalues}
df_2 = pd.DataFrame(d)

y_wt = df_2['resid_']
X_wt = df_2['preds_']
X_wt = sm.add_constant(X_wt) # add constant to predictor variables
fit_2 = sm.OLS(y_wt, X_wt).fit() # fit linear regression model

# print(fit_2.summary());
wt_2 = 1 / fit_2.fittedvalues**2

# %%
# fit weighted least squares regression model
X = X_train_normal_grass[indp_vars]
X = sm.add_constant(X)
Y = y_train_grass.values.astype(float)

fit_wls = sm.WLS(Y, X, weights=wt_2).fit()

# view summary of weighted least squares regression model
print(fit_wls.summary())

# %%

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, 
                         gridspec_kw={"hspace": 0.45, "wspace": 0.05}, dpi=dpi_)

axes[0].scatter(fit_wls.fittedvalues, fit_wls.resid, c="dodgerblue", s=2);

title_ = f"train weighted LS: $(y) = f(T, P)$"
axes[0].set_title(title_); axes[0].set_ylabel("residual");
##############################################################################
##############################################################################
Xnew = X_test_normal_grass[indp_vars]
Xnew = sm.add_constant(Xnew)

y_test_grass_pred_weighted = fit_wls.predict(Xnew)
u = y_test_grass - y_test_grass_pred_weighted
###########################################
axes[1].scatter(y_test_grass_pred_weighted, u, c="dodgerblue", s=2);

title_ = f"test weighted LS: $(y) = f(T, P)$"
axes[1].set_title(title_);
axes[1].set_xlabel("prediction"); axes[1].set_ylabel("residual");
plt.xlim(-1000, 7500)
fig.subplots_adjust(top=0.9, bottom=0.2, left=0.12, right=0.981)
file_name = bio_plots + "grassland_WLS.png"
plt.savefig(file_name, dpi=400)

# %%

# %%
