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
# copy of Weather_monthly_models_II.ipynb 

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
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as MSE 


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
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
min_bio_dir = rangeland_bio_base

rangeland_base = rangeland_bio_base
rangeland_reOrganized = rangeland_base


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
FID_veg = pd.read_pickle(bio_reOrganized + "FID_veg.sav")
FIDs_weather_ANPP_common = FID_veg["FIDs_weather_ANPP_common"]
FIDs_weather_ANPP_common.head(2)

# %% [markdown]
# ### Subset to common FIDs:

# %%
ANPP    = ANPP[ANPP["fid"].isin(list(FIDs_weather_ANPP_common["fid"]))]
monthly_weather = monthly_weather[monthly_weather["fid"].isin(list(FIDs_weather_ANPP_common["fid"]))]
SF_west = SF_west[SF_west["fid"].isin(list(FIDs_weather_ANPP_common["fid"]))]

# %%
monthly_weather.head(2)

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
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error as MSE

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", TP_cols

# fit model no training data
XGB_model = XGBRegressor()
XGB_model.fit(X_train[indp_vars], y_train)

# %%
# make predictions for test data
y_test_pred = XGB_model.predict(X_test[indp_vars])
y_train_pred = XGB_model.predict(X_train[indp_vars])

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"] = r2_score(y_train.values, y_train_pred).round(4)
df_metric.loc["test", "R2"]  = r2_score(y_test.values, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train.values, y_train_pred)).round(4)
df_metric.loc["test", "RMSE"]  = np.sqrt(MSE(y_test.values, y_test_pred)).round(4)
df_metric

# %% [markdown]
# ### Grid search

# %%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

# %%
# %%time
# from https://machinelearningmastery.com/xgboost-for-regression/
depen_var, indp_vars = "mean_lb_per_acr", TP_cols
XGB_model = XGBRegressor()

# define model evaluation method
cv_ = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(XGB_model, X_train[indp_vars], y_train, 
                         scoring='neg_mean_absolute_error', cv=cv_, n_jobs=-1)

# force scores to be positive
scores = np.abs(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

# %% [markdown]
# ### Parameters
# [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", TP_cols
parameters = {'n_jobs':[-1],
              'eta': [.1, .2, .3, .4, .5], # learning rate
              'gamma' : [0], # default=0, alias: min_split_loss
              'max_depth': [5, 6, 7, 8, 9],
              'lambda': [1, 2, 3, 4, 5, 6], # L2 regularization
              'objective' : ["reg:squarederror"]
             } # , 

XGB_search = GridSearchCV(XGBRegressor(random_state=0), 
                          parameters, cv=5, verbose=1,
                          error_score='raise')

XGB_search.fit(X_train[indp_vars], y_train);
XGB_search.best_params_

# %%
y_train_pred = XGB_search.predict(X_train[indp_vars])
y_test_pred = XGB_search.predict(X_test[indp_vars])

df_metric = pd.DataFrame(columns=["R2", "RMSE"], index=("train", "test"))
df_metric.loc["train", "R2"] = r2_score(y_train.values, y_train_pred).round(4)
df_metric.loc["test", "R2"]  = r2_score(y_test.values, y_test_pred).round(4)
df_metric.loc["train", "RMSE"] = np.sqrt(MSE(y_train.values, y_train_pred)).round(4)
df_metric.loc["test", "RMSE"]  = np.sqrt(MSE(y_test.values, y_test_pred)).round(4)
df_metric

# %%
# XGB_search.best_estimator_.get_booster().get_score(importance_type='weight')

# %%
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
          "axes.labelsize": tick_legend_FontSize * 1.4,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1,
          "ytick.labelsize": tick_legend_FontSize * 1,
          "axes.titlepad": 10}
plt.rcParams.update(params)

# %%
feature_important = XGB_search.best_estimator_.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

FI_df = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
FI_df["new_idx"] = FI_df.index

for idx in FI_df.index:
    month =  (idx.split("_"))[-1]
    if "Temp" in idx:
        var = "temp" 
    elif "precip" in idx:
        var = "precip" 
    FI_df.loc[idx, "new_idx"] = var + "_" + month

FI_df.set_index('new_idx', inplace=True)
FI_df.rename_axis(index=None, inplace=True)

fig, axes = plt.subplots(1, 1, figsize=(2.2, 4), sharex=True, dpi=100)
## plot top 40 features
FI_df.nlargest(40, columns="score").plot(kind='barh', ax=axes, legend=False); 
plt.title("XGBoost feature importance", fontsize=8);
plt.xlabel("score")
# plt.ylabel("variable")

fig.subplots_adjust(top=0.9, bottom=0.1, left=0.28, right=0.9)
file_name = bio_plots + "XGBoost_feature_importance.pdf"
plt.savefig(file_name, dpi=400)

# %%
import shap

# %%
# %%time
A = X_train[indp_vars].copy()
for a_col in A.columns:
    month =  a_col.split("_")[-1]
    if "Temp" in a_col:
        A.rename(columns={a_col: "temp. " + month}, inplace=True)
    elif  "precip" in a_col:
        A.rename(columns={a_col: "precip. " + month}, inplace=True)
    elif  "RH_" in a_col:
        A.rename(columns={a_col: "RH. " + month}, inplace=True)

Xd = xgboost.DMatrix(A, label=y_train)
explainer = shap.TreeExplainer(XGB_search.best_estimator_)
explanation = explainer(Xd)
shap.summary_plot(explanation, A, plot_size=(6, 6), show=False);
plt.savefig(bio_plots + "SHAP_XGBoost_train.pdf", dpi=300);
# plt.show();

# %%

# %%
# %%time
A = X_test[indp_vars].copy()
for a_col in A.columns:
    month =  a_col.split("_")[-1]
    if "Temp" in a_col:
        A.rename(columns={a_col: "temp. " + month}, inplace=True)
    elif  "precip" in a_col:
        A.rename(columns={a_col: "precip. " + month}, inplace=True)
    elif  "RH_" in a_col:
        A.rename(columns={a_col: "RH. " + month}, inplace=True)

Xd = xgboost.DMatrix(A, label=y_test)
explanation = explainer(Xd)

shap.summary_plot(explanation, A, plot_size=(6, 6), show=False);
plt.savefig(bio_plots + "SHAP_XGBoost_test.pdf", dpi=300)

# %%
# fig, axes = plt.subplots(1, 1, figsize=(10, 2), sharex=True, 
#                          gridspec_kw={"hspace": 0.25, "wspace": 0.05}, dpi=dpi_)

# axes.scatter(m5.predy, m5.u, c="dodgerblue", s=2);

# title_ = f"train: NPP = $f(T)$"
# axes.set_title(title_);
# axes.set_xlabel("prediction"); axes.set_ylabel("residual");

# %%
# y_test_pred = rc.pred_via_spreg_regime(regime_col="groupveg", a_model=m5, data_df=X_test_normal)
# print (f"train: {m5.r2.round(2) = }")
# print ("test R2 = {}".format(r2_score(y_test.values, y_test_pred.values).round(2)))

# %% [markdown]
# # Model based on Precip

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", precip_cols

# %% [markdown]
# # Model by Temp and Precip

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", TP_cols

# %% [markdown]
# # Model with interaction terms

# %%
# %%time
depen_var = "mean_lb_per_acr"
indp_vars = [x for x in numeric_cols if ("Temp" in x) or ("precip" in x) ]

# %% [markdown]
# # Temp, precipitation, humidity

# %%
TPH_cols = TP_cols + [x for x in numeric_cols if "rel_hum" in x]

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", TPH_cols

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

# %%

# %% [markdown]
# # log of Y based on Temp, Precip 

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", TP_cols

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
y_train_grass.head(2)

# %%

# %%
depen_var, indp_vars = "mean_lb_per_acr", TP_cols

# %%
tick_legend_FontSize = 4
params = {"legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * 1.4,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 1,
          "ytick.labelsize": tick_legend_FontSize * 1,
          "axes.titlepad": 10}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
