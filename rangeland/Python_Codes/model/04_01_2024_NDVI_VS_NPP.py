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
import shutup

shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

from datetime import datetime, date

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# %%
from statsmodels.formula.api import ols
# fit = ols('inventory ~ C(state_dummy_int) + max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982', 
#           data=all_df_normalized_needed).fit() 

# fit.summary()

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
print (len(state_fips_SoI))
state_fips_SoI.head(2)

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_df = all_data_dict["all_df_outerjoined"]
all_df_normal = all_data_dict["all_df_outerjoined_normalized"]

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df_normal = all_df_normal[all_df_normal.state_fips.isin(list(state_fips_SoI.state_fips))].copy()

all_df.reset_index(drop=True, inplace=True)
all_df_normal.reset_index(drop=True, inplace=True)

# %%

# %%
npp_cols  = [x for x in all_df_normal.columns if "npp" in x ]
NDVI_cols = [x for x in all_df_normal.columns if "ndvi" in x ]
NDVI_cols = [x for x in NDVI_cols if "modis" in x ]
NDVI_cols

# %%
npp_cols

# %%
data_dir_base

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})

axs.grid(axis="y", which="both")

axs.scatter(all_df_normal.max_ndvi_in_year_modis, all_df_normal.unit_matt_npp, 
            s = 20, c="r", marker="x");

axs.set_xlabel("max NDVI in a year (normalized)")
axs.set_ylabel("unit Matt's NPP in a year (normalized)")

plots_dir = data_dir_base + "00_plots/"
fig_name = plots_dir + "NormalNDVI_mattUnitNPP_Scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
fit = ols('unit_matt_npp ~ max_ndvi_in_year_modis', data=all_df_normal).fit() 
fit.summary()

# %%
fit.params['Intercept'].round(2)
fit.params['max_ndvi_in_year_modis'].round(2)

# %%

# # This automatically considers intercept, i.e. no need to add a column on 1s to Xs

# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# # Create linear regression object
# regr = linear_model.LinearRegression()

# df_ = all_df[["max_ndvi_in_year_modis", "unit_matt_npp"]].copy()
# df_.dropna(how="any", inplace=True)
# # Train the model using the training sets
# regr.fit(df_.max_ndvi_in_year_modis.values.reshape(-1, 1), 
#          df_.unit_matt_npp.values.reshape(-1, 1))

# # Make predictions using the testing set
# y_pred = regr.predict(df_.max_ndvi_in_year_modis.values.reshape(-1, 1))

# print("Coefficients: \n", regr.coef_)
# print("Mean squared error: %.2f" % mean_squared_error(y_pred, df_.unit_matt_npp.values))
# print("Coefficient of determination: %.2f" % r2_score(y_pred, df_.unit_matt_npp.values))

# %%
X = all_df[["max_ndvi_in_year_modis", "unit_matt_npp"]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X["unit_matt_npp"].astype(float)
X = X.drop("unit_matt_npp", axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
ks_result.summary()

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})

axs.grid(axis="y", which="both")

axs.scatter(all_df.max_ndvi_in_year_modis, all_df.unit_matt_npp, 
            s = 20, c="r", marker="x");

axs.plot(X["max_ndvi_in_year_modis"], y_pred, color="blue", linewidth=3)

constant = ks_result.params["const"].round(2)
slope = ks_result.params["max_ndvi_in_year_modis"].round(2)
plt.text(0.2, 5100, f'y = {constant} + {slope} $x$')

axs.set_xlabel("max NDVI in a year")
axs.set_ylabel("unit Matt's NPP in a year")

plots_dir = data_dir_base + "00_plots/"
fig_name = plots_dir + "NDVI_mattUnitNPP_Scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")


# %% [markdown]
# # Transfered model

# %%
X = df_["max_ndvi_in_year_modis"]
X = sm.add_constant(X)
Y = np.sqrt(df_["unit_matt_npp"].astype(float))
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
ks_result.summary()


fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05})

axs.grid(axis="y", which="both")

axs.scatter(X.max_ndvi_in_year_modis, Y, 
            s = 20, c="dodgerblue", marker="x");

axs.plot(X["max_ndvi_in_year_modis"], y_pred, color="red", linewidth=3)


constant = ks_result.params["const"].round(2)
slope = ks_result.params["max_ndvi_in_year_modis"].round(2)
plt.text(0.2, 70, f'y = {constant} + {slope} $x$')

axs.set_xlabel("max NDVI in a year")
axs.set_ylabel("sqrt unit Matt's NPP in a year")

plots_dir = data_dir_base + "00_plots/"
fig_name = plots_dir + "NDVI_sqrtMattUnitNPP_Scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
X = df_["max_ndvi_in_year_modis"]
X = sm.add_constant(X)
Y = np.log(df_["unit_matt_npp"].astype(float))
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
ks_result.summary()

fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05})

axs.grid(axis="y", which="both")

axs.scatter(X.max_ndvi_in_year_modis, Y, 
            s = 20, c="dodgerblue", marker="x");

axs.plot(X["max_ndvi_in_year_modis"], y_pred, color="red", linewidth=3)


constant = ks_result.params["const"].round(2)
slope = ks_result.params["max_ndvi_in_year_modis"].round(2)
plt.text(0.2, 8.5, f'y = {constant} + {slope} $x$')
axs.set_xlabel("max NDVI in a year")
axs.set_ylabel("log unit Matt's NPP in a year")

plots_dir = data_dir_base + "00_plots/"
fig_name = plots_dir + "NDVI_LogMattUnitNPP_Scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %% [markdown]
# # Square model

# %%
# fit a second degree polynomial to the economic data
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot

# define the true objective function
def objective(x, a, b, c):
    return a * x + b * x**2 + c

# choose the input and output variables
x, y = df_["max_ndvi_in_year_modis"], df_["unit_matt_npp"].astype(float)
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b, c = popt

print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
# plot input vs output

fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})

axs.grid(axis="y", which="both")

axs.scatter(all_df.max_ndvi_in_year_modis, all_df.unit_matt_npp, 
            s = 20, c="dodgerblue", marker="x");


x_line = arange(min(x), max(x), 0.01)
y_line = objective(x_line, a, b, c)
axs.plot(x_line, y_line, color="r", linewidth=4)
plt.text(0.2, 5100, f'y = {c.round(1)} + {a.round(1)} $x$ + {b.round(1)} $x^2$')

axs.set_xlabel("max NDVI in a year")
axs.set_ylabel("unit Matt's NPP in a year")

plots_dir = data_dir_base + "00_plots/"
fig_name = plots_dir + "squaredNDVI_mattUnitNPP_Scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")


# %%
def objective(x, a, b, c):
    return a * (x**2) + b*x + c

# choose the input and output variables
x, y = df_["max_ndvi_in_year_modis"], df_["unit_matt_npp"].astype(float)
# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b, c = popt

print(f'y =  {a.round(1)} * x^2 + {b} * x  + {c}')

fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})

axs.grid(axis="y", which="both")

axs.scatter(all_df.max_ndvi_in_year_modis, all_df.unit_matt_npp, 
            s = 20, c="dodgerblue", marker="x");


x_line = arange(min(x), max(x), 0.01)
y_line = objective(x_line, a, b, c)
axs.plot(x_line, y_line, color="r", linewidth=4)
plt.text(0.2, 5100, f'y = {a.round(1)} $x^2$ + {b.round(1)} $x$ + {c.round(1)}')

axs.set_xlabel("max NDVI in a year")
axs.set_ylabel("unit Matt's NPP in a year")

plots_dir = data_dir_base + "00_plots/"
fig_name = plots_dir + "squaredNDVI_mattUnitNPP_Scatter.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
del(x_line)

# %%

# X = df_["max_ndvi_in_year_modis"].copy()
# X = sm.add_constant(X)
# X["max_ndvi_in_year_modis_sq"] = X["max_ndvi_in_year_modis"] ** 2
# Y = (df_["unit_matt_npp"].astype(float))

# ks = sm.OLS(Y, X)
# ks_result = ks.fit()
# ks_result.summary()

# fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
# axs.grid(axis="y", which="both")

# axs.scatter(all_df.max_ndvi_in_year_modis, all_df.unit_matt_npp, 
#             s = 20, c="dodgerblue", marker="x");

# x_line = pd.DataFrame(arange(min(x), max(x), 0.01))
# x_line.columns = ["x"]
# x_line["x_sq"] = x_line["x"]**2
# x_line = sm.add_constant(x_line)
# x_line
# y_line = ks_result.predict(x_line)
# axs.plot(x_line["x"], (y_line), color="r", linewidth=4)

# axs.set_xlabel("max NDVI in a year")
# axs.set_ylabel("unit Matt's NPP in a year")

# fig_name = plots_dir + "test.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")


# %%
fit = ols('unit_matt_npp ~ max_ndvi_in_year_modis', data=all_df).fit() 
fit.summary()

# %%
fit = ols('total_matt_npp ~ max_ndvi_in_year_modis', data=all_df_normal).fit() 
fit.summary()

# %%
# # !pip3 install plspm

# %%
import plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode


# %%

# %%
