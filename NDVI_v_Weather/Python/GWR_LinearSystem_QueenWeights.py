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
# It seems all the libraryes want to do it the bandwidth way; no pre-specified weight matrix!
# Lets just do it outselves 
#
# $$\beta(u_i, v_i) = (X^T W(u_i, v_i) X) ^ {-1} X^T W(u_i, v_i) y$$

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
from datetime import datetime, date
from scipy import sparse
import statsmodels.api as sm
current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
# import rangeland_core as rc

# %%
dpi_ = 300

plot_dir = "/Users/hn/Documents/01_research_data/RangeLand/Mike_Results/plots/"
os.makedirs(plot_dir, exist_ok=True)

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

data_dir_base = research_db + "RangeLand/Data/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"

NDVI_weather_data_dir = research_db + "/NDVI_v_Weather/data/"

bio_data_dir_base = research_db + "/RangeLand_bio/Data/"
bio_reOrganized_dir = bio_data_dir_base + "reOrganized/"

# %%
# %%time
filename = (NDVI_weather_data_dir + "monthly_NDVI_county_weight_for_GWR_9trainYears_sparse.sav")

GWR_9trainYears_sparse = pd.read_pickle(filename)
print (GWR_9trainYears_sparse.keys())

# %%
x_train = GWR_9trainYears_sparse["x_train"]
y_train = GWR_9trainYears_sparse["y_train"]

x_test  = GWR_9trainYears_sparse["x_test"]
y_test  = GWR_9trainYears_sparse["y_test"]

weightMatrix = GWR_9trainYears_sparse["weightMatrix"]
test_weightMatrix = GWR_9trainYears_sparse["test_weightMatrix"]

# %%
print (type(weightMatrix));
csr_weightMatrix = weightMatrix.tocsr()
test_weightMatrix = test_weightMatrix.tocsr()
type(csr_weightMatrix)

# %% [markdown]
# ## Predict
#
# Write a function that does prediction. What we want is $$\hat{y}(u_i, v_i) = x_{\text{new}}^T \beta(u_i, v_i)$$

# %%
from sklearn.metrics import r2_score

def GWR_predict(X_, betas):
    return X_.dot(betas)

def rmse(v, w):
    rmse = np.sqrt(np.mean((v - w)**2))
    return rmse


# %%

# %%
# %%time
# are_equal = csr_weightMatrix == weightMatrix

# # If the matrices are equal, the result will be a sparse matrix of booleans, 
# # where True indicates equality at that position and False otherwise.
# # To check if all elements are equal, one can use the .all() method.
# print(are_equal.toarray())
# print((csr_matrix1 != coo_matrix1).nnz == 0)

# %%
indp_vars = ['tavg_avg', 'ppt', 'tavg_avg_lag1', 'ppt_lag1', 'MODIS_NDVI_lag1']
y_var = 'MODIS_NDVI'

# %%
X = x_train[indp_vars].copy()
X["const"] = 1 # add constant term
X.head(2)

# %%
# Compute left hand side above: matrix of A in Ax=y
WX = csr_weightMatrix.dot(X)
XtWX = X.T.dot(WX)

# Compute right hand side above
Wy = csr_weightMatrix.dot(y_train)
XtWy = X.T.dot(Wy)

## Moment of Truth
betas = np.linalg.solve(XtWX, XtWy)
betas

# %%
model = sm.OLS(XtWy, XtWX)
results = model.fit()
p_values = results.pvalues
results.summary()

# %%
X_test = x_test[indp_vars].copy()
# add constant term to X matrix
X_test["const"] = 1

y_hat = GWR_predict(X_=X_test, betas=betas)

print(f"R2: {r2_score(y_test, y_hat):.2f}")
print(f"RMSE: {rmse(y_test, y_hat):.2f}")
print(f"RMSE^2: {rmse(y_test, y_hat)**2:.4f}")

# %% [markdown]
# ### Leave Current month's temp and preip. out
#
# Does this makes sense? precipitation of last month might be more important precipitation of this months.

# %%
indp_vars = ['tavg_avg_lag1', 'ppt_lag1', 'MODIS_NDVI_lag1']
y_var = 'MODIS_NDVI'

X = x_train[indp_vars].copy()
X["const"] = 1

# %%
# Compute left hand side above: matrix of A in Ax=y
WX = csr_weightMatrix.dot(X)
XtWX = X.T.dot(WX)

# Compute right hand side above
Wy = csr_weightMatrix.dot(y_train)
XtWy = X.T.dot(Wy)

## Moment of Truth
betas = np.linalg.solve(XtWX, XtWy)
betas

# %%
model = sm.OLS(XtWy, XtWX)
results = model.fit()
p_values = results.pvalues
results.summary()

# %%
X_test = x_test[indp_vars].copy()
# add constant term to X matrix
X_test["const"] = 1

y_hat = GWR_predict(X_=X_test, betas=betas)

print(f"R2: {r2_score(y_test, y_hat):.2f}")
print(f"RMSE: {rmse(y_test, y_hat):.2f}")
print(f"RMSE^2: {rmse(y_test, y_hat)**2:.4f}")

# %% [markdown]
# ### Leave last months precipitation and temp out

# %%
indp_vars = ['tavg_avg', 'ppt', 'MODIS_NDVI_lag1']
y_var = 'MODIS_NDVI'

X = x_train[indp_vars].copy()
X["const"] = 1

# %%
# Compute left hand side above: matrix of A in Ax=y
WX = csr_weightMatrix.dot(X)
XtWX = X.T.dot(WX)

# Compute right hand side above
Wy = csr_weightMatrix.dot(y_train)
XtWy = X.T.dot(Wy)

## Moment of Truth
betas = np.linalg.solve(XtWX, XtWy)
betas

# %%
model = sm.OLS(XtWy, XtWX)
results = model.fit()
p_values = results.pvalues
results.summary()

# %%
X_test = x_test[indp_vars].copy()
# add constant term to X matrix
X_test["const"] = 1

y_hat = GWR_predict(X_=X_test, betas=betas)

print(f"R2: {r2_score(y_test, y_hat):.2f}")
print(f"RMSE: {rmse(y_test, y_hat):.2f}")
print(f"RMSE^2: {rmse(y_test, y_hat)**2:.4f}")

# %%

# %%
