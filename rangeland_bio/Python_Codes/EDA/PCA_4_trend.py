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

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
import statsmodels.api as sm
from datetime import datetime
import matplotlib.ticker as plticker
from sklearn.decomposition import PCA



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
y_var = "mean_lb_per_acr"
dpi_ = 200
save_dpi=400

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_Spearman_no2012 = pd.read_pickle(filename)
ANPP_MK_Spearman_no2012 = ANPP_MK_Spearman_no2012["ANPP_MK_df"]
ANPP_MK_Spearman_no2012.head(2)

# %%
bpszone_ANPP_no2012 = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
print (bpszone_ANPP_no2012["Date"])
bpszone_ANPP_no2012 = bpszone_ANPP_no2012["bpszone_ANPP"]
bpszone_ANPP_no2012.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP_no2012.rename(columns={"area": "area_sqMeter", 
                                    "count": "pixel_count",
                                    "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP_no2012.sort_values(by=['fid', 'year'], inplace=True)
bpszone_ANPP_no2012.reset_index(drop=True, inplace=True)
bpszone_ANPP_no2012.head(2)

# %%
all_FIDs_list = list(bpszone_ANPP_no2012["fid"].unique())

# %%

# %%
################# Compute PCA for a field

a_fid = all_FIDs_list[0] # 22591 #  all_FIDs_list[20] # try 1 and 100, 20

X = bpszone_ANPP_no2012[["fid", "year", "mean_lb_per_acr"]].copy()
X = X[X["fid"] == a_fid].copy()
X = X[["year", "mean_lb_per_acr"]].copy()
# X = X.select_dtypes(include='number')

# We better standardize first
# X["year"] = X["year"] / X["year"].std()
# X["mean_lb_per_acr"] = X["mean_lb_per_acr"] /X["mean_lb_per_acr"].std()

pca = PCA(n_components=1) 
pca.fit(X) # Fit the PCA model

# Access the loadings for the first principal component
# loadings = pca.components_[0]
loadings = pd.DataFrame(pca.components_.T, columns=['PC1'], index=X.columns)
slope = loadings.loc["mean_lb_per_acr", "PC1"] / loadings.loc["year", "PC1"] 
print (slope)
print (pca.components_[0][1] / pca.components_[0][0])


fig, axes = plt.subplots(1, 1, figsize=(2, 6), dpi=100)
df = bpszone_ANPP_no2012[bpszone_ANPP_no2012["fid"] == a_fid]
axes.plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);
axes.scatter(df.year, df[y_var], marker='o', facecolors='r', edgecolors='r', s=5, zorder=2);

################# plot PCA line
x1, y1 = df["year"].mean(), df["mean_lb_per_acr"].mean()
x = np.linspace(df["year"].min(), df["year"].max(), 100)
y = slope * (x - x1) + y1
axes.plot(x, y);


# del(pca, slope)

# %%

# %%

# %%
A = np.dot(X.values.T, X.values)
U, S, Vh = np.linalg.svd(X)

# %%
S

# %%
Vh

# %%
U.shape

# %%
loadings

# %%

# %%
loadings.loc["year", "PC1"] * X["year"] + loadings.loc["mean_lb_per_acr", "PC1"] * X["mean_lb_per_acr"]

# %%
loadings

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate some sample data
np.random.seed(0)
X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a PCA object
pca = PCA(n_components=1)
pca.fit(X_std) # Fit the PCA model

# Transform the data
X_pca = pca.transform(X_std)

# Plot the original data
plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, color="red")

# Plot the PCA lines
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')
plt.show()


# %%

# %%

# %%
del(X_std)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


X = bpszone_ANPP_no2012[["fid", "year", "mean_lb_per_acr"]].copy()
X = X[X["fid"] == a_fid].copy()
X = X[["year", "mean_lb_per_acr"]].copy()


# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a PCA object
pca = PCA(n_components=1)

# Fit the PCA model
# pca.fit(X_std)
pca.fit(X)

# Plot the original data
plt.scatter(X.loc[:, "year"], X.loc[:, "mean_lb_per_acr"], alpha=0.5)

# Plot the PCA lines
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([1984, v[0]], [1984, v[1]], '-k', lw=2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')
plt.show()


# %%
v[0]

# %%
v[1]

# %%
del(X_std)

# %%
# Generate some sample data
np.random.seed(0)
X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

# Create a PCA object
pca = PCA(n_components=1)
pca.fit(X) # Fit the PCA model


# Plot the original data
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)

# Plot the PCA lines
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')
plt.show()

# %%
X_pca

# %%
X

# %%
