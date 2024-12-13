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
# https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf
#
# https://en.wikipedia.org/wiki/Autoregressive_moving-average_model
#
# https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html

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

from datetime import datetime

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
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_Spearman = pd.read_pickle(filename)
ANPP_MK_Spearman = ANPP_MK_Spearman["ANPP_MK_df"]
ANPP_MK_Spearman.head(2)

# %%
ANPP_MK_Spearman.columns

# %%
filename = bio_reOrganized + "bpszone_ANPP_no2012.sav"
bpszone_ANPP = pd.read_pickle(filename)
bpszone_ANPP = bpszone_ANPP["bpszone_ANPP"]
bpszone_ANPP.head(2)

# %%

# %% [markdown]
# ### Durbin-Watson test: This test is commonly used to detect autocorrelation in the residuals of a regression model
#
# **High positive autocorrelation**
#
#    - Near 2: Low autocorrelation 
#    - Near 0: Strong positive autocorrelation 
#    - Near 4: Strong negative autocorrelation

# %%
import statsmodels.api as sm

data = {'fid': bpszone_ANPP["fid"].unique(), 
        'dw_stat': [-666]*len(bpszone_ANPP["fid"].unique())}

dw_statistics_df = pd.DataFrame(data)


# Perform the Durbin-Watson test
for a_fid in bpszone_ANPP["fid"].unique():
    TS = bpszone_ANPP.loc[bpszone_ANPP["fid"] == a_fid, "mean_lb_per_acr"]
    dw_statistics_df.loc[dw_statistics_df["fid"] == a_fid, "dw_stat"] = sm.stats.stattools.durbin_watson(TS)

dw_statistics_df.head(2)

# %%
print (len(dw_statistics_df["fid"]))
print (dw_statistics_df[dw_statistics_df["dw_stat"] < 0.3].shape)
print (dw_statistics_df[dw_statistics_df["dw_stat"] < 0.4].shape)
print (dw_statistics_df[dw_statistics_df["dw_stat"] < 0.5].shape)

# %%
dpi_ = 300
map_dpi_ = 350
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds')
plt.rc("font", family="Helvetica")

# %%
tick_legend_FontSize = 8
params = {"legend.fontsize": tick_legend_FontSize*.8,
          "axes.labelsize": tick_legend_FontSize * .8,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 0.8,
          "ytick.labelsize": tick_legend_FontSize * 0.8,
          "axes.titlepad": 5,    'legend.handlelength': 2}

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
y_var = "mean_lb_per_acr"

# %%
fig, axes = plt.subplots(2, 1, figsize=(6, 2), sharex=True, sharey=False, dpi=200,
                        gridspec_kw={"hspace": 0.5, "wspace": 0.05})

axes[0].grid(axis="y", which="both"); axes[1].grid(axis="y", which="both");

######### Maximum correlation by DW test
auto_corr_idx = dw_statistics_df['dw_stat'].idxmin()
fid_corr = dw_statistics_df.loc[auto_corr_idx, "fid"]
df = bpszone_ANPP[bpszone_ANPP["fid"] == fid_corr]
axes[0].plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);
axes[0].scatter(df.year, df[y_var], marker='o', facecolors='r', edgecolors='r', s=5, zorder=2);

state = ANPP_MK_Spearman.loc[ANPP_MK_Spearman["fid"] == fid_corr, "state_majority_area"].item()
axes[0].set_title("Maximum correlation by DW test (" + state + ", FID: " + str(fid_corr) + ")");

######### Minimum correlation by DW test
auto_corr_idx = dw_statistics_df['dw_stat'].idxmax()
fid_corr = dw_statistics_df.loc[auto_corr_idx, "fid"]
df = bpszone_ANPP[bpszone_ANPP["fid"] == fid_corr]
axes[1].plot(df.year, df[y_var], linewidth=2, color="dodgerblue", zorder=1);
axes[1].scatter(df.year, df[y_var], marker='o', facecolors='r', edgecolors='r', s=5, zorder=2);
state = ANPP_MK_Spearman.loc[ANPP_MK_Spearman["fid"] == fid_corr, "state_majority_area"].item()
axes[1].set_title("Minimum correlation by DW test (" + state + ", FID: " + str(fid_corr) + ")");
axes[1].set_ylim([0, 210])

axes[0].axvline(x = 2012, color = 'c', label = '2012', linestyle="--")
axes[0].legend(loc = 'best');

# %% [markdown]
# ### Ljung-Box Q-test 
#
# Does not return anything significant for the location with max DW significance!

# %%
auto_corr_idx = dw_statistics_df['dw_stat'].idxmin()
fid_corr = dw_statistics_df.loc[auto_corr_idx, "fid"]

TS = bpszone_ANPP.loc[bpszone_ANPP["fid"] == fid_corr, "mean_lb_per_acr"]

sm.stats.diagnostic.acorr_ljungbox(x=TS.values, lags=None, boxpierce=False, 
                                   model_df=0, period=None, return_df=True, auto_lag=True)

# %%
auto_corr_idx = dw_statistics_df['dw_stat'].idxmax()
fid_corr = dw_statistics_df.loc[auto_corr_idx, "fid"]

TS = bpszone_ANPP.loc[bpszone_ANPP["fid"] == fid_corr, "mean_lb_per_acr"]

sm.stats.diagnostic.acorr_ljungbox(x=TS.values, lags=None, boxpierce=False, 
                                   model_df=0, period=None, return_df=True, auto_lag=True)

# %%
# %%time
data = {'fid': bpszone_ANPP["fid"].unique(), 
        'lb_stat': [-666]*len(bpszone_ANPP["fid"].unique()),
        'lb_pvalue': [-666]*len(bpszone_ANPP["fid"].unique())}

LB_df = pd.DataFrame(data)


for a_fid in bpszone_ANPP["fid"].unique():
    TS = bpszone_ANPP.loc[bpszone_ANPP["fid"] == a_fid, "mean_lb_per_acr"]
    curr_test = sm.stats.diagnostic.acorr_ljungbox(x=TS.values, lags=None, boxpierce=False,
                                                   model_df=0, period=None, return_df=True, auto_lag=True)
    
    LB_df.loc[LB_df["fid"] == a_fid, ["lb_stat", "lb_pvalue"]] = \
                            curr_test.loc[curr_test["lb_pvalue"].idxmin()].values

# %%
# %%time
data = {'fid': bpszone_ANPP["fid"].unique(), 
        'lb_stat': [-666]*len(bpszone_ANPP["fid"].unique()),
        'lb_pvalue': [-666]*len(bpszone_ANPP["fid"].unique())}

LB_df_insignificant = pd.DataFrame(data)

for a_fid in bpszone_ANPP["fid"].unique():
    TS = bpszone_ANPP.loc[bpszone_ANPP["fid"] == a_fid, "mean_lb_per_acr"]
    curr_test = sm.stats.diagnostic.acorr_ljungbox(x=TS.values, lags=None, boxpierce=False,
                                                   model_df=0, period=None, return_df=True, auto_lag=True)
    
    LB_df_insignificant.loc[LB_df_insignificant["fid"] == a_fid, ["lb_stat", "lb_pvalue"]] = \
                            curr_test.loc[curr_test["lb_pvalue"].idxmax()].values

# %%
LB_df_insignificant.head(2)

# %%
print (LB_df.shape)
print (LB_df[LB_df["lb_pvalue"]<0.05].shape)
print (LB_df_insignificant[LB_df_insignificant["lb_pvalue"]<0.05].shape)

# %%
dta = sm.datasets.sunspots.load_pandas().data
dta.head(5)

# %%
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
dta.head(5)

# %%
del dta["YEAR"]
dta.head(5)

# %%
sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
plt.show()

# %%
df = bpszone_ANPP[bpszone_ANPP.fid == 1]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5)
plt.show()

# %%
an_fid = dw_statistics_df.loc[dw_statistics_df['dw_stat'].idxmax(), "fid"]
df = bpszone_ANPP[bpszone_ANPP.fid == an_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5)
plt.show()

# %%
an_fid = dw_statistics_df.loc[dw_statistics_df['dw_stat'].idxmin(), "fid"]
df = bpszone_ANPP[bpszone_ANPP.fid == an_fid]
sm.graphics.tsa.plot_acf(df.mean_lb_per_acr.squeeze(), lags=5)
plt.show()

# %%

# %%
