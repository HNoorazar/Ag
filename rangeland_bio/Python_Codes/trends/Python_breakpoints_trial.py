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
# # breakpoints in R.
#
# Lets do the breakpoints in R. Here we can analyze and plot things.

# %%
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os, os.path, pickle, sys
import pymannkendall as mk

import statistics
import statsmodels.formula.api as smf

import statsmodels.stats.api as sms
import statsmodels.api as sm

from scipy import stats
import scipy.stats as scipy_stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_core as rpc


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
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

rangeland_bio_base = research_db + "/RangeLand_bio/"
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
ANPP.head(2)

# %%
ANPP.to_csv("/Users/hn/Desktop/" + 'bpszone_ANPP_no2012.csv', index=False)

# %%
# from statsmodels.stats.diagnostic import breaks_cusumolsresid

# # Fit model
# X = sm.add_constant(x)
# model = sm.OLS(y, X).fit()

# # CUSUM test for structural breaks
# stat, pval, crit = breaks_cusumolsresid(model.resid, ddof=1)

# %%
import ruptures as rpt
# Example using Binary Segmentation
algo = rpt.Binseg(model="l2").fit(data)
result = algo.predict(n_bkps=3) # Detect 3 breakpoints
result

# %%

# %%

# %%

# %%

# %%
import importlib
importlib.reload(rc)
importlib.reload(rpc)

# %%
# Example usage
x = range(100)
np.random.seed(42)
y = [i + np.random.normal(0, 10) if i < 50 else 2 * i + np.random.normal(0, 10) for i in range(100)]

data = pd.DataFrame({'x': x,'y': y})
formula = "y ~ x"
split_point = 50

chow_stat, p_value = rc.Gemini_chow_test(data, formula, split_point)

print(f"Chow Statistic: {chow_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a structural break.")
else:
    print("Fail to reject the null hypothesis: There is no significant structural break.")

print  ("------------------ chatGPT ------------------")
chow_stat, p_value = rc.ChatGPT_chow_test(y, np.array(x), split_index=50)
print (chow_stat, p_value)

# %%

# %%

# %%
# chow_test(y, np.array(x), 50)
# from chow_test import chow_test

# %%

# %%
from statsmodels.datasets import nile

# Load the Nile dataset
nile_data = nile.load_pandas().data
y_nile = nile_data['volume'].values
x_nile = nile_data['year'].values

# %%

# %%
# Example with a simple time series
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.random.randn(100) + np.linspace(0, 5, 100)  # Add a linear trend
time_series = pd.DataFrame({'value': values}, index=dates)

# Add a time trend variable
time_series['time'] = range(len(time_series))

# Fit a linear regression model
X = time_series['time']
y = time_series['value']
X = sm.add_constant(X)  # Add a constant term
model = sm.OLS(y, X).fit()

# Get the F-statistic and p-value
f_statistic = model.fvalue
p_value = model.f_pvalue

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")


# %%

# %%
def compute_fstats(y, min_size=10):
    """
    Compute F-statistics for structural breaks in a univariate time series using a constant-only model (intercept).
    
    Parameters:
        y (array-like): 1D time series data.
        min_size (int): Minimum number of observations before and after a breakpoint.
        
    Returns:
        pd.Series: F-statistics indexed by candidate breakpoints.
    """
    y = np.asarray(y)
    T = len(y)
    
    if T < 2 * min_size:
        raise ValueError("Time series too short for given min_size.")
    
    f_stats = []

    for t in range(min_size, T - min_size):
        # Full model (intercept only)
        X_full = np.ones((T, 1))
        model_full = sm.OLS(y, X_full).fit()
        rss_full = np.sum(model_full.resid**2)
        
        # Two-segment models
        X1 = np.ones((t, 1))
        X2 = np.ones((T - t, 1))
        model1 = sm.OLS(y[:t], X1).fit()
        model2 = sm.OLS(y[t:], X2).fit()
        rss1 = np.sum(model1.resid**2)
        rss2 = np.sum(model2.resid**2)
        
        # F-statistic
        num = (rss_full - (rss1 + rss2)) / 1
        denom = (rss1 + rss2) / (T - 2)
        f_stat = num / denom
        f_stats.append(f_stat)
    
    return pd.Series(f_stats, index=range(min_size, T - min_size))


# %%
# Load Nile river dataset
T = len(y_nile)

# %%
# Minimum number of observations before and after break
min_size = 5
f_stat_series = compute_fstats(y_nile, min_size=min_size)
# Plot the F-statistics time series
plt.figure(figsize=(10, 5))
f_stat_series.plot(title="F-statistics for Structural Break Test")
plt.xlabel("Break Point")
plt.ylabel("F-statistic")
plt.grid(True)
plt.show()

# %%
min_size = 10 # Minimum number of observations before and after break

f_stat_series = compute_fstats(y_nile, min_size=min_size)
# Plot the F-statistics time series
plt.figure(figsize=(10, 5))
f_stat_series.plot(title="F-statistics for Structural Break Test")
plt.xlabel("Break Point")
plt.ylabel("F-statistic")
plt.grid(True)
plt.show()

# %%
print (f_stat_series.idxmax())
print (f_stat_series[f_stat_series.idxmax()])

# %%
x_nile[f_stat_series.idxmax()-1]


# %%

# %%
def find_breakpoints(y, model="l2", n_bkps=1):
    """
    Find structural breakpoints in a univariate time series using the 'ruptures' package.
    
    Parameters:
        y (array-like): The original time series data.
        model (str): Cost model to use ('l2' for mean shift).
        n_bkps (int): Number of breakpoints to detect.
        
    Returns:
        list: List of breakpoint indices (end of each segment).
    """
    algo = rpt.Binseg(model=model).fit(y)
    breakpoints = algo.predict(n_bkps=n_bkps)
    return breakpoints

bkps = find_breakpoints(y, model="l2", n_bkps=1)
print("Detected breakpoints:", bkps)

# Plot
rpt.display(y_nile, bkps, figsize=(10, 5));

# %%

# %%
model = rpt.Window(width=14, model="l2").fit(data)
result = model.predict(n_bkps=1)  # Number of breakpoints to detect
print (f"{result=}")
# Plot the result
rpt.display(data, result, figsize=(10, 6));

# %%
x_nile[result[0]-1]

# %%
from scipy.signal import find_peaks

def find_breakpoints_from_fstats(f_stats, threshold=None):
    """
    Find breakpoints from F-statistics series by locating peaks.
    
    Parameters:
        f_stats (pd.Series): Time series of F-statistics.
        threshold (float): Minimum F-stat to count as breakpoint (optional).
        
    Returns:
        list: Indices of detected breakpoints.
    """
    peaks, _ = find_peaks(f_stats, height=threshold)
    return f_stats.index[peaks].tolist()


# %%
find_breakpoints_from_fstats(f_stats=f_stat_series, threshold=f_stat_series.max()-1)

# %%
print("Breakpoint year:", x_nile[bkps[0]-1]) 

# %%
bkps = find_breakpoints(y_nile, model="l2", n_bkps=1)
print("Breakpoint index (Python):", bkps)  # typically [30]
print("Breakpoint year:", x_nile[bkps[0]-1])

# %%
from statsmodels.stats.diagnostic import breaks_cusumolsresid

# Fit model
X = sm.add_constant(x_nile)
X = sm.add_constant(np.arange(len(y_nile)))
model = sm.OLS(y_nile, X).fit()

# CUSUM test for structural breaks
stat, pval, crit = breaks_cusumolsresid(model.resid, ddof=1)
print(f"stat={stat:.2f}, pval={pval:.2f}, crit={crit}")

# %%
model.summary()

# %%

# %%

# %%

# %%
