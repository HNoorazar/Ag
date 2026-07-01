# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3.13.11 (Conda)
#     language: python
#     name: py313
# ---

# %%
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from statsmodels.graphics.tsaplots import plot_acf

# %%
dpi_ = 300

# %%
tick_legend_FontSize = 6
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * .8,
          "axes.labelsize":  tick_legend_FontSize * 1,
          "axes.titlesize":  tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .8,
          "ytick.labelsize": tick_legend_FontSize * .8,
          "axes.titlepad": 5,
          'legend.handlelength': 2,
          "axes.titleweight": 'bold',
          'axes.linewidth' : .05,
          'xtick.major.width': 0.1,
          'ytick.major.width': 0.1,
          'xtick.major.size': 2,
          'ytick.major.size': 2,
         }

plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
data_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/"

# %%
npp = pd.read_csv(data_base + "bps_annual_productivity_1982_2025_updated.csv")
npp.sort_values(by=['bps', 'year'], inplace=True)
npp.reset_index(drop=True, inplace=True)
npp.head(2)

# %%
print(f"There are {len(npp["bps"].unique()):,} locations.")

# %%
bps = 9
df = npp[npp.bps == bps]

n = len(df)
span_5yr = 5 / n
span_7yr = 7 / n
span_10yr = 10 / n

# ----------------------------
# Apply LOWESS
# ----------------------------
loess_5 = lowess(df["productivity"], df["year"], frac=span_5yr, return_sorted=False)
loess_7 = lowess(df["productivity"], df["year"], frac=span_7yr, return_sorted=False)
loess_10 = lowess(df["productivity"], df["year"], frac=span_10yr, return_sorted=False)

# moving_avg_7 = df["productivity"].rolling(window=7).mean()

# %%
fig, axes = plt.subplots(1, 1, figsize=(4, 2), sharey=False, sharex=False, dpi=dpi_)

axes.plot(df["year"], df["productivity"], label="Raw data", alpha=0.7, linewidth=1.2, color="dodgerblue")
axes.plot(df["year"], loess_5,  label="LOESS ~5-year (span=0.125)", linewidth=1, color="red")
axes.plot(df["year"], loess_7,  label="LOESS ~7-year (span=0.175)", linewidth=1, color="black")
axes.plot(df["year"], loess_10, label="LOESS ~10-year (span=0.25)", linewidth=1, color="green")
# axes.plot(df["year"], moving_avg_7, label="moving_avg_7", linewidth=1, linestyle="--", color="cyan")

plt.xlabel("Year")
plt.ylabel("productivity")
plt.title("LOESS smoothing of productivity")
plt.legend()
plt.show()

# %%
fig, ax = plt.subplots()
plot_acf(df["productivity"], ax=ax)
plt.show()

# %%
y = loess_10
X = df["year"]
X = sm.add_constant(X)

ols = sm.OLS(y, X).fit()
resid = ols.resid
rho = np.corrcoef(resid[1:], resid[:-1])[0, 1]
Sigma = np.fromfunction(lambda i, j: rho ** np.abs(i - j), (n, n))

# %%
model = sm.GLS(y, X, sigma=Sigma)
results = model.fit()
print(results.summary())

# %%
GLSAR_fit = sm.GLSAR(y, X, rho=1).iterative_fit()
GLSAR_fit.summary()

# %%
ols_HAC = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
ols_HAC.summary()

# %%

# %%
len(npp["bps"].unique())

# %%
