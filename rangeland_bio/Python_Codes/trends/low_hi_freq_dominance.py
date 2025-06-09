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
# # !pip3 install PyWavelets
import numpy as np
import pywt
import matplotlib.pyplot as plt

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

breakpoint_plot_base = bio_plots + "breakpoints/"
os.makedirs(breakpoint_plot_base, exist_ok=True)

breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/"
os.makedirs(breakpoint_TS_dir, exist_ok=True)


G_breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/greening/"
B_breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/browning/"
noTrend_breakpoint_TS_dir = breakpoint_plot_base + "breakpoints_TS/notrend/"

os.makedirs(G_breakpoint_TS_dir, exist_ok=True)
os.makedirs(B_breakpoint_TS_dir, exist_ok=True)
os.makedirs(noTrend_breakpoint_TS_dir, exist_ok=True)

# %%

# %%
np.random.seed(42)  # For reproducibility
n = 1000  # Length of the signal
t = np.arange(n)
x = np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 1, n)  # A sine wave with noise

# Perform MODWT decomposition
max_level = 5  # Number of decomposition levels
# wavelet_str = ''  # Daubechies wavelet (can be changed)

# Perform the MODWT (with wavelet "db1")
coeffs = pywt.wavedec(x, 'db1', level=max_level)

# %%
# Plot original + all wavelet components
n_plots = len(coeffs) + 1  # original + levels
fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=False)

# Plot original signal
axes[0].plot(t, x)
axes[0].set_title("Original Signal")

# Plot wavelet coefficients (details + approximation)
for i, coeff in enumerate(coeffs):
    label = f"Level {i} - Approximation" if i == 0 else f"Level {i} - Detail"
    axes[i+1].plot(coeff)
    axes[i+1].set_title(label)

plt.tight_layout()
plt.show()


# %%
variances = []

# Calculate the variance for the approximation and each detail component
for coeff in coeffs:
    variances.append(np.var(coeff))

print("Variances at each scale:", variances)

# %%
for i, coeff in enumerate(coeffs):
    print (len(coeff))

# %%
coeffs = pywt.wavedec(x, 'db1', level=0)
print (np.array_equal(coeffs[0], x))

coeffs = pywt.wavedec(x, 'db1', level=2)
print (np.array_equal(coeffs[0], x))

# %%
n = 1000  # Length of the signal
t = np.arange(n)
x = np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 1, n)  # A sine wave with noise

# %%
# wavelet = 'db4'
sampling_period = 1.0
max_level = pywt.dwt_max_level(data_len=1024, filter_len=pywt.Wavelet('db4').dec_len)

for level in range(1, max_level + 1):
    freq = pywt.scale2frequency('db4', level) / sampling_period
    print(f"Level {level}: approx frequency = {freq:.5f} Hz")

# %% [markdown]
# # Determine the right level of decomposition

# %%
pywt.Wavelet('db4').dec_len

# %%
import math
N = len(x)
max_level = int(math.floor(math.log2(N))) - k

# %%
# wavelet = 'db4'  # or any wavelet
max_theoretical_level = pywt.dwt_max_level(len(x), pywt.Wavelet('db4').dec_len)

# Choose something slightly smaller
max_level = max_theoretical_level - 1  # or -2


# %%
def auto_max_level(x, wavelet_name='db4', offset=2):
    wavelet = pywt.Wavelet(wavelet_name)
    N = len(x)
    max_level = pywt.dwt_max_level(N, wavelet.dec_len)
    return max(1, max_level - offset)  # prevent 0 or negative levels


# %%

# %% [markdown]
# # Check shift in variability of low freq and hi. freq

# %%
np.random.seed(42)
n = 512
high_freq = np.random.normal(scale=1.0, size=n//2)
low_freq = np.cumsum(np.random.normal(scale=0.3, size=n//2))
signal = np.concatenate([high_freq, low_freq])

# %%
level = pywt.dwt_max_level(len(signal), pywt.Wavelet('db4').dec_len)
coeffs = pywt.wavedec(signal, 'db4', mode='periodization', level=level)

# %%
variances = [np.var(c) for c in coeffs[:-1]]  # exclude approximation coeffs

# %%
# Plot variance per level
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(variances)+1), variances)
plt.xlabel('MODWT Level (Detail coefficients)')
plt.ylabel('Variance')
plt.title('Variance of Wavelet Detail Coefficients')
plt.xticks(range(1, len(variances)+1))
plt.gca().invert_xaxis()  # Level 1 is highest frequency, so put it on the right
plt.grid(True)
plt.show()

# %%
[len(c) for c in coeffs[:-1]]

# %%

# %%
variances

# %%
window_size = 128
step_size = 16
n_windows = (len(signal) - window_size) // step_size + 1

level = 4
var_matrix = []

for i in range(n_windows):
    window = signal[i*step_size : i*step_size + window_size]
    
    wavelet = pywt.Wavelet('db4')
    max_level = pywt.dwt_max_level(window_size, wavelet.dec_len)
    level = min(4, max_level) 
    
    coeffs = pywt.wavedec(window, 'db4', mode='periodization', level=level)
    variances = [np.var(c) for c in coeffs[:-1]]
    var_matrix.append(variances)

var_matrix = np.array(var_matrix)

# Plot heatmap of variance over time and scale
plt.figure(figsize=(12, 6))
plt.imshow(var_matrix.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Variance')
plt.xlabel('Window index (time)')
plt.ylabel('Wavelet Level (1 = High Freq)')
plt.title('Shift in Variance Across MODWT Scales Over Time')
plt.yticks(range(level), [f'Level {i+1}' for i in range(level)])
plt.show()


# %%
## hard-coded level
window_size = 128
step_size = 16
n_windows = (len(signal) - window_size) // step_size + 1

level = 4
var_matrix = []

for i in range(n_windows):
    window = signal[i*step_size : i*step_size + window_size]
    
    coeffs = pywt.wavedec(window, 'db4', mode='periodization', level=level)
    variances = [np.var(c) for c in coeffs[:-1]]
    var_matrix.append(variances)

var_matrix = np.array(var_matrix)

# Plot heatmap of variance over time and scale
plt.figure(figsize=(12, 6))
plt.imshow(var_matrix.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Variance')
plt.xlabel('Window index (time)')
plt.ylabel('Wavelet Level (1 = High Freq)')
plt.title('Shift in Variance Across MODWT Scales Over Time')
plt.yticks(range(level), [f'Level {i+1}' for i in range(level)])
plt.show()

# %%
var_matrix

# %%
variances

# %%

# %%

# %%

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

# %%
a_fid_anpp = ANPP[ANPP["fid"]==1]
signal = a_fid_anpp["mean_lb_per_acr"].values

# %%
level = pywt.dwt_max_level(len(signal), pywt.Wavelet('db4').dec_len)
coeffs = pywt.wavedec(signal, 'db4', mode='periodization', level=level)

# %%

# %%
from modwtpy import modwt, modwtmra

# Simulate time series
n = 512
signal = np.concatenate([np.random.normal(0, 1, n//2), np.cumsum(np.random.normal(0, 0.3, n//2))])

# Perform MODWT
w, V = modwt(signal, 'la8', 5)  # w is JxN matrix

# Multiresolution reconstruction
mra = modwtmra(w, 'la8')

# Compute variance at each level
variances = np.var(mra, axis=1)

plt.bar(range(1, len(variances)+1), variances)
plt.gca().invert_xaxis()
plt.xlabel('Level (High → Low Frequency)')
plt.ylabel('Variance')
plt.title('MODWT MRA Variance by Level')
plt.show()


# %%
# !pip3 install modwtpy

# %%

# %%
