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
import numpy as np
import pywt
import matplotlib.pyplot as plt

# %%
np.random.seed(42)  # For reproducibility
n = 1000  # Length of the signal
t = np.arange(n)
x = np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 1, n)  # A sine wave with noise

# Perform MODWT decomposition
max_level = 5  # Number of decomposition levels
wavelet = 'db1'  # Daubechies wavelet (can be changed)

# Perform the MODWT (with wavelet "db1")
coeffs = pywt.wavedec(x, wavelet, level=max_level)

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
coeffs = pywt.wavedec(x, wavelet, level=0)
print (np.array_equal(coeffs[0], x))

coeffs = pywt.wavedec(x, wavelet, level=2)
print (np.array_equal(coeffs[0], x))

# %%
n = 1000  # Length of the signal
t = np.arange(n)
x = np.sin(2 * np.pi * 0.01 * t) + np.random.normal(0, 1, n)  # A sine wave with noise

# %%
wavelet = 'db4'
sampling_period = 1.0
max_level = pywt.dwt_max_level(data_len=1024, filter_len=pywt.Wavelet(wavelet).dec_len)

for level in range(1, max_level + 1):
    freq = pywt.scale2frequency(wavelet, level) / sampling_period
    print(f"Level {level}: approx frequency = {freq:.5f} Hz")

# %%
pywt.Wavelet(wavelet).dec_len

# %%
import math
N = len(x)
max_level = int(math.floor(math.log2(N))) - k

# %%
wavelet = 'db4'  # or any wavelet
max_theoretical_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)

# Choose something slightly smaller
max_level = max_theoretical_level - 1  # or -2


# %%
def auto_max_level(x, wavelet_name='db4', offset=2):
    wavelet = pywt.Wavelet(wavelet_name)
    N = len(x)
    max_level = pywt.dwt_max_level(N, wavelet.dec_len)
    return max(1, max_level - offset)  # prevent 0 or negative levels
