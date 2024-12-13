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

# %%
import os, os.path, pickle, sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymannkendall as mk

from scipy import stats


# %%

def exponential_moving_average(prices, period, weighting_factor=0.2):
    ema = np.zeros(len(prices))
    sma = np.mean(prices[:period])
    ema[period - 1] = sma
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
    return ema


def exponential_moving_average_v2(TS_list, window, weighting_factor=0.2):
    """
    Let us repeat the first entry 5 times, so, the beginning of time-series
    comes out not too bad
    """
    TS_list = list(TS_list)
    TS_list = TS_list[:window] + TS_list
    ema = np.zeros(len(TS_list))
    sma = np.mean(TS_list[:window])
    ema[window - 1] = sma
    for i in range(window, len(TS_list)):
        ema[i] = (TS_list[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
    return ema[window:]


# %%
y_var = "mean_lb_per_acr"

# %%
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

plot_dir = rangeland_bio_base + "plots/for_Matt/"
os.makedirs(plot_dir, exist_ok=True)

# %%
dpi_ = 300

txt_font_dict = {'fontsize':10, 'fontweight':'bold'}

tick_legend_FontSize = 12
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize * 1,
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05
         }

plt.rcParams.update(params)

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print (len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
len(ANPP_MK_df[ANPP_MK_df["trend"] == "increasing"])

# %%
df = ANPP_MK_df[(ANPP_MK_df["trend"] == "increasing") & (ANPP_MK_df["sens_slope"] <= 20)]
print ("s < 20:    {:,}".format(len(df)))

df = ANPP_MK_df[(ANPP_MK_df["trend"] == "increasing") & 
               (ANPP_MK_df["sens_slope"] > 20) & 
               ((ANPP_MK_df["sens_slope"] <= 30))]
print ("s in 20-30: {:,}".format(len(df)))

df = ANPP_MK_df[(ANPP_MK_df["trend"] == "increasing") & (ANPP_MK_df["sens_slope"] > 30)]
print ("s > 30:     {:,}".format(len(df)))

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
print (2012 in sorted(ANPP.year.unique()))
ANPP.head(2)

# %%
a_fid = 25509 # 25509 # 14926

df = ANPP[ANPP["fid"] == a_fid].copy()
df['moving_average'] = df['mean_lb_per_acr'].rolling(window=10).mean()
TS = df[y_var].values

x_vec = list(df["year"].values)
period = 2 # 2
weighting_factor = 0.5 # 0.5
ema = exponential_moving_average_v2(TS, period, weighting_factor)
########################
x_text, x_text_SG = 1983, 1997
y_txt = TS.max() * .99
########################
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, dpi=dpi_)

######## NPP
axes.plot(x_vec, TS, linewidth=3, color="dodgerblue", label="NPP");
axes.scatter(x_vec, TS, zorder=3, color="dodgerblue");

####### exponential average
axes.plot(x_vec, ema, linewidth=3, color="black", label="smooth");
axes.scatter(x_vec, ema, zorder=3, color="black");

##############################
trend_, _, p_, _, Tau_, _, _, slope_, _ = mk.original_test(ema)
trend_yue, _, p_yue, _, Tau_yue, _, var_s_yue, slope_yue, _ = mk.yue_wang_modification_test(ema)

text_ = "original:\ntrend: {} ({:.4f})\nSen's slope: {:.2f}, \nTau: {:.2f}\n".format(trend_, p_, slope_, Tau_)
axes.text(x_text, y_txt, text_, color="black", fontsize=tick_legend_FontSize*1.2, va="top");

text_ = "Yue:\ntrend: {} ({:.4f})\n".format(trend_yue, p_yue)
axes.text(x_text_SG, y_txt, text_, color="black", fontsize=tick_legend_FontSize*1.2, va="top");

axes.legend(loc="lower right");

file_name = plot_dir + "FID_" + str(a_fid) + "_strange_MK_test_EMA.pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

del(ema, trend_, p_, slope_, Tau_, trend_yue, p_yue, text_)

# %%

# %%
a_fid = 25509 # 25509 # 14926

df = ANPP[ANPP["fid"] == a_fid].copy()
df['moving_average'] = df['mean_lb_per_acr'].rolling(window=10).mean()
TS = df[y_var].values

x_vec = list(df["year"].values)
########################
x_text, x_text_SG = 1983, 1997
y_txt = TS.max() * .99
########################
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, dpi=dpi_)

######## NPP
axes.plot(x_vec, TS, linewidth=3, color="dodgerblue", label="NPP");
axes.scatter(x_vec, TS, zorder=3, color="dodgerblue");

####### moving average
axes.plot(x_vec, df["moving_average"], linewidth=3, color="black", label="rolling");
axes.scatter(x_vec, df["moving_average"], zorder=3, color="black");

##############################
trend_, _, p_, _, Tau_, _, _, slope_, _ = mk.original_test(df["moving_average"])
trend_yue, _, p_yue, _, Tau_yue, _, var_s_yue, slope_yue, _ = mk.yue_wang_modification_test(df["moving_average"])

text_ = "original:\ntrend: {} ({:.4f})\nSen's slope: {:.2f}, \nTau: {:.2f}\n".format(trend_, p_, slope_, Tau_)
axes.text(x_text, y_txt, text_, color="black", fontsize=tick_legend_FontSize*1.2, va="top");

text_ = "Yue:\ntrend: {} ({:.4f})\n".format(trend_yue, p_yue)
axes.text(x_text_SG, y_txt, text_, color="black", fontsize=tick_legend_FontSize*1.2, va="top");

axes.legend(loc="lower right");

file_name = plot_dir + "FID_" + str(a_fid) + "_strange_MK_test_MA.pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %%
slope, intercept, _, _ = stats.theilslopes(df[y_var])
trend_, _, p_, _, Tau_, _, _, slope_, _ = mk.original_test(df[y_var])
print (slope.round(4), slope_.round(4))
print (intercept.round(4), intercept.round(4))

# %%
slope, intercept, _, _ = stats.theilslopes(df[y_var], method="joint")
print (slope.round(4), slope_.round(4))

slope, intercept, _, _ = stats.theilslopes(df[y_var], method="separate")
print (slope.round(4), slope_.round(4))

# %%
mk.original_test(df[y_var])

# %%
mk.yue_wang_modification_test(df[y_var], lag=5)

# %% [markdown]
# # Change lag in Yue

# %%
a_fid = 14926 # 25509 # 14926
lag_ = 1
df = ANPP[ANPP["fid"] == a_fid].copy()
df['moving_average'] = df['mean_lb_per_acr'].rolling(window=10).mean()
TS = df[y_var].values

x_vec = list(df["year"].values)
period = 2 # 2
weighting_factor = 0.5 # 0.5
ema = exponential_moving_average_v2(TS, period, weighting_factor)
########################
x_text, x_text_SG = 1983, 1997
y_txt = TS.max() * .99
########################
fig, axes = plt.subplots(1, 1, figsize=(10, 3), sharex=True, dpi=dpi_)

######## NPP
axes.plot(x_vec, TS, linewidth=3, color="dodgerblue", label="NPP");
axes.scatter(x_vec, TS, zorder=3, color="dodgerblue");

####### exponential average
axes.plot(x_vec, ema, linewidth=3, color="black", label="smooth");
axes.scatter(x_vec, ema, zorder=3, color="black");
##############################
trend_, _, p_, _, Tau_, _, _, slope_, intercept_ = mk.original_test(ema)
trend_yue, _, p_yue, _, Tau_yue, _, var_s_yue, slope_yue, _ = mk.yue_wang_modification_test(ema, lag=lag_)

text_ = "original:\ntrend: {} ({:.4f})\nSen's slope: {:.2f}, \nTau: {:.2f}\n".format(trend_, p_, slope_, Tau_)
axes.text(x_text, y_txt, text_, color="black", fontsize=tick_legend_FontSize*1.2, va="top");

text_ = "Yue:\ntrend: {} ({:.4f})\n".format(trend_yue, p_yue)
axes.text(x_text_SG, y_txt, text_, color="black", fontsize=tick_legend_FontSize*1.2, va="top");

######## Trend line
trend_line = np.arange(len(ema)) * slope_ + intercept_
axes.plot(x_vec, trend_line, linewidth=3, color="red");

########
axes.legend(loc="lower right");

file_name = plot_dir + "FID_" + str(a_fid) + "_strange_MK_test_EMA.pdf"
# plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

del(ema, trend_, p_, slope_, Tau_, trend_yue, p_yue, text_)

# %%
import statsmodels.api as sm

fig, ax = plt.subplots(figsize=(8, 4))
sm.graphics.tsa.plot_acf(df[y_var], lags=20, ax=ax);

# %%
