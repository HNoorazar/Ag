# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %%
import numpy as np
import pandas as pd

import time, random
from datetime import date
from random import seed, random

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow

import sys, os, os.path

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc
import NASA_plot_core as ncp

# %%
data_dir = "/Users/hn/Documents/01_research_data/NASA/merged_trend_ML_preds/"
meta_dir = (
    "/Users/hn/Documents/01_research_data/NASA/shapefiles/10_intersect_East_Irr_2008_2018_2cols/"
)

plot_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/trend/"
os.makedirs(plot_dir, exist_ok=True)

# %%
SF_data = pd.read_csv(meta_dir + "10_intersect_East_Irr_2008_2018_2cols_data_part.csv")
SF_data_large = SF_data[SF_data.acreage > 10]

# %%
EVI_SG_preds = pd.read_csv(data_dir + "EVI_SG_preds_intersect.csv")
NDVI_SG_preds = pd.read_csv(data_dir + "NDVI_SG_preds_intersect.csv")

EVI_regular_preds = pd.read_csv(data_dir + "EVI_regular_preds_intersect.csv")
NDVI_regular_preds = pd.read_csv(data_dir + "NDVI_regular_preds_intersect.csv")

# %%

# %%
EVI_SG_preds_pre2008 = pd.read_csv(data_dir + "EVI_SG_preds_intersect_pre2008.csv")
NDVI_SG_preds_pre2008 = pd.read_csv(data_dir + "NDVI_SG_preds_intersect_pre2008.csv")

EVI_regular_preds_pre2008 = pd.read_csv(data_dir + "EVI_regular_preds_intersect_pre2008.csv")
NDVI_regular_preds_pre2008 = pd.read_csv(data_dir + "NDVI_regular_preds_intersect_pre2008.csv")

# %%
EVI_SG_preds = pd.concat([EVI_SG_preds, EVI_SG_preds_pre2008])
EVI_regular_preds = pd.concat([EVI_regular_preds, EVI_regular_preds_pre2008])

NDVI_SG_preds = pd.concat([NDVI_SG_preds, NDVI_SG_preds_pre2008])
NDVI_regular_preds = pd.concat([NDVI_regular_preds, NDVI_regular_preds_pre2008])

# %%

# %%
prob_EVI = 0.4
prob_NDVI = 0.9

colName = "EVI_SG_DL_p4"
EVI_SG_preds[colName] = -1
EVI_SG_preds.loc[EVI_SG_preds.EVI_SG_DL_p_single < prob_EVI, colName] = 2
EVI_SG_preds.loc[EVI_SG_preds.EVI_SG_DL_p_single >= prob_EVI, colName] = 1
EVI_SG_preds.drop(["EVI_SG_DL_p_single"], axis=1, inplace=True)


colName = "EVI_regular_DL_p4"
EVI_regular_preds[colName] = -1
EVI_regular_preds.loc[EVI_regular_preds.EVI_regular_DL_p_single < prob_EVI, colName] = 2
EVI_regular_preds.loc[EVI_regular_preds.EVI_regular_DL_p_single >= prob_EVI, colName] = 1
EVI_regular_preds.drop(["EVI_regular_DL_p_single"], axis=1, inplace=True)
EVI_regular_preds.head(2)

# %%
NDVI_SG_preds.head(2)

# %%
sorted(NDVI_SG_preds.year.unique())

# %%
NDVI_regular_preds.head(2)

# %%
colName = "NDVI_SG_DL_p9"
NDVI_SG_preds[colName] = -1
NDVI_SG_preds.loc[NDVI_SG_preds.NDVI_SG_DL_p_single < prob_NDVI, colName] = 2
NDVI_SG_preds.loc[NDVI_SG_preds.NDVI_SG_DL_p_single >= prob_NDVI, colName] = 1
NDVI_SG_preds.drop(["NDVI_SG_DL_p_single"], axis=1, inplace=True)

colName = "NDVI_regular_DL_p9"
NDVI_regular_preds[colName] = -1
NDVI_regular_preds.loc[NDVI_regular_preds.NDVI_regular_DL_p_single < prob_NDVI, colName] = 2
NDVI_regular_preds.loc[NDVI_regular_preds.NDVI_regular_DL_p_single >= prob_NDVI, colName] = 1
NDVI_regular_preds.drop(["NDVI_regular_DL_p_single"], axis=1, inplace=True)

# %%

# %%
NDVI_SG_preds = pd.merge(NDVI_SG_preds, SF_data, on=["ID"], how="left")
EVI_SG_preds = pd.merge(EVI_SG_preds, SF_data, on=["ID"], how="left")

NDVI_regular_preds = pd.merge(NDVI_regular_preds, SF_data, on=["ID"], how="left")
EVI_regular_preds = pd.merge(EVI_regular_preds, SF_data, on=["ID"], how="left")

# %%
NDVI_SG_preds_large = NDVI_SG_preds[NDVI_SG_preds.acreage > 10].copy()
NDVI_regular_preds_large = NDVI_regular_preds[NDVI_regular_preds.acreage > 10].copy()

EVI_SG_preds_large = EVI_SG_preds[EVI_SG_preds.acreage > 10].copy()
EVI_regular_preds_large = EVI_regular_preds[EVI_regular_preds.acreage > 10].copy()

EVI_regular_preds_large.reset_index(drop=True, inplace=True)
NDVI_regular_preds_large.reset_index(drop=True, inplace=True)
EVI_SG_preds_large.reset_index(drop=True, inplace=True)
NDVI_SG_preds_large.reset_index(drop=True, inplace=True)


# %%
# NDVI_SG_preds_large.groupby(['year', 'RF_NDVI_SG_preds'])['acreage'].sum()


# %%
def group_sum_area(df, group_cols):
    """groups by 2 columns given by group_cols.
    group_cols[0] is something like
                            SVM_NDVI_SG_preds
                            SVM_NDVI_regular_preds
                            SVM_EVI_SG_preds
                            SVM_EVI_regular_preds
    """
    df = df[group_cols + ["acreage"]]
    col = (
        df.groupby([group_cols[0], group_cols[1]])["acreage"]
        .sum()
        .reset_index(name=group_cols[0] + "_acr_sum")
    )
    col.rename(
        columns={group_cols[0]: "label", group_cols[0] + "_acr_sum": group_cols[0]}, inplace=True
    )
    return col


# %% [markdown]
# # Acre-Wise
#
# ### NDVI SG

# %%
size = 10
tick_legend_FontSize = 10
params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.2,
    "xtick.labelsize": tick_legend_FontSize,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
# plt.rcParams['axes.grid'] = False
# plt.rcParams(which='major', axis='y', linestyle='--')
plt.rcParams.update(params)

color_dict = {"SVM": "#DDCC77",
              "kNN": "#E69F00",
              "RF": "#332288", # "#6699CC",
              "DL":'#0072B2'
             }

# %%
fig_size = (12, 2.5)

# %%
NDVI_SG_summary_L = pd.DataFrame(columns=["label"])

col1 = group_sum_area(NDVI_SG_preds_large, [NDVI_SG_preds_large.columns[2], "year"])
col2 = group_sum_area(NDVI_SG_preds_large, [NDVI_SG_preds_large.columns[3], "year"])
col3 = group_sum_area(NDVI_SG_preds_large, [NDVI_SG_preds_large.columns[4], "year"])
col4 = group_sum_area(NDVI_SG_preds_large, [NDVI_SG_preds_large.columns[5], "year"])

NDVI_SG_summary_L = pd.concat([NDVI_SG_summary_L, col1])
NDVI_SG_summary_L = pd.merge(NDVI_SG_summary_L, col2, on=(["label", "year"]), how="left")
NDVI_SG_summary_L = pd.merge(NDVI_SG_summary_L, col3, on=(["label", "year"]), how="left")
NDVI_SG_summary_L = pd.merge(NDVI_SG_summary_L, col4, on=(["label", "year"]), how="left")

NDVI_SG_summary_L.year = NDVI_SG_summary_L.year.astype(int)
NDVI_SG_summary_L.head(2)

NDVI_SG_summary_L.rename(
    columns={
        "RF_NDVI_SG_preds": "RF",
        "SVM_NDVI_SG_preds": "SVM",
        "KNN_NDVI_SG_preds": "KNN",
        "NDVI_SG_DL_p9": "DL",
    },
    inplace=True,
)

NDVI_SG_summary_L_double = NDVI_SG_summary_L[NDVI_SG_summary_L.label == 2].copy()
NDVI_SG_summary_L.head(2)

# %%
y_label_ = "area (acreage)"
title_base = "" # "double-cropped area "

# %%
# NDVI_SG_summary_L_double.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )

# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(NDVI, 5-step smoothed)")
# plt.grid(axis="y")
# #file_name = plot_dir + "NDVI_SG_double_area_trend.pdf"

# %%

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False, # sharey='col', # sharex=True, sharey=True,
                       gridspec_kw={'hspace': 0.35, 'wspace': .05});
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
NDVI_SG_summary_L_double_2008 = NDVI_SG_summary_L_double[NDVI_SG_summary_L_double.year>=2008].copy()

ncp.trend_prePost2008(df = NDVI_SG_summary_L_double_2008, ax = axs,
                      ylabel_ = y_label_,
                      title_=title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)
# yLims = [0, 460000],


file_name = plot_dir + "NDVI_SG_double_area_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %%

# %% [markdown]
# ### EVI SG

# %%
EVI_SG_summary_L = pd.DataFrame(columns=["label"])

col1 = group_sum_area(EVI_SG_preds_large, [EVI_SG_preds_large.columns[2], "year"])
col2 = group_sum_area(EVI_SG_preds_large, [EVI_SG_preds_large.columns[3], "year"])
col3 = group_sum_area(EVI_SG_preds_large, [EVI_SG_preds_large.columns[4], "year"])
col4 = group_sum_area(EVI_SG_preds_large, [EVI_SG_preds_large.columns[5], "year"])

EVI_SG_summary_L = pd.concat([EVI_SG_summary_L, col1])
EVI_SG_summary_L = pd.merge(EVI_SG_summary_L, col2, on=(["label", "year"]), how="left")
EVI_SG_summary_L = pd.merge(EVI_SG_summary_L, col3, on=(["label", "year"]), how="left")
EVI_SG_summary_L = pd.merge(EVI_SG_summary_L, col4, on=(["label", "year"]), how="left")

EVI_SG_summary_L.year = EVI_SG_summary_L.year.astype(int)
EVI_SG_summary_L.head(2)

EVI_SG_summary_L.rename(
    columns={
        "RF_EVI_SG_preds": "RF",
        "SVM_EVI_SG_preds": "SVM",
        "KNN_EVI_SG_preds": "KNN",
        "EVI_SG_DL_p4": "DL",
    },
    inplace=True,
)

EVI_SG_summary_L_double = EVI_SG_summary_L[EVI_SG_summary_L.label == 2].copy()
EVI_SG_summary_L.head(2)

# %%
# EVI_SG_summary_L_double.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(EVI, 5-step smoothed)")
# plt.grid(axis="y")
# # file_name = plot_dir + "EVI_SG_double_area_trend.pdf"

# %%
# sharey='col', # sharex=True, sharey=True,
# gridspec_kw={'hspace': 0.35, 'wspace': .05}
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
EVI_SG_summary_L_double_2008 = EVI_SG_summary_L_double[EVI_SG_summary_L_double.year>=2008].copy()

ncp.trend_prePost2008(df = EVI_SG_summary_L_double_2008, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary = color_dict,
                      linewidth_ = 4)
# yLims = [0, 460000],
file_name = plot_dir + "EVI_SG_double_area_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %% [markdown]
# ### NDVI regular

# %%
NDVI_regular_summary_L = pd.DataFrame(columns=["label"])

col1 = group_sum_area(NDVI_regular_preds_large, [NDVI_regular_preds_large.columns[2], "year"])
col2 = group_sum_area(NDVI_regular_preds_large, [NDVI_regular_preds_large.columns[3], "year"])
col3 = group_sum_area(NDVI_regular_preds_large, [NDVI_regular_preds_large.columns[4], "year"])
col4 = group_sum_area(NDVI_regular_preds_large, [NDVI_regular_preds_large.columns[5], "year"])

NDVI_regular_summary_L = pd.concat([NDVI_regular_summary_L, col1])
NDVI_regular_summary_L = pd.merge(NDVI_regular_summary_L, col2, on=(["label", "year"]), how="left")
NDVI_regular_summary_L = pd.merge(NDVI_regular_summary_L, col3, on=(["label", "year"]), how="left")
NDVI_regular_summary_L = pd.merge(NDVI_regular_summary_L, col4, on=(["label", "year"]), how="left")

NDVI_regular_summary_L.year = NDVI_regular_summary_L.year.astype(int)
NDVI_regular_summary_L.rename(
    columns={
        "RF_NDVI_regular_preds": "RF",
        "SVM_NDVI_regular_preds": "SVM",
        "KNN_NDVI_regular_preds": "KNN",
        "NDVI_regular_DL_p9": "DL",
    },
    inplace=True,
)

NDVI_regular_summary_L_double = NDVI_regular_summary_L[NDVI_regular_summary_L.label == 2].copy()
NDVI_regular_summary_L.head(2)

# %%
# NDVI_regular_summary_L_double.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(NDVI, 4-step smoothed)")
# plt.grid(axis="y")
# # file_name = plot_dir + "NDVI_regular_double_area_trend.pdf"

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
NDVI_regular_summary_L_double_2008 = NDVI_regular_summary_L_double[NDVI_regular_summary_L_double.year>=2008].copy()

ncp.trend_prePost2008(df = NDVI_regular_summary_L_double_2008, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)

axs.set_ylim([0, 425000])
# axs.yaxis.set_ticks(np.arange(50000, 400000, 50000))
file_name = plot_dir + "NDVI_regular_double_area_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %% [markdown]
# ### EVI regular

# %%
EVI_regular_summary_L = pd.DataFrame(columns=["label"])

col1 = group_sum_area(EVI_regular_preds_large, [EVI_regular_preds_large.columns[2], "year"])
col2 = group_sum_area(EVI_regular_preds_large, [EVI_regular_preds_large.columns[3], "year"])
col3 = group_sum_area(EVI_regular_preds_large, [EVI_regular_preds_large.columns[4], "year"])
col4 = group_sum_area(EVI_regular_preds_large, [EVI_regular_preds_large.columns[5], "year"])

EVI_regular_summary_L = pd.concat([EVI_regular_summary_L, col1])
EVI_regular_summary_L = pd.merge(EVI_regular_summary_L, col2, on=(["label", "year"]), how="left")
EVI_regular_summary_L = pd.merge(EVI_regular_summary_L, col3, on=(["label", "year"]), how="left")
EVI_regular_summary_L = pd.merge(EVI_regular_summary_L, col4, on=(["label", "year"]), how="left")

EVI_regular_summary_L.year = EVI_regular_summary_L.year.astype(int)
EVI_regular_summary_L.rename(
    columns={
        "RF_EVI_regular_preds": "RF",
        "SVM_EVI_regular_preds": "SVM",
        "KNN_EVI_regular_preds": "KNN",
        "EVI_regular_DL_p4": "DL",
    },
    inplace=True,
)

EVI_regular_summary_L_double = EVI_regular_summary_L[EVI_regular_summary_L.label == 2].copy()
EVI_regular_summary_L.head(2)

# %%
# EVI_regular_summary_L_double.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(EVI, 4-step smoothed)")
# plt.grid(axis="y")
# # file_name = plot_dir + "EVI_regular_double_area_trend.pdf"

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
EVI_regular_summary_L_double_2008 = EVI_regular_summary_L_double[EVI_regular_summary_L_double.year>=2008].copy()

ncp.trend_prePost2008(df = EVI_regular_summary_L_double_2008, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)

axs.set_ylim([0, 420000])
axs.yaxis.set_ticks(np.arange(50000, 480000, 50000))
file_name = plot_dir + "EVI_regular_double_area_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %% [markdown]
# # Acre-%-Wise

# %%
total_acr_largeFs = SF_data_large.acreage.sum()

# %%
NDVI_SG_summary_L_double_perc = NDVI_SG_summary_L_double.copy()
EVI_SG_summary_L_double_perc = EVI_SG_summary_L_double.copy()

NDVI_regular_summary_L_double_perc = NDVI_regular_summary_L_double.copy()
EVI_regular_summary_L_double_perc = EVI_regular_summary_L_double.copy()

# %%
ML_cols = ["RF", "SVM", "KNN", "DL"]
EVI_regular_summary_L_double_perc[ML_cols] = (
    EVI_regular_summary_L_double_perc[ML_cols] / total_acr_largeFs
) * 100
NDVI_regular_summary_L_double_perc[ML_cols] = (
    NDVI_regular_summary_L_double_perc[ML_cols] / total_acr_largeFs
) * 100

EVI_SG_summary_L_double_perc[ML_cols] = (
    EVI_SG_summary_L_double_perc[ML_cols] / total_acr_largeFs
) * 100
NDVI_SG_summary_L_double_perc[ML_cols] = (
    NDVI_SG_summary_L_double_perc[ML_cols] / total_acr_largeFs
) * 100

# %%
y_label_ = "area (%)"
title_base = "double-cropped area as percentage "

# %%
# NDVI_SG_summary_L_double_perc.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(NDVI, 5-step smoothed)")
# plt.grid(axis="y")
# # file_name = plot_dir + "NDVI_SG_double_areaPerc_trend.pdf"

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = NDVI_SG_summary_L_double_perc[NDVI_SG_summary_L_double_perc.year>=2008].copy()

ncp.trend_prePost2008(df = df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)

file_name = plot_dir + "NDVI_SG_double_areaPerc_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %%
# EVI_SG_summary_L_double_perc.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(EVI, 5-step smoothed)")
# plt.grid(axis="y")
# # file_name = plot_dir + "EVI_SG_double_areaPerc_trend.pdf"

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = EVI_SG_summary_L_double_perc[EVI_SG_summary_L_double_perc.year>=2008].copy()

ncp.trend_prePost2008(df = df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary = color_dict,
                      linewidth_ = 4)

file_name = plot_dir + "EVI_SG_double_areaPerc_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %%
# EVI_regular_summary_L_double_perc.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(EVI, 4-step smoothed)")
# plt.grid(axis="y")
# # file_name = plot_dir + "EVI_regular_double_areaPerc_trend.pdf"

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = EVI_regular_summary_L_double_perc[EVI_regular_summary_L_double_perc.year>=2008].copy()

ncp.trend_prePost2008(df = df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)
axs.set_ylim([0, 20])
axs.yaxis.set_ticks(np.arange(2.5, 22, 2.5))

file_name = plot_dir + "EVI_regular_double_areaPerc_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %%
# NDVI_regular_summary_L_double_perc.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(NDVI, 4-step smoothed)")
# plt.grid(axis="y")
# # file_name = plot_dir + "NDVI_regular_double_areaPerc_trend.pdf"


# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = NDVI_regular_summary_L_double_perc[NDVI_regular_summary_L_double_perc.year>=2008].copy()

ncp.trend_prePost2008(df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)

file_name = plot_dir + "NDVI_regular_double_areaPerc_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %% [markdown]
# # Count-Wise


# %%
def group_count(df, group_cols):
    """groups by 2 columns given by group_cols.
    group_cols[0] is something like
                            SVM_NDVI_SG_preds
                            SVM_NDVI_regular_preds
                            SVM_EVI_SG_preds
                            SVM_EVI_regular_preds
    """
    df = df[group_cols]
    col = (
        df.groupby([group_cols[0], group_cols[1]])[group_cols[0]]
        .count()
        .reset_index(name=group_cols[0] + "_count")
    )
    col.rename(
        columns={group_cols[0]: "label", group_cols[0] + "_count": group_cols[0]}, inplace=True
    )
    return col


# %% [markdown]
# ### NDVI SG

# %%
NDVI_SG_summary_L = pd.DataFrame(columns=["label"])

col1 = group_count(NDVI_SG_preds_large, [NDVI_SG_preds_large.columns[2], "year"])
col2 = group_count(NDVI_SG_preds_large, [NDVI_SG_preds_large.columns[3], "year"])
col3 = group_count(NDVI_SG_preds_large, [NDVI_SG_preds_large.columns[4], "year"])
col4 = group_count(NDVI_SG_preds_large, [NDVI_SG_preds_large.columns[5], "year"])

NDVI_SG_summary_L = pd.concat([NDVI_SG_summary_L, col1])
NDVI_SG_summary_L = pd.merge(NDVI_SG_summary_L, col2, on=(["label", "year"]), how="left")
NDVI_SG_summary_L = pd.merge(NDVI_SG_summary_L, col3, on=(["label", "year"]), how="left")
NDVI_SG_summary_L = pd.merge(NDVI_SG_summary_L, col4, on=(["label", "year"]), how="left")

NDVI_SG_summary_L.year = NDVI_SG_summary_L.year.astype(int)
NDVI_SG_summary_L.head(2)

NDVI_SG_summary_L.rename(
    columns={
        "RF_NDVI_SG_preds": "RF",
        "SVM_NDVI_SG_preds": "SVM",
        "KNN_NDVI_SG_preds": "KNN",
        "NDVI_SG_DL_p9": "DL",
    },
    inplace=True,
)

NDVI_SG_summary_L_double = NDVI_SG_summary_L[NDVI_SG_summary_L.label == 2].copy()
NDVI_SG_summary_L_double.head(2)

# %%
y_label_ = "field count"
title_base = "double-cropped field count "

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = NDVI_SG_summary_L_double[NDVI_SG_summary_L_double.year>=2008].copy()

ncp.trend_prePost2008(df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)
axs.set_ylim([-500, 5500])
axs.yaxis.set_ticks(np.arange(0, 5500, 1000))
file_name = plot_dir + "NDVI_SG_double_Count_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %% [markdown]
# ### EVI SG

# %%
EVI_SG_summary_L = pd.DataFrame(columns=["label"])

col1 = group_count(EVI_SG_preds_large, [EVI_SG_preds_large.columns[2], "year"])
col2 = group_count(EVI_SG_preds_large, [EVI_SG_preds_large.columns[3], "year"])
col3 = group_count(EVI_SG_preds_large, [EVI_SG_preds_large.columns[4], "year"])
col4 = group_count(EVI_SG_preds_large, [EVI_SG_preds_large.columns[5], "year"])

EVI_SG_summary_L = pd.concat([EVI_SG_summary_L, col1])
EVI_SG_summary_L = pd.merge(EVI_SG_summary_L, col2, on=(["label", "year"]), how="left")
EVI_SG_summary_L = pd.merge(EVI_SG_summary_L, col3, on=(["label", "year"]), how="left")
EVI_SG_summary_L = pd.merge(EVI_SG_summary_L, col4, on=(["label", "year"]), how="left")

EVI_SG_summary_L.year = EVI_SG_summary_L.year.astype(int)
EVI_SG_summary_L.head(2)

EVI_SG_summary_L.rename(
    columns={
        "RF_EVI_SG_preds": "RF",
        "SVM_EVI_SG_preds": "SVM",
        "KNN_EVI_SG_preds": "KNN",
        "EVI_SG_DL_p4": "DL",
    },
    inplace=True,
)

EVI_SG_summary_L_double = EVI_SG_summary_L[EVI_SG_summary_L.label == 2].copy()
EVI_SG_summary_L_double.head(2)

# %%
# EVI_SG_summary_L_double.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(EVI, 5-step smoothed)")
# plt.grid(axis="y")
# # file_name = plot_dir + "EVI_SG_double_Count_trend.pdf"

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = EVI_SG_summary_L_double[EVI_SG_summary_L_double.year>=2008].copy()

ncp.trend_prePost2008(df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)

axs.set_ylim([-500, 6500])
axs.yaxis.set_ticks(np.arange(0, 6500, 1000))

file_name = plot_dir + "EVI_SG_double_Count_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %% [markdown]
# ### NDVI regular

# %%
NDVI_regular_summary_L = pd.DataFrame(columns=["label"])

col1 = group_count(NDVI_regular_preds_large, [NDVI_regular_preds_large.columns[2], "year"])
col2 = group_count(NDVI_regular_preds_large, [NDVI_regular_preds_large.columns[3], "year"])
col3 = group_count(NDVI_regular_preds_large, [NDVI_regular_preds_large.columns[4], "year"])
col4 = group_count(NDVI_regular_preds_large, [NDVI_regular_preds_large.columns[5], "year"])

NDVI_regular_summary_L = pd.concat([NDVI_regular_summary_L, col1])
NDVI_regular_summary_L = pd.merge(NDVI_regular_summary_L, col2, on=(["label", "year"]), how="left")
NDVI_regular_summary_L = pd.merge(NDVI_regular_summary_L, col3, on=(["label", "year"]), how="left")
NDVI_regular_summary_L = pd.merge(NDVI_regular_summary_L, col4, on=(["label", "year"]), how="left")

NDVI_regular_summary_L.year = NDVI_regular_summary_L.year.astype(int)
NDVI_regular_summary_L.head(2)

NDVI_regular_summary_L.rename(
    columns={
        "RF_NDVI_regular_preds": "RF",
        "SVM_NDVI_regular_preds": "SVM",
        "KNN_NDVI_regular_preds": "KNN",
        "NDVI_regular_DL_p9": "DL",
    },
    inplace=True,
)

NDVI_regular_summary_L_double = NDVI_regular_summary_L[NDVI_regular_summary_L.label == 2].copy()
NDVI_regular_summary_L_double.head(2)

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = NDVI_regular_summary_L_double[NDVI_regular_summary_L_double.year>=2008].copy()

ncp.trend_prePost2008(df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)
axs.set_ylim([-500, 4800])
axs.yaxis.set_ticks(np.arange(0, 4300, 1000))
axs.legend(loc="upper left")
file_name = plot_dir + "NDVI_regular_double_Count_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %% [markdown]
# ### EVI regular

# %%
EVI_regular_summary_L = pd.DataFrame(columns=["label"])

col1 = group_count(EVI_regular_preds_large, [EVI_regular_preds_large.columns[2], "year"])
col2 = group_count(EVI_regular_preds_large, [EVI_regular_preds_large.columns[3], "year"])
col3 = group_count(EVI_regular_preds_large, [EVI_regular_preds_large.columns[4], "year"])
col4 = group_count(EVI_regular_preds_large, [EVI_regular_preds_large.columns[5], "year"])

EVI_regular_summary_L = pd.concat([EVI_regular_summary_L, col1])
EVI_regular_summary_L = pd.merge(EVI_regular_summary_L, col2, on=(["label", "year"]), how="left")
EVI_regular_summary_L = pd.merge(EVI_regular_summary_L, col3, on=(["label", "year"]), how="left")
EVI_regular_summary_L = pd.merge(EVI_regular_summary_L, col4, on=(["label", "year"]), how="left")

EVI_regular_summary_L.year = EVI_regular_summary_L.year.astype(int)
EVI_regular_summary_L.head(2)

EVI_regular_summary_L.rename(
    columns={
        "RF_EVI_regular_preds": "RF",
        "SVM_EVI_regular_preds": "SVM",
        "KNN_EVI_regular_preds": "KNN",
        "EVI_regular_DL_p4": "DL",
    },
    inplace=True,
)

EVI_regular_summary_L_double = EVI_regular_summary_L[EVI_regular_summary_L.label == 2].copy()
EVI_regular_summary_L_double.head(2)

# %%
# EVI_regular_summary_L_double.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(EVI, 4-step smoothed)")
# plt.grid(axis="y")
# file_name = plot_dir + "EVI_regular_double_Count_trend.pdf"

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = EVI_regular_summary_L_double[EVI_regular_summary_L_double.year>=2008].copy()

ncp.trend_prePost2008(df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)

file_name = plot_dir + "EVI_regular_double_Count_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %% [markdown]
# # Count-%-Wise

# %%
print(f"{len(SF_data_large.ID.unique())=}")
print(f"{len(SF_data.ID.unique())=}")
total_number_largeFs = len(SF_data_large.ID.unique())

# %%
NDVI_SG_summary_L_double_perc = NDVI_SG_summary_L_double.copy()
EVI_SG_summary_L_double_perc = EVI_SG_summary_L_double.copy()

NDVI_regular_summary_L_double_perc = NDVI_regular_summary_L_double.copy()
EVI_regular_summary_L_double_perc = EVI_regular_summary_L_double.copy()

# %%
ML_cols = ["RF", "SVM", "KNN", "DL"]
EVI_regular_summary_L_double_perc[ML_cols] = (
    EVI_regular_summary_L_double_perc[ML_cols] / total_number_largeFs
) * 100
NDVI_regular_summary_L_double_perc[ML_cols] = (
    NDVI_regular_summary_L_double_perc[ML_cols] / total_number_largeFs
) * 100

EVI_SG_summary_L_double_perc[ML_cols] = (
    EVI_SG_summary_L_double_perc[ML_cols] / total_number_largeFs
) * 100
NDVI_SG_summary_L_double_perc[ML_cols] = (
    NDVI_SG_summary_L_double_perc[ML_cols] / total_number_largeFs
) * 100

# %%

# %%
y_label_ = "field count (%)"
title_base = "double-cropped field count as percentage "

# %%
# NDVI_SG_summary_L_double_perc.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(NDVI, 5-step smoothed)")
# plt.grid(axis="y")

# # file_name = plot_dir + "NDVI_SG_double_countPerc_trend.pdf"
# # plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = NDVI_SG_summary_L_double_perc[NDVI_SG_summary_L_double_perc.year>=2008].copy()

ncp.trend_prePost2008(df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)
axs.set_ylim([-1, 15])
axs.yaxis.set_ticks(np.arange(0, 16, 2.5))
file_name = plot_dir + "NDVI_SG_double_countPerc_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %%
# EVI_SG_summary_L_double_perc.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(EVI, 5-step smoothed)")
# plt.grid(axis="y")
# file_name = plot_dir + "EVI_SG_double_countPerc_trend.pdf"
# plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = EVI_SG_summary_L_double_perc[EVI_SG_summary_L_double_perc.year>=2008].copy()

ncp.trend_prePost2008(df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)

axs.set_ylim([-1, 16])
axs.yaxis.set_ticks(np.arange(0, 17, 2))
file_name = plot_dir + "EVI_SG_double_countPerc_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %%
# NDVI_regular_summary_L_double_perc.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(NDVI, 4-step smoothed)")
# plt.grid(axis="y")
# file_name = plot_dir + "NDVI_regular_double_countPerc_trend.pdf"
# plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = NDVI_regular_summary_L_double_perc[NDVI_regular_summary_L_double_perc.year>=2008].copy()
ncp.trend_prePost2008(df, ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)
# axs.set_ylim([-1, 10])
axs.yaxis.set_ticks(np.arange(0, 14, 2.5))
file_name = plot_dir + "NDVI_regular_double_countPerc_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %%
# EVI_regular_summary_L_double_perc.plot(
#     x="year", y=["RF", "SVM", "KNN", "DL"], kind="line", figsize = fig_size, linewidth=4
# )
# plt.xlabel("year")
# plt.ylabel(y_label_)
# plt.title(title_base + "(EVI, 4-step smoothed)")
# plt.grid(axis="y")

# # file_name = plot_dir + "EVI_regular_double_countPerc_trend.pdf"

# %%
fig, axs = plt.subplots(1, 1, figsize = fig_size, sharex=False);
axs.grid(axis='y', which='both')

axs.set_xlim([2007, 2022])
df = EVI_regular_summary_L_double_perc[EVI_regular_summary_L_double_perc.year>=2008].copy()

ncp.trend_prePost2008(df , ax = axs,
                      ylabel_ = y_label_,
                      title_ = title_base,
                      color_dictionary=color_dict,
                      linewidth_ = 4)
# axs.set_ylim([-1, 13])
axs.yaxis.set_ticks(np.arange(0, 16, 2))
file_name = plot_dir + "EVI_regular_double_countPerc_trend.pdf"
plt.savefig(fname=file_name, dpi=400, bbox_inches="tight", transparent=False)
# plt.close('all')

# %%
EVI_regular_summary_L_double_perc.year.unique()
