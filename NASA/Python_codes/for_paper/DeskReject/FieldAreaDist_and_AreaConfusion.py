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
# Sep. 19, 2024
#
#

# %%
import numpy as np
import pandas as pd
import time, datetime
import sys, os, os.path
from datetime import date, datetime

import matplotlib.pyplot as plt

# %%
plot_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/plots/"

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
ground_truth_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
GT_wMeta = pd.merge(ground_truth_labels, meta, on="ID", how="left")
print (GT_wMeta.shape)
GT_wMeta.head(2)

# %%
import seaborn as sns
sns.displot(GT_wMeta, x="ExctAcr", kind="kde")

# %%
tick_legend_FontSize = 12

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
x_label = "field size (acre)"
title_ = "distribution of field sizes in ground-truth set"

# %%
x = GT_wMeta["ExctAcr"]
fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
fig.tight_layout(pad=5.0)
# axes.tick_params(labelrotation=90)
# plt.yticks(rotation=90)
axes.grid(axis='y', which='both')

plt.hist(x, density=True, bins=100)  # density=False would make counts
plt.ylabel('density')
plt.xlabel(x_label);
plt.title(title_)

ymin, ymax = axes.get_ylim()
axes.set(ylim=(ymin-0.0001, ymax+0.003), axisbelow=True);

file_name = plot_dir + "GT_fieldArea_prob.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%

# %%
x = GT_wMeta["ExctAcr"]

fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
fig.tight_layout(pad=5.0)
# axes.tick_params(labelrotation=90)
# plt.yticks(rotation=90)
axes.grid(axis='y', which='both')

plt.hist(x, density=False, bins=100) # density=False would make counts
plt.ylabel('count')
plt.xlabel(x_label);
plt.title(title_)

ymin, ymax = axes.get_ylim()
axes.set(ylim=(ymin-1, ymax+25), axisbelow=True);

file_name = plot_dir + "GT_fieldArea_Freq.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"hspace": 0.15, "wspace": 0.05},
                         sharex=True, sharey = True)
ax1 , ax2 = axs[0], axs[1]
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
ax1.set_axisbelow(True), ax2.set_axisbelow(True); # sends the grids underneath the plot

x1 = GT_wMeta[GT_wMeta["Vote"] == 1]["ExctAcr"].values
ax1.hist(x1, density=True, bins=100);

x2 = GT_wMeta[GT_wMeta["Vote"] == 2]["ExctAcr"].values
ax2.hist(x2, density=True, bins=100);

plt.suptitle(title_, fontsize=15, y=.94);
ax1.legend(['single-cropped']);
ax2.legend(['double-cropped']);

ax1.set_ylabel('density');
ax2.set_ylabel('density');
ax2.set_xlabel(x_label);
file_name = plot_dir + "GT_fieldArea_prob_SD.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%

# %%
# x = GT_wMeta["ExctAcr"]

# fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
# fig.tight_layout(pad=5.0)
# # axes.tick_params(labelrotation=90)
# # plt.yticks(rotation=90)
# axes.grid(axis='y', which='both')

# plt.hist(x, density=True, bins=100) # density=False would make counts
# plt.ylabel('count')
# plt.xlabel(x_label);
# plt.title(title_)

# ymin, ymax = axes.get_ylim()
# axes.set(ylim=(ymin-1, ymax+25), axisbelow=True);

# file_name = plot_dir + "GT_fieldArea_Freq.pdf"
# # plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"hspace": 0.15, "wspace": 0.05},
                       sharey=True, sharex=True)
ax1, ax2 = axs[0], axs[1]
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
ax1.set_axisbelow(True), ax2.set_axisbelow(True); # sends the grids underneath the plot

x1 = GT_wMeta[GT_wMeta["Vote"] == 1]["ExctAcr"].values
ax1.hist(x1, density=False, bins=100);

x2 = GT_wMeta[GT_wMeta["Vote"] == 2]["ExctAcr"].values
ax2.hist(x2, density=False, bins=100);

plt.suptitle(title_, fontsize=15, y=.94);
ax1.legend(['single-cropped']);
ax2.legend(['double-cropped']);

ax1.set_ylabel('count');
ax2.set_ylabel('count');
ax2.set_xlabel(x_label);
file_name = plot_dir + "GT_fieldArea_prob_Freq_sharey.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%
GT_wMeta.head(2)

# %%
####
#### Directories
####
path_to_data = "/Users/hn/Documents/01_research_data/NASA/Amin/"
file_name = "six_OverSam_TestRes_and_InclusionProb.sav"

####
#### Read file
####
file_path = path_to_data + file_name

all_data_dict = pd.read_pickle(file_path)
print (f"{list(all_data_dict.keys()) = }")
test_set1_DL_res = all_data_dict["six_OverSam_TestRes"]["test_results_DL"]["train_ID0"]["a_test_set_df"]
test_set1_DL_res.head(2)

# %%
print (test_set1_DL_res.shape)

test_set1_DL_res = pd.merge(test_set1_DL_res, GT_wMeta[["ID", "ExctAcr"]], how="left", on="ID")
test_set1_DL_res.head(2)

# %%
# fig, axs = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"hspace": 0.15, "wspace": 0.05},
#                        sharey=True, sharex=True)
# ax1, ax2 = axs[0], axs[1]
# ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
# ax1.set_axisbelow(True), ax2.set_axisbelow(True); # sends the grids underneath the plot

# x1 = test_set1_DL_res[(test_set1_DL_res["Vote"]==1) & 
#                       (test_set1_DL_res["DL_NDVI_SG_prob_point3"]==2)]["ExctAcr"].values
# ax1.hist(x1, density=False, bins=100);
# ax1.legend(['single-cropped, false predict.']);

# x2 = test_set1_DL_res[(test_set1_DL_res["Vote"]==2) & 
#                       (test_set1_DL_res["DL_NDVI_SG_prob_point3"]==1)]["ExctAcr"].values
# ax2.hist(x2, density=False, bins=100);
# ax2.legend(['double-cropped, false predict.']);

# title_2 = "distribution of field sizes in test set that are mislabeled (original split, DL)"
# plt.suptitle(title_2, fontsize=15, y=.94);

# ax1.set_ylabel('count');
# ax2.set_ylabel('count');
# ax2.set_xlabel(x_label);
# file_name = plot_dir + "testSplit0DL_fieldSizedist_.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"
meta_dir = dir_base + "/parameters/"
SF_data_dir   = dir_base + "/data_part_of_shapefile/"
pred_dir_base = dir_base + "/RegionalStatData/"
pred_dir = pred_dir_base + "02_ML_preds_oversampled/"

all_preds_overSample = pd.read_csv(pred_dir_base + "all_preds_overSample.csv")
all_preds_overSample = all_preds_overSample[all_preds_overSample.ExctAcr > 10]
all_preds_overSample.head(2)

# %%
[x for x in list(all_preds_overSample.columns) if "DL" in x]

# %%
# print (GT_wMeta.shape)
# print (all_preds_overSample.shape)
GT_preds_ = all_preds_overSample[all_preds_overSample.ID.isin(list(GT_wMeta.ID.unique()))].copy()
GT_preds_ = GT_preds_[["ID", "ExctAcr", "DL_NDVI_SG_prob_point3"]]
# GT_preds_.rename(columns={"DL_NDVI_regular_prob_point3": "NDVI_SG_DL_p3"}, inplace=True)
GT_preds_ = pd.merge(GT_preds_, GT_wMeta[["ID", "Vote"]], how="left", on="ID")
GT_preds_.head(2)

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"hspace": 0.15, "wspace": 0.05},
                       sharey=True, sharex=True)
ax1, ax2 = axs[0], axs[1]
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
ax1.set_axisbelow(True), ax2.set_axisbelow(True); # sends the grids underneath the plot

x1 = GT_preds_[(GT_preds_["Vote"]==1) & (GT_preds_["DL_NDVI_SG_prob_point3"]==2)]["ExctAcr"].values
ax1.hist(x1, density=False, bins=100);
ax1.legend(['single-cropped, false predict.']);

x2 = GT_preds_[(GT_preds_["Vote"]==2) & (GT_preds_["DL_NDVI_SG_prob_point3"]==1)]["ExctAcr"].values
ax2.hist(x2, density=False, bins=100);
ax2.legend(['double-cropped, false predict.']);

title_2 = "distribution of field sizes in GT set that are mislabeled (original split, DL)"
plt.suptitle(title_2, fontsize=15, y=.94);

ax1.set_ylabel('count');
ax2.set_ylabel('count');
ax2.set_xlabel(x_label);
file_name = plot_dir + "GT_Split0DL_fieldSizedist_.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"hspace": 0.15, "wspace": 0.05},
                       sharey=True, sharex=True)
ax1, ax2 = axs[0], axs[1]
ax1.grid(axis="y", which="both"); ax2.grid(axis="y", which="both")
ax1.set_axisbelow(True), ax2.set_axisbelow(True); # sends the grids underneath the plot

x1 = test_set1_DL_res[(test_set1_DL_res["Vote"]==1) & 
                      (test_set1_DL_res["DL_NDVI_SG_prob_point3"]==2)]["ExctAcr"].values

x2 = test_set1_DL_res[(test_set1_DL_res["Vote"]==2) & 
                      (test_set1_DL_res["DL_NDVI_SG_prob_point3"]==1)]["ExctAcr"].values


_ = ax1.hist(x1, density=False, bins=100);
_ = ax1.hist(x2, density=False, bins=100);
# ax1.legend(['single-cropped, false predict.']);
# ax1.legend(['double-cropped, false predict.']);
ax1.legend(('A1-P2', 'A2-P1'), loc='best');
###########################################################################################
x1 = GT_preds_[(GT_preds_["Vote"]==1) & 
                      (GT_preds_["DL_NDVI_SG_prob_point3"]==2)]["ExctAcr"].values

x2 = GT_preds_[(GT_preds_["Vote"]==2) & 
                      (GT_preds_["DL_NDVI_SG_prob_point3"]==1)]["ExctAcr"].values


_ = ax2.hist(x1, density=False, bins=100);
_ = ax2.hist(x2, density=False, bins=100);
# ax1.legend(['single-cropped, false predict.']);
# ax1.legend(['double-cropped, false predict.']);
ax2.legend(('A1-P2', 'A2-P1'), loc='best');

title_2 = "distribution of field sizes in test (top) and GT" +\
           "(bottom) set that are mislabeled"
plt.suptitle(title_2, fontsize=13, y=.94);

ax1.set_ylabel('count');
ax2.set_ylabel('count');
ax2.set_xlabel(x_label);
file_name = plot_dir + "GT_testSplit0DL_fieldSizedist_mistakes.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%

# %%
tick_legend_FontSize = 12

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize,  #  * 0.75
    "axes.titlepad": 10
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
from matplotlib import rcParams

# sns.set_style("whitegrid")
sns.set_style({'axes.grid' : True})
ax = sns.displot(GT_wMeta, x="ExctAcr", kind="hist", kde=True, height=5, aspect=2,  bins=100); # height=5
ax.set(xlabel=x_label, ylabel='count', title=title_);

ax.despine(ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
# ax.set_xlim([0, 270])

plt.xlim(0, 270);
plt.rcParams["axes.grid.axis"] ="y"

file_name = plot_dir + "GT_fieldArea_Freq_seaborn.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%
GT_wMeta_1 = GT_wMeta[GT_wMeta["Vote"]==1].copy()
GT_wMeta_2 = GT_wMeta[GT_wMeta["Vote"]==2].copy()

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharey=True, sharex=True)
sns.set_style({'axes.grid' : True})
sns.histplot(data=GT_wMeta_1["ExctAcr"], ax=axes[0], bins=100, kde=True); # height=5
sns.histplot(data=GT_wMeta_2["ExctAcr"], ax=axes[1], bins=100, kde=True); # height=5
# sns.countplot(ax=axes[1], x="NDVI", data=L7_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index())
# x="ExctAcr", kind="hist", kde=True, height=5, aspect=2,

plt.suptitle(title_, fontsize=15, y=.94);
axes[0].legend(['single-cropped']);
axes[1].legend(['double-cropped']);

axes[0].set_ylabel("count");
axes[1].set_ylabel("count");
axes[1].set_xlabel(x_label);

file_name = plot_dir + "GT_fieldArea_Freq_seaborn_shareY_SD.pdf"
plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharey=False, sharex=True)
sns.set_style({'axes.grid' : True})
sns.histplot(data=GT_wMeta_1["ExctAcr"], ax=axes[0], bins=100, kde=True); # height=5
sns.histplot(data=GT_wMeta_2["ExctAcr"], ax=axes[1], bins=100, kde=True); # height=5
# sns.countplot(ax=axes[1], x="NDVI", data=L7_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index())
# x="ExctAcr", kind="hist", kde=True, height=5, aspect=2,

plt.suptitle(title_, fontsize=15, y=.94);
axes[0].legend(['single-cropped']);
axes[1].legend(['double-cropped']);

axes[0].set_ylabel("count");
axes[1].set_ylabel("count");
axes[1].set_xlabel(x_label);

file_name = plot_dir + "GT_fieldArea_Freq_seaborn_SD.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%

# %%
A1_P2 = GT_preds_[(GT_preds_["Vote"]==1) & (GT_preds_["DL_NDVI_SG_prob_point3"]==2)].copy()
A1_P2["legend"] = "A1-P2"

A2_P1 = GT_preds_[(GT_preds_["Vote"]==2) & (GT_preds_["DL_NDVI_SG_prob_point3"]==1)].copy()
A2_P1["legend"] = "A2-P1"

false_preds_GT = pd.concat([A1_P2, A2_P1])

A1_P2 = test_set1_DL_res[(test_set1_DL_res["Vote"]==1) & (test_set1_DL_res["DL_NDVI_SG_prob_point3"]==2)].copy()
A1_P2["legend"] = "A1-P2"

A2_P1 = test_set1_DL_res[(test_set1_DL_res["Vote"]==2) & (test_set1_DL_res["DL_NDVI_SG_prob_point3"]==1)].copy()
A2_P1["legend"] = "A2-P1"

false_preds_test = pd.concat([A1_P2, A2_P1])

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharey=True, sharex=True)
sns.set_style({'axes.grid' : True})

g=sns.histplot(data = false_preds_GT, x="ExctAcr", hue="legend", ax=axes[0], bins=100, kde=False, legend=True);
axes[0].set_xlabel(x_label);
g.legend_.set_title(None)
title_2 = "distribution of field sizes in GT (top) and test set (bottom) that are mislabeled"
plt.suptitle(title_2, fontsize=13, y=.95);

g=sns.histplot(data=false_preds_test, x="ExctAcr", hue="legend", ax=axes[1], bins=100, kde=False, legend=True);
axes[1].set_xlabel(x_label);
g.legend_.set_title(None)
title_2 = "distribution of field sizes in test set that are mislabeled"
# plt.suptitle(title_2, fontsize=15, y=1);

# handles, labels = axes.get_legend_handles_labels()
# axes.legend(handles=handles[1:], labels=labels[1:]);

file_name = plot_dir + "GT_testSplit0DL_fieldSizedist_mistakes_seaborn.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%

# %%
# fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharey=True, sharex=True)
# sns.set_style({'axes.grid' : True})
# sns.histplot(data = test_set1_DL_res[(test_set1_DL_res["Vote"]==1) & 
#                                      (test_set1_DL_res["DL_NDVI_SG_prob_point3"]==2)]["ExctAcr"], 
#              ax=axes[0], bins=100, kde=False); # height=5

# sns.histplot(data = test_set1_DL_res[(test_set1_DL_res["Vote"]==2) & 
#                                      (test_set1_DL_res["DL_NDVI_SG_prob_point3"]==1)]["ExctAcr"], 
#              ax=axes[1], bins=100, kde=False); # height=5

# # sns.countplot(ax=axes[1], x="NDVI", data=L7_NDVI.groupby(["ID", "year"])["NDVI"].count().reset_index())
# # x="ExctAcr", kind="hist", kde=True, height=5, aspect=2,
# title_2 = "distribution of field sizes in test set that are mislabeled (original split, DL)"
# plt.suptitle(title_2, fontsize=15, y=.94);
# axes[0].legend(['single-cropped, mislabeled']);
# axes[1].legend(['double-cropped, mislabeled']);

# axes[0].set_ylabel("count");
# axes[1].set_ylabel("count");
# axes[1].set_xlabel(x_label);

# file_name = plot_dir + "testSplit0DL_fieldSizedist_seaborn.pdf"
# plt.savefig(fname = file_name, dpi=400, bbox_inches='tight', transparent=False);

# %%
del(test_set1_DL_res, GT_preds_, all_data_dict, all_preds_overSample)

# %%

# %% [markdown]
# # Area-Based Confusions

# %%
####
#### Directories
####
path_to_data = "/Users/hn/Documents/01_research_data/NASA/Amin/"
file_name = "six_OverSam_TestRes_and_InclusionProb.sav"

# %%
####
#### Read file
####
file_path = path_to_data + file_name

all_data_dict = pd.read_pickle(file_path)
print (f"{list(all_data_dict.keys()) = }")

field_areas = all_data_dict["field_info"][["ID", "ExctAcr"]]
test_set1_DL_res = all_data_dict["six_OverSam_TestRes"]["test_results_DL"]["train_ID1"]["a_test_set_df"]

field_areas.head(2)

# %%
field_areas.shape

# %%
test_set1_DL_res.head(3)

# %%

# %%
stats = ["A1_P1", "A2_P2", "A1_P2", "A2_P1"]
MLs   = ["SVM", "RF", "KNN", "DL"]

splits = ['train_ID0', 'train_ID1', 'train_ID2', 'train_ID3', 'train_ID4', 'train_ID5']

# %%

# %%
all_ABCM = {} # all Area-Based Confusion Matrices
all_ABCM_ratios = {}

for a_split in splits:
    split_summary = pd.DataFrame(columns=["stats"] + MLs, index=np.arange(len(stats)))
    split_summary["stats"] = stats
    for a_col in split_summary.columns[1:]:
        split_summary[a_col] = split_summary[a_col].astype(np.float64)
        
    for an_ML in MLs:
        currSplit_currML_dict = all_data_dict["six_OverSam_TestRes"]["test_results_" + an_ML][a_split].copy()

        curr_test = currSplit_currML_dict["a_test_set_df"].copy()
        curr_ML_col = [x for x in curr_test.columns if "NDVI_SG_" in x]
        curr_ML_col = curr_ML_col[0]

        curr_test = pd.merge(curr_test, GT_wMeta[["ID", "ExctAcr"]], on="ID", how="left")
        curr_test.head(2)

        split_summary.loc[split_summary["stats"]=="A1_P1", an_ML] = \
           int(curr_test[(curr_test["Vote"] == 1 ) & (curr_test[curr_ML_col]==1)]["ExctAcr"].sum())

        split_summary.loc[split_summary["stats"]=="A2_P2", an_ML] = \
           int(curr_test[(curr_test["Vote"] == 2 ) & (curr_test[curr_ML_col]==2)]["ExctAcr"].sum())


        split_summary.loc[split_summary["stats"]=="A1_P2", an_ML] = \
           int(curr_test[(curr_test["Vote"] == 1 ) & (curr_test[curr_ML_col]==2)]["ExctAcr"].sum())


        split_summary.loc[split_summary["stats"]=="A2_P1", an_ML] = \
           int(curr_test[(curr_test["Vote"] == 2) & (curr_test[curr_ML_col]==1)]["ExctAcr"].sum())
        
        cols = list(split_summary.columns)[1:]
        split_summary_ratios = split_summary.copy()
        split_summary_ratios[cols] = split_summary_ratios[cols] / split_summary_ratios.sum()[1:].values

        
    for a_col in split_summary.columns[1:]:
        split_summary[a_col] = split_summary[a_col].astype(np.int64)
        
    for a_col in split_summary.columns[1:]:
        for idx in split_summary.index:
            split_summary.loc[idx, a_col] = "{:,}".format(split_summary.loc[idx, a_col])
            
    for a_col in split_summary_ratios.columns[1:]:
        split_summary_ratios[a_col] = split_summary_ratios[a_col].astype(np.float64)
        
    split_summary_ratios = split_summary_ratios
            
    all_ABCM[a_split] = split_summary
    all_ABCM_ratios[a_split] = split_summary_ratios

# %%

# %%
import pickle
from datetime import datetime

filename = path_to_data + "area_based_confusion.sav"
# pickle.dump(all_ABCM, open(filename, 'wb'))

export_ = {"area_based_confusion": all_ABCM,
           "area_based_confusion_ratios": all_ABCM_ratios,
           "source_code" : "FieldAreaDist_and_AreaConfusion",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
all_ABCM_ratios["train_ID5"].values

# %%

# %%

# %%

# %%

# %%
