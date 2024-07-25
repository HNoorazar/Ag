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
import pandas as pd
import time, datetime
import sys, os, os.path
import scipy, scipy.signal

from datetime import date, datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score, \
                            confusion_matrix, balanced_accuracy_score, \
                            classification_report
import matplotlib.pyplot as plt

# from patsy import cr
# from pprint import pprint
# from statsmodels.sandbox.regression.predstd import wls_prediction_std

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
import NASA_plot_core as ncp

# %%
NASA_dir_base = "/Users/hn/Documents/01_research_data/NASA/"
meta_dir = NASA_dir_base + "0000_parameters/"
train_TS_dir_base = NASA_dir_base + "VI_TS/"

data_part_of_shapefile_dir = NASA_dir_base + "data_part_of_shapefile/"


# %%
out_name = "/Users/hn/Documents/01_research_data/NASA/" + "all_fields_correct_year_irr_noNass.csv"
pool = pd.read_csv(out_name)
pool.head(2)

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
training_set_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
ground_truth_labels = pd.read_csv(training_set_dir+"groundTruth_labels_Oct17_2022.csv")
print ("Unique Votes: ", ground_truth_labels.Vote.unique())
print (len(ground_truth_labels.ID.unique()))
ground_truth_labels.head(2)

# %%
GT_wMeta = pd.merge(ground_truth_labels, meta, on="ID", how="left")
print (GT_wMeta.shape)
GT_wMeta.head(2)

# %%
GT_crops = GT_wMeta["CropTyp"].unique()
meta_crops = meta["CropTyp"].unique()

print (len(GT_crops))
print (len(meta_crops))

# %%
[x for x in meta_crops if not(x in GT_crops)]

# %%

# %%
# csv_data_parts = os.listdir(data_part_of_shapefile_dir)
# csv_data_parts = [x for x in csv_data_parts if x.endswith("csv")]
# csv_data_parts

# %%
# irriigated_SF_data_concatenated = pd.read_csv(data_part_of_shapefile_dir + "irriigated_SF_data_concatenated.csv")
# irriigated_SF_data_concatenated["survey_year"] = pd.to_datetime(irriigated_SF_data_concatenated["LstSrvD"], 
#                                                                 format='mixed').dt.year

# print (irriigated_SF_data_concatenated.county.unique())

# irriigated_SF_data_concatenated["SF_year"] = irriigated_SF_data_concatenated["ID"].str.split("_", expand=True)[3]
# irriigated_SF_data_concatenated["SF_year"] = irriigated_SF_data_concatenated["SF_year"].astype(int)
# irriigated_SF_data_concatenated.head(2)

# %%
# (irriigated_SF_data_concatenated["SF_year"] == irriigated_SF_data_concatenated["survey_year"]).sum()

# print (irriigated_SF_data_concatenated.shape)
# # limit to large fields
# irriigated_SF_data_concatenated = irriigated_SF_data_concatenated[irriigated_SF_data_concatenated.ExctAcr > 10]
# print (len(list(irriigated_SF_data_concatenated.CropTyp.unique())))
# irriigated_SF_data_concatenated.head(2)

# %%
print (len(pool.CropTyp.unique()))
print (len(GT_wMeta.CropTyp.unique()))

# %%
# pool = pool[pool.ExctAcr > 10].copy()
pool = pool[pool.CropTyp.isin(list(GT_wMeta.CropTyp.unique()))]

# %%
GT_wMeta.head(2)

# %%
numer = pd.DataFrame(GT_wMeta.groupby(["CropTyp"])["ID"].count()).reset_index()
numer.rename(columns={"ID": "numer"}, inplace=True)
numer.head(2)

# %%
denom = pd.DataFrame(pool.groupby(["CropTyp"])["ID"].count()).reset_index()
denom.rename(columns={"ID": "denom"}, inplace=True)
denom.head(2)

# %%
in_df = pd.DataFrame({"CropTyp" : sorted(pool.CropTyp.unique())})
in_df.head(2)

# %%
print (in_df.shape)
in_df = pd.merge(in_df, numer, on="CropTyp", how="outer")
in_df = pd.merge(in_df, denom, on="CropTyp", how="outer")
print (in_df.shape)
in_df.head(2)

# %%
in_df["inclusion_prob"] = in_df["numer"] / in_df["denom"]
in_df = in_df.round(3)
in_df.head(2)

# %%
GT_wMeta.head(2)

# %%
numer_acr = pd.DataFrame(GT_wMeta.groupby(["CropTyp"])["ExctAcr"].sum()).reset_index()
numer_acr.rename(columns={"ExctAcr": "numer_acr"}, inplace=True)
numer_acr.head(2)

# %%
denom_acr = pd.DataFrame(pool.groupby(["CropTyp"])["ExctAcr"].sum()).reset_index()
denom_acr.rename(columns={"ExctAcr": "denom_acr"}, inplace=True)
denom_acr.head(2)

# %%
in_df = pd.merge(in_df, numer_acr, on="CropTyp", how="outer")
in_df = pd.merge(in_df, denom_acr, on="CropTyp", how="outer")
in_df = in_df.round(3)
in_df.head(2)

# %%
# I need to comment this out after fixing original ID
# out_name = "/Users/hn/Documents/01_research_data/Amin/inclusion.csv"
# in_df.to_csv(out_name, index = False)

# %%
pool.head(2)

# %%
sorted(list(pool.Irrigtn.unique()))

# %%
sorted(list(pool.DataSrc.unique()))

# %%
pool.county.unique()

# %%
pool["SF_year"] = pool["ID"].str.split("_", expand=True)[3]
pool["SF_year"] = pool["SF_year"].astype(int)

pool_Adams = pool[pool.county == "Adams"].copy()
pool_Benton = pool[pool.county == "Benton"].copy()
pool_Franklin = pool[pool.county == "Franklin"].copy()
pool_Yakima = pool[pool.county == "Yakima"].copy()
pool_Grant = pool[pool.county == "Grant"].copy()
pool_Walla = pool[pool.county == "Walla Walla"].copy()

# %%
print (pool_Adams.SF_year.unique())
print (pool_Benton.SF_year.unique())
print (pool_Franklin.SF_year.unique())
print (pool_Yakima.SF_year.unique())
print (pool_Grant.SF_year.unique())
print (pool_Walla.SF_year.unique())

# %%
import pickle
from datetime import datetime
dir_ = "/Users/hn/Documents/01_research_data/NASA/Amin/"
filename = dir_ + "5_OverSampled_TestResults.sav"
five_OverSam_TestRes = pd.read_pickle(filename)
five_OverSam_TestRes.keys()

# %%
five_OverSam_TestRes["Date"]

# %%
five_OverSam_TestRes = five_OverSam_TestRes["ML_test_results"]
five_OverSam_TestRes.keys()

# %%
for _key in five_OverSam_TestRes.keys():
    # remove the 6th split. That's the one with different size
    if "train_ID6" in five_OverSam_TestRes[_key].keys():
        five_OverSam_TestRes[_key].pop("train_ID6")


# %%
for _key in five_OverSam_TestRes.keys():
    print (_key)
    print (five_OverSam_TestRes[_key].keys())

# %%
for _key in five_OverSam_TestRes.keys():
    for train_ID in five_OverSam_TestRes[_key].keys():
        five_OverSam_TestRes[_key][train_ID]["a_test_set_df"] = \
             pd.merge(five_OverSam_TestRes[_key][train_ID]["a_test_set_df"], 
                      pool[["ID", "CropTyp"]], how="left", on="ID")

# %%
five_OverSam_TestRes[_key][train_ID]["a_test_set_df"].head(2)

# %%
five_OverSam_TestRes.keys()

# %%
five_OverSam_TestRes['inclusion_prob'] = in_df

# %%
five_OverSam_TestRes.keys()

# %%
GT_wMeta.shape

# %%
five_OverSam_TestRes.keys()

# %%
five_OverSam_TestRes["inclusion_prob"]

# %%
five_OverSam_TestRes["test_results_SVM"]["train_ID0"]["a_test_set_df"]

# %%
import pickle
from datetime import datetime
dir_ = "/Users/hn/Documents/01_research_data/NASA/Amin/"
filename = dir_ + "six_OverSam_TestRes_and_InclusionProb.sav"

export_ = {"six_OverSam_TestRes": five_OverSam_TestRes,
           "field_info" : GT_wMeta,
           "source_code" : "inclusion.ipynb",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%

# %%
