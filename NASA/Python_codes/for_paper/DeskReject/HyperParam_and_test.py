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
import pandas as pd
import numpy as np

# %%
merged_pred_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/DeskRejectResults/01_merged_preds/"


# %%
def kappa(A2P2, A1P2, A1P1, A2P1):
    num = 2 * (A2P2*A1P1 - A2P1 * A1P2)
    denom = (A2P2+A1P2) * (A1P2+A1P1) + (A2P2 + A2P1) * (A2P1 + A1P1)
    return round((num/denom), 2)


# %%
DL_preds = pd.read_csv(merged_pred_dir + "all_DL_preds.csv")
DL_preds.head(2)

# %%
DL_preds_train = DL_preds[DL_preds.train_test == "train"].copy()
DL_preds_train.reset_index(drop=True, inplace=True)
DL_preds_train.head(2)

# %%
print (sorted(DL_preds_train.SR.unique()))
print (sorted(DL_preds_train.train_ID.unique()))

# %%
# prob_NDVI = 0.3
# colName = "NDVI_SG_DL_p3"
# NDVI_SG_preds[colName] = -1
# NDVI_SG_preds.loc[NDVI_SG_preds.NDVI_SG_DL_p_single < prob_NDVI, colName] = 2
# NDVI_SG_preds.loc[NDVI_SG_preds.NDVI_SG_DL_p_single >= prob_NDVI, colName] = 1
# NDVI_SG_preds.drop(['NDVI_SG_DL_p_single'], axis=1, inplace=True)


# %%
cuts_ = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_params = {}

for a_train_ID in sorted(DL_preds_train.train_ID.unique()):
    print (f"{a_train_ID = }")
    best_in_set = {# "train_ID": a_train_ID,
                   "cut":0, 
                   "ratio":0, 
                   "acc.":-1, 
                   "UA":-1, 
                   "PA":-1, 
                   "A1P1":-10, 
                   "A1P2":-10, 
                   "A2P2":-10, 
                   "A2P1":-10, 
                   "err_count":100000}
    a_train_set = DL_preds_train[DL_preds_train.train_ID == a_train_ID].copy()
    
    for a_SR in sorted(a_train_set.SR.unique()):
        a_train_set_SR = a_train_set[a_train_set.SR == a_SR].copy()
        
        for a_cut in cuts_:
            colName = "NDVI_SG_DL_p" + str(int(a_cut*10))
            a_train_set_SR[colName] = -1
            a_train_set_SR.loc[a_train_set_SR.prob_single < a_cut, colName] = 2
            a_train_set_SR.loc[a_train_set_SR.prob_single >= a_cut, colName] = 1
            
            error_count = (a_train_set_SR.Vote != a_train_set_SR[colName]).sum()
            accuracy = (len(a_train_set_SR) - error_count) / len(a_train_set_SR)
            
            A2P2_df = a_train_set_SR[a_train_set_SR.Vote==2]
            A2P2 = (A2P2_df.Vote == A2P2_df[colName]).sum()
            A2P1 = len(A2P2_df) - A2P2
            
            if error_count < best_in_set["err_count"]:
                best_in_set["cut"] = a_cut
                best_in_set["ratio"] = a_SR
                best_in_set["acc."] = round(accuracy, 3)
                best_in_set["A2P2"] = A2P2
                best_in_set["A2P1"] = A2P1
                best_in_set["err_count"] = error_count
                
            elif error_count == best_in_set["err_count"]:
                if A2P2 > best_in_set["A2P2"]:
                    best_in_set["cut"] = a_cut
                    best_in_set["ratio"] = a_SR
                    best_in_set["acc."] = round(accuracy, 3)
                    best_in_set["A2P2"] = A2P2
                    best_in_set["A2P1"] = A2P1
                    best_in_set["err_count"] = error_count
                    
    print (best_in_set)
    best_params["train_ID" + str(a_train_ID)] = best_in_set
    print ("=============================================================================")

# %%

# %% [markdown]
# # Test Time

# %%
del(DL_preds_train)
DL_preds_test = DL_preds[DL_preds.train_test == "test"].copy()
DL_preds_test.reset_index(drop=True, inplace=True)
DL_preds_test.head(2)

# %%
test_results = {}
for a_train_ID in sorted(DL_preds_test.train_ID.unique()):
    a_test_set = DL_preds_test[DL_preds_test.train_ID == a_train_ID].copy()
    curr_best_params = best_params["train_ID" + str(a_train_ID)]
    
    best_cut = curr_best_params["cut"]
    best_ratio = curr_best_params["ratio"]
    
    a_test_set_SR = a_test_set[a_test_set.SR == best_ratio].copy()
    
#     print (f"{a_train_ID = }")
#     print (f"{a_test_set_SR.shape = }") 

    colName = "NDVI_SG_DL_p" + str(int(best_cut*10))
    a_test_set_SR[colName] = -1
    a_test_set_SR.loc[a_test_set_SR.prob_single < a_cut, colName] = 2
    a_test_set_SR.loc[a_test_set_SR.prob_single >= a_cut, colName] = 1
    
    
    error_count = (a_test_set_SR.Vote != a_test_set_SR[colName]).sum()
    accuracy = (len(a_test_set_SR) - error_count) / len(a_test_set_SR)

    A2P2_df = a_test_set_SR[a_test_set_SR.Vote==2]
    A2P2 = (A2P2_df.Vote == A2P2_df[colName]).sum()
    A2P1 = len(A2P2_df) - A2P2
    
    A1P1_df = a_test_set_SR[a_test_set_SR.Vote==1]
    A1P1 = (A1P1_df.Vote == A1P1_df[colName]).sum()
    A1P2 = len(A1P1_df) - A1P1
    
    curr_test_result = {"acc" : round(accuracy, 3),
                        "UA": round((A2P2) / (A1P2 + A2P2), 3),
                        "PA" : round((A2P2) / (A2P1 + A2P2), 3),
                        "err_count" : error_count,
                        "A2P2": A2P2,
                        "A2P1": A2P1,
                        "A1P1": A1P1,
                        "A1P2":A1P2, 
                        "kappa": kappa(A2P2, A1P2, A1P1, A2P1),
                        "a_test_set_df" : a_test_set_SR
                       }
    test_results["train_ID" + str(a_train_ID)] = curr_test_result

# %%
a_train_ID

# %%
# for a_key in test_results.keys():
#     print (a_key)
#     print (test_results[a_key])
#     print ("==========================================================================")


# %%
# 0.679 0.737 0.76 0.806  0.826 .92

# %%
test_results_DL = test_results.copy()

# %%
test_results.keys()

# %%

# %% [markdown]
# ### Non-DL MLs

# %%
all_preds = pd.read_csv(merged_pred_dir+"all_preds.csv")
all_preds.head(2)

# %%
best_params_RF = {}
best_params_SVM = {}
best_params_KNN={}

for a_ML in ["RF", "SVM", "KNN"]:
    pred_col = a_ML + "_NDVI_SG_preds"
    ML_preds_df = all_preds[["ID", "Vote", "train_test", "train_ID", "SR", pred_col]].copy()
    
    ML_preds_train = ML_preds_df[ML_preds_df.train_test == "train"].copy()
    ML_preds_train.reset_index(drop=True, inplace=True)

    for a_train_ID in sorted(ML_preds_train.train_ID.unique()):
        # print (f"{a_train_ID = }")
        best_in_set = {# "train_ID": a_train_ID,
                       # "cut":0, 
                       "ratio":0, 
                       "acc.":-1, 
                       "double_agreement":-10, 
                       "double_disagreement":-10, 
                       "err_count":100000}
        a_train_set = ML_preds_train[ML_preds_train.train_ID == a_train_ID].copy()

        for a_SR in sorted(a_train_set.SR.unique()):
            a_train_set_SR = a_train_set[a_train_set.SR == a_SR].copy()
            
            error_count = (a_train_set_SR.Vote != a_train_set_SR[pred_col]).sum()
            accuracy = (len(a_train_set_SR) - error_count) / len(a_train_set_SR)

            double_agreement_df = a_train_set_SR[a_train_set_SR.Vote==2]
            double_agreement = (double_agreement_df.Vote == double_agreement_df[pred_col]).sum()
            double_disagreement = len(double_agreement_df) - double_agreement

            if error_count < best_in_set["err_count"]:
                best_in_set["ratio"] = a_SR
                best_in_set["acc."] = round(accuracy, 3)
                best_in_set["double_agreement"] = double_agreement
                best_in_set["double_disagreement"] = double_disagreement
                best_in_set["err_count"] = error_count

            elif error_count == best_in_set["err_count"]:
                if double_agreement > best_in_set["double_agreement"]:
                    best_in_set["ratio"] = a_SR
                    best_in_set["acc."] = round(accuracy, 3)
                    best_in_set["double_agreement"] = double_agreement
                    best_in_set["double_disagreement"] = double_disagreement
                    best_in_set["err_count"] = error_count

#         print (a_ML)
#         print (best_in_set)
        if a_ML == "RF":
            best_params_RF["train_ID" + str(a_train_ID)] = best_in_set
        elif a_ML == "SVM":
            best_params_SVM["train_ID" + str(a_train_ID)] = best_in_set
        elif a_ML == "KNN":
            best_params_KNN["train_ID" + str(a_train_ID)] = best_in_set
        # print ("=============================================================================")


# %%
best_params = {"best_params_RF": best_params_RF,
               "best_params_SVM" : best_params_SVM,
               "best_params_KNN" : best_params_KNN
              }

# %%
curr_best_params

# %% [markdown]
# ### Test Set - MLs

# %%
test_results_SVM = {}
test_results_KNN = {}
test_results_RF = {}
counter = 0

for a_ML in ["RF", "SVM", "KNN"]:
    pred_col = a_ML + "_NDVI_SG_preds"
    best_param_name = "best_params_" + a_ML
    ML_preds_df = all_preds[["ID", "Vote", "train_test", "train_ID", "SR", pred_col]].copy()
    
    ML_preds_test = ML_preds_df[ML_preds_df.train_test == "test"].copy()
    ML_preds_test.reset_index(drop=True, inplace=True)

    for a_train_ID in sorted(ML_preds_test.train_ID.unique()):
#         counter += 1
#         print (f"{counter = }")
        a_test_set = ML_preds_test[ML_preds_test.train_ID == a_train_ID].copy()
        
       #  print (f"{a_train_ID = }")
       #  print (f"{a_ML = }")
        # print (a_test_set.train_test.unique())
        # print (a_test_set.ID[-4:])
        
        curr_best_params = best_params[best_param_name]["train_ID" + str(a_train_ID)]
        # print (curr_best_params)
        best_ratio = curr_best_params["ratio"]
        a_test_set_SR = a_test_set[a_test_set.SR == best_ratio].copy()

        error_count = (a_test_set_SR.Vote != a_test_set_SR[pred_col]).sum()
        accuracy = (len(a_test_set_SR) - error_count) / len(a_test_set_SR)

        A2P2_df = a_test_set_SR[a_test_set_SR.Vote==2]
        A2P2 = (A2P2_df.Vote == A2P2_df[pred_col]).sum()
        A2P1 = len(A2P2_df) - A2P2

        A1P1_df = a_test_set_SR[a_test_set_SR.Vote==1]
        A1P1 = (A1P1_df.Vote == A1P1_df[pred_col]).sum()
        A1P2 = len(A1P1_df) - A1P1
        
        if A2P2!=0 or A1P2 != 0:
            UA = round(A2P2 / (A1P2 + A2P2), 3)
        else:
            UA = "NA"

        curr_test_result = {"acc" : round(accuracy, 3),
                            "UA": UA,
                            "PA": round(A2P2 / (A2P1 + A2P2), 3),
                            "err_count" : error_count,
                            "A2P2": A2P2,
                            "A2P1":A2P1,
                            "A1P1": A1P1,
                            "A1P2":A1P2, 
                            "kappa": kappa(A2P2, A1P2, A1P1, A2P1),
                            "a_test_set_df" : a_test_set_SR
                           }

        if a_ML == "SVM":
            test_results_SVM["train_ID" + str(a_train_ID)] = curr_test_result
        elif a_ML == "RF":
            test_results_RF["train_ID" + str(a_train_ID)] = curr_test_result
        elif a_ML == "KNN":
            test_results_KNN["train_ID" + str(a_train_ID)] = curr_test_result
        
        # del(error_count, accuracy, A2P2_df, A2P2, A2P1)
        del(curr_best_params, a_train_ID)

# %%
ML_test_results = {"test_results_SVM" : test_results_SVM,
                   "test_results_RF" : test_results_RF,
                   "test_results_KNN" : test_results_KNN}

# %%
for a_key in test_results_SVM.keys():
    print (a_key)
    print (test_results_SVM[a_key])
    print ("==========================================================================")

# %%
# for a_key in test_results_KNN.keys():
#     print (a_key)
#     print (test_results_KNN[a_key])
#     print ("==========================================================================")

# %%
# for a_key in test_results_RF.keys():
#     print (a_key)
#     print (test_results_RF[a_key])
#     print ("==========================================================================")

# %%
ML_test_results.keys()

# %%
ML_test_results["test_results_SVM"]["train_ID1"]["a_test_set_df"]

# %%
ML_test_results["test_results_DL"] = test_results_DL

# %%
ML_test_results.keys()

# %%
ML_test_results["test_results_SVM"].keys()

# %% [markdown]
# # Original Split

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
pred_cols = [x for x in all_preds_overSample.columns if 'NDVI' in x]
pred_cols = [x for x in pred_cols if 'SG' in x]
pred_cols
all_preds_overSample = all_preds_overSample[["ID", "CropTyp"] + pred_cols].copy()

# %%
ML_data_Oct17_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
original_test = pd.read_csv(ML_data_Oct17_dir + "test20_split_2Bconsistent_Oct17.csv")
original_test.head(2)

# %%
###########
########### we need to pick up train_ID from original split.
###########
original_test_IDs = list(original_test["ID"])
all_preds_overSample = all_preds_overSample[all_preds_overSample.ID.isin(original_test_IDs)].copy()
all_preds_overSample.reset_index(drop=True, inplace=True)
all_preds_overSample.head(3)

# %%
all_preds_overSample.shape

# %%
DL_orig = all_preds_overSample[["ID", "DL_NDVI_SG_prob_point3"]].copy()
DL_orig = pd.merge(DL_orig, original_test, how="left", on="ID")
DL_orig.head(2)

# %%
DL_orig[(DL_orig["Vote"] == 1) & (DL_orig["DL_NDVI_SG_prob_point3"]==1)].shape

# %%
# A = ML_test_results["test_results_SVM"]["train_ID1"]["a_test_set_df"]
# A.head(2)

# %%
original_split_test_SVM = all_preds_overSample.copy()
original_split_test_SVM = original_split_test_SVM[["ID", "SVM_NDVI_SG_preds"]].copy()
original_split_test_SVM = pd.merge(original_split_test_SVM, original_test, on="ID", how="left")
original_split_test_SVM["train_test"] = "test"
original_split_test_SVM["train_ID"] = 0
original_split_test_SVM.head(2)

# %%
DL_orig[(DL_orig["Vote"] == 1) & (DL_orig["DL_NDVI_SG_prob_point3"]==1)].shape

# %%
pred_col = "SVM_NDVI_SG_preds"
error_count = (original_split_test_SVM.Vote != original_split_test_SVM[pred_col]).sum()
accuracy = (len(original_split_test_SVM) - error_count) / len(original_split_test_SVM)

A2P2_df = original_split_test_SVM[original_split_test_SVM.Vote==2]
A2P2 = (A2P2_df.Vote == A2P2_df[pred_col]).sum()
A2P1 = len(A2P2_df) - A2P2

A1P1_df = original_split_test_SVM[original_split_test_SVM.Vote==1]
A1P1 = (A1P1_df.Vote == A1P1_df[pred_col]).sum()
A1P2 = len(A1P1_df) - A1P1

if A2P2!=0 or A1P2 != 0:
    UA = round(A2P2 / (A1P2 + A2P2), 3)
else:
    UA = "NA"

original_split_test_SVM_dict = {"acc" : round(accuracy, 3),
                                "UA": UA,
                                "PA": round(A2P2 / (A2P1 + A2P2), 3),
                                "err_count" : error_count,
                                "A2P2": A2P2,
                                "A2P1":A2P1,
                                "A1P1": A1P1,
                                "A1P2":A1P2, 
                                "kappa": kappa(A2P2, A1P2, A1P1, A2P1),
                                "a_test_set_df" : original_split_test_SVM
                               }

# %%

# %%
original_split_test_RF = all_preds_overSample.copy()
original_split_test_RF = original_split_test_RF[["ID", "RF_NDVI_SG_preds"]].copy()
original_split_test_RF = pd.merge(original_split_test_RF, original_test, on="ID", how="left")
original_split_test_RF["train_test"] = "test"
original_split_test_RF["train_ID"] = 0
original_split_test_RF.head(2)

# %%
pred_col = "RF_NDVI_SG_preds"
error_count = (original_split_test_RF.Vote != original_split_test_RF[pred_col]).sum()
accuracy = (len(original_split_test_RF) - error_count) / len(original_split_test_RF)

A2P2_df = original_split_test_RF[original_split_test_RF.Vote==2]
A2P2 = (A2P2_df.Vote == A2P2_df[pred_col]).sum()
A2P1 = len(A2P2_df) - A2P2

A1P1_df = original_split_test_RF[original_split_test_RF.Vote==1]
A1P1 = (A1P1_df.Vote == A1P1_df[pred_col]).sum()
A1P2 = len(A1P1_df) - A1P1

if A2P2!=0 or A1P2 != 0:
    UA = round(A2P2 / (A1P2 + A2P2), 3)
else:
    UA = "NA"

original_split_test_RF_dict = {"acc" : round(accuracy, 3),
                                "UA": UA,
                                "PA": round(A2P2 / (A2P1 + A2P2), 3),
                                "err_count" : error_count,
                                "A2P2": A2P2,
                                "A2P1":A2P1,
                                "A1P1": A1P1,
                                "A1P2":A1P2, 
                                "kappa": kappa(A2P2, A1P2, A1P1, A2P1),
                                "a_test_set_df" : original_split_test_RF
                               }

# %%

# %%
original_split_test_KNN = all_preds_overSample.copy()
original_split_test_KNN = original_split_test_KNN[["ID", "KNN_NDVI_SG_preds"]].copy()
original_split_test_KNN = pd.merge(original_split_test_KNN, original_test, on="ID", how="left")
original_split_test_KNN["train_test"] = "test"
original_split_test_KNN["train_ID"] = 0
original_split_test_KNN.head(2)

# %%
pred_col = "KNN_NDVI_SG_preds"
error_count = (original_split_test_KNN.Vote != original_split_test_KNN[pred_col]).sum()
accuracy = (len(original_split_test_KNN) - error_count) / len(original_split_test_KNN)

A2P2_df = original_split_test_KNN[original_split_test_KNN.Vote==2]
A2P2 = (A2P2_df.Vote == A2P2_df[pred_col]).sum()
A2P1 = len(A2P2_df) - A2P2

A1P1_df = original_split_test_KNN[original_split_test_KNN.Vote==1]
A1P1 = (A1P1_df.Vote == A1P1_df[pred_col]).sum()
A1P2 = len(A1P1_df) - A1P1

if A2P2!=0 or A1P2 != 0:
    UA = round(A2P2 / (A1P2 + A2P2), 3)
else:
    UA = "NA"

original_split_test_KNN_dict = {"acc" : round(accuracy, 3),
                                "UA": UA,
                                "PA": round(A2P2 / (A2P1 + A2P2), 3),
                                "err_count" : error_count,
                                "A2P2": A2P2,
                                "A2P1":A2P1,
                                "A1P1": A1P1,
                                "A1P2":A1P2, 
                                "kappa": kappa(A2P2, A1P2, A1P1, A2P1),
                                "a_test_set_df" : original_split_test_KNN
                               }

# %%

# %%
original_split_test_DL = all_preds_overSample.copy()
original_split_test_DL = original_split_test_DL[["ID", "DL_NDVI_SG_prob_point3"]].copy()
original_split_test_DL = pd.merge(original_split_test_DL, original_test, on="ID", how="left")
original_split_test_DL["train_test"] = "test"
original_split_test_DL["train_ID"] = 0
original_split_test_DL.head(2)

# %%
original_split_test_DL[(original_split_test_DL["Vote"] == 2) & \
                       (original_split_test_DL["DL_NDVI_SG_prob_point3"] == 2)].shape

# %%

# %%
pred_col = "DL_NDVI_SG_prob_point3"
error_count = (original_split_test_DL.Vote != original_split_test_DL[pred_col]).sum()
accuracy = (len(original_split_test_DL) - error_count) / len(original_split_test_DL)

A2P2_df = original_split_test_DL[original_split_test_DL.Vote==2]
A2P2 = (A2P2_df.Vote == A2P2_df[pred_col]).sum()
A2P1 = len(A2P2_df) - A2P2

A1P1_df = original_split_test_DL[original_split_test_DL.Vote==1]
A1P1 = (A1P1_df.Vote == A1P1_df[pred_col]).sum()
A1P2 = len(A1P1_df) - A1P1

if A2P2!=0 or A1P2 != 0:
    UA = round(A2P2 / (A1P2 + A2P2), 3)
else:
    UA = "NA"

original_split_test_DL_dict = {"acc" : round(accuracy, 3),
                                "UA": UA,
                                "PA": round(A2P2 / (A2P1 + A2P2), 3),
                                "err_count" : error_count,
                                "A2P2": A2P2,
                                "A2P1":A2P1,
                                "A1P1": A1P1,
                                "A1P2":A1P2, 
                                "kappa": kappa(A2P2, A1P2, A1P1, A2P1),
                                "a_test_set_df" : original_split_test_DL
                               }

# %%
ML_test_results.keys()

# %%
ML_test_results["test_results_SVM"]["train_ID0"] = original_split_test_SVM_dict
ML_test_results["test_results_RF"]["train_ID0"] = original_split_test_RF_dict
ML_test_results["test_results_KNN"]["train_ID0"] = original_split_test_KNN_dict
ML_test_results["test_results_DL"]["train_ID0"] = original_split_test_DL_dict

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
# meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
# print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
import pickle
from datetime import datetime
dir_ = "/Users/hn/Documents/01_research_data/NASA/Amin/"
filename = dir_ + "5_OverSampled_TestResults.sav"

export_ = {"ML_test_results": ML_test_results, 
           "areas" : meta,
           "source_code" : "HyperParam_and_test",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
ML_test_results["test_results_SVM"]["train_ID0"]["a_test_set_df"].head(2)

# %%
ML_test_results["test_results_RF"]["train_ID0"]["a_test_set_df"].head(2)

# %%
ML_test_results["test_results_DL"]["train_ID0"]["a_test_set_df"].head(2)

# %%
ML_test_results["test_results_KNN"]["train_ID0"]["a_test_set_df"].head(2)

# %%

# %%

# %%

# %%
all_preds = pd.read_csv(merged_pred_dir+"all_preds.csv")
all_preds.head(2)

# %%
SVM_preds = all_preds[["ID", "SVM_NDVI_SG_preds", "train_ID", "SR", "train_test"]]
SVM_preds = SVM_preds[SVM_preds.train_test == "train"]
SVM_preds.train_test.unique()

# %%
SVM_preds_TR1_SR3 = SVM_preds[(SVM_preds.train_ID==1) & (SVM_preds.SR==3)].copy()
SVM_preds_TR2_SR3 = SVM_preds[(SVM_preds.train_ID==2) & (SVM_preds.SR==3)].copy()
SVM_preds_TR5_SR3 = SVM_preds[(SVM_preds.train_ID==4) & (SVM_preds.SR==3)].copy()

SVM_preds_TR1_SR3.sort_values(by="ID", inplace=True)
SVM_preds_TR2_SR3.sort_values(by="ID", inplace=True)
SVM_preds_TR5_SR3.sort_values(by="ID", inplace=True)

SVM_preds_TR1_SR3.reset_index(drop=True, inplace=True)
SVM_preds_TR2_SR3.reset_index(drop=True, inplace=True)
SVM_preds_TR5_SR3.reset_index(drop=True, inplace=True)


# %%
SVM_preds_TR1_SR3.head(2)

# %%
SVM_preds_TR2_SR3.head(2)

# %%
sum(SVM_preds_TR1_SR3.ID == SVM_preds_TR2_SR3.ID)

# %%
sum(SVM_preds_TR1_SR3.SVM_NDVI_SG_preds == SVM_preds_TR2_SR3.SVM_NDVI_SG_preds)

# %%
SVM_preds_TR1_SR3.shape

# %%

# %%
SVM_preds = all_preds[["ID", "SVM_NDVI_SG_preds", "train_ID", "SR", "train_test"]]
SVM_preds = SVM_preds[SVM_preds.train_test == "test"]

# %%
SVM_preds_TR1_SR3 = SVM_preds[(SVM_preds.train_ID == 1) & (SVM_preds.SR == 3)].copy()
SVM_preds_TR1_SR8 = SVM_preds[(SVM_preds.train_ID == 1) & (SVM_preds.SR == 8)].copy()
SVM_preds_TR5_SR8 = SVM_preds[(SVM_preds.train_ID == 5) & (SVM_preds.SR == 8)].copy()
SVM_preds_TR5_SR3 = SVM_preds[(SVM_preds.train_ID == 5) & (SVM_preds.SR == 3)].copy()

# %%
SVM_preds_TR1_SR3.reset_index(drop=True, inplace=True)
SVM_preds_TR1_SR8.reset_index(drop=True, inplace=True)
SVM_preds_TR5_SR8.reset_index(drop=True, inplace=True)
SVM_preds_TR5_SR3.reset_index(drop=True, inplace=True)

# %%
SVM_preds_TR1_SR3.head(2)

# %%
SVM_preds_TR1_SR8.head(2)

# %%
SVM_preds_TR5_SR8.head(6)

# %%
sum(SVM_preds_TR5_SR3.SVM_NDVI_SG_preds == SVM_preds_TR5_SR8.SVM_NDVI_SG_preds)

# %%
SVM_preds_TR5_SR3.shape

# %%

# %%
kappa(55, 5, 568, 4)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# Hyper by test set
cuts_ = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_params = {}

for a_train_ID in sorted(DL_preds_test.train_ID.unique()):
    print (f"{a_train_ID = }")
    best_in_set = {# "train_ID": a_train_ID,
                   "cut":0, 
                   "ratio":0, 
                   "acc.":-1, 
                   "double_agreement":-10, 
                   "double_disagreement":-10, 
                   "err_count":100000}
    a_test_set = DL_preds_test[DL_preds_test.train_ID == a_train_ID].copy()
    a_test_set.sort_values(by="ID", inplace=True)
    
    for a_SR in sorted(a_test_set.SR.unique()):
        a_test_set_SR = a_test_set[a_test_set.SR == a_SR].copy()
        
        for a_cut in cuts_:
            colName = "NDVI_SG_DL_p" + str(int(a_cut*10))
            a_test_set_SR[colName] = -1
            a_test_set_SR.loc[a_test_set_SR.prob_single < a_cut, colName] = 2
            a_test_set_SR.loc[a_test_set_SR.prob_single >= a_cut, colName] = 1
            
            error_count = (a_test_set_SR.Vote != a_test_set_SR[colName]).sum()
            accuracy = (len(a_test_set_SR) - error_count) / len(a_test_set_SR)
            
            double_agreement_df = a_test_set_SR[a_test_set_SR.Vote==2]
            double_agreement = (double_agreement_df.Vote == double_agreement_df[colName]).sum()
            double_disagreement = len(double_agreement_df) - double_agreement
            
            if error_count < best_in_set["err_count"]:
                best_in_set["cut"] = a_cut
                best_in_set["ratio"] = a_SR
                best_in_set["acc."] = round(accuracy, 3)
                best_in_set["double_agreement"] = double_agreement
                best_in_set["double_disagreement"] = double_disagreement
                best_in_set["err_count"] = error_count
                
            elif error_count == best_in_set["err_count"]:
                if double_agreement > best_in_set["double_agreement"]:
                    best_in_set["cut"] = a_cut
                    best_in_set["ratio"] = a_SR
                    best_in_set["acc."] = round(accuracy, 3)
                    best_in_set["double_agreement"] = double_agreement
                    best_in_set["double_disagreement"] = double_disagreement
                    best_in_set["err_count"] = error_count
                    
    print (best_in_set)
    best_params["train_ID" + str(a_train_ID)] = best_in_set
    print ("=============================================================================")

# %%

# %%
