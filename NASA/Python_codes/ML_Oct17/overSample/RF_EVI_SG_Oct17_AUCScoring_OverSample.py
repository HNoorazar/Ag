# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import numpy as np
import pandas as pd

from datetime import date
import time

import random
from random import seed
from random import random

import os, os.path
import shutil

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow
import pickle
import h5py
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# %%
sys.path.append('/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/')
import NASA_core as nc
# import NASA_plot_core.py as rcp

# %%
from tslearn.metrics import dtw as dtw_metric

# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# %% [markdown]
# # Metadata

# %%
meta_dir = "/Users/hn/Documents/01_research_data/NASA/parameters/"
meta = pd.read_csv(meta_dir+"evaluation_set.csv")
meta_moreThan10Acr=meta[meta.ExctAcr>10]
print (meta.shape)
print (meta_moreThan10Acr.shape)
meta.head(2)

# %%
# print (len(meta.ID.unique()))
# meta_lessThan10Acr=meta[meta.ExctAcr<10]
# print (meta_lessThan10Acr.shape)

# %% [markdown]
# # EVI - SG - SR 0.3

# %%
VI_idx = "EVI"
smooth_type = "SG"

# %%
model_dir = "/Users/hn/Documents/01_research_data/NASA/ML_Models_Oct17/overSample/"
os.makedirs(model_dir, exist_ok=True)

overSamples_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/overSamples/"

# %%
f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample3.csv"
# train80_GT_wide:
EVI_SG_wide_overSample3 = pd.read_csv(overSamples_data_folder + f_name)
print ("train set size of sample ratio 3 is ", EVI_SG_wide_overSample3.shape)

x_train_df=EVI_SG_wide_overSample3.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=EVI_SG_wide_overSample3[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
(list(x_train_df.ID)==list(y_train_df.ID))

# %% [markdown]
# ### Train on EVI - SG - SR 0.3

# %%
# %%time
RF_1_default = RandomForestClassifier(n_estimators=100, 
                                      criterion='gini', max_depth=None, 
                                      min_samples_split=2, min_samples_leaf=1, 
                                      min_weight_fraction_leaf=0.0,
                                      max_features='sqrt', max_leaf_nodes=None, 
                                      min_impurity_decrease=0.0, 
                                      bootstrap=True, oob_score=False, n_jobs=None, 
                                      random_state=1, verbose=0, 
                                      warm_start=False, class_weight=None, 
                                      ccp_alpha=0.0, max_samples=None)

RF_1_default.fit(x_train_df.iloc[:, 1:], y_train_df.iloc[:, 1:].values.ravel())

# filename = model_dir + "regular_" + VI_idx + "_RF1_default" + "_Oct17_AUCScoring_Oversample_SR0.3.sav"
# pickle.dump(RF_1_default, open(filename, 'wb'))

ML_data_folder = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
x_test_df = pd.read_csv(ML_data_folder + "widen_test_TS/" + VI_idx + "_" + smooth_type + \
                        "_wide_test20_split_2Bconsistent_Oct17.csv")

y_test_df = x_test_df[["ID", "Vote"]].copy()
x_test_df.drop(columns=["Vote"], inplace=True)

RF_1_default_predictions = RF_1_default.predict(x_test_df.iloc[:, 1:])
RF_1_default_y_test_df = y_test_df.copy()
RF_1_default_y_test_df["prediction"]=list(RF_1_default_predictions)
RF_1_default_y_test_df.head(2)

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_1_default_y_test_df.index:
    curr_vote=list(RF_1_default_y_test_df[RF_1_default_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_1_default_y_test_df[\
                                            RF_1_default_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
RF_default_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                                               index=range(2))
RF_default_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_default_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_default_confus_tbl_test['Predict_Single']=0
RF_default_confus_tbl_test['Predict_Double']=0

RF_default_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_default_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_default_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_default_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_default_confus_tbl_test

# %%
# # %%time
# parameters = {'n_jobs':[6],
#               'criterion': ["gini", "entropy"], # log_loss 
#               'max_depth':[1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17],
#               'min_samples_split':[4],
#               'max_features': ["log2"],
#               'class_weight':[None],
#               'ccp_alpha':[0.0], 
#               'max_samples':[None]
#              } # , 
# RF_grid_1 = GridSearchCV(RandomForestClassifier(random_state=0), 
#                                      parameters, cv=5, verbose=1, scoring="roc_auc",
#                                      error_score='raise')

# RF_grid_1.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

# print (RF_grid_1.best_params_)
# print (RF_grid_1.best_score_)

# RF_grid_1_predictions = RF_grid_1.predict(x_test_df.iloc[:, 1:])
# RF_grid_1_y_test_df=y_test_df.copy()
# RF_grid_1_y_test_df["prediction"]=list(RF_grid_1_predictions)

# true_single_predicted_single=0
# true_single_predicted_double=0

# true_double_predicted_single=0
# true_double_predicted_double=0

# for index_ in RF_grid_1_y_test_df.index:
#     curr_vote=list(RF_grid_1_y_test_df[RF_grid_1_y_test_df.index==index_].Vote)[0]
#     curr_predict=list(RF_grid_1_y_test_df[RF_grid_1_y_test_df.index==index_].prediction)[0]
#     if curr_vote==curr_predict:
#         if curr_vote==1: 
#             true_single_predicted_single+=1
#         else:
#             true_double_predicted_double+=1
#     else:
#         if curr_vote==1:
#             true_single_predicted_double+=1
#         else:
#             true_double_predicted_single+=1
            
# RF_grid_1_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
#                                index=range(2))
# RF_grid_1_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
# RF_grid_1_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
# RF_grid_1_confus_tbl_test['Predict_Single']=0
# RF_grid_1_confus_tbl_test['Predict_Double']=0

# RF_grid_1_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
# RF_grid_1_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
# RF_grid_1_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
# RF_grid_1_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
# RF_grid_1_confus_tbl_test

# %% [markdown]
# ### More parameters

# %%
# %%time
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                                     parameters, cv=5, verbose=1, scoring="roc_auc",
                                     error_score='raise')

RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)

RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_grid_2_confus_tbl_test

# %%
del(regular_forest_grid_2_confus_tbl_test, regular_forest_grid_2, regular_forest_grid_2_y_test_df)

# %% [markdown]
# ## SG - EVI - SR 0.4

# %%
del(x_train_df, EVI_SG_wide_overSample3, f_name, x_test_df, RF_grid_2, RF_grid_2)

f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample4.csv"

EVI_SG_wide_overSample3 = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 4 is", EVI_SG_wide_overSample3.shape)

x_train_df=EVI_SG_wide_overSample3.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=EVI_SG_wide_overSample3[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
(list(x_train_df.ID)==list(y_train_df.ID))

# %%
# %%time
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                                     parameters, cv=5, verbose=1, scoring="roc_auc",
                                     error_score='raise')

RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)

##########
##########    Test
##########
RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - EVI - SR 0.5

# %%
# %%time

del(x_train_df, EVI_SG_wide_overSample3, f_name, x_test_df, RF_grid_2, RF_grid_2)

f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample5.csv"

EVI_SG_wide_overSample3 = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 5 is", EVI_SG_wide_overSample3.shape)

x_train_df=EVI_SG_wide_overSample3.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=EVI_SG_wide_overSample3[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                                     parameters, cv=5, verbose=1, scoring="roc_auc",
                                     error_score='raise')

RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)

##########
##########    Test
##########
RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - EVI - SR 0.6

# %%
# %%time

del(x_train_df, EVI_SG_wide_overSample3, f_name, x_test_df, RF_grid_2, RF_grid_2)

f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample6.csv"

EVI_SG_wide_overSample3 = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 6 is", EVI_SG_wide_overSample3.shape)

x_train_df=EVI_SG_wide_overSample3.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=EVI_SG_wide_overSample3[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                                     parameters, cv=5, verbose=1, scoring="roc_auc",
                                     error_score='raise')

RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)

##########
##########    Test
##########
RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - EVI - SR 0.7

# %%
# %%time

del(x_train_df, EVI_SG_wide_overSample3, f_name, x_test_df, RF_grid_2, RF_grid_2)

f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample7.csv"

EVI_SG_wide_overSample3 = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 7 is", EVI_SG_wide_overSample3.shape)

x_train_df=EVI_SG_wide_overSample3.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=EVI_SG_wide_overSample3[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                                     parameters, cv=5, verbose=1, scoring="roc_auc",
                                     error_score='raise')

RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)

##########
##########    Test
##########
RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_grid_2_confus_tbl_test

# %% [markdown]
# ## SG - EVI - SR 0.8

# %%
# %%time

del(x_train_df, EVI_SG_wide_overSample3, f_name, x_test_df, RF_grid_2, RF_grid_2)

f_name = VI_idx + "_" + smooth_type + "_wide_train80_split_2Bconsistent_Oct17_overSample8.csv"

EVI_SG_wide_overSample3 = pd.read_csv(overSamples_data_folder + f_name) # train80_GT_wide:
print ("train set size of sample ratio 8 is", EVI_SG_wide_overSample3.shape)

x_train_df=EVI_SG_wide_overSample3.copy()
x_train_df.drop(columns=["Vote"], inplace=True)
y_train_df=EVI_SG_wide_overSample3[["ID", "Vote"]]

#### Make sure rows of ```ground_truth_allBands``` and ```ground_truth_labels``` are in the same order
print ((list(x_train_df.ID)==list(y_train_df.ID)))
parameters = {'n_jobs':[6],
              'criterion': ["gini", "entropy"], # log_loss 
              'max_depth':[2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20],
              'min_samples_split':[2, 3, 4, 5],
              'max_features': ["sqrt", "log2", None],
              'class_weight':['balanced', 'balanced_subsample', None],
              'ccp_alpha':[0.0, 1, 2, 3], 
             # 'min_impurity_decreasefloat':[0, 1, 2], # roblem with sqrt stuff?
              'max_samples':[None, 1, 2, 3, 4, 5]
             }

RF_grid_2 = GridSearchCV(RandomForestClassifier(random_state=0), 
                                     parameters, cv=5, verbose=1, scoring="roc_auc",
                                     error_score='raise')

RF_grid_2.fit(x_train_df.iloc[:, 1:], y_train_df.Vote.values.ravel())

print (RF_grid_2.best_params_)
print (RF_grid_2.best_score_)

##########
##########    Test
##########
RF_grid_2_predictions = RF_grid_2.predict(x_test_df.iloc[:, 1:])
RF_grid_2_y_test_df=y_test_df.copy()
RF_grid_2_y_test_df["prediction"]=list(RF_grid_2_predictions)

print ()
print (RF_grid_2_y_test_df.head(2))
print ()

true_single_predicted_single=0
true_single_predicted_double=0

true_double_predicted_single=0
true_double_predicted_double=0

for index_ in RF_grid_2_y_test_df.index:
    curr_vote=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].Vote)[0]
    curr_predict=list(RF_grid_2_y_test_df[RF_grid_2_y_test_df.index==index_].prediction)[0]
    if curr_vote==curr_predict:
        if curr_vote==1: 
            true_single_predicted_single+=1
        else:
            true_double_predicted_double+=1
    else:
        if curr_vote==1:
            true_single_predicted_double+=1
        else:
            true_double_predicted_single+=1
            
RF_grid_2_confus_tbl_test = pd.DataFrame(columns=['None', 'Predict_Single', 'Predict_Double'], 
                               index=range(2))
RF_grid_2_confus_tbl_test.loc[0, 'None'] = 'Actual_Single'
RF_grid_2_confus_tbl_test.loc[1, 'None'] = 'Actual_Double'
RF_grid_2_confus_tbl_test['Predict_Single']=0
RF_grid_2_confus_tbl_test['Predict_Double']=0

RF_grid_2_confus_tbl_test.loc[0, "Predict_Single"]=true_single_predicted_single
RF_grid_2_confus_tbl_test.loc[0, "Predict_Double"]=true_single_predicted_double
RF_grid_2_confus_tbl_test.loc[1, "Predict_Single"]=true_double_predicted_single
RF_grid_2_confus_tbl_test.loc[1, "Predict_Double"]=true_double_predicted_double
RF_grid_2_confus_tbl_test

# %%
