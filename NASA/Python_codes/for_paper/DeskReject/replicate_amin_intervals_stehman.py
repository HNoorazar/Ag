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
# ## Stehman Replication
#
# I'm replicating Stehman's paper. Check against Amin's code. etc. ec.
#
# Created this notebook on Aug. 26. 2024

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os, os.path, sys

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%

# %%
stratum = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
           4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

map_class = ["A", "A", "A", "A", "A", "A", "A", 
             "B", "B", "B", "A", "B", "B", "B", "B", "B", "B", "B", 
             "B", "B", "B", "B", "C", "C", "C", "C", "C", "C", "B", 
             "B", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D"]

ref_class = ["A", "A", "A", "A", "A", "C", "B", "A", "B",
             "C", "A", "B", "B", "B", "B", "B", "A",
             "A", "B", "B", "C", "C", "C", "C", "C",
             "D", "D", "B", "B", "A", "D", "D", "D",
             "D", "D", "D", "D", "C", "C", "B"]

area_class_A_yu = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

area_class_C_yu = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0] 

overall_ac_yu = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 
                 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

UA_class_B_yu = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 
                 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

UA_class_B_xu = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

PA_class_B_yu = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 
                 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

PA_class_B_xu = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 
                 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

P23_yu = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
          1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# %%
# area_perc_in_R2_C3_yu is area proportion in row 2 column 3. I suspect
# they are using confusion matrix terminology; here everythin is in columns!
# 
stehman_dict = {"stratum" : stratum, 
                "map_class" : map_class, 
                "ref_class" : ref_class, 
                "area_class_A_yu" : area_class_A_yu, 
                "area_class_C_yu" : area_class_C_yu, 
                "overall_ac_yu" : overall_ac_yu,
                "UA_class_B_yu" : UA_class_B_yu, 
                "UA_class_B_xu" : UA_class_B_xu, 
                "PA_class_B_yu" : PA_class_B_yu, 
                "PA_class_B_xu" : PA_class_B_xu,
                "area_perc_in_R2_C3_yu" : P23_yu}

stehman_df_table2 = pd.DataFrame(stehman_dict)
stehman_df_table2.head(5)

# %%
stehman_df_table2.tail(5)

# %%

# %%
variable_cols = list(stehman_dict.keys())[3:]
cov_variables = [["UA_class_B_yu", "UA_class_B_xu"], ["PA_class_B_yu", "PA_class_B_xu"]]

Stehman_Table3 = nc.mean_var_covar_table(df = stehman_df_table2, 
                                         stratum_col = "stratum", 
                                         variable_cols = variable_cols,
                                         cov_variables = cov_variables)

for a_col in list(Stehman_Table3.columns[2:]):
    Stehman_Table3[a_col] = Stehman_Table3[a_col].astype(float)

Stehman_Table3 = Stehman_Table3.round(3)

Stehman_Table3

# %%

# %%
stratum_areas = {"stratum" : [1, 2, 3, 4], 
                 "population_area" : [40000, 30000, 20000, 10000],
                 "sample_area" : [10, 10, 10, 10],
                 "population_count" : [40000, 30000, 20000, 10000],
                 "sample_count" : [10, 10, 10, 10]}

stehman_stratum_areas = pd.DataFrame(stratum_areas)
stehman_stratum_areas

# %%
area_class_col = "area_class_A_yu"

stratum_area_df = stehman_stratum_areas.copy()
stratum_col = "stratum" 
strarum_area_col = "population_area"

# %%
nc.AreaClassProportion_and_OA(Stehman_T3 = Stehman_Table3, 
                              yu_col = "area_class_A_yu", 
                              stratum_area_df = stehman_stratum_areas, 
                              stratum_col = "stratum", 
                              stratum_area_count_col = "population_area")

# %%
nc.AreaClassProportion_and_OA(Stehman_T3 = Stehman_Table3, 
                              yu_col = "area_class_C_yu", 
                              stratum_area_df = stehman_stratum_areas, 
                              stratum_col = "stratum", 
                              stratum_area_count_col = "population_area")

# %%
nc.AreaClassProportion_and_OA(Stehman_T3 = Stehman_Table3, 
                              yu_col = "overall_ac_yu",
                              stratum_area_df = stehman_stratum_areas, 
                              stratum_col = "stratum", 
                              stratum_area_count_col = "population_area")

# %%
nc.UA_PA_Rhat_Eq27(Stehman_T3 = Stehman_Table3, 
                   yu_col = "UA_class_B_yu", 
                   xu_col = "UA_class_B_xu",
                   stratum_area_df = stehman_stratum_areas, 
                   stratum_col = "stratum", 
                   stratum_area_count_col = "population_area")

# %%
nc.UA_PA_Rhat_Eq27(Stehman_T3 = Stehman_Table3, 
                   yu_col = "PA_class_B_yu", 
                   xu_col = "PA_class_B_xu",
                   stratum_area_df = stehman_stratum_areas, 
                   stratum_col = "stratum", 
                   stratum_area_count_col = "population_area")

# %%
# Page 13 of the PDF 4935 of the paper.
# Cell (i, j) of the error matrix, Pij (i = 2, j = 3):
#
nc.AreaClassProportion_and_OA(Stehman_T3 = Stehman_Table3, 
                              yu_col = "area_perc_in_R2_C3_yu", 
                              stratum_area_df = stehman_stratum_areas, 
                              stratum_col = "stratum", 
                              stratum_area_count_col = "population_area")

# %%

# %%
nc.SE_4_OA_and_PropArea(stehman_T2 = stehman_df_table2, 
                        stehman_T3 = Stehman_Table3, 
                        stratum_area_df = stratum_area_df, 
                        area_or_count="area", 
                        variable = "area_class_A_yu")

# %%
nc.SE_4_OA_and_PropArea(stehman_T2 = stehman_df_table2, 
                        stehman_T3 = Stehman_Table3, 
                        stratum_area_df = stratum_area_df, 
                        area_or_count="area", 
                        variable = "area_class_C_yu")

# %%
nc.SE_4_OA_and_PropArea(stehman_T2 = stehman_df_table2, 
                        stehman_T3 = Stehman_Table3, 
                        stratum_area_df = stratum_area_df, 
                        area_or_count="area", 
                        variable = "overall_ac_yu")

# %%
nc.SE_4_UA_PA(Stehman_T3 = Stehman_Table3, 
              stratum_area_df = stratum_area_df, 
              area_or_count = "area", 
              yu_col = "UA_class_B_yu", 
              xu_col = "UA_class_B_xu", 
              stratum_col = "stratum")

# %%

# %%

# %% [markdown]
# # Our Data

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

six_OverSam_TestRes_IncluProb = pd.read_pickle(file_path)
print (six_OverSam_TestRes_IncluProb.keys())

# %%

# %%
field_areas = six_OverSam_TestRes_IncluProb["field_info"][["ID", "ExctAcr"]]

test_set1_DL_res = six_OverSam_TestRes_IncluProb["six_OverSam_TestRes"]\
                                                      ["test_results_DL"]["train_ID1"]["a_test_set_df"]

field_areas.head(2)

# %%
test_set1_DL_res.head(2)

# %%
confusion_matrix(test_set1_DL_res["NDVI_SG_DL_p3"], test_set1_DL_res["Vote"])

# %%
inclusion_prob = six_OverSam_TestRes_IncluProb["six_OverSam_TestRes"]["inclusion_prob"]
inclusion_prob.head(2)

# %%
test_set1_DL_res = test_set1_DL_res.merge(inclusion_prob, on="CropTyp", how="right")
test_set1_DL_res = test_set1_DL_res.merge(field_areas, on="ID", how="inner")
test_set1_DL_res.head(2)

# %%

# %%
test_df = test_set1_DL_res.copy()
nc.area_count_refClass_yu_Eq14_Eq23(test_df = test_df, ref_class = 1)
nc.area_count_refClass_yu_Eq14_Eq23(test_df = test_df, ref_class = 2)
nc.overal_acc_yu_Eq12(test_df = test_df, ML_pred_col="NDVI_SG_DL_p3")
test_df.head(2)

# %%
ML_pred_col = "NDVI_SG_DL_p3"
nc.yu_4_UA_Eq18(test_df, map_class = 1, ML_pred_col = ML_pred_col)
nc.yu_4_UA_Eq18(test_df, map_class = 2, ML_pred_col = ML_pred_col)

test_df.head(2)

# %%
test_df[["ref_class", ML_pred_col, "UA_single_yu_NDVI_SG_DL_p3", "UA_double_yu_NDVI_SG_DL_p3"]]

# %%

# %%

# %%
nc.xu_4_UA_Eq19(test_df, map_class = 1, ML_pred_col = ML_pred_col)
nc.xu_4_UA_Eq19(test_df, map_class = 2, ML_pred_col = ML_pred_col)

# %%
test_df.head(2)

# %%

# %%
nc.yu_4_PA_Eq22(test_df, ref_class = 1, ML_pred_col = ML_pred_col)
nc.yu_4_PA_Eq22(test_df, ref_class = 2, ML_pred_col = ML_pred_col)

# %%
test_df.head(2)

# %%
test_df[["ref_class", ML_pred_col, "PA_single_yu_NDVI_SG_DL_p3", "PA_double_yu_NDVI_SG_DL_p3"]]

# %%
stehman_df_table2.head(2)

# %%
# Test our functions to create Table 2 of Stehman here.

# %%
stehman_T2 = stehman_df_table2[list(stehman_df_table2.columns[0:3])].copy()
stehman_T2.head(2)


# %% [markdown]
# # I adjust these functions slightly 
# to accomodate the 4 classes in Stehman

# %%
def area_count_refClass_yu_Eq14_Eq23(test_df, ref_class):
    """
    Arguments
    ---------
    ref_class : int
        here ref_class can be either 1 for single- or 2 for double-cropped
        according to truth/vote/reality

    Returns
    ---------
    Modifies test_df in place.
    Adds a column single_class_yu or double_class_yu to the dataframe
    which indicates whether a field is single-cropped or double-cropped.
    by vote/reference/truth.
    This is Eq. 14 of Stehman's paper
    """
    # assert ref_class in [1, 2]

    if "Vote" in test_df:
        test_df.rename(columns={"Vote": "ref_class"}, inplace=True)

 #   if ref_class == 1:
    new_variable = str(ref_class) + "_class_yu"
 #   else:
 #       new_variable = "double_class_yu"

    test_df[new_variable] = 0

    indices = test_df[test_df["ref_class"] == ref_class].index
    test_df.loc[indices, new_variable] = 1


# %%
area_count_refClass_yu_Eq14_Eq23(test_df=stehman_T2, ref_class="A")
area_count_refClass_yu_Eq14_Eq23(test_df=stehman_T2, ref_class="C")
stehman_T2.head(2)

# %%
stehman_T2["C_class_yu"].equals(stehman_df_table2["area_class_C_yu"])

# %%
stehman_df_table2.head(2)

# %%
nc.overal_acc_yu_Eq12(test_df = stehman_T2, ML_pred_col = "map_class")
stehman_T2.head(2)

# %%
stehman_T2["map_class_yu"].equals(stehman_df_table2["overall_ac_yu"])


# %%
def yu_4_UA_Eq18(test_df, map_class, ML_pred_col):
    """
    Arguments
    ---------
    map_class : int
        here map_class can be either 1 for single- or 2 for double-cropped
        according to prediction

    ML_pred_col : str
        Name of the ML model we want, since we have trained
        more than one model

    Returns
    ---------
    Modifies test_df in place.
    Adds a column UA_singleClass or UA_singleClass to the dataframe
    which indicates whether a field is classified correctly or not
    in a given class given by map_class

    This is Eq. 18 of Stehman's paper
    """
    # assert map_class in [1, 2]

    if "Vote" in test_df:
        test_df.rename(columns={"Vote": "ref_class"}, inplace=True)

#    if map_class == 1:
    new_variable = "UA_" + map_class + "_yu_" + ML_pred_col
#    else:
#        new_variable = "UA_double_yu_" + ML_pred_col

    test_df[new_variable] = 0

    correctly_classified = test_df[test_df["ref_class"] == test_df[ML_pred_col]]
    correc_class_targetClass = correctly_classified[
        correctly_classified[ML_pred_col] == map_class
    ]
    idx = correc_class_targetClass.index
    test_df[new_variable] = 0
    test_df.loc[idx, new_variable] = 1


# %%
stehman_T2.head(2)

# %%
stehman_df_table2.head(2)

# %%
yu_4_UA_Eq18(test_df = stehman_T2, map_class = "B", ML_pred_col = "map_class")
stehman_T2.head(2)

# %%
stehman_T2["UA_B_yu_map_class"].equals(stehman_df_table2["UA_class_B_yu"])


# %%
def xu_4_UA_Eq19(test_df, map_class, ML_pred_col):
    """
    This is the same as Eq 14 (area_count_refClass_yu_Eq14()).
    Just uses map_class instead of ref_class
    Arguments
    ---------
    map_class : int
        here map_class can be either 1 for single- or 2 for double-cropped
        according to ML prediction

    ML_pred_col : str
        Name of the column containing predictions of a given ML.

    Returns
    ---------
    Modifies test_df in place.
    Adds a column single__yu or double_class_yu to the dataframe
    which indicates whether a field is single-cropped or double-cropped.
    by vote/reference/truth.

    This is Eq. 19 of Stehman's paper
    """
    # assert map_class in [1, 2]

#    if map_class == 1:
    new_variable = map_class + "_Pred_xu_" + ML_pred_col
#    else:
#        new_variable = "doublePred_xu_" + ML_pred_col

    test_df[new_variable] = 0

    indices = test_df[test_df[ML_pred_col] == map_class].index
    test_df.loc[indices, new_variable] = 1


# %%
xu_4_UA_Eq19(test_df=stehman_T2, map_class="B", ML_pred_col="map_class")
stehman_T2.head(2)

# %%
stehman_T2["B_Pred_xu_map_class"].equals(stehman_df_table2["UA_class_B_xu"])


# %%
def yu_4_PA_Eq22(test_df, ref_class, ML_pred_col):
    
    # assert ref_class in [1, 2]

    if "Vote" in test_df:
        test_df.rename(columns={"Vote": "ref_class"}, inplace=True)

#    if ref_class == 1:
    new_variable = "PA_" + ref_class + "_yu_" + ML_pred_col
#    else:
#        new_variable = "PA_double_yu_" + ML_pred_col

    test_df[new_variable] = 0

    correctly_classified = test_df[test_df["ref_class"] == test_df[ML_pred_col]]
    correc_class_targetClass = correctly_classified[
        correctly_classified[ML_pred_col] == ref_class
    ]
    idx = correc_class_targetClass.index
    test_df[new_variable] = 0
    test_df.loc[idx, new_variable] = 1


# %%
yu_4_PA_Eq22(test_df=stehman_T2, ref_class="B", ML_pred_col="map_class")
stehman_T2.head(2)

# %%
stehman_T2.drop(columns=["PA_single_yu_map_class"], in)
