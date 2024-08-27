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
# I am replicating Stehman's paper. Check against Amin's code. etc. ec.
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
path_to_data = "/Users/hn/Documents/01_research_data/NASA/Amin/"

# %%
file_name = "six_OverSam_TestRes_and_InclusionProb.sav"
file_path = path_to_data + file_name

six_OverSam_TestRes_IncluProb = pd.read_pickle(file_path)
print (six_OverSam_TestRes_IncluProb.keys())

# %%
field_areas = six_OverSam_TestRes_IncluProb["field_info"][["ID", "ExctAcr"]]

test_set1_DL_res = six_OverSam_TestRes_IncluProb["six_OverSam_TestRes"]\
                                                      ["test_results_DL"]["train_ID1"]["a_test_set_df"]

# %%
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
                "map_class" : map_class, "ref_class" : ref_class, 
                "area_class_A_yu" : area_class_A_yu, "area_class_C_yu" : area_class_C_yu, 
                "overall_ac_yu" : overall_ac_yu,
                "UA_class_B_yu" : UA_class_B_yu, "UA_class_B_xu" : UA_class_B_xu, 
                "PA_class_B_yu" : PA_class_B_yu, "PA_class_B_xu" : PA_class_B_xu,
                "area_perc_in_R2_C3_yu" : P23_yu}

stehman_df = pd.DataFrame(stehman_dict)
stehman_df.head(5)

# %%
stehman_df.tail(5)

# %%
strata_1_areaClassA_yu = stehman_df[stehman_df.stratum == 1].copy()
strata_1_areaClassA_yu.area_class_A_yu.mean()
strata_1_areaClassA_yu.area_class_A_yu.std()**2


# %%
def mean_var_covar_table(df, strarum_col, variable_cols, cov_variables):
    """
    Arguments
    ---------
    df : DataFrame
        That looks like Table 2 of Stehman paper.

    strarum_col : str
        name of strarums
        
    variable_cols : list
        list of variables for which we want to compute mean and var
        
    cov_variables : list
        This is list of lists. 
        Each list has two column names in it: for covariance we need two variables.
        
    Returns
    ---------
    mean_var_covar_df : DataFrame
         table of means and variances and covariances
         similar to table 3 of Stehman paper
    """
    stratas = df[strarum_col].unique()
    cols = [strarum_col, "parameters"] + variable_cols
    meanVarCovar_df = pd.DataFrame(columns = cols, index=range(len(stratas)*3))

    # fill in the strarum column
    # since we want mean, var, and covar, we put 3 in line below
    meanVarCovar_df[strarum_col] = np.repeat(stratas, 3)
    
    param_list = ['mean', 'var', 'covar']
    meanVarCovar_df["parameters"] = np.stack((param_list) * len(stratas), axis=0)
    
    # populate the table:
    mean_df = df.groupby(by=strarum_col)[variable_cols].mean().reset_index()
    replacement = mean_df.loc[:, variable_cols].values
    meanVarCovar_df.loc[meanVarCovar_df["parameters"] == "mean", variable_cols] = replacement
    
    #
    var_df = df.groupby(by=strarum_col)[variable_cols].std().reset_index()
    var_df[variable_cols] = var_df[variable_cols]**2
    replacement = var_df.loc[:, variable_cols].values
    meanVarCovar_df.loc[meanVarCovar_df["parameters"] == "var", variable_cols] = replacement
    
    for curr_cov_var in cov_variables:
        cov_df = df.groupby(by=strarum_col)[[curr_cov_var[0], curr_cov_var[1]]].cov().reset_index()
        # we need one of the off-diagonal entries
        # so, we drop first column and every other row

        # cov_df = cov_df.loc[cov_df.index[::2,]].copy() # or the following line the same thing

        cov_df = cov_df[cov_df["level_1"] == curr_cov_var[0]].copy()
        # copy values of off diagonal into diagonal as well
        # just for simplifying the process of populating of 
        # main dataframe
        cov_df[curr_cov_var[0]] = cov_df[curr_cov_var[1]]


        replacement = cov_df[curr_cov_var].values
        meanVarCovar_df.loc[meanVarCovar_df["parameters"] == "covar", curr_cov_var] = replacement
    
    return meanVarCovar_df

# %%

# %%
variable_cols = list(stehman_dict.keys())[3:]
cov_variables = [["UA_class_B_yu", "UA_class_B_xu"], ["PA_class_B_yu", "PA_class_B_xu"]]

mean_var_covar_table(df=stehman_df, strarum_col="stratum", 
                     variable_cols=variable_cols,
                     cov_variables = cov_variables)

# %%
df = stehman_df.copy()
strarum_col = "stratum"
variable_cols = list(stehman_dict.keys())[3:]

##### function body
stratas = df[strarum_col].unique()
cols = [strarum_col, "parameters"] + variable_cols
meanVarCovar_df = pd.DataFrame(columns = cols, index=range(len(stratas)*3))

# fill in the strarum column
# since we want mean, var, and covar, we put 3 in line below
meanVarCovar_df[strarum_col] = np.repeat(stratas, 3)
param_list = ['mean', 'var', 'covar']
meanVarCovar_df["parameters"] = np.stack((param_list) * len(stratas), axis=0)
meanVarCovar_df

# %%
mean_df = df.groupby(by=strarum_col)[variable_cols].mean().reset_index()
replacement = mean_df.loc[:, variable_cols].values
meanVarCovar_df.loc[meanVarCovar_df["parameters"] == "mean", variable_cols] = replacement

#
var_df = df.groupby(by=strarum_col)[variable_cols].std().reset_index()
var_df[variable_cols] = var_df[variable_cols]**2
replacement = var_df.loc[:, variable_cols].values
meanVarCovar_df.loc[meanVarCovar_df["parameters"] == "var", variable_cols] = replacement

# %%
meanVarCovar_df

# %%
df_stratum1 = df[df.stratum == 4].copy()
np.cov(df_stratum1["PA_class_B_yu"], df_stratum1["PA_class_B_xu"])

# %%

# %%
cov_variables = [["UA_class_B_yu", "UA_class_B_xu"], ["PA_class_B_yu", "PA_class_B_xu"]]
curr_cov_var = cov_variables[1]
curr_cov_var

# %%

# %%
meanVarCovar_df

# %%
cov_df = cov_df[cov_df["level_1"] == curr_cov_var[0]].copy()
print (cov_df)
# # copy values of off diagonal into diagonal as well
# just for simplifying the process of populating of 
# main dataframe
cov_df[curr_cov_var[0]] = cov_df[curr_cov_var[1]]


replacement = cov_df[curr_cov_var].values
meanVarCovar_df.loc[meanVarCovar_df["parameters"] == "covar", curr_cov_var] = replacement
