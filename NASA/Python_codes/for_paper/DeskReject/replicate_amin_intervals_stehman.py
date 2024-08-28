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

stehman_df_table2 = pd.DataFrame(stehman_dict)
stehman_df_table2.head(5)

# %%
stehman_df_table2.tail(5)

# %%

# %%
variable_cols = list(stehman_dict.keys())[3:]
cov_variables = [["UA_class_B_yu", "UA_class_B_xu"], ["PA_class_B_yu", "PA_class_B_xu"]]

Stehman_Table3 = nc.mean_var_covar_table(df = stehman_df_table2, 
                                         strarum_col = "stratum", 
                                         variable_cols = variable_cols,
                                         cov_variables = cov_variables)

for a_col in list(Stehman_Table3.columns[2:]):
    Stehman_Table3[a_col] = Stehman_Table3[a_col].astype(float)

Stehman_Table3 = Stehman_Table3.round(3)

Stehman_Table3

# %%

# %%
stratum_areas = {"stratum" : [1, 2, 3, 4], 
                 "total_area" : [40000, 30000, 20000, 10000]}

stehman_stratum_areas = pd.DataFrame(stratum_areas)
stehman_stratum_areas

# %%
area_class_col = "area_class_A_yu"

strarum_area_df = stehman_stratum_areas.copy()
strarum_col = "stratum" 
strarum_area_col = "total_area"

# %%
nc.proportion_of_area_class(Stehman_Table3 = Stehman_Table3, 
                            area_class_col = "area_class_A_yu", 
                            strarum_area_df = stehman_stratum_areas, 
                            strarum_col = "stratum", 
                            strarum_area_col = "total_area")

# %%
nc.proportion_of_area_class(Stehman_Table3 = Stehman_Table3, 
                            area_class_col = "area_class_C_yu", 
                            strarum_area_df = stehman_stratum_areas, 
                            strarum_col = "stratum", 
                            strarum_area_col = "total_area")

# %%
nc.proportion_of_area_class(Stehman_Table3 = Stehman_Table3, 
                            area_class_col = "overall_ac_yu", 
                            strarum_area_df = stehman_stratum_areas, 
                            strarum_col = "stratum", 
                            strarum_area_col = "total_area")

# %%

# %%
