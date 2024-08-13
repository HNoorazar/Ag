# %% [markdown]
# # Author: Amin Norouzi Kandlati

# %% [markdown]
# # Accuracy assessment of double cropping paper
# This notebook is based on the methodologies described in the following paper:
#
# Stehman, Stephen V. "Estimating area and map accuracy for stratified
# random sampling when the strata are different from the map classes."
# International Journal of Remote Sensing 35.13 (2014): 4923-4939.
#
# - Why first do dictionary and then convert back to dataframe?

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import os, os.path, sys

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
                "Projects/Double_Crop_Mapping/")

path_to_data = "/Users/hn/Documents/01_research_data/NASA/Amin/"

# %%
file_path = path_to_data + "six_OverSam_TestRes_and_InclusionProb.sav"
data = pd.read_pickle(file_path)
field_info = data["field_info"][["ID", "ExctAcr"]]
test_set = data["six_OverSam_TestRes"]["test_results_DL"]["train_ID1"]["a_test_set_df"]

# drop the columns that are specific to a model
# test_set = test_set[["ID", "Vote", "CropTyp"]]
test_set.head(2)


# %%
prob = data["six_OverSam_TestRes"]["inclusion_prob"]
test_set = test_set.merge(prob, on="CropTyp", how="right")
test_set = test_set.merge(field_info, on="ID", how="inner")
test_set.head(3)

# %%
id_dict = defaultdict(list)
for idx, row in test_set.iterrows():
    id_dict[(row["Vote"], row["NDVI_SG_DL_p3"]), row["CropTyp"]].append(
        (row["ID"], row["inclusion_prob"], row["ExctAcr"]))


# %%
[list(id_dict.keys())[0]]

# %%
id_dict[list(id_dict.keys())[0]]

# %%

# %% [markdown]
# ## Overall accuracy

# %%
# n_star_h - len(A_yu_list)

# %%
A_N = data["six_OverSam_TestRes"]["inclusion_prob"]["denom_acr"].sum()
N = sum(data["six_OverSam_TestRes"]["inclusion_prob"]["denom"])


# %%
# Things below are from Eqs.
#

# %%
acr_data = data["six_OverSam_TestRes"]["inclusion_prob"]
master_dict = defaultdict(list)
# Numbers of strata 1
for strata in test_set["CropTyp"].unique():
    strata_subset = {key: value for key, value in id_dict.items() if key[1] == strata}
    A_n_star_h_list = [value[2] for key, values in strata_subset.items() for value in values]
    A_n_star_h = sum(A_n_star_h_list)

    idx = acr_data[acr_data["CropTyp"] == strata].index[0]
    # Now use .at to access the specific value
    A_N_star_h = acr_data.at[idx, "denom_acr"]
    N_star_h = acr_data.at[idx, "denom"]
    n_star_h = len(A_n_star_h_list)

    master_dict[(strata, "n_star_h")].append(n_star_h)
    master_dict[(strata, "A_n_star_h")].append(A_n_star_h)
    master_dict[(strata, "A_N_star_h")].append(A_N_star_h)
    master_dict[(strata, "N_star_h")].append(N_star_h)

    A_yu_list = [value[2] for key, values in strata_subset.items() for value in values
                 if key[0][0] == key[0][1]]
    A_yu = sum(A_yu_list)
    y_bar_h = A_yu / A_n_star_h

    # Sample variance (based on counts not area)
    y_bar_h_count = len(A_yu_list) / master_dict[(strata, "n_star_h")][0]
    yu_0_1 = np.append(np.ones(len(A_yu_list)), np.zeros(n_star_h - len(A_yu_list)))
    sy_h_2 = sum((yu_0_1 - y_bar_h_count) ** 2 / master_dict[(strata, "n_star_h")][0])

    master_dict[strata, "y_bar_h"].append(y_bar_h)
    master_dict[strata, "sy_h_2"].append(sy_h_2)

# %%
list(master_dict.keys())[:5]

# %%
list(master_dict.keys())[0]

# %%
master_dict[list(master_dict.keys())[0]]

# %%
master_df = nc.amin_UA_defaultdict_to_df(master_dict)
master_df = master_df.dropna()

# %% [markdown]
# **Eq. 25** of the paper:
#
# $$\hat V(\hat{\overline Y}) = (1/N^2) \sum_{h=1}^H (1 - \frac{n_h^*}{N_h^*}) \frac{s_{yh}^2}{n_h^*},$$
#
# where (**Eq. 26** of the paper:)
#
# $$s_{yh}^2 = \sum_{u \in h} (y_u - \overline y_h)^2 / (n_h^* - 1) $$

# %%

# %%
Y_bar_list = []
variance_list = []
variance_list_countbased = []
for strata in master_df["strata"].unique(): # 1
    A_N_star_h = master_df.loc[master_df["strata"] == strata, "A_N_star_h"].values[0]
    A_n_star_h = master_df.loc[master_df["strata"] == strata, "A_n_star_h"].values[0]
    sy_h_2 = master_df.loc[master_df["strata"] == strata, "sy_h_2"].values[0]
    y_bar_h = master_df.loc[master_df["strata"] == strata, "y_bar_h"].values[0]

    Y_bar_list.append(A_N_star_h * y_bar_h)

    variance_list.append(A_N_star_h**2 * (1 - A_n_star_h / A_N_star_h) * sy_h_2 / A_n_star_h)
    
Overall_acc = sum(Y_bar_list) / A_N
print("Overall Accuracy = ", Overall_acc)

# Variance of overall accuracy
variance_o = (1 / (A_N**2)) * sum(variance_list)

# variance_o_countbased = (1 / N**2) * sum(variance_list_countbased)
print("Area-based Variance of overall accuracy = ", variance_o)
# print("Count-based Variance of overall accuracy = ", variance_o_countbased)


# %%
master_df.head(3)

# %% [markdown]
# ### User and Producer Accuracy

# %%
start_b = "\033[1m"
end_b = "\033[0;0m"

c = 2  # We have ony two classes

for c in [1, 2]:
    ######################################################################
    #
    # USER ACCURACY AND SE
    #
    ######################################################################
    # Filter for instances that are mapped as c.
    denom_dict = {key: value for key, value in id_dict.items() if key[0][0] == c}
    # Filter for instances that are mapped as c and referenced as c, too (cc).
    numerator_dict = {key: value for key, value in id_dict.items()
                      if (key[0][0] == c and key[0][1] == c)}

    # List stratas for c () and cc (diagonals of )
    numerator_strata_list = [key[1] for key, _ in numerator_dict.items()]  # numerator
    denom_strata_list_ = [key[1] for key, _ in denom_dict.items()]  # denominator
    # numerator sum
    acr_data = data["six_OverSam_TestRes"]["inclusion_prob"]

    master_dict = defaultdict(list)
    # Numbers of strata 2 # why there are two of these? and the line below it
    master_dict = nc.number_of_strata(test_df = test_set, m_dict = master_dict, 
                                      IDs_dictionary = id_dict, area_df = acr_data)
    master_dict = nc.numer_sum_for_acc_intervals(numer_strata_list = numerator_strata_list, 
                                                 m_dict = master_dict,
                                                 numer_dict = numerator_dict)

    ###########  Calculate denominator sum  ###########
    # Why there are two of these?
    master_dict = nc.denom_sum_for_acc_intervals(denom_strata_list = denom_strata_list_, 
                                                 m_dict = master_dict,
                                                 denom_dictionary = denom_dict)
    master_dict = {key: master_dict[key] for key in sorted(master_dict.keys())}
    master_dict = defaultdict(list, master_dict)

    # put yu and xu of 0 - 1s in the master dict # 1
    xu_id = {key[0]: np.array(sorted(value)) for key, values in master_dict.items()
             for value in values if key[1] == "xu_IDs"}
    yu_id = {key[0]: np.array(sorted(value)) for key, values in master_dict.items()
             for value in values if key[1] == "yu_IDs"}

    for key, value in xu_id.items():
        if key not in yu_id:
            master_dict[(key, "yu_0_1")].append(np.zeros(len(xu_id[key])))
        else:
            yu_in_xu_0_1 = np.array((np.isin(xu_id[key], yu_id[key])).astype(int))
            master_dict[(key, "xu_0_1")].append(np.ones(len(yu_in_xu_0_1)))
            master_dict[(key, "yu_0_1")].append(yu_in_xu_0_1)

    master_dict = {key: master_dict[key] for key in sorted(master_dict.keys())}
    master_dict = defaultdict(list, master_dict)

    master_df = nc.amin_UA_defaultdict_to_df(master_dict) # Convert master_dict to a dataframe
    master_df = master_df.dropna()
    master_df = nc.s_xy_h_func(master_df) # Calculate s_xy_h
    # Calculate user accuracy
    Y_bar_list = [value[0] for key, value in master_dict.items() if key[1] == "Y_bar"]
    numerator_sum = sum(Y_bar_list)

    X_bar_list = [value[0] for key, value in master_dict.items() if key[1] == "X_bar"]
    denominator_sum = sum(X_bar_list)

    users_acc = numerator_sum / denominator_sum
    
    print(start_b + f"Class: {c}"  + end_b)
    nsr = numerator_sum.round(2)
    dsr = denominator_sum.round(2)
    print(f"Area-based user accuracy = {nsr} / {dsr} = {users_acc.round(2)}")

    # Calculate variance of user accuracy # why there are 2 of these?
    variance_sum_list = nc.user_acc_variance(master_df, user_accuracy=users_acc)
    variance_u = (1 / master_df["x_hat"].sum()) * sum(variance_sum_list)
    print("Area-based SE of user accuracy = ", np.sqrt(variance_u).round(2))

    ######################################################################
    #
    # PRODUCER ACCURACY AND SE
    #
    ######################################################################

    # Filter for instances that are mapped as c.
    denom_dict = {key: value for key, value in id_dict.items() if key[0][1] == c}
    
    # Filter for instances that are mapped as c and referenced as c, too (cc).
    numerator_dict = {key: value for key, value in id_dict.items() if (key[0][0] == c and key[0][1] == c)}

    # List stratas for c and cc Why there are two of everything?
    numerator_strata_list = [key[1] for key, _ in numerator_dict.items()] 
    denom_strata_list_ = [key[1] for key, _ in denom_dict.items()]
    acr_data = data["six_OverSam_TestRes"]["inclusion_prob"] # numerator sum
    
    master_dict = defaultdict(list)
    # Numbers of strata 3. why there are two of these? and the one below it
    master_dict = nc.number_of_strata(test_set, m_dict = master_dict, 
                                      IDs_dictionary = id_dict, 
                                      area_df = acr_data)
    master_dict = nc.numer_sum_for_acc_intervals(numer_strata_list = numerator_strata_list, 
                                                 m_dict=master_dict, 
                                                 numer_dict = numerator_dict)

    ###########  Calculate denominator sum  ###########
    # Why there are two of these?
    master_dict = nc.denom_sum_for_acc_intervals(denom_strata_list = denom_strata_list_, 
                                                 m_dict = master_dict,
                                                 denom_dictionary = denom_dict)
    master_dict = {key: master_dict[key] for key in sorted(master_dict.keys())}
    master_dict = defaultdict(list, master_dict)

    # put yu and xu of 0 - 1s in the master dict # 2
    xu_id = {key[0]: np.array(sorted(value)) for key, values in master_dict.items()
             for value in values if key[1] == "xu_IDs"}
    yu_id = {key[0]: np.array(sorted(value)) for key, values in master_dict.items()
             for value in values if key[1] == "yu_IDs"}

    for key, value in xu_id.items():
        if key not in yu_id:
            master_dict[(key, "yu_0_1")].append(np.zeros(len(xu_id[key])))
        else:
            yu_in_xu_0_1 = np.array((np.isin(xu_id[key], yu_id[key])).astype(int))
            master_dict[(key, "xu_0_1")].append(np.ones(len(yu_in_xu_0_1)))
            master_dict[(key, "yu_0_1")].append(yu_in_xu_0_1)

    master_dict = {key: master_dict[key] for key in sorted(master_dict.keys())}
    master_dict = defaultdict(list, master_dict)
    master_df = nc.amin_UA_defaultdict_to_df(master_dict) # Convert master_dict to dataframe
    master_df = master_df.dropna()
    master_df = nc.s_xy_h_func(master_df) # Calculate s_xy_h. Why there are two of these?

    # Calculate user accuracy (user or producer?)
    Y_bar_list = [value[0] for key, value in master_dict.items() if key[1] == "Y_bar"]
    X_bar_list = [value[0] for key, value in master_dict.items() if key[1] == "X_bar"]
    numerator_sum, denominator_sum = sum(Y_bar_list), sum(X_bar_list)
    users_acc = numerator_sum / denominator_sum

    nsr = numerator_sum.round(2)
    dsr = denominator_sum.round(2)
    print(f"Area-based user producer = {nsr}/{dsr} = {users_acc.round(2)}")

    # Calculate variance of user accuracy (user or producer?)
    variance_sum_list = nc.user_acc_variance(UAV_df=master_df, user_accuracy=users_acc)

    variance_u = (1 / master_df["x_hat"].sum()) * sum(variance_sum_list)
    print("Area-based SE of producer accuracy = ", np.sqrt(variance_u))
    print ()

# %%
master_df.head(5)

# %%

# %%
