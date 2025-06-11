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

# %%
dir_ = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/ACF1/"
new_dir_ = dir_ + "new_ACF1/"

# %%
f_name_pref = "rolling_autocorrelations_ws7"
ACF = pd.read_pickle(dir_ + f_name_pref + ".sav")
ACF_new = pd.read_pickle(new_dir_ + f_name_pref + "_mean_lb_per_acr.sav")

# %%
print (ACF.keys())
print (ACF_new.keys())

# %%
ACF = ACF[f_name_pref]
ACF_new = ACF_new[f_name_pref+"_mean_lb_per_acr"]

# %%
ACF_new.head(2)

# %%
ACF.head(2)

# %%
(ACF_new["autocorr_lag1_ws7"] - ACF["autocorr_lag1_ws7"]).sum()

# %%
print ((ACF['years'] == ACF_new['years']).sum() == len(ACF_new))
print ((ACF['fid'] == ACF_new['fid']).sum() == len(ACF_new))

# %%

# %%
