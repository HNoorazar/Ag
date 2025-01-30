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
import os, sys

sys.path.append("/Users/hn/Documents/00_github/Ag/NASA/Python_codes/")
import NASA_core as nc

# %%
dir_base = "/Users/hn/Documents/01_research_data/NASA/"

ML_dir_ = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
raw_dir = dir_base + "VI_TS/data_for_train_individual_counties/"
out_dir = "/Users/hn/Documents/01_research_data/NASA/for_paper/"

# %%
eval_ = pd.read_csv('/Users/hn/Documents/01_research_data/NASA/parameters/evaluation_set.csv')
GT = pd.read_csv(ML_dir_ + 'groundTruth_labels_Oct17_2022.csv')

GT_IDs = list(GT.ID.unique())

# %%
print (f"{eval_.shape = }")
print (f"{GT.shape = }")

# %%
csv_files = os.listdir(raw_dir)
csv_files = [x for x in csv_files if x.endswith(".csv")]
csv_files

# %%
raw_ = pd.DataFrame()
for file_ in csv_files:
    if not("Monterey" in file_):
        df = pd.read_csv(raw_dir + file_)
        df = df[df.ID.isin(GT_IDs)]
        
        df = df[["ID", "NDVI", "system_start_time"]]
        df.dropna(inplace=True)
        
        # add human time and pick proper year
        df = nc.add_human_start_time_by_system_start_time(df)
        df["year"] = df.human_system_start_time.dt.year
        df = df[df["year"] == sorted(df["year"].unique())[1]]

        raw_ = pd.concat([raw_, df])
raw_.sort_values(by=['ID', 'system_start_time'], inplace=True)
raw_.reset_index(inplace=True, drop=True)

# %%
raw_.drop(columns=['year'], inplace=True)
raw_.head(2)

# %%
raw_.to_csv(out_dir + 'raw_GEE_NDVI_Zenodo.csv', index=False)

# %%
print (raw_.shape)
print (len(raw_["ID"].unique()))

# %%
(raw_[raw_["ID"] == GT_IDs[3000]]).shape

# %%
