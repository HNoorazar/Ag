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
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
min_dir = data_dir_base + "Min_Data/"
RPMS_dir = min_dir + "RPMS/"
bpszone_dir = min_dir + "bpszone/"

# %%
subsec_annual_RPMS_ANPP = pd.read_csv(RPMS_dir + "subsection_annual_productivity_rpms_SUM.csv")
subsec_annual_RPMS_ANPP.head(2)

# %%
county_yrly_rpms_ANPP = pd.read_csv(RPMS_dir + "county_annual_productivity_rpms_MEAN.csv")
county_yrly_rpms_ANPP.head(2)

# %%
bpszone_ANPP = pd.read_csv(bpszone_dir + "bpszone_annual_productivity_rpms_MEAN.csv")
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP.FID.unique())

# %%
