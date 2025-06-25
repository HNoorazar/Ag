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
bio_reOrganized = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/reOrganized/"
f = "bpszone_ANPP_no2012_detrended.sav"

# %%
ANPP = pd.read_pickle(bio_reOrganized + f)

# %%
ANPP.keys()

# %%
ANPP['source_code']

# %%
y_="diff"

if ("diff".lower() in y_.lower()):
    print ("hell")

# %%

# %%
