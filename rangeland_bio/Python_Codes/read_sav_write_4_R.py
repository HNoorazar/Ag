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

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import os, os.path, pickle, sys

from datetime import datetime

# sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
# import rangeland_core as rc
# import rangeland_plot_core as rcp

# import importlib
# importlib.reload(rc);

# %%
research_data_ = "/Users/hn/Documents/01_research_data/"

common_data = research_data_ + "common_data/"

rangeland_bio_base = research_data_ + "RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir_v11 = rangeland_bio_data + "Min_Data_v1.1/"

rangeland_base = research_data_ + "RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
bio_reOrganized_temp = rangeland_bio_data + "temp_reOrganized/"

bio_plots = rangeland_bio_base + "plots/vegAreaChange/"
os.makedirs(bio_plots, exist_ok=True)

# %%
filename = bio_reOrganized + "bpszone_ANPP_no2012.sav"
A = pd.read_pickle(filename)
print (A["Date"])

# %%
bpszone_ANPP_no2012 = A["bpszone_ANPP"]
bpszone_ANPP_no2012.to_csv(bio_reOrganized + 'bpszone_ANPP_no2012_for_R.csv', index=False)

# %%
bpszone_ANPP_no2012.head(2)

# %% [markdown]
# #### see if number of years are identical for all locations

# %%
A["source_code"]

# %%
# %%time
unique_number_of_years = {}

for a_fid in bpszone_ANPP_no2012.fid.unique():
    LL = str(len(bpszone_ANPP_no2012[bpszone_ANPP_no2012.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)
Albers_SF.head(2)

# %%
WA_SF = Albers_SF[Albers_SF["state_majority_area"] == "Washington"].copy()
print (WA_SF.shape)
WA_SF.head(2)

# %%
bpszone_ANPP_no2012.head(2)

# %%
WA_ANPP_no2012 = bpszone_ANPP_no2012[bpszone_ANPP_no2012["fid"].isin(list(WA_SF["fid"].unique()))].copy()
WA_ANPP_no2012.head(2)

# %%
# %%time
unique_number_of_years = {}

for a_fid in WA_ANPP_no2012.fid.unique():
    LL = str(len(WA_ANPP_no2012[WA_ANPP_no2012.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%
