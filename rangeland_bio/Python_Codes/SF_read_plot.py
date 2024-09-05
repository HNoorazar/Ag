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
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

import matplotlib
import matplotlib.pyplot as plt

# %%

# %%
fips_dir = "/Users/hn/Documents/01_research_data/RangeLand/Data/reOrganized/"
county_fips_dict = pd.read_pickle(fips_dir + "county_fips.sav")

county_fips = county_fips_dict["county_fips"]
full_2_abb = county_fips_dict["full_2_abb"]
abb_2_full_dict = county_fips_dict["abb_2_full_dict"]
abb_full_df = county_fips_dict["abb_full_df"]
filtered_counties_29States = county_fips_dict["filtered_counties_29States"]
SoI = county_fips_dict["SoI"]
state_fips = county_fips_dict["state_fips"]

state_fips = state_fips[state_fips.state != "VI"].copy()
state_fips.head(2)

# %%
county_fips.head(2)
county_fips_westMeridian = county_fips[county_fips["EW_meridian"] == "W"].copy()
county_fips_westMeridian.head(2)

# %%
state_fips.head(2)
state_fips_westMeridian = state_fips[state_fips["EW_meridian"] == "W"].copy()

# also drop alaska and hawaii
state_fips_westMeridian = state_fips_westMeridian[state_fips_westMeridian.state_full != "Alaska"].copy()
state_fips_westMeridian = state_fips_westMeridian[state_fips_westMeridian.state_full != "Hawaii"].copy()
state_fips_westMeridian.head(2)

# %%
import geopandas
SF_dir = "/Users/hn/Documents/01_research_data/shapefiles/"
US_counties_SF = geopandas.read_file(SF_dir + "cb_2018_us_county_500k")
US_counties_SF.head(2)

# %%
len(US_counties_SF["STATEFP"].unique())

# %%
US_counties_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
US_counties_SF.head(2)

# %%
US_counties_SF.rename(columns={"statefp": "state_fips", 
                               "countyfp": "county_fip",
                               "geoid" : "county_fips"}, inplace=True)

# %%
print (state_fips.shape)

# %%
US_counties_SF = pd.merge(US_counties_SF, state_fips[["state_fips", "state"]], how="left", on="state_fips")
US_counties_SF.head(2)

# %%

# %%
## subset to US main states

US_counties_SF = US_counties_SF[US_counties_SF["state_fips"].isin(list(state_fips["state_fips"].unique()))].copy()
len(US_counties_SF["state_fips"].unique())

# %%
state_fips_westMeridian.shape

# %%

# %%

# %%
WM_stateFips_list = list(state_fips_westMeridian["state_fips"].unique())
US_counties_SF_westMeridian = US_counties_SF[US_counties_SF["state_fips"].isin(WM_stateFips_list)].copy()
US_counties_SF_westMeridian.reset_index(drop=True, inplace=True)

# %%
print (f"{US_counties_SF.shape = }")
print (f"{US_counties_SF_westMeridian.shape = }")

# %%
US_counties_SF_westMeridian.head(2)

# %%
sorted(list(US_counties_SF_westMeridian["state_fips"].unique()))

# %%
sorted(state_fips_westMeridian["state_fips"].unique()) == \
sorted(US_counties_SF_westMeridian["state_fips"].unique())

# %%
f_name = SF_dir + "US_counties_SF_westMeridian"
US_counties_SF_westMeridian.to_file(filename=f_name, driver='ESRI Shapefile')

# %%
sorted(US_counties_SF_westMeridian["state_fips"].unique())

# %%

# %%
