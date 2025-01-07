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
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import pymannkendall as mk
from scipy.stats import variation
from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas
from geopy.distance import geodesic

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from datetime import datetime

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

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
county_fips_dict = pd.read_pickle(common_data + "county_fips.sav")

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
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)
Albers_SF.head(2)

# %%
Albers_SF["centroid"] = Albers_SF["geometry"].centroid
Albers_SF["centroid"] = Albers_SF["centroid"].astype(str)

# %%
Albers_SF.head(2)

# %%

# %%
f_name = rangeland_bio_data + 'Albers_BioRangeland_Min_Ehsan_withCentroids.shp.zip'
# Albers_SF.drop(columns=["centroid"], inplace=True) # it does not let to have geometry and point ()2 geometries!
Albers_SF.to_file(filename=f_name, driver='ESRI Shapefile')

# %%
