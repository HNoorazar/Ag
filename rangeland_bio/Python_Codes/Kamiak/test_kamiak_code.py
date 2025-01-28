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

from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda
import geopandas, contextily
from scipy.stats import ttest_ind
import statistics
from sklearn.metrics import r2_score
import statsmodels.api as sm

from pyproj import CRS, Transformer

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW, Gaussian, Poisson
import spglm


import pandas as pd
import numpy as np
import os, os.path, pickle, sys

import pyproj
import geopandas
from geopy.distance import geodesic
from datetime import datetime

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rc("font", family="Palatino")

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

# %%
research_data_ = "/Users/hn/Documents/01_research_data/"
SF_dir = research_data_ + "shapefiles/"
rangeland_bio_base = research_data_ + "/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"

common_data = research_data_ + "common_data/"

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

# %%
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
Albers_SF.rename(
    columns={"minstatsid": "fid", "satae_max": "state_majority_area"}, inplace=True
)

Albers_SF = pd.merge(
    Albers_SF,
    state_fips[["EW_meridian", "state_full"]],
    how="left",
    left_on="state_majority_area",
    right_on="state_full",
)

Albers_SF.drop(columns=["state_full"], inplace=True)
Albers_SF.head(2)


print(Albers_SF.shape)
Albers_SF = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF.reset_index(drop=True, inplace=True)
print(Albers_SF.shape)


Albers_SF["centroid"] = Albers_SF["geometry"].centroid
Albers_SF.head(2)


source_crs = pyproj.CRS("EPSG:5070")
target_crs = pyproj.CRS("EPSG:4326")  # WGS84 (latitude/longitude)
transformer = pyproj.Transformer.from_crs(source_crs, target_crs)


# %%
def convert_5070Centroids_to_lat_long(row):
    lat, long = transformer.transform(
        row["centroid"].coords[0][0], row["centroid"].coords[0][1]
    )
    return (lat, long)


Albers_SF[["lat", "long"]] = Albers_SF.apply(convert_5070Centroids_to_lat_long, axis=1, result_type="expand")
Albers_SF.head(2)

Albers_SF["lat_long_centroid"] = geopandas.points_from_xy(
    Albers_SF["long"], Albers_SF["lat"]
)
Albers_SF.head(2)


# %%

# %%
Albers_SF.set_index("fid", inplace=True)
# Create a new DataFrame for pairwise distances
Albers_geodesic_dist = pd.DataFrame(index=Albers_SF.index, columns=Albers_SF.index)

Albers_geodesic_dist

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rcp


# %%
# Calculate distances
for i in Albers_SF.index:
    for j in Albers_SF.index:
        Albers_geodesic_dist.loc[i, j] = rc.calculate_geodesic_distance(
            Albers_SF.loc[i, "lat_long_centroid"], Albers_SF.loc[j, "lat_long_centroid"]
        )

# %%
Albers_geodesic_dist

# %%
