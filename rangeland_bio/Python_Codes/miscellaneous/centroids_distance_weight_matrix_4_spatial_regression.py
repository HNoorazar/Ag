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

# %% [markdown]
# <font color='red'>**GeoPandas distance:**</font>
# Uses the CRS of your GeoDataFrame to calculate distances.
# May not be ideal for long distances or when high precision is needed, especially if your CRS is not a true spherical model.
# Can be faster for calculations within a local area with a suitable CRS.
#
#
# <font color='red'>**Geopy geodesic:**</font>
# Always calculates geodesic distances (great circle distances) on a spherical Earth.
# Provides more accurate results for long-distance calculations between geographic coordinates.
# Might be slightly slower than gdf.geometry.distance for local calculations. 

# %%

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
Albers_SF = pd.merge(Albers_SF, state_fips[["EW_meridian", "state_full"]], 
                     how="left", left_on="state_majority_area", right_on="state_full")

Albers_SF.drop(columns=["state_full"], inplace=True)
Albers_SF.head(2)

# %%
print (Albers_SF.shape)
Albers_SF = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF.reset_index(drop=True, inplace=True)
print (Albers_SF.shape)

# %%
Albers_SF["centroid"] = Albers_SF["geometry"].centroid
Albers_SF.head(2)

# %%
# from shapely.geometry import Point


# point1 = Point(48.8566, 2.3522)  # Paris
# point2 = Point(51.5074, 0.1278)  # London

# # Calculate distance in kilometers
# distance = geodesic(point1.coords[0], point2.coords[0]).km
# distance

# %%
# # Create a DataFrame with Shapely Point objects
# data = {'id': [1, 2], 'geometry': [Point(1, 2), Point(3, 4)]}
# df = geopandas.GeoDataFrame(data, crs="EPSG:4326")

# # Calculate distance between points
# distances = df.geometry.distance(df.geometry.iloc[0])

# print(distances)

# %%
# # Create a DataFrame with Shapely Point geometries
# df = pd.DataFrame({'id': [1, 2, 3],
#                    'geometry': [Point(1, 2), Point(3, 4), Point(5, 6)]
#                   })

# # Convert the DataFrame to a GeoDataFrame
# gdf = geopandas.GeoDataFrame(df, geometry='geometry')

# # Calculate the pairwise distance matrix
# distance_matrix = gdf.geometry.apply(lambda x: gdf.geometry.distance(x))

# print(distance_matrix)

# %%
# # Create a DataFrame with Shapely Point geometries
# df = pd.DataFrame({'id': [1, 2],
#                    'geometry': [Point(1, 2), Point(3, 4)]
#                   })

# # Convert the DataFrame to a GeoDataFrame
# gdf = geopandas.GeoDataFrame(df, geometry='geometry')

# # Calculate the pairwise distance matrix
# distance_matrix = gdf.geometry.apply(lambda x: gdf.geometry.distance(x))

# print(distance_matrix)

# %%
# geodesic(gdf["geometry"][0].coords[0], gdf["geometry"][1].coords[0]).km

# %%
# gdf.geometry.distance(gdf["geometry"][0])

# %%
# %%time
CRS_distance_matrix = Albers_SF["centroid"].apply(lambda x: Albers_SF["centroid"].distance(x))
CRS_distance_matrix.head(2)

# %%

# %%
# # Sample DataFrame
# df = pd.DataFrame({
#     'point1': [Point(10, 20), Point(30, 40)],
#     'point2': [Point(15, 25), Point(35, 45)]
# })

# # Calculate geodesic distance
# def calculate_distance(row):
#     return geodesic(
#         (row['point1'].y, row['point1'].x), 
#         (row['point2'].y, row['point2'].x)
#     ).km

# df['distance'] = df.apply(calculate_distance, axis=1)

# print(df)

# %%
import importlib
importlib.reload(rc)

# %%
# Create sample DataFrame
df = pd.DataFrame({
    'point': [Point(10, 20), Point(30, 40), Point(50, 60)]
})

# Define a function to calculate pairwise distance

# Create a new DataFrame for pairwise distances
distances = pd.DataFrame(index=df.index, columns=df.index)

# Calculate distances
for i in df.index:
    for j in df.index:
        distances.loc[i, j] = rc.calculate_geodesic_distance(df.loc[i, 'point'], df.loc[j, 'point'])

print(distances)

# %%
Albers_SF.set_index('fid', inplace=True)

# %%
# %%time 
# Create a new DataFrame for pairwise distances
Albers_geodesic_distances = pd.DataFrame(index=Albers_SF.index, columns=Albers_SF.index)

# Calculate distances
for i in Albers_SF.index:
    for j in Albers_SF.index:
        Albers_geodesic_distances.loc[i, j] = rc.calculate_geodesic_distance(Albers_SF.loc[i, 'centroid'], 
                                                                             Albers_SF.loc[j, 'centroid'])

print(Albers_geodesic_distances)

# %%
Albers_SF.head(2)

# %%
Albers_SF.crs

# %%
