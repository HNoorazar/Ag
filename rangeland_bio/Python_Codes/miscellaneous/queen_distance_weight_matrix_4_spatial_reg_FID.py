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
import rangeland_plot_core as rcp

import importlib
importlib.reload(rc)

# %% [markdown]
# <font color='red'>**GeoPandas distance:**</font>
# Uses the CRS of your GeoDataFrame to calculate distances.
# May not be ideal for long distances or when high precision is needed, especially if your CRS is not a true spherical model.
# Can be faster for calculations within a local area with a suitable CRS. (This is just Euclidean distance)
#
#
# <font color='red'>**Geopy geodesic:**</font>
# Always calculates geodesic distances (great circle distances) on a spherical Earth.
# Provides more accurate results for long-distance calculations between geographic coordinates.
# Might be slightly slower than gdf.geometry.distance for local calculations. 

# %%
dpi_=300

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

# %%
# ## Supriya's columns need to change. So, read, write.
# WGS84_4326_name = common_data + "WGS84_4326/Albers_BioRangeland_Centroid_WGS84_4326/" + \
#                    "Albers_BioRangeland_Centroid_WGS84_4326.shp"
# WGS84_4326 = geopandas.read_file(WGS84_4326_name)

# WGS84_4326.rename(columns={"lat": "supriya_long", "lon": "supriya_lat"}, inplace=True)

# Centroid_WGS84_4326 = pd.read_csv(common_data + "WGS84_4326/" + "Centroid_WGS84_4326.csv")
# Centroid_WGS84_4326.rename(columns={"lat": "supriya_long", "lon": "supriya_lat"}, inplace=True)


# Centroid_WGS84_4326 = Centroid_WGS84_4326[["fid", "supriya_long", "supriya_lat"]]
# Centroid_WGS84_4326.to_csv(common_data + "WGS84_4326/" + "centroid_WGS84_4326_correctColNames.csv", index=False)
# Centroid_WGS84_4326.head(2)

# %%
Centroid_WGS84_4326 = pd.read_csv(common_data + "WGS84_4326/" + "centroid_WGS84_4326_correctColNames.csv")
Centroid_WGS84_4326.head(2)

# %%
from shapely.geometry import Polygon
us_states = geopandas.read_file(common_data +'cb_2018_us_state_500k.zip')

us_states.rename(columns={"STUSPS": "state"}, inplace=True)
us_states = us_states[~us_states.state.isin(["PR", "VI", "AS", "GU", "MP"])]
us_states = pd.merge(us_states, state_fips[["EW_meridian", "state"]], how="left", on="state")


visframe = us_states.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

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

print (Albers_SF.shape)

# %%
Albers_SF["centroid"] = Albers_SF["geometry"].centroid
Albers_SF.head(2)

# %%
tick_legend_FontSize = 5
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * .71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .7,
          "ytick.labelsize": tick_legend_FontSize * .7,
          "axes.titlepad": 5,
          "legend.handlelength": 2,
          "xtick.bottom": False,
          "ytick.left": False,
          "xtick.labelbottom": False,
          "ytick.labelleft": False,
          'axes.linewidth' : .05}
plt.rcParams.update(params)

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 3), sharex=True, sharey=True, dpi=dpi_)
Albers_SF["centroid"].plot(ax=ax, color='dodgerblue', markersize=0.051);
Albers_SF.plot(column='value', ax=ax, legend=False);

# %%
Albers_SF.set_index('fid', inplace=True)

# %%
# %%time
import pyproj
source_crs = pyproj.CRS("EPSG:5070")
target_crs = pyproj.CRS("EPSG:4326") # WGS84 (latitude/longitude)
transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

def convert_5070Centroids_to_lat_long(row):
    lat, long = transformer.transform(row["centroid"].coords[0][0], row["centroid"].coords[0][1])
    return (lat, long)

Albers_SF[["lat", "long"]] = Albers_SF.apply(convert_5070Centroids_to_lat_long, axis=1, result_type='expand')
Albers_SF.head(2)

# %% [markdown]
# #### Check if centroids are in the "middle" of polygons

# %%
two_polys = Albers_SF.iloc[:3].copy()
two_polys

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 3), sharex=True, sharey=True, dpi=dpi_)

rcp.plot_SF(SF=visframe_mainLand_west[visframe_mainLand_west.state=="SD"], ax_=ax, col="EW_meridian")
two_polys.plot(column='value', ax=ax, legend=False);
two_polys["centroid"].plot(ax=ax, color='red', markersize=.05);

# %%
Centroid_WGS84_4326.head(2)

# %%
Albers_SF.head(2)

# %%
Albers_SF['lat_long_centroid'] = geopandas.points_from_xy(Albers_SF['long'], Albers_SF['lat'])
Albers_SF.head(2)

# %%
Albers_SF.head(2)

# %%
Albers_SF.crs

# %%
import libpysal as ps

# Create a simple GeoDataFrame
gdf = geopandas.GeoDataFrame({
    'id': [1, 2, 3],
    'geometry': [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    ]
})

# Create the binary matrix
w = ps.weights.contiguity.Queen.from_dataframe(two_polys)
binary_matrix = w.full()[0]

print(binary_matrix)

# %%
# two_polys.iloc[[1, 2]].plot();
two_polys.plot();

# %%
# %%time
# Assuming you have a GeoDataFrame named 'gdf' with your polygon data
w = ps.weights.contiguity.Queen.from_dataframe(Albers_SF)
fid_contiguity_Queen_neighbors = w.full()[0]

print(fid_contiguity_Queen_neighbors)

# %%

# %%
fid_contiguity_Queen_neighbors = pd.DataFrame(fid_contiguity_Queen_neighbors, 
                                              index=Albers_SF.index, columns=list(Albers_SF.index))
fid_contiguity_Queen_neighbors = fid_contiguity_Queen_neighbors.astype(int)
fid_contiguity_Queen_neighbors.head(2)

# %%
# %%time
filename = bio_reOrganized + "fid_contiguity_Queen_neighbors.sav"
export_ = {"fid_contiguity_Queen_neighbors": fid_contiguity_Queen_neighbors,
           "source_code": "queen_distance_weight_matrix_4_spatial_reg",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %%
# %%time
queen_weights_std =
fid_contiguity_Queen_neighbors.div(fid_contiguity_Queen_neighbors.sum(axis=1), axis=0)

filename = bio_reOrganized + "fid_contiguity_Queen_neighbors_rowSTD.sav"
export_ = {"fid_contiguity_Queen_neighbors_rowSTD": queen_weights_std,
           "source_code": "queen_distance_weight_matrix_4_spatial_reg",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %%
queen_weights_std.head(2)

# %%
# %%time
queen_weights_std.to_csv(bio_reOrganized+'fid_contiguity_Queen_neighbors_rowSTD.csv', index=False)

# %%
# # !pip3 install pyreadstat

# %%
# %%time
# What's the point of saving it this way???
import pyreadstat
queen_weights_std.rename(columns=lambda x: "var_" + str(x), inplace=True)
pyreadstat.write_sav(queen_weights_std, 
                     bio_reOrganized + 'fid_contiguity_Queen_neighbors_rowSTD_for_R.sav')

# %% [markdown]
# ## Washington Only 
# for sake of model development

# %%
fid_contiguity_Queen_neighbors.head(2)

# %%
WA_SF = Albers_SF[Albers_SF["state_majority_area"] == "Washington"].copy()
print (WA_SF.shape)
WA_SF.head(2)

# %%
WA_FIDs = list(WA_SF.index.unique())
WA_FIDs[:4]

# %%
WA_fid_Queen_neighbors = fid_contiguity_Queen_neighbors[
                                        fid_contiguity_Queen_neighbors.index.isin(WA_FIDs)].copy()
print (WA_fid_Queen_neighbors.shape)
WA_fid_Queen_neighbors.head(2)

# %%
WA_fid_Queen_neighbors = WA_fid_Queen_neighbors[WA_FIDs].copy()
WA_fid_Queen_neighbors.shape

# %% [markdown]
# ### form queen neighborhoods from scratch and compare it to above

# %%
# # %%time
# # Assuming you have a GeoDataFrame named 'gdf' with your polygon data
# w_WA = ps.weights.contiguity.Queen.from_dataframe(WA_SF)
# WA_fid_Queen_neighbors_scratch = w_WA.full()[0]

# WA_fid_Queen_neighbors_scratch = pd.DataFrame(WA_fid_Queen_neighbors_scratch, 
#                                               index=WA_SF.index, columns=list(WA_SF.index))
# WA_fid_Queen_neighbors_scratch = WA_fid_Queen_neighbors_scratch.astype(int)

# print (WA_fid_Queen_neighbors.equals(WA_fid_Queen_neighbors_scratch)) # It was True
# WA_fid_Queen_neighbors_scratch.head(2)

# %%
WA_fid_Queen_neighbors_std = WA_fid_Queen_neighbors.div(WA_fid_Queen_neighbors.sum(axis=1), axis=0)
WA_fid_Queen_neighbors_std.head(2)

# %%
WA_fid_Queen_neighbors.to_csv(bio_reOrganized + 'WA_fid_Queen_neighbors.csv', index=True)
WA_fid_Queen_neighbors_std.to_csv(bio_reOrganized + 'WA_fid_Queen_neighbors_rowSTD.csv', index=True)

# %%
print (type(WA_SF))
WA_SF.head(2)

# %%
WA_SF.head(2)

# %%
WA_SF.drop(columns=["lat_long_centroid", "centroid"], inplace=True)
f_name = bio_reOrganized + 'Albers_BioRangeland_Min_Ehsan_WA/Albers_BioRangeland_Min_Ehsan_WA.shp'
WA_SF.to_file(filename=f_name, driver='ESRI Shapefile')

# %%

# %%
