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
from shapely.geometry import Polygon
us_states = geopandas.read_file(common_data +'cb_2018_us_state_500k.zip')

us_states.rename(columns={"STUSPS": "state"}, inplace=True)
us_states = us_states[~us_states.state.isin(["PR", "VI", "AS", "GU", "MP"])]
us_states = pd.merge(us_states, state_fips[["EW_meridian", "state"]], how="left", on="state")


# visframe = us_states.to_crs({'init':'epsg:5070'})
visframe = us_states.to_crs({'init':'epsg:4269'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%
# %%time
SF_name = common_data + "cb_2018_us_county_500k"
county_SF = geopandas.read_file(SF_name)
county_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
county_SF.rename(columns={"statefp": "state_fips", 
                          "countyfp": "county_fips"}, inplace=True)

county_SF.drop(columns =["countyns", "affgeoid", "lsad", "aland", "awater"], inplace=True)
county_SF["county_fips"] = county_SF["state_fips"] + county_SF["county_fips"]
county_SF.head(2)

# %%
state_fips.head(2)

# %%
state_fips[state_fips["state_full"].isin(["Alaska", "Hawaii"])]

# %%
county_SF = pd.merge(county_SF, state_fips[["EW_meridian", "state_fips", "state_full"]], 
                     how="left", on="state_fips")

print (county_SF.shape)
county_SF = county_SF[county_SF["EW_meridian"] == "W"]
# remove Alaska and Hawaii
county_SF = county_SF[~(county_SF["state_fips"].isin(["02", "15"]))]
county_SF["centroid"] = county_SF["geometry"].centroid
county_SF.reset_index(drop=True, inplace=True)
print (county_SF.shape)

county_SF.head(2)

# %% [markdown]
# ### Shannon County, SD
#
# Shannon County, SD (```FIPS code = 46113```) was renamed Oglala Lakota County and assigned anew FIPS code (```46102```) effective in 2014.
#
#
# Old county fips for this county is ```46113``` which is what Min has in its dataset.
#
# How can I take care of this? If I get an old county shapefile, then, which year?

# %%
county_SF[county_SF["county_fips"] == "46102"]

# %%
county_SF[county_SF["county_fips"] == "48199"]

# %%

# %%
filename = "/Users/hn/Documents/01_research_data/NDVI_v_Weather/data/NDVI_weather.sav"
NDVI_weather = pd.read_pickle(filename)
print (NDVI_weather["source_code"])
NDVI_weather = NDVI_weather["NDVI_weather_input"]
NDVI_weather.head(2)

# %%
NDVI_missing_from_weights = []

for a_county in list(NDVI_weather["county_fips"].unique()):
    if not(a_county in list(county_SF['county_fips'].unique())):
        NDVI_missing_from_weights = NDVI_missing_from_weights + [a_county]

weights_missing_from_NDVI = []

for a_county in list(county_SF['county_fips'].unique()):
    if not(a_county in list(NDVI_weather["county_fips"].unique())):
        weights_missing_from_NDVI = weights_missing_from_NDVI + [a_county]
print (len(weights_missing_from_NDVI))
weights_missing_from_NDVI

# %%
NDVI_missing_from_weights

# %%
weather_counties = set(NDVI_weather["county_fips"].unique())
all_counties = set(county_SF['county_fips'].unique())

print (len(weather_counties))
print (len(all_counties))

# %% [markdown]
# ### Just pick the counties in common

# %%
common_counties = list(weather_counties.intersection(all_counties))

# 53055 is in WA, is island. no neighbor. Toss it. Or, turn NaN in its row std into 0.

# %%
print (county_SF.shape)
county_SF = county_SF[county_SF["county_fips"].isin(common_counties)].copy()
print (county_SF.shape)

# %%
"46113" in common_counties

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
six_CA_cnty = county_SF[county_SF["county_fips"].isin(['06001', '06013', '06085', '06099', '06077', '06075'])]

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 3), sharex=True, sharey=True, dpi=dpi_)
county_SF["centroid"].plot(ax=ax, color='dodgerblue', markersize=0.051);
county_SF.plot(ax=ax, legend=False);
six_CA_cnty.plot(ax=ax, legend=False, color="red");

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 3), sharex=True, sharey=True, dpi=dpi_)
six_CA_cnty.plot(ax=ax, legend=False, color="red");
six_CA_cnty["centroid"].plot(ax=ax, color='dodgerblue', markersize=0.051);

for idx, row in six_CA_cnty.iterrows():
    ax.text(row["centroid"].x, row["centroid"].y, row["county_fips"], fontsize=3, ha='center')

# %%

# %%
county_SF.set_index('county_fips', inplace=True)
county_SF.head(2)

# %%

# %%
county_SF.crs

# %%
visframe_mainLand_west.crs

# %%
# # %%time
### This did not work here. So, rather, above, I used 
### visframe = us_states.to_crs({'init':'epsg:4269'}) 
### rather than 5070 from what I did initially in 
### queen_distance_weight_matrix_4_spatial_reg_FID.ipynb

# import pyproj

# # 4269 too 5070 seems to be the correct way, but lat long come out infinity!!!
# source_crs = pyproj.CRS("EPSG:5070")

# # california and 3 counties come out wrong. 
# # Is this the reason? 4269 did not help neither
# target_crs = pyproj.CRS("EPSG:4269") # WGS84 (latitude/longitude) 
# transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

# def convert_5070Centroids_to_lat_long(row):
#     lat, long = transformer.transform(row["centroid"].coords[0][0], row["centroid"].coords[0][1])
#     return (lat, long)

# county_SF[["lat", "long"]] = county_SF.apply(convert_5070Centroids_to_lat_long, axis=1, result_type='expand')
# county_SF.head(2)

county_SF["lat"] = county_SF["centroid"].y
county_SF["long"] = county_SF["centroid"].x
county_SF.head(2)

# %% [markdown]
# #### Check if centroids are in the "middle" of polygons

# %%
three_polys = county_SF.iloc[:3].copy()
three_polys

# %%
state_fips[state_fips["state_fips"] == "06"]

# %%
fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=dpi_)

rcp.plot_SF(SF=visframe_mainLand_west[visframe_mainLand_west.state=="CA"], ax_=ax, col="EW_meridian")
three_polys.plot(ax=ax, legend=False);
three_polys["centroid"].plot(ax=ax, color='red', markersize=.05);

# %%
county_SF.head(2)

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
w = ps.weights.contiguity.Queen.from_dataframe(three_polys)
binary_matrix = w.full()[0]

print(binary_matrix)

# %%
# two_polys.iloc[[1, 2]].plot();
three_polys.plot();

# %%
county_SF.head(2)

# %%
# sort by index
county_SF = county_SF.sort_index()

# %%
# %%time
# Assuming you have a GeoDataFrame named 'gdf' with your polygon data
w = ps.weights.contiguity.Queen.from_dataframe(county_SF)
county_contiguity_Queen_neighbors = w.full()[0]

print(county_contiguity_Queen_neighbors)

# %%

# %%
county_contiguity_Queen_neighbors = pd.DataFrame(county_contiguity_Queen_neighbors, 
                                                  index=county_SF.index, columns=list(county_SF.index))
county_contiguity_Queen_neighbors = county_contiguity_Queen_neighbors.astype(int)
np.fill_diagonal(county_contiguity_Queen_neighbors.values, 1)
county_contiguity_Queen_neighbors.head(2)

# %%

# %%
# %%time
filename = bio_reOrganized + "county_contiguity_Queen_neighbors.sav"
export_ = {"county_contiguity_Queen_neighbors": county_contiguity_Queen_neighbors,
           "source_code": "queen_distance_weight_matrix_4_spatial_reg_county",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %%
# %%time
queen_weights_std = county_contiguity_Queen_neighbors.div(county_contiguity_Queen_neighbors.sum(axis=1), axis=0)

filename = bio_reOrganized + "county_contiguity_Queen_neighbors_rowSTD.sav"
export_ = {"county_contiguity_Queen_neighbors_rowSTD": queen_weights_std,
           "source_code": "queen_distance_weight_matrix_4_spatial_reg_county",
           "Author": "HN",
           "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

pickle.dump(export_, open(filename, "wb"))

# %%
queen_weights_std = county_contiguity_Queen_neighbors.div(county_contiguity_Queen_neighbors.sum(axis=1), axis=0)
queen_weights_std.loc["53055"]

# %%

# %%
queen_weights_std.loc["53055"] = 0

# %%
queen_weights_std.loc["53055"]

# %%
queen_weights_std.head(2)

# %%
# %%time
queen_weights_std.to_csv(bio_reOrganized+'county_contiguity_Queen_neighbors_rowSTD.csv', index=False)

# %%
# %%time
# What's the point of saving it this way???
import pyreadstat
queen_weights_std.rename(columns=lambda x: "var_" + str(x), inplace=True)
pyreadstat.write_sav(queen_weights_std, 
                     bio_reOrganized + 'county_contiguity_Queen_neighbors_rowSTD_for_R.sav')

# %% [markdown]
# ## Washington Only 
# for sake of model development

# %%
county_contiguity_Queen_neighbors.head(2)

# %%
county_SF.head(2)

# %%
WA_SF = county_SF[county_SF["state_full"] == "Washington"].copy()
print (WA_SF.shape)
WA_SF.head(2)

# %%
WA_counties = list(WA_SF.index.unique())
WA_counties[:4]

# %%
WA_county_Queen_neighbors = county_contiguity_Queen_neighbors[
                                        county_contiguity_Queen_neighbors.index.isin(WA_counties)].copy()
print (WA_county_Queen_neighbors.shape)
WA_county_Queen_neighbors.head(2)

# %%
WA_county_Queen_neighbors = WA_county_Queen_neighbors[WA_counties].copy()
WA_county_Queen_neighbors.shape

# %% [markdown]
# ### form queen neighborhoods from scratch and compare it to above

# %%
# # %%time
# # Assuming you have a GeoDataFrame named 'gdf' with your polygon data
# w_WA = ps.weights.contiguity.Queen.from_dataframe(WA_SF)
# WA_county_Queen_neighbors_scratch = w_WA.full()[0]

# WA_county_Queen_neighbors_scratch = pd.DataFrame(WA_county_Queen_neighbors_scratch, 
#                                               index=WA_SF.index, columns=list(WA_SF.index))
# WA_county_Queen_neighbors_scratch = WA_county_Queen_neighbors_scratch.astype(int)

# print (WA_county_Queen_neighbors.equals(WA_county_Queen_neighbors_scratch)) # It was True
# WA_county_Queen_neighbors_scratch.head(2)

# %%
WA_county_Queen_neighbors_std = WA_county_Queen_neighbors.div(WA_county_Queen_neighbors.sum(axis=1), axis=0)
WA_county_Queen_neighbors_std.head(2)

# %%
WA_county_Queen_neighbors.to_csv(bio_reOrganized + 'WA_county_Queen_neighbors.csv', index=True)
WA_county_Queen_neighbors_std.to_csv(bio_reOrganized + 'WA_county_Queen_neighbors_rowSTD.csv', index=True)

# %%
print (type(WA_SF))
WA_SF.head(2)

# %%
WA_SF.head(2)

# %%
WA_SF.drop(columns=["centroid"], inplace=True)
f_name = bio_reOrganized + 'county_BioRangeland_Min_Ehsan_WA/county_BioRangeland_Min_Ehsan_WA.shp'
WA_SF.to_file(filename=f_name, driver='ESRI Shapefile')
