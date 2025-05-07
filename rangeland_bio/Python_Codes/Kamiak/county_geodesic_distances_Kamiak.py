import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

import pyproj
import geopandas
from geopy.distance import geodesic
from datetime import datetime

sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc

#####################################################################################
#####################################################################################

research_data_ = "/data/project/agaid/h.noorazar/"
rangeland_bio_base = research_data_ + "rangeland_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
bio_reOrganized = rangeland_bio_data + "reOrganized/"
common_data = research_data_ + "common_data/"

NDVI_weather_base = research_data_ + "NDVI_v_Weather/"
NDVI_weather_data_dir = NDVI_weather_base + "data/"

#####################################################################################

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

#####################################################################################
county_SF_name = common_data + "cb_2018_us_county_500k"
county_SF = geopandas.read_file(county_SF_name)
county_SF.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
county_SF.rename(
    columns={"statefp": "state_fips", "countyfp": "county_fips"}, inplace=True
)

county_SF["county_fips"] = county_SF["state_fips"] + county_SF["county_fips"]


county_SF = pd.merge(
    county_SF,
    state_fips[["EW_meridian", "state_fips", "state_full"]],
    how="left",
    on="state_fips",
)

print(county_SF.shape)
county_SF = county_SF[county_SF["EW_meridian"] == "W"]
# remove Alaska and Hawaii
county_SF = county_SF[~(county_SF["state_fips"].isin(["02", "15"]))]
county_SF["centroid"] = county_SF["geometry"].centroid
county_SF.reset_index(drop=True, inplace=True)
print(county_SF.shape)

############################################################################################
### Just pick the counties in common

NDVI_weather = pd.read_pickle(NDVI_weather_data_dir + "NDVI_weather.sav")
NDVI_weather = NDVI_weather["NDVI_weather_input"]


weather_counties = set(NDVI_weather["county_fips"].unique())
all_counties = set(county_SF["county_fips"].unique())
print(len(weather_counties))
print(len(all_counties))

common_counties = list(weather_counties.intersection(all_counties))
county_SF = county_SF[county_SF["county_fips"].isin(common_counties)].copy()
county_SF.set_index("county_fips", inplace=True)
# Create a new DataFrame for pairwise distances
county_geodesic_dist = pd.DataFrame(index=county_SF.index, columns=county_SF.index)
county_geodesic_dist = county_geodesic_dist.astype(int)
# distance of a county to itself is zero. But in queen, we made diagonals equal to 1.
# np.fill_diagonal(county_geodesic_dist.values, 1)


print("line 97")
print(county_geodesic_dist)
# Calculate distances
for i in county_SF.index:
    for j in county_SF.index:
        county_geodesic_dist.loc[i, j] = rc.calculate_geodesic_distance(
            county_SF.loc[i, "centroid"], county_SF.loc[j, "centroid"]
        )

print(county_geodesic_dist)

code_src = "county_geodesic_distances_Kamiak"
filename = bio_reOrganized + "county_geodesic_dist.sav"
export_ = {
    "county_geodesic_dist": county_geodesic_dist,
    "source_code": code_src,
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

county_geodesic_dist_rowSTD = county_geodesic_dist.div(
    county_geodesic_dist.sum(axis=1), axis=0
)
filename = bio_reOrganized + "county_geodesic_dist_rowSTD.sav"
export_ = {
    "county_geodesic_dist_rowSTD": county_geodesic_dist_rowSTD,
    "source_code": code_src,
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

#######################################################

county_geodesic_dist_rowSTD.to_csv(
    bio_reOrganized + "county_geodesic_dist_rowSTD.csv", index=True
)
county_geodesic_dist.to_csv(bio_reOrganized + "county_geodesic_dist.csv", index=True)

#######################################################
county_geodesic_dist = 1 / county_geodesic_dist
county_geodesic_dist_rowSTD = county_geodesic_dist.div(
    county_geodesic_dist.sum(axis=1), axis=0
)

filename = bio_reOrganized + "county_geodesic_dist_reciprocal.sav"
export_ = {
    "county_geodesic_dist_reciprocal": county_geodesic_dist,
    "source_code": code_src,
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
pickle.dump(export_, open(filename, "wb"))

filename = bio_reOrganized + "county_geodesic_dist_reciprocal_rowSTD.sav"
export_ = {
    "county_geodesic_dist_reciprocal_rowSTD": county_geodesic_dist_rowSTD,
    "source_code": code_src,
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
pickle.dump(export_, open(filename, "wb"))
#######################################################
county_geodesic_dist.to_csv(
    bio_reOrganized + "county_geodesic_dist_reciprocal.csv", index=True
)
county_geodesic_dist_rowSTD.to_csv(
    bio_reOrganized + "county_geodesic_dist_reciprocal_rowSTD.csv", index=True
)

#######################################################
