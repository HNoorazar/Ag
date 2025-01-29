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


def convert_5070Centroids_to_lat_long(row):
    lat, long = transformer.transform(
        row["centroid"].coords[0][0], row["centroid"].coords[0][1]
    )
    return (lat, long)


Albers_SF[["lat", "long"]] = Albers_SF.apply(
    convert_5070Centroids_to_lat_long, axis=1, result_type="expand"
)
Albers_SF.head(2)


Albers_SF["lat_long_centroid"] = geopandas.points_from_xy(
    Albers_SF["long"], Albers_SF["lat"]
)
Albers_SF.head(2)

Albers_SF.set_index("fid", inplace=True)
# Create a new DataFrame for pairwise distances
Albers_geodesic_dist = pd.DataFrame(index=Albers_SF.index, columns=Albers_SF.index)

print("line 97")
print(Albers_geodesic_dist)
# Calculate distances
for i in Albers_SF.index:
    for j in Albers_SF.index:
        Albers_geodesic_dist.loc[i, j] = rc.calculate_geodesic_distance(
            Albers_SF.loc[i, "lat_long_centroid"], Albers_SF.loc[j, "lat_long_centroid"]
        )

print(Albers_geodesic_dist)

filename = bio_reOrganized + "Albers_geodesic_dist.sav"
export_ = {
    "Albers_geodesic_dist": Albers_geodesic_dist,
    "source_code": "geodesic_distances_Kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

Albers_geodesic_dist_rowSTD = Albers_geodesic_dist.div(
    Albers_geodesic_dist.sum(axis=1), axis=0
)
filename = bio_reOrganized + "Albers_geodesic_dist_rowSTD.sav"
export_ = {
    "Albers_geodesic_dist_rowSTD": Albers_geodesic_dist_rowSTD,
    "source_code": "geodesic_distances_Kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

#######################################################

Albers_geodesic_dist_rowSTD.to_csv(
    bio_reOrganized + "Albers_geodesic_dist_rowSTD.csv", index=True
)
Albers_geodesic_dist.to_csv(bio_reOrganized + "Albers_geodesic_dist.csv", index=True)

#######################################################
Albers_geodesic_dist = 1 / Albers_geodesic_dist
Albers_geodesic_dist_rowSTD = Albers_geodesic_dist.div(
    Albers_geodesic_dist.sum(axis=1), axis=0
)

filename = bio_reOrganized + "Albers_geodesic_dist_reciprocal.sav"
export_ = {
    "Albers_geodesic_dist_reciprocal": Albers_geodesic_dist,
    "source_code": "geodesic_distances_Kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
pickle.dump(export_, open(filename, "wb"))

filename = bio_reOrganized + "Albers_geodesic_dist_reciprocal_rowSTD.sav"
export_ = {
    "Albers_geodesic_dist_reciprocal_rowSTD": Albers_geodesic_dist_rowSTD,
    "source_code": "geodesic_distances_Kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
pickle.dump(export_, open(filename, "wb"))
#######################################################
Albers_geodesic_dist.to_csv(
    bio_reOrganized + "Albers_geodesic_dist_reciprocal.csv", index=True
)
Albers_geodesic_dist_rowSTD.to_csv(
    bio_reOrganized + "Albers_geodesic_dist_reciprocal_rowSTD.csv", index=True
)

#######################################################
