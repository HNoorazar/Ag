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

# %% [markdown]
# https://rspatial.org/analysis/6-local_regression.html

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

import geopandas
from pyproj import CRS, Transformer

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW, Gaussian, Poisson
import spglm


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rc("font", family="Palatino")

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

# %%
dpi_ = 300

# %%
research_data_ = "/Users/hn/Documents/01_research_data/"
rangeland_bio_base = research_data_ + "/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"

common_data = research_data_ + "common_data/"

# %%
rangeland_bio_data

# %%
# # %%time
                               
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
# SF_west = geopandas.read_file(f_name)

# %%

# %%
r_data_dir = "/Users/hn/Documents/01_research_data/RangeLand_bio/data_from_R/"

# %% [markdown]
# # California House Price Data

# %%
research_data_ = "/Users/hn/Documents/01_research_data/"
common_data = research_data_ + "common_data/"

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
state_fips[state_fips["state_full"] == "California"]

# %%

# %%
houses = pd.read_csv(r_data_dir + "california_houses.csv")
houses.head(2)

# %%
houses_gdf = geopandas.GeoDataFrame(houses, geometry=geopandas.points_from_xy(houses.longitude, houses.latitude))

# Set the coordinate reference system (CRS) if you know it
houses_gdf.crs = 'EPSG:4326' # 'EPSG:4326' # WGS84
houses_gdf.head(2)

# %%
tick_legend_FontSize = 5
params = {"font.family": "Palatino",
          "legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * 2,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * 1,
          "ytick.labelsize": tick_legend_FontSize * 1,
          "axes.titlepad": 5,
          "legend.handlelength": 2,
          "xtick.bottom": True,
          "ytick.left": True,
          "xtick.labelbottom": True,
          "ytick.labelleft": True,
          'axes.linewidth' : .05,
          "xtick.major.width" : .51,
          "ytick.major.width" : .51,
          "xtick.major.size" : 2,
          "ytick.major.size" : 2}
plt.rcParams.update(params)

fig, ax = plt.subplots(1, 1, figsize=(2, 3), sharex=True, sharey=True, dpi=dpi_)
houses_gdf["geometry"].plot(ax=ax, color='dodgerblue', markersize=0.051);
# houses_gdf.plot(column='value', ax=ax, legend=False);

# %%
SF_dir = "/Users/hn/Documents/01_research_data/shapefiles/"
US_counties_SF = geopandas.read_file(SF_dir + "cb_2018_us_county_500k")

US_counties_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

US_counties_SF.rename(columns={"statefp": "state_fips", 
                               "countyfp": "county_fip",
                               "geoid" : "county_fips"}, inplace=True)

US_counties_SF = US_counties_SF[["state_fips", "county_fip", "county_fips", "name", "geometry"]]
US_counties_SF.head(2)

# %%
CA_counties_SF = US_counties_SF[US_counties_SF["state_fips"] == '06'].copy()
print (CA_counties_SF.shape)
CA_counties_SF.head(2)

# %%

# %%
houses_gdf.head(2)

# %%
houses_gdf.crs

# %%
CA_counties_SF.crs

# %%
houses_gdf = houses_gdf.to_crs(CA_counties_SF.crs)
houses_gdf.crs

# %%

# %%
cnty = geopandas.sjoin(houses_gdf, CA_counties_SF, how="left", op="intersects");

# %%
cnty.head(3)

# %%
totpop = cnty[["name", "population"]].groupby(["name"]).sum().reset_index()
totpop.head(2)

# %%
hd = cnty.copy()

# %%
hd["suminc"] = hd["income"] * hd["households"]
hd.head(2)

# %%
csum = hd[["name", 'suminc', 'households']].groupby(["name"]).sum().reset_index()
print (csum.shape)
csum.head(2)

# %%
csum_R = pd.read_csv(r_data_dir + "csum.csv")
csum_R.rename(columns={"Group.1": "name"}, inplace=True)

print (csum_R.shape)
csum_R.head(2)

# %%
print ((csum["name"] == csum_R["name"]).sum())
print ((csum["suminc"] - csum_R["suminc"]).sum())

# %%
print (hd.shape)
hd.head(2)

# %%
hd_R = pd.read_csv(r_data_dir + "hd.csv")
hd_R.rename(columns={"NAME": "county_name"}, inplace=True)
hd_R.drop('LSAD_TRANS', axis=1, inplace=True)
print (hd_R.shape)
hd_R.head(2)

# %%
hd_hd_R = pd.concat([hd, hd_R["county_name"]], axis=1)
hd_hd_R.head(2)

# %%
np.where(hd_hd_R["name"] != hd_hd_R["county_name"])

# %%
hd.iloc[394]

# %%
hd_hd_R.iloc[394]

# %%
hd_R.iloc[394]

# %%
hd_noNACounty_R = pd.read_csv(r_data_dir + "hd_noNACounty.csv")
hd_noNACounty_R.rename(columns={"NAME": "county_name"}, inplace=True)
hd_noNACounty_R.drop('LSAD_TRANS', axis=1, inplace=True)
print (hd_noNACounty_R.shape)
hd_noNACounty_R.head(2)

# %%
hd_noNACounty = hd.copy()
hd_noNACounty.dropna(subset=['name'], inplace=True)

# %%
hd_noNACounty.shape

# %%
hd_noNACounty.head(2)

# %%
hd_noNACounty_R.head(2)

# %%
hd_noNACounty_R.head(2)

# %%
A = hd_noNACounty.copy()
A = A[["houseValue", "income", "houseAge", "rooms", "bedrooms", "latitude", "longitude", "name"]]

A_R = hd_noNACounty_R.copy()
A_R = A_R[["houseValue", "income", "houseAge", "rooms", "bedrooms", "latitude", "longitude", "county_name"]]


# %%
v = ["houseValue", "income", "houseAge", "rooms", "bedrooms", "latitude", "longitude"]
hd_noNACounty_merged = pd.merge(A, A_R, how="left", on=v)
hd_noNACounty_merged.head(2)

# %%
mismatch_locs = (np.where(hd_noNACounty_merged["name"] != hd_noNACounty_merged["county_name"])[0])
hd_noNACounty_merged.loc[mismatch_locs]

# %%

# %%

# %%

# %%

# %%
from pyproj import CRS

crs = CRS.from_epsg(32633)  # EPSG code for UTM zone 33N

# Set the CRS of the GeoDataFrame
gdf_planar.set_crs(crs, allow_override=True, inplace=True)
gdf_planar.head(2)

# %%
# # !pip3 install rpy2

# %%
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# Import the terra library in R
ro.r('library(terra)')

# %%
df1 = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
df2 = pd.DataFrame({'x': [2, 3, 4], 'y': [5, 6, 7]})

# Convert pandas dataframes to R dataframes
r_df1 = pandas2ri.py2rpy(df1)
r_df2 = pandas2ri.py2rpy(df2)

# %%

# %%

# %%
vect = t.vect(gdf_planar["geometry"], crs="EPSG:32632")

# %%

# %%
# Create a geometry column from the latitude and longitude
gdf_4326 = geopandas.GeoDataFrame(p, geometry=geopandas.points_from_xy(p.LONG, p.LAT))

# Set the coordinate reference system (CRS) if you know it
gdf_4326.crs = 'EPSG:4326' # 'EPSG:4326' # WGS84

gdf_4326.head(2)

# %%
gdf.crs

# %%
import terra as t

# Create a SpatVector from scratch
coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
geom = t.geom(coords, "polygon")
vect = t.vect(geom, crs="EPSG:32632")  # Use a planar CRS, e.g., UTM Zone 32N

print(vect)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
import pandas as pd
import numpy as np
from mgwr.gwr import GWR
# from mgwr.sel_bwch import Sel_BW
from mgwr.sel_bw import Sel_BW
from libpysal.weights import Queen
from sklearn.preprocessing import StandardScaler
from libpysal.weights import DistanceBand
# from libpysal.weights import Distance

# %%
X = gdf[['ALT']].values  # Predictor variable(s)
y = gdf['pan'].values  # Dependent variable
coords = gdf[['LONG', 'LAT']].values 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# w = Queen.from_dataframe(gdf, use_index=False)
# w = DistanceBand(coords, threshold=0.1)
# w = Distance(coords, threshold=0.1)
w = DistanceBand(coords, threshold=0.1, binary=True)
# w = w.transform('b')
# w = w.weights
selector = Sel_BW(coords, y, X_scaled).search(criterion='CV')
selector

# %%

# %%

# %%
import mgwr.gwr as mgwr

# Set your dependent and independent variables
y = gdf[y_var]
X = gdf[indp_vars]


# Create a GWR model object
gwr_model = mgwr.GWR(coords=gdf[['LONG', 'LAT']], y=y, X=X, kernel='gaussian')

# %%

# %%

# %%
# %%time
depen_var, indp_vars = "mean_lb_per_acr", temp_cols

m5 = spreg.OLS_Regimes(y = y_train.values, x = X_train_normal[indp_vars].values, 
                       regimes = X_train_normal["groupveg"].tolist(),
                       constant_regi="many", regime_err_sep=False,
                       name_y=depen_var, name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(),
                           "Std. Error": m5.std_err.flatten(),
                           "P-Value": [i[1] for i in m5.t_stat]}, index=m5.name_x)

Conifer_m   = [i for i in m5_results.index if "Conifer"   in i]
Grassland_m = [i for i in m5_results.index if "Grassland" in i]
Hardwood_m  = [i for i in m5_results.index if "Hardwood"  in i]
Shrubland_m = [i for i in m5_results.index if "Shrubland" in i]

veg_ = "Conifer" ## Subset results to Conifer
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Conifer = m5_results.loc[Conifer_m, :].rename(lambda i: i.replace(rep_, ""))
Conifer.columns = pd.MultiIndex.from_product([[veg_], Conifer.columns])

veg_ = "Grassland" ## Subset results to Grassland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Grassland = m5_results.loc[Grassland_m, :].rename(lambda i: i.replace(rep_, ""))
Grassland.columns = pd.MultiIndex.from_product([[veg_], Grassland.columns])

veg_ = "Hardwood" ## Subset results to Hardwood
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Hardwood = m5_results.loc[Hardwood_m, :].rename(lambda i: i.replace(rep_, ""))
Hardwood.columns = pd.MultiIndex.from_product([[veg_], Hardwood.columns])

veg_ = "Shrubland" ## Subset results to Shrubland
rep_ = [x for x in groupveg if veg_ in x][0] + "_"
Shrubland = m5_results.loc[Shrubland_m, :].rename(lambda i: i.replace(rep_, ""))
Shrubland.columns = pd.MultiIndex.from_product([[veg_], Shrubland.columns])

# Concat models
table_ = pd.concat([Conifer, Grassland, Hardwood, Shrubland], axis=1).round(5).transpose()
table_.rename(columns=lambda x: x.replace('avg_of_dailyAvgTemp_C_', 'temp_'), inplace=True)
table_

# %%

# %%

# %%

# %%
