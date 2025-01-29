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

# %% [markdown]
# - Original link: https://rspatial.org/analysis/6-local_regression.html
# - New link: https://rspatial.org/analysis/7-spregression.html
#
# The original link does not do anything about weights or spatial-correlation. So, here we are, doing the New link!

# %%
import os, sys
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

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rc("font", family="Palatino")

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

# %%
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rcp

import importlib
importlib.reload(rc);

# %%

# %%
dpi_ = 300

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
# # %%time
                               
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
# SF_west = geopandas.read_file(f_name)

# %%

# %%
r_data_dir = "/Users/hn/Documents/01_research_data/RangeLand_bio/data_from_R/"

# %% [markdown]
#

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
state_fips[state_fips["state_full"] == "California"]

# %%
print(geopandas.__version__) # Macbook: 0.14.3
                             # Macbook: 1.0.1

# %%

# %%
US_counties_SF = geopandas.read_file(common_data + "cb_2018_us_county_500k")

US_counties_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

US_counties_SF.rename(columns={"statefp": "state_fips", 
                               "countyfp": "county_fip",
                               "geoid" : "county_fips"}, inplace=True)

US_counties_SF = US_counties_SF[["state_fips", "county_fip", "county_fips", "name", "geometry"]]
US_counties_SF.head(2)

# %% [markdown]
# # California House Price Data

# %%
CA_counties_SF = US_counties_SF[US_counties_SF["state_fips"] == '06'].copy()
CA_counties_SF.reset_index(inplace=True, drop=True)
print (CA_counties_SF.shape)
CA_counties_SF.head(2)

# %% [markdown]
# # Spatial lag model

# %%
hh_R = pd.read_csv(r_data_dir + "hh_cali_corr_4spatialLagModel.csv")
weights_R = pd.read_csv(r_data_dir + "weight_matrix_Cali_houses_corr.csv")

# %%
print (hh_R.shape)
hh_R.head(2)

# %%
weights_R.head(2)

# %%

# %%
print (weights_R.shape)
weights_R.columns = list(hh_R["County"])
weights_R["County"] = list(hh_R["County"])
weights_R.set_index('County', inplace=True)

weights_R.head(2)

# %%
weights_R.columns[np.where(weights_R.loc["Yuba"]!=0)[0]]

# %%
from spreg import ML_Lag, ML_Error
from libpysal.weights.util import full2W


pysal_weights = full2W(weights_R.values)
print (f"{type(pysal_weights) = }")
print ("------------------------------")

y = hh_R['houseValue'].values.reshape(-1, 1)  # Dependent variable
X = hh_R[['age', 'nBedrooms']].values
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding a constant term

model = ML_Lag(y, X_with_intercept, w=pysal_weights, epsilon=1.0e-30);

# Print model summary
print(model.summary)


# %% [markdown]
# # Spatial error model

# %%
error_model = ML_Error(y, X_with_intercept, w=pysal_weights, epsilon=1.0e-30)
# Print model summary
print(error_model.summary)

# %%

# %%
f_name = r_data_dir + 'hh_corr/hh_corr.shp'
hh_corr_SF = geopandas.read_file(f_name)
hh_corr_SF["error_model_residuals"] = error_model.u
# sorted(hh_corr_SF.columns)

# %%

# %%

# %%
col = "fWhite"
sum(hh_corr_SF_2[col] - hh_corr_SF[col])

# %%

# %%
# Define the number of groups
grps = 5

# Calculate quantiles
brks = np.quantile(hh_corr_SF['error_model_residuals'].dropna(), np.linspace(0, 1, grps))

# Create a color palette similar to 'RdBu'
palette = sns.color_palette("RdBu", grps)
cmap = ListedColormap(palette[::-1])  # Reverse the palette

# Assign colors based on quantiles
# Ensure to use the right number of labels
hh_corr_SF['color'] = pd.cut(hh_corr_SF['error_model_residuals'], 
                             bins=brks, 
                             labels=palette[:-1], include_lowest=True)

# Convert color assignments to categorical, ensuring proper dtype
hh_corr_SF['color'] = hh_corr_SF['color'].astype('category')


hh_corr_SF.plot(color=hh_corr_SF["color"]);

# %%

# %%
hh_corr_SF.loc[20]

# %%

# %%

# %%

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
import seaborn as sns

# Create a sample GeoDataFrame (replace this with your actual GeoDataFrame)
# Sample data
data = {
    'geometry': [None] * 5,  # Placeholder for geometries
    'residuals': [0.5, 1.5, -0.5, -1.5, 0.0]  # Example residuals
}
hh = gpd.GeoDataFrame(data)

# Define the number of groups
grps = 5

# Calculate quantiles
brks = np.quantile(hh['residuals'].dropna(), np.linspace(0, 1, grps))

# Create a color palette similar to 'RdBu'
palette = sns.color_palette("RdBu", grps)
cmap = ListedColormap(palette[::-1])  # Reverse the palette

# Assign colors based on quantiles
# Ensure to use the right number of labels
hh['color'] = pd.cut(hh['residuals'], bins=brks, labels=palette[:-1], include_lowest=True)

# Convert color assignments to categorical, ensuring proper dtype
hh['color'] = hh['color'].astype('category')

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Create dummy geometries for visualization if none are provided
# Here we simply create random points for demonstration; replace with actual geometries
hh['geometry'] = gpd.points_from_xy(np.random.rand(len(hh)), np.random.rand(len(hh)))

# If geometries exist, plot them
hh.boundary.plot(ax=ax, color='k')  # Plot boundaries if geometries exist
hh.plot(column='color', ax=ax, legend=True, edgecolor='k', cmap=cmap)

# Adjust legend
norm = Normalize(vmin=brks.min(), vmax=brks.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib
plt.colorbar(sm, ax=ax, orientation='vertical', label='Residuals')

plt.title('Residuals Plot with Quantile Breaks')
plt.show()


# %%

# %%

# %%

# %%

# %%
import numpy as np
from spreg import ML_Lag
from libpysal.weights.util import full2W

# Example data
np.random.seed(42)
y = np.random.rand(10).reshape(-1, 1)  # Dependent variable (10 observations)

# Create a NumPy-based weight matrix (10x10)
W_numpy = np.array([
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
])

# Row-standardize the weight matrix
W_standardized = W_numpy / W_numpy.sum(axis=1, keepdims=True)

# Convert to a PySAL weights object
pysal_weights = full2W(W_standardized)

# Create independent variables (adding an intercept term)
X = np.random.rand(10, 3)  # Three independent variables (10 observations)
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding a constant term

# Fit the spatial lag model
model = ML_Lag(y, X_with_intercept, pysal_weights)

# Print model summary
print(model.summary)


# %%
X

# %%

# %%
# Example NumPy weight matrix
W_numpy = np.array([
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
])

# Row-standardize the weight matrix
W_standardized = W_numpy / W_numpy.sum(axis=1, keepdims=True)

# Convert the NumPy array to a PySAL W object
pysal_weights = full2W(W_standardized)

# Print information about the created PySAL weights object
print(f"Number of observations: {pysal_weights.n}")


# %%

# %%

# %%

# %%
import libpysal as ps

# Example contiguity matrix
contiguity_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

# Convert to tuple 
contiguity_tuple = tuple(map(tuple, contiguity_matrix))

W = ps.weights.W(contiguity_tuple) 

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
