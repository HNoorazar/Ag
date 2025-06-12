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
# There is another script called ```autocorr_moving_window_analysis_archived.ipynb```.
#
# This is a copy of that with modifications. That notebook was just fine. Here I am editting it so that we have more stuff from detrended ANPP in it. AND, variance of ACF1's are removed here. We may need them again.
#
# **June 12, 2025**

# %%
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os, os.path, pickle, sys
import pymannkendall as mk

import seaborn as sns
from scipy import stats
import scipy.stats as scipy_stats
from statsmodels.tsa.stattools import acf
import geopandas

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rpc

# %%
dpi_, map_dpi_=300, 900
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

rangeland_bio_base = research_db + "/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

# %%
bio_plots = rangeland_bio_base + "plots/"
ACF_plot_base = bio_plots + "ACF1/"
os.makedirs(ACF_plot_base, exist_ok=True)

# %%
ACF_data = rangeland_bio_data + "ACF1/"

# %%

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

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
gdf = geopandas.read_file(common_data +'cb_2018_us_state_500k.zip')
# gdf = geopandas.read_file(common_data +'cb_2018_us_state_500k')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%
# %%time
## bad 2012
# f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman_no2012.shp.zip'
SF_west = geopandas.read_file(f_name)
SF_west["centroid"] = SF_west["geometry"].centroid
SF_west.head(2)

# %%
print (ANPP.year.min())
print (ANPP.year.max())

# %% [markdown]
# ## Read all rolling window ACFs

# %%
ys = ["anpp", "anpp_detrendLinReg", "anpp_detrendDiff", "anpp_detrendSens"]

# %%
ACF_dict = {}
for window_size in np.arange(5, 11):
    for y_ in ys:
        key_ = f"rolling_autocorrelations_ws{window_size}_{y_}"
        f_name = ACF_data + key_ + ".sav"
        ACF_df = pd.read_pickle(f_name)
        ACF_df = ACF_df[key_]
        key_ = f"ACF1_ws{window_size}_{y_}"
        ACF_dict[key_] = ACF_df

# %% [markdown]
# ### Find trends of ACF1 time-series via MK again!

# %%
(list(ACF_dict.keys()))

# %%
ACF_dict["ACF1_ws5_anpp"].head(2)

# %%
ACF_dict["ACF1_ws5_anpp_detrendLinReg"].head(2)

# %%
import re

# %%
import importlib;
importlib.reload(rc);
importlib.reload(rpc);

# %%
# %%time

ACF_trends_MK_dict = {}

for a_key in ACF_dict.keys():
    ws = re.search(r'ws(\d+)', a_key).group(1)
    curr_df = ACF_dict[a_key]
    curr_col = f'autocorr_lag1_ws{ws}'
    ACF_trends_MK_dict[a_key] = rc.compute_mk_by_fid(df=curr_df, groupby_='fid', value_col=curr_col)
    del(a_key)

# %%
ACF_trends_MK_dict[list(ACF_dict.keys())[0]].head(3)

# %%
for key_ in ACF_trends_MK_dict.keys():
    ACF_trends_MK_dict[key_].rename(columns={"trend": f"trend_{key_}",
                                             "p_value": f"p_value_{key_}",
                                             "slope": f"slope_{key_}"}, 
                                    inplace=True)
del(key_)

# %%
ACF_trends_MK_dict[list(ACF_dict.keys())[0]].head(3)

# %%
from functools import reduce

# Convert dict values to a list of DataFrames
df_list = list(ACF_trends_MK_dict.values())

# Perform left merges iteratively
ACF_trends_MK_df = reduce(lambda left, right: pd.merge(left, right, on='fid', how='left'), df_list)

ACF_trends_MK_df.head(2)

# %%
# %%
filename = bio_reOrganized + "ACFs_rollingWindow_trends.sav"

export_ = {
    "ACF_trends_MK_df": ACF_trends_MK_df,
    "source_code": "autocorr_moving_window_compute_trend",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, 'wb'))

# %%
SF_west = pd.merge(SF_west, ACF_trends_MK_df, how="left", on="fid")

# %%
# %%time
f_name = rangeland_bio_data + 'SF_west_movingACF1s.shp.zip'

SF_west_2write = SF_west.copy()
SF_west_2write["centroid"] = SF_west_2write["centroid"].astype(str)

# SF_west_2write.drop(columns=["centroid"], inplace=True) # it does not like 2 geometries!
SF_west_2write.to_file(filename=f_name, driver='ESRI Shapefile')
del(SF_west_2write)

# %%
