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


# import cv2
from PIL import Image
import tifffile as tiff
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import os, os.path, pickle, sys

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from datetime import datetime


sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rcp

import importlib
importlib.reload(rc);

# %% [markdown]
# Google AI Choosing a Library:
#
# - **tifffile**: The best option for most cases.
# - **Pillow (PIL)**: Good for simple TIFF images.
# - **OpenCV**: If you are already using OpenCV for image processing tasks.
# - **Rasterio**: If you are working with geospatial data.

# %%
research_data_ = "/Users/hn/Documents/01_research_data/"
NASA_dir = research_data_ + "NASA/"
GCI_dir = NASA_dir + "GCI13_data/"

common_data = research_data_ + "common_data/"

# %%
# # Read the TIFF file
# tiff_image = tiff.imread(GCI_dir + "Cropping_Intensity_30m_2016_2018_N50W120.tif")

# # Now you can work with the image data, which is typically a NumPy array
# print(tiff_image.shape)

# cv2_image = cv2.imread(GCI_dir + "Cropping_Intensity_30m_2016_2018_N50W120.tif")

# with rasterio.open(GCI_dir + "Cropping_Intensity_30m_2016_2018_N50W120.tif") as dataset:
#     rasterio_image_array = dataset.read()

## PIL_image = Image.open(GCI_dir + "Cropping_Intensity_30m_2016_2018_N50W120.tif")
## PIL_image_array = np.array(PIL_image)

# with rasterio.open(GCI_dir + "Cropping_Intensity_30m_2016_2018_N50W120.tif") as src:
#     # Read the data
#     data = src.read(1)  # Read the first band
#     show(data, transform=src.transform, cmap='terrain') # Use a colormap
#     plt.show()

# with rasterio.open(GCI_dir + "Cropping_Intensity_30m_2016_2018_N50W130.tif") as src:
#     # Read the data
#     data = src.read(1)  # Read the first band
#     show(data, transform=src.transform, cmap='terrain') # Use a colormap
#     plt.show()
    
# # plt.imshow(tiff_image)
# # plt.show()

# %%
dpi_=300

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


visframe = us_states.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

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
from shapely.geometry import Polygon
us_states = geopandas.read_file(common_data +'cb_2018_us_state_500k.zip')

us_states.rename(columns={"STUSPS": "state"}, inplace=True)
us_states = us_states[~us_states.state.isin(["PR", "VI", "AS", "GU", "MP"])]
us_states = pd.merge(us_states, state_fips[["EW_meridian", "state"]], how="left", on="state")

visframe = us_states.to_crs({'init':'epsg:4326'})

fig, ax = plt.subplots(1, 1, figsize=(2, 2), sharex=True, sharey=True, dpi=dpi_)
rcp.plot_SF(SF=visframe[visframe.state.isin(["WA"])], ax_=ax, col="EW_meridian")

# %%
from shapely.geometry import Polygon
supriya_sf = geopandas.read_file(GCI_dir + "GCI13_Supriya_SF_NASA/" + 'Irrigated_WSDA_2020.shp')

supriya_sf.head(2)

# %%
Fieldlevel_pixel_info = pd.read_csv(GCI_dir + "Fieldlevel_pixel_information.csv")
Fieldlevel_pixel_info["OBJECTID"] = Fieldlevel_pixel_info["OBJECTID"].astype(int)
Fieldlevel_pixel_info.head(2)

# %%
Fieldlevel_pixel_info['OBJECTID'].max()

# %%
regional_stat_dir = "/Users/hn/Documents/01_research_data/NASA/RegionalStatData/"
preds = pd.read_csv(regional_stat_dir + "all_preds_overSample.csv")
preds.head(2)

# %%

# %%
preds_clean = preds[["ID", "DL_EVI_regular_prob_point9", "county"]].copy()
preds_clean.head(2)

# %%

# %%
[x for x in preds.columns if "DL_" in x]

# %%
preds_clean.shape

# %%
ML_data_Oct17_dir = "/Users/hn/Documents/01_research_data/NASA/ML_data_Oct17/"
A = pd.read_csv(ML_data_Oct17_dir + "01_TL_results/01_SG_EVI_TL_testPreds.csv")
A.shape

# %% [markdown]
# ### Read old shapefiles and see if we can merge if with 2020

# %%
SF_dir = research_data_ + "/NASA/000_shapefiles/"

# %%
# %%time
SF_2015 = geopandas.read_file(SF_dir +'Eastern_2015')
SF_2015.reset_index(drop=True, inplace=True)
print (len(SF_2015))
SF_2015 = SF_2015[SF_2015.ID.isin(list(preds_clean["ID"].unique()))].copy()
print (len(SF_2015))
SF_2015.head(2)

# %%
# %%time
SF_2016 = geopandas.read_file(SF_dir +'Eastern_2016')
SF_2016.reset_index(drop=True, inplace=True)
print (len(SF_2016))
SF_2016 = SF_2016[SF_2016.ID.isin(list(preds_clean["ID"].unique()))].copy()
print (len(SF_2016))
SF_2016.head(2)

# %%
# %%time
SF_2017 = geopandas.read_file(SF_dir +'Eastern_2017')
SF_2017.reset_index(drop=True, inplace=True)
print (len(SF_2017))
SF_2017 = SF_2017[SF_2017.ID.isin(list(preds_clean["ID"].unique()))].copy()
print (len(SF_2017))
SF_2017.head(2)

# %%
# %%time
SF_2018 = geopandas.read_file(SF_dir +'Eastern_2018')
SF_2018.reset_index(drop=True, inplace=True)
print (len(SF_2018))
SF_2018 = SF_2018[SF_2018.ID.isin(list(preds_clean["ID"].unique()))].copy()
print (len(SF_2018))
SF_2018.head(2)

# %%

# %%
print (preds_clean.shape)
print (sorted(preds_clean["county"].unique()))
preds_clean.head(2)

# %%
preds_clean_AdamBenton = preds_clean[preds_clean["county"].isin(["Adams", "Benton"])].copy()
preds_clean_AdamBenton.reset_index(drop=True, inplace=True)
print (preds_clean_AdamBenton.shape)
preds_clean_AdamBenton.head(2)

# %%
preds_clean_AdamBenton = pd.merge(preds_clean_AdamBenton, SF_2016[["ID", "geometry"]],
                                 how="left", on="ID")
preds_clean_AdamBenton.head(2)

# %%
preds_clean_AdamBenton.head(2)

# %%
supriya_sf.head(2)

# %%
Fieldlevel_pixel_info.head(2)

# %%
Fieldlevel_pixel_info = pd.merge(Fieldlevel_pixel_info, supriya_sf[["OBJECTID", "geometry"]],
                                 how="left", on="OBJECTID")
Fieldlevel_pixel_info.head(2)

# %%
preds_clean_AdamBenton.head(2)

# %%
Fieldlevel_pixel_info.drop(columns=["ID", "CropType", "Acres", "Irrigation", "County", "CropGroup"], inplace=True)
Fieldlevel_pixel_info.head(2)

# %%
preds_clean_AdamBenton = pd.merge(preds_clean_AdamBenton, Fieldlevel_pixel_info, how="left", on="geometry")
preds_clean_AdamBenton.head(2)

# %%

# %%
