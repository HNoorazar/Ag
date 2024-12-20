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
# !pip3 install pymannkendall

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

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from datetime import datetime

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%
dpi_ = 300
custom_cmap_coral = ListedColormap(['lightcoral', 'black'])
custom_cmap_BW = ListedColormap(['white', 'black'])
cmap_G = cm.get_cmap('Greens') # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap('Reds') 

# %%
from matplotlib import colormaps
print (list(colormaps)[:4])

# %%

# %%
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

min_bio_dir_v11 = rangeland_bio_data + "Min_Data_v1.1/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
bio_reOrganized_temp = rangeland_bio_data + "temp_reOrganized/"

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

# %%
county_fips_dict = pd.read_pickle(rangeland_reOrganized + "county_fips.sav")

county_fips = county_fips_dict["county_fips"]
full_2_abb = county_fips_dict["full_2_abb"]
abb_2_full_dict = county_fips_dict["abb_2_full_dict"]
abb_full_df = county_fips_dict["abb_full_df"]
filtered_counties_29States = county_fips_dict["filtered_counties_29States"]
SoI = county_fips_dict["SoI"]
state_fips = county_fips_dict["state_fips"]

state_fips = state_fips[state_fips.state != "VI"].copy()
state_fips.head(2)

# %% [markdown]
# ## Read the shapefile
# And keep the vegtype in subsequent dataframes

# %%
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)
Albers_SF.head(2)

# %%
len(Albers_SF["fid"].unique())

# %% [markdown]
# # Focus only on West Meridian

# %%
print ((Albers_SF["state_majority_area"] == Albers_SF["state_1"]).sum())
print ((Albers_SF["state_majority_area"] == Albers_SF["state_2"]).sum())
print (Albers_SF.shape)
print (len(Albers_SF) - (Albers_SF["state_1"] == Albers_SF["state_2"]).sum())
print ((Albers_SF["state_1"] == Albers_SF["state_2"]).sum())

# %%
Albers_SF = pd.merge(Albers_SF, state_fips[["EW_meridian", "state_full"]], 
                     how="left", left_on="state_majority_area", right_on="state_full")

Albers_SF.drop(columns=["state_full"], inplace=True)

print (Albers_SF.shape)
Albers_SF.head(2)

# %%
Albers_SF = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF.reset_index(drop=True, inplace=True)
Albers_SF.shape

# %%
print (len(Albers_SF["fid"].unique()))
print (len(Albers_SF["value"].unique()))
print (len(Albers_SF["hucsgree_4"].unique()))

print ((Albers_SF["hucsgree_4"] - Albers_SF["value"]).unique())
print ((list(Albers_SF.index) == Albers_SF.fid).sum())

Albers_SF.drop(columns=["value"], inplace=True)
Albers_SF.head(2)

# %% [markdown]
# ## Read NPP Data

# %%
bpszone_ANPP = pd.read_csv(min_bio_dir + "2012_bpszone_annual_productivity_rpms_MEAN.csv")

bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.sort_values(by= ['fid', 'year'], inplace=True)
bpszone_ANPP.reset_index(drop=True, inplace=True)
bpszone_ANPP.head(2)

# %%
# I think Min mentioned that FID is the same as Min_statID
# So, let us subset the west metidians
print (bpszone_ANPP.shape)
bpszone_ANPP = bpszone_ANPP[bpszone_ANPP["fid"].isin(list(Albers_SF["fid"]))].copy()
print (bpszone_ANPP.shape)

# %% [markdown]
# # Remove 2012 data?

# %%
bpszone_ANPP = bpszone_ANPP[bpszone_ANPP.year != 2012]

# %%
bpszone_ANPP = pd.merge(bpszone_ANPP, Albers_SF[["fid", "groupveg"]], how="left", on="fid")
bpszone_ANPP.head(2)

# %%
bpszone_ANPP.to_csv(min_bio_dir_v11 + "bpszone_annual_productivity_rpms_MEAN.csv", index=False)

# %%
filename = bio_reOrganized + "bpszone_ANPP_no2012.sav"
export_ = {"bpszone_ANPP": bpszone_ANPP, 
           "source_code" : "Min_2_Min_v1.1",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
bpszone_ANPP["fid"].unique()[-8::]

# %%

# %%
print (len(bpszone_ANPP["fid"].unique()))
print (len(Albers_SF["hucsgree_4"].unique()))

print (bpszone_ANPP["fid"].unique().max())
print (Albers_SF["bps_code"].unique().max())

# %% [markdown]
# ### Check if all locations have all years in it

# %%
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP[bpszone_ANPP.fid == 1])

# %%
# %%time
unique_number_of_years = {}

for a_fid in bpszone_ANPP.fid.unique():
    LL = str(len(bpszone_ANPP[bpszone_ANPP.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%
print (f'{len(Albers_SF["fid"].unique()) = }')
print (f'{len(bpszone_ANPP["fid"].unique())= }')

# %%
bpszone_ANPP.head(2)

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
if not ("EW_meridian" in bpszone_ANPP.columns):
    bpszone_ANPP = pd.merge(bpszone_ANPP, Albers_SF[cols_], how="left", on = "fid")
bpszone_ANPP.head(2)

# %% [markdown]
# # Make some plots

# %%
Albers_SF.plot(column='EW_meridian', categorical=True, legend=True);

# %%
from shapely.geometry import Polygon

gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]
gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%
2012 in list(bpszone_ANPP.year.unique())

# %%
num_locs = len(bpszone_ANPP["fid"].unique())
num_locs

# %% [markdown]
# ## Read and dump weather data

# %%
bps_weather = pd.read_csv(min_bio_dir + "bps_gridmet_mean_indices.csv")
bps_weather.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bps_weather.head(2)

bps_weather.rename(columns={"rmin_min" : "min_of_dailyMin_rel_hum", 
                            "ravg_avg" : "avg_of_dailyAvg_rel_hum",
                            "rmin_avg" : "avg_of_dailyMin_rel_hum",
                            "rmax_max" : "max_of_dailyMax_rel_hum",
                            "rmax_avg" : "max_of_dailyAvg_rel_hum",
                            "tmin_min" : "min_of_dailyMinTemp_C",
                            "tmin_avg" : "avg_of_dailyMinTemp_C",
                            "tmax_max" : "max_of_dailyMaxTemp_C",
                            "tmax_avg" : "avg_of_dailyMaxTemp_C",
                            "tavg_avg" : "avg_of_dailyAvgTemp_C",
                            "ppt" : "precip_mm_month",
                            "bpshuc" : "fid" # I have no other choice at this time!
                            }, 
                    inplace=True)
bps_weather.head(2)

drop_cols = ['alert', 'danger', 'emergency', 'thi_90', 'thi_std', "normal",
             
             'min_of_dailyMin_rel_hum',
             'avg_of_dailyMin_rel_hum',
             'max_of_dailyMax_rel_hum',
             'max_of_dailyAvg_rel_hum',
             
             'avg_of_dailyMaxTemp_C', 
             'max_of_dailyMaxTemp_C', 
             'avg_of_dailyMinTemp_C', 
             'min_of_dailyMinTemp_C']

bps_weather.drop(columns = drop_cols, axis="columns", inplace=True)
bps_weather = bps_weather[[bps_weather.columns[-1]] + list(bps_weather.columns[:-1])]
bps_weather.head(2)

# %%
Albers_SF.EW_meridian.unique()

# %%
print (bps_weather.shape)
bps_weather = bps_weather[bps_weather.fid.isin(list(Albers_SF.fid.unique()))]
print (bps_weather.shape)

# %%

# %%
filename = bio_reOrganized + "bps_weather.sav"

bps_weather.reset_index(drop=True, inplace=True)

export_ = {"bps_weather": bps_weather, 
           "source_code" : "Min_2_Min_v1.1",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

del(bps_weather)

# %%
filename = bio_reOrganized + "bps_weather.sav"
monthly_weather = pd.read_pickle(filename)
monthly_weather = monthly_weather["bps_weather"]
monthly_weather.head(2)

# %%
print (f'{len(monthly_weather["fid"].unique())=}')

# %%
# %%time
monthly_weather_wide = monthly_weather.copy()
monthly_weather_wide.sort_values(by= ['fid', 'year', "month"], inplace=True)
monthly_weather_wide["month"] = monthly_weather_wide["month"].astype(str)
df1 = monthly_weather_wide[['fid', 'year', "month", 'avg_of_dailyAvg_rel_hum']].copy()
df2 = monthly_weather_wide[['fid', 'year', "month", 'avg_of_dailyAvgTemp_C']].copy()
df3 = monthly_weather_wide[['fid', 'year', "month", 'thi_avg']].copy()
df4 = monthly_weather_wide[['fid', 'year', "month", 'precip_mm_month']].copy()
########################################################################
df1 = df1.pivot(index=['fid', 'year'], columns=['month'])
df2 = df2.pivot(index=['fid', 'year'], columns=['month'])
df3 = df3.pivot(index=['fid', 'year'], columns=['month'])
df4 = df4.pivot(index=['fid', 'year'], columns=['month'])
########################################################################
df1.reset_index(drop=False, inplace=True)
df2.reset_index(drop=False, inplace=True)
df3.reset_index(drop=False, inplace=True)
df4.reset_index(drop=False, inplace=True)
########################################################################
df1.columns = ["_".join(tup) for tup in df1.columns.to_flat_index()]
df2.columns = ["_".join(tup) for tup in df2.columns.to_flat_index()]
df3.columns = ["_".join(tup) for tup in df3.columns.to_flat_index()]
df4.columns = ["_".join(tup) for tup in df4.columns.to_flat_index()]
########################################################################
df1.rename(columns={"fid_": "fid", "year_":"year"}, inplace=True)
df2.rename(columns={"fid_": "fid", "year_": "year"}, inplace=True)
df3.rename(columns={"fid_": "fid", "year_": "year"}, inplace=True)
df4.rename(columns={"fid_": "fid", "year_": "year"}, inplace=True)

df1.head(2)

wide_WA = pd.merge(df1, df2, how="left", on=["fid", "year"])
wide_WA = pd.merge(wide_WA, df3, how="left", on=["fid", "year"])
wide_WA = pd.merge(wide_WA, df4, how="left", on=["fid", "year"])

# %%
print (wide_WA.shape)
print (wide_WA[wide_WA.fid.isin(list(Albers_SF.fid.unique()))].shape)

# %%
filename = bio_reOrganized + "bps_weather_wide.sav"
export_ = {"bps_weather_wide": wide_WA, 
           "source_code" : "Min_2_Min_v1.1",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
pickle.dump(export_, open(filename, 'wb'))
wide_WA.head(2)

# %%
FIDs_weather_ANPP_common = Albers_SF[['fid', 'state_majority_area', 'groupveg']].copy()
print (FIDs_weather_ANPP_common.shape)
FIDs_weather_ANPP_common = FIDs_weather_ANPP_common[FIDs_weather_ANPP_common.fid.isin(list(\
                                                                        monthly_weather.fid.unique()))]

print (FIDs_weather_ANPP_common.shape)

FIDs_weather_ANPP_common = FIDs_weather_ANPP_common[FIDs_weather_ANPP_common.fid.isin(list(\
                                                                        bpszone_ANPP.fid.unique()))]

print (FIDs_weather_ANPP_common.shape)

# %%
FID_veg = Albers_SF[['fid', 'groupveg']].copy()
FID_veg.reset_index(drop=True, inplace=True)


filename = bio_reOrganized + "FID_veg.sav"
export_ = {"FID_veg": FID_veg, 
           "FIDs_weather_ANPP_common" : FIDs_weather_ANPP_common,
           "source_code" : "Min_2_Min_v1.1",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))
del(FID_veg)

# %% [markdown]
# # Rangeland Landcover

# %%
landcover_dir = min_bio_dir + "Rangeland_Landcover/"

# %%
landcover_files = os.listdir(landcover_dir)
landcover_files = sorted([x for x in landcover_files if x.endswith(".csv")])
landcover_files[0:4]

# %%
# DF = pd.DataFrame()

# prefix = "Rangeland_rap_mean_vegcover_allpixels_"
# for a_year in np.arange(1986, 2024):
#     curr_df = pd.read_csv(landcover_dir + prefix + str(a_year) + ".csv")
#     DF = pd.concat([DF, curr_df])


# all_yrs_Min.sort_values(by= ['MinStatsID', 'year'], inplace=True)
# DF.sort_values(by= ['MinStatsID', 'year'], inplace=True)

# DF.reset_index(drop=True, inplace=True)
# all_yrs_Min.reset_index(drop=True, inplace=True)

# all_yrs_Min = all_yrs_Min[list(DF.columns)]

# DF.head(2)

# %%
rangeland_rap = pd.read_csv(landcover_dir + "Rangeland_rap_mean_vegcover_allpixels_1986_2023.csv")

rangeland_rap.rename(columns={"MinStatsID": "fid", 
                              "AFG" : "Annual_Forb_Grass",
                              "BGR" : "Bare_Ground",
                              "LTR" : "Litter",
                              "PFG" : "Perennial_Forb_Grass",
                              "SHR" : "Shrub",
                              "TRE" : "Tree",
                             }, inplace=True)

rangeland_rap.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
rangeland_rap.head(2)

# %%
L_ = sorted(rangeland_rap.columns)
L_ = [x for x in L_ if not (x in (["fid", "year"]))]
L_ = ["fid", "year"] + L_
rangeland_rap = rangeland_rap[L_]
rangeland_rap.head(2)

# %%
# Multiply by 100.
# For some reason Matt says this works better.
L_ = [x for x in L_ if not (x in (["fid", "year"]))]
for a_col in L_:
    rangeland_rap[a_col] = rangeland_rap[a_col] * 100

rangeland_rap["total_area"] = rangeland_rap[L_].sum(axis=1)
rangeland_rap.head(2)

# %%
print (list(Albers_SF["EW_meridian"].unique()))
Albers_SF.head(2)

# %%
# Subset to west side
print (len(rangeland_rap))
rangeland_rap = rangeland_rap[rangeland_rap["fid"].isin(list(Albers_SF["fid"].unique()))].copy()
print (len(rangeland_rap))
rangeland_rap.reset_index(drop=True, inplace=True)

# %%
filename = bio_reOrganized + "rangeland_rap.sav"

export_ = {"rangeland_rap": rangeland_rap, 
           "source_code" : "Min_2_Min_v1.1",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %% [markdown]
# ## New shapefile of Min
#
# Different from old ones?

# %%
# %%time
new_Albers_SF_name = landcover_dir + "albersHucsGreeningBpSAtts250_For_Zonal_Stats.zip"
new_Albers_SF = geopandas.read_file(new_Albers_SF_name)

new_Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
new_Albers_SF.rename(columns={"minstatsid": "fid", 
                              "satae_max": "state_majority_area"}, inplace=True)
new_Albers_SF.head(2)

# %%
# Subset to west side
print (len(new_Albers_SF))
new_Albers_SF = new_Albers_SF[new_Albers_SF["fid"].isin(list(Albers_SF["fid"].unique()))].copy()
print (len(new_Albers_SF))
new_Albers_SF.reset_index(drop=True, inplace=True)

# %%
print (len(new_Albers_SF.fid.unique()))
print (len(Albers_SF.fid.unique()))

# %%
Albers_SF.reset_index(drop=True, inplace=True)
Albers_SF.head(5)

# %%
new_Albers_SF.head(5)

# %%
col_ = "bps_name"
(new_Albers_SF[col_] == Albers_SF[col_]).sum()

# %%
