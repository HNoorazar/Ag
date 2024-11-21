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
def plot_SF(SF, ax_, cmap_ = "Pastel1", col="EW_meridian"):
    SF.plot(column=col, ax=ax_, alpha=1, cmap=cmap_, edgecolor='k', legend=False, linewidth=0.1)


# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import pymannkendall as mk

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

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)
####### Laptop
# rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
# min_bio_dir = rangeland_bio_base

# rangeland_base = rangeland_bio_base
# rangeland_reOrganized = rangeland_base

# %% [markdown]
# ## Read the shapefile
# And keep the vegtype in subsequent dataframes

# %%
# %%time
Albers_SF_name = min_bio_dir + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)
Albers_SF.head(2)

# %%

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
filename = bio_reOrganized + "bps_weather.sav"

bps_weather.reset_index(drop=True, inplace=True)

export_ = {"bps_weather": bps_weather, 
           "source_code" : "clean_dump",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

del(bps_weather)

# %%
bpszone_ANPP = pd.read_csv(min_bio_dir + "bpszone_annual_productivity_rpms_MEAN.csv")

bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.sort_values(by= ['fid', 'year'], inplace=True)
bpszone_ANPP.reset_index(drop=True, inplace=True)
bpszone_ANPP.head(2)

# %%
bpszone_ANPP_2012 = bpszone_ANPP[bpszone_ANPP.year == 2012].copy()
print (f"{len(bpszone_ANPP.fid.unique()) = }")
print (f"{len(bpszone_ANPP_2012.fid.unique()) = }")

# %%

# %%

# %% [markdown]
# # Remove 2012 data?

# %%
# bpszone_ANPP = bpszone_ANPP[bpszone_ANPP.year != 2012]

# %%
# bpszone_ANPP = pd.merge(bpszone_ANPP, Albers_SF[["fid", "groupveg"]], how="left", on="fid")
# bpszone_ANPP.head(2)

# %%
filename = bio_reOrganized + "bpszone_ANPP.sav"
export_ = {"bpszone_ANPP": bpszone_ANPP, 
           "source_code" : "clean_dump",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
print (len(Albers_SF["fid"].unique()))
print (len(Albers_SF["value"].unique()))
print (len(Albers_SF["hucsgree_4"].unique()))

print ((Albers_SF["hucsgree_4"] - Albers_SF["value"]).unique())
print ((list(Albers_SF.index) == Albers_SF.fid).sum())

# %%
Albers_SF.drop(columns=["value"], inplace=True)
Albers_SF.head(2)

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
print (f'{bpszone_ANPP["fid"].unique().max()= }')
print (f'{Albers_SF["fid"].unique().max()= }')

# %% [markdown]
# ### Read State FIPS, abbreviation, etc

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
Albers_SF_west = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF_west.shape

# %%

# %%
bps_weather = pd.read_pickle(bio_reOrganized + "bps_weather.sav")
bps_weather = bps_weather["bps_weather"]

FIDs_weather_ANPP_common = set(bps_weather["fid"].unique()).intersection(set(bpszone_ANPP["fid"].unique()))
FIDs_weather_ANPP_common = list(set(FIDs_weather_ANPP_common).intersection(Albers_SF_west["fid"].unique()))
FIDs_weather_ANPP_common = pd.DataFrame(columns = ["fid"], data=FIDs_weather_ANPP_common)

FIDs_weather_ANPP_common = pd.merge(FIDs_weather_ANPP_common, 
                                    Albers_SF_west[["fid", "state_majority_area", "groupveg"]],
                                    on="fid", how="left")


FID_veg = Albers_SF[['fid', 'groupveg']].copy()
filename = bio_reOrganized + "FID_veg.sav"
FID_veg.reset_index(drop=True, inplace=True)
export_ = {"FID_veg": FID_veg, 
           "FIDs_weather_ANPP_common" : FIDs_weather_ANPP_common,
           "source_code" : "clean_dump",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))
del(FID_veg, bps_weather)

# %%
FIDs_weather_ANPP_common.head(2)

# %%

# %%
bpszone_ANPP.head(2)

# %%
# I think Min mentioned that FID is the same as Min_statID
# So, let us subset the west metidians
bpszone_ANPP_west = bpszone_ANPP[bpszone_ANPP["fid"].isin(list(Albers_SF_west["fid"]))].copy()

print (bpszone_ANPP.shape)
print (bpszone_ANPP_west.shape)

print (len(bpszone_ANPP) - len(bpszone_ANPP_west))

# %%
dict_ = {}
for a_year in sorted(bpszone_ANPP_west.year.unique()):
    df = bpszone_ANPP_west[bpszone_ANPP_west["year"] == a_year]
    dict_[a_year] = len(df["fid"].unique())
dict_

# %%

# %%

# %%
# %%time
unique_number_of_years = {}
for a_fid in bpszone_ANPP_west.fid.unique():
    LL = str(len(bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid])) + "_years"
    
    if not (LL in unique_number_of_years.keys()):
        unique_number_of_years[LL] = 1
    else:
        unique_number_of_years[LL] = \
            unique_number_of_years[LL] + 1

unique_number_of_years

# %%

# %%
bpszone_ANPP_west.head(2)

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
bpszone_ANPP_west = pd.merge(bpszone_ANPP_west, Albers_SF[cols_], how="left", on = "fid")
bpszone_ANPP_west.head(2)

# %% [markdown]
# # Make some plots

# %%
Albers_SF.plot(column='EW_meridian', categorical=True, legend=True);

# %%
from shapely.geometry import Polygon

gdf = geopandas.read_file(rangeland_base +'cb_2018_us_state_500k.zip')
# gdf = geopandas.read_file(rangeland_bio_base +'cb_2018_us_state_500k')

gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]

gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
visframe = gdf.to_crs({'init':'epsg:5070'})
visframe_mainLand = visframe[~visframe.state.isin(["AK", "HI"])].copy()

visframe_mainLand_west = visframe[visframe.EW_meridian.isin(["W"])].copy()
visframe_mainLand_west = visframe_mainLand_west[~visframe_mainLand_west.state.isin(["AK", "HI"])].copy()

# %%

# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize,
          "axes.labelsize": tick_legend_FontSize * .71,
          "axes.titlesize": tick_legend_FontSize * 1,
          "xtick.labelsize": tick_legend_FontSize * .7,
          "ytick.labelsize": tick_legend_FontSize * .7,
          "axes.titlepad": 5,
          'legend.handlelength': 2}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = False
plt.rcParams["ytick.left"] = False
plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["ytick.labelleft"] = False
plt.rcParams.update(params)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True, dpi=dpi_)
ax.set_xticks([]); ax.set_yticks([])
plt.title('rangeland polygons in Albers shapefile')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="1%", pad=0, alpha=1)
plot_SF(SF=visframe_mainLand, ax_=ax, col="EW_meridian",
       cmap_ = "Pastel1")
Albers_SF["geometry"].centroid.plot(ax=ax, color='dodgerblue', markersize=0.4)

plt.rcParams['axes.linewidth'] = .051
plt.tight_layout()
# plt.legend(fontsize=10) # ax.axis('off')
# plt.show();
file_name = bio_plots + "Albers_SF_locations.png"
# plt.savefig(file_name)

# %%
bpszone_ANPP_west.head(2)

# %%
num_locs = len(bpszone_ANPP_west["fid"].unique())
num_locs

# %%
median_diff = bpszone_ANPP_west[["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]].copy()
print (median_diff.shape)

median_diff.drop_duplicates(inplace=True)
median_diff.reset_index(drop=True, inplace=True)

print (median_diff.shape)
median_diff.head(3)

# %%
# %%time
## Check if each ID has unique state_1, state_2, and state_majority_area
bad_FIDs = []
for a_FID in median_diff["fid"].unique():
    curr_df = median_diff[median_diff.fid == a_FID]
    if len(curr_df) > 1:
        print (a_FID)
        bad_FIDs += bad_FIDs + [a_FID]

# %%
median_diff.head(2)

# %%
# Not all locations have the same number of data in them
# Lets just assume they do. The missing year
median_diff["first_10_years_median_ANPP"] = -666
median_diff["last_10_years_median_ANPP"]  = -666

# %%
# %%time
# Find median of first decade and last decade of ANPP

for a_FID in median_diff["fid"].unique():
    curr_df = bpszone_ANPP_west[bpszone_ANPP_west["fid"] == a_FID]
    
    min_year = curr_df["year"].min()
    max_year = curr_df["year"].max()
    
    first_decade = curr_df[curr_df["year"] < min_year + 10]
    last_decade  = curr_df[curr_df["year"] > max_year - 10]
    
    median_diff.loc[median_diff["fid"] == a_FID, "first_10_years_median_ANPP"] = \
                                                first_decade['mean_lb_per_acr'].median()

    median_diff.loc[median_diff["fid"] == a_FID, "last_10_years_median_ANPP"] = \
                                                    last_decade['mean_lb_per_acr'].median()

# %%
median_diff.head(2)

# %%
year_diff = bpszone_ANPP_west["year"].max() - bpszone_ANPP_west["year"].min()

median_diff["medians_diff_ANPP"] = median_diff["last_10_years_median_ANPP"] - \
                                      median_diff["first_10_years_median_ANPP"]

median_diff["medians_diff_slope_ANPP"] = median_diff["medians_diff_ANPP"] / year_diff
median_diff.head(2)

# %%
print (median_diff["medians_diff_slope_ANPP"].min())
print (median_diff["medians_diff_slope_ANPP"].max())

# %%
median_diff[median_diff["medians_diff_slope_ANPP"] < -19]

# %% [markdown]
# ### change as percenatge of first decade

# %%
median_diff["median_ANPP_change_as_perc"] = (100 * median_diff["medians_diff_ANPP"]) / \
                                                  median_diff["first_10_years_median_ANPP"]
median_diff.head(2)

# %%
bpszone_ANPP_west.head(2)

# %% [markdown]
# # MK test for ANPP and Spearman's rank

# %% [markdown]
# # Auto correlation test

# %%
a_FID = bpszone_ANPP_west["fid"].unique()[0]
ANPP_TS = bpszone_ANPP_west.loc[bpszone_ANPP_west.fid==a_FID, "mean_lb_per_acr"].values
year_TS = bpszone_ANPP_west.loc[bpszone_ANPP_west.fid==a_FID, "year"].values

print (len(mk.original_test(ANPP_TS)))
print (len(mk.yue_wang_modification_test(ANPP_TS)))
print (len(mk.hamed_rao_modification_test(ANPP_TS)))
mk.hamed_rao_modification_test(ANPP_TS)

# %%
mk.yue_wang_modification_test(ANPP_TS)

# %%
mk.original_test(ANPP_TS)

# %% [markdown]
# ### Test to see if Slope, intercept, Tau are identical

# %%
# %%time
diff_fid = {}
# populate the dataframe with MK test result now
for a_FID in bpszone_ANPP_west["fid"].unique():
    ANPP_TS = bpszone_ANPP_west.loc[bpszone_ANPP_west.fid==a_FID, "mean_lb_per_acr"].values    
    # MK test original
    _, _, _, _, Tau, s, _, slope, intercept = mk.original_test(ANPP_TS)
    
    # MK test rao
    _, _, _, _, Tau_rao, s_rao, _, slope_rao, intercept_rao = mk.hamed_rao_modification_test(ANPP_TS)
    
    # MK test yue
    _, _, _, _, Tau_yue, s_yue, _, slope_yue, intercept_yue = mk.yue_wang_modification_test(ANPP_TS)
    
    if (Tau != Tau_rao) or (Tau != Tau_yue) or (Tau_rao != Tau_yue):
        diff_fid[a_FID + "Tau"] = [Tau, Tau_yue, Tau_rao]
        
    if (s != s_rao) or (s != s_yue) or (s_rao != s_yue):
        diff_fid[a_FID + "s"] = [s, s_yue, s_rao]
        
    if (slope != slope_rao) or (slope != slope_yue) or (slope_rao != slope_yue):
        diff_fid[a_FID + "slope"] = [slope, slope_yue, slope_rao]

    if (intercept != intercept_rao) or (intercept != intercept_yue) or (intercept_rao != intercept_yue):
        diff_fid[a_FID + "intercept"] = [intercept, intercept_yue, intercept_rao]

len(diff_fid)

# %%
need_cols = ["fid", "state_majority_area", "EW_meridian"]
ANPP_MK_df = bpszone_ANPP_west[need_cols].copy()
print (ANPP_MK_df.shape)

ANPP_MK_df.drop_duplicates(inplace=True)
ANPP_MK_df.reset_index(drop=True, inplace=True)

print (ANPP_MK_df.shape)
ANPP_MK_df.head(3)


##### z: normalized test statistics
##### Tau: Kendall Tau
MK_test_cols = ["trend", "trend_yue", "trend_rao",
                "p", "p_yue", "p_rao",
                "z", "Tau", "MK_score", 
                "var_s", "var_s_yue", "var_s_rao",
                "sens_slope", "intercept",
                "Spearman", "p_Spearman"]

ANPP_MK_df = pd.concat([ANPP_MK_df, pd.DataFrame(columns = MK_test_cols)])
ANPP_MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols)-1)

# Why data type changed?!
ANPP_MK_df["fid"] = ANPP_MK_df["fid"].astype(np.int64)
ANPP_MK_df.head(2)

# %%
# %%time
# populate the dataframe with MK test result now
for a_FID in ANPP_MK_df["fid"].unique():
    ANPP_TS = bpszone_ANPP_west.loc[bpszone_ANPP_west.fid==a_FID, "mean_lb_per_acr"].values
    year_TS = bpszone_ANPP_west.loc[bpszone_ANPP_west.fid==a_FID, "year"].values
    
    # MK test
    trend, _, p, z, Tau, MK_score, var_s, slope, intercept = mk.original_test(ANPP_TS)
    trend_yue, _, p_yue, _, _, _, var_s_yue, _, _ = mk.yue_wang_modification_test(ANPP_TS)
    trend_rao, _, p_rao, _, _, _, var_s_rao, _, _ = mk.hamed_rao_modification_test(ANPP_TS)    

    # Spearman's rank
    Spearman, p_Spearman = stats.spearmanr(year_TS, ANPP_TS)

    # Update dataframe by MK result
    L_ = [trend, trend_yue, trend_rao,
          p, p_yue, p_rao, 
          z, Tau, MK_score, 
          var_s, var_s_yue, var_s_rao,
          slope, intercept, Spearman, p_Spearman]
    ANPP_MK_df.loc[ANPP_MK_df["fid"]==a_FID, MK_test_cols] = L_

# Round the columns to 6-decimals
for a_col in list(ANPP_MK_df.columns[7:]):
    ANPP_MK_df[a_col] = ANPP_MK_df[a_col].round(6)

ANPP_MK_df.head(2)

# %%
some_col = ["fid", "medians_diff_ANPP", "medians_diff_slope_ANPP", "median_ANPP_change_as_perc"]

ANPP_MK_df = pd.merge(ANPP_MK_df, median_diff[some_col], on="fid", how="left")
ANPP_MK_df.head(2)

# %%
trend_col = "trend"
trend_count_orig = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_orig.rename(columns={"fid": "fid_original"}, inplace=True)

trend_count_orig

# %%
trend_col = "trend_yue"
trend_count_yue = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_yue.rename(columns={"fid": "fid_yue",
                                "trend_yue" : "trend"}, inplace=True)
trend_count_yue

# %%
trend_col = "trend_rao"
trend_count_rao = ANPP_MK_df[[trend_col, "fid"]].groupby([trend_col]).count().reset_index()
trend_count_rao.rename(columns={"fid": "fid_rao", 
                               "trend_rao" : "trend"}, inplace=True)
trend_count_rao

# %%
trend_counts = pd.merge(trend_count_orig, trend_count_yue, on="trend", how="outer")
trend_counts = pd.merge(trend_counts, trend_count_rao, on="trend", how="outer")
trend_counts

# %%
spearman_increase_pval5 = ANPP_MK_df[ANPP_MK_df["p_Spearman"] < 0.05]
spearman_increase_pval5 = spearman_increase_pval5[spearman_increase_pval5["Spearman"] > 0]
spearman_increase_pval5.shape

# %%
spearman_decrease_pval5 = ANPP_MK_df[ANPP_MK_df["p_Spearman"] < 0.05]
spearman_decrease_pval5 = spearman_decrease_pval5[spearman_decrease_pval5["Spearman"] < 0]
spearman_decrease_pval5.shape

# %%
len(ANPP_MK_df) - (len(spearman_increase_pval5) + len(spearman_decrease_pval5))

# %%
trend_col = "trend"
decreasing_df_orig = ANPP_MK_df[ANPP_MK_df[trend_col] == "decreasing"].copy()
decreasing_df_orig = decreasing_df_orig[["fid", "state_majority_area"]].groupby(
                            ["state_majority_area"]).count().reset_index()

decreasing_df_orig

# %%
trend_col = "trend_yue"
decreasing_df_yue = ANPP_MK_df[ANPP_MK_df[trend_col] == "decreasing"].copy()
decreasing_df_yue = decreasing_df_yue[["fid", "state_majority_area"]].groupby(
                                    ["state_majority_area"]).count().reset_index()

decreasing_df_yue

# %%
trend_col = "trend_rao"
decreasing_df_rao = ANPP_MK_df[ANPP_MK_df[trend_col] == "decreasing"].copy()
decreasing_df_rao = decreasing_df_rao[["fid", "state_majority_area"]].groupby(\
                                                    ["state_majority_area"]).count().reset_index()

decreasing_df_rao

# %%
Albers_SF.head(2)

# %%
print (MK_test_cols)

# %%
some_col = ["fid", "sens_slope", "trend", 'trend_yue', 'trend_rao', 
            "Tau", "Spearman", "p_Spearman",
            "medians_diff_ANPP", "medians_diff_slope_ANPP", "median_ANPP_change_as_perc"]

Albers_SF_west = pd.merge(Albers_SF_west, ANPP_MK_df[some_col], on="fid", how="left")
Albers_SF_west.head(2)

Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid
Albers_SF_west.head(2)

# %%

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman.sav"

export_ = {"ANPP_MK_df": ANPP_MK_df, 
           "source_code" : "clean_dump",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
Albers_SF_west_noCentroid = Albers_SF_west.copy()
Albers_SF_west_noCentroid.drop(columns=["centroid"], inplace=True)
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
Albers_SF_west_noCentroid.to_file(filename=f_name, driver='ESRI Shapefile')


# %%
Albers_SF_west_noCentroid.columns

# %%
# %%time
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
A = geopandas.read_file(f_name)
A.head(2)

# %%
A.columns

# %% [markdown]
# ### Plot a couple of examples

# %%
sorted(list(ANPP_MK_df.loc[ANPP_MK_df["trend"] == "increasing", "sens_slope"]))[:10]

# %%
sorted(list(ANPP_MK_df.loc[ANPP_MK_df["trend"] == "increasing", "sens_slope"]))[-10:]

# %% [markdown]
# # 400 difference in size between slope and Spearmans rank! 
# We need to check if the smaller set is subset of the larger set

# %%

# %%
