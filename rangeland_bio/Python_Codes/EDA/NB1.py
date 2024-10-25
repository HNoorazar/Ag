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
# # !pip3 install pymannkendall

# %%
def plot_SF(SF, ax_, cmap_ = "Pastel1", col="EW_meridian"):
    SF.plot(column=col, ax=ax_, alpha=1, cmap=cmap_, edgecolor='k', legend=False, linewidth=0.1)


# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import pymannkendall as mk

import statistics
import statsmodels.api as sm

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

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
# ####### Laptop
# rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
# min_bio_dir = rangeland_bio_base

# rangeland_base = rangeland_bio_base
# rangeland_reOrganized = rangeland_base

# %%
bpszone_ANPP = pd.read_csv(min_bio_dir + "bpszone_annual_productivity_rpms_MEAN.csv")

bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.sort_values(by= ['fid', 'year'], inplace=True)
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP["fid"].unique())

# %%
# f_name = "albers_HucsGreeningBpSAtts250_For_Zonal_Stats"
# bps_SF = geopandas.read_file(min_bio_dir + f_name + "/" + f_name + ".shp")
# bps_SF.head(2)

# %%
# %%time
Albers_SF_name = min_bio_dir + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)

Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "fid", 
                          "satae_max": "state_majority_area"}, inplace=True)

Albers_SF.head(2)

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
bpszone_ANPP.head(2)

# %%
# I think Min mentioned that FID is the same as Min_statID
# So, let us subset the west metidians
bpszone_ANPP_west = bpszone_ANPP[bpszone_ANPP["fid"].isin(list(Albers_SF_west["fid"]))].copy()

print (bpszone_ANPP.shape)
print (bpszone_ANPP_west.shape)

print (len(bpszone_ANPP) - len(bpszone_ANPP_west))

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
# {'39_years': 22430,
#  '40_years': 4332,
#  '38_years': 447,
#  '37_years': 4,
#  '35_years': 16}

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
plt.savefig(file_name)

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

# %%
ANPP_MK_df = bpszone_ANPP_west[["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]].copy()
print (ANPP_MK_df.shape)

ANPP_MK_df.drop_duplicates(inplace=True)
ANPP_MK_df.reset_index(drop=True, inplace=True)

print (ANPP_MK_df.shape)
ANPP_MK_df.head(3)

# %%
##### z: normalized test statistics
##### Tau: Kendall Tau
MK_test_cols = ["trend", "p", "z", "Tau", "Mann_Kendal_score", "var_s", "sens_slope", "intercept",
                "Spearman", "p_valueSpearman"]


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
    trend, _, p, z, Tau, s, var_s, slope, intercept = mk.original_test(ANPP_TS)

    # Spearman's rank
    Spearman, p_valueSpearman = stats.spearmanr(year_TS, ANPP_TS)

    # Update dataframe by MK result
    L_ = [trend, p, z, Tau, s, var_s, slope, intercept, Spearman, p_valueSpearman]
    ANPP_MK_df.loc[median_diff["fid"]==a_FID, MK_test_cols] = L_

# Round the columns to 6-decimals
for a_col in list(ANPP_MK_df.columns[6:]):
    ANPP_MK_df[a_col] = ANPP_MK_df[a_col].round(6)

ANPP_MK_df.head(2)

# %%

# %%
some_col = ["fid", "medians_diff_ANPP", "medians_diff_slope_ANPP", "median_ANPP_change_as_perc"]

ANPP_MK_df = pd.merge(ANPP_MK_df, median_diff[some_col], on="fid", how="left")
ANPP_MK_df.head(2)

# %%
ANPP_MK_df[["trend", "fid"]].groupby(["trend"]).count().reset_index()

# %%
Albers_SF.head(2)

# %%
some_col = ["fid", "sens_slope", "trend", "Tau", "Spearman", "p_valueSpearman",
            "medians_diff_ANPP", "medians_diff_slope_ANPP", "median_ANPP_change_as_perc"]

Albers_SF_west = pd.merge(Albers_SF_west, ANPP_MK_df[some_col], on="fid", how="left")
Albers_SF_west.head(2)

Albers_SF_west["centroid"] = Albers_SF_west["geometry"].centroid
Albers_SF_west.head(2)

# %%
import pickle
from datetime import datetime

filename = bio_reOrganized + "ANPP_MK_Spearman.sav"

export_ = {"ANPP_MK_df": ANPP_MK_df, 
           "source_code" : "NB1",
           "Author": "HN",
           "Date" : datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

pickle.dump(export_, open(filename, 'wb'))

# %%
Albers_SF_west_noCentroid = Albers_SF_west.copy()
Albers_SF_west_noCentroid.drop(columns=["centroid"], inplace=True)
f_name = bio_reOrganized + 'Albers_SF_west_ANPP_MK_Spearman.shp.zip'
Albers_SF_west_noCentroid.to_file(filename=f_name, driver='ESRI Shapefile')


# %%

# %% [markdown]
# ### Plot a couple of examples

# %%
font = {"size": 14}
matplotlib.rc("font", **font)
tick_legend_FontSize = 10
params = {"legend.fontsize": tick_legend_FontSize * 1.2,
          "axes.labelsize" : tick_legend_FontSize * 1.2,
          "axes.titlesize" : tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,
          "ytick.labelsize": tick_legend_FontSize * 1.1,
          "axes.titlepad": 10}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2, ax3) = axes
# ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both"); 
# ax3.grid(axis="both", which="both")
y_var = "mean_lb_per_acr"
######
###### subplot 1
######
target_idx = ANPP_MK_df["sens_slope"].max()
a_fid = ANPP_MK_df.loc[ANPP_MK_df["sens_slope"] == target_idx, "fid"].values[0]

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
ax1.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax1.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, reg. slope {}\nstate {}".format(trend_, slope_, reg_slope, state_)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/4)
ax1.text(1984, y_txt, text_, fontsize = 12);
# ax1.set_ylim(3000, 4500);
######
###### subplot 2
######
target_idx = ANPP_MK_df["sens_slope"].min()
a_fid = ANPP_MK_df.loc[ANPP_MK_df["sens_slope"] == target_idx, "fid"].values[0]

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
ax2.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax2.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, reg. slope {}\nstate {}".format(trend_, slope_, reg_slope, state_)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/1.5)
ax2.text(1984, y_txt, text_, fontsize = 12);
# ax2.set_ylim(3000, 4500);
######
###### subplot 3
###### a location with no trend
a_fid = ANPP_MK_df.loc[ANPP_MK_df["trend"] == "no trend", "fid"].values[0]
df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
ax3.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax3.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, reg. slope {}\nstate {}".format(trend_, slope_, reg_slope, state_)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/3)
ax3.text(1984, y_txt, text_, fontsize = 12);
# ax3.set_ylim(3000, 4500);

# %%

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=dpi_)
(ax1, ax2) = axes
# ax1.grid(axis='y', which='both')
a_fid = 100
df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].values[0])
state_ = list(df['state_majority_area'].unique())[0]
ax1.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax1.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, \nregression slope: {}\nstate: {}".format(trend_, slope_, reg_slope, state_)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/1.2)
ax1.text(1984, y_txt, text_, fontsize = 12);

######### Remove outlier and plot
df = bpszone_ANPP_west[bpszone_ANPP_west["fid"] == 100].copy()

# remove outlier
outlier_index = df[['mean_lb_per_acr']].idxmin()
outlier_index = list(outlier_index)
df.drop(index=outlier_index, inplace=True)

# MK test
ANPP_TS = df["mean_lb_per_acr"].values
trend, _, p, z, Tau, s, var_s, slope, intercept = mk.original_test(ANPP_TS)
ax2.plot(df.year, df[y_var], linewidth=3);

## regression line
X = df[["year", y_var]].copy()
X.dropna(how="any", inplace=True)
X = sm.add_constant(X)
Y = X[y_var].astype(float)
X = X.drop(y_var, axis=1)
ks = sm.OLS(Y, X)
ks_result = ks.fit()
y_pred = ks_result.predict(X)
reg_slope = int(ks_result.params["year"].round())
ax2.plot(X["year"], y_pred, color="red", linewidth=3);

text_ = "trend: {}\nSen's slope {}, \nregression slope: {}\nstate: {}".format(trend_, slope_, reg_slope, state_)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/2.5)
ax2.text(2010, y_txt, text_, fontsize = 12);

# %%

# %%
ANPP_MK_df.head(3)

# %%
sorted(list(ANPP_MK_df.loc[ANPP_MK_df["trend"] == "increasing", "sens_slope"]))[:10]

# %%
sorted(list(ANPP_MK_df.loc[ANPP_MK_df["trend"] == "increasing", "sens_slope"]))[-10:]

# %%

# %% [markdown]
# ### Plot everything and color based on slope

# %%

# %%
tick_legend_FontSize = 6
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
fig, ax = plt.subplots(1,1, figsize=(3, 3), sharex=True, sharey=True, dpi=dpi_)
plt.title(r"rangeland trends (Sen's slope)")

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))
cent_plt = Albers_SF_west.plot(column='sens_slope', legend=False, ax=ax, cmap = cm.get_cmap('RdYlGn'))
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.3, 
                     cmap = cm.get_cmap('RdYlGn'))
cbar1.set_label(r"Sen's slope")

ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show();

# %% [markdown]
# In order to have the center at ```yellow``` we manipulated ```vmin``` and ```vmax```.
# Another way is [TwoSlopeNorm](https://matplotlib.org/stable/users/explain/colors/colormapnorms.html). Not pretty.
#
# Or from AI?
# ```norm = colors.MidpointNormalize(midpoint=midpoint, vmin=data.min(), vmax=data.max())```?

# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True, dpi=dpi_)
plt.title("rangeland trends (Sen's slope)")

min_max = max(np.abs(Albers_SF_west['sens_slope'].min()), np.abs(Albers_SF_west['sens_slope'].max()))
norm1 = Normalize(vmin = -min_max, vmax = min_max, clip=True)

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_=ListedColormap(['dodgerblue', 'white']))


cent_plt = Albers_SF_west.plot(column='sens_slope', ax=ax, legend=False,
                               cmap = cm.get_cmap('RdYlGn'), norm=norm1)

cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.3, 
                     cmap = cm.get_cmap('RdYlGn'), norm=norm1)

ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show();

# %%

# %% [markdown]
# ### Plot increasing trends and color based on slope
#
# The labels seem tobe based on p-values. increasing means **```p-value < 0.05```**.

# %%
print (ANPP_MK_df[ANPP_MK_df["trend"] == "increasing"]["p"].max())
print (ANPP_MK_df[ANPP_MK_df["trend"] == "increasing"]["p"].min())

# %%
Albers_SF_west_increase = Albers_SF_west[Albers_SF_west["trend"] == "increasing"]

# %%
fig, ax = plt.subplots(1,1, figsize=(3, 3), sharex=True, sharey=True, dpi=dpi_)
plt.title(r"rangelands with greening trends")

plot_SF(SF=visframe_mainLand_west, ax_=ax, col="EW_meridian", cmap_="Pastel1")
cent_plt = Albers_SF_west_increase.plot(column='sens_slope', ax=ax)

cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.5)
cbar1.set_label(r"Sen's slope")

ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show();

# %%

# %%
fig, ax = plt.subplots(1,1, figsize=(3, 3), sharex=True, sharey=True, dpi=dpi_)
plt.title(r"rangelands with greening trends (Kendall's $\tau$)")

visframe_mainLand_west.plot(column='EW_meridian', ax=ax, alpha=1, # cax=cax,
                            cmap=ListedColormap(['white', 'black']), 
                            edgecolor='k', legend=False, linewidth=0.1)

cent_plt = Albers_SF_west_increase.plot(column='Tau', ax=ax, cmap=cm.get_cmap('RdYlGn'))
cbar1 = fig.colorbar(cent_plt.collections[1], ax=ax, orientation='vertical', shrink=0.5)
cbar1.set_label(r"Kendall's $\tau$")

ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show();

# %%

# %% [markdown]
# ### Plot positive Spearman's with p-value smaller than 0.05

# %%
Albers_SF_west["Spearman"].min()

# %%
Albers_SF_west.head(2)

# %%
Albers_SF_west_spearmanP5 = Albers_SF_west[(Albers_SF_west["Spearman"] > 0) & 
                                            (Albers_SF_west["p_valueSpearman"] < 0.05)]


# %%
# Creating the figure and axes
fig, axes = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True, dpi=dpi_)

plot_SF(SF=visframe_mainLand_west, ax_=axes, cmap_ = "Pastel1", col="EW_meridian")

spearman_plot_s = Albers_SF_west_spearmanP5.plot(column='Spearman', ax=axes)

# Add colorbar for Spearman's plot
cbar1 = fig.colorbar(spearman_plot_s.collections[1], ax=axes, orientation='vertical', shrink=0.6)
cbar1.set_label('Spearman\'s rank')

axes.set_xticks([])
axes.set_yticks([])
axes.set_title("rangelands with greening trends (Spearman's rank)")

plt.tight_layout()
plt.show();

# %%
Albers_SF_west.head(2)

# %%
# Creating the figure and axes
fig, axes = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True, dpi=dpi_)

visframe_mainLand_west.plot(column='EW_meridian', ax=axes, alpha=1, cmap='Pastel1', edgecolor='k', 
                            legend=False, linewidth=0.1)

Albers_SF_west_median_diff_increase = Albers_SF_west[Albers_SF_west["median_ANPP_change_as_perc"] > 0].copy()
plot_s = Albers_SF_west_median_diff_increase.plot(column='median_ANPP_change_as_perc', ax=axes)

# Add colorbar for Spearman's plot
cbar1 = fig.colorbar(plot_s.collections[1], ax=axes, orientation='vertical', shrink=0.6)
cbar1.set_label('median ANPP change %')

axes.set_xticks([])
axes.set_yticks([])
axes.set_title(r"greening trends. median ANPP change %")

plt.tight_layout()
plt.show();

# %%

# %% [markdown]
# # 400 difference in size between slope and Spearmans rank! 
# We need to check if the smaller set is subset of the larger set

# %%
print (Albers_SF_west_spearmanP5.shape)
Albers_SF_west_increase.shape

# %%
