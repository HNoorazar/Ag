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
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys
import pymannkendall as mk

import statistics
import statsmodels.api as sm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/Data/"
min_dir = data_dir_base + "Min_Data/"
reOrganized_dir = data_dir_base + "reOrganized/"

# %%
bpszone_ANPP = pd.read_csv(min_dir + "bpszone_annual_productivity_rpms_MEAN.csv")

bpszone_ANPP.sort_values(by= ['FID', 'year'], inplace=True)
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP["FID"].unique())

# %%
import geopandas
f_name = "albers_HucsGreeningBpSAtts250_For_Zonal_Stats"
bps_SF = geopandas.read_file(min_dir + f_name + "/" + f_name + ".shp")
bps_SF.head(2)

# %%
print (len(bps_SF["MinStatsID"].unique()))
print (len(bps_SF["Value"].unique()))
print (len(bps_SF["hucsgree_4"].unique()))

# %%
print ((bps_SF["hucsgree_4"] - bps_SF["Value"]).unique())

print ((list(bps_SF.index) == bps_SF.MinStatsID).sum())

# %%
bps_SF.drop(columns=["Value"], inplace=True)
bps_SF.head(2)

# %%
bpszone_ANPP["FID"].unique()[-10::]

# %% [markdown]
# #### rename columns

# %%
bpszone_ANPP.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
bpszone_ANPP.rename(columns={"area": "area_sqMeter", 
                             "count": "pixel_count",
                             "mean" : "mean_lb_per_acr"}, inplace=True)

bpszone_ANPP.head(2)

# %%
print (len(bpszone_ANPP["fid"].unique()))
print (len(bps_SF["hucsgree_4"].unique()))

# %%
print (bpszone_ANPP["fid"].unique().max())
print (bps_SF["BPS_CODE"].unique().max())

# %% [markdown]
# ### Check if all locations have all years in it

# %%
bpszone_ANPP.head(2)

# %%
len(bpszone_ANPP[bpszone_ANPP.fid == 1])

# %%
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
len(bps_SF["MinStatsID"].unique())

# %%
len(bpszone_ANPP["fid"].unique())

# %%
bpszone_ANPP["fid"].unique().max()

# %%
bps_SF["MinStatsID"].unique().max()

# %%

# %%
Albers_SF_name = min_dir + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)

Albers_SF.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
Albers_SF.rename(columns={"minstatsid": "Min_statID", 
                          "satae_max": "satae_majority_area"}, inplace=True)

Albers_SF.head(2)

# %%
bps_SF.head(2)

# %%
fips_dir = "/Users/hn/Documents/01_research_data/RangeLand/Data/reOrganized/"
county_fips_dict = pd.read_pickle(fips_dir + "county_fips.sav")

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
(Albers_SF["satae_majority_area"] == Albers_SF["state_1"]).sum()

# %%
(Albers_SF["satae_majority_area"] == Albers_SF["state_2"]).sum()

# %%
Albers_SF.shape

# %%
len(Albers_SF) - (Albers_SF["state_1"] == Albers_SF["state_2"]).sum()

# %%
(Albers_SF["state_1"] == Albers_SF["state_2"]).sum()

# %%
Albers_SF = pd.merge(Albers_SF, state_fips[["EW_meridian", "state_full"]], 
                     how="left", left_on="satae_majority_area", right_on="state_full")

Albers_SF.drop(columns=["state_full"], inplace=True)

print (Albers_SF.shape)
Albers_SF.head(3)

# %%
Albers_SF_west = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF_west.shape

# %%
bpszone_ANPP.head(2)

# %%
# I think Min mentioned that FID is the same as Min_statID
# So, let us subset the west metidians
bpszone_ANPP_west = bpszone_ANPP[bpszone_ANPP["fid"].isin(list(Albers_SF_west["Min_statID"]))].copy()

print (bpszone_ANPP.shape)
print (bpszone_ANPP_west.shape)

print (len(bpszone_ANPP) - len(bpszone_ANPP_west))

# %%
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
cols_ = ["Min_statID", "satae_majority_area", "state_1", "state_2", "EW_meridian"]
bpszone_ANPP_west = pd.merge(bpszone_ANPP_west, Albers_SF[cols_], 
                             how="left", left_on = "fid", right_on="Min_statID")
bpszone_ANPP_west.head(2)

# %% [markdown]
# # Make some plots

# %%
Albers_SF.plot(column='EW_meridian', categorical=True, legend=True);

# %%
from shapely.geometry import Polygon

gdf = geopandas.read_file("/Users/hn/Documents/01_research_data/RangeLand/Data/"+'cb_2018_us_state_500k.zip')
gdf.rename(columns={"STUSPS": "state"}, inplace=True)
gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]

gdf = pd.merge(gdf, state_fips[["EW_meridian", "state"]], how="left", on="state")

# %%
tick_legend_FontSize = 10

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * .71,
    "axes.titlesize": tick_legend_FontSize * 1,
    "xtick.labelsize": tick_legend_FontSize * .7,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * .7,  #  * 0.75
    "axes.titlepad": 5,
    'legend.handlelength': 2
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = False
plt.rcParams["ytick.left"] = False
plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["ytick.labelleft"] = False
plt.rcParams.update(params)

# %%
import warnings
warnings.filterwarnings("ignore")

from mpl_toolkits.axes_grid1 import make_axes_locatable

f, ax = plt.subplots(1,1, figsize=(8, 6), sharex=True, sharey=True, dpi=300)
plt.title('rangeland polygons on western meridian')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="1%", pad=0, alpha=1)

visframe = gdf.to_crs({'init':'epsg:5070'})
visframe[~visframe.state.isin(["AK", "HI"])].plot('EW_meridian', ax=ax, 
                                                  alpha=1, cmap='Pastel1', 
                                                  edgecolor='k', legend=True, linewidth=0.1,
                                                  # cax=cax
                                                  )

Albers_SF_west["geometry"].centroid.plot(ax=ax, color='dodgerblue', markersize=0.1)

# ax.axis('off')
plt.rcParams['axes.linewidth'] = .051
# plt.legend(fontsize=10)
plt.show();

# %%
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# f, ax = plt.subplots(1, 1, figsize=(8,6), sharex=True, sharey=True, dpi=300)
# divider = make_axes_locatable(ax)

# plt.title('rangeland polygons on western meridian', fontsize=10)

# gdf[~gdf.state.isin(["AK", "HI"])].plot('EW_meridian', ax=ax, 
#                                         alpha=0.5, cmap='Pastel1', 
#                                         edgecolor='k', legend=True, cax=cax, linewidth=0.1)

# # Albers_SF.plot(column='EW_meridian', categorical=True, legend=True);
# # Albers_SF["geometry"].centroid.plot(ax=ax, marker='o', color='red', markersize=2)
# # ax.legend(loc="best")
# plt.show()

# %%
bpszone_ANPP_west.head(2)

# %%
if (bpszone_ANPP_west["fid"] == bpszone_ANPP_west["Min_statID"]).sum() == len(bpszone_ANPP_west):
    bpszone_ANPP_west.drop(columns=["Min_statID"], inplace=True)

bpszone_ANPP_west.head(2)

# %%
num_locs = len(bpszone_ANPP_west["fid"].unique())
num_locs

# %%
median_diff = bpszone_ANPP_west[["fid", "satae_majority_area", "state_1", "state_2", "EW_meridian"]].copy()
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
# Find median of first decare and last decade of ANPP
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
median_diff.head(4)

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

# %%
bpszone_ANPP_west.head(2)

# %% [markdown]
# # MK test for ANPP

# %%
ANPP_MK_df = bpszone_ANPP_west[["fid", "satae_majority_area", "state_1", "state_2", "EW_meridian"]].copy()
print (ANPP_MK_df.shape)

ANPP_MK_df.drop_duplicates(inplace=True)
ANPP_MK_df.reset_index(drop=True, inplace=True)

print (ANPP_MK_df.shape)
ANPP_MK_df.head(3)

# %%
type(ANPP_MK_df["fid"][0])

# %%
##### z: normalized test statistics
##### Tau: Kendall Tau
MK_test_cols = ["trend", "p", "z", "Tau", "Mann_Kendal_score", "var_s", "slope", "intercept" ]

# %%
ANPP_MK_df = pd.concat([ANPP_MK_df, pd.DataFrame(columns = MK_test_cols)])
ANPP_MK_df[MK_test_cols] = ["-666", -666, -666, -666, -666, -666, -666, -666]
ANPP_MK_df.head(2)

# %%
# Why data type changed?!

ANPP_MK_df["fid"] = ANPP_MK_df["fid"].astype(np.int64)

# %%
# %%time
# populate the dataframe with MK test result now
for a_FID in ANPP_MK_df["fid"].unique():
    ANPP_TS = bpszone_ANPP_west.loc[bpszone_ANPP_west.fid==a_FID, "mean_lb_per_acr"].values
    
    # MK test
    trend, _, p, z, Tau, s, var_s, slope, intercept = mk.original_test(ANPP_TS)

    # Update dataframe by MK result
    ANPP_MK_df.loc[median_diff["fid"]==a_FID, MK_test_cols] = [trend, p, z, Tau, s, var_s, slope, intercept]

# %%
ANPP_MK_df.head(2)

# %% [markdown]
# ### Plot a couple of examples

# %%
font = {"size": 14}
matplotlib.rc("font", **font)

tick_legend_FontSize = 10

params = {
    "legend.fontsize": tick_legend_FontSize * 1.2,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.2,
    "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)
# Times New Roman

# %%
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=400)
(ax1, ax2, ax3) = axs
# ax1.grid(axis="both", which="both")
# ax2.grid(axis="both", which="both")
# ax3.grid(axis="both", which="both")

y_var = "mean_lb_per_acr"

######
###### subplot 1
######
target_idx = ANPP_MK_df["slope"].max()
a_fid = ANPP_MK_df.loc[ANPP_MK_df["slope"] == target_idx, "fid"].values[0]

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "slope"].values[0])
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

text_ = "trend: {}\nSen's slope {}, reg. slope {}".format(trend_, slope_, reg_slope)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/4)
ax1.text(1984, y_txt, text_, fontsize = 12);
# ax1.set_ylim(3000, 4500);

######
###### subplot 2
######

target_idx = ANPP_MK_df["slope"].min()
a_fid = ANPP_MK_df.loc[ANPP_MK_df["slope"] == target_idx, "fid"].values[0]

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "slope"].values[0])
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

text_ = "trend: {}\nSen's slope {}, reg. slope {}".format(trend_, slope_, reg_slope)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/1.5)
ax2.text(1984, y_txt, text_, fontsize = 12);
# ax2.set_ylim(3000, 4500);

######
###### subplot 3
###### a location with no trend

a_fid = ANPP_MK_df.loc[ANPP_MK_df["trend"] == "no trend", "fid"].values[0]

df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "slope"].values[0])
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

text_ = "trend: {}\nSen's slope {}, reg. slope {}".format(trend_, slope_, reg_slope)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/3)
ax3.text(1984, y_txt, text_, fontsize = 12);
# ax3.set_ylim(3000, 4500);

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True, 
                        gridspec_kw={"hspace": 0.15, "wspace": 0.05}, dpi=400)

(ax1, ax2) = axs
y_var = "mean_lb_per_acr"

a_fid = 100
df = bpszone_ANPP_west[bpszone_ANPP_west.fid == a_fid]
trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend"].values[0]
slope_ = int(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "slope"].values[0])
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

text_ = "trend: {}\nSen's slope {}, reg. slope {}".format(trend_, slope_, reg_slope)
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


y_var = "mean_lb_per_acr"

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

text_ = "trend: {}\nSen's slope {}, reg. slope {}".format(trend_, slope_, reg_slope)
y_txt = int(df[y_var].max()) - (int(df[y_var].max())/2.5)
ax2.text(2010, y_txt, text_, fontsize = 12);

# %%

# %%
