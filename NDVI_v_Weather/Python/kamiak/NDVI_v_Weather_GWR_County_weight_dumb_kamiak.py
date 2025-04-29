import shutup

shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

from sklearn import preprocessing
from datetime import datetime, date
from scipy import sparse

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

####################################################################################
###
###      Parameters
###
####################################################################################

train_perc_ = int(sys.argv[1])

# do the following since walla walla has two parts and we have to use walla_walla in terminal
print("Terminal Arguments are: ")
print("train_perc_= ", train_perc_)
print("__________________________________________")

######################################################################################
######################################################################################
######################################################################################
#############
#############  Define Directories
#############

dpi_ = 300

data_basement = "/data/project/agaid/h.noorazar/"
NDVI_weather_base = data_basement + "NDVI_v_Weather/"
NDVI_weather_data_dir = NDVI_weather_base + "data/"
common_data = data_basement + "common_data/"

bio_reOrganized_dir = data_basement + "rangeland_bio/Data/reOrganized/"
output_dir = NDVI_weather_data_dir + "GWR_weights/"
os.makedirs(output_dir, exist_ok=True)
######################################################################################
######################################################################################
######################################################################################
abb_dict = pd.read_pickle(common_data + "county_fips.sav")
county_fips_df = abb_dict["county_fips"]
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())


state_name_fips = pd.DataFrame(
    {
        "state_full": list(abb_dict["full_2_abb"].keys()),
        "state": list(abb_dict["full_2_abb"].values()),
    }
)

state_name_fips = pd.merge(
    state_name_fips,
    abb_dict["state_fips"][["state_fips", "EW_meridian", "state"]],
    on=["state"],
    how="left",
)
state_name_fips.head(2)

state_fips_SoI = state_name_fips[state_name_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
state_fips_SoI.head(2)

state_fips_west = list(
    state_fips_SoI[state_fips_SoI["EW_meridian"] == "W"]["state_fips"].values
)
state_fips_west[:3]

# ****************************************************************************************
# ****************************************************************************************
# %% [markdown]
# **western meridian** saved data is already western side (```NDVI_v_Weather_dataPrep.ipynb```)
WM_counties = county_fips_df[county_fips_df["EW_meridian"] == "W"]
WM_counties = list(WM_counties["county_fips"])
len(WM_counties)

weight_rowSTD_sav = pd.read_pickle(
    bio_reOrganized_dir + "county_contiguity_Queen_neighbors_rowSTD.sav"
)
print(weight_rowSTD_sav["source_code"])

weight_rowSTD_sav = weight_rowSTD_sav["county_contiguity_Queen_neighbors_rowSTD"]
weight_rowSTD_sav.head(3)

# %%
NDVI_weather = pd.read_pickle(NDVI_weather_data_dir + "NDVI_weather.sav")
NDVI_weather = NDVI_weather["NDVI_weather_input"]
NDVI_weather.head(2)

NDVI_weather[(NDVI_weather["county_fips"] == "04001") & (NDVI_weather["year"] == 2002)]

# %%
indp_vars = [
    "county_fips",
    "year",
    "month",
    "tavg_avg",
    "ppt",
    "tavg_avg_lag1",
    "ppt_lag1",
    "MODIS_NDVI_lag1",
]
y_var = "MODIS_NDVI"

NDVI_weather = NDVI_weather[[y_var] + indp_vars]

print(NDVI_weather.shape)
NDVI_weather.dropna(inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)
print(NDVI_weather.shape)
# 258300 - 256250

# X = NDVI_weather[indp_vars].copy()
# y = NDVI_weather[y_var].copy()

# %%
print(NDVI_weather.shape)
NDVI_weather = NDVI_weather[NDVI_weather["county_fips"].isin(WM_counties)]
print(NDVI_weather.shape)

# %%
NDVI_weather[(NDVI_weather["county_fips"] == "04001") & (NDVI_weather["year"] == 2002)]

# %%
NDVI_weather[(NDVI_weather["county_fips"] == "04001") & (NDVI_weather["year"] == 2003)]
# ### Shannon County, SD
#
# Shannon County, SD (```FIPS code = 46113```) was renamed Oglala Lakota County and assigned anew FIPS code (```46102```) effective in 2014.
# Old county fips for this county is ```46113``` which is what Min has in its dataset.
# How can I take care of this? If I get an old county shapefile, then, which year?
# Lets just figure out how many mismatches are there and exclude them

# %%
NDVI_counties = list(NDVI_weather["county_fips"].unique())

# %%
weight_rowSTD_sav_counties = list(weight_rowSTD_sav.index)

NDVI_missing_from_weights = []

for a_county in NDVI_counties:
    if not (a_county in list(weight_rowSTD_sav.index)):
        NDVI_missing_from_weights = NDVI_missing_from_weights + [a_county]

print(NDVI_missing_from_weights)
weights_missing_from_NDVI = []
for a_county in weight_rowSTD_sav_counties:
    if not (a_county in NDVI_counties):
        weights_missing_from_NDVI = weights_missing_from_NDVI + [a_county]
print(len(weights_missing_from_NDVI))
weights_missing_from_NDVI


# %% [markdown]
# # Toss 46113

# %%
"46113" in list(NDVI_weather["county_fips"].unique())

# %%
NDVI_weather = NDVI_weather[NDVI_weather["county_fips"] != "46113"].copy()

# %%
"46113" in list(NDVI_weather["county_fips"].unique())

# %% [markdown]
# ## Split train and test set

# %%
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(NDVI_weather[indp_vars], NDVI_weather[y_var],
#                                                     test_size=0.4, random_state=0, shuffle=True)

# %%
print(f"{len(sorted(NDVI_weather.year.unique())) = }")
print(sorted(NDVI_weather.year.unique()))

# %%
years_list = sorted(NDVI_weather.year.unique())
train_perc = train_perc_ / 100
year_count = len(years_list)
train_years = years_list[: int(np.ceil(train_perc * year_count))]

# %%
train_DF = NDVI_weather[NDVI_weather.year.isin(train_years)].copy()
x_train = train_DF[indp_vars].copy()
y_train = train_DF[y_var].copy()

# %%
print(x_train.shape)
x_train.head(2)


# %%
weight_rowSTD_sav.head(2)

# %% [markdown]
# ### Form weight matrix for x_train.
# Sort ```x_train``` by year, so that from year to year a given county is not its own neighbor!
# Or we can explore the idea of doing this for few years prior to a given year to consider time-neighboring!!!

# %%
NDVI_weather.head(2)

# %%
x_train.head(2)

# %%
y_train.head(2)

# %%
### If we split randomly:

# Doing it this way, automatically merges x_train an y_train
# and sorting by year in one step!
# train_Xy = NDVI_weather[NDVI_weather.index.isin(x_train.index)].copy()
# train_Xy.head(2)

# print (sorted(list(train_Xy.index)) == sorted(list(x_train.index)))
# print (NDVI_weather.shape[0] - train_Xy.shape[0] == x_test.shape[0])
# # to double check
# A = x_train[x_train.index.isin(list(train_Xy.index))].copy()
# A.equals(x_train)

# # redefine
# x_train = train_Xy[indp_vars].copy()
# y_train = train_Xy[y_var].copy()

# %%
## If the data is complete; for all years and months
## we have all data, then we can compute the neighbors once
## and repeat it on main diagonal of bigger matrix.
## Let us check that? This is not possible. split of train and test was done
## randomly.

train_unique_years = sorted(x_train["year"].unique())
train_unique_months = sorted(x_train["month"].unique())
train_unique_counties = x_train["county_fips"].unique()

# %%
# # %%time
### This crashes the Kernel. Too big.
### Lets try numpy matrix, if that does not work, go into sparse matrix.

# # the data is monthly. So, This is bad operation. Kernel dies.
w = np.zeros((len(x_train), len(x_train)))
weightMatrix = pd.DataFrame(w)

idx = list(
    x_train["county_fips"]
    + "_"
    + x_train["year"].astype(str)
    + "_"
    + x_train["month"].astype(str)
)
# weightMatrix.index = idx
weightMatrix.columns = idx
weightMatrix.index = idx
weightMatrix.head(3)

#####################################################
# weightMatrix["county_fips"] = x_train["county_fips"]
# weightMatrix["year"] = x_train["year"]
# weightMatrix["month"] = x_train["month"]
# weightMatrix = weightMatrix.set_index(['county_fips', 'year', 'month'])
#####################################################
# idx_arrays = [x_train['county_fips'], x_train['year'], x_train['month']]
# idx_tuples = list(zip(*idx_arrays))
# index_ = pd.MultiIndex.from_tuples(idx_tuples, names=["county_fips", "year", "month"])
# weightMatrix.index = index_
# weightMatrix.head(13)

# %%
weight_rowSTD_sav.head(2)

# %%
a_county = "06001"
a_year = train_unique_years[0]
a_month = train_unique_months[1]

# %%
curr_month_data = NDVI_weather[
    (NDVI_weather["month"] == a_month) & (NDVI_weather["year"] == a_year)
]
len(curr_month_data) > 0

# %%
existing_counties = list(curr_month_data["county_fips"])
all_neighbor_idx = np.nonzero(
    weight_rowSTD_sav[weight_rowSTD_sav.index == a_county].values
)[1]
curr_cnty_all_neighbors = list(weight_rowSTD_sav.columns[all_neighbor_idx])
curr_cnty_all_neighbors

# %%

# %%
# %%time
for a_county in train_unique_counties:
    for a_year in train_unique_years:
        for a_month in train_unique_months:
            ### If data is complete some of the followings would be redundant
            ### but we do not take that chance, do we?

            curr_month_data = NDVI_weather[
                (NDVI_weather["month"] == a_month) & (NDVI_weather["year"] == a_year)
            ]
            if len(curr_month_data) > 0:
                # locations for which we have data for current month
                existing_counties = list(curr_month_data["county_fips"])

                # all neighbors of current location
                all_neighbor_idx = np.nonzero(
                    weight_rowSTD_sav[weight_rowSTD_sav.index == a_county].values
                )[1]
                curr_cnty_all_neighbors = list(
                    weight_rowSTD_sav.columns[all_neighbor_idx]
                )

                # existing neighbors for a given county in a given month
                # This would be redundant if we have data for all counties for all months
                curr_t_neighbors = list(
                    curr_month_data[
                        curr_month_data["county_fips"].isin(curr_cnty_all_neighbors)
                    ]["county_fips"]
                )

                row_idx = "_".join([a_county, str(a_year), str(a_month)])

                # update all columns at once
                post = "_" + str(a_year) + "_" + str(a_month)
                update_col_names = [s + post for s in curr_t_neighbors]

                weightMatrix.loc[row_idx, update_col_names] = weight_rowSTD_sav.loc[
                    a_county, curr_t_neighbors
                ].values

#                 for a_neighb in curr_t_neighbors:
#                     update_col_name = a_neighb + "_" + str(a_year) + "_" + str(a_month)
#                     weightMatrix.loc[row_idx, update_col_name] = weight_rowSTD_sav.loc[a_county, a_neighb]


# %%
weightMatrix.head(2)

# %%
weightMatrix.iloc[0].unique()

# %%
### Dump sparse matrix form
sparse_matrix = sparse.coo_matrix(weightMatrix.values)
filename = (
    output_dir
    + "monthly_NDVI_county_weight_for_GWR_"
    + str(len(train_years))
    + "trainYears_sparse.sav"
)
export_ = {
    "weightMatrix": sparse_matrix,
    "x_train": x_train,  # for sake of knowing what's where
    "y_train": y_train,
    "source_code": "NDVI_v_Weather_GWR_County_weight_dumb_kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
pickle.dump(export_, open(filename, "wb"))

### Dump full marix
filename = (
    output_dir
    + "monthly_NDVI_county_weight_for_GWR_"
    + str(len(train_years))
    + "trainYears.sav"
)
export_ = {
    "weightMatrix": weightMatrix,
    "source_code": "NDVI_v_Weather_GWR_County_weight_dumb_kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

# %%

# %%
