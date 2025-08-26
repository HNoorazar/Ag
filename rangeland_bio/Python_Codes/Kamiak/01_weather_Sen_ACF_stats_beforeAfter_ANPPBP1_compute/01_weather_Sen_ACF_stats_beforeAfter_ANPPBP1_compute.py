"""
On Aug 19 2025 I am moving this to Kamiak.
Drought are added and I do not think my computer can do it.
It was already taking 33 minutes.
"""

"""
In this notebook we use breakpoints of ANPP and we compute Sen's slope before and after that
breakpoint for weather variables, as well as mean/median before and after of ANPP-BP1

The reason this notebook's name is starting with ```01_``` is that we need 
to convert monthly data of Min to annual scale. That is done in the notebook called 
```00_weather_monthly2Annual_and_40yearsMK.ipynb```.
"""
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import random
import os, os.path, pickle, sys
import pymannkendall as mk

from scipy import stats
import scipy.stats as scipy_stats


sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc
import rangeland_plot_core as rcp
from datetime import datetime
from datetime import date
import time

start_time = time.time()
###############################################################
#######
#######    Terminal arguments
#######
# Do we want this? Lets try and see if we can get away with it.
# variable_set = str(sys.argv[1])  # drought or weather:
###############################################################
#######
#######    Directories
#######
research_data_ = "/data/project/agaid/h.noorazar/"
rangeland_bio_base = research_data_ + "rangeland_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
bio_reOrganized = rangeland_bio_data + "reOrganized/"
common_data = research_data_ + "common_data/"
breakpoints_dir = rangeland_bio_data + "breakpoints/"

###############################################################
#######
#######    Read Files
#######
weather = pd.read_pickle(
    bio_reOrganized + "bpszone_annualWeatherByHN_and_deTrended.sav"
)
print(weather.keys())
print(weather["source_code"])
weather = weather["bpszone_annual_weather_byHN"]
# weather.drop(columns=["state_1", "state_2", "state_majority_area", "EW_meridian"], inplace=True)
weather.head(2)

list(weather.columns)[:4]

###########
###########
###########
ANPP_breaks = pd.read_csv(breakpoints_dir + "ANPP_break_points.csv")
ANPP_breaks = ANPP_breaks[ANPP_breaks["breakpoint_count"] > 0]
ANPP_breaks.reset_index(drop=True, inplace=True)
ANPP_breaks.head(2)


bp_cols = ANPP_breaks["breakpoint_years"].str.split("_", expand=True)
bp_cols.columns = [f"BP_{i+1}" for i in range(bp_cols.shape[1])]
bp_cols = bp_cols.apply(pd.to_numeric, errors="coerce")
ANPP_breaks = pd.concat([ANPP_breaks, bp_cols], axis=1)
ANPP_breaks.head(2)

print(ANPP_breaks.shape)
ANPP_breaks["BP_1"] = ANPP_breaks["BP_1"].dropna().astype(int)
print(ANPP_breaks.shape)


weather.head(2)


static_columns = ["fid", "year", "state_majority_area", "EW_meridian"]
y_vars = [x for x in weather.columns if not (x in static_columns)]
y_vars[:4]

##
## Some FIDs have no breakpoints. Toss them
##
lag = 1

print(len(weather["fid"].unique()))
print(len(ANPP_breaks["fid"].unique()))

# %%
fids_ = list(ANPP_breaks["fid"].unique())
weather = weather[weather["fid"].isin(fids_)]


# %%
# %%time
# Iterate through each row in ANPP_breaks
results = []

for _, row in ANPP_breaks.iterrows():
    fid = row["fid"]
    bp_year = row["BP_1"]
    subset = weather[weather["fid"] == fid]
    a_fid_results = {}

    for y_var in y_vars:
        # Separate before and after BP_1
        before = subset[subset["year"] < bp_year][y_var]
        after = subset[subset["year"] >= bp_year][y_var]

        # Apply Mann-Kendall test if sufficient data
        result = {
            "fid": fid,
            "BP_1": bp_year,
            "n_before": len(before),
            "n_after": len(after),
            f"{y_var}_slope_before": None,
            f"{y_var}_slope_after": None,
            f"{y_var}_intercept_before": None,
            f"{y_var}_intercept_after": None,
            f"{y_var}_trend_before": None,
            f"{y_var}_trend_after": None,
            f"{y_var}_mean_before": None,
            f"{y_var}_mean_after": None,
            f"{y_var}_median_before": None,
            f"{y_var}_median_after": None,
            f"{y_var}_variance_before": None,
            f"{y_var}_variance_after": None,
            f"{y_var}_ACF1_before": None,
            f"{y_var}_ACF1_after": None,
        }
        # Why 3? is 2 enough?
        # We can count the number of cases that we had 2
        if len(before) >= 3:
            trend, _, _, _, _, _, _, slope, intercept = mk.original_test(before)
            result[f"{y_var}_trend_before"] = trend
            result[f"{y_var}_slope_before"] = slope.round(2)
            result[f"{y_var}_intercept_before"] = intercept.round(2)

        if len(after) >= 3:
            trend, _, _, _, _, _, _, slope, intercept = mk.original_test(after)
            result[f"{y_var}_trend_after"] = trend
            result[f"{y_var}_slope_after"] = slope.round(2)
            result[f"{y_var}_intercept_after"] = intercept.round(2)

        #########  Mean. Median. Variance.
        if len(before) >= 1:
            result[f"{y_var}_mean_before"] = before.mean()
            result[f"{y_var}_median_before"] = before.median()
            result[f"{y_var}_variance_before"] = before.var()

        if len(after) >= 1:
            result[f"{y_var}_mean_after"] = after.mean()
            result[f"{y_var}_median_after"] = after.median()
            result[f"{y_var}_variance_after"] = after.var()

        autocorr = before.autocorr(lag=lag) if before.nunique() > 1 else np.nan
        result[f"{y_var}_ACF1_before"] = autocorr

        autocorr = after.autocorr(lag=lag) if after.nunique() > 1 else np.nan
        result[f"{y_var}_ACF1_after"] = autocorr

        a_fid_results.update(result)

    results.append(a_fid_results)

# Create results DataFrame
slope_results = pd.DataFrame(results)
slope_results.head(3)

print(slope_results.shape)
list(slope_results.columns)

# %%
slope_results.head(2)

# ## Compute differences and ratios here
# %%
weather_ANPPBP1 = slope_results.copy()

# %%
stats_tuple_ = ("slope", "mean", "median", "variance", "ACF1")
y_cols = [x for x in weather_ANPPBP1.columns if any(k in x for k in stats_tuple_)]
y_cols[:4]

# %%
## remove before and after to get patterns
y_cols_patterns = [x.replace("before", "").replace("after", "") for x in y_cols]
y_cols_patterns = list(set(y_cols_patterns))
y_cols_patterns[:4]

# %%
for a_pattern in y_cols_patterns:
    weather_ANPPBP1[f"{a_pattern}diff"] = (
        weather_ANPPBP1[f"{a_pattern}after"] - weather_ANPPBP1[f"{a_pattern}before"]
    )

    weather_ANPPBP1[f"{a_pattern}ratio"] = (
        weather_ANPPBP1[f"{a_pattern}after"] / weather_ANPPBP1[f"{a_pattern}before"]
    )
weather_ANPPBP1.head(2)

# %%
weather_ANPPBP1.shape

# %% [markdown]
# ## Separate the diff/ratio DF from the actual values

# %%
diff_ratio_cols = [x for x in weather_ANPPBP1 if ("diff" in x) or ("ratio" in x)]
keep_cols = ["fid", "BP_1", "n_before", "n_after"] + diff_ratio_cols

# %%
weather_ANPPBP1 = weather_ANPPBP1[keep_cols]
weather_ANPPBP1.shape

# %%

# %%
filename = breakpoints_dir + "01_weather_Sen_ACF_stats_beforeAfter_ANPPBP1.sav"

export_ = {
    "sensSlope_stats_ACF_beforeAfter_ANPPBP1": slope_results,
    "weather_diffsRatios": weather_ANPPBP1,
    "source_code": "01_weather_Sen_ACF_stats_beforeAfter_ANPPBP1_compute",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
