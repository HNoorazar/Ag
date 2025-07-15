import numpy as np
import pandas as pd

import time, datetime
from pprint import pprint
import os, os.path, sys
import scipy
from scipy.linalg import inv
from geopy.distance import geodesic
import tensorflow as tf
from scipy import stats as scipy_stats
import pymannkendall as mk

"""
for whatever reason, the following 2 lines dont work
even tho tensorflow in my environemnt is the same as that
on my computer!!! So, we comment out and then add it in 
different way!
"""
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

from keras import losses, optimizers, metrics
from keras import backend as K

"""
There are scipy.linalg.block_diag() and scipy.sparse.block_diag()
"""


def calculate_stat_beforeAfterBP(group, y_col, stat):
    breakpoint_year = group["BP_1"].iloc[0]  # Get the BP_1 year for the group

    # Split the data into before and after the breakpoint year
    before_bp = group[group["year"] < breakpoint_year]
    after_bp = group[group["year"] >= breakpoint_year]

    # Calculate variances
    if stat == "variance":
        stat_before = before_bp[y_col].var() if not before_bp.empty else None
        stat_after = after_bp[y_col].var() if not after_bp.empty else None
    elif stat == "mean":
        stat_before = before_bp[y_col].mean() if not before_bp.empty else None
        stat_after = after_bp[y_col].mean() if not after_bp.empty else None
    elif stat == "median":
        stat_before = before_bp[y_col].median() if not before_bp.empty else None
        stat_after = after_bp[y_col].median() if not after_bp.empty else None

    return pd.Series(
        {
            f"{y_col}_{stat}_before": stat_before,
            f"{y_col}_{stat}_after": stat_after,
        }
    )


def calculate_variance_beforeAfterBP(group, y_col):
    breakpoint_year = group["BP_1"].iloc[0]  # Get the BP_1 year for the group

    # Split the data into before and after the breakpoint year
    before_bp = group[group["year"] < breakpoint_year]
    after_bp = group[group["year"] >= breakpoint_year]

    # Calculate variances
    variance_before = before_bp[y_col].var() if not before_bp.empty else None
    variance_after = after_bp[y_col].var() if not after_bp.empty else None

    return pd.Series(
        {
            f"{y_col}_variance_before": variance_before,
            f"{y_col}_variance_after": variance_after,
        }
    )


def rolling_variance_df_prealloc(df, y_var="mean_lb_per_acr", window_size=5):
    # Step 1: Count how many total windows we'll need
    total_windows = 0
    for _, group in df.groupby("fid"):
        years = sorted(group["year"].unique())
        total_windows += max(0, len(years) - window_size + 1)

    # Step 2: Preallocate an empty DataFrame
    var_col_name = f"variance_ws{window_size}"
    columns = ["fid", "years", var_col_name]
    dtype_map = {"fid": "Int64", "years": "str", var_col_name: "float"}

    preallocated_df = pd.DataFrame(
        {
            col: pd.Series(index=range(total_windows), dtype=dtype)
            for col, dtype in dtype_map.items()
        }
    )

    # Step 3: Populate DataFrame by index
    idx = 0
    for fid, group in df.groupby("fid"):
        group = group.sort_values("year").reset_index(drop=True)
        years = group["year"].tolist()

        for i in range(len(years) - window_size + 1):
            window_years = years[i : i + window_size]
            window_data = group[group["year"].isin(window_years)]

            # is the following necessary? (can it be violated? and if does, then what?)
            if len(window_data["year"]) == window_size:
                values = window_data[y_var]
                variance_ = values.var() if len(values.dropna()) > 1 else np.nan

                preallocated_df.loc[idx] = {
                    "fid": fid,
                    "years": "_".join(map(str, window_years)),
                    var_col_name: variance_,
                }
                idx += 1

    # Step 4: Truncate to actual number of rows (if some skipped)
    return preallocated_df.iloc[:idx].reset_index(drop=True)


def compute_mk_by_fid(df: pd.DataFrame, groupby_: str, value_col: str) -> pd.DataFrame:
    """
    Apply Mann-Kendall test grouped by 'fid' for a given value column.

    Parameters:
    - df (pd.DataFrame): Input dataframe with 'fid' and value_col.
    - value_col (str): Column name on which to apply MK test.

    Returns:
    - pd.DataFrame: Results with trend, p-value, and slope for each fid.
    """

    def apply_mk(group):
        """applies MK test to a given group
        (in our case a time-series for a given FID)

        Returns the result as pd.Series.

        When we use it via .apply() function,
        we'll automatically have a dataframe.
        """
        result = mk.original_test(group[value_col])
        return pd.Series(
            {"trend": result.trend, "p_value": result.p, "slope": result.slope}
        )

    return df.groupby(groupby_).apply(apply_mk).reset_index()


def rolling_autocorr_df_prealloc(df, y_var="mean_lb_per_acr", window_size=5, lag=1):
    # Step 1: Count how many total windows we'll need
    total_windows = 0
    for _, group in df.groupby("fid"):
        years = sorted(group["year"].unique())
        total_windows += max(0, len(years) - window_size + 1)

    # Step 2: Preallocate an empty DataFrame
    corr_col_name = f"autocorr_lag{lag}_ws{window_size}"
    columns = ["fid", "years", corr_col_name]
    dtype_map = {
        "fid": "Int64",
        "years": "str",
        corr_col_name: "float",
    }

    preallocated_df = pd.DataFrame(
        {
            col: pd.Series(index=range(total_windows), dtype=dtype)
            for col, dtype in dtype_map.items()
        }
    )

    # Step 3: Populate DataFrame by index
    idx = 0
    for fid, group in df.groupby("fid"):
        group = group.sort_values("year").reset_index(drop=True)
        years = group["year"].tolist()

        for i in range(len(years) - window_size + 1):
            window_years = years[i : i + window_size]
            window_data = group[group["year"].isin(window_years)]

            # if window_data["year"].nunique() == window_size:
            if len(window_data["year"]) == window_size:
                values = window_data[y_var]
                autocorr = (
                    values.autocorr(lag=lag)
                    if values.dropna().nunique() > 1
                    else np.nan
                )

                preallocated_df.loc[idx] = {
                    "fid": fid,
                    "years": "_".join(map(str, window_years)),
                    corr_col_name: autocorr,
                }
                idx += 1

    # Step 4: Truncate to actual number of rows (if some skipped)
    return preallocated_df.iloc[:idx].reset_index(drop=True)


def rolling_autocorr(df, y_var="mean_lb_per_acr", window_size=5, lag=1):
    """
    Compute rolling autocorrelation (using pandas .autocorr()) for each FID and year window.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['fid', 'year', y_var].
        window_size (int): Number of years in each window.
        lag (int): Lag at which to compute autocorrelation (typically 1).

    Returns:
        pd.DataFrame: Resulting DataFrame with rolling autocorrelation per window.
    """
    results = []

    for fid, group in df.groupby("fid"):
        group = group.sort_values("year").reset_index(drop=True)
        years = group["year"].tolist()

        for i in range(len(years) - window_size + 1):
            window_years = years[i : i + window_size]
            window_data = group[group["year"].isin(window_years)]

            if len(window_data) == window_size:
                values = window_data[y_var]
                if values.dropna().nunique() > 1:  # .dropna() is new
                    # if len(values) > 1:
                    autocorr = values.autocorr(lag=lag)

                    ## we can replace the line above with
                    ## from statsmodels.tsa.stattools import acf
                    ## acf_vals = acf(values, nlags=lag, fft=False)
                    ## autocorr = acf_vals[lag]
                else:
                    autocorr = np.nan

                results.append(
                    {
                        "fid": fid,
                        "years": "_".join(map(str, window_years)),
                        f"autocorr_lag{lag}_ws{window_size}": autocorr,
                    }
                )

    return pd.DataFrame(results)


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


import numpy as np
import statsmodels.api as sm


def Gemini_chow_test(data, formula, split_point):
    """
    Performs the Chow test for structural break.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        formula (str): Regression formula (e.g., "y ~ x1 + x2").
        split_point (int): Index where the data is split.

    Returns:
        tuple: F-statistic and p-value of the Chow test.
    """
    # Split the data
    data1 = data[:split_point]
    data2 = data[split_point:]

    # Fit regressions
    model1 = sm.OLS.from_formula(formula, data1).fit()
    model2 = sm.OLS.from_formula(formula, data2).fit()
    model_combined = sm.OLS.from_formula(formula, data).fit()

    # Calculate RSS
    rss1 = model1.ssr
    rss2 = model2.ssr
    rss_combined = model_combined.ssr

    # Calculate Chow statistic
    k = len(model1.params)  # Number of coefficients
    n1 = len(data1)
    n2 = len(data2)
    chow_stat = ((rss_combined - (rss1 + rss2)) / k) / (
        (rss1 + rss2) / (n1 + n2 - 2 * k)
    )

    # Calculate p-value
    p_value = 1 - scipy_stats.f.cdf(chow_stat, k, n1 + n2 - 2 * k)

    return chow_stat, p_value


def ChatGPT_chow_test(y, X, split_index):
    # Ensure constant term
    X = sm.add_constant(X)

    # Full model
    model_full = sm.OLS(y, X).fit()
    rss_full = np.sum(model_full.resid**2)

    # Split models
    y1, X1 = y[:split_index], X[:split_index]
    y2, X2 = y[split_index:], X[split_index:]

    model1 = sm.OLS(y1, X1).fit()
    model2 = sm.OLS(y2, X2).fit()

    rss1 = np.sum(model1.resid**2)
    rss2 = np.sum(model2.resid**2)

    k = X.shape[1]  # number of parameters
    n = len(y)

    chow_num = (rss_full - (rss1 + rss2)) / k
    chow_den = (rss1 + rss2) / (n - 2 * k)
    chow_stat = chow_num / chow_den

    p_value = 1 - scipy_stats.f.cdf(chow_stat, k, n - 2 * k)

    return chow_stat, p_value


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def r_squared(y_true, y_pred):
    # Total sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # Residual sum of squares
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    return 1 - ss_res / ss_tot


def calculate_geodesic_distance(point1, point2):
    return geodesic((point1.y, point1.x), (point2.y, point2.x)).km


def pred_via_spreg_regime(regime_col, a_model, data_df):
    """
    model_results : a trained spreg.OLS_Regimes() model

    """
    model_results = pd.DataFrame(
        {
            "Coeff.": a_model.betas.flatten(),
            "Std. Error": a_model.std_err.flatten(),
            "P-Value": [i[1] for i in a_model.t_stat],
        },
        index=a_model.name_x,
    )

    betas = model_results["Coeff."]

    # Create a list to store predictions
    y_pred = pd.DataFrame(index=data_df.index)

    # Get regime assignments (which regime each observation belongs to)
    # This column defines which regime each observation belongs to
    regimes = data_df[regime_col].unique()

    # Loop over each observation and assign the correct betas based on its regime
    for a_regime in regimes:
        curr_X = data_df[data_df[regime_col] == a_regime].copy()
        # curr_X = curr_X[indp_vars]
        curr_coeffs_idx = [x for x in betas.index if a_regime in x]

        ## we can be anal and use curr_coeffs_idx to re-order the columns of curr_X
        ## so that we are SURE, the betas and columns are correctly multiplied!
        ## first one is CONSTANT
        ind_cols = [x.replace(a_regime, "")[1:] for x in curr_coeffs_idx][1:]
        curr_pred = betas[curr_coeffs_idx][0] + np.dot(
            curr_X[ind_cols], betas[curr_coeffs_idx][1:]
        )
        y_pred.loc[curr_X.index, "preds"] = curr_pred
    return y_pred.values


def GLS(X, y, weight_):
    """
    This function returns a generalized least square solution
    where the weight matrix is inverted in analytical form.
    """
    return (
        inv(X.T.values @ inv(weight_.values) @ X.values)
        @ X.T.values
        @ inv(weight_.values)
        @ y
    )


def WLS(X, y, weight_):
    """
    This function returns a weighted least square solution
    where the weight matrix is not inverted in analytical form.
    The weight matrix is already degined as inverse of the diagonal matrix
    where each random variable has different variance but they are un-correlated.
    """
    return inv(X.T.values @ weight_.values @ X.values) @ X.T.values @ weight_.values @ y


def create_adj_weight_matrix(data_df, adj_df, fips_var="state_fips"):
    """
    We need to make a block-diagonal weight matrix
    out of adjacency matrix.

    We need data_df and fips_var since the data might not be complete.
    For example, an state might be missing from a given year. So, we cannot
    use the adj_df in full.

    We need fips_var for distinguishing state or county

    data_df : dataframe of data to use for iterating over years.
    fips_var : string
    adj_df : dataframe of adjacency matrix
    """

    # Sort data
    data_df.sort_values(by=[fips_var, "year"], inplace=True)

    blocks = []
    diag_col_names = []
    for a_year in data_df.year.unique():
        curr_df = data_df[data_df.year == a_year]

        assert len(curr_df[fips_var].unique()) == len(curr_df[fips_var])
        curr_fips = list(curr_df[fips_var].unique())
        curr_adj_block = adj_df.loc[curr_fips, curr_fips].copy()
        assert (curr_adj_block.columns == curr_adj_block.index).all()

        blocks.append(curr_adj_block)
        diag_col_names = diag_col_names + list(curr_adj_block.columns)
    # the * allows to use a list.

    diag_block = scipy.linalg.block_diag(*blocks)
    diag_block = pd.DataFrame(diag_block)

    # rename columns so we know what's what
    diag_block.columns = diag_col_names
    diag_block.index = diag_col_names

    return diag_block


def convert_lb_2_kg(df, matt_total_npp_col, new_col_name):
    """
    Convert weight in lb to kg 0.45359237
    """
    df[new_col_name] = df[matt_total_npp_col] / 2.2046226218
    return df


def convert_lbperAcr_2_kg_in_sqM(df, matt_unit_npp_col, new_col_name):
    """
    Convert lb/acr to kg/m2

    1 acre is 4046.86 m2
    1 lb is 0.453592 kg (multiplying by 0.453592 is the same as diving by 2.205)
    Or just multiply by 0.000112085
    """
    # lb_2_kg = df[matt_unit_npp_col] / 2.205
    # lbAcr_2_kgm2 = lb_2_kg / 4046.86
    # df[new_col_name] = lbAcr_2_kgm2

    df[new_col_name] = df[matt_unit_npp_col] * 0.000112085
    return df


def add_lags_avg(df, lag_vars_, year_count, fips_name):
    """
    This function adds lagged variables in the sense of average.
    if year_count is 3, then average of past year, 2 years before, and 3 years before
          are averaged.
    df : pandas dataframe
    lag_vars_ : list of variable/column names to create the lags for
    year_count : integer: number of lag years we want.
    fips_name : str : name of column of fips; e.g. state_fips/county_fips
    """
    df_lag = df[["year", fips_name] + lag_vars_]
    df_lag = df_lag.groupby([fips_name]).rolling(year_count, on="year").mean()
    df_lag.reset_index(drop=False, inplace=True)
    df_lag.drop(columns=["level_1"], inplace=True)
    df_lag.dropna(subset=lag_vars_, inplace=True)

    df_lag["year"] = df_lag["year"] + 1
    df_lag.reset_index(inplace=True, drop=True)

    for a_col in lag_vars_:
        new_col = a_col + "_lagAvg" + str(year_count)
        df_lag.rename(columns={a_col: new_col}, inplace=True)
        df_lag.dropna(subset=[new_col], inplace=True)

    df = pd.merge(df, df_lag, on=["year", fips_name], how="left")
    df.dropna(subset=new_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_lags(df, merge_cols, lag_vars_, year_count):
    """
    This function adds lagged variables.
    df : pandas dataframe
    merge_cols : list of column names to merge on: state_fips/county_fips, year
    lag_vars_ : list of variable/column names to create the lags for
    year_count : integer: number of lag years we want.
    """
    cc_ = merge_cols + lag_vars_
    for yr_lag in np.arange(1, year_count + 1):
        df_needed_yrbefore = df[cc_].copy()
        df_needed_yrbefore["year"] = df_needed_yrbefore["year"] + yr_lag
        lag_col_names = [x + "_lag" + str(yr_lag) for x in lag_vars_]
        df_needed_yrbefore.columns = merge_cols + lag_col_names

        df = pd.merge(df, df_needed_yrbefore, on=merge_cols, how="left")
    return df


def compute_herbRatio_totalArea(hr):
    """
    We want to use average herb ratio and pixel count
    to compute total herb space.
    """
    pixel_length = 250
    pixel_area = pixel_length**2
    hr["herb_area_m2"] = pixel_area * hr["pixel_count"] * (hr["herb_avg"] / 100)

    # convert to acres for sake of consistency
    hr["herb_area_acr"] = hr["herb_area_m2"] / 4047
    hr.drop(labels=["herb_area_m2"], axis=1, inplace=True)
    return hr


def covert_totalNpp_2_unit(NPP_df, npp_total_col_, area_m2_col_, npp_unit_col_name_):
    """
    Min has unit NPP on county level.

    So, for state level, we have to compute total NPP first
    and then unit NPP for the state.

    Convert the total NPP to unit NPP.
    Total area can be area of rangeland in a county or an state

    Units are Kg * C / m^2

    1 m^2 = 0.000247105 acres
    """
    NPP_df[npp_unit_col_name_] = NPP_df[npp_total_col_] / NPP_df[area_m2_col_]
    return NPP_df


def covert_MattunitNPP_2_total(NPP_df, npp_unit_col_, acr_area_col_, npp_total_col_):
    """
    Convert the unit NPP to total area.
    Total area can be area of rangeland in a county or an state

    Units are punds per acre

    Arguments
    ---------
    NPP_df : dataframe
           whose one column is unit NPP

    npp_unit_col_ : str
           name of the unit NPP column

    acr_area_col_ : str
           name of the column that gives area in acres

    npp_area_col_ : str
           name of new column that will have total NPP

    Returns
    -------
    NPP_df : dataframe
           the dataframe that has a new column in it: total NPP

    """
    NPP_df[npp_total_col_] = NPP_df[npp_unit_col_] * NPP_df[acr_area_col_]

    return NPP_df


def covert_unitNPP_2_total(NPP_df, npp_unit_col_, acr_area_col_, npp_area_col_):
    """
    Convert the unit NPP to total area.
    Total area can be area of rangeland in a county or an state

    Units are Kg * C / m^2

    1 m^2 = 0.000247105 acres

    Arguments
    ---------
    NPP_df : dataframe
           whose one column is unit NPP

    npp_unit_col_ : str
           name of the unit NPP column

    acr_area_col_ : str
           name of the column that gives area in acres

    npp_area_col_ : str
           name of new column that will have total NPP

    Returns
    -------
    NPP_df : dataframe
           the dataframe that has a new column in it: total NPP

    """
    meterSq_to_acr = 0.000247105
    acr_2_m2 = 4046.862669715303
    NPP_df["area_m2"] = NPP_df[acr_area_col_] * acr_2_m2
    NPP_df[npp_area_col_] = NPP_df[npp_unit_col_] * NPP_df["area_m2"]
    # NPP_df[npp_area_col_] = (
    #     NPP_df[npp_unit_col_] * NPP_df[acr_area_col_]
    # ) / meterSq_to_acr
    return NPP_df


def census_stateCntyAnsi_2_countyFips(
    df, state_fip_col="state_ansi", county_fip_col="county_ansi"
):
    df[state_fip_col] = df[state_fip_col].astype("int32")
    df[county_fip_col] = df[county_fip_col].astype("int32")

    df[state_fip_col] = df[state_fip_col].astype("str")
    df[county_fip_col] = df[county_fip_col].astype("str")

    for idx in df.index:
        if len(df.loc[idx, state_fip_col]) == 1:
            df.loc[idx, state_fip_col] = "0" + df.loc[idx, state_fip_col]

        if len(df.loc[idx, county_fip_col]) == 1:
            df.loc[idx, county_fip_col] = "00" + df.loc[idx, county_fip_col]
        elif len(df.loc[idx, county_fip_col]) == 2:
            df.loc[idx, county_fip_col] = "0" + df.loc[idx, county_fip_col]

    df["county_fips"] = df[state_fip_col] + df[county_fip_col]
    return df


def clean_census(df, col_, col_to_lower=True):
    """
    Census data is weird;
        - Column can have ' (D)' or ' (Z)' in it.
        - Numbers are as strings.
    """
    if col_to_lower == True:
        df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
        col_ = col_.lower()
    if "state" in df.columns:
        df.state = df.state.str.title()
    if "county" in df.columns:
        df.county = df.county.str.title()

    df.reset_index(drop=True, inplace=True)

    """
    It is possible that this column is all numbers. 
    So, I put the following If there. 
    I am not sure how many cases are possible!!!
    So, maybe I should convert it to str first!
    But, then we might have produced NaN and who knows how many different
    patterns!!!
    """
    if df[col_].dtype == "O" or df[col_].dtype == "str":
        # df = df[df[col_] != " (D)"]
        # df = df[df[col_] != " (Z)"]
        # df = df[~(df[col_].str.contains(pat="(D)", case=False))]
        # df = df[~(df[col_].str.contains(pat="(Z)", case=False))]

        df = df[~(df[col_].str.contains(pat="(D)", case=False, na=False))]
        df = df[~(df[col_].str.contains(pat="(Z)", case=False, na=False))]
        df = df[~(df[col_].str.contains(pat="(S)", case=False, na=False))]
        df = df[~(df[col_].str.contains(pat="(NA)", case=False, na=False))]

    df.reset_index(drop=True, inplace=True)

    # this is not good condition. maybe the first one is na whose
    # type would be float.

    # if type(df[col_][0]) == str:
    #     df[col_] = df[col_].str.replace(",", "")
    #     df[col_] = df[col_].astype(float)

    df[col_] = df[col_].astype(str)
    df[col_] = df[col_].str.replace(",", "")
    df[col_] = df[col_].astype(float)

    if (
        ("state_ansi" in df.columns)
        and ("county_ansi" in df.columns)
        and not ("county_fips" in df.columns)
    ):
        df = census_stateCntyAnsi_2_countyFips(df)

    return df


def correct_Mins_county_6digitFIPS(df, col_):
    """
    Min has added a leading 1 to FIPS
    since some FIPs starts with 0.

    Get rid of 1 and convert to strings.
    """
    df[col_] = df[col_].astype("str")
    df[col_] = df[col_].str.slice(1)

    ## if county name is missing, that is for
    ## all of state. or sth. drop them. They have ' ' in them, no NA!
    if "county_name" in df.columns:
        df = df[df.county_name != " "].copy()
        df.reset_index(drop=True, inplace=True)
    return df


def correct_2digit_countyStandAloneFips(df, col_):
    """
    If the leading digit is zero, it will be gone.
    So, stand alone county FIPS can end up being 2 digit.
    We add zero back and FIPS will be string.
    """
    df[col_] = df[col_].astype("str")
    for idx in df.index:
        if len(df.loc[idx, col_]) == 2:
            df.loc[idx, col_] = "0" + df.loc[idx, col_]
        if len(df.loc[idx, col_]) == 1:
            df.loc[idx, col_] = "00" + df.loc[idx, col_]
    return df


def correct_4digitFips(df, col_):
    """
    If the leading digit is zero, it will be gone.
    So, county FIPS can end up being 4 digit.
    We add zero back and FIPS will be string.
    """
    df[col_] = df[col_].astype("str")
    for idx in df.index:
        if len(df.loc[idx, col_]) == 4:
            df.loc[idx, col_] = "0" + df.loc[idx, col_]
    return df


def correct_3digitStateFips_Min(df, col_):
    # Min has an extra 1 in his data. just get rid of it.
    df[col_] = df[col_].astype("str")
    df[col_] = df[col_].str.slice(1, 3)
    return df


def correct_state_int_fips_to_str(df, col_):
    # Min has an extra 1 in his data. just get rid of it.
    df[col_] = df[col_].astype("str")
    for idx in df.index:
        if (len(df.loc[idx, col_])) == 1:
            df.loc[idx, col_] = "0" + df.loc[idx, col_]
    return df
