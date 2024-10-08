import numpy as np
import pandas as pd

# import geopandas as gpd
from IPython.display import Image

# from shapely.geometry import Point, Polygon
import time, datetime
import scipy, math
from math import factorial

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr
from pprint import pprint
from collections import defaultdict

import os, os.path, sys

from tensorflow.keras.utils import to_categorical, load_img, img_to_array

# from keras.models import Sequential, Model, load_model
# from keras.applications.vgg16 import VGG16
# import tensorflow as tf

# # from keras.optimizers import SGD
# from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
# from tensorflow.keras.optimizers import SGD
# from keras.preprocessing.image import ImageDataGenerator

###
### These will be more generalized functions of remote_sensing_core.py
### Hence, less hard coding, which implies column/variavle wise we
### will be minimalistic. e.g. column: lastSurveydate should not be included
### here.
###


###########################################################


def yu_4_PA_Eq22(test_df, ref_class, ML_pred_col):
    """
    This is the same as Eq. 18 of the paper.
    Same as function yu_4_UA_Eq18(.)
    Except it uses ref_class instead of map_class

    Arguments
    ---------
    ref_class : int
        here ref_class can be either 1 for single- or 2 for double-cropped
        according to prediction

    ML_pred_col : str
        Name of the ML model we want, since we have trained
        more than one model

    Returns
    ---------
    Modifies test_df in place.
    Adds a column UA_singleClass or UA_singleClass to the dataframe
    which indicates whether a field is classified correctly or not
    in a given class given by ref_class

    This is Eq. 22 of Stehman's paper
    """
    assert ref_class in [1, 2]

    if "Vote" in test_df:
        test_df.rename(columns={"Vote": "ref_class"}, inplace=True)

    if ref_class == 1:
        new_variable = "PA_single_yu_" + ML_pred_col
    else:
        new_variable = "PA_double_yu_" + ML_pred_col

    test_df[new_variable] = 0

    correctly_classified = test_df[test_df["ref_class"] == test_df[ML_pred_col]]
    correc_class_targetClass = correctly_classified[
        correctly_classified[ML_pred_col] == ref_class
    ]
    idx = correc_class_targetClass.index
    test_df[new_variable] = 0
    test_df.loc[idx, new_variable] = 1


def xu_4_UA_Eq19(test_df, map_class, ML_pred_col):
    """
    # Tested Against Stehman's Table 2
    This is the same as Eq 14 (area_count_refClass_yu_Eq14()).
    Just uses map_class instead of ref_class
    Arguments
    ---------
    map_class : int
        here map_class can be either 1 for single- or 2 for double-cropped
        according to ML prediction

    ML_pred_col : str
        Name of the column containing predictions of a given ML.

    Returns
    ---------
    Modifies test_df in place.
    Adds a column single__yu or double_class_yu to the dataframe
    which indicates whether a field is single-cropped or double-cropped.
    by vote/reference/truth.

    This is Eq. 19 of Stehman's paper
    """
    assert map_class in [1, 2]

    if map_class == 1:
        new_variable = "UA_single_xu_" + ML_pred_col
    else:
        new_variable = "UA_double_xu_" + ML_pred_col

    test_df[new_variable] = 0

    indices = test_df[test_df[ML_pred_col] == map_class].index
    test_df.loc[indices, new_variable] = 1


def yu_4_UA_Eq18(test_df, map_class, ML_pred_col):
    """
    Arguments
    ---------
    map_class : int
        here map_class can be either 1 for single- or 2 for double-cropped
        according to prediction

    ML_pred_col : str
        Name of the ML model we want, since we have trained
        more than one model

    Returns
    ---------
    Modifies test_df in place.
    Adds a column UA_singleClass or UA_singleClass to the dataframe
    which indicates whether a field is classified correctly or not
    in a given class given by map_class

    This is Eq. 18 of Stehman's paper
    """
    assert map_class in [1, 2]

    if "Vote" in test_df:
        test_df.rename(columns={"Vote": "ref_class"}, inplace=True)

    if map_class == 1:
        new_variable = "UA_single_yu_" + ML_pred_col
    else:
        new_variable = "UA_double_yu_" + ML_pred_col

    test_df[new_variable] = 0

    correctly_classified = test_df[test_df["ref_class"] == test_df[ML_pred_col]]
    correc_class_targetClass = correctly_classified[
        correctly_classified[ML_pred_col] == map_class
    ]
    idx = correc_class_targetClass.index
    test_df[new_variable] = 0
    test_df.loc[idx, new_variable] = 1


def overal_acc_yu_Eq12(test_df, ML_pred_col):
    """
    # Tested against Table 2 of Stehman
    Arguments
    ---------
    ML_pred_col : str
        name of the column that contains predictions by a
        given ML

    Returns
    ---------
    Modifies test_df in place.
    Adds a column that indicates whether predictions are
    correct (1) or not (0)

    This is Eq. 12 of Stehman's paper
    """

    if "Vote" in test_df:
        test_df.rename(columns={"Vote": "ref_class"}, inplace=True)

    new_variable = "overal_acc_yu_" + ML_pred_col

    test_df[new_variable] = 0

    indices = test_df[test_df["ref_class"] == test_df[ML_pred_col]].index
    test_df.loc[indices, new_variable] = 1


def xu_4_PA_Eq23_yuEq14(test_df, ref_class):
    """
    Checked agains Stehman.
    Arguments
    ---------
    ref_class : int
        here ref_class can be either 1 for single- or 2 for double-cropped
        according to truth/vote/reality

    Returns
    ---------
    Modifies test_df in place.
    Adds a column single_class_yu or double_class_yu to the dataframe
    which indicates whether a field is single-cropped or double-cropped.
    by vote/reference/truth.
    This is Eq. 14 of Stehman's paper
    """
    assert ref_class in [1, 2]

    if "Vote" in test_df:
        test_df.rename(columns={"Vote": "ref_class"}, inplace=True)

    if ref_class == 1:
        new_variable = "PA_single_xu"
    else:
        new_variable = "PA_double_xu"

    test_df[new_variable] = 0

    indices = test_df[test_df["ref_class"] == ref_class].index
    test_df.loc[indices, new_variable] = 1


def SE_4_UA_PA(Stehman_T3, stratum_area_df, area_or_count, yu_col, xu_col, stratum_col):
    """
    as it is possible that there is only one instance of a given strata
    in the set, we need to drop them. This is already done in Stehman_T3.
    We must do the same for stehman_T2
    """

    stratum_area_df = stratum_area_df[
        stratum_area_df[stratum_col].isin(list(Stehman_T3[stratum_col]))
    ]

    Stehman_T3.sort_values(by=[stratum_col], inplace=True)
    Stehman_T3.reset_index(drop=True, inplace=True)

    stratum_area_df.sort_values(by=[stratum_col], inplace=True)
    stratum_area_df.reset_index(drop=True, inplace=True)

    if area_or_count == "area":
        pop_area_count = "population_area"
        sample_area_count = "sample_area"
    else:
        pop_area_count = "population_count"
        sample_area_count = "sample_count"

    Rhat = UA_PA_Rhat_Eq27(
        Stehman_T3=Stehman_T3,
        yu_col=yu_col,
        xu_col=xu_col,
        stratum_area_df=stratum_area_df,
        stratum_col=stratum_col,
        stratum_area_count_col=pop_area_count,
    )

    outer_denom = mean_times_a_column(
        Stehman_T3=Stehman_T3,
        yu_col=xu_col,
        stratum_area_df=stratum_area_df,
        stratum_col=stratum_col,
        stratum_area_count_col=pop_area_count,
    )
    outer_denom = 1 / (outer_denom.sum() ** 2)

    # first term inside the Sigma is Nh_satar_sq
    Nh_star_sq = stratum_area_df[pop_area_count].values ** 2

    second_term = (
        1 - (stratum_area_df[sample_area_count] / stratum_area_df[pop_area_count])
    ).values

    # compute the third term in Eq 28 of Stehman's paper
    sy_sq = Stehman_T3[Stehman_T3["parameters"] == "var"][yu_col].values
    RhatSq_X_sxSq = (Rhat**2) * Stehman_T3[Stehman_T3["parameters"] == "var"][
        xu_col
    ].values
    twoRhat_sxy = (2 * Rhat) * Stehman_T3[Stehman_T3["parameters"] == "covar"][
        yu_col
    ].values
    third_term = sy_sq + RhatSq_X_sxSq - twoRhat_sxy

    inner_denoms = 1 / stratum_area_df[sample_area_count].values

    a = np.multiply(Nh_star_sq, second_term)
    b = np.multiply(a, third_term)
    c = np.multiply(b, inner_denoms)
    # variance_ = np.abs(c.sum() * outer_denom)
    variance_ = c.sum() * outer_denom
    return (np.sqrt(variance_)).round(3)


def SE_4_OA_and_PropArea(
    stehman_T2, stehman_T3, stratum_area_df, area_or_count, variable, stratum_col
):
    """
    as it is possible that there is only one instance of a given strata
    in the set, we need to drop them. This is already done in stehman_T3.
    We must do the same for stehman_T2
    """
    stehman_T2 = stehman_T2[stehman_T2[stratum_col].isin(list(stehman_T3[stratum_col]))]
    stratum_area_df = stratum_area_df[
        stratum_area_df[stratum_col].isin(list(stehman_T3[stratum_col]))
    ]

    stehman_T2.sort_values(by=[stratum_col], inplace=True)
    stehman_T2.reset_index(drop=True, inplace=True)

    stehman_T3.sort_values(by=[stratum_col], inplace=True)
    stehman_T3.reset_index(drop=True, inplace=True)

    stratum_area_df.sort_values(by=[stratum_col], inplace=True)
    stratum_area_df.reset_index(drop=True, inplace=True)

    if area_or_count == "area":
        pop_area_count = "population_area"
        sample_area_count = "sample_area"
    else:
        pop_area_count = "population_count"
        sample_area_count = "sample_count"

    outer_denom = stratum_area_df[pop_area_count].sum() ** 2

    Nh_star_sq = stratum_area_df[pop_area_count].values ** 2
    second_term = (
        1 - (stratum_area_df[sample_area_count] / stratum_area_df[pop_area_count])
    ).values
    syh_sq = stehman_T3[stehman_T3["parameters"] == "var"][variable].values
    inner_denoms = 1 / stratum_area_df[sample_area_count].values

    first_X_second = np.multiply(Nh_star_sq, second_term)
    first_X_second_X_third = np.multiply(first_X_second, syh_sq)
    sum_ = np.multiply(first_X_second_X_third, inner_denoms).sum()
    SE_sq = sum_ / outer_denom
    return (np.sqrt(SE_sq)).round(3)


def UA_PA_Rhat_Eq27(
    Stehman_T3, yu_col, xu_col, stratum_area_df, stratum_col, stratum_area_count_col
):
    numer = mean_times_a_column(
        Stehman_T3, yu_col, stratum_area_df, stratum_col, stratum_area_count_col
    )
    denom = mean_times_a_column(
        Stehman_T3, xu_col, stratum_area_df, stratum_col, stratum_area_count_col
    )
    return (numer.sum() / denom.sum()).round(3)


def AreaClassProportion_and_OA(
    Stehman_T3, yu_col, stratum_area_df, stratum_col, stratum_area_count_col
):
    """
    Arguments
    ---------
    Stehman_T3 : dataframe
        Table 3 from Stehman paper. It is assumed the column "parameters"
        exist in this datatable

    stratum_col : str
        the name of the stratum column. e.g. stratum, cropType.

    yu_col : str
        name of column in Stehman_T3 to be used

    stratum_area_df : dataframe
        dataframe containing area per stratum

    stratum_col : str
        name of the column containing the name of each stratum
        This column must be identical in both Stehman_T3 and stratum_area_df

    stratum_area_count_col : str
        name of the column containing the area of each stratum

    Returns
    ---------

    """

    numerator = mean_times_a_column(
        Stehman_T3, yu_col, stratum_area_df, stratum_col, stratum_area_count_col
    )
    # stratum_area_df = pd.merge(
    #     stratum_area_df,
    #     Stehman_T3[Stehman_T3["parameters"] == "mean"][[stratum_col, yu_col]],
    #     how="left",
    #     on=stratum_col,
    # )
    return (numerator.sum() / stratum_area_df[stratum_area_count_col].sum()).round(3)


def mean_times_a_column(
    Stehman_T3, yu_col, stratum_area_df, stratum_col, stratum_area_count_col
):
    """
    a_column in the name of this function can be the area of each stratum or.

    Arguments
    ---------
    Stehman_T3 : dataframe
        Table 3 from Stehman paper. It is assumed the column "parameters"
        exist in this datatable

    stratum_col : str
        the name of the stratum column. e.g. stratum, cropType.

    yu_col : str
        name of column in Stehman_T3 to be used

    stratum_area_df : dataframe
        dataframe containing area per stratum

    stratum_col : str
        name of the column containing the name of each stratum
        This column must be identical in both Stehman_T3 and stratum_area_df

    stratum_area_count_col : str
        name of the column containing the area of each stratum

    Returns
    ---------

    """
    stratum_area_df = pd.merge(
        stratum_area_df,
        Stehman_T3[Stehman_T3["parameters"] == "mean"][[stratum_col, yu_col]],
        how="left",
        on=stratum_col,
    )

    return stratum_area_df[stratum_area_count_col] * stratum_area_df[yu_col]


def mean_var_covar_table(df, stratum_col, variable_cols, cov_variables):
    """
    Arguments
    ---------
    df : DataFrame
        That looks like Table 2 of Stehman paper.

    stratum_col : str
        name of strarums

    variable_cols : list
        list of variables for which we want to compute mean and var

    cov_variables : list
        This is list of lists.
        Each list has two column names in it: for covariance we need two variables.

    Returns
    ---------
    mean_var_covar_df : DataFrame
         table of means and variances and covariances
         similar to table 3 of Stehman paper

    Stehman paper: http://dx.doi.org/10.1080/01431161.2014.930207
    """

    ### It's possible that there is only one example
    ### from a given stratum. This creates NaN for variance and covariance stuff.
    ### Get rid of them.
    ###
    if not ("ID" in df.columns):
        df["ID"] = df.index

    count_df = df.groupby(by=stratum_col).count().reset_index()
    alone_stratums_ = list(count_df[count_df["ID"] == 1][stratum_col])
    df = df[~(df[stratum_col].isin(alone_stratums_))]

    df.sort_values(by=[stratum_col], inplace=True)
    df.reset_index(drop=True, inplace=True)

    stratas = df[stratum_col].unique()
    cols = [stratum_col, "parameters"] + variable_cols
    meanVarCovar_df = pd.DataFrame(columns=cols, index=range(len(stratas) * 3))

    # fill in the strarum column
    # since we want mean, var, and covar, we put 3 in line below
    meanVarCovar_df[stratum_col] = np.repeat(stratas, 3)

    param_list = ["mean", "var", "covar"]
    meanVarCovar_df["parameters"] = np.stack((param_list) * len(stratas), axis=0)

    # populate the table:
    mean_df = df.groupby(by=stratum_col)[variable_cols].mean().reset_index()
    replacement = mean_df.loc[:, variable_cols].values
    meanVarCovar_df.loc[
        meanVarCovar_df["parameters"] == "mean", variable_cols
    ] = replacement

    #
    var_df = df.groupby(by=stratum_col)[variable_cols].std().reset_index()
    var_df[variable_cols] = var_df[variable_cols] ** 2
    replacement = var_df.loc[:, variable_cols].values
    meanVarCovar_df.loc[
        meanVarCovar_df["parameters"] == "var", variable_cols
    ] = replacement

    for curr_cov_var in cov_variables:
        cov_df = (
            df.groupby(by=stratum_col)[[curr_cov_var[0], curr_cov_var[1]]]
            .cov()
            .reset_index()
        )
        # we need one of the off-diagonal entries
        # so, we drop first column and every other row

        # cov_df = cov_df.loc[cov_df.index[::2,]].copy() # or the following line the same thing

        cov_df = cov_df[cov_df["level_1"] == curr_cov_var[0]].copy()
        # copy values of off diagonal into diagonal as well
        # just for simplifying the process of populating of
        # main dataframe
        cov_df[curr_cov_var[0]] = cov_df[curr_cov_var[1]]

        replacement = cov_df[curr_cov_var].values
        meanVarCovar_df.loc[
            meanVarCovar_df["parameters"] == "covar", curr_cov_var
        ] = replacement

    for a_col in meanVarCovar_df.columns[2:]:
        meanVarCovar_df[a_col] = meanVarCovar_df[a_col].astype(np.float64)

    return meanVarCovar_df


def number_of_strata(test_df, m_dict, IDs_dictionary, area_df):
    """
    Author : Amin Norouzi Kandelati

    Arguments
    ---------
    test_df : dataframe
        This dataframe includes

    m_dict : dictionary
        Master dictionary that includes

    IDs_dictionary : dictionary

    area_df : dataframe
        includes area of all fields of the same crop?

    Returns
    ---------
    m_dict : dictionary
         Adds X, Y, Z to the m_dict

    """
    # Numbers of strata 2
    for cropType in test_df["CropTyp"].unique():
        cropType_subset = {
            key: value for key, value in IDs_dictionary.items() if key[1] == cropType
        }
        A_n_star_h_list = [
            value[2] for key, values in cropType_subset.items() for value in values
        ]
        A_n_star_h = sum(A_n_star_h_list)

        # Now use .at to access the specific value
        # idx = area_df[area_df["CropTyp"] == cropType].index[0]
        # A_N_star_h = area_df.at[idx, "denom_acr"]
        # N_star_h = area_df.at[idx, "denom"]
        A_N_star_h = area_df.loc[area_df["CropTyp"] == cropType, "denom_acr"].values[0]
        N_star_h = area_df.loc[area_df["CropTyp"] == cropType, "denom"].values[0]

        m_dict[(cropType, "n_star_h")].append(len(A_n_star_h_list))
        m_dict[(cropType, "A_n_star_h")].append(A_n_star_h)
        m_dict[(cropType, "A_N_star_h")].append(A_N_star_h)
        m_dict[(cropType, "N_star_h")].append(N_star_h)
    return m_dict


def numer_sum_for_acc_intervals(numer_strata_list, m_dict, numer_dict):
    """
    Author : Amin Norouzi Kandelati
    """
    for strata in np.unique(np.array(numer_strata_list)):
        strata_subset = {
            key: value for key, value in numer_dict.items() if key[1] == strata
        }

        A_yu_list = [
            value[2]
            for key, values in strata_subset.items()
            for value in values
            if key[0][0] == key[0][1]
        ]
        yu_IDs = np.array(
            [
                value[0]
                for key, values in strata_subset.items()
                for value in values
                if key[0][0] == key[0][1]
            ]
        )
        A_yu = sum(A_yu_list)

        # Sample variance (based on counts not area)
        y_bar_h_count = len(A_yu_list) / m_dict[(strata, "n_star_h")][0]
        sy_h_2 = (len(A_yu_list) - y_bar_h_count) ** 2 / m_dict[(strata, "n_star_h")][0]

        m_dict[(strata, "n_yu")].append(len(A_yu_list))
        m_dict[(strata, "yu_IDs")].append(yu_IDs)
        m_dict[(strata, "y_bar_h")].append(A_yu / m_dict[(strata, "A_n_star_h")][0])
        m_dict[(strata, "y_bar_h_count")].append(y_bar_h_count)
        m_dict[(strata, "sy_h_2")].append(sy_h_2)
        m_dict[(strata, "Y_bar")].append(
            m_dict[(strata, "A_N_star_h")][0] * m_dict[(strata, "y_bar_h")][0]
        )
    return m_dict


def denom_sum_for_acc_intervals(denom_strata_list, m_dict, denom_dictionary):
    """
    Author : Amin Norouzi Kandelati
    """
    for strata in np.unique(np.array(denom_strata_list)):
        strata_subset = {
            key: value for key, value in denom_dictionary.items() if key[1] == strata
        }
        A_xu_list = [
            value[2] for key, values in strata_subset.items() for value in values
        ]
        xu_IDs = np.array(
            [value[0] for key, values in strata_subset.items() for value in values]
        )
        A_xu = sum(A_xu_list)

        # Sample variance (based on counts not area)
        x_bar_h_count = len(A_xu_list) / m_dict[(strata, "n_star_h")][0]
        sx_h_2 = (len(A_xu_list) - x_bar_h_count) ** 2 / m_dict[(strata, "n_star_h")][0]

        m_dict[(strata, "n_xu")].append(len(A_xu_list))
        m_dict[(strata, "xu_IDs")].append(xu_IDs)
        m_dict[(strata, "x_bar_h")].append(A_xu / m_dict[(strata, "A_n_star_h")][0])
        m_dict[(strata, "x_bar_h_count")].append(x_bar_h_count)
        m_dict[(strata, "sx_h_2")].append(sx_h_2)
        m_dict[(strata, "X_bar")].append(
            m_dict[(strata, "A_N_star_h")][0] * m_dict[(strata, "x_bar_h")][0]
        )
    return m_dict


def user_acc_variance(UAV_df, user_accuracy):
    """
    Author : Amin Norouzi Kandelati
    """
    v_sum_list = []
    for strata in UAV_df["strata"].unique():  # 5
        A_N_star_h = UAV_df.loc[UAV_df["strata"] == strata, "A_N_star_h"].values[0]
        A_n_star_h = UAV_df.loc[UAV_df["strata"] == strata, "A_n_star_h"].values[0]
        sy_h_2 = UAV_df.loc[UAV_df["strata"] == strata, "sy_h_2"].values[0]
        sx_h_2 = UAV_df.loc[UAV_df["strata"] == strata, "sx_h_2"].values[0]
        s_xy_h = UAV_df.loc[UAV_df["strata"] == strata, "s_xy_h"].values[0]

        v_sum_list.append(
            A_N_star_h**2
            * (1 - A_n_star_h / A_N_star_h)
            * (sy_h_2 + user_accuracy**2 * sx_h_2 - 2 * user_accuracy * s_xy_h)
            / A_n_star_h
        )
    return v_sum_list


def s_xy_h_func(m_df):
    """
    Author : Amin Norouzi Kandelati
    """
    for strata in m_df["strata"].unique():  # 4
        yu = m_df.loc[m_df["strata"] == strata, "yu_0_1"].values[0]
        xu = m_df.loc[m_df["strata"] == strata, "xu_0_1"].values[0]
        ybar_h = m_df.loc[m_df["strata"] == strata, "y_bar_h_count"].values[0]
        xbar_h = m_df.loc[m_df["strata"] == strata, "x_bar_h_count"].values[0]
        n_star_h = m_df.loc[m_df["strata"] == strata, "n_star_h"].values[0]

        s_xy_h = sum((yu - ybar_h) * (xu - xbar_h) / n_star_h - 1)
        m_df.loc[m_df["strata"] == strata, "s_xy_h"] = s_xy_h

        # Calculate X_hat
        A_N_star_h = m_df.loc[m_df["strata"] == strata, "A_N_star_h"].values[0]
        m_df.loc[m_df["strata"] == strata, "x_hat"] = A_N_star_h * xbar_h
    return m_df


def amin_UA_defaultdict_to_df(master_dictionary):
    """
    Author: Amin Norouzi Kandlati
    Extract all unique first and second keys

    Arguments
    ---------
    master_dictionary : dictionary
        This dictionary includes


    Returns
    -------
    """
    strata = set()
    columns = set()

    for key in master_dictionary.keys():
        strata.add(key[0])
        columns.add(key[1])

    # Convert to sorted lists
    strata = sorted(strata)
    columns = sorted(columns)

    # Create a list to store each row as a dictionary
    rows = []

    # Populate the list with rows
    for s in strata:
        row = {"strata": s}
        for c in columns:
            # Get the first value from the list or None if key doesn't exist
            # are there more than one value? why first value?
            row[c] = master_dictionary.get((s, c), [None])[0]
        rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows, columns=["strata"] + columns)
    return df


def regularize_a_field_annual_basis(
    a_df, V_idks="NDVI", interval_size=10, start_year=2008, end_year=2021
):
    """
    This is a modification of regularize_a_field() function.
    The update is that this function is "less flexible"!!!
    Here we will have intervals go from Jan 1-Jan 10, and so on.
    In other words, the time origin is Jan 1.

    In the regularize_a_field() function the origin of time
    was the first data point! So, we ended up having 36 data for
    some years and 37 for some other when we were looking at 3 years of data!
    The root cause was that ML was not part of the plan.... same old, same old!
    """
    """Returns a dataframe where data points are interval_size-day apart.
       This function regularizes the data between the minimum and maximum dates
       present in the data. 

    Arguments
    ---------
    a_df : dataframe 
           of a given field for only one satellite

    Returns
    -------
    regularized_df : dataframe
    """
    if not ("human_system_start_time" in a_df.columns):
        a_df = add_human_start_time_by_system_start_time(a_df)

    a_df["human_system_start_time"] = pd.to_datetime(a_df["human_system_start_time"])
    a_df.sort_values(by="human_system_start_time", inplace=True)
    a_df.reset_index(drop=True, inplace=True)

    assert len(a_df.ID.unique()) == 1
    # assert (len(a_df.dataset.unique()) == 1)
    #
    # see how many days there are between the first and last image
    #
    a_df_coverage_days = (
        max(a_df.human_system_start_time) - min(a_df.human_system_start_time)
    ).days
    assert a_df_coverage_days >= interval_size

    # see how many data points we need.
    all_years = sorted(a_df.human_system_start_time.dt.year.unique())
    no_steps_per_year = 365 // interval_size
    no_steps = len(all_years) * no_steps_per_year

    """
       I am reducing the flexibility of the code we had before!
       I want to make it that all fields have the same exact dates
       for their time steps. Jan. 1, Jan 10, ...
    """
    regular_time_stamps = []
    for a_year in all_years:
        regular_time_stamps = regular_time_stamps + list(
            pd.date_range(
                pd.Timestamp(str(a_year) + "-01-01"),
                pd.Timestamp(str(a_year) + "-12-25"),
                freq=str(interval_size) + "D",
            )
        )

    # initialize output dataframe
    if "dataset" in a_df.columns:
        regular_cols = ["ID", "dataset", "human_system_start_time", V_idks]
    else:
        regular_cols = ["ID", "human_system_start_time", V_idks]

    regular_df = pd.DataFrame(
        data=None, index=np.arange(no_steps), columns=regular_cols
    )

    regular_df["ID"] = a_df.ID.unique()[0]
    if "dataset" in a_df.columns:
        regular_df["dataset"] = a_df.dataset.unique()[0]

    if len(regular_time_stamps) == no_steps + 1:
        regular_df.human_system_start_time = regular_time_stamps[:-1]
    elif len(regular_time_stamps) == no_steps:
        regular_df.human_system_start_time = regular_time_stamps
    else:
        raise ValueError(
            f"There is a mismatch between no. days needed and '{interval_size}-day' interval array!"
        )

    # Pick the maximum of every interval_size-days
    for start_date in regular_df.human_system_start_time:
        """
        The following will crate an array (of length 2)
        it goes from a day to 10 days later; end points of the interval_size-day interval.

              # Here we add 1 day to the right end point (end_date)
        because the way pandas/python slices the dataframe;
        does not include the last row of sub-dataframe
        """
        dateRange = pd.date_range(
            start_date,
            start_date + pd.Timedelta(days=interval_size - 1),
            freq=str(1) + "D",
        )
        assert len(dateRange) == interval_size

        curr_time_window = a_df[a_df.human_system_start_time.isin(dateRange)]
        if len(curr_time_window) == 0:
            regular_df.loc[
                regular_df.human_system_start_time == start_date, V_idks
            ] = -1.5
        else:
            regular_df.loc[
                regular_df.human_system_start_time == start_date, V_idks
            ] = max(curr_time_window[V_idks])
    ##### end the for-loop
    regular_df.reset_index(drop=True, inplace=True)
    return regular_df


def create_calendar_table(SF_year):
    start = str(SF_year) + "-01-01"
    end = str(SF_year) + "-12-31"

    df = pd.DataFrame({"human_system_start_time": pd.date_range(start, end)})

    # add day of year
    df["doy"] = 1 + np.arange(len(df))

    # df['Weekday'] = df['Date'].dt.day_name()

    # Drop the last element if the year is leap-year.
    # we want the data to have equal size
    if len(df) == 366:
        df.drop(index=365, axis=0, inplace=True)
    return df


def filter_out_NASS(dt_df):
    dt_cf_NASS = dt_df.copy()
    dt_cf_NASS["DataSrc"] = dt_cf_NASS["DataSrc"].astype(str)
    dt_cf_NASS["DataSrc"] = dt_cf_NASS["DataSrc"].str.lower()

    dt_cf_NASS = dt_cf_NASS[~dt_cf_NASS["DataSrc"].str.contains("nass")]
    return dt_cf_NASS


def filter_by_lastSurvey(dt_df_su, year):
    dt_surv = dt_df_su.copy()
    dt_surv = dt_surv[dt_surv["LstSrvD"].str.contains(str(year))]
    return dt_surv


def filter_out_nonIrrigated(dt_df_irr):
    dt_irrig = dt_df_irr.copy()
    #
    # drop NA rows in irrigation column
    #
    dt_irrig.dropna(subset=["Irrigtn"], inplace=True)

    dt_irrig["Irrigtn"] = dt_irrig["Irrigtn"].astype(str)

    dt_irrig["Irrigtn"] = dt_irrig["Irrigtn"].str.lower()
    dt_irrig = dt_irrig[~dt_irrig["Irrigtn"].str.contains("none")]
    dt_irrig = dt_irrig[~dt_irrig["Irrigtn"].str.contains("unknown")]
    dt_irrig = dt_irrig[~dt_irrig["Irrigtn"].str.contains("empty")]

    return dt_irrig


def Null_SOS_EOS_by_DoYDiff(pd_TS, min_season_length=40):
    """
    input: pd_TS is a pandas dataframe
           it includes a column SOS and a column EOS

    output: create a vector that measures distance between DoY
            of an SOS and corresponding EOS.

    It is possible that the number of one of the SOS and EOS is
    different from the other. (perhaps just by 1)

    So, we need to keep that in mind.
    """
    pd_TS_DoYDiff = pd_TS.copy()

    # find indexes of SOS and EOS
    SOS_indexes = pd_TS_DoYDiff.index[pd_TS_DoYDiff["SOS"] != 0].tolist()
    EOS_indexes = pd_TS_DoYDiff.index[pd_TS_DoYDiff["EOS"] != 0].tolist()

    """
    It seems it is possible to only have 1 SOS with no EOS. (or vice versa).
    In this case we can consider we only have 1 season!
    """
    """
    We had the following in the code, which is fine for computing 
    the tables (since we count the seasons by counting SOS), but, if 
    there is no SOS and only 1 EOS, then the EOS will not be nullified. and will show
    up in the plots.

    if len(SOS_indexes) == 0 or len(EOS_indexes) == 0:
        return pd_TS_DoYDiff
    """
    # if len(SOS_indexes) == 0 or len(EOS_indexes) == 0:
    #     return pd_TS_DoYDiff

    if len(SOS_indexes) == 0:
        if len(EOS_indexes) == 0:
            return pd_TS_DoYDiff
        else:
            if len(EOS_indexes) == 1:
                EOS_indexes[0] = 0
                pd_TS_DoYDiff.EOS = 0
                return pd_TS_DoYDiff

            else:
                raise ValueError("too many EOS and no SOS whatsoever!")

    if len(EOS_indexes) == 0:
        if len(SOS_indexes) == 1:
            return pd_TS_DoYDiff
        else:
            raise ValueError("too many SOS and no EOS whatsoever!")

    SOS_indexes = pd_TS_DoYDiff.index[pd_TS_DoYDiff["SOS"] != 0].tolist()
    EOS_indexes = pd_TS_DoYDiff.index[pd_TS_DoYDiff["EOS"] != 0].tolist()

    """
    First we need to fix the prolems such as having 2 SOS and only 1 EOS, or,
                                                    2 EOS and only 1 SOS, or,
    it is possible that number of SOSs and number of EOSs are identical,
    but the plot starts with EOS and ends with SOS.
    """
    #
    # Check if first EOS is less than first SOS
    #
    SOS_pointer = SOS_indexes[0]
    EOS_pointer = EOS_indexes[0]
    if (
        pd_TS_DoYDiff.loc[EOS_pointer, "human_system_start_time"]
        < pd_TS_DoYDiff.loc[SOS_pointer, "human_system_start_time"]
    ):
        # Remove the false EOS from dataFrame
        pd_TS_DoYDiff.loc[EOS_pointer, "EOS"] = 0

        # remove the first element of EOS indexes
        EOS_indexes.pop(0)

    #
    # Check if last SOS is greater than last EOS
    #
    if len(EOS_indexes) == 0 or len(EOS_indexes) == 0:
        return pd_TS_DoYDiff

    SOS_pointer = SOS_indexes[-1]
    EOS_pointer = EOS_indexes[-1]
    if (
        pd_TS_DoYDiff.loc[EOS_pointer, "human_system_start_time"]
        < pd_TS_DoYDiff.loc[SOS_pointer, "human_system_start_time"]
    ):
        # Remove the false EOS from dataFrame
        pd_TS_DoYDiff.loc[SOS_pointer, "SOS"] = 0

        # remove the first element of EOS indexes
        SOS_indexes.pop()

    if len(SOS_indexes) != len(EOS_indexes):
        #
        # in this case we have an extra SOS (at the end) or EOS (at the beginning)
        #
        print("Error occured at {}.".format(pd_TS.ID.unique()[0]))
        # print (pd_TS.image_year.unique()[0])
        raise ValueError("SOS and EOS are not of the same length.")

    """
    Go through seasons and invalidate them if their length is too short
    """
    for ii in np.arange(len(SOS_indexes)):
        SOS_pointer = SOS_indexes[ii]
        EOS_pointer = EOS_indexes[ii]

        current_growing_season_Length = (
            pd_TS_DoYDiff.loc[EOS_pointer, "human_system_start_time"]
            - pd_TS_DoYDiff.loc[SOS_pointer, "human_system_start_time"]
        ).days

        #  Kill/invalidate season if its length is too short.
        if current_growing_season_Length < min_season_length:
            pd_TS_DoYDiff.loc[SOS_pointer, "SOS"] = 0
            pd_TS_DoYDiff.loc[EOS_pointer, "EOS"] = 0

    return pd_TS_DoYDiff


def addToDF_SOS_EOS_White(pd_TS, VegIdx="EVI", onset_thresh=0.3, offset_thresh=0.3):
    """
    In this methods the NDVI_Ratio = (NDVI - NDVI_min) / (NDVI_Max - NDVI_min)
    is computed.

    SOS or onset is when NDVI_ratio exceeds onset-threshold
    and EOS is when NDVI_ratio drops below off-set-threshold.
    """
    pandaFrame = pd_TS.copy()

    VegIdx_min = pandaFrame[VegIdx].min()
    VegIdx_max = pandaFrame[VegIdx].max()
    VegRange = VegIdx_max - VegIdx_min + sys.float_info.epsilon

    colName = VegIdx + "_ratio"
    pandaFrame[colName] = (pandaFrame[VegIdx] - VegIdx_min) / VegRange

    # if (onset_thresh == offset_thresh):
    #     SOS_EOS_candidates = pandaFrame[colName] - onset_thresh
    #     sign_change = find_signChange_locs_EqualOnOffset(SOS_EOS_candidates.values)
    # else:
    #     SOS_candidates = pandaFrame[colName] - onset_thresh
    #     EOS_candidates = offset_thresh - pandaFrame[colName]
    #     sign_change = find_signChange_locs_DifferentOnOffset(SOS_candidates.values, EOS_candidates.values)
    # pandaFrame['SOS_EOS'] = sign_change * pandaFrame[VegIdx]

    SOS_candidates = pandaFrame[colName] - onset_thresh
    EOS_candidates = offset_thresh - pandaFrame[colName]

    BOS, EOS = find_signChange_locs_DifferentOnOffset(SOS_candidates, EOS_candidates)
    pandaFrame["SOS"] = BOS * pandaFrame[VegIdx]
    pandaFrame["EOS"] = EOS * pandaFrame[VegIdx]

    return pandaFrame


def find_signChange_locs_DifferentOnOffset(SOS_candids, EOS_candids):
    if type(SOS_candids) != np.ndarray:
        SOS_candids = SOS_candids.values

    if type(EOS_candids) != np.ndarray:
        EOS_candids = EOS_candids.values

    SOS_sign_change = np.zeros(len(SOS_candids))
    EOS_sign_change = np.zeros(len(EOS_candids))

    pointer = 0
    for pointer in np.arange(0, len(SOS_candids) - 1):
        """
        On Feb. 23, 2023 we came upon a rare case where SOS_candids[pointer+1] was exactly zero!
        So, we changed the line if SOS_candids[pointer+1] >0: to if SOS_candids[pointer+1] >= 0:.
        """

        if SOS_candids[pointer] < 0:
            if SOS_candids[pointer + 1] >= 0:
                # if SOS_candids[pointer]*SOS_candids[pointer+1]<=0:
                SOS_sign_change[pointer + 1] = 1

        if EOS_candids[pointer] < 0:
            if EOS_candids[pointer + 1] >= 0:
                # if EOS_candids[pointer]*EOS_candids[pointer+1]<=0:
                EOS_sign_change[pointer + 1] = 1

    # sign_change = SOS_sign_change + EOS_sign_change
    return (SOS_sign_change, EOS_sign_change)


def correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie, give_col, maxjump_perDay=0.015):
    """
    This is a modified version of correct_big_jumps_1DaySeries()
    Here if the big jumps happen in Dec. Jan, or Feb. we take the high value down
    (as opposed to lower value up)

    Returns: a dataframe with no big jumps in it
    Arguments
    ---------
    dataTMS_jumpie : dataframe
        A dataframe in which

    give_col : String
        A string indicating which column/VI should be filled in.

    Returns
    -------
    dataTMS_jumpie : dataframe
        the same dataframe with no big jumps! (just one iteration)
    """
    # dataTMS_jumpie = initial_clean(df = dataTMS_jumpie, column_to_be_cleaned = give_col)

    dataTMS_jumpie["human_system_start_time"] = pd.to_datetime(
        dataTMS_jumpie["human_system_start_time"]
    )
    dataTMS_jumpie.sort_values(by=["human_system_start_time"], inplace=True)
    dataTMS_jumpie.reset_index(drop=True, inplace=True)

    thyme_vec = dataTMS_jumpie["human_system_start_time"].values.copy()
    Veg_indks = dataTMS_jumpie[give_col].values.copy()

    time_diff = (
        pd.to_datetime(thyme_vec[1:])
        - pd.to_datetime(thyme_vec[0 : len(thyme_vec) - 1])[0]
    ).days

    Veg_indks_diff = Veg_indks[1:] - Veg_indks[0 : len(thyme_vec) - 1]
    jump_indexes = np.where(Veg_indks_diff > maxjump_perDay)
    jump_indexes = jump_indexes[0]
    jump_indexes = jump_indexes.tolist()

    thyme_vec = dataTMS_jumpie["human_system_start_time"].values.copy()
    Veg_indks = dataTMS_jumpie[give_col].values.copy()
    time_diff = thyme_vec[1:] - thyme_vec[0 : len(thyme_vec) - 1]

    # time_diff_in_days = time_diff / 86400
    time_diff_in_days = time_diff.astype("timedelta64[D]")
    time_diff_in_days = time_diff_in_days.astype(int)

    # It is possible that the very first one has a big jump in it.
    # we cannot interpolate this. so, lets just skip it.
    if len(jump_indexes) > 0:
        if jump_indexes[0] == 0:
            jump_indexes.pop(0)

    if len(jump_indexes) > 0:
        for jp_idx in jump_indexes:
            # for count, jp_idx in enumerate(jump_indexes):
            # Veg_indks_diff >= (time_diff_in_days * maxjump_perDay)
            if Veg_indks_diff[jp_idx] >= (time_diff_in_days[jp_idx] * maxjump_perDay):
                #
                # form a line using the adjacent points of the big jump:
                #
                if pd.to_datetime(thyme_vec[jp_idx]).month in [1, 2, 12]:
                    # take the big value down in Jan, Feb, or Dec.
                    """
                    It is possible that the big jump is the last data point.
                    In this case, let it go!
                    Perhaps we can do this in a faster way: remove the indices from jump_indexes
                    above. rather than checking the if statement below. Or maybe not?
                    We have 2 cases here: Jan-Feb-Dec and other months!
                    """
                    if (jp_idx + 2) < len(Veg_indks):
                        x1, y1 = thyme_vec[jp_idx], Veg_indks[jp_idx]
                        x2, y2 = thyme_vec[jp_idx + 2], Veg_indks[jp_idx + 2]

                        m = float(y2 - y1) / (x2 - x1).astype(
                            pd.Timedelta
                        )  # slope or float(x2-x1)
                        b = y2 - (m * int(x2))  # intercept

                        # replace the big jump with linear interpolation
                        # only if the new value is smaller that it was in the raw
                        new_val = m * thyme_vec[jp_idx + 1].astype(int) + b
                        if new_val < Veg_indks[jp_idx + 1]:
                            Veg_indks[jp_idx + 1] = new_val
                else:
                    """
                    It is possible that the big jump is the last data point.
                    or first one. In these cases let it go!!!
                    """
                    if (jp_idx + 1) < len(Veg_indks) and (jp_idx - 1) >= 0:
                        # take the low value upper, in !(Jan, Feb, Dec)
                        x1, y1 = thyme_vec[jp_idx - 1], Veg_indks[jp_idx - 1]
                        x2, y2 = thyme_vec[jp_idx + 1], Veg_indks[jp_idx + 1]

                        m = float(y2 - y1) / (x2 - x1).astype(
                            pd.Timedelta
                        )  # slope or float(x2-x1)
                        b = y2 - (m * int(x2))  # intercept

                        # replace the big jump with linear interpolation
                        Veg_indks[jp_idx] = m * thyme_vec[jp_idx].astype(int) + b

    dataTMS_jumpie[give_col] = Veg_indks
    return dataTMS_jumpie


def correct_big_jumps_1DaySeries(dataTMS_jumpie, give_col, maxjump_perDay=0.015):
    """
    in the function correct_big_jumps_preDefinedJumpDays(.) we have
    to define the jump_amount and the no_days_between_points.
    For example if we have a jump more than 0.4 in less than 20 dats, then
    that is an outlier detected.

    Here we modify the approach to be flexible in the following sense:
    if the amount of increase in NDVI is more than #_of_Days * 0.02 then
    an outlier is detected and we need interpolation.

    0.015 came from the SG based paper that used 0.4 jump in NDVI for 20 days.
    That translates into 0.02 = 0.4 / 20 per day.
    But we did choose 0.015 as default
    """
    dataTMS_jumpie = initial_clean(df=dataTMS_jumpie, column_to_be_cleaned=give_col)

    dataTMS_jumpie["human_system_start_time"] = pd.to_datetime(
        dataTMS_jumpie["human_system_start_time"]
    )
    dataTMS_jumpie.sort_values(by=["human_system_start_time"], inplace=True)
    dataTMS_jumpie.reset_index(drop=True, inplace=True)

    thyme_vec = dataTMS_jumpie["human_system_start_time"].values.copy()
    Veg_indks = dataTMS_jumpie[give_col].values.copy()

    time_diff = (
        pd.to_datetime(thyme_vec[1:])
        - pd.to_datetime(thyme_vec[0 : len(thyme_vec) - 1])[0]
    ).days

    Veg_indks_diff = Veg_indks[1:] - Veg_indks[0 : len(thyme_vec) - 1]
    jump_indexes = np.where(Veg_indks_diff > maxjump_perDay)
    jump_indexes = jump_indexes[0]
    jump_indexes = jump_indexes.tolist()

    thyme_vec = dataTMS_jumpie["human_system_start_time"].values.copy()
    Veg_indks = dataTMS_jumpie[give_col].values.copy()
    time_diff = thyme_vec[1:] - thyme_vec[0 : len(thyme_vec) - 1]

    # time_diff_in_days = time_diff / 86400
    time_diff_in_days = time_diff.astype("timedelta64[D]")
    time_diff_in_days = time_diff_in_days.astype(int)

    # It is possible that the very first one has a big jump in it.
    # we cannot interpolate this. so, lets just skip it.
    if len(jump_indexes) > 0:
        if jump_indexes[0] == 0:
            jump_indexes.pop(0)
    if len(jump_indexes) > 0:
        for jp_idx in jump_indexes:
            if Veg_indks_diff[jp_idx] >= (time_diff_in_days[jp_idx] * maxjump_perDay):
                #
                # form a line using the adjacent points of the big jump:
                #
                x1, y1 = thyme_vec[jp_idx - 1], Veg_indks[jp_idx - 1]
                x2, y2 = thyme_vec[jp_idx + 1], Veg_indks[jp_idx + 1]
                if (x2 - x1).astype(pd.Timedelta) == 0:
                    print(jp_idx)
                m = float(y2 - y1) / (x2 - x1).astype(
                    pd.Timedelta
                )  # slope or float(x2-x1)
                b = y2 - (m * int(x2))  # intercept

                # replace the big jump with linear interpolation
                Veg_indks[jp_idx] = m * thyme_vec[jp_idx].astype(int) + b

    dataTMS_jumpie[give_col] = Veg_indks
    return dataTMS_jumpie


def interpolate_outliers_EVI_NDVI(outlier_input, given_col):
    """
    outliers are those that are beyond boundaries. For example and EVI value of 2.
    Big jump in the other function means we have a big jump but we are still
    within the region of EVI values. If in 20 days we have a jump of 0.3 then that is noise.

    in 2017 data I did not see outlier in NDVI. It only happened in EVI.
    """
    outlier_input = initial_clean(df=outlier_input, column_to_be_cleaned=given_col)

    outlier_input["human_system_start_time"] = pd.to_datetime(
        outlier_input["human_system_start_time"]
    )
    assert len(outlier_input.ID.unique()) == 1

    # ID below is for sanity check. otherwise the input must be one field
    outlier_input.sort_values(by=["ID", "human_system_start_time"], inplace=True)
    outlier_input.reset_index(drop=True, inplace=True)

    # 1st block
    time_vec = outlier_input["human_system_start_time"].values.copy()
    vec = outlier_input[given_col].values.copy()

    # find out where are outliers
    high_outlier_inds = np.where(vec > 1)[0]
    low_outlier_inds = np.where(vec < -1)[0]

    all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
    all_outliers_idx = np.sort(all_outliers_idx)
    non_outiers = np.arange(len(vec))[~np.in1d(np.arange(len(vec)), all_outliers_idx)]

    # 2nd block
    if len(all_outliers_idx) == 0:
        return outlier_input

    """
    it is possible that for a field we only have x=2 data points
    where all the EVI/NDVI is outlier. Then, there is nothing to 
    use for interpolation. So, we return an empty datatable
    """
    if len(all_outliers_idx) == len(outlier_input):
        outlier_input = initial_clean(df=outlier_input, column_to_be_cleaned=given_col)
        outlier_input = outlier_input[outlier_input[given_col] < 1.5]
        outlier_input = outlier_input[outlier_input[given_col] > -1.5]
        return outlier_input

    # 3rd block

    # Get rid of outliers that are at the beginning of the time series
    # if len(non_outiers) > 0 :
    if non_outiers[0] > 0:
        vec[0 : non_outiers[0]] = vec[non_outiers[0]]

        # find out where are outliers
        high_outlier_inds = np.where(vec > 1)[0]
        low_outlier_inds = np.where(vec < -1)[0]

        all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
        all_outliers_idx = np.sort(all_outliers_idx)
        non_outiers = np.arange(len(vec))[
            ~np.in1d(np.arange(len(vec)), all_outliers_idx)
        ]
        if len(all_outliers_idx) == 0:
            outlier_input[given_col] = vec
            return outlier_input

    # 4th block
    # Get rid of outliers that are at the end of the time series
    if non_outiers[-1] < (len(vec) - 1):
        vec[non_outiers[-1] :] = vec[non_outiers[-1]]

        # find out where are outliers
        high_outlier_inds = np.where(vec > 1)[0]
        low_outlier_inds = np.where(vec < -1)[0]

        all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
        all_outliers_idx = np.sort(all_outliers_idx)
        non_outiers = np.arange(len(vec))[
            ~np.in1d(np.arange(len(vec)), all_outliers_idx)
        ]
        if len(all_outliers_idx) == 0:
            outlier_input[given_col] = vec
            return outlier_input
    """
    At this point outliers are in the middle of the vector
    and beginning and the end of the vector are clear.
    """
    for out_idx in all_outliers_idx:
        """
        Right here at the beginning we should check
        if vec[out_idx] is outlier or not. The reason is that
        there might be consecutive outliers at position m and m+1
        and we fix the one at m+1 when we are fixing m ...
        """
        # if ~(vec[out_idx] <= 1 and vec[out_idx] >= -1):
        if vec[out_idx] >= 1 or vec[out_idx] <= -1:
            left_pointer = out_idx - 1
            right_pointer = out_idx + 1
            while ~(vec[right_pointer] <= 1 and vec[right_pointer] >= -1):
                right_pointer += 1

            # form the line and fill in the outlier valies
            x1, y1 = time_vec[left_pointer], vec[left_pointer]
            x2, y2 = time_vec[right_pointer], vec[right_pointer]

            time_diff = x2 - x1
            y_diff = y2 - y1

            slope = y_diff / time_diff.astype(pd.Timedelta)
            intercept = y2 - (slope * int(x2))
            vec[left_pointer + 1 : right_pointer] = (
                slope * ((time_vec[left_pointer + 1 : right_pointer]).astype(int))
                + intercept
            )
    outlier_input[given_col] = vec
    return outlier_input


def initial_clean(df, column_to_be_cleaned):
    #     dt_copy = df.copy()
    # remove the useles system:index column
    if "system:index" in list(df.columns):
        df = df.drop(columns=["system:index"])
    df.drop_duplicates(inplace=True)

    if "human_system_start_time" in df.columns:
        df["human_system_start_time"] = pd.to_datetime(df["human_system_start_time"])

    # Drop rows whith NA in column_to_be_cleaned column.
    df = df[df[column_to_be_cleaned].notna()]

    if column_to_be_cleaned in ["NDVI", "EVI"]:
        df.loc[df[column_to_be_cleaned] > 1, column_to_be_cleaned] = 1.5
        df.loc[df[column_to_be_cleaned] < -1, column_to_be_cleaned] = -1.5
    return df


def fill_theGap_linearLine(a_regularized_TS, V_idx="NDVI"):
    """Returns a dataframe that has replaced the missing parts of regular_TS.

    Arguments
    ---------
    regular_TS : dataframe
        A regularized (data points are squidistant from each other) dataframe
        with missing data points; -1.5 is indication of missing values.
        This dataframe is the output of the function regularize_a_field(.)
        We will assume the regular_TS is for a given unique field from a given unique satellite.

    V_idx : String
        A string indicating which column/VI should be filled in.

    Returns
    -------
    regular_TS : dataframe
        the same dataframe with missing data points filled in by linear interpolation
    """
    # a_regularized_TS = regular_TS.copy()

    a_regularized_TS["human_system_start_time"] = pd.to_datetime(
        a_regularized_TS["human_system_start_time"]
    )
    TS_array = a_regularized_TS[V_idx].copy().values

    # aaa = a_regularized_TS["human_system_start_time"].values[1]
    # bbb = a_regularized_TS["human_system_start_time"].values[0]
    # time_step_size = (aaa - bbb).astype('timedelta64[D]')/np.timedelta64(1, 'D')

    """
    -1.5 is an indicator of missing values, i.e. a gap.
    The -1.5 was used as indicator in the function regularize_movingWindow_windowSteps_2Yrs()
    """
    missing_indicies = np.where(TS_array == -1.5)[0]
    Notmissing_indicies = np.where(TS_array != -1.5)[0]

    #
    #    Check if the first or last k values are missing
    #    if so, replace them with proper number and shorten the task
    #
    left_pointer = Notmissing_indicies[0]
    right_pointer = Notmissing_indicies[-1]

    if left_pointer > 0:
        TS_array[:left_pointer] = TS_array[left_pointer]

    if right_pointer < (len(TS_array) - 1):
        TS_array[right_pointer:] = TS_array[right_pointer]
    #
    # update indexes.
    #
    missing_indicies = np.where(TS_array == -1.5)[0]
    Notmissing_indicies = np.where(TS_array != -1.5)[0]

    # left_pointer = Notmissing_indicies[0]
    stop = right_pointer
    right_pointer = left_pointer + 1

    missing_indicies = np.where(TS_array == -1.5)[0]

    while len(missing_indicies) > 0:
        left_pointer = missing_indicies[0] - 1
        left_value = TS_array[left_pointer]
        right_pointer = missing_indicies[0]

        while TS_array[right_pointer] == -1.5:
            right_pointer += 1
        right_value = TS_array[right_pointer]

        if (right_pointer - left_pointer) == 2:
            # if there is a single gap, then we have just average of the
            # values
            # Avoid extra computation!
            #
            TS_array[left_pointer + 1] = 0.5 * (
                TS_array[left_pointer] + TS_array[right_pointer]
            )
            missing_indicies = np.where(TS_array == -1.5)[0]
        else:
            # form y = ax + b
            # to see what the "x_axis" was look at the same function in remote_sensing_core.py

            # denom = (x_axis[right_pointer]-x_axis[left_pointer]).astype('timedelta64[D]')/ \
            # np.timedelta64(int(time_step_size), 'D')

            denom = right_pointer - left_pointer
            slope = (right_value - left_value) / denom

            # b = right_value - (slope * x_axis[right_pointer])
            # 150 is a random number below.
            # The thing that matters is the number of steps, not actual values on x-axis.
            # I did it this way to avoid dealing with timestamp values and figuring out its stuff
            # Stuff means both finding out the right script and relation of timestamp to day of year stuff.
            # We can just use right_pointer itself instead of 150!
            b = right_value - (slope * right_pointer)
            TS_array[left_pointer + 1 : right_pointer] = (
                slope * np.arange(right_pointer - denom + 1, right_pointer) + b
            )
            missing_indicies = np.where(TS_array == -1.5)[0]

    a_regularized_TS[V_idx] = TS_array
    return a_regularized_TS


def is_leap_year(year):
    """Determine whether a year is a leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def regularize_a_field(
    a_df, V_idks="NDVI", interval_size=10, start_year=2008, end_year=2021
):
    """Returns a dataframe where data points are interval_size-day apart.
       This function regularizes the data between the minimum and maximum dates
       present in the data.

    Arguments
    ---------
    a_df : dataframe
           of a given field for only one satellite

    Returns
    -------
    regularized_df : dataframe
    """
    if not ("human_system_start_time" in a_df.columns):
        a_df = add_human_start_time_by_system_start_time(a_df)

    a_df["human_system_start_time"] = pd.to_datetime(a_df["human_system_start_time"])
    a_df.sort_values(by="human_system_start_time", inplace=True)
    a_df.reset_index(drop=True, inplace=True)

    assert len(a_df.ID.unique()) == 1
    # assert (len(a_df.dataset.unique()) == 1)
    #
    # see how many days there are between the first and last image
    #
    a_df_coverage_days = (
        max(a_df.human_system_start_time) - min(a_df.human_system_start_time)
    ).days
    assert a_df_coverage_days >= interval_size

    # see how many data points we need in terms of interval_size-day intervals for a_df_coverage_days
    no_steps = a_df_coverage_days // interval_size

    # initialize output dataframe
    if "dataset" in a_df.columns:
        regular_cols = ["ID", "dataset", "human_system_start_time", V_idks]
    else:
        regular_cols = ["ID", "human_system_start_time", V_idks]

    regular_df = pd.DataFrame(
        data=None, index=np.arange(no_steps), columns=regular_cols
    )

    regular_df["ID"] = a_df.ID.unique()[0]
    if "dataset" in a_df.columns:
        regular_df["dataset"] = a_df.dataset.unique()[0]

    # the following is an array of time stamps where each entry is the beginning
    # of the interval_size-day period
    regular_time_stamps = pd.date_range(
        min(a_df.human_system_start_time),
        max(a_df.human_system_start_time),
        freq=str(interval_size) + "D",
    )

    if len(regular_time_stamps) == no_steps + 1:
        regular_df.human_system_start_time = regular_time_stamps[:-1]
    elif len(regular_time_stamps) == no_steps:
        regular_df.human_system_start_time = regular_time_stamps
    else:
        raise ValueError(
            f"There is a mismatch between no. days needed and '{interval_size}-day' interval array!"
        )

    # Pick the maximum of every interval_size-days
    # for row_or_count in np.arange(len(no_steps)-1):
    #     curr_time_window = a_df[a_df.human_system_start_time >= first_year_steps[row_or_count]]
    #     curr_time_window = curr_time_window[curr_time_window.doy < first_year_steps[row_or_count+1]]

    #     if len(curr_time_window)==0:
    #         regular_df.loc[row_or_count, V_idks] = -1.5
    #     else:
    #         regular_df.loc[row_or_count, V_idks] = max(curr_time_window[V_idks])

    #     regular_df.loc[row_or_count, 'image_year'] = curr_year
    #     regular_df.loc[row_or_count, 'doy'] = first_year_steps[row_or_count]

    for start_date in regular_df.human_system_start_time:
        """
        The following will crate an array (of length 2)
        it goes from a day to 10 days later; end points of the interval_size-day interval.

              # Here we add 1 day to the right end point (end_date)
        because the way pandas/python slices the dataframe;
        does not include the last row of sub-dataframe
        """
        dateRange = pd.date_range(
            start_date,
            start_date + pd.Timedelta(days=interval_size - 1),
            freq=str(1) + "D",
        )
        assert len(dateRange) == interval_size

        curr_time_window = a_df[a_df.human_system_start_time.isin(dateRange)]
        if len(curr_time_window) == 0:
            regular_df.loc[
                regular_df.human_system_start_time == start_date, V_idks
            ] = -1.5
        else:
            regular_df.loc[
                regular_df.human_system_start_time == start_date, V_idks
            ] = max(curr_time_window[V_idks])
    ##### end the for-loop

    ##
    ## Some days will be missing from the beginning and end of the whole time series.
    ##
    #         all_years = np.arange(start_year, end_year+1)
    #         leapyear_count = np.sum([is_leap_year(item) for item in all_years])
    #         total_no_days = (leapyear_count*366) + ((end_year - start_year + 1 - leapyear_count)*365)
    #         total_no_points = total_no_days//interval_size
    #         missing_count = total_no_points - regular_df.shape[0]
    #         missing_from_beginning = (min(regular_df.human_system_start_time) - \
    #                                   pd.to_datetime(datetime.datetime(start_year, 1, 1, 0, 0))).days // interval_size

    #         missing_from_end = missing_count - missing_from_beginning
    A = pd.date_range(
        pd.Timestamp(start_year, 1, 1),
        min(regular_df.human_system_start_time),
        freq=str(interval_size) + "D",
    )

    missing_begin_df = pd.DataFrame(
        data=None, index=np.arange(len(A[:-1])), columns=regular_cols
    )

    missing_begin_df.human_system_start_time = A[:-1]
    missing_begin_df.ID = regular_df.ID.unique()[0]
    mm = min(regular_df.human_system_start_time)
    missing_begin_df[V_idks] = np.array(
        regular_df[regular_df.human_system_start_time == mm][V_idks]
    )[0]
    if "dataset" in regular_cols:
        missing_begin_df.dataset = regular_df.dataset.unique()[0]

    #
    # The tail of the TS
    #
    A = pd.date_range(
        max(regular_df.human_system_start_time),
        pd.Timestamp(end_year, 12, 31),
        freq=str(interval_size) + "D",
    )

    missing_end_df = pd.DataFrame(
        data=None, index=np.arange(len(A[1:])), columns=regular_cols
    )
    missing_end_df.human_system_start_time = A[1:]
    missing_end_df.ID = regular_df.ID.unique()[0]
    mm = max(regular_df.human_system_start_time)
    missing_end_df[V_idks] = np.array(
        regular_df[regular_df.human_system_start_time == mm][V_idks]
    )[0]
    if "dataset" in regular_cols:
        missing_end_df.dataset = regular_df.dataset.unique()[0]

    regular_df = pd.concat([missing_begin_df, regular_df, missing_end_df])
    regular_df.reset_index(drop=True, inplace=True)
    return regular_df


def set_negatives_to_zero(df, indeks="NDVI"):
    df.loc[df[indeks] < 0, indeks] = 0
    return df


def clip_outliers(df, idx="NDVI"):
    # dt_copy = df.copy()
    df.loc[df[idx] > 1, idx] = 1
    df.loc[df[idx] < -1, idx] = -1
    return df


def add_human_start_time_by_system_start_time(HDF):
    """Returns human readable time (conversion of system_start_time)

    Arguments
    ---------
    HDF : dataframe

    Returns
    -------
    HDF : dataframe
        the same dataframe with added column of human readable time.
    """
    HDF.system_start_time = HDF.system_start_time / 1000
    time_array = HDF["system_start_time"].values.copy()
    human_time_array = [
        time.strftime("%Y-%m-%d", time.localtime(x)) for x in time_array
    ]
    HDF["human_system_start_time"] = human_time_array

    if type(HDF["human_system_start_time"] == str):
        HDF["human_system_start_time"] = pd.to_datetime(HDF["human_system_start_time"])

    """
    Lets do this to go back to the original number:
    I added this when I was working on Colab on March 30, 2022.
    Keep an eye on it and see if we have ever used "system_start_time"
    again. If we do, how we use it; i.e. do we need to get rid of the 
    following line or not.
    """
    HDF.system_start_time = HDF.system_start_time * 1000
    return HDF


def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


# We need this for DL
# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # load the image
    img = img_to_array(img)  # convert to array
    img = img.reshape(1, 224, 224, 3)  # reshape into a single sample with 3 channels
    img = img.astype("float32")  # center pixel data
    img = img - [123.68, 116.779, 103.939]
    return img
