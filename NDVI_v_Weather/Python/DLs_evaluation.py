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

# %%
import shutup
shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
from datetime import datetime, date

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%
import tensorflow as tf

import keras
from keras import losses, optimizers, metrics, layers
from keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from scikeras.wrappers import KerasRegressor
# from sklearn.model_selection import GridSearchCV

import keras_tuner

print(tf.__version__)
print(tf.version.VERSION)


# %%
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def r_squared(y_true, y_pred):
    # Total sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # Residual sum of squares
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    return (1 - ss_res / ss_tot)


# %%
dpi_ = 300

plot_dir = "/Users/hn/Documents/01_research_data/RangeLand/Mike_Results/plots/"
os.makedirs(plot_dir, exist_ok=True)

# %%
research_db = "/Users/hn/Documents/01_research_data/"
common_data = research_db + "common_data/"

data_dir_base = research_db + "RangeLand/Data/"
param_dir = data_dir_base + "parameters/"
Min_data_base = data_dir_base + "Min_Data/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

NDVI_weather_data_dir = research_db + "/NDVI_v_Weather/data/"
NDVI_weather_model_dir = research_db + "/NDVI_v_Weather/models/"
DL_model_dir = NDVI_weather_model_dir + "DL_LastLayerRelu/"

# %%
abb_dict = pd.read_pickle(common_data + "county_fips.sav")
county_fips_df = abb_dict["county_fips"]
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
#######
state_name_fips = pd.DataFrame({"state_full" : list(abb_dict["full_2_abb"].keys()),
                                "state" : list(abb_dict["full_2_abb"].values())})

state_name_fips = pd.merge(state_name_fips, 
                           abb_dict["state_fips"][["state_fips", "EW_meridian", "state"]], 
                           on=["state"], how="left")
state_name_fips.head(2)

# %%
state_fips_SoI = state_name_fips[state_name_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
state_fips_SoI.head(2)

# %%
state_fips_west = list(state_fips_SoI[state_fips_SoI["EW_meridian"] == "W"]["state_fips"].values)
state_fips_west[:3]

# %% [markdown]
# # west Meridian

# %%
WM_counties = county_fips_df[county_fips_df["EW_meridian"] == "W"]
WM_counties = list(WM_counties["county_fips"])
len(WM_counties)

# %%
filename = NDVI_weather_data_dir + "NDVI_weather.sav"
NDVI_weather = pd.read_pickle(filename)
NDVI_weather=NDVI_weather["NDVI_weather_input"]

# %%
indp_vars = ['county_fips', 'year', 'month', 'tavg_avg_lag1', 'ppt_lag1', 'delta_NDVI']
y_var = 'MODIS_NDVI'

NDVI_weather = NDVI_weather[[y_var] + indp_vars]

print (NDVI_weather.shape)
NDVI_weather.dropna(inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)
print (NDVI_weather.shape)
# 258300 - 256250

X = NDVI_weather[indp_vars].copy()
y = NDVI_weather[y_var].copy()

# %%
# it was working before, without doing this!!!
X['county_fips'] = X['county_fips'].astype(np.float64)

# %%
from sklearn.model_selection import train_test_split

x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# %%
input_shape_ = x_train_df.shape[1]

# %% [markdown]
# ### So far
#
# everything was copied from ```NDVI_Weather_KerasTuner_prep.ipynb```

# %%
all_DL_models = [x for x in os.listdir(DL_model_dir) if x.endswith("sav")]
print (f"{len(all_DL_models) = }")
all_DL_models[:3]

# %%
a_model = pd.read_pickle(DL_model_dir + all_DL_models[0])
a_model = a_model['best_model']
# Evaluate the model
R2, loss, mean_squared_error = a_model.evaluate(x_test_df, y_test_df)
print(f"R2:{R2}, Loss: {loss}, MSE: {mean_squared_error}")

# %%

# %%

# %%
a_model.summary()

# %%
all_DL_models[2]

# %%
a_model = pd.read_pickle(DL_model_dir + all_DL_models[2])
a_model["source_code"]

# %%
param_grid = {
        "optimizer__learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "model__l2_lambda": [0.001, 0.01, 0.1],
        # "l2_lambda": [0.001, 0.01, 0.1],
        "epochs": [10, 20, 50, 100],
    }

# %%
param_grid["Hi"] = "hello"

# %%
param_grid

# %%
