"""
Global Normalized Difference Vegetation Index forecasting from air 
temperature, soil moisture and precipitation using a deep neural network

"""
from datetime import datetime, date

current_time = datetime.now()
print("Today's date:", date.today())
print("Current Time =", current_time.strftime("%H:%M:%S"))


import shutup

shutup.please()
import pandas as pd
import numpy as np
import os, os.path, pickle, sys

from sklearn import preprocessing
import tensorflow as tf
import keras_tuner

print(tf.__version__)
print(tf.version.VERSION)

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

from keras import losses, optimizers, metrics, regularizers
from keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from keras import backend as K

sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc


####################################################################################
###
###      Parameters
###
####################################################################################

NDVI_lag_or_delta = sys.argv[1]  # let be either lag or delta
architecture = sys.argv[2]  # let be either paper or Network2

# do the following since walla walla has two parts and we have to use walla_walla in terminal
print("Terminal Arguments are: ")
print("NDVI_lag_or_delta= ", NDVI_lag_or_delta)
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

models_dir = NDVI_weather_base + "models/DL/"
os.makedirs(models_dir, exist_ok=True)

plot_dir = NDVI_weather_base + "plots/"
os.makedirs(plot_dir, exist_ok=True)


common_data = data_basement + "common_data/"
# data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
# param_dir = data_dir_base + "parameters/"
# Min_data_base = data_dir_base + "Min_Data/"

# Shannon_data_dir = data_dir_base + "Shannon_Data/"
# Min_data_dir_base = data_dir_base + "Min_Data/"
# NASS_downloads = data_dir_base + "/NASS_downloads/"
# NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
# mike_dir = data_dir_base + "Mike/"
# reOrganized_dir = data_dir_base + "reOrganized/"
# os.makedirs(reOrganized_dir, exist_ok=True)
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


NDVI_weather = pd.read_pickle(NDVI_weather_data_dir + "NDVI_weather.sav")
NDVI_weather = NDVI_weather["NDVI_weather_input"]

######################################################################################
######################################################################################
######################################################################################

if NDVI_lag_or_delta == "deta":
    indp_vars = [
        "county_fips",
        "year",
        "month",
        "tavg_avg_lag1",
        "ppt_lag1",
        "delta_NDVI",
    ]
else:
    indp_vars = [
        "county_fips",
        "year",
        "month",
        "tavg_avg_lag1",
        "ppt_lag1",
        "MODIS_NDVI_lag1",
    ]

y_var = "MODIS_NDVI"

NDVI_weather = NDVI_weather[[y_var] + indp_vars]
NDVI_weather.dropna(inplace=True)
NDVI_weather.reset_index(drop=True, inplace=True)


X = NDVI_weather[indp_vars].copy()
y = NDVI_weather[y_var].copy()


x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True
)

#### do a validation set for the KerasTuner.
x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(
    x_train_df, y_train_df, test_size=0.2, random_state=0, shuffle=True
)
input_shape_ = x_train_df.shape[1]


######################################################################################
######################################################################################
######################################################################################
#############
#############
#############
#############
#############
def create_model(l2_lambda):
    model = Sequential()
    model.add(
        Dense(
            150,
            input_shape=(input_shape_,),
            activation="relu",
            kernel_regularizer=l2(l2_lambda),
        )
    )
    model.add(Dense(100, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(50, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(25, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(10, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(1, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["mean_squared_error", "R2Score"],
    )
    return model


if architecture == "paper":

    def call_existing_code(epochs, learning_rate, l2_lambda):
        model = Sequential()
        model.add(
            Dense(
                250,
                input_shape=(input_shape_,),
                activation="relu",
                kernel_regularizer=l2(l2_lambda),
            )
        )
        model.add(Dense(200, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(150, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(100, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(50, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(25, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(10, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(1, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mean_squared_error", "R2Score"],
        )
        return model


if architecture == "Network2":

    def call_existing_code(epochs, learning_rate, l2_lambda):
        model = Sequential()
        model.add(
            Dense(
                150,
                input_shape=(input_shape_,),
                activation="relu",
                kernel_regularizer=l2(l2_lambda),
            )
        )
        model.add(Dense(100, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(50, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(25, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(10, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.add(Dense(1, activation="relu", kernel_regularizer=l2(l2_lambda)))
        model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mean_squared_error", "R2Score"],
        )
        return model


def build_model(hp):
    """
    https://keras.io/keras_tuner/api/hyperparameters/
    hp.Float("learning_rate", min_value=0.001, max_value=10, step=10, sampling="log")
    When sampling="log", the step is multiplied between samples.
    The possible values are [0.001, 0.01, 0.1, 1, 10].
    """
    epochs = hp.Int("epochs", min_value=10, max_value=30, step=10)
    l2_lambda = hp.Float("L2s", min_value=0.001, max_value=10, step=10, sampling="log")
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(
        epochs=epochs, learning_rate=learning_rate, l2_lambda=l2_lambda
    )
    return model


#########################################################################################################
#########
#########    Search
#########
max_trials_ = 10
executions_per_trial_ = 3
print(f"{build_model(keras_tuner.HyperParameters()) = }")
project_name_ = architecture + "_" + NDVI_lag_or_delta
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=max_trials_,
    executions_per_trial=executions_per_trial_,
    overwrite=True,
    directory=NDVI_weather_data_dir,
    project_name=project_name_,
)
print("---------------------------------------------------------------------")
tuner.search_space_summary()
print("---------------------------------------------------------------------")
tuner.search(x_train_df, y_train_df, validation_data=(x_val_df, y_val_df))


# Get the top 2 models.
models = tuner.get_best_models(num_models=1)
best_model = models[0]
print(best_model.summary())
print("---------------------------------------------------------------------")
print(tuner.results_summary())


#########################################################################################################
filename = (
    models_dir
    + "KerasTuner_"
    + project_name_
    + "_executions_per_trial_"
    + str(executions_per_trial_)
    + "_max_trials_"
    + str(max_trials_)
    + "best_model.sav"
)

export_ = {
    "best_model": best_model,
    "source_code": "KerasTuner",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))


final_time = datetime.now()
print("Today's date:", date.today())
print("Current Time =", final_time.strftime("%H:%M:%S"))
print(
    "it took the following to run the code (in hours): ",
    (final_time - current_time).total_seconds() / 3600,
)
