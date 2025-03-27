# %% [markdown]
"""
This has architecture of the paper 

Global Normalized Difference Vegetation Index forecasting from air 
temperature, soil moisture and precipitation using a deep neural network
"""
# %%
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

print(tf.__version__)
print(tf.version.VERSION)

# %% [markdown]
"""
for whatever reason, the following 2 lines dont work
even tho tensorflow in my environemnt is the same as that
on my computer!!! So, we comment out and then add it in 
different way!
"""
# %%
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

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc


# %% [markdown]
# ##      Parameters

# %%
NDVI_lag_or_delta = "delta"  # let be either lag or delta

# do the following since walla walla has two parts and we have to use walla_walla in terminal
print("Terminal Arguments are: ")
print("NDVI_lag_or_delta= ", NDVI_lag_or_delta)
print("__________________________________________")

# %% [markdown]
# ###  Define Directories

# %%
dpi_ = 300

plot_dir = "/Users/hn/Documents/01_research_data/RangeLand/Mike_Results/plots/"
os.makedirs(plot_dir, exist_ok=True)

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
# %%
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

# %%
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


# %%
def create_model(l2_lambda):
    model = Sequential()
    model.add(Dense(25, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(10, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(1, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["mean_squared_error", "R2Score"],
    )
    return model


model = KerasRegressor(model=create_model, verbose=0)

param_grid = {
    "optimizer__learning_rate": [0.01, 0.1],
    "model__l2_lambda": [0.1],
    "batch_size": [16],
    "epochs": [10],
}

seed = 7
tf.random.set_seed(seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(x_train_df, y_train_df)
# %% [markdown]
# From https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

# %%
# load dataset
dataset = np.loadtxt("/Users/hn/Desktop/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model

# %%
# Use scikit-learn to grid search the learning rate and momentum
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
# fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)

# %%
model = KerasClassifier(model=create_model, loss="binary_crossentropy", optimizer="SGD", epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(optimizer__learning_rate=learn_rate, optimizer__momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# %%

# %%
