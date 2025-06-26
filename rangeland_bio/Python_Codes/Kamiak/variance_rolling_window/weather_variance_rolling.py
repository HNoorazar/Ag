import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

import pyproj
import geopandas
from geopy.distance import geodesic

# do not do import datetime. if you do it this way, datetime.now() wont work
# then you have ot do datetime.datetime.now()
from datetime import datetime
from datetime import date
import time


sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc

start_time = time.time()
#####################################################################################
ws = int(sys.argv[1])
y_ = str(sys.argv[2])
#####################################################################################
#####################################################################################
#####################################################################################

research_data_ = "/data/project/agaid/h.noorazar/"
rangeland_bio_base = research_data_ + "rangeland_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
bio_reOrganized = rangeland_bio_data + "reOrganized/"
common_data = research_data_ + "common_data/"

NDVI_weather_base = research_data_ + "NDVI_v_Weather/"
NDVI_weather_data_dir = NDVI_weather_base + "data/"


rolling_variances_dir = rangeland_bio_data + "rolling_variances/"
os.makedirs(rolling_variances_dir, exist_ok=True)
#####################################################################################
weather_detrended = pd.read_pickle(bio_reOrganized + "weather_detrended.sav")
weather_detrended = weather_detrended["weather_detrended"]
weather_detrended.head(2)

if "diff".lower() in y_.lower():
    weather_detrended.dropna(subset=[y_], inplace=True)
    weather_detrended.reset_index(drop=True, inplace=True)

variances = rc.rolling_variance_df_prealloc(
    df=weather_detrended, y_var=y_, window_size=ws
)

ws_str = str(ws)

if y_ == "avg_of_dailyAvgTemp_C":
    fnamePref = f"rolling_variance_ws{ws_str}_temp"
elif y_ == "precip_mm":
    fnamePref = f"rolling_variance_ws{ws_str}_prec"

else:
    fnamePref = f"rolling_variance_ws{ws_str}_{y_}"

filename = rolling_variances_dir + fnamePref + ".sav"

export_ = {
    fnamePref: variances,
    "source_code": "weather_variances_rolling.py from Kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

filename = rolling_variances_dir + fnamePref + ".csv"
variances.to_csv(filename, index=False)


end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
