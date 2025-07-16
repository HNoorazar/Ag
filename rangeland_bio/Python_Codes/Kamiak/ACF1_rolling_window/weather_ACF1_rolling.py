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


ACF_data = rangeland_bio_data + "ACF1/"
os.makedirs(ACF_data, exist_ok=True)
#####################################################################################
ff_ = bio_reOrganized + "bpszone_annualWeatherByHN_and_deTrended.sav"
weather_detrended = pd.read_pickle(ff_)
weather_detrended = weather_detrended["bpszone_annual_weather_byHN"]

weather_detrended.head(2)

## On july 25, I am eliminated first-diff deterending
# if "diff".lower() in y_.lower():
#     weather_detrended.dropna(subset=[y_], inplace=True)
#     weather_detrended.reset_index(drop=True, inplace=True)

weather_detrended.dropna(subset=[y_], inplace=True)
weather_detrended.reset_index(drop=True, inplace=True)

ACF1s_window = rc.rolling_autocorr_df_prealloc(
    df=weather_detrended, y_var=y_, window_size=ws, lag=1
)

ws_str = str(ws)
fnamePref = f"rolling_ACF1_ws{ws_str}_{y_}"

export_ = {
    fnamePref: ACF1s_window,
    "source_code": "weather_ACF1_rolling from Kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

filename = ACF_data + fnamePref + ".sav"
pickle.dump(export_, open(filename, "wb"))

filename = ACF_data + fnamePref + ".csv"
ACF1s_window.to_csv(filename, index=False)


end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
