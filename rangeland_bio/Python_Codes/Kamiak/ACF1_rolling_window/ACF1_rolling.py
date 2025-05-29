import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

import pyproj
import geopandas
from geopy.distance import geodesic
from datetime import datetime

sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc

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
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

#####################################################################################
ws = int(sys.argv[1])

#####################################################################################
y_ = "mean_lb_per_acr"
ACF1s_window = rc.rolling_autocorr_df_prealloc(df=ANPP, y_var=y_, window_size=ws, lag=1)

ws_str = str(ws)
filename = ACF_data + f"rolling_autocorrelations_ws{ws_str}.sav"

export_ = {
    f"rolling_autocorrelations_ws{ws_str}": ACF1s_window,
    "source_code": "ACF1_rolling from Kamiak",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

pickle.dump(export_, open(filename, "wb"))

filename = ACF_data + f"rolling_autocorrelations_ws{ws_str}.csv"
ACF1s_window.to_csv(filename, index=False)
