import shutup, time  # , random

shutup.please()

import numpy as np
import pandas as pd

from datetime import date, datetime
from random import seed, random
import sys, os, os.path, shutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import scipy, scipy.signal
import pickle, h5py

# from tslearn.metrics import dtw as dtw_metric
# https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")

####################################################################################
###
###                      Kamiak Core path
###
####################################################################################

sys.path.append("/home/h.noorazar/NASA/")
import NASA_core as nc

####################################################################################
###
###      Parameters
###
####################################################################################

VI_idx = sys.argv[1]
smooth = sys.argv[2]
batch_no = str(sys.argv[3])
model = sys.argv[4]

print("Passed Args. are: ", VI_idx, ",", smooth, ",", batch_no, ",", model)
####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"

if model != "DL":
    if smooth == "regular":
        in_dir = data_base + "VI_TS/04_regularized_TS/"
    else:
        in_dir = data_base + "VI_TS/05_SG_TS/"
else:
    in_dir = data_base + "06_cleanPlots_4_DL_pre2008/" + VI_idx + "_" + smooth + "_plots/"

out_dir = data_base + "trend_ML_preds/"
os.makedirs(out_dir, exist_ok=True)

param_dir = data_base + "parameters/"
# model_dir = data_base + "ML_Models_Oct17/"
model_dir_base = data_base + "ML_Models_Oct17/overSample/"


#####################################################################
######
######                           Body
######
#####################################################################
# We need this for KNN
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


####################################################################################
###
###      Read
###
####################################################################################

winnerModels = pd.read_csv(param_dir + "winnerModels_overSample.csv")

winnerModel = np.array(
    winnerModels.loc[
        (winnerModels.VI_idx == VI_idx)
        & (winnerModels.smooth == smooth)
        & (winnerModels.model == model)
    ].output_name
)[0]
print("winnerModel=", winnerModel)

##
##    Read Model
##
if winnerModel.endswith(".sav"):
    f_name = VI_idx + "_" + smooth + "_intersect_batchNumber" + batch_no + "_wide_JFD_pre2008.csv"
    wide_TS = pd.read_csv(in_dir + f_name)
    print("wide_TS.shape: ", wide_TS.shape)

    # ML_model = pickle.load(open(model_dir + winnerModel, "rb"))
    ML_model = pickle.load(open(model_dir_base + model + "/" + winnerModel, "rb"))
    predictions = ML_model.predict(wide_TS.iloc[:, 2:])
    pred_colName = model + "_" + VI_idx + "_" + smooth + "_preds"
    A = pd.DataFrame(columns=["ID", "year", pred_colName])
    A.ID = wide_TS.ID.values
    A.year = wide_TS.year.values
    A[pred_colName] = predictions
    predictions = A.copy()
    del A
else:
    # from keras.utils import to_categorical
    from tensorflow.keras.utils import to_categorical, load_img, img_to_array
    from keras.models import Sequential, Model, load_model
    from keras.applications.vgg16 import VGG16
    import tensorflow as tf

    # from keras.optimizers import SGD
    from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.optimizers import SGD
    from keras.preprocessing.image import ImageDataGenerator

    # ML_model = load_model(model_dir + winnerModel)
    ML_model = load_model(model_dir_base + model + "/" + winnerModel)
    prob_thresholds = [
        3,
        3.4,
        3.5,
        3.6,
        4,
        5,
        6,
        7,
        8,
        9,
        9.1,
        9.2,
        9.3,
        9.4,
        9.5,
        9.6,
        9.7,
        9.8,
        9.9,
    ]

    plot_dir = in_dir
    # p_filenames = os.listdir(plot_dir)

    f_name = "NDVI_SG_intersect_batchNumber" + batch_no + "_wide_JFD.csv"
    wide_TS = pd.read_csv(data_base + "VI_TS/05_SG_TS/" + f_name)
    p_filenames_clean = list(wide_TS.ID + "_" + wide_TS.year.astype(str) + ".jpg")

    # p_filenames_clean = []
    # for a_file in p_filenames:
    #     if a_file.endswith(".jpg"):
    #         # if a_file.split(".")[0] in SF_data.ID.unique():
    #         p_filenames_clean += [a_file]

    # print ("len(p_filenames_clean) is [{}].".format(len(p_filenames_clean)))

    predictions = pd.DataFrame({"filename": p_filenames_clean})
    predictions["prob_single"] = -1.0

    for idx in predictions.index:
        img = load_image(plot_dir + predictions.loc[idx, "filename"])
        predictions.loc[idx, "prob_single"] = ML_model.predict(img, verbose=False)[0][0]

    # for prob in np.divide(prob_thresholds, 10).round(2):
    #     colName = "prob_point" + str(prob)[2:]
    #     # print ("line 39: " + str(prob))
    #     # print ("line 40: " + colName)
    #     predictions.loc[predictions.prob_single < prob, colName] = "d"
    #     predictions.loc[predictions.prob_single >= prob, colName] = "s"


######  Export Output
pred_colName = VI_idx + "_" + smooth + "_" + model + "_batchNumber" + batch_no + "_preds"
out_name = out_dir + pred_colName + "_pre2008.csv"
predictions.to_csv(out_name, index=False)

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")
