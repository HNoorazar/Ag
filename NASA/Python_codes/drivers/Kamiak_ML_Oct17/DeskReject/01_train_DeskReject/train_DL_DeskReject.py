import shutup  # , random

shutup.please()


from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis


import numpy as np
import pandas as pd
from datetime import date, datetime
from random import seed, random

import sys, os, os.path, shutil, h5py, time
import matplotlib
import matplotlib.pyplot as plt

from pylab import imshow

# vgg16 model used for transfer learning on the dogs and cats dataset
from matplotlib import pyplot

# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf

# from keras.optimizers import SGD

# from keras.optimizers import gradient_descent_v2
# SGD = gradient_descent_v2.SGD(...)

from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

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
smooth_type = sys.argv[2]
train_ID = sys.argv[3]  # we have different training sets: 1, 2, 3, 4, 5, 6
SR = sys.argv[4]  # sample Ratio 3, 4, 5, 6, 7, 8

print("Passed Args. are: ", VI_idx, ",", smooth_type, ",", train_ID, ",", SR)
####################################################################################
###
###      Directories
###
####################################################################################
data_base = "/data/project/agaid/h.noorazar/NASA/"

ML_data_dir_base = data_base + "/ML_data_Oct17/"
overSamp_data_base = ML_data_dir_base + "overSamples/"

model_dir = data_base + "ML_Models_Oct17/DeskReject/"
os.makedirs(model_dir, exist_ok=True)

train_test_dir = overSamp_data_base + "train_test_DL_" + str(train_ID) + "/"

train_plot_dir = (
    train_test_dir + "/oversample" + str(SR) + "/" + smooth_type + "_" + VI_idx + "_train/"
)

train_test_split_dir = ML_data_dir_base + "train_test_DL_" + str(train_ID) + "/"


#####################################################################
######
######                           Body
######
#####################################################################
# We need this for KNN
def DTW_prune(ts1, ts2):
    d, _ = dtw.warping_paths(ts1, ts2, window=10, use_pruning=True)
    return d


LL = "_wide_train80_split_2Bconsistent_Oct17_overSample"
train_fileName = VI_idx + "_" + smooth_type + LL + str(SR) + ".csv"
train80_wide = pd.read_csv(train_test_dir + train_fileName)

print(train80_wide.shape)

train_plot_count = len(os.listdir(train_plot_dir + "/single/")) + len(
    os.listdir(train_plot_dir + "/double/")
)
print(len(train80_wide) == train_plot_count)
print("===================================================")


# train_folder_80 = train_plot_dir +"/train80/"
# define cnn model
def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
    output = Dense(1, activation="sigmoid")(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# run the test harness for evaluating a model
def run_test_harness():
    # define model
    _model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    # train_separate_dir = train_folder_80 + "/separate_singleDouble/"
    train_it = datagen.flow_from_directory(
        train_plot_dir, class_mode="binary", batch_size=16, target_size=(224, 224)
    )
    # fit model
    _model.fit(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=1)
    os.makedirs(model_dir, exist_ok=True)
    _model.save(
        model_dir
        + "01_TL_"
        + VI_idx
        + "_"
        + smooth_type
        + "_train80_SR_"
        + str(SR)
        + "_DL_"
        + str(train_ID)
        + ".h5"
    )


run_test_harness()


##############################################################################

print("--------------------------------------------------------------")
print(date.today(), "-", datetime.now().strftime("%H:%M:%S"))
print("--------------------------------------------------------------")
