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
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys

from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from datetime import datetime

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%
rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
min_bio_dir = rangeland_bio_data + "Min_Data/"

rangeland_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
rangeland_reOrganized = rangeland_base + "reOrganized/"

bio_reOrganized = rangeland_bio_data + "reOrganized/"
os.makedirs(bio_reOrganized, exist_ok=True)

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)
####### Laptop
# rangeland_bio_base = "/Users/hn/Documents/01_research_data/RangeLand_bio/"
# min_bio_dir = rangeland_bio_base
# rangeland_base = rangeland_bio_base
# rangeland_reOrganized = rangeland_base

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman.sav"
ANPP_MK_Spearman = pd.read_pickle(filename)
ANPP_MK_Spearman = ANPP_MK_Spearman["ANPP_MK_df"]
ANPP_MK_Spearman.head(2)

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_Spearman_no2012 = pd.read_pickle(filename)
ANPP_MK_Spearman_no2012 = ANPP_MK_Spearman_no2012["ANPP_MK_df"]
ANPP_MK_Spearman_no2012.head(2)
