import os, os.path, pickle, sys
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

# %%
# Major ticks every 5, minor ticks every 1
major_ticks = np.arange(1984, 2024, 5)
minor_ticks = np.arange(1984, 2024, 1)
y_var = "mean_lb_per_acr"

dpi_ = 300

# %%
rangeland_bio_base = "/data/project/agaid/h.noorazar/rangeland_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
bio_reOrganized = rangeland_bio_data + "reOrganized/"

bio_plots = rangeland_bio_base + "plots/"
os.makedirs(bio_plots, exist_ok=True)

yue_plots = bio_plots + "Yue_99CL_TS_dismissedByOrig/"
os.makedirs(yue_plots, exist_ok=True)

# %%
ANPP = pd.read_pickle(bio_reOrganized + "bpszone_ANPP_no2012.sav")
ANPP = ANPP["bpszone_ANPP"]
ANPP.head(2)

# %%
filename = bio_reOrganized + "ANPP_MK_Spearman_no2012.sav"
ANPP_MK_df = pd.read_pickle(filename)
ANPP_MK_df = ANPP_MK_df["ANPP_MK_df"]

print(len(ANPP_MK_df["fid"].unique()))
ANPP_MK_df.head(2)

# %%
len(ANPP_MK_df["EW_meridian"].unique())

# %%

# %%
Yue_green = ANPP_MK_df[["fid", "trend", "trend_yue", "p_yue"]].copy()
print(len(Yue_green))
Yue_green = Yue_green[Yue_green["trend_yue"] == "increasing"]
print(len(Yue_green))
Yue_green = Yue_green[Yue_green["p_yue"] < 0.01].copy()
print(len(Yue_green))
Yue_green = Yue_green[Yue_green["trend"] != "increasing"]
print(len(Yue_green))

print(Yue_green["trend"].unique())
print(Yue_green["trend_yue"].unique())
Yue_green.head(2)

# %%
Yue_green = list(Yue_green["fid"].unique())
len(Yue_green)

# %%

# %%
tick_legend_FontSize = 12
params = {
    "legend.fontsize": tick_legend_FontSize * 1.5,
    "axes.labelsize": tick_legend_FontSize * 1.2,
    "axes.titlesize": tick_legend_FontSize * 1.5,
    "xtick.labelsize": tick_legend_FontSize * 1.1,
    "ytick.labelsize": tick_legend_FontSize * 1.1,
    "axes.titlepad": 10,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.labelbottom": True,
    "ytick.labelleft": True,
    #          'axes.linewidth' : .05
}

plt.rcParams.update(params)

# %%

for a_fid in Yue_green:
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(10, 3),
        sharex=True,
        gridspec_kw={"hspace": 0.15, "wspace": 0.05},
        dpi=dpi_,
    )
    # axes.grid(which='major', alpha=0.5, axis="both")
    axes.set_xticks(major_ticks)
    axes.set_xticks(minor_ticks, minor=True)
    axes.grid(which="minor", alpha=0.2, axis="x")
    axes.grid(which="major", alpha=0.5, axis="x")

    ########## plot 1
    df = ANPP[ANPP.fid == a_fid].copy()
    axes.plot(df.year, df[y_var], linewidth=3, color="dodgerblue")
    axes.scatter(df.year, df[y_var], zorder=3, color="dodgerblue")

    ###
    ### Text
    trend_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "trend_yue"].item()
    slope_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "sens_slope"].item(), 2)
    Tau_ = round(ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "Tau"].item(), 2)
    state_ = ANPP_MK_df.loc[ANPP_MK_df.fid == a_fid, "state_majority_area"].item()

    text_ = "trend: {}\nSen's slope: {}, \nTau: {}, \n{} (FID: {})".format(
        trend_, slope_, Tau_, state_, a_fid
    )

    y_txt = df[y_var].max() * 0.99
    axes.text(1983, y_txt, text_, fontsize=tick_legend_FontSize * 1.2, va="top")
    ####
    axes.set_title("99% Yue Greening. Dismissed by Original MK.")
    axes.set_ylabel(r"$\mu_{NPP}$ (lb/acr)")

    if slope_ <= 20:
        yue_plots_less20 = yue_plots + "slope_less20/"
        os.makedirs(yue_plots_less20, exist_ok=True)
        file_name = (
            yue_plots_less20 + "FID_" + str(a_fid) + "_Yue99PercCL_dismissedOrig.pdf"
        )
    elif (slope_ >= 20) and (slope_ < 30):
        yue_plots_20to30 = yue_plots + "slope_20to30/"
        os.makedirs(yue_plots_20to30, exist_ok=True)

        file_name = (
            yue_plots_20to30 + "FID_" + str(a_fid) + "_Yue99PercCL_dismissedOrig.pdf"
        )
    elif slope_ >= 30:
        yue_plots_ge30 = yue_plots + "slope_ge30/"
        os.makedirs(yue_plots_ge30, exist_ok=True)
        file_name = (
            yue_plots_ge30 + "FID_" + str(a_fid) + "_Yue99PercCL_dismissedOrig.pdf"
        )

    plt.savefig(file_name, dpi=dpi_, bbox_inches="tight")
    plt.close("all")

    # %%
