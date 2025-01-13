import pandas as pd
import numpy as np

import sys, scipy, scipy.signal

import datetime
from datetime import date, timedelta
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import seaborn as sb


def plot_SF(SF, ax_, cmap_="Pastel1", col="EW_meridian"):
    SF.plot(
        column=col,
        ax=ax_,
        alpha=1,
        cmap=cmap_,
        edgecolor="k",
        legend=False,
        linewidth=0.1,
    )


def makeColorColumn(gdf, variable, vmin, vmax):
    # apply a function to a column to create a new column of assigned colors & return full frame
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGnBu)
    gdf["value_determined_color"] = gdf[variable].apply(
        lambda x: mcolors.to_hex(mapper.to_rgba(x))
    )
    return gdf
