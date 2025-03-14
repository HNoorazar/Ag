# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# The link Mike sent about [spatial correlation](https://geographicdata.science/book/notebooks/11_regression.html)
#
# https://scholar.google.com/scholar?hl=en&as_sdt=0%2C48&q=Conley+T%2C+Spatial+econometrics.New+Palgrave+Dictionary+of+Economic&btnG=
#
# https://www.nber.org/system/files/working_papers/t0055/t0055.pdf
#
# https://pysal.org/libpysal/generated/libpysal.weights.KNN.html

# %% [raw]
# An interesting question arises around the relevance of the regimes. Are estimates for each variable across regimes statistically different? For this, the model object also calculates for us what is called a ```Chow``` test. This is a statistic that tests the null hypothesis that estimates from different regimes are undistinguishable. If we reject the null, we have evidence suggesting the regimes actually make a difference.
#
#
# The first value represents the ```statistic```, while the second one captures the ```p-value```. 
#
# In this case, the two regimes are statistically different from each other.
#
# The text above this line is for the following model (for the record, just in case the model is changed in above cell):
#
# ### Allows for different slopes per category:
#
# depen_var_name = "inventoryDiv1000"
# indp_vars = ["metric_total_matt_nppDiv10M"]
# m5 = spreg.OLS_Regimes(y = inv_prices_ndvi_npp[depen_var_name].values, # Dependent variable
#                        x = inv_prices_ndvi_npp[indp_vars].values,  # Independent variables
#
#                        # Variable specifying neighborhood membership
#                        regimes = inv_prices_ndvi_npp["EW_meridian"].tolist(),
#               
#                        # Variables to be allowed to vary (True) or kept
#                        # constant (False). Here we set all to False
#                        # cols2regi=[False] * len(indp_vars),
#                         
#                        # Allow the constant term to vary by group/regime
#                        constant_regi="many",
#                         
#                        # Allow separate sigma coefficients to be estimated
#                        # by regime (False so a single sigma)
#                        regime_err_sep=False,
#                        name_y=depen_var_name, # Dependent variable name
#                         
#                        # Independent variables names
#                        name_x=indp_vars)
#
# m5_results = pd.DataFrame({# Pull out regression coefficients and
#                            # flatten as they are returned as Nx1 array
#                            "Coeff.": m5.betas.flatten(),
#                            # Pull out and flatten standard errors
#                            "Std. Error": m5.std_err.flatten(),
#                            # Pull out P-values from t-stat object
#                            "P-Value": [i[1] for i in m5.t_stat],
#                            }, index=m5.name_x)
#
#
# \# West regime ## Extract variables for the coastal regime
# west_m = [i for i in m5_results.index if "W_" in i]
#
# \## Subset results to coastal and remove the 1_ underscore
# west = m5_results.loc[west_m, :].rename(lambda i: i.replace("W_", ""))
#
#
# \## Build multi-index column names
# west.columns = pd.MultiIndex.from_product([["West Meridian"], west.columns])
#
# \# East model ## Extract variables for the non-coastal regime
# east_m = [i for i in m5_results.index if "E_" in i]
#
# \## Subset results to non-coastal and remove the 0_ underscore
# east = m5_results.loc[east_m, :].rename(lambda i: i.replace("E_", ""))
#
# \## Build multi-index column names
# east.columns = pd.MultiIndex.from_product([["East Meridian"], east.columns])
#
# \# Concat both models
# pd.concat([east, west], axis=1)
#
# [Reference is here](https://geographicdata.science/book/notebooks/11_regression.html)

# %%
import shutup

shutup.please()

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
import statistics
import statsmodels.api as sm

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc
import rangeland_plot_core as rcp

from datetime import datetime, date

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

# %%
from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda
import geopandas, contextily

from scipy.stats import ttest_ind

# %%
tick_legend_FontSize = 8

params = {"legend.fontsize": tick_legend_FontSize,  # medium, large
          "axes.labelsize": tick_legend_FontSize * 1.5,
          "axes.titlesize": tick_legend_FontSize * 1.3,
          "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
          "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
          "axes.titlepad": 10}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %% [markdown]
# #### Fit OLS model with Spreg
#
# \begin{equation}
# m1 = spreg.OLS(db[["log_price"]].values, #Dependent variable
#                     db[variable_names].values, # Independent variables
#                name_y="log_price", # Dependent variable name
#                name_x=variable_names # Independent variable name
#                )
# \end{equation}

# %%
from statsmodels.formula.api import ols
# fit = ols('inventory ~ C(state_dummy_int) + max_ndvi_in_year_modis + beef_price_at_1982 + hay_price_at_1982', 
#           data=all_df_normalized_needed).fit() 

# fit.summary()

# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
param_dir = data_dir_base + "parameters/"

Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"
NASS_downloads = data_dir_base + "/NASS_downloads/"
NASS_downloads_state = data_dir_base + "/NASS_downloads_state/"
mike_dir = data_dir_base + "Mike/"
reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
plots_dir = data_dir_base + "00_plots/"

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_name_fips = pd.DataFrame({"state_full" : list(abb_dict["full_2_abb"].keys()),
                                "state" : list(abb_dict["full_2_abb"].values())})
state_name_fips = pd.merge(state_name_fips, abb_dict["state_fips"], on=["state"], how="left")
state_name_fips.head(2)

# %%
state_fips = abb_dict["state_fips"]
state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
print (len(state_fips_SoI))
state_fips_SoI.head(2)

# %% [markdown]
# ### Read beef and hay prices
#
# #### non-normalized

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%
all_df = all_data_dict["all_df_outerjoined"]
all_df_normal = all_data_dict["all_df_outerjoined_normalized"]
print ([x for x in list(all_df.columns) if "npp" in x])

# %%
all_df = rc.convert_lb_2_kg(df=all_df, 
                            matt_total_npp_col="total_matt_npp", 
                            new_col_name="metric_total_matt_npp")

# %%
len(all_df.state_fips.unique())

# %%
all_df.head(2)

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df.reset_index(drop=True, inplace=True)

# %%
print (all_df.shape)
# print (all_df_old.shape)

# %% [markdown]
# # Subset to states of interest
#
# **EW_meridian** exist only for the 29 states of interest. So, in the cell above we are automatically subseting.

# %%
all_df = all_df[all_df.state_fips.isin(list(state_fips_SoI.state_fips))].copy()
all_df.reset_index(drop=True, inplace=True)

# %%
all_df.head(2)

# %%
all_df["log_inventory"] = np.log(all_df["inventory"])
all_df["inventoryDiv1000"] = all_df["inventory"]/1000
all_df["log_total_matt_npp"] = np.log(all_df["total_matt_npp"])
all_df["total_matt_nppDiv1B"] = all_df["total_matt_npp"] / 1000000000

all_df["log_metric_total_matt_npp"] = np.log(all_df["metric_total_matt_npp"])
all_df["metric_total_matt_nppDiv10M"] = all_df["metric_total_matt_npp"] / 10000000
all_df["metric_total_matt_nppDiv500K"] = all_df["metric_total_matt_npp"] / 500000

# %%

# %%

# %%
print (all_df.metric_total_matt_npp.min())
print (all_df.metric_total_matt_npp.max())
print()
print (all_df.inventory.min())
print (all_df.inventory.max())
print("---------------------------------------------------------------------------")
print (all_df.metric_total_matt_nppDiv10M.min())
print (all_df.metric_total_matt_nppDiv10M.max())
print()
print (all_df.metric_total_matt_nppDiv500K.min())
print (all_df.metric_total_matt_nppDiv500K.max())
print()
print (all_df.inventoryDiv1000.min())
print (all_df.inventoryDiv1000.max())

# %%
453,592,370

# %%
# EW_meridian exist only for the 29 states of interest. So, here we are automatically subseting
needed_cols = ["year", "state_fips", 
               "metric_total_matt_npp", "metric_total_matt_nppDiv500K", "metric_total_matt_nppDiv10M",
               "log_metric_total_matt_npp",
               "total_matt_npp", "log_total_matt_npp", "total_matt_nppDiv1B",
               "inventory", "log_inventory", "inventoryDiv1000",
               "unit_matt_npp", "rangeland_acre",
               "max_ndvi_in_year_modis", 
               "beef_price_at_1982", "hay_price_at_1982",
               "EW_meridian", "state_dummy_int"]

inv_prices_ndvi_npp = all_df[needed_cols].copy()
inv_prices_ndvi_npp.head(2)

# %%
# [x for x in sorted(list(all_df.columns)) if not ("dumm" in x)]

# %%
inv_prices_ndvi_npp

# %%

# %%
len(inv_prices_ndvi_npp.state_fips.unique())

# %%
inv_prices_ndvi_npp.dropna(how="any", inplace=True)
inv_prices_ndvi_npp.reset_index(drop=True, inplace=True)
len(inv_prices_ndvi_npp.state_fips.unique())

# %%
[x for x in sorted(all_df.state_fips.unique()) if not (x in sorted(inv_prices_ndvi_npp.state_fips.unique()))]

# %%
all_df[all_df.state_fips == "21"].unit_matt_npp.unique()

# %%
state_fips = abb_dict["state_fips"]
state_fips[state_fips["state_fips"]=="21"]

# %%
inv_prices_ndvi_npp.head(2)

# %%
inv_prices_ndvi_npp.columns

# %% [markdown]
# # Add mean and variance of NPP to the data

# %%
mean_NPP = inv_prices_ndvi_npp.groupby(["state_fips"])["metric_total_matt_nppDiv10M"].mean()
mean_NPP = mean_NPP.reset_index()
mean_NPP.rename(columns={"metric_total_matt_nppDiv10M": "mean_metric_total_matt_nppDiv10M_2001_2020"}, 
                inplace=True)
mean_NPP.head(2)

# %%
var_NPP = inv_prices_ndvi_npp.groupby(["state_fips"])["metric_total_matt_nppDiv10M"].var()
var_NPP = var_NPP.reset_index()
var_NPP.rename(columns={"metric_total_matt_nppDiv10M": "var_metric_total_matt_nppDiv10M_2001_2020"}, inplace=True)
var_NPP.head(2)

# %%
inv_prices_ndvi_npp = pd.merge(inv_prices_ndvi_npp, mean_NPP, on=["state_fips"], how="left")
inv_prices_ndvi_npp = pd.merge(inv_prices_ndvi_npp, var_NPP, on=["state_fips"], how="left")

# %%
inv_prices_ndvi_npp.head(2)

# %%
# non-normal data
# log_inventory, inventoryDiv1000, C(EW_meridian)
# [inv_prices_ndvi_npp.EW_meridian=="E"]

depen_var_name = "inventory"
indp_vars = ["max_ndvi_in_year_modis", "beef_price_at_1982", "hay_price_at_1982"]

fit = ols(depen_var_name + " ~ " + " + ".join(indp_vars), 
           data=inv_prices_ndvi_npp).fit()

fit.summary()

# %%
short_summ = pd.DataFrame({"Coeff.": fit.params.values,          
                           "Std. Error": fit.bse.values.round(2),
                           "t": fit.tvalues.values,
                           "P-Value": fit.pvalues.values},
                           index = list(fit.params.index))
short_summ

# %%
spreg_fit = spreg.OLS(y = inv_prices_ndvi_npp[depen_var_name].values, # Dependent variable
                      x = inv_prices_ndvi_npp[indp_vars].values, # Independent variables
                      name_y=depen_var_name, # Dependent variable name
                      name_x=indp_vars)

pd.DataFrame({"Coeff.": spreg_fit.betas.flatten(),
              "Std. Error": spreg_fit.std_err.flatten(),
              "P-Value": [i[1] for i in spreg_fit.t_stat]},
              index = spreg_fit.name_x).round(4)

# %%
# variable_list = fit.params.index.to_list() 
# coef_dict = fit.params.to_dict()  # coefficient dictionary
# pval_dict = fit.pvalues.to_dict()  # pvalues dictionary
# std_error_dict = fit.bse.to_dict()

# aic_val = round(fit.aic, 2) # aic value
# adj_rsqured = round(fit.rsquared_adj, 3) # adjusted rsqured
# info_index = ['Num', 'AIC', 'Adjusted R2']
# index_list = variable_list + info_index

# %%
print (len(inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "E"].state_fips.unique()))
print (len(inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].state_fips.unique()))

# %%
state_fips_SoI[state_fips_SoI.state_fips == "47"]

# %%
len(inv_prices_ndvi_npp.state_fips.unique())

# %% [markdown]
# ## NPP Models

# %%
print (inv_prices_ndvi_npp.beef_price_at_1982.min())
print (inv_prices_ndvi_npp.beef_price_at_1982.max())
print ()
print (inv_prices_ndvi_npp.log_total_matt_npp.min())
print (inv_prices_ndvi_npp.log_total_matt_npp.max())
print ()
print (inv_prices_ndvi_npp.hay_price_at_1982.min())
print (inv_prices_ndvi_npp.hay_price_at_1982.max())

# %%
inv_prices_ndvi_npp[inv_prices_ndvi_npp.hay_price_at_1982 == inv_prices_ndvi_npp.hay_price_at_1982.max()]

# %%
inv_prices_ndvi_npp[inv_prices_ndvi_npp.year==2011].beef_price_at_1982.unique()

# %%
state_fips_SoI[state_fips_SoI.state_fips=="35"]

# %%
[x for x in inv_prices_ndvi_npp.columns if "npp" in x]

# %%
[x for x in inv_prices_ndvi_npp.columns if "inventory" in x]

# %%
inv_prices_ndvi_npp.columns

# %%
depen_var_name = "log_inventory"
indp_vars = ["metric_total_matt_nppDiv10M", "beef_price_at_1982", "hay_price_at_1982", 
             # "var_metric_total_matt_nppDiv10M_2001_2020"
            ]
m5 = spreg.OLS_Regimes(y = inv_prices_ndvi_npp[depen_var_name].values, # Dependent variable
                       x = inv_prices_ndvi_npp[indp_vars].values, # Independent variables

                       # Variable specifying neighborhood membership
                       regimes = inv_prices_ndvi_npp["EW_meridian"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var_name, # Dependent variable name
                       name_x=indp_vars)

print (m5.r2.round(2))

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results

# West regime
## Extract variables for the west side 
west_m = [i for i in m5_results.index if "W_" in i]

## Subset results to west side and remove the W_
west = m5_results.loc[west_m, :].rename(lambda i: i.replace("W_", ""))
## Build multi-index column names
west.columns = pd.MultiIndex.from_product([["West Meridian"], west.columns])

# East model
## Extract variables for the east side
east_m = [i for i in m5_results.index if "E_" in i]
east = m5_results.loc[east_m, :].rename(lambda i: i.replace("E_", ""))
## Build multi-index column names
east.columns = pd.MultiIndex.from_product([["East Meridian"], east.columns])

# Concat both models
pd.concat([east, west], axis=1).round(5)

# %%
m5.chow.joint

# %% [markdown]
# The next step then is to check whether each of the coefficients in our model differs across regimes. For this, we can pull them out into a table:

# %%
pd.DataFrame(m5.chow.regi, # Chow results by variable
             index = m5.name_x_r, # Name of variables
             columns = ["Statistic", "P-value"])

# %%
### west of Meridian
# + C(state_dummy_int)
# + C(EW_meridian)
# [inv_prices_ndvi_npp.EW_meridian == "W"]
# + beef_price_at_1982 + hay_price_at_1982
x_vars = ["metric_total_matt_nppDiv10M", "beef_price_at_1982", "hay_price_at_1982", 
          "var_metric_total_matt_nppDiv10M_2001_2020"]
fit = ols('inventoryDiv1000 ~ ' + "+".join(x_vars),
          data = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"]).fit() 

print (f"{fit.pvalues['metric_total_matt_nppDiv10M'] = }")
fit.summary()

# %%

# %%
y_var = "log_inventory" # log_inventory, inventoryDiv1000
# -1 came from Mike's Tutorial. sth about under "C(state_dummy_int) - 1" "C(EW_meridian)"
x_vars = ["metric_total_matt_nppDiv10M",  "beef_price_at_1982", "hay_price_at_1982"]

fit = ols(y_var + ' ~ ' + "+".join(x_vars), data = inv_prices_ndvi_npp).fit() 

print (f"{fit.pvalues['metric_total_matt_nppDiv10M'].round(3) = }")
fit.summary()

# %%
x_vars = ["metric_total_matt_nppDiv10M",  "beef_price_at_1982", "hay_price_at_1982",
          "var_metric_total_matt_nppDiv10M_2001_2020"]
yhats = fit.predict(inv_prices_ndvi_npp[x_vars])

d_ = {"yhats":yhats, y_var: inv_prices_ndvi_npp[y_var]}
y_and_hats = pd.DataFrame(d_)
y_and_hats.loc[y_and_hats.yhats.idxmin()]

# %%

# %%

# %%
# fit.params.filter(like="state_dummy_int")

# %%
print (inv_prices_ndvi_npp.shape)
inv_prices_ndvi_npp_west = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].copy()
print (inv_prices_ndvi_npp_west.shape)
inv_prices_ndvi_npp_west_noTexas = inv_prices_ndvi_npp_west[inv_prices_ndvi_npp_west.state_fips != "48"].copy()
print (inv_prices_ndvi_npp_west_noTexas.shape)

# %%
y_col = "inventoryDiv1000"
x_col = "metric_total_matt_nppDiv10M"
west_fit = ols(y_col + "~" +  x_col, data = inv_prices_ndvi_npp_west).fit()
west_noTexas_fit = ols(y_col + "~" +  x_col, data = inv_prices_ndvi_npp_west_noTexas).fit()

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=False, gridspec_kw={"hspace": 1, "wspace": 0.1})
axs[0].grid(axis="y", which="both"); axs[1].grid(axis="y", which="both")

####### Texas included predictions

x = inv_prices_ndvi_npp_west[x_col]
y = inv_prices_ndvi_npp_west[y_col]
axs[0].scatter(x, y, s = 20, c="dodgerblue", marker="x");

x_line = np.arange(min(x), max(x), 0.01)
y_line = west_fit.predict(pd.DataFrame(x_line).rename(columns = {0:x_col}))

axs[0].plot(x_line, y_line, color="r", linewidth=4, label="Texas Included")
axs[0].text(min(x), max(y)-.2, 'West of meridian.')

axs[0].set_xlabel(x_col);
axs[0].set_ylabel(y_col);

####### Texas Excluded predictions
x_noTexas = inv_prices_ndvi_npp_west_noTexas[x_col]
y_noTexas = inv_prices_ndvi_npp_west_noTexas[y_col]

x_line_noTexas = np.arange(min(x_noTexas), max(x_noTexas), 0.01)
y_line_noTexas = west_noTexas_fit.predict(pd.DataFrame(x_line_noTexas).rename(columns = {0:x_col}))
axs[0].plot(x_line_noTexas, y_line_noTexas, color="k", linewidth=4, label="Texas Removed")
axs[0].legend(loc="lower right")

####### Zoom in on No Texas on the right
axs[1].scatter(x_noTexas, y_noTexas, s = 20, c="dodgerblue", marker="x");

x_line_noTexas = np.arange(min(x_noTexas), max(x_noTexas), 0.01)
y_line_noTexas = west_noTexas_fit.predict(pd.DataFrame(x_line_noTexas).rename(columns = {0:x_col}))
axs[1].plot(x_line_noTexas, y_line_noTexas, color="k", linewidth=4, label="Texas Removed")
axs[1].legend(loc="lower right")
axs[1].set_xlabel(x_col);
axs[1].text(min(x_noTexas), max(y_noTexas)-.2, 'West of meridian. No texas')

fig_name = plots_dir + y_col + "_" + x_col + "_WestMeridian.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
print (west_fit.params)
print ()
print (west_noTexas_fit.params)

# %%
df_ = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].copy()
df_[df_["inventoryDiv1000"]>3000].state_fips.unique()

# %%
state_fips_SoI[state_fips_SoI.state_fips=="48"]

# %%
tick_legend_FontSize = 12

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.5,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
df = inv_prices_ndvi_npp.copy()
y_col = "metric_total_matt_npp"
df = df[["year", y_col, "state_fips"]]

df = df[df.state_fips == "48"]
df.dropna(subset=["year"], inplace=True)
df.dropna(subset=[y_col], inplace=True)

fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(which="both"); # axs.grid(axis="y", which="both")

axs.plot(df.year, df[y_col], color="dodgerblue", linewidth=4);
axs.set_xticks(np.arange(2001, 2021, 2))
axs.set_xlabel("year");
axs.set_ylabel(y_col.replace("_", " "));

axs.title.set_text(y_col.replace("_", " ") + " in Texas (kg)")

fig_name = plots_dir + "Texas_" + y_col + "_WestMeridian.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
tick_legend_FontSize = 8

params = {
    "legend.fontsize": tick_legend_FontSize,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.5,
    "axes.titlesize": tick_legend_FontSize * 1.3,
    "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
### Remove that outlier in (20, 13)
# for ```log_inventory``` against ```log_metric_total_matt_npp```


### west of Meridian
# + C(state_dummy_int)
# + C(EW_meridian)
# [inv_prices_ndvi_npp.EW_meridian == "W"]
# + beef_price_at_1982 + hay_price_at_1982

# %%
df_ = inv_prices_ndvi_npp.copy()
df_ = df_[df_.EW_meridian == "W"]
df_ = df_[["log_inventory", "log_metric_total_matt_npp"]]

m_ = df_["log_metric_total_matt_npp"].min()

print (df_.shape)
df_noMin = df_[df_["log_metric_total_matt_npp"] != m_]
print (df_noMin.shape)

log_log_noMin_fit = ols('log_inventory ~ log_metric_total_matt_npp', data = df_noMin).fit() 
log_log_fit = ols('log_inventory ~ log_metric_total_matt_npp', data = df_).fit() 
print (f"{log_log_noMin_fit.pvalues['log_metric_total_matt_npp'] = }")

######## Plot
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(axis="y", which="both")

x_col = "log_metric_total_matt_npp"
y_col = "log_inventory"

axs.scatter(df_[x_col], df_[y_col], s = 20, c="r", marker="x");
axs.scatter(df_noMin[x_col], df_noMin[y_col], s = 20, c="g", marker="x");

# minumum excluded fit
x_line = np.arange(min(df_noMin[x_col]), max(df_noMin[x_col]), 0.01)
y_line = log_log_noMin_fit.predict(pd.DataFrame(x_line).rename(columns = {0:x_col}))
axs.plot(x_line, y_line, color="dodgerblue", linewidth=4, label="minimum excluded for fit")


# minumum included fit
x_line = np.arange(min(df_[x_col]), max(df_[x_col]), 0.01)
y_line = log_log_fit.predict(pd.DataFrame(x_line).rename(columns = {0:x_col}))
axs.plot(x_line, y_line, color="r", linewidth=4, label="minimum included for fit")

axs.legend(loc="lower right")
plt.text(min(df_[x_col]), max(df_[y_col])-.2, 'West of meridian.')
axs.set_xlabel(x_col.replace("_", " "));
axs.set_ylabel(y_col.replace("_", " "));

fig_name = plots_dir + "log_log_metric_westMeridian.pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
# # !pip3 install spreg

# %%
inv_prices_ndvi_npp.head(2)

# %%
list(inv_prices_ndvi_npp.columns)

# %%
inv_prices_ndvi_npp["W_meridian_bool"] = True
inv_prices_ndvi_npp.loc[inv_prices_ndvi_npp.EW_meridian=="E", "W_meridian_bool"] = False

inv_prices_ndvi_npp.W_meridian_bool.head(2)

# %%
26786792214/755465

# %%
### Washington
inv_prices_ndvi_npp.total_matt_npp = np.log(inv_prices_ndvi_npp.total_matt_npp)

fit = ols('inventory ~ total_matt_npp + beef_price_at_1982 + hay_price_at_1982 + C(EW_meridian)', 
          data = inv_prices_ndvi_npp[inv_prices_ndvi_npp.state_fips=="53"]).fit() 

inv_prices_ndvi_npp.total_matt_npp = np.exp(inv_prices_ndvi_npp.total_matt_npp)
fit.summary()

# %% [raw]
# m4 = spreg.OLS_Regimes(
#     # Dependent variable
#     db[["log_price"]].values,
#     # Independent variables
#     db[variable_names].values,
#     # Variable specifying neighborhood membership
#     db["neighborhood"].tolist(),
#     # Allow the constant term to vary by group/regime
#     constant_regi="many",
#     # Variables to be allowed to vary (True) or kept
#     # constant (False). Here we set all to False
#     cols2regi=[False] * len(variable_names),
#     # Allow separate sigma coefficients to be estimated
#     # by regime (False so a single sigma)
#     regime_err_sep=False,
#     # Dependent variable name
#     name_y="log_price",
#     # Independent variables names
#     name_x=variable_names,
# )
#
# # Allow different coefficients per region
# # Pysal spatial regimes implementation
# m5 = spreg.OLS_Regimes(
#     # Dependent variable
#     db[["log_price"]].values,
#     # Independent variables
#     db[variable_names].values,
#     # Variable specifying neighborhood membership
#     db["coastal"].tolist(),
#     # Allow the constant term to vary by group/regime
#     constant_regi="many",
#     # Allow separate sigma coefficients to be estimated
#     # by regime (False so a single sigma)
#     regime_err_sep=False,
#     # Dependent variable name
#     name_y="log_price",
#     # Independent variables names
#     name_x=variable_names,
# )

# %% [markdown]
# # NDVI and rangeland area herb ratio

# %%
inv_prices_ndvi_npp.columns

# %%
depen_var_name = "log_inventory"
indp_vars = ["max_ndvi_in_year_modis", "beef_price_at_1982", "hay_price_at_1982", "rangeland_acre"]
m5 = spreg.OLS_Regimes(y = inv_prices_ndvi_npp[depen_var_name].values, # Dependent variable
                       x = inv_prices_ndvi_npp[indp_vars].values, # Independent variables

                       # Variable specifying neighborhood membership
                       regimes = inv_prices_ndvi_npp["EW_meridian"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                       
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var_name, # Dependent variable name
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results

# West regime
## Extract variables for the west side regime
west_m = [i for i in m5_results.index if "W_" in i]

## Subset results to west side and remove the W_
west = m5_results.loc[west_m, :].rename(lambda i: i.replace("W_", ""))
## Build multi-index column names
west.columns = pd.MultiIndex.from_product([["West Meridian"], west.columns])

# East model
## Extract variables for the eastern regime
east_m = [i for i in m5_results.index if "E_" in i]
east = m5_results.loc[east_m, :].rename(lambda i: i.replace("E_", ""))
## Build multi-index column names
east.columns = pd.MultiIndex.from_product([["East Meridian"], east.columns])
# Concat both models
pd.concat([east, west], axis=1)

# %%
depen_var_name = "log_inventory" # C(EW_meridian)
indp_vars = ["max_ndvi_in_year_modis", "beef_price_at_1982",
             "hay_price_at_1982", "rangeland_acre", "C(EW_meridian)"]

fit = ols(depen_var_name + " ~ " + " + ".join(indp_vars), data=inv_prices_ndvi_npp).fit()

print (f"{fit.rsquared.round(2) = }")
pd.DataFrame({"Coeff.": fit.params.values,          
              "Std. Error": fit.bse.values.round(2),
              "t": fit.tvalues.values,
              "P-Value": fit.pvalues.values},
              index = list(fit.params.index))

# %% [markdown]
# # Indp. Lagged

# %%
lag_vars = ['metric_total_matt_npp', 'metric_total_matt_nppDiv10M', 'log_metric_total_matt_npp',
             'total_matt_npp', 'log_total_matt_npp', 'unit_matt_npp',
             'inventory', 'log_inventory', 'inventoryDiv1000',
             'beef_price_at_1982', 'hay_price_at_1982']

inv_prices_ndvi_npp_lagged = rc.add_lags(df = inv_prices_ndvi_npp, 
                                         merge_cols = ["state_fips", "year"], 
                                         lag_vars_ = lag_vars, 
                                         year_count = 3)

print (f"{inv_prices_ndvi_npp.shape = }")
print (f"{inv_prices_ndvi_npp.year.min() = }")
print (f"{inv_prices_ndvi_npp.year.max() = }")
inv_prices_ndvi_npp_lagged.dropna(subset=["hay_price_at_1982_lag3"], inplace=True)
print ()
print (f"{inv_prices_ndvi_npp_lagged.shape = }")
print (f"{inv_prices_ndvi_npp_lagged.year.min() = }")
print (f"{inv_prices_ndvi_npp_lagged.year.max() = }")

# %%
lagged_vars = sorted([x for x in inv_prices_ndvi_npp_lagged.columns if "lag" in x])

# %%

# %%
y_var = "inventoryDiv1000"

x_vars = ["metric_total_matt_nppDiv10M",      
          'metric_total_matt_nppDiv10M_lag1', # 'beef_price_at_1982_lag1', 'hay_price_at_1982_lag1',
          'metric_total_matt_nppDiv10M_lag2', # 'beef_price_at_1982_lag2', 'hay_price_at_1982_lag2',
          'metric_total_matt_nppDiv10M_lag3', # 'beef_price_at_1982_lag3', 'hay_price_at_1982_lag3',
          'beef_price_at_1982',      'hay_price_at_1982',
          "C(EW_meridian)"
         ]

fit = ols(y_var + ' ~ ' + "+".join(x_vars), data = inv_prices_ndvi_npp_lagged).fit() 
fit.summary()

# %%
inv_prices_ndvi_npp_lagged.columns

# %%
y_var = "inventoryDiv1000"

x_vars = ["metric_total_matt_nppDiv10M",      
          'metric_total_matt_nppDiv10M_lag1', # 'beef_price_at_1982_lag1', 'hay_price_at_1982_lag1',
          'metric_total_matt_nppDiv10M_lag2', # 'beef_price_at_1982_lag2', 'hay_price_at_1982_lag2',
          'metric_total_matt_nppDiv10M_lag3', # 'beef_price_at_1982_lag3', 'hay_price_at_1982_lag3',
          'beef_price_at_1982',      'hay_price_at_1982',
         ]

### Allows for different slopes per category:
m5 = spreg.OLS_Regimes(y = inv_prices_ndvi_npp_lagged[y_var].values, # Dependent variable
                       x = inv_prices_ndvi_npp_lagged[x_vars].values, # Independent variables

                       # Variable specifying neighborhood membership
                       regimes = inv_prices_ndvi_npp_lagged["EW_meridian"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(x_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y = y_var, # Dependent variable name 
                       name_x = x_vars)
                      
m5_results = pd.DataFrame({# Pull out regression coefficients and flatten
                           "Coeff.": m5.betas.flatten(),
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

# West regime
## Extract variables for the West regime
west_m = [i for i in m5_results.index if "W_" in i]

## Subset results to West and remove the 1_ underscore
west = m5_results.loc[west_m, :].rename(lambda i: i.replace("W_", ""))
## Build multi-index column names
west.columns = pd.MultiIndex.from_product([["West Meridian"], west.columns])

# East model
## Extract variables for the east regime
east_m = [i for i in m5_results.index if "E_" in i]
## Subset results to east and remove the 0_ underscore
east = m5_results.loc[east_m, :].rename(lambda i: i.replace("E_", ""))
## Build multi-index column names
east.columns = pd.MultiIndex.from_product([["East Meridian"], east.columns])

# Concat both models
result_ = pd.concat([east, west], axis=1)
result_

# %%
print (list(result_["West Meridian", "Coeff."].round(2)))

# %% [markdown]
# # Indp. Avg. Lagged

# %%
inv_prices_ndvi_npp.sort_values(by=["state_fips", "year"], inplace=True)
inv_prices_ndvi_npp.reset_index(inplace=True, drop=True)
inv_prices_ndvi_npp.head(2)

# %%
lag_vars_ = ["metric_total_matt_nppDiv10M", "beef_price_at_1982", "hay_price_at_1982"]

inv_prices_ndvi_npp_lagAvg3 = rc.add_lags_avg(df = inv_prices_ndvi_npp, lag_vars_ = lag_vars_, 
                                              year_count = 3, fips_name = "state_fips")

# %%
y_var = "inventoryDiv1000"

x_vars = ["metric_total_matt_nppDiv10M", 'metric_total_matt_nppDiv10M_lagAvg3',
          'beef_price_at_1982',          'hay_price_at_1982',
#           'beef_price_at_1982_lagAvg3', 'hay_price_at_1982_lagAvg3',
#           "C(EW_meridian)"
         ]

m5 = spreg.OLS_Regimes(y = inv_prices_ndvi_npp_lagAvg3[y_var].values, # Dependent variable
                       x = inv_prices_ndvi_npp_lagAvg3[x_vars].values, # Independent variables
                       regimes = inv_prices_ndvi_npp_lagAvg3["EW_meridian"].tolist(),
                       constant_regi="many",
                       regime_err_sep=False,
                       name_y = y_var, 
                       name_x = x_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(),
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)
# West regime
## Extract variables for the West regime
west_m = [i for i in m5_results.index if "W_" in i]
west = m5_results.loc[west_m, :].rename(lambda i: i.replace("W_", ""))
west.columns = pd.MultiIndex.from_product([["West Meridian"], west.columns])

# East model
## Extract variables for the east regime
east_m = [i for i in m5_results.index if "E_" in i]
east = m5_results.loc[east_m, :].rename(lambda i: i.replace("E_", ""))
east.columns = pd.MultiIndex.from_product([["East Meridian"], east.columns])

# Concat both models
result_ = pd.concat([east, west], axis=1)
result_

# %%
y_var = "inventoryDiv1000"

x_vars = ["metric_total_matt_nppDiv10M", 'metric_total_matt_nppDiv10M_lagAvg3',
          'beef_price_at_1982',          'hay_price_at_1982',
#          "C(EW_meridian)"
         ]

fit = ols(y_var + ' ~ ' + "+".join(x_vars), data = inv_prices_ndvi_npp_lagAvg3).fit() 
fit.summary()

# %%

# %% [markdown]
# # Residual Plots

# %%
y_var = "log_inventory"
# y_var = "inventoryDiv1000"

x_vars = ['metric_total_matt_nppDiv10M', 'beef_price_at_1982', 'hay_price_at_1982']

fit = ols(y_var + '~' + " + ".join(x_vars),
          data = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"]).fit() 

print (f"{fit.pvalues['metric_total_matt_nppDiv10M'] = }")

fit.summary()

# %%
sm.graphics.influence_plot(fit);

# %%
X = inv_prices_ndvi_npp[inv_prices_ndvi_npp.EW_meridian == "W"].copy()
y_pred = fit.predict(X[x_vars])
y_pred.min()

# %%
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(axis="y", which="both")

axs.axhline(y = 0, color = 'r', linestyle = '-', linewidth=4, zorder=0)
axs.scatter(y_pred, fit.resid, s = 20, c="dodgerblue", marker="x");
axs.set_xlabel("$\\hat y$");
axs.set_ylabel("residual");

# fig_name = plots_dir + ".pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %% [markdown]
# # External studentized residuals

# %%
from statsmodels.stats.outliers_influence import OLSInfluence

# %%
fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
axs.grid(axis="y", which="both")

# change order of layers: zorder=0
axs.axhline(y = 0, color = 'r', linestyle = '-', linewidth=4, zorder=0)
axs.scatter(y_pred, OLSInfluence(fit).resid_studentized_external, s = 20, c="dodgerblue", marker="x");
axs.set_xlabel("$\\hat y$");
axs.set_ylabel("externally studentized residual");

# fig_name = plots_dir + ".pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %% [markdown]
# # Residuals on the map
#
# train east and west separate: both with log and 1000-head.

    # %%
    depen_var_name = "inventoryDiv1000"
indp_vars = ["metric_total_matt_nppDiv10M", "beef_price_at_1982", "hay_price_at_1982"]
m5 = spreg.OLS_Regimes(y = inv_prices_ndvi_npp[depen_var_name].values, # Dependent variable
                       x = inv_prices_ndvi_npp[indp_vars].values, # Independent variables

                       # Variable specifying neighborhood membership
                       regimes = inv_prices_ndvi_npp["EW_meridian"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var_name, # Dependent variable name
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

residuals_1000head = m5.u

# %%
depen_var_name = "log_inventory"
indp_vars = ["metric_total_matt_nppDiv10M", "beef_price_at_1982", "hay_price_at_1982"]
m5 = spreg.OLS_Regimes(y = inv_prices_ndvi_npp[depen_var_name].values, # Dependent variable
                       x = inv_prices_ndvi_npp[indp_vars].values, # Independent variables

                       # Variable specifying neighborhood membership
                       regimes = inv_prices_ndvi_npp["EW_meridian"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var_name, # Dependent variable name
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

residuals_logInv = m5.u

# %%
A = {"state_fips" : list(inv_prices_ndvi_npp["state_fips"].values.flatten()), 
     "year" : list(inv_prices_ndvi_npp["year"].values.flatten()),
     "residuals_1000head" : list(residuals_1000head.flatten()),
     "residuals_logInv" : list(residuals_logInv.flatten())}
residuals = pd.DataFrame(A)
residuals.head(3)

# %%
residuals_median = residuals.groupby(["state_fips"]).median()
residuals_median.head(5)
residuals_median = residuals_median.reset_index().round(2)
residuals_median.drop(columns=["year"], inplace=True)

residuals_median.rename(columns={"residuals_1000head": "residuals_1000head_median", 
                                 "residuals_logInv": "residuals_logInv_median"}, inplace=True)

residuals_median.head(5)

# %%

# %%
good_col = ["state_fips", "residuals_1000head", "residuals_logInv"]
residuals_color = pd.merge(residuals.loc[residuals.year==2017, good_col],
                           residuals_median, on=["state_fips"], how="left")
residuals_color = residuals_color.round(2)
residuals_color.head(4)

# %%
import warnings
warnings.filterwarnings('ignore')

import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors

import geopandas as gpd
from shapely.geometry import Polygon
# import missingno as msno
# import wget
# import openpyxl
import math

# %%
# we need geometry of states which is in this file
# we have to add residuals to it to use for coloring the map
gdf = gpd.read_file(data_dir_base + 'cb_2018_us_state_500k.zip')
gdf.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

gdf.rename(columns={"statefp": "state_fips", "stusps": "residuals_logInv_median",
                      "name": "state_full", "stusps" : "state" }, inplace=True)

gdf = gdf[~gdf.state.isin(["PR", "VI", "AS", "GU", "MP"])]

gdf = pd.merge(gdf, residuals_color, on=["state_fips"], how="left")

gdf["SoI"] = 0
gdf.loc[gdf.state_full.isin(abb_dict["SoI"]), "SoI"] = 1


gdf.head(3)

# %%

# %%

# %%
tick_legend_FontSize = 15

params = {
    "legend.fontsize": tick_legend_FontSize * 1.5,  # medium, large
    # 'figure.figsize': (6, 4),
    "axes.labelsize": tick_legend_FontSize * 1.5,
    "axes.titlesize": tick_legend_FontSize * 1.8, # this changes legend title,
    "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
    "axes.titlepad": 10,
}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

title_font_size="40"

# %%
# This color map cannot change or it also need to change in 
# makeColorColumn() in rangeland_plot_core.py as well.
colormap = "YlGnBu" 

# %%
plot_dir = data_dir_base + "00_plots/residual_on_map/"
os.makedirs(plot_dir, exist_ok=True)

# %%
# **************************
# set the value column that will be visualised
variable = "residuals_1000head"

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf[variable].min(), gdf[variable].max() # math.ceil(gdf.pct_food_insecure.max())
# from https://matplotlib.org/stable/tutorials/colors/colormaps.html
gdf = rcp.makeColorColumn(gdf,variable,vmin,vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init':'epsg:2163'})


# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

# set the font for the visualization to Helvetica
hfont = {'fontname':'Helvetica'}

# add a title and annotation
title_ = 'Food Insecurity by Percentage of State Households\n2019-2021'
ax.set_title(title_, **hfont, fontdict={'fontsize': '42', 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

title_2 = "'Percentage of state households\nexperiencing food insecurity\n'"
cbax.set_title(title_2, **hfont, fontdict={'fontsize': '15', 'fontweight' : '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, \
                 norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
comma_fmt = FuncFormatter(lambda x, p: format(x/100, '.0%'))
fig.colorbar(sm, cax=cbax, format=comma_fmt)
tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)
# annotate the data source, date of access, and hyperlink
ann_txt_ = "Data: USDA Economic Research Service, accessed 15 Jan 23"
ax.annotate(ann_txt_, xy=(0.22, .085), xycoords='figure fraction', fontsize=14, color='#555555')


# create map
# Note: we're going state by state here because of unusual 
# coloring behavior when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = gdf[gdf.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')



# %%

# %%

# %%
# https://medium.com/@alex_44314/use-python-geopandas-to-make-a-us-map-with-alaska-and-hawaii-39a9f5c222c6

# **************************
# set the value column that will be visualised
variable = "residuals_1000head"

# make a column for value_determined_color in gdf
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = rcp.makeColorColumn(gdf, variable, vmin, vmax)

# create "visframe" as a re-projected gdf using EPSG 2163 for CONUS
visframe = gdf.to_crs({'init':'epsg:2163'})

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box

txt_ = "y = f(metric_total_matt_nppDiv10M, $b_p$, $h_p$)\n year=2017"
ax.set_title(txt_, fontdict={'fontsize': title_font_size, 'fontweight' : '1'})

fig = ax.get_figure() # Create colorbar legend
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = variable
cbax.set_title(txt_) #

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
# comma_fmt = FuncFormatter(lambda x, p: format(x/100, '.0%'))
comma_fmt = FuncFormatter(lambda x, p: format(int(x)))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)
# annotate the data source, date of access, and hyperlink
text_ = "Data: USDA Economic Research Service"  
# ax.annotate(text_, xy=(0.22, .085), xycoords='figure fraction', fontsize=14, color='#555555')


# Note: we're going state by state here because of unusual coloring behavior 
# when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = visframe[visframe.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

fig.savefig(plot_dir + 'residuals_2017_1000Head.pdf', dpi=200, bbox_inches="tight")

# %%
import plotly.graph_objects as go

variable = "residuals_1000head"

fig = go.Figure(data=go.Choropleth(
    locations= gdf['state'],
    z = gdf[variable].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=False,
    # text=df['text'], # hover text
    marker_line_color='white', # line markers between states
    colorbar_title = variable
))

fig.update_layout(
    title_text = "y = f(metric_total_matt_nppDiv10M, b_p, h_p),\n year=2017",
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)

fig.show()
fig.write_image(plot_dir + 'residuals_2017_1000Head_goFigure.pdf')

# %%
gdf.head(2)

# %%

# %%

# %%

# %%

# %%

# %%
# **************************
# set the value column that will be visualised
variable = "residuals_logInv"

vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = rcp.makeColorColumn(gdf,variable,vmin,vmax)

visframe = gdf.to_crs({'init':'epsg:2163'})

fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box

txt_ = "y = f(metric_total_matt_nppDiv10M, $b_p$, $h_p$)\n year=2017"
ax.set_title(txt_, fontdict={'fontsize': title_font_size, 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = variable
cbax.set_title(txt_, fontdict={'fontweight' : '0'}) # 'fontsize': '20', 

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = [] # reformat tick labels on legend
comma_fmt = FuncFormatter(lambda x, p: format(int(x)))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)

for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = gdf[gdf.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

fig.savefig(plot_dir + 'residuals_2017_logInv.pdf', dpi=200, bbox_inches="tight")

# %%

# %%
# **************************
# set the value column that will be visualised
variable = "residuals_1000head_median"

vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = rcp.makeColorColumn(gdf, variable,vmin,vmax)

visframe = gdf.to_crs({'init':'epsg:2163'})

fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box

txt_ = "y = f(metric_total_matt_nppDiv10M, $b_p$, $h_p$)\n year=median"
ax.set_title(txt_, fontdict={'fontsize': title_font_size, 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = variable
cbax.set_title(txt_, fontdict={'fontsize': '25', 'fontweight' : '0'}) # 

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = [] # reformat tick labels on legend
comma_fmt = FuncFormatter(lambda x, p: format(int(x)))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)

for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = gdf[gdf.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

fig.savefig(plot_dir + 'residuals_median_1000head.pdf', dpi=200, bbox_inches="tight")

# %%

# %%
# **************************
# set the value column that will be visualised
variable = "residuals_logInv_median"

vmin, vmax = gdf[variable].min(), gdf[variable].max() #math.ceil(gdf.pct_food_insecure.max())
gdf = rcp.makeColorColumn(gdf,variable,vmin,vmax)

visframe = gdf.to_crs({'init':'epsg:2163'})

fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box

txt_ = "y = f(metric_total_matt_nppDiv10M, $b_p$, $h_p$)\n year=median"
ax.set_title(txt_, fontdict={'fontweight' : '1'}) # 'fontsize': '20',

# Create colorbar legend
fig = ax.get_figure()
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = variable
cbax.set_title(txt_, fontdict={'fontsize': '25', 'fontweight' : '0'}) # 

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = [] # reformat tick labels on legend
comma_fmt = FuncFormatter(lambda x, p: format(int(x)))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)

for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = gdf[gdf.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

fig.savefig(plot_dir + 'residuals_median_logInv.pdf', dpi=200, bbox_inches="tight")

# %% [markdown]
# # Cross Sectional model:2017

# %%
cross_2017 = inv_prices_ndvi_npp.copy()
cross_2017 = cross_2017[cross_2017.year==2017]
cross_2017.head(3)

# %%
depen_var_name = "log_inventory"
indp_vars = ["metric_total_matt_nppDiv10M", "beef_price_at_1982", "hay_price_at_1982"]
m5 = spreg.OLS_Regimes(y = cross_2017[depen_var_name].values, # Dependent variable
                       x = cross_2017[indp_vars].values, # Independent variables

                       # Variable specifying neighborhood membership
                       regimes = cross_2017["EW_meridian"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var_name, # Dependent variable name
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

cross_2017_residuals_logInv = m5.u

# %%
cross_2017 = inv_prices_ndvi_npp.copy()
cross_2017 = cross_2017[cross_2017.year==2017]
depen_var_name = "inventoryDiv1000"
indp_vars = ["metric_total_matt_nppDiv10M", "beef_price_at_1982", "hay_price_at_1982"]
m5 = spreg.OLS_Regimes(y = cross_2017[depen_var_name].values, # Dependent variable
                       x = cross_2017[indp_vars].values, # Independent variables

                       # Variable specifying neighborhood membership
                       regimes = cross_2017["EW_meridian"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       # cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi="many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y=depen_var_name, # Dependent variable name
                       name_x=indp_vars)

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

cross_2017_residuals_inventoryDiv1000 = m5.u

# %%
A = {"state_fips" : list(cross_2017["state_fips"].values.flatten()), 
     # "year" : list(cross_2017["year"].values.flatten()),
     "cross_2017_residuals_1000head" : list(cross_2017_residuals_inventoryDiv1000.flatten()),
     "cross_2017_residuals_logInv" : list(cross_2017_residuals_logInv.flatten())}
cross_2017_residuals = pd.DataFrame(A).round(2)
cross_2017_residuals.head(3)

# %%
gdf_cross = gpd.read_file(data_dir_base + 'cb_2018_us_state_500k.zip')
gdf_cross.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

gdf_cross.rename(columns={"statefp": "state_fips", "stusps": "residuals_logInv_median",
                      "name": "state_full", "stusps" : "state" }, inplace=True)

gdf_cross = gdf_cross[~gdf_cross.state.isin(["PR", "VI", "AS", "GU", "MP"])]

gdf_cross = pd.merge(gdf_cross, cross_2017_residuals, on=["state_fips"], how="left")

gdf_cross["SoI"] = 0
gdf_cross.loc[gdf_cross.state_full.isin(abb_dict["SoI"]), "SoI"] = 1


gdf_cross.head(2)

# %%
# **************************
# set the value column that will be visualised
variable = "cross_2017_residuals_logInv"

vmin, vmax = gdf_cross[variable].min(), gdf_cross[variable].max() #math.ceil(gdf_cross.pct_food_insecure.max())
gdf_cross = rcp.makeColorColumn(gdf_cross,variable,vmin,vmax)

visframe = gdf_cross.to_crs({'init':'epsg:2163'})

fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box

txt_ = "y = f(metric_total_matt_nppDiv10M, $b_p$, $h_p$)\n year=cross 2017"
ax.set_title(txt_, fontdict={'fontweight' : '1'}) # 'fontsize': '20',

# Create colorbar legend
fig = ax.get_figure()
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = variable
cbax.set_title(txt_, fontdict={'fontsize': '25', 'fontweight' : '0'}) # 

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = [] # reformat tick labels on legend
comma_fmt = FuncFormatter(lambda x, p: format(float(x.round(3))))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)

for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = gdf_cross[gdf_cross.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

fig.savefig(plot_dir + 'cross_2017_residuals_logInv.pdf', dpi=200, bbox_inches="tight")

# %%

# %%

# %%

# %%
# **************************
# set the value column that will be visualised
variable = "cross_2017_residuals_1000head"

vmin, vmax = gdf_cross[variable].min(), gdf_cross[variable].max() #math.ceil(gdf_cross.pct_food_insecure.max())
gdf_cross = rcp.makeColorColumn(gdf_cross,variable,vmin,vmax)

visframe = gdf_cross.to_crs({'init':'epsg:2163'})

fig, ax = plt.subplots(1, figsize=(18, 14))
ax.axis('off') # remove the axis box

txt_ = "y = f(metric_total_matt_nppDiv10M, $b_p$, $h_p$)\n year=cross 2017"
ax.set_title(txt_, fontdict={'fontweight' : '1'}) # 'fontsize': '20',

# Create colorbar legend
fig = ax.get_figure()
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = variable
cbax.set_title(txt_, fontdict={'fontsize': '25', 'fontweight' : '0'}) # 

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = [] # reformat tick labels on legend
comma_fmt = FuncFormatter(lambda x, p: format(int(x)))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)

for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = gdf_cross[gdf_cross.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')

fig.savefig(plot_dir + 'cross_2017_residuals_1000head.pdf', dpi=200, bbox_inches="tight")

# %%

# %%

# %%

# %%

# %%

# %%
gdf_copy = gdf.copy()

# values = {"A": 0, "B": 1, "C": 2, "D": 3}
# gdf_copy.fillna(value=values)
# gdf_copy.fillna(100000, inplace=True)
gdf_copy.dropna(inplace=True)

# %%
# **************************
# set the value column that will be visualised
colormap = "YlOrBr"
variable = "residuals_1000head"

# make a column for value_determined_color in gdf_copy
# set the range for the choropleth values with the upper bound the rounded up maximum value
vmin, vmax = gdf_copy[variable].min(), gdf_copy[variable].max() #math.ceil(gdf_copy.pct_food_insecure.max())
gdf_copy = rcp.makeColorColumn(gdf_copy,variable,vmin,vmax)

# create "visframe" as a re-projected gdf_copy using EPSG 2163 for CONUS
visframe = gdf_copy.to_crs({'init':'epsg:2163'})

# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(18, 14))
# remove the axis box around the vis
ax.axis('off')

# set the font for the visualization to Helvetica
hfont = {'fontname':'Helvetica'}

# add a title and annotation
txt_ = "y = f(metric_total_matt_nppDiv10M, $b_p$, $h_p$)"
ax.set_title(txt_, **hfont, fontdict={'fontsize': '20', 'fontweight' : '1'})

# Create colorbar legend
fig = ax.get_figure()
# add colorbar axes to the figure
# This will take some iterating to get it where you want it [l,b,w,h] right
# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)
cbax = fig.add_axes([0.89, 0.21, 0.03, 0.31])   

txt_ = "residuals_1000head"
cbax.set_title(txt_, **hfont, fontdict={'fontsize': '20', 'fontweight' : '0'})

# add color scale
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# reformat tick labels on legend
sm._A = []
# comma_fmt = FuncFormatter(lambda x, p: format(x/100, '.0%'))
comma_fmt = FuncFormatter(lambda x, p: format(int(x)))
fig.colorbar(sm, cax=cbax, format=comma_fmt)

tick_font_size = 16
cbax.tick_params(labelsize=tick_font_size)
# annotate the data source, date of access, and hyperlink
text_ = "Data: USDA Economic Research Service"  
# ax.annotate(text_, xy=(0.22, .085), xycoords='figure fraction', fontsize=14, color='#555555')


# create map
# Note: we're going state by state here because of unusual coloring behavior 
# when trying to plot the entire dataframe using the "value_determined_color" column
for row in visframe.itertuples():
    if row.state not in ['AK','HI']:
        vf = visframe[visframe.state==row.state]
        c = gdf_copy[gdf_copy.state==row.state][0:1].value_determined_color.item()
        vf.plot(color=c, linewidth=0.8, ax=ax, edgecolor='0.8')


# fig.savefig(os.getcwd()+'study_area.pdf', dpi=400, bbox_inches="tight")
# bbox_inches="tight" keeps the vis from getting cut off at the edges in the saved png

# %%
print (f"{inv_prices_ndvi_npp.year.min() = }")
print (f"{inv_prices_ndvi_npp.year.max() = }")

# %%
