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

# %% [markdown]
# Dec. 2, 2024
#
# Mike had run some regression. He wanted me to replicate them to see if the results are identical

# %% [markdown]
# ### Why hay price is not available for all states prior to 1948?

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

from datetime import datetime, date
from scipy.linalg import inv

current_time = datetime.now().strftime("%H:%M:%S")
print("Today's date:", date.today())
print("Current Time =", current_time)

from pysal.model import spreg

# %%
dpi_ = 300

plot_dir = "/Users/hn/Documents/01_research_data/RangeLand/Mike_Results/plots/"
os.makedirs(plot_dir, exist_ok=True)

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

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_name_fips = pd.DataFrame({"state_full" : list(abb_dict["full_2_abb"].keys()),
                                "state" : list(abb_dict["full_2_abb"].values())})

state_name_fips = pd.merge(state_name_fips, 
                           abb_dict["state_fips"][["state_fips", "EW_meridian", "state"]], 
                           on=["state"], how="left")
state_name_fips.head(2)

# %%
state_fips_SoI = state_name_fips[state_name_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
state_fips_SoI.head(2)

# %%
state_fips_west = list(state_fips_SoI[state_fips_SoI["EW_meridian"] == "W"]["state_fips"].values)
state_fips_west[:3]

# %%

# %%
DF_csv = pd.read_csv(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.csv")
DF_csv.head(2)

# %%
# DF = pd.read_csv(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.csv")
DF = pd.read_pickle(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.sav")
DF = DF["NPP_NDVI_Invent_Mike_2May2024"]

if len(DF["EW_meridian_x"])==(DF["EW_meridian_x"] == DF["EW_meridian_y"]).sum():
    DF.drop('EW_meridian_x', axis=1, inplace=True)
    DF.rename(columns={'EW_meridian_y': 'EW_meridian'}, inplace=True)
###
### Subset years and drop Kentucky
###
DF = DF[DF.state_full != "Kentucky"]
DF = DF[DF["year"] >= 1984].copy()
DF = DF[DF["year"] <= 2021].copy()

DF["inv_per_sqkm"] = DF["inventory"] / DF["rangeland_sq_kilometer"]
DF["log_inv_per_sqkm"] = np.log(DF["inv_per_sqkm"])
 
DF.head(5)

# %%

# %%
DF_inventory = DF[["year", "state_fips", 
                   "inventory", "inv_per_sqkm", "log_inv_per_sqkm",
                   "rangeland_sq_kilometer"]].copy()
DF_inventory.dropna(inplace=True)
DF_inventory.reset_index(inplace=True, drop=True)
DF_inventory.head(2)

# %%
DF_inventory.describe()

# %%

# %%
cc = "diff_of_logs_of_inv_per_sqkm"
cols_ = ["year", "state_fips", cc]
DF_inventory_diff = pd.DataFrame(columns=cols_)

for a_state in DF_inventory.state_fips.unique():
    curr_DF = DF_inventory[DF_inventory["state_fips"] == a_state].copy()
    curr_DF.sort_values(by="year", inplace=True)
    curr_DF.reset_index(drop=True, inplace=True)
    
    for a_year in curr_DF.year.unique()[1:]:
        curr_diff = pd.DataFrame(index=range(1), columns=cols_)
        curr_diff["state_fips"] = a_state
        curr_diff["year"] = str(a_year) + "-" + str(a_year-1)
        
        curr_diff[cc] = np.log(curr_DF.loc[curr_DF["year"] == a_year, "inv_per_sqkm"].item()) - \
                        np.log(curr_DF.loc[curr_DF["year"] == a_year-1, "inv_per_sqkm"].item())

        DF_inventory_diff = pd.concat([DF_inventory_diff, curr_diff])

# %%

# %%
cc = "ratio_of_logs_of_inv_per_sqkm"
cols_ = ["year", "state_fips", cc]
DF_inventory_ratio = pd.DataFrame(columns=cols_)

for a_state in DF_inventory.state_fips.unique():
    curr_DF = DF_inventory[DF_inventory["state_fips"] == a_state].copy()
    curr_DF.sort_values(by="year", inplace=True)
    curr_DF.reset_index(drop=True, inplace=True)
    
    for a_year in curr_DF.year.unique()[1:]:
        curr_ratio = pd.DataFrame(index=range(1), columns=cols_)
        curr_ratio["state_fips"] = a_state
        curr_ratio["year"] = str(a_year) + "-" + str(a_year-1)
        
        curr_ratio[cc] = np.log(curr_DF.loc[curr_DF["year"] == a_year, "inv_per_sqkm"].item()) / \
                        np.log(curr_DF.loc[curr_DF["year"] == a_year-1, "inv_per_sqkm"].item())

        DF_inventory_ratio = pd.concat([DF_inventory_ratio, curr_ratio])

# %%
DF_inventory_diff_ratio = pd.merge(DF_inventory_ratio, DF_inventory_diff, on=["year", "state_fips"], how="left")
DF_inventory_diff_ratio.head(2)

# %%
tick_legend_FontSize = 5
params = {"legend.fontsize": tick_legend_FontSize,
          "legend.title_fontsize" : tick_legend_FontSize * 1.3,
          "legend.markerscale" : 2,
          "axes.labelsize": tick_legend_FontSize * 2,
          "axes.titlesize": tick_legend_FontSize * 2,
          "xtick.labelsize": tick_legend_FontSize * 1.7,
          "ytick.labelsize": tick_legend_FontSize * 1.7,
          "axes.titlepad": 10,
          'axes.linewidth' : .05}
plt.rcParams.update(params)

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 2), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : False})
axes.grid(axis='y', which='both', zorder=0)

DF_plot = DF_inventory_diff_ratio.copy()
DF_plot = DF_plot[DF_plot["state_fips"].isin(state_fips_west)]

sns.histplot(data=DF_plot["diff_of_logs_of_inv_per_sqkm"], 
             ax=axes, bins=40, kde=False, zorder=3, color="dodgerblue", linewidth=0.1);

axes.set_xlabel(r"log$_{t+1}$(inv/sqKilometer) - log$_{t}$(inv/sqKilometer)");
axes.tick_params(length=2, width=.51) # axis='both', which='major', 
# min_loc = DF_inventory_diff_ratio["diff_of_logs_of_inv_per_sqKilometer"].min()
# text_ = min_loc + ", " + str(min_year)
# axes.text(df.loc[0, "season_length"]-2, 3, text_, fontsize = 12);

####################
# sns.histplot(data=DF_inventory_diff_ratio["ratio_of_logs_of_inv_per_sqKilometer"], 
#              ax=axes[1], bins=100, kde=False);
# axes[1].set_xlabel(r"log$_{t+1}$(inv/sqKilometer) / log$_{t}$(inv/sqKilometer)");

# fig.suptitle('log(yield) distribution (yield: bushel/acre)', y=0.95, fontsize=suptitle_fontsize)
axes.set_title('all years of inventory present', y=.95);
file_name = plot_dir + "difference_of_logInvPerSqKm.pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %%
DF.head(2)

# %% [markdown]
# ## Figure 2 replication

# %%
DF_NPP = DF[["year", "state_fips", "unit_matt_npp_kg_per_sq_kilometer", "total_matt_npp_kg"]].copy()
print (DF_NPP.shape)
DF_NPP.dropna(inplace=True)
DF_NPP.reset_index(inplace=True, drop=True)
print (DF_NPP.shape)
DF_NPP.head(2)

# %%
cc = "diff_of_logs_of_unitMetricNPP"
cols_ = ["year", "state_fips", cc]
DF_unitNPP_diff = pd.DataFrame(columns=cols_)

column_2_diff = "unit_matt_npp_kg_per_sq_kilometer"
print (column_2_diff)

for a_state in DF_NPP.state_fips.unique():
    curr_DF = DF_NPP[DF_NPP["state_fips"] == a_state].copy()
    curr_DF.sort_values(by="year", inplace=True)
    curr_DF.reset_index(drop=True, inplace=True)
    
    for a_year in curr_DF.year.unique()[1:]:
        curr_diff = pd.DataFrame(index=range(1), columns=cols_)
        curr_diff["state_fips"] = a_state
        curr_diff["year"] = str(a_year) + "-" + str(a_year-1)
        
        curr_diff[cc] = np.log(curr_DF.loc[curr_DF["year"] == a_year, column_2_diff].item()) - \
                        np.log(curr_DF.loc[curr_DF["year"] == a_year-1, column_2_diff].item())

        DF_unitNPP_diff = pd.concat([DF_unitNPP_diff, curr_diff])
        
DF_unitNPP_diff.head(2)

# %%
fig, axes = plt.subplots(1, 1, figsize=(4, 2), sharey=False, sharex=False, dpi=dpi_,
                         gridspec_kw={'hspace': .5, 'wspace': .05})
sns.set_style({'axes.grid' : False})
axes.grid(axis='y', which='both', zorder=0)

DF_plot = DF_unitNPP_diff.copy()
DF_plot = DF_plot[DF_plot["state_fips"].isin(state_fips_west)]

sns.histplot(data=DF_plot["diff_of_logs_of_unitMetricNPP"], 
             ax=axes, bins=40, kde=False, zorder=3, color="dodgerblue", linewidth=0.1);

axes.set_xlabel("diff_of_logs_of_unitMetricNPP".replace("_", " "));
axes.set_title('all years of NPP present', y=.95);
axes.tick_params(length=2, width=.51) # axis='both', which='major', 

file_name = plot_dir + "diff_of_logs_of_unitMetricNPP.pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %% [markdown]
# ## Regression Replication

# %%
DF.head(2)

# %%
DF.describe()

# %% [markdown]
# ## Form Delta DataFrames
# Since for a given location, some years might be missing, we dont want to do this is a vectorized fashion!

# %%
# %%time
target_col = "inv_per_sqkm"
new_cols = ["year", "diff_year", "state_fips", "log_of_ratio_of_inv_per_sqkm"]
DF_w_logRatioInv = pd.DataFrame(index=list(DF.index), columns=new_cols)
print (DF_w_logRatioInv.shape)
DF_w_logRatioInv.head(2)

counter = 0
for a_state in DF["state_fips"].unique():
    curr_state_df = DF[DF["state_fips"] == a_state].copy()
    curr_state_df.sort_values(by="year", inplace=True)
    curr_state_df.reset_index(drop=True, inplace=True)
    
    years_ = sorted(curr_state_df["year"].unique())[1:]
    for a_year in years_:
        curr_state_yr_df = curr_state_df[curr_state_df["year"].isin([a_year-1, a_year])].copy()
        
        if len(curr_state_yr_df) == 2:
            curr_log_ratio = np.log(curr_state_yr_df[target_col].values[1] / 
                                    curr_state_yr_df[target_col].values[0])
            
            diff_year = str(a_year) + "_" + str(a_year-1)
            
            DF_w_logRatioInv.loc[counter, "state_fips"] = a_state
            DF_w_logRatioInv.loc[counter, "year"] = a_year
            DF_w_logRatioInv.loc[counter, "diff_year"] = diff_year
            DF_w_logRatioInv.loc[counter, "log_of_ratio_of_inv_per_sqkm"] = curr_log_ratio
            counter += 1
        elif len(curr_state_yr_df) > 2:
            print ("something is wrong: too many years", a_state, a_year)
        elif len(curr_state_yr_df) < 2:
            print ("something is wrong (too little yearss)", a_state, a_year)

DF_w_logRatioInv.dropna(inplace=True)
DF_w_logRatioInv["year"] = DF_w_logRatioInv["year"].astype(int)
DF_w_logRatioInv["log_of_ratio_of_inv_per_sqkm"] = DF_w_logRatioInv["log_of_ratio_of_inv_per_sqkm"].astype(float)
print (DF_w_logRatioInv.shape)
DF_w_logRatioInv.head(3)

# %%
Mike_DF = DF_w_logRatioInv.copy()
del(DF_w_logRatioInv)

# %% [markdown]
# ### NPP deltas

# %%
# %%time
target_col = "unit_matt_npp_kg_per_sq_kilometer"
new_col = "diff_unit_matt_npp_kg_per_sq_kilometer"

new_cols = ["year", "diff_year", "state_fips", new_col]
DF_w_NPP_diff = pd.DataFrame(index=list(DF.index), columns=new_cols)
print (DF_w_NPP_diff.shape)
DF_w_NPP_diff.head(2)

counter = 0
for a_state in DF["state_fips"].unique():
    curr_state_df = DF[DF["state_fips"] == a_state].copy()
    curr_state_df.dropna(subset=[target_col], inplace=True)
    curr_state_df.sort_values(by="year", inplace=True)
    curr_state_df.reset_index(drop=True, inplace=True)
    
    years_ = sorted(curr_state_df["year"].unique())[1:]
    for a_year in years_:
        curr_state_yr_df = curr_state_df[curr_state_df["year"].isin([a_year-1, a_year])].copy()
        
        if len(curr_state_yr_df) == 2:
            curr_diff = curr_state_yr_df[target_col].values[1] - curr_state_yr_df[target_col].values[0]
            
            diff_year = str(a_year) + "_" + str(a_year-1)
            
            DF_w_NPP_diff.loc[counter, "state_fips"] = a_state
            DF_w_NPP_diff.loc[counter, "year"] = a_year
            DF_w_NPP_diff.loc[counter, "diff_year"] = diff_year
            DF_w_NPP_diff.loc[counter, new_col] = curr_diff
            counter += 1
        elif len(curr_state_yr_df) > 2:
            print ("something is wrong: too many years", a_state, a_year)
        elif len(curr_state_yr_df) < 2:
            print ("something is wrong (too little yearss)", a_state, a_year)

DF_w_NPP_diff.dropna(inplace=True)
DF_w_NPP_diff.reset_index(inplace=True, drop=True)

DF_w_NPP_diff["year"] = DF_w_NPP_diff["year"].astype(int)
c_ = "diff_unit_matt_npp_kg_per_sq_kilometer"
DF_w_NPP_diff[c_] = DF_w_NPP_diff[c_].astype(float)

print (DF_w_NPP_diff.shape)
DF_w_NPP_diff.head(3)

# %%
Mike_DF = pd.merge(Mike_DF, DF_w_NPP_diff, how="outer", on=["year", "diff_year", "state_fips"])
del(DF_w_NPP_diff)
Mike_DF.head(2)

# %%

# %% [markdown]
# ### NDVI deltas

# %%
# %%time
target_col = "max_ndvi_in_year_modis"
new_col = "diff_max_ndvi_in_year_modis"

new_cols = ["year", "diff_year", "state_fips", new_col]
DF_w_NDVI_diff = pd.DataFrame(index=list(DF.index), columns=new_cols)
print (DF_w_NDVI_diff.shape)
DF_w_NDVI_diff.head(2)

counter = 0
for a_state in DF["state_fips"].unique():
    curr_state_df = DF[DF["state_fips"] == a_state].copy()
    curr_state_df.dropna(subset=[target_col], inplace=True)
    curr_state_df.sort_values(by="year", inplace=True)
    curr_state_df.reset_index(drop=True, inplace=True)
    
    years_ = sorted(curr_state_df["year"].unique())[1:]
    for a_year in years_:
        curr_state_yr_df = curr_state_df[curr_state_df["year"].isin([a_year-1, a_year])].copy()
        
        if len(curr_state_yr_df) == 2:
            curr_diff = curr_state_yr_df[target_col].values[1] - curr_state_yr_df[target_col].values[0]
            
            diff_year = str(a_year) + "_" + str(a_year-1)
            
            DF_w_NDVI_diff.loc[counter, "state_fips"] = a_state
            DF_w_NDVI_diff.loc[counter, "year"] = a_year
            DF_w_NDVI_diff.loc[counter, "diff_year"] = diff_year
            DF_w_NDVI_diff.loc[counter, new_col] = curr_diff
            counter += 1
        elif len(curr_state_yr_df) > 2:
            print ("something is wrong: too many years", a_state, a_year)
        elif len(curr_state_yr_df) < 2:
            print ("something is wrong (too little yearss)", a_state, a_year)

DF_w_NDVI_diff.dropna(inplace=True)
DF_w_NDVI_diff.reset_index(inplace=True, drop=True)

DF_w_NDVI_diff["year"] = DF_w_NDVI_diff["year"].astype(int)
DF_w_NDVI_diff["diff_max_ndvi_in_year_modis"] = DF_w_NDVI_diff["diff_max_ndvi_in_year_modis"].astype(float)
print (DF_w_NDVI_diff.shape)
DF_w_NDVI_diff.head(3)

# %%
Mike_DF = pd.merge(Mike_DF, DF_w_NDVI_diff, how="outer", on=["year", "diff_year", "state_fips"])
del(DF_w_NDVI_diff)
Mike_DF.head(2)

# %%
Mike_DF = pd.merge(Mike_DF, 
                   DF[["year", "state_fips", "beef_price_at_1982", "hay_price_at_1982"]], 
                   how="outer", 
                   on=["year", "state_fips"])
Mike_DF.head(2)

# %% [markdown]
# ## Form relative X/state-mean DataFrames

# %%
DF_NPP_ratio = DF[["year", "state_fips", "unit_matt_npp_kg_per_sq_kilometer"]].copy()
DF_NPP_ratio.dropna(inplace=True)
DF_NPP_ratio.reset_index(inplace=True, drop=True)
DF_NPP_ratio.head(2)

# %%
DF_NPP_state_means = DF_NPP_ratio.groupby(["state_fips"]).mean().reset_index()
DF_NPP_state_means = DF_NPP_state_means[["state_fips", "unit_matt_npp_kg_per_sq_kilometer"]]
DF_NPP_state_means.rename(columns={'unit_matt_npp_kg_per_sq_kilometer': 
                                   'TemporalAvg_unit_matt_npp_kg_per_sq_kilometer'}, inplace=True)
DF_NPP_state_means.head(2)

# %%
DF_NPP_ratio = pd.merge(DF_NPP_ratio, DF_NPP_state_means, how="left", on="state_fips")
DF_NPP_ratio["NPP_div_by_AvgNPP_unitMetric"] = DF_NPP_ratio["unit_matt_npp_kg_per_sq_kilometer"] / \
                                               DF_NPP_ratio["TemporalAvg_unit_matt_npp_kg_per_sq_kilometer"]

DF_NPP_ratio.head(2)

# %%
DF_NPP_ratio.drop(['unit_matt_npp_kg_per_sq_kilometer',
                    "TemporalAvg_unit_matt_npp_kg_per_sq_kilometer"], axis=1, inplace=True)
DF_NPP_ratio.head(2)

# %%
Mike_DF = pd.merge(Mike_DF, DF_NPP_ratio, how="outer", on=["year", "state_fips"])
Mike_DF.head(2)

# %% [markdown]
# ### NDVI ratios

# %%
DF_NDVI_ratio = DF[["year", "state_fips", "max_ndvi_in_year_modis"]].copy()
DF_NDVI_ratio.dropna(inplace=True)
DF_NDVI_ratio.reset_index(inplace=True, drop=True)
DF_NDVI_ratio.head(2)

# %%
DF_NDVI_state_means = DF_NDVI_ratio.groupby(["state_fips"]).mean().reset_index()
DF_NDVI_state_means = DF_NDVI_state_means[["state_fips", "max_ndvi_in_year_modis"]]
DF_NDVI_state_means.rename(columns={'max_ndvi_in_year_modis': 
                                   'TemporalAvg_max_ndvi_in_year_modis'}, inplace=True)
DF_NDVI_state_means.head(2)

# %%
DF_NDVI_ratio = pd.merge(DF_NDVI_ratio, DF_NDVI_state_means, how="left", on="state_fips")
DF_NDVI_ratio["NDVI_div_by_AvgNDVI"] = DF_NDVI_ratio["max_ndvi_in_year_modis"] / \
                                               DF_NDVI_ratio["TemporalAvg_max_ndvi_in_year_modis"]

DF_NDVI_ratio.drop(['max_ndvi_in_year_modis',
                    "TemporalAvg_max_ndvi_in_year_modis"], axis=1, inplace=True)

DF_NDVI_ratio.head(2)

# %%
Mike_DF = pd.merge(Mike_DF, DF_NDVI_ratio, how="outer", on=["year", "state_fips"])
Mike_DF.head(2)

# %%
Mike_DF = pd.merge(Mike_DF, state_name_fips[["state_fips", "state_full", "EW_meridian"]],
                  how="left", on="state_fips")
Mike_DF.head(2)

# %%

# %% [markdown]
# ## Check some computations

# %%
yr_ = 1985
fips = "04"
DF.loc[(DF["year"].isin([yr_-1, yr_])) & (DF["state_fips"] == fips)]

# %%

# %%
# inventory / area
print ("inventory / area\n_____________________")
print (272000.0 / 220460.267862)
print (293000.0 / 220460.267862)
print ()
print ("log(inventory / area)\n_____________________")
print (np.log(272000.0 / 220460.267862))
print (np.log(293000.0 / 220460.267862))

# %%
yr_ = 2010
fips = "04"

DF.loc[(DF["year"].isin([yr_-1, yr_])) & (DF["state_fips"] == fips), "max_ndvi_in_year_modis"]

# %%
Mike_DF.loc[(Mike_DF["year"].isin([yr_-1, yr_])) & (Mike_DF["state_fips"] == fips)]

# %%
[x for x in Mike_DF.columns if "ndvi" in x]

# %%
print (0.246282 - 0.207875)
print (0.246282 / DF.loc[DF["state_fips"] == fips, "max_ndvi_in_year_modis"].mean())

# %%

# %%
Mike_DF.describe()

# %%
Mike_DF.head(2)

# %%
for a_state in sorted(Mike_DF["state_full"].unique()):
    df = Mike_DF[Mike_DF["state_full"] == a_state]
    df_inv = df[["year", "log_of_ratio_of_inv_per_sqkm"]].copy()
    df_delta_NPP = df[["year", "diff_unit_matt_npp_kg_per_sq_kilometer"]].copy()
    df_ratio_NPP = df[["year", "NPP_div_by_AvgNPP_unitMetric"]].copy()
    
    df_delta_ndvi = df[["year", "diff_max_ndvi_in_year_modis"]].copy()
    df_ratio_ndvi = df[["year", "NDVI_div_by_AvgNDVI"]].copy()
    
    df_inv.dropna(inplace=True)
    df_delta_NPP.dropna(inplace=True)
    df_ratio_NPP.dropna(inplace=True)
    df_delta_ndvi.dropna(inplace=True)
    df_ratio_ndvi.dropna(inplace=True)
    
    print (f"{a_state}, {len(df_inv)=}")
    print (f"{a_state}, {len(df_delta_NPP)=}")
    print (f"{a_state}, {len(df_ratio_NPP)=}")
    print (f"{a_state}, {len(df_delta_ndvi)=}")
    print (f"{a_state}, {len(df_ratio_ndvi)=}")
    print ("________________________________________")

# %%

# %%

# %% [markdown]
# # Start Modeling
#
# ### Model with x-deltas
#
# statsmodels.regression.linear_model

# %% [raw]
# indp_vars = ["county_total_npp"]
# y_var = "inventory"
#
# #################################################################
# X_npp = inv_2017_NPP_SW_heat_avg_normal[indp_vars]
# X_npp = sm.add_constant(X_npp)
# Y = np.log(inv_2017_NPP_SW_heat_avg_normal[y_var].astype(float))
# npp_inv_model = sm.OLS(Y, X_npp)
# npp_inv_model_result = npp_inv_model.fit()
# npp_inv_model_result.summary()

# %%

# %%
west_fips = state_fips_SoI.copy()
west_fips = west_fips[west_fips["EW_meridian"] == "W"]
west_fips = list(west_fips["state_fips"].values)

# %%

# %% [markdown]
# ### 1st Row of Mike's Table?

# %%
depen_var_name = "log_of_ratio_of_inv_per_sqkm"
indp_vars = ["diff_unit_matt_npp_kg_per_sq_kilometer", "beef_price_at_1982", "hay_price_at_1982"]
extra_cols = ["state_fips"]

df_model = Mike_DF.copy()
df_model = df_model[[depen_var_name] + indp_vars + extra_cols].copy()
df_model.dropna(inplace=True)
df_model.reset_index(inplace=True, drop=True)
print (f"Number of samples here is {len(df_model)}")


m5 = spreg.OLS_Regimes(y = df_model[depen_var_name].values,
                       x = df_model[indp_vars].values,
                       
                       # Variable specifying neighborhood membership
                       regimes = df_model["state_fips"].tolist(),
              
                       # Variables to be allowed to vary (True) or kept
                       # constant (False). Here we set all to False
                       cols2regi=[False] * len(indp_vars),
                        
                       # Allow the constant term to vary by group/regime
                       constant_regi = "many",
                        
                       # Allow separate sigma coefficients to be estimated
                       # by regime (False so a single sigma)
                       regime_err_sep=False,
                       name_y = depen_var_name, # Dependent variable name
                       name_x = indp_vars)

print (f"R2 = {m5.r2.round(2)}")

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results[25:]

# %%
df_model.describe()

# %%
depen_var_name = "log_of_ratio_of_inv_per_sqkm"
indp_vars = ["diff_unit_matt_npp_kg_per_sq_kilometer", "beef_price_at_1982", "hay_price_at_1982"]
extra_cols = ["state_fips"]

df_model = Mike_DF.copy()
df_model = df_model[[depen_var_name] + indp_vars + extra_cols].copy()
df_model.dropna(inplace=True)
df_model.reset_index(inplace=True, drop=True)
print (f"Number of samples here is {len(df_model)}")


m5 = spreg.OLS_Regimes(y = df_model[depen_var_name].values,
                       x = df_model[indp_vars].values,
                       regimes = df_model["state_fips"].tolist(),              
                       cols2regi=[False] * len(indp_vars),
                       constant_regi = "many",
                       regime_err_sep=False,
                       name_y = depen_var_name,
                       name_x = indp_vars)


m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

print (f"R2 = {m5.r2.round(2)}")
m5_results[25:]

# %%

# %% [markdown]
# ### 5th Row of Mike's Table?

# %%
#####
##### Just west Meridian
#####
depen_var_name = "log_of_ratio_of_inv_per_sqkm"
indp_vars = ["diff_unit_matt_npp_kg_per_sq_kilometer", "beef_price_at_1982", "hay_price_at_1982"]
extra_cols = ["state_fips"]

df_model = Mike_DF[[depen_var_name] + indp_vars + extra_cols].copy()
df_model = df_model[df_model["state_fips"].isin(west_fips)] # west side
df_model.dropna(inplace=True)
df_model.reset_index(inplace=True, drop=True)
print (f"Number of samples here is {len(df_model)}")

m5 = spreg.OLS_Regimes(y = df_model[depen_var_name].values,
                       x = df_model[indp_vars].values,
                       regimes = df_model["state_fips"].tolist(),

                       cols2regi=[False] * len(indp_vars),
                       constant_regi = "many",
                       regime_err_sep=False,
                       name_y = depen_var_name, # Dependent variable name
                       name_x = indp_vars)

print (m5.r2.round(2))

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results[15:]

# %%

# %% [markdown]
# ### 2nd row of Mike's Table?

# %%
#### All states
depen_var_name = "log_of_ratio_of_inv_per_sqkm"
indp_vars = ["NPP_div_by_AvgNPP_unitMetric", "beef_price_at_1982", "hay_price_at_1982"]
extra_cols = ["state_fips"]

df_model = Mike_DF[[depen_var_name] + indp_vars + extra_cols].copy()
df_model.dropna(inplace=True)
df_model.reset_index(inplace=True, drop=True)
print (f"Number of samples here is {len(df_model)}")

m5 = spreg.OLS_Regimes(y = df_model[depen_var_name].values,
                       x = df_model[indp_vars].values,
                       regimes = df_model["state_fips"].tolist(),
                       cols2regi=[False] * len(indp_vars),
                       constant_regi = "many",               
                       regime_err_sep=False,
                       name_y = depen_var_name,
                       name_x = indp_vars)

print (f"R2 = {m5.r2.round(2)}")

m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results[25:]

# %% [markdown]
# ### 6th row of Mike's Table?

# %%
#####
##### Just west Meridian
#####
depen_var_name = "log_of_ratio_of_inv_per_sqkm"
indp_vars = ["NPP_div_by_AvgNPP_unitMetric", "beef_price_at_1982", "hay_price_at_1982"]
extra_cols = ["state_fips"]

df_model = Mike_DF[[depen_var_name] + indp_vars + extra_cols].copy()
df_model = df_model[df_model["state_fips"].isin(west_fips)] # west side
df_model.dropna(inplace=True)
df_model.reset_index(inplace=True, drop=True)
print (f"Number of samples here is {len(df_model)}")

m5 = spreg.OLS_Regimes(y = df_model[depen_var_name].values,
                       x = df_model[indp_vars].values,
                       regimes = df_model["state_fips"].tolist(),
                       cols2regi=[False] * len(indp_vars),
                       constant_regi = "many",
                       regime_err_sep=False,
                       name_y = depen_var_name,
                       name_x = indp_vars)

print (f"R2 = {m5.r2.round(2)}")
m5_results = pd.DataFrame({"Coeff.": m5.betas.flatten(), # Pull out regression coefficients and
                           "Std. Error": m5.std_err.flatten(), # Pull out and flatten standard errors
                           "P-Value": [i[1] for i in m5.t_stat], # Pull out P-values from t-stat object
                           }, index=m5.name_x)

m5_results[15:]

# %%

# %%

# %%

# %%
