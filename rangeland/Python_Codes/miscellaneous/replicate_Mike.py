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
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_name_fips = pd.DataFrame({"state_full" : list(abb_dict["full_2_abb"].keys()),
                                "state" : list(abb_dict["full_2_abb"].values())})


state_name_fips.head(2)

# %%
state_name_fips = pd.merge(state_name_fips, 
                           abb_dict["state_fips"][["state_fips", "EW_meridian", "state"]], 
                           on=["state"], how="left")
state_name_fips.head(2)

# %%
state_fips_SoI = state_name_fips[state_name_fips.state.isin(SoI_abb)].copy()
state_fips_SoI.reset_index(drop=True, inplace=True)
state_fips_SoI.head(2)

# %%
# DF = pd.read_csv(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.csv")
DF = pd.read_pickle(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.sav")
DF = DF["NPP_NDVI_Invent_Mike_2May2024"]
DF.head(2)

# %%
DF_inventory = DF[["year", "inventory", "state_fips", "rangeland_sq_kilometer"]].copy()
DF_inventory.dropna(inplace=True)
DF_inventory.reset_index(inplace=True, drop=True)

DF_inventory["inv_per_sqKilometer"] = DF_inventory["inventory"] / DF_inventory["rangeland_sq_kilometer"]

print (DF_inventory.shape)
DF_inventory.head(2)

# %%
cc = "diff_of_logs_of_inv_per_sqKilometer"
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
        
        curr_diff[cc] = np.log(curr_DF.loc[curr_DF["year"] == a_year, "inv_per_sqKilometer"].item()) - \
                        np.log(curr_DF.loc[curr_DF["year"] == a_year-1, "inv_per_sqKilometer"].item())

        DF_inventory_diff = pd.concat([DF_inventory_diff, curr_diff])

# %%
cc = "ratio_of_logs_of_inv_per_sqKilometer"
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
        
        curr_ratio[cc] = np.log(curr_DF.loc[curr_DF["year"] == a_year, "inv_per_sqKilometer"].item()) / \
                        np.log(curr_DF.loc[curr_DF["year"] == a_year-1, "inv_per_sqKilometer"].item())

        DF_inventory_ratio = pd.concat([DF_inventory_ratio, curr_ratio])

# %%
DF_inventory_diff_ratio = pd.merge(DF_inventory_ratio, DF_inventory_diff, on=["year", "state_fips"], how="left")
DF_inventory_diff_ratio.head(2)

# %%
tick_legend_FontSize = 5
params = {"legend.fontsize": tick_legend_FontSize,
          "legend.title_fontsize" : tick_legend_FontSize * 1.3,
          "legend.markerscale" : 2,
          "axes.labelsize": tick_legend_FontSize * 1.3,
          "axes.titlesize": tick_legend_FontSize * 1.5,
          "xtick.labelsize": tick_legend_FontSize * 1.3,
          "ytick.labelsize": tick_legend_FontSize * 1.3,
          "axes.titlepad": 10,
          'axes.linewidth' : .05}
plt.rcParams.update(params)

# %%
state_fips_west = list(state_fips_SoI[state_fips_SoI["EW_meridian"] == "W"]["state_fips"].values)
state_fips_west[:3]

# %%

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 2), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : False})
axes.grid(axis='y', which='both', zorder=0)

DF_plot = DF_inventory_diff_ratio.copy()
DF_plot = DF_plot[DF_plot["state_fips"].isin(state_fips_west)]

sns.histplot(data=DF_plot["diff_of_logs_of_inv_per_sqKilometer"], 
             ax=axes, bins=40, kde=False, zorder=3, color="dodgerblue", linewidth=0.1);

axes.set_xlabel(r"log$_{t+1}$(inv/sqKilometer) - log$_{t}$(inv/sqKilometer)");

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

# %%
DF_noNA = DF.copy()
DF_noNA.dropna(inplace=True)
common_years = list(DF_noNA.year.unique())

# %%
DF_inventory_commonYears = DF_noNA[["year", "inventory", "state_fips", "rangeland_sq_kilometer"]].copy()
DF_inventory_commonYears.reset_index(inplace=True, drop=True)
DF_inventory_commonYears["inv_per_sqKilometer"] = DF_inventory_commonYears["inventory"] / \
                                                  DF_inventory_commonYears["rangeland_sq_kilometer"]

print (DF_inventory_commonYears.shape)
DF_inventory_commonYears.head(2)

# %%
cc = "diff_of_logs_of_inv_per_sqKilometer"
cols_ = ["year", "state_fips", cc]
DF_inventory_diff_commonYears = pd.DataFrame(columns=cols_)

for a_state in DF_inventory_commonYears.state_fips.unique():
    curr_DF = DF_inventory_commonYears[DF_inventory_commonYears["state_fips"] == a_state].copy()
    curr_DF.sort_values(by="year", inplace=True)
    curr_DF.reset_index(drop=True, inplace=True)
    
    for a_year in curr_DF.year.unique()[1:]:
        curr_diff = pd.DataFrame(index=range(1), columns=cols_)
        curr_diff["state_fips"] = a_state
        curr_diff["year"] = str(a_year) + "-" + str(a_year-1)
        
        curr_diff[cc] = np.log(curr_DF.loc[curr_DF["year"] == a_year, "inv_per_sqKilometer"].item()) - \
                        np.log(curr_DF.loc[curr_DF["year"] == a_year-1, "inv_per_sqKilometer"].item())

        DF_inventory_diff_commonYears = pd.concat([DF_inventory_diff_commonYears, curr_diff])

DF_inventory_diff_commonYears.head(2)

# %%

# %%
fig, axes = plt.subplots(1, 1, figsize=(5, 2), sharey=False, sharex=False, dpi=dpi_)
sns.set_style({'axes.grid' : False})
axes.grid(axis='y', which='both', zorder=0)

DF_plot = DF_inventory_diff_commonYears.copy()
DF_plot = DF_plot[DF_plot["state_fips"].isin(state_fips_west)]
# DF_plot = DF_plot[DF_plot.year.isin(common_years)]

sns.histplot(data=DF_plot["diff_of_logs_of_inv_per_sqKilometer"], 
             ax=axes, bins=40, kde=False, zorder=3, color="dodgerblue", linewidth=0.1);
axes.set_xlabel(r"log$_{t+1}$(inv/sqKilometer) - log$_{t}$(inv/sqKilometer)");
axes.set_title('only common years present', y=.95);

file_name = plot_dir + "difference_of_logInvPerSqKm_commonYrs.pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %%

# %% [markdown]
# # Figure 2 replication

# %%

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

file_name = plot_dir + "diff_of_logs_of_unitMetricNPP.pdf"
plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %% [markdown]
# ### Figure 2 replication: common years

# %%
c_ = ["year", "state_fips", "unit_matt_npp_kg_per_sq_kilometer", "total_matt_npp_kg"]
DF_NPP_commonYears = DF_noNA[c_].copy()
print (DF_NPP.shape)
print (DF_NPP_commonYears.shape)

DF_NPP_commonYears.dropna(inplace=True)
DF_NPP_commonYears.reset_index(inplace=True, drop=True)
print (DF_NPP_commonYears.shape)
DF_NPP_commonYears.head(2)

# %%
cc = "diff_of_logs_of_unitMetricNPP"
cols_ = ["year", "state_fips", cc]
DF_unitNPP_diff_commonYrs = pd.DataFrame(columns=cols_)

column_2_diff = "unit_matt_npp_kg_per_sq_kilometer"
print (column_2_diff)

for a_state in DF_NPP_commonYears.state_fips.unique():
    curr_DF = DF_NPP_commonYears[DF_NPP_commonYears["state_fips"] == a_state].copy()
    curr_DF.sort_values(by="year", inplace=True)
    curr_DF.reset_index(drop=True, inplace=True)
    
    for a_year in curr_DF.year.unique()[1:]:
        curr_diff = pd.DataFrame(index=range(1), columns=cols_)
        curr_diff["state_fips"] = a_state
        curr_diff["year"] = str(a_year) + "-" + str(a_year-1)
        
        curr_diff[cc] = np.log(curr_DF.loc[curr_DF["year"] == a_year, column_2_diff].item()) - \
                        np.log(curr_DF.loc[curr_DF["year"] == a_year-1, column_2_diff].item())

        DF_unitNPP_diff_commonYrs = pd.concat([DF_unitNPP_diff_commonYrs, curr_diff])
        
DF_unitNPP_diff_commonYrs.head(2)

# %%
fig, axes = plt.subplots(1, 1, figsize=(4, 2), sharey=False, sharex=False, dpi=dpi_,
                         gridspec_kw={'hspace': .5, 'wspace': .05})

axes.grid(axis='y', which='both', zorder=0)
# sns.set_style({'axes.grid' : False},)
# sns.set_context(rc={"lines.linewidth": .1})

DF_plot = DF_unitNPP_diff.copy()
DF_plot = DF_plot[DF_plot["state_fips"].isin(state_fips_west)]

sns.histplot(data=DF_plot["diff_of_logs_of_unitMetricNPP"], 
             ax=axes, bins=40, kde=False, zorder=3, color="dodgerblue", linewidth=0.1);

axes.set_xlabel("diff_of_logs_of_unitMetricNPP".replace("_", " "));
axes.set_title('only common years present', y=.95);

file_name = plot_dir + "diff_of_logs_of_unitMetricNPP_commonYrs.pdf"
# plt.savefig(file_name, dpi=dpi_, bbox_inches='tight')

# %% [markdown]
# ## Regression Replication

# %%
# DF = pd.read_csv(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.csv")
DF = pd.read_pickle(reOrganized_dir + "NPP_NDVI_Invent_Mike_2May2024.sav")
DF = DF["NPP_NDVI_Invent_Mike_2May2024"]

#### We must have inventory. So, drop NAs
DF.dropna(subset=['inventory'], inplace=True)
DF.reset_index(inplace=True, drop=True)
len(DF)

DF.head(2)

# %%
if len(DF["EW_meridian_x"])==(DF["EW_meridian_x"] == DF["EW_meridian_y"]).sum():
    DF.drop('EW_meridian_x', axis=1, inplace=True)
    DF.rename(columns={'EW_meridian_y': 'EW_meridian'}, inplace=True)

# %%
# ndvi_cols = [x for x in DF.columns if "ndvi" in x]
# npp_cols = [x for x in DF.columns if "npp" in x]

# cc = ["year", "inventory", "beef_price_at_1982", "hay_price_at_1982",
#       "rangeland_sq_kilometer", 'state', 'state_fips', 'state_full']
# needed_cols = cc + ndvi_cols + npp_cols

# DF_Inventory_NPP_NDVI = DF.copy()

# DF_Inventory_NPP_NDVI.dropna(inplace=True)
# len(DF_Inventory_NPP_NDVI)

# %%
print (len(DF))
DF_common_years = DF.copy()
DF_common_years.dropna(inplace=True)
print (len(DF_common_years))

# %%
DF.head(2)

# %%
DF["inv_per_sqkm"] = DF["inventory"] / DF["rangeland_sq_kilometer"]
DF.head(2)

# %%
DF.drop(columns=["inventory"], inplace=True)
DF.head(2)

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
print (DF_w_logRatioInv.shape)
DF_w_logRatioInv.head(3)

# %%
[x for x in sorted(DF.columns) if "npp" in x]

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
print (DF_w_NPP_diff.shape)
DF_w_NPP_diff.head(3)

# %%
[x for x in sorted(DF.columns) if "ndvi" in x]

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
print (DF_w_NDVI_diff.shape)
DF_w_NDVI_diff.head(3)

# %% [markdown]
# ### Model with x-deltas

# %%
DF_w_NPP_diff.head(2)

# %%
DF_w_logRatioInv.head(2)

# %%
y_deltaNPP_df = pd.merge(DF_w_logRatioInv, DF_w_NPP_diff, 
                         how="left", 
                         on=["year", "diff_year", "state_fips"])

print (f"{DF_w_NPP_diff.shape = }")
print (f"{DF_w_logRatioInv.shape = }")
print (f"{y_deltaNPP_df.shape = }")

y_deltaNPP_df.dropna(inplace=True)
y_deltaNPP_df.reset_index(inplace=True, drop=True)

print (f"{y_deltaNPP_df.shape = }")
y_deltaNPP_df.head(2)

# %%
print (y_deltaNPP_df.shape)
y_deltaNPP_df = pd.merge(y_deltaNPP_df, 
                         DF[["year", "state_fips", "beef_price_at_1982", "hay_price_at_1982"]], 
                         how="left", 
                         on=["year", "state_fips"])

print (y_deltaNPP_df.shape)
y_deltaNPP_df.head(2)

# %%

# %%
A.head(2)

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Form X/state-mean DataFrames

# %% [markdown]
# ### Model with x/state_mean

# %%

# %%
a_state = DF["state_fips"].unique()[0]
curr_state_df = DF[DF["state_fips"] == a_state].copy()
curr_state_df.sort_values(by="year", inplace=True)
curr_state_df.reset_index(drop=True, inplace=True)
years_ = sorted(curr_state_df["year"].unique())[1:]
a_year = years_[0]

curr_state_yr_df = curr_state_df[curr_state_df["year"].isin([a_year-1, a_year])].copy()
curr_state_yr_df

# %%

# %%

# %%

# %%
DF_noNA = DF.copy()
DF_noNA.dropna(inplace=True)
len(DF_noNA)

# %%
DF_common_NDVI_Inv = DF.copy()
print (len(DF_common_NDVI_Inv))
DF_common_NDVI_Inv.dropna(subset=['max_ndvi_in_year_modis'], inplace=True)
print (len(DF_common_NDVI_Inv))

# %%
npp_cols = [x for x in DF.columns if "npp" in x]
DF_common_NPP_Inv = DF.copy()
print (len(DF_common_NPP_Inv))
DF_common_NPP_Inv.dropna(subset=npp_cols, inplace=True)
print (len(DF_common_NPP_Inv))

# %%

# %%
