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
# When we met on June 10, 2024 Mike mentioned it would be good to look into how much and what kind of interaction/dynamic exist between inventory and slaughter. Lets see what we can do!
#
# I think here we need Jan 1 inventory. and then see how many were slaughtered thereafter during the same year. dammit.

# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os, os.path, pickle, sys

sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# %%
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# %%
data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
Shannon_data_dir = data_dir_base + "Shannon_Data/"
Min_data_dir_base = data_dir_base + "Min_Data/"

reOrganized_dir = data_dir_base + "reOrganized/"
os.makedirs(reOrganized_dir, exist_ok=True)

# %%
plot_dir = data_dir_base + "00_plots/slaughter_inventory_exploration/"

os.makedirs(plot_dir, exist_ok=True)

# %%

# %%
graph_dict = {"region_10" : ["region_8", "region_9"],
              "region_9" : ["region_6", "region_8", "region_10"],
              "region_8" : ["region_10", "region_9", "region_7", "region_6", "region_5"],
              "region_7" : ["region_8", "region_6", "region_5", "region_4"],
              "region_6" : ["region_9", "region_8", "region_7", "region_4"],
              "region_5" : ["region_8", "region_7", "region_4", "region_3"],
              "region_4" : ["region_7", "region_6", "region_5", "region_3"],
              "region_3" : ["region_5", "region_4", "region_1_region_2"],
              "region_1_region_2" : ["region_3"]
               }

region_Adj = np.array([
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 1, 1, 0, 0, 0],
                       [0, 1, 1, 0, 0, 1, 1, 0, 0],
                       [0, 0, 1, 0, 0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0, 1, 1],
                       [0, 0, 0, 0, 1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1, 0],
                      ])
region_Adj

# %%
# These colors are from US_map_study_area.py to be consistent with regions.
col_dict = {"R1&2": "cyan",
            "R3": "black", 
            "R4": "green",
            "R5": "tomato",
            "R6": "red",
            "R7": "dodgerblue",
            "R8": "dimgray", # gray: "#C0C0C0"
            "R9": "#ffd343", # mild yellow
            "R10": "steelblue"}

# %%
import networkx as nx
np.random.seed(11)

G = nx.DiGraph(directed=True)
G.add_weighted_edges_from([('R1&2', 'R3', 3.0), 
                  ('R3', 'R4', 1), ('R3', 'R5', 1), 
                  ('R4', 'R5', 1), ('R4', 'R6', 1), ('R4', 'R7', 1),
                  ('R5', 'R7', 1), ('R5', 'R8', 1),
                  ('R6', 'R7', 1), ('R6', 'R8', 1), ('R6', 'R9', 1),
                  ('R7', 'R8', 1),
                  ('R8', 'R9', 1), ('R8', 'R10', 1),
                  ('R9', 'R10', 1)])

# G.add_edges_from([('R1&2', 'R3'), 
#                   ('R3', 'R4'), ('R3', 'R5'), 
#                   ('R4', 'R5'), ('R4', 'R6'), ('R4', 'R7'),
#                   ('R5', 'R7'), ('R5', 'R8'),
#                   ('R6', 'R7'), ('R6', 'R8'), ('R4', 'R9'),
#                   ('R7', 'R8'),
#                   ('R8', 'R9'), ('R8', 'R10'),
#                   ('R9', 'R10')])

# G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])

values = [col_dict.get(node, 2) for node in G.nodes()]

options = {# 'node_color': 'dodgerblue',
           'node_color' : values,
           'node_size': 1000,
           'width': 2,
           'arrowstyle': '-|>',
           'arrowsize': 10,
           'font_color' : "white"}

nx.draw_networkx(G, arrows=True, **options)
limits = plt.axis("off")

# %%

# %%
# G=nx.Graph()
# i=1
# G.add_node(i, pos=(i, i))
# G.add_node(2, pos=(1, 2))
# G.add_node(3, pos=(1, 4))
# G.add_edge(1, 2, weight=0.5)
# G.add_edge(1, 3, weight=9.8)
# pos=nx.get_node_attributes(G,'pos')
# nx.draw(G,pos)
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=labels);

# %%
DGraph = nx.DiGraph()

vertex_list = ['R1&2', "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
edges_list = [('R1&2', 'R3', "M23"), 
              ('R3', 'R4', "M34"), ('R3', 'R5', "M35"), 
              ('R4', 'R5', "M45"), ('R4', 'R6', "M46"), ('R4', 'R7', "M47"),
              ('R5', 'R7', "M57"), ('R5', 'R8', "M58"),
              ('R6', 'R7', "M67"), ('R6', 'R8', "M68"), ('R6', 'R9', "M69"),
              ('R7', 'R8', "M78"),
              ('R8', 'R9', "M89"), ('R8', 'R10', "M810"),
              ('R9', 'R10', "M910")]

DGraph.add_nodes_from(vertex_list)
DGraph.add_weighted_edges_from(edges_list)

DGraph._node['R1&2']['pos'] = (10, 9)
DGraph._node['R3']['pos'] = (8, 7)
DGraph._node['R4']['pos'] = (5, 5)
DGraph._node['R5']['pos'] = (5, 9)
DGraph._node['R6']['pos'] = (-1, 5)
DGraph._node['R7']['pos'] = (2, 7)
DGraph._node['R8']['pos'] = (-2, 9)
DGraph._node['R9']['pos'] = (-6, 5)
DGraph._node['R10']['pos'] = (-7, 9)

node_pos=nx.get_node_attributes(DGraph,'pos')
arc_weight=nx.get_edge_attributes(DGraph,'weight')

nx.draw_networkx_nodes(DGraph , pos=node_pos, node_color=values, node_size=1000)
nx.draw_networkx_labels(DGraph , pos=node_pos, font_color="white")
nx.draw_networkx_edges(DGraph , node_pos, connectionstyle='arc3, rad = 0', 
                       arrowsize=20, # arrowstyle='-|>', label="S"
                       width=1,)
nx.draw_networkx_edge_labels(DGraph , node_pos , arc_weight)

plt.axis('off');

fig_name = plot_dir + "directedRegions.pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight");

# %%

# %%

# %%
# These colors are from US_map_study_area.py to be consistent with regions.
col_dict = {"region_1_region_2": "cyan",
            "region_3": "black", 
            "region_4": "green",
            "region_5": "tomato",
            "region_6": "red",
            "region_7": "dodgerblue",
            "region_8": "dimgray", # gray: "#C0C0C0"
            "region_9": "#ffd343", # mild yellow
            "region_10": "steelblue"}

# %%

# %%
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]
list(abb_dict.keys())

# %%
state_fips = abb_dict["state_fips"]
# state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
# state_fips_SoI.reset_index(drop=True, inplace=True)
# print (len(state_fips_SoI))
print (state_fips.shape)
state_fips.head(2)

# %%
state_fips = abb_dict["state_fips"]
# state_fips_SoI = state_fips[state_fips.state.isin(SoI_abb)].copy()
# state_fips_SoI.reset_index(drop=True, inplace=True)
# print(len(state_fips_SoI))
# state_fips_SoI.head(2)

# %%
filename = reOrganized_dir + "shannon_slaughter_data.sav"

slaughter = pd.read_pickle(filename)
print (slaughter["Date"])
(list(slaughter.keys()))

# %%
beef_slaught_complete_yrs = slaughter["beef_slaught_complete_yrs_tall"]
beef_slaught_complete_yrs.rename(columns={"slaughter_count": "slaughter"}, inplace=True)
beef_slaught_complete_yrs.head(2)

# %%
regions = list(beef_slaught_complete_yrs["region"].unique())
regions

# %% [markdown]
# ### Compute annual slaughter

# %%
monthly_slaughter = beef_slaught_complete_yrs[["year", "region", "slaughter"]].copy()
monthly_slaughter.dropna(subset=["slaughter"], inplace=True)
monthly_slaughter.reset_index(drop=True, inplace=True)
monthly_slaughter.head(2)

# %%
annual_slaughter = monthly_slaughter.groupby(["region", "year"])["slaughter"].sum().reset_index()
annual_slaughter.head(2)

# %%
df = annual_slaughter[annual_slaughter["region"] == "region_8"].copy()
print (df.shape)

df = annual_slaughter[annual_slaughter["region"] == "region_9"].copy()
print (df.shape)

# %% [markdown]
# ### Compute regional inventory

# %%
filename = reOrganized_dir + "state_data_and_deltas_and_normalDelta_OuterJoined.sav"
all_data_dict = pd.read_pickle(filename)
print (all_data_dict["Date"])
list(all_data_dict.keys())

# %%

# %%
all_data = all_data_dict["all_df_outerjoined"].copy()
inventory_df = all_data[["year", "inventory", "state_fips"]].copy()

inventory_df.dropna(subset=["inventory"], inplace=True)
inventory_df["inventory"] = inventory_df["inventory"].astype(int)
# do NOT subset to states of interest as slaughter data ignores that
# inventory_df = inventory_df[inventory_df["state_fips"].isin(list(state_fips_SoI["state_fips"].unique()))]
inventory_df.reset_index(drop=True, inplace=True)

print (f"{inventory_df.year.max() = }")
inventory_df.head(2)

# %%
inventory_df = pd.merge(inventory_df, state_fips[['state', 'state_fips', 'state_full']], 
                        on=["state_fips"], how="left")

inventory_df.head(2)

# %%
shannon_regions_dict_abbr = {"region_1_region_2" : ['CT', 'ME', 'NH', 'VT', 'MA', 'RI', 'NY', 'NJ'], 
                             "region_3" : ['DE', 'MD', 'PA', 'WV', 'VA'],
                             "region_4" : ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
                             "region_5" : ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
                             "region_6" : ['AR', 'LA', 'NM', 'OK', 'TX'],
                             "region_7" : ['IA', 'KS', 'MO', 'NE'],
                             "region_8" : ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
                             "region_9" : ['AZ', 'CA', 'HI', 'NV'],
                             "region_10": ['AK', 'ID', 'OR', 'WA']}

# %%
inventory_df["region"] = "NA"
for a_key in shannon_regions_dict_abbr.keys():
    inventory_df.loc[inventory_df["state"].isin(shannon_regions_dict_abbr[a_key]), 'region'] = a_key
inventory_df.head(2)

# %% [markdown]
# ### compute inventory in each region

# %%
region_inventory = inventory_df.copy()
region_inventory = region_inventory[["year", "region", "inventory"]]

region_inventory = region_inventory.groupby(["region", "year"])["inventory"].sum().reset_index()
region_inventory.head(2)

# %%
annual_slaughter.head(2)

# %% [markdown]
# ## Add one year to each inventory data so we have Jan 1st inventory.

# %%
region_inventory["year"] = region_inventory["year"] + 1

region_inventory.rename(columns={"inventory": "inventory_Jan1"}, inplace=True)

# slaughter goes only up to 2022. So, lets do that to inventory
region_inventory = region_inventory[region_inventory["year"] < 2023].copy()

region_inventory.head(2)

# %%
region_slaughter_inventory = pd.merge(region_inventory, annual_slaughter, 
                                      on=["region", "year"], how="outer")

print (f"{region_inventory.shape = }")
print (f"{annual_slaughter.shape = }")
print (f"{region_slaughter_inventory.shape = }")
region_slaughter_inventory.head(2)

# %%
annual_slaughter.head(3)

# %%
print (region_inventory.shape)
region_inventory[region_inventory.year == 2022]

# %%

# %%
#### it seems in some years some of the data are not available
# do we want to drop NAs? Region 8 has a lot of missing slaughter
# but its inventory might be important!

# region_slaughter_inventory.dropna(how="any", inplace=True)
region_slaughter_inventory.reset_index(drop=True, inplace=True)
print (region_slaughter_inventory.shape)

# %%
region_slaughter_inventory.head(2)

# %%
NotInteresting_regions_L = ["region_1_region_2", "region_3", "region_5"]
high_inv_regions = ["region_" + str(x) for x in [4, 6, 7, 8]]
low_inv_regions = ["region_" + str(x) for x in [9, 10]]

# %%
font = {'size' : 14}
matplotlib.rc('font', **font)

tick_legend_FontSize = 10

params = {"legend.fontsize": tick_legend_FontSize * 1.2,  # medium, large
          # 'figure.figsize': (6, 4),
          "axes.labelsize": tick_legend_FontSize * 1.2,
          "axes.titlesize": tick_legend_FontSize * 1.2,
          "xtick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
          "ytick.labelsize": tick_legend_FontSize * 1.1,  #  * 0.75
          "axes.titlepad": 10}

plt.rc("font", family="Palatino")
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.labelbottom"] = True
plt.rcParams["ytick.labelleft"] = True
plt.rcParams.update(params)

# %%
region_slaughter_inventory.head(2)

# %%

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs;
ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both")
y_var = "inventory_Jan1"
for a_region in high_inv_regions:
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax1.plot(df.year, df[y_var], 
             color = col_dict[a_region], linewidth=3, 
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #
    ax1.legend(loc="best");
    

for a_region in low_inv_regions:
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax2.plot(df.year, df[y_var], 
             color = col_dict[a_region], linewidth=3, 
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #
    ax2.legend(loc="best");
    
space = 5
ax1.xaxis.set_major_locator(ticker.MultipleLocator(space)) 

fig_name = plot_dir + "inventory_TS_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
region_slaughter_inventory.head(2)

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs;
ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both")
y_var = "slaughter"
for a_region in high_inv_regions:
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax1.plot(df.year, df[y_var], 
             color = col_dict[a_region], linewidth=3, 
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #
    ax1.legend(loc="best");
    

for a_region in low_inv_regions:
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax2.plot(df.year, df[y_var], 
             color = col_dict[a_region], linewidth=3, 
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #
    ax2.legend(loc="best");
    
fig_name = plot_dir + "slaughter_TS_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs;
ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both")

# fig.suptitle("slaughter: dashed lines");
ax1.title.set_text('slaughter: dashed lines')

y_var = "inventory_Jan1"
for a_region in high_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    ax1.plot(df.year, df[y_var], color = col_dict[a_region], linewidth=3,
            label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); # 

for a_region in low_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    ax2.plot(df.year, df[y_var], color = col_dict[a_region], linewidth=3,
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #

y_var = "slaughter"
for a_region in high_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    ax1.plot(df.year, df[y_var], linestyle="dashed",
             color = col_dict[a_region], linewidth=3)

for a_region in low_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax2.plot(df.year, df[y_var], linestyle="dashed",
             color = col_dict[a_region], linewidth=3)

ax1.legend(loc="best"); ax2.legend(loc="best");

space = 5
ax1.xaxis.set_major_locator(ticker.MultipleLocator(space)) 

fig_name = plot_dir + "inv_slau_TS_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
annual_slaughter.head(2)

# %%
region_inventory.head(2)

# %%
min_year = annual_slaughter.year.unique().min()
min_year = max([region_inventory.year.unique().min(), min_year])

# %%

# %%
print (region_slaughter_inventory.shape)
region_slaughter_inventory = region_slaughter_inventory[region_slaughter_inventory.year >= 1984].copy()
print (region_slaughter_inventory.shape)

# %% [markdown]
# # compute inventory deltas carefully
#
# There might be missing years!

# %%
region_slaughter_inventory.head(2)

# %%
region_inventory_Jan1_means = region_slaughter_inventory.groupby(["region"])["inventory_Jan1"].mean().reset_index()

region_inventory_Jan1_means.rename(columns={"inventory_Jan1": "inventory_Jan1_mean"}, inplace=True)
region_inventory_Jan1_means.head(2)

# %%
print (region_slaughter_inventory.shape)
region_slaughter_inventory = pd.merge(region_slaughter_inventory, region_inventory_Jan1_means, 
                                      on="region", how="left")
print (region_slaughter_inventory.shape)

region_slaughter_inventory.head(2)

# %%

# %%
inventory_annal_diff = pd.DataFrame()
for a_region in regions:
    curr_df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    curr_df = curr_df[['region', 'year', 'inventory_Jan1', 'inventory_Jan1_mean']].copy()
    curr_df.sort_values("year", inplace=True)
    curr_region_diff = pd.DataFrame(columns=["region", "year", "inventory_delta", 'inventory_Jan1_mean'])
    for a_year in sorted(curr_df.year.unique()):
        curr_df_yr = curr_df[curr_df.year.isin([a_year, a_year-1])].copy()
        if len(curr_df_yr) == 2:
            I_year = curr_df_yr.loc[curr_df_yr["year"] == a_year, "inventory_Jan1"].item()
            I_past_year = curr_df_yr.loc[curr_df_yr["year"] == a_year-1, "inventory_Jan1"].item()
            curr_diff = I_year - I_past_year

            perc_change = (curr_diff / I_past_year) * 100
            perc_change_rel_2_mean = (curr_diff / curr_df_yr.iloc[0]["inventory_Jan1_mean"]) * 100
            
            d = pd.DataFrame.from_dict({'region': [a_region], 
                                        'year': [str(a_year) + "_" + str(a_year-1)], 
                                        'inventory_delta': [curr_diff],
                                        "inv_perc_change" : [perc_change],
                                        "inv_perc_change_rel_2_mean" : [perc_change_rel_2_mean]
                                       })
            
            curr_region_diff = pd.concat([curr_region_diff, d])
    
    inventory_annal_diff = pd.concat([inventory_annal_diff, curr_region_diff])

inventory_annal_diff = inventory_annal_diff[["region", "year", 
                                             "inventory_delta", "inv_perc_change",
                                             "inv_perc_change_rel_2_mean"]]

inventory_annal_diff.reset_index(drop=True, inplace=True)
inventory_annal_diff["inventory_delta"] = inventory_annal_diff["inventory_delta"].astype(int)
inventory_annal_diff.reset_index(drop=True, inplace=True)

inventory_annal_diff.rename(columns={"year": "diff_years"}, inplace=True)
inventory_annal_diff.head(2)

# %%
inventory_annal_diff[(inventory_annal_diff["region"]== "region_6") & 
                     (inventory_annal_diff["inventory_delta"] < 0)].shape

# %%
fig, axs = plt.subplots(1, 1, figsize=(12, 3), sharex=True, gridspec_kw={"hspace":0.15, "wspace":0.05})
axs.grid(axis="y", which="both")
axs.grid(axis="x", which="major")

region = "region_6"
df = inventory_annal_diff.copy()
df = df[df["region"] == region].copy()
axs.plot(df["diff_years"], df.inventory_delta, linewidth=1,color="dodgerblue")
axs.scatter(df["diff_years"], df.inventory_delta, label="inventory delta " +  region)

plt.xticks(rotation=90);
# axs.plot(df.year, df.slaughter, linewidth=3, label="slaughter "+ region, 
#          color="dodgerblue", linestyle="dashed");
space = 1
axs.xaxis.set_major_locator(ticker.MultipleLocator(space)) 

plt.title("inventory deltas")
plt.legend(loc = "best");

# %%

# %% [markdown]
# ### plot inventory diff against slaughter

# %%
inventory_annal_diff.head(2)

# %%
inventory_annal_diff["year"] = inventory_annal_diff["diff_years"].str.split("_", expand=True)[1]
inventory_annal_diff["year"] = inventory_annal_diff["year"].astype(int)
inventory_annal_diff.head(2)

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
(ax1, ax2) = axs;
ax1.grid(axis="both", which="both"); ax2.grid(axis="both", which="both")

# fig.suptitle("slaughter: dashed lines");
ax1.title.set_text('slaughter: dashed lines: multiplied by negative')

y_var = "inventory_delta"
for a_region in high_inv_regions:
    df = inventory_annal_diff.copy()
    df = df[df["region"] == a_region].copy()
    ax1.plot(df["year"], df[y_var], color = col_dict[a_region], linewidth=3,
            label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); # 

for a_region in low_inv_regions:
    df = inventory_annal_diff.copy()
    df = df[df["region"] == a_region].copy()
    ax2.plot(df["year"], df[y_var], color = col_dict[a_region], linewidth=3,
             label = y_var[:3].title() + ". " +  a_region.replace("_", " ").title()); #

y_var = "slaughter"

for a_region in high_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    ax1.plot(df.year, -df[y_var], linestyle="dashed", color = col_dict[a_region], linewidth=3)

for a_region in low_inv_regions:
    df = region_slaughter_inventory.copy()
    df = df[df["region"] == a_region].copy()
    df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
    ax2.plot(df.year, -df[y_var], linestyle="dashed", color = col_dict[a_region], linewidth=3)

ax1.legend(loc="best"); ax2.legend(loc="best");

space = 1
ax1.xaxis.set_major_locator(ticker.MultipleLocator(space))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(space))
# plt.xticks(rotation=90);

# ax1.xaxis.set_ticks(list(region_slaughter_inventory.year))
# ax2.xaxis.set_ticks(list(region_slaughter_inventory.year))

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90);
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90);

fig_name = plot_dir + "invDiff_slau_TS_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
inventory_annal_diff[(inventory_annal_diff.region == "region_6") &
                     (inventory_annal_diff.diff_years == "2012_2011")]

# %%
region_slaughter_inventory[(region_slaughter_inventory.region == "region_6") &
                           (region_slaughter_inventory.year == 2012)]

# %%
aa = inventory_annal_diff.loc[inventory_annal_diff.region == "region_6", "inventory_delta"].idxmin()
aa

# %%
inventory_annal_diff.loc[aa]

# %%
region_slaughter_inventory[(region_slaughter_inventory["region"] == "region_6") &
                           (region_slaughter_inventory["year"] == 2011)]

# %%
Rs = ["region_4", "region_7", "region_8", "region_9"]

region_slaughter_inventory[(region_slaughter_inventory["region"].isin(Rs)) &
                           (region_slaughter_inventory["year"].isin([2011])) ]

# %%
Rs = ["region_4", "region_7", "region_8", "region_9"]

inventory_annal_diff[(inventory_annal_diff["region"].isin(Rs)) &
                     (inventory_annal_diff["diff_years"].isin(["2012_2011"])) ]

# %%
annual_slaughter.head(2)

# %%
region_slaughter_inventory.head(2)

# %%
# ## inventory of region 8

# fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharex=True, gridspec_kw={"hspace": 0.15, "wspace": 0.05})
# axs.grid(axis="y", which="both");
# y_var = "inventory"

# for a_region in ["region_8"]:
#     df = region_slaughter_inventory[region_slaughter_inventory["region"] == a_region].copy()
#     axs.scatter(df["year"], df["inventory"], label="inventory " +  a_region)
#     axs.legend(loc="best");

# %%
inventory_annal_diff.sort_values(["region", "year"], inplace=True)
inventory_annal_diff.reset_index(drop=True, inplace=True)
inventory_annal_diff.head(4)

# %%
region_slaughter_inventory.sort_values(["region", "year"], inplace=True)
region_slaughter_inventory.reset_index(drop=True, inplace=True)
region_slaughter_inventory.head(4)

# %%
inventory_annal_diff = pd.merge(inventory_annal_diff, 
                                region_slaughter_inventory[['region', 'year', 'slaughter']], 
                                on=["region", "year"], how="left")

inventory_annal_diff.head(2)

# %% [markdown]
# # Inventory vs. Slaughter
#
# There are 3 cases:
#
#   - Inventory decline is less than slaughter. Which means some of slaughters came into the region to be slaughtered.
#   
#     e.g. inventory goes from 100 to 80 and slaughter is 50. So, 30 cows were impoprted to be killed.
#   
# --------
#   - Inventory decline is more than slaughter, which means some of inventory moved out of state?
#   
#     e.g. inventory goes from 100 to 20 and slaughter is 50. So, 30 cows are lost (moved out of state?)
# --------
#   - Inventory increases even tho there are slaughters. So, some cows are imported and added to the inventory.
#   
#     e.g. inventory goes from 100 to 120 and slaughter is 50. So, 70 cows are imported to the region.
#     either for slaughter or all added to the inventory.

# %%
inventory_annal_diff.head(2)

# %%
incresed_inv_case = inventory_annal_diff[inventory_annal_diff.inventory_delta > 0].copy()
incresed_inv_case.reset_index(drop=True, inplace=True)

print (f"{incresed_inv_case.shape = }")
incresed_inv_case.head(2)

# %%
inventory_annal_diff.index

# %%
# we need to look at the years of overlap
# between inventory and slaughter!

A = inventory_annal_diff.copy()
A = A[A.inventory_delta < 0].copy()
A = A[abs(A["inventory_delta"]) < A["slaughter"]].copy()
A.reset_index(drop=True, inplace=True)
slr_more_than_inv_decline = A.copy()
del(A)

print (slr_more_than_inv_decline.shape)
slr_more_than_inv_decline.head(2)

# %%
A = inventory_annal_diff.copy()
A = A[A.inventory_delta < 0].copy()
A = A[abs(A["inventory_delta"]) > A["slaughter"]].copy()
A.reset_index(drop=True, inplace=True)
slr_less_than_inv_decline = A.copy()
del(A)

print (slr_less_than_inv_decline.shape)
slr_less_than_inv_decline.head(2)

# %%
A = inventory_annal_diff.copy()
A = A[A.inventory_delta < 0].copy()
A = A[abs(A["inventory_delta"]) == A["slaughter"]].copy()
A.reset_index(drop=True, inplace=True)
slr_equal_inv_decline = A.copy()
del(A)

print (slr_equal_inv_decline.shape)
slr_equal_inv_decline.head(2)

# %% [markdown]
# # Region_1_2 is not intersting
#
# and there is only 1 case of ```region_10```. Let us look at ```region_8```.

# %%
slr_less_than_inv_decline.groupby(["region"]).count()

# %%
slr_less_than_inv_decline[slr_less_than_inv_decline.region == "region_8"]

# %%
aa_ = slr_less_than_inv_decline[slr_less_than_inv_decline.region == "region_8"].year.values
aa_

# %% [markdown]
# #### Look at neighbors and those years
#
# to see if we can find anything. 
#
# Slaughter has been less than inventory decline. So, if we assume inventory/slaughter data are correctly recorded.:
#
# - either some of inventory has gone to neighbors.
# - or some cows from other states have been imported here solely for slaughtering.
# - or both?
#
# So, we need to find a neghbor(s) whose
# - This is not doable. What if a neighboring state's slaughter is also contamintated by importing cows to it for only slaughtering? Lets say that has not happened. If they had capacity to slaughter, they would have slaughtered their own cows.
#
# So, we need to find a neghbor(s) whose
# - relation between ```inventory_decline``` and ```slaughter``` is opposited of what we have above; ```abs(inventory_decline) < slaughter```. BUT, those neighbors have neighbors of their own. They might be interacting with them too. it is a hard thing to attack in a brute force fashion!

# %%
# we are looking for this discrepancy
target_year = aa_[0]
df_ = slr_less_than_inv_decline[(slr_less_than_inv_decline.region == "region_8") &
                                (slr_less_than_inv_decline.year == target_year)]

abs(df_["inventory_delta"].item()) - df_["slaughter_count"].item()

# %%
explore_df = inventory_annal_diff[(inventory_annal_diff["region"].isin(graph_dict["region_8"])) &
                                  (inventory_annal_diff["year"] == target_year)].copy()

explore_df

# %%
explore_df["slaughter_minus_abs_inv_decline"] = explore_df["slaughter_count"] - abs(explore_df["inventory_delta"])
explore_df

# %%

# %%
region_slaughter_inventory.head(2)

# %%
inventory_annal_diff["invDelta_plusSla"] = inventory_annal_diff["inventory_delta"] + \
                                         inventory_annal_diff["slaughter_count"]

# %%
inventory_annal_diff[inventory_annal_diff["invDelta_plusSla"] < 0 ]

# %%
annual_slaughter.head(2)

# %%
region_inventory.head(2)

# %%
region_slaughter_inventory.head(2)

# %%
region_inventory[(region_inventory["region"] == "region_1_region_2") & 
                 (region_inventory["year"].isin([2013, 2012]))]

# %%
141500 - 159000 + 6100

# %%
annual_slaughter[(annual_slaughter["region"] == "region_1_region_2") & 
                 (annual_slaughter["year"].isin([2012]))]

# %%
inventory_annal_diff.head(2)

# %%
inventory_annal_diff.head(2)

# %%
inventory_annal_diff.inventory_delta.min()

# %%
inventory_annal_diff.inventory_delta.max()

# %%
print (inventory_annal_diff.shape)
inventory_annal_diff.dropna(how="any", inplace=True)
inventory_annal_diff.shape

# %%
fig, axs = plt.subplots(1, 1, figsize=(5*0.7, 5), sharex=True, gridspec_kw={"hspace":0.15, "wspace":0.05})
axs.grid(axis="y", which="both")
axs.grid(axis="x", which="major")
axs.set_axisbelow(True)

region = "region_6"
df = inventory_annal_diff.copy()
df = df[df["region"] == region].copy()

axs.scatter(df["inventory_delta"], df["slaughter_count"], label=region.replace("_", " ").title())
# plt.xticks(rotation=90);
axs.set_xlabel("inventory change")
axs.set_ylabel("slaughter")

# plt.title("inventory deltas")
plt.legend(loc = "best");

# %%
l = 5
fig, axs = plt.subplots(1, 1, figsize=(l*0.7, l), sharex=True, gridspec_kw={"hspace":0.15, "wspace":0.05})

axs.grid(axis="y", which="both")
axs.grid(axis="x", which="major")
axs.set_axisbelow(True) 

region = "region_6"
df = inventory_annal_diff.copy()
df = df[df["region"] == region].copy()

axs.scatter(-df["inventory_delta"], df["slaughter_count"], label=region.replace("_", " ").title())
# plt.xticks(rotation=90);
axs.set_xlabel("negative inventory change")
axs.set_ylabel("slaughter")

plt.legend(loc = "best");

# %%
len(inventory_annal_diff.region.unique())

# %%
fig, ax = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'hspace': 0.1, 'wspace': .15});

scale_1000 = 1e3

region_count = -1
for ii in [0, 1, 2]:
    for jj in [0, 1, 2]:
        region_count += 1
        region = regions[region_count]
        ax[ii][jj].grid(True);
        df = inventory_annal_diff.copy()
        df = df[df["region"] == region].copy()
        
        ax[ii][jj].scatter(-df["inventory_delta"], df["slaughter_count"], c = col_dict[region],
                           label=region.replace("_", " ").title())
        if region_count >= 6 :
            ax[ii][jj].set_xlabel("negative inventory change (1000 heads)")
        if region_count in [0, 3, 6]:
            ax[ii][jj].set_ylabel("slaughter (1000 heads)");
        ax[ii][jj].legend(loc = "best");
#        ax[ii][jj].set_yticklabels(ax[ii][jj].get_yticklabels(), rotation=0); # makes 10^6 go away!!!
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_1000))
        ax[ii][jj].yaxis.set_major_formatter(ticks_y)
        
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_1000))
        ax[ii][jj].xaxis.set_major_formatter(ticks_x)
        ax[ii][jj].set_axisbelow(True)
        
fig_name = plot_dir + "regionsSla_NegInvt_Scatter_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%

# %%
fig, ax = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'hspace': 0.1, 'wspace': .15});

region_count = -1

scale_1000 = 1e3

for ii in [0, 1, 2]:
    for jj in [0, 1, 2]:
        region_count += 1
        region = regions[region_count]
        ax[ii][jj].grid(True);
        df = inventory_annal_diff.copy()
        df = df[df["region"] == region].copy()
        
        ax[ii][jj].scatter(df["inventory_delta"], df["slaughter_count"], c = col_dict[region],
                           label=region.replace("_", " ").title())
        if region_count >= 6 :
            ax[ii][jj].set_xlabel("inventory change (1000 heads)")
        if region_count in [0, 3, 6]:
            ax[ii][jj].set_ylabel("slaughter (1000 heads)");
        ax[ii][jj].legend(loc = "best");
#        ax[ii][jj].set_yticklabels(ax[ii][jj].get_yticklabels(), rotation=0); # makes 10^6 go away!!!
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_1000))
        ax[ii][jj].yaxis.set_major_formatter(ticks_y)
        
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_1000))
        ax[ii][jj].xaxis.set_major_formatter(ticks_x)
        ax[ii][jj].set_axisbelow(True)
        
fig_name = plot_dir + "regionsSla_Invt_Scatter_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %%
# fig, ax = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'hspace': 0.1, 'wspace': .15});

# region_count = -1

# scale_1000 = 1e3

# for ii in [0, 1, 2]:
#     for jj in [0, 1, 2]:
#         region_count += 1
#         region = regions[region_count]
#         ax[ii][jj].grid(True);
#         df = inventory_annal_diff.copy()
#         df = df[df["region"] == region].copy()
        
#         ax[ii][jj].scatter(df["inventory_delta"], df["slaughter_count"], c = col_dict[region],
#                            label=region.replace("_", " ").title())
#         if region_count >= 6 :
#             ax[ii][jj].set_xlabel("inventory change")
#         if region_count in [0, 3, 6]:
#             ax[ii][jj].set_ylabel("slaughter");
#         ax[ii][jj].legend(loc = "best");
        
# fig_name = plot_dir + "AAA" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
# plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %% [markdown]
# # National Level
#
# gets rid of movements.

# %%
region_slaughter_inventory.head(2)

# %%
national_slaughter = region_slaughter_inventory.groupby(["year"])["slaughter_count"].sum().reset_index()
national_inv = region_slaughter_inventory.groupby(["year"])["inventory_Jan1"].sum().reset_index()

national_inv_slaut = pd.merge(national_slaughter, national_inv, how="outer", on="year")

print (national_inv_slaut.shape)
national_inv_slaut.head(2)

# %%
national_inv_annal_diff = pd.DataFrame()
for a_year in sorted(national_inv_slaut.year.unique()):
    curr_df_yr = national_inv_slaut[national_inv_slaut.year.isin([a_year, a_year-1])].copy()
    if len(curr_df_yr) == 2:
        curr_diff = curr_df_yr.iloc[1]["inventory_Jan1"] - \
                    curr_df_yr.iloc[0]["inventory_Jan1"]

        d = pd.DataFrame.from_dict({'year': [str(a_year) + "_" + str(a_year-1)], 
                                    'inventory_delta': [curr_diff]})

        national_inv_annal_diff = pd.concat([national_inv_annal_diff, d])

national_inv_annal_diff = national_inv_annal_diff[["year", "inventory_delta"]]

national_inv_annal_diff.reset_index(drop=True, inplace=True)
national_inv_annal_diff["inventory_delta"] = national_inv_annal_diff["inventory_delta"].astype(int)
national_inv_annal_diff.reset_index(drop=True, inplace=True)
national_inv_annal_diff.rename(columns={"year": "diff_years"}, inplace=True)


national_inv_annal_diff["year"] = national_inv_annal_diff["diff_years"].str.split("_", expand=True)[1]
national_inv_annal_diff["year"] = national_inv_annal_diff["year"].astype(int)

print (national_inv_annal_diff.shape)
national_inv_annal_diff.head(2)

# %%
national_inv_annal_diff = pd.merge(national_inv_annal_diff, 
                                   national_inv_slaut[['year', 'slaughter_count']], 
                                   on=["year"], how="left")

national_inv_annal_diff.head(2)

# %% [markdown]
# We can see above that even on national level the inventory change and slaughter are not identical.

# %%

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, gridspec_kw={"hspace":0.15, "wspace":0.15})
(ax1, ax2) = axs;

ax1.grid(True); ax2.grid(True)
ax1.set_axisbelow(True); ax2.set_axisbelow(True)

df = national_inv_annal_diff.copy()

ax1.scatter(df["inventory_delta"], df["slaughter_count"], label="National")
ax1.set_xlabel("inventory change")
ax1.set_ylabel("slaughter")
ax1.legend(loc = "best");

ax2.scatter(-df["inventory_delta"], df["slaughter_count"], label="National")
ax2.set_xlabel("negative inventory change")
ax2.legend(loc = "best");

fig_name = plot_dir + "National_Sla_Invt_Scatter_" + datetime.now().strftime('%Y-%m-%d time-%H.%M') + ".pdf"
plt.savefig(fname=fig_name, dpi=100, bbox_inches="tight")

# %% [markdown]
# ### Slaughter % change relative to average of that region

# %%
region_slaughter_inventory.head(2)

# %% [markdown]
# #### Compute regional slaughter averages.

# %%
# %who

# %%
slaughter_means = annual_slaughter.groupby(["region"])["slaughter_count"].mean().reset_index()
slaughter_means.rename(columns={"slaughter_count": "slaughter_mean"}, inplace=True)
slaughter_means

# %%
