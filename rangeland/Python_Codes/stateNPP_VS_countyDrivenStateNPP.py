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
import pandas as pd
import sys
sys.path.append("/Users/hn/Documents/00_GitHub/Ag/rangeland/Python_Codes/")
import rangeland_core as rc

# %%
dir_ = "/Users/hn/Downloads/Mac-to-iMac/"

data_dir_base = "/Users/hn/Documents/01_research_data/RangeLand/Data/"
min_dir = data_dir_base + "Min_Data/"

# %%
county_annual_productivity = pd.read_csv(min_dir+ "county_annual_productivity.csv")
Min_state_NPP = pd.read_csv(min_dir+ "statefips_annual_productivity.csv")
county_RA = pd.read_csv(min_dir+ "county_rangeland_and_totalarea_fraction.csv")

# %%
county_RA.head(2)

# %%
print (county_RA.shape)
county_RA.drop_duplicates(inplace=True)
print (county_RA.shape)

# %%
county_RA.to_csv(dir_+ "county_rangeland_and_totalarea_fraction.csv", index=False)

# %%
county_RA.head(2)

# %%

# %%
county_RA = rc.correct_Mins_county_6digitFIPS(county_RA, "FIPS_ID")
county_annual_productivity = rc.correct_Mins_county_6digitFIPS(county_annual_productivity, "county")

county_RA.head(2)

# %%
county_RA["state_fips"] = county_RA["FIPS_ID"].str.slice(0, 2)
county_RA.head(2)

# %%
county_RA.rename(columns={"FIPS_ID" : "county_fips"}, inplace=True)
county_annual_productivity.rename(columns={"county" : "county_fips"}, inplace=True)

# %%
unique_county_fips_from_county_RA = list(county_RA["county_fips"].unique())
unique_county_fips_from_county_annual_productivity = list(county_annual_productivity["county_fips"].unique())

# %%
print (len(unique_county_fips_from_county_RA))
print (len(unique_county_fips_from_county_annual_productivity))

# %%
all_present = True

for a_cnty in unique_county_fips_from_county_annual_productivity:
    if not (a_cnty in unique_county_fips_from_county_RA):
        print (a_cnty)

# %% [markdown]
# ## Subset counties in county_RA to the counties that are present in NPP file

# %%
print (county_RA.shape)
county_RA = county_RA[county_RA.county_fips.isin(unique_county_fips_from_county_annual_productivity)]
print (county_RA.shape)

# %%

# %%
state_RA = county_RA[["state_fips", "Rangeland_Acre"]].groupby(["state_fips"]).sum().reset_index()
state_RA.rename(columns={"Rangeland_Acre" : "rangeland_acre"}, inplace=True)
print (state_RA.shape)
state_RA.head(2)

# %%
state_RA[state_RA.state_fips == "47"]

# %%
county_RA.head(2)

# %%
county_annual_productivity.head(2)

# %%
county_annual_productivity = pd.merge(county_annual_productivity, 
                                      county_RA[["county_fips", "Rangeland_Acre", "state_fips"]],
                                      how = "left",
                                      on = "county_fips")

county_annual_productivity.head(2)

# %%
county_annual_productivity.rename(columns={"productivity" : "unit_NPP",
                                           "Rangeland_Acre" : "rangeland_acre"}, inplace=True)
county_annual_productivity.head(2)

# %%
county_annual_productivity["total_NPP_lb"] = county_annual_productivity["unit_NPP"] * \
                                             county_annual_productivity["rangeland_acre"]
    
county_annual_productivity.head(3)

# %%
A = county_annual_productivity[["state_fips", "total_NPP_lb", "year"]].copy()
my_state_NPP = A.groupby(["year", "state_fips"]).sum().reset_index()

del(A)
my_state_NPP.head(2)

# %%
state_RA.head(2)

# %%

# %%
my_state_NPP = pd.merge(my_state_NPP, state_RA, how="left", on="state_fips")
my_state_NPP.head(2)

# %%
my_state_NPP["unit_NPP_lb_per_acr"] = my_state_NPP["total_NPP_lb"] / my_state_NPP["rangeland_acre"]
my_state_NPP.head(2)

# %%
my_state_NPP.shape

# %%
Min_state_NPP.shape

# %%
my_state_NPP.head(2)

# %%
Min_state_NPP = rc.correct_Mins_county_6digitFIPS(Min_state_NPP, "statefips90m")
Min_state_NPP.rename(columns={"statefips90m" : "state_fips",
                              "productivity" : "Min_unit_NPP"}, inplace=True)
Min_state_NPP.head(2)

# %%
my_state_NPP.rename(columns={"productivity" : "my_productivity"}, inplace=True)
my_state_NPP.head(2)

# %%
df = pd.merge(Min_state_NPP, my_state_NPP, how="left", on=["state_fips", "year"])

# %%
df = df[['year', 'state_fips', 'total_NPP_lb', 'rangeland_acre',
         'Min_unit_NPP','unit_NPP_lb_per_acr']]
df.head(10)

# %%
dir_ = "/Users/hn/Downloads/Mac-to-iMac/"

state_unitNPP_withArea = pd.read_csv(dir_ + "statefips90m_annual_productivity_MEAN.csv")
state_unitNPP = pd.read_csv(min_dir + "statefips_annual_productivity.csv")


state_unitNPP_withArea["statefips90m"] = state_unitNPP_withArea["statefips90m"].astype(str)
state_unitNPP["statefips90m"] = state_unitNPP["statefips90m"].astype(str)

state_unitNPP_withArea["state_fips"] = state_unitNPP_withArea["statefips90m"].str.slice(1, 3)
state_unitNPP["state_fips"] = state_unitNPP["statefips90m"].str.slice(1, 3)

# %%
state_unitNPP_withArea.head(3)

# %%
state_unitNPP.head(3)

# %%
reOrganized_dir = data_dir_base + "reOrganized/"
abb_dict = pd.read_pickle(reOrganized_dir + "county_fips.sav")
SoI = abb_dict["SoI"]
SoI_abb = [abb_dict["full_2_abb"][x] for x in SoI]

# %%
state_fips_names_df = abb_dict["state_fips"].copy()
state_fips_names_df.head(2)

# %%
state_unitNPP_withArea = pd.merge(state_unitNPP_withArea, state_fips_names_df, how="left", on="state_fips")
state_unitNPP = pd.merge(state_unitNPP, state_fips_names_df, how="left", on="state_fips")

# %%

# %%
state_unitNPP_withArea[state_unitNPP_withArea.state_full == "Tennessee"]

# %%
