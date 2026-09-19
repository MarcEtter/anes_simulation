
# Code style guidelines

Write code in a compact and concise way.
Add line breaks between between major chunks of code and a sparing amount of comments.
An illustration of the style is shown below:

# Code Examples
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.formula.api as smf

ANES_YEAR_MIN = 1972 # Lower limit of years to consider for ANES-based model

anes_full = pd.read_csv('raw_data/anes_timeseries_cdf_csv_20260205/anes_timeseries_cdf_csv_20260205.csv')
anes_full = anes_full.rename(columns={
    "VCF0803": "voter_ideo", #Ideological self-placement scale
    "VCF9096": "gop_ideo", #Respondent rating of gop candidate on ideology scale
    "VCF9088": "dem_ideo", #Respondentt rating of dem candidate on ideology scale
    "VCF0830": "voter_blacks", #Government aid to blacks question, selected due to time continuity from 1972-2024
    "VCF9092": "gop_blacks", #Respondent rating of gop candidate on gov't aid to blacks
    "VCF9084": "dem_blacks", #Respondent rating of dem candidate on gov't aid to blacks
    "VCF0901a": "state_fips",
    "VCF0004": "year",
    "VCF0704": "vote_pres",
    "VCF0704a": "vote_pres_2party"})
anes_full['state_fips'] = pd.to_numeric(anes_full['state_fips'], errors='coerce').astype('Int64')

# Merge with state postal codes (for interpretability) and region codes
state_codes = pd.read_csv('raw_data/state_fips.csv')
anes_full = pd.merge(anes_full, state_codes, on='state_fips', how='left')
regions = pd.read_csv('raw_data/us_regions_divisions.csv')
regions = regions.rename(columns = {'State': 'state_name', 'State Code': 'state', 'Region': 'census_region', 'Division': 'division'})
anes_full = pd.merge(anes_full, regions[['state','census_region','division']], on = 'state', how = 'left')
bea_regions = pd.read_csv('raw_data/statelevel_predictors.csv')
anes_full = pd.merge(anes_full, bea_regions[['state','bea_region','pol_south']], on = 'state', how = 'left')

# Add state partisanship data to better predict distribution of ideology within states
state_partisanship = pd.read_stata('raw_data/1868_2020_presvote.dta')
state_partisanship['natl_rvote'] = (state_partisanship.groupby('year')['rvote']
        .transform(lambda x: x[state_partisanship.loc[x.index, 'state'] == 'US'].iloc[0]))
state_partisanship['rlean'] = state_partisanship['rvote'] - state_partisanship['natl_rvote']
state_partisanship['rlean_rolling2'] = (state_partisanship.groupby('state')['rlean']
        .rolling(window = 2, min_periods = 1).mean().reset_index(level=0, drop=True) 
)
state_partisanship = state_partisanship[state_partisanship['year'] >= ANES_YEAR_MIN]
anes_full = pd.merge(anes_full, state_partisanship[['year','state','rlean','rlean_rolling2']], on = ['state', 'year'], how = 'left')
anes_full['rlean'] = anes_full.groupby('state')['rlean'].ffill() #fill missing values in mid-election year with rlean from previous year
anes_full['rlean_rolling2'] = anes_full.groupby('state')['rlean_rolling2'].ffill()
# TODO: Extract 'totev' from the state partisanship data to model electoral votes

# Join mass economic and policy liberalism measures from Caughey Warshaw 2018 with ANES respondent ideology data
caughey_warshaw = pd.read_stata('/Users/marcetter/Dropbox/writing_sample/caughey_warshaw_2018_replication/caughey_warshaw_summary.dta')
caughey_warshaw = caughey_warshaw.rename(columns = {'stpo': 'state'})
caughey_warshaw['year'] = caughey_warshaw['year'].astype('int64')
anes_full = pd.merge(anes_full, caughey_warshaw[['year','state','masseconlib_est','masssociallib_est']], on = ['state','year'], how = 'left')
#extrapolate most recent estimates in 2014 to 2024
anes_full['masseconlib_est'] = anes_full.groupby('state')['masseconlib_est'].ffill()
anes_full['masssociallib_est'] = anes_full.groupby('state')['masssociallib_est'].ffill()

anes_full['voter_ideo'] = pd.to_numeric(anes_full['voter_ideo'], errors='coerce')
anes_full['voter_ideo'] = anes_full['voter_ideo'].apply(lambda x: np.nan if x == 9.0 or x == 0.0 else x)
anes_full['voter_blacks'] = pd.to_numeric(anes_full['voter_blacks'], errors='coerce')
anes_full['voter_blacks'] = anes_full['voter_blacks'].apply(lambda x: np.nan if x == 9.0 or x == 0.0 else x)

## Create predictor variables for ordinal model
anes_full['state_fips'] = anes_full['state_fips'].astype('string')
anes_full['decade'] = ((anes_full['year'] // 10) * 10).astype('string')
#anes['year_categorical'] = anes['year'].astype('category')
anes_full['year_center'] = anes_full['year'] - 2000
anes_full['year_center_sq'] = anes_full['year_center']**2

anes = anes_full[[
    'state','state_fips',
    'voter_ideo','voter_blacks',
    'year', 'year_center', 'year_center_sq', 'decade',
    'census_region','division','bea_region',
    'masseconlib_est', 'masssociallib_est',
    'rlean','rlean_rolling2']]
anes = anes[anes['year'] >= ANES_YEAR_MIN]