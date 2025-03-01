import pandas as pd
import statsmodels.formula.api as smf
from code_to_category import *

def fit_ols_model():
    predictions = pd.DataFrame()
    regress_data = pd.read_csv('model_data/regression_data_clean.csv')
    states = pd.read_csv('model_data/state_demographics.csv')
    for yr in raking_yr_range:#Exclude current year for cross-validation
        subs = regress_data[regress_data['year'] == yr]
        train = (regress_data[regress_data['year'] != yr])[['year','state','dem_share', 'lean_prev2', 'lean_prev',
                                                            'hlean_prev', 'hlean_prev2', 'dem_inflation_yoy', 
                                                            'dem_rdi_yr_to_election','inc_party','dem_inc_tenure',
                                                            'berry_citizen']]
        regression_str = 'dem_share ~ lean_prev2 + lean_prev + dem_inflation_yoy + dem_rdi_yr_to_election'
        regression_str += ' + inc_party + dem_inc_tenure + berry_citizen'
        linear_model = smf.ols(regression_str, data = train).fit()
        subs['ols_pred_dem_vote'] = linear_model.predict(subs)
        subs = subs[subs['state'] != 'USA']
        subs['fips'] = subs['state'].apply(lambda x: code_to_fips[x])
        subs = subs.set_index('state')
        subs.loc['DC','ols_pred_dem_vote'] = subs.loc['DC','dem_lean_prev2'] + 0.5
        subs['ols_pred_gop_vote'] = 1 - subs['ols_pred_dem_vote']
        predictions = pd.concat([predictions, subs])
    
    predictions = predictions.reset_index().drop('Unnamed: 0', axis = 'columns')
    keys = ['fips','year','state','dem_share','ols_pred_dem_vote','ols_pred_gop_vote']
    predictions[keys].to_csv('model_data/OLS_predictions.csv')

def read_counties():
    print('Reading county_demographics.csv...')
    counties = pd.read_csv('model_data/county_demographics.csv')
    counties = counties.set_index(['year','fips'], drop = False)
    counties = counties.sort_index(level = ['year','fips'])
    print('Finished reading county_demographics.csv.')

raking_yr_range = range(1972,2004,4)
states = pd.read_csv('model_data/state_demographics.csv')
fit_ols_model()
ols = pd.read_csv('model_data/OLS_predictions.csv').rename({'predict': 'ols_prediction'}, axis = 'columns')
keys = ['ols_pred_dem_vote','ols_pred_gop_vote']
ols = ols.set_index(['year','fips'])
#moving .set_index after the merge caused caused states to become lex unsorted
#states = states.set_index(['year','fips'], drop = False) 
#states = states.sort_index(level = ['year','fips'])
states = states.join(ols[keys], on = ['year','fips'], how = 'left')
states = states.set_index(['year','fips'], drop = False) 
#states = states.sort_index(level = ['year','fips'])
#states = pd.merge(states, ols[keys], on = ['year','fips'], how = 'outer')
anes_family_income = pd.read_csv('model_data/anes_income_percentiles.csv').set_index('year')
anes_family_income = anes_family_income.drop(columns=['17','34','68','96'])

#Keys from state level gis data representing variables to be used in the 
#American National Election studies raking function
#Note: there are other variables whose marginal probabilities may be estimated (e.g. urban/rural status),
#but they are omitted in the function's first version
census_keys =  {
    'gender': [
    "Persons: Male",
    "Persons: Female"],
    'age': [
        "Persons: 18-24",
        "Persons: 25-34",
        "Persons: 35-44",
        "Persons: 45-54",
        "Persons: 55-64",
        "Persons: 65+"
    ],
    'race': [
    "Persons: White (single race)",
    "Persons: Black or African American (single race)",
    "Persons: American Indian and Alaska Native (single race)",
    "Persons: Asian and Pacific Islander and Other and Two or More Races"],

    'education': [
    "Persons: 25 years and over ~ Less than 9th grade",
    "Persons: 25 years and over ~ 9th grade to 3 years of college (until 1980) or to some college or associate's degree (since 1990)",
    "Persons: 25 years and over ~ 4 or more years of college (until 1980) or bachelor's degree or higher (since 1990)"],
    
    'family_income': [
    "Families: Income less than $10,000",
    "Families: Income $10,000 to $14,999",
    "Families: Income $15,000 to $24,999",
    "Families: Income $25,000 to $49,999",
    "Families: Income $50,000 or more"],

    'vote': #['ols_pred_dem_vote','ols_pred_gop_vote'],
    ['dem_vote_lean_prev2', 'gop_vote_lean_prev2'],
    #['dem_lean2_plus_poll', 'gop_lean2_plus_poll'],
    #['dem_vote', 'gop_vote']

    #minimum range 0 and maximum range infinity dropped
     'family_income_numeric':
     [
         10000,
         15000,
         25000,
         50000
     ]
}

#Below dict stores mapping from anes keys to census keys with dictionaries relating 
#anes categories to corresponding census categories.
#In order to rake sample data, each variable to rake must have a mapping
#from the anes variable to the corresponding variable in the census data.
#This rule applies to all variables, with the exception of family_income,
#where the marginals are calculated in advance. 
mapping = {
    'vote': {
        0: census_keys['vote'][0],#'dem_vote', #'dem_vote_lean_prev2',
        1: census_keys['vote'][1]#'gop_vote'#'gop_vote_lean_prev2'
    },
    'gender': {1: "Persons: Male",
                    2: "Persons: Female",
                    3: "Persons: Female",
                    0: "Presons: Female"},
    'age': {
        17: "Persons: 18-24",
        18: "Persons: 18-24",
        19: "Persons: 18-24",
        20: "Persons: 18-24",
        21: "Persons: 18-24",
        22: "Persons: 18-24",
        23: "Persons: 18-24",
        24: "Persons: 18-24",
        25: "Persons: 25-34",
        26: "Persons: 25-34",
        27: "Persons: 25-34",
        28: "Persons: 25-34",
        29: "Persons: 25-34",
        30: "Persons: 25-34",
        31: "Persons: 25-34",
        32: "Persons: 25-34",
        33: "Persons: 25-34",
        34: "Persons: 25-34",
        35: "Persons: 35-44",
        36: "Persons: 35-44",
        37: "Persons: 35-44",
        38: "Persons: 35-44",
        39: "Persons: 35-44",
        40: "Persons: 35-44",
        41: "Persons: 35-44",
        42: "Persons: 35-44",
        43: "Persons: 35-44",
        44: "Persons: 35-44",
        45: "Persons: 45-54",
        46: "Persons: 45-54",
        47: "Persons: 45-54",
        48: "Persons: 45-54",
        49: "Persons: 45-54",
        50: "Persons: 45-54",
        51: "Persons: 45-54",
        52: "Persons: 45-54",
        53: "Persons: 45-54",
        54: "Persons: 45-54",
        55: "Persons: 55-64",
        56: "Persons: 55-64",
        57: "Persons: 55-64",
        58: "Persons: 55-64",
        59: "Persons: 55-64",
        60: "Persons: 55-64",
        61: "Persons: 55-64",
        62: "Persons: 55-64",
        63: "Persons: 55-64",
        64: "Persons: 55-64",
        65: "Persons: 65+",
        66: "Persons: 65+",
        67: "Persons: 65+",
        68: "Persons: 65+",
        69: "Persons: 65+",
        70: "Persons: 65+",
        71: "Persons: 65+",
        72: "Persons: 65+",
        73: "Persons: 65+",
        74: "Persons: 65+",
        75: "Persons: 65+",
        76: "Persons: 65+",
        77: "Persons: 65+",
        78: "Persons: 65+",
        79: "Persons: 65+",
        80: "Persons: 65+",
        81: "Persons: 65+",
        82: "Persons: 65+",
        83: "Persons: 65+",
        84: "Persons: 65+",
        85: "Persons: 65+",
        86: "Persons: 65+",
        87: "Persons: 65+",
        88: "Persons: 65+",
        89: "Persons: 65+",
        90: "Persons: 65+",
        91: "Persons: 65+",
        92: "Persons: 65+",
        93: "Persons: 65+",
        94: "Persons: 65+",
        95: "Persons: 65+",
        96: "Persons: 65+",
        97: "Persons: 65+",
        98: "Persons: 65+",
        99: "Persons: 65+",
        0: "Persons: 35-44",
    },
    #mapping of race categories in anes to race categories in census
    #anes race categories are commented to the right of census categories
    'race': { 
    1: "Persons: White (single race)", #White non-Hispanic (1948-2012)
    2: "Persons: Black or African American (single race)",#Black non-Hispanic (1948-2012)
    3: "Persons: Asian and Pacific Islander and Other and Two or More Races",#Asian or Pacific Islander, non-Hispanic (1966-2012)
    4: "Persons: American Indian and Alaska Native (single race)", #American Indian or Alaska Native non-Hispanic (1966-2012) 
    5: "Persons: Asian and Pacific Islander and Other and Two or More Races",#Hispanic (1966-2012)
    6: "Persons: Asian and Pacific Islander and Other and Two or More Races",#Other or multiple races, non-Hispanic (1968-2012)
    7: "Persons: Asian and Pacific Islander and Other and Two or More Races",#Non-white and non-black (1948-1964)
    
    #conversion for anes codes converted to anes categories
    'White non-Hispanic':                               "Persons: White (single race)",
    'Hispanic':                                         "Persons: Asian and Pacific Islander and Other and Two or More Races",
    'Black non-Hispanic':                               "Persons: Black or African American (single race)",
    'Asian or Pacific Islander, non-Hispanic':          "Persons: Asian and Pacific Islander and Other and Two or More Races",
    'Other or multiple races, non-Hispanic':            "Persons: Asian and Pacific Islander and Other and Two or More Races",
    'American Indian or Alaska Native non-Hispanic':    "Persons: American Indian and Alaska Native (single race)",
    'no matching code':                                 "Persons: White (single race)"

    },
    
    'education':{
        '8 grades or less':                         "Persons: 25 years and over ~ Less than 9th grade",
        '12 grades':                                "Persons: 25 years and over ~ 9th grade to 3 years of college (until 1980) or to some college or associate's degree (since 1990)",
        '12 grades plus non-academic training':     "Persons: 25 years and over ~ 9th grade to 3 years of college (until 1980) or to some college or associate's degree (since 1990)",
        '9-12 grades':                              "Persons: 25 years and over ~ 9th grade to 3 years of college (until 1980) or to some college or associate's degree (since 1990)",
        'Advanced degree':                          "Persons: 25 years and over ~ 4 or more years of college (until 1980) or bachelor's degree or higher (since 1990)",
        'BA level degree':                          "Persons: 25 years and over ~ 4 or more years of college (until 1980) or bachelor's degree or higher (since 1990)",
        'Some college, no degree; AA degree':       "Persons: 25 years and over ~ 9th grade to 3 years of college (until 1980) or to some college or associate's degree (since 1990)",
        'no matching code':                         "Persons: 25 years and over ~ 9th grade to 3 years of college (until 1980) or to some college or associate's degree (since 1990)"
    }
}

