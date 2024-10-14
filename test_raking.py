from test_regression import *
from code_to_category import *
import pandas as pd

#This code uses two-way raking to determine geographic units' survey weights from a national survey
#First a matrix of weights is obtained by raking the national survey for each geographic unit,
#where columns correspond to the survey weights in each geographic unit
#The quotient between the national survey weights and the summed weights gives the vector that adjusts
#the geographic unit weights so that they sum to the national marginals.
#We multiply the geographic weight matrix by this vector to find the final weights for each geographic unit.

#Edit: I do not think that this step is necessary
##We sum over all survey weights in geographic units and rake again to determine a new vector of survey weights
#that produce the marginals in the national survey

yr = 1976
df_1976 = default['df'].copy()
df_1976 = df_1976[df_1976['year'] == 1976]
rake_keys = list(census_keys.keys())[:-1]
state_weights = pd.DataFrame()
#Find weights of each respondent within state by raking
for state in state_postal_codes:
    df = rake_state(1976, state, df_1976, rake_keys)
    state_pop = states.loc[(yr,code_to_fips[state]), 'Persons: Total'] / 10**6
    state_weights[f'{state}_weight1'] = df['weight1'] * state_pop

national_weights = state_weights.sum(axis = 'columns')

#Below code may not be necessary
"""
#Sum weights across all geographic units to find equivalent weights in theoretical national sample
#Then rake so that summed proportions equal proportions in national sample
target = df_1976.copy()
target['weight1'] = national_weights
#Get marginal probabilities of national sample
target_marginal_probs = dict()
for key in rake_keys:
    target_marginal_probs[key] = target.groupby(key)['weight1'].agg(pd.Series.sum) / target['weight1'].sum() 

adj_weights = rake(df_1976, target_marginal_probs, rake_keys)[0]['weight1'] #get weights from the returned df indexed at 0
quotient = adj_weights / national_weights
state_weights *= quotient
"""

quotient = df_1976['weight1'] / national_weights
state_weights *= quotient

