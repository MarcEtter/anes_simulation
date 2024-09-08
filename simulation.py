from test_regression import *
import pandas as pd

ELECTION_YR = 2020
INC_PARTY_CAND_APPROVAL = 0
CONTINUOUS_VARS = ['ideology']

candidates = {
'Harris':
    {
    'party': 'Democrat',
    'incumbent': True,
    'ideology': 2.1,
    'expected_vote': 0.5},

'Trump':
    {
    'party': 'Republican',
    'incumbent': False,
    'ideology': 5.25,
    'expected_vote': 0.5},

'Kennedy':
    {'party': 'Independent',
    'incumbent': False,
    'ideology': 4.25,
    'expected_vote': 0.05}
}

cand_df = pd.DataFrame(candidates)

for var in CONTINUOUS_VARS:
    var_mean = CONVERSION_TABLE[f'{var}_mean']
    var_std = CONVERSION_TABLE[f'{var}_std']
    cand_df[var] = cand_df[var].apply(lambda x: (x - var_mean) / var_std)  

anes_yr = anes[anes['year'] == ELECTION_YR]
anes_yr['dem_ideo'] = cand_df['Harris']['ideology']
anes_yr['gop_ideo'] = cand_df['Trump']['ideology']
anes_yr['rfk_ideo'] = cand_df['Kennedy']['ideology']

for candidate in cand_df.index():
    party = cand_df[candidate]['party']
    anes_yr[f'{party}_diff'] = abs(anes_yr['dem_ideo'] - anes_yr['ideology'])

#MAKE sure to SUBTRACT REPUBLICAN from DEMOCRAT, or the measure will be reversed
#note: in the multi-candidate case, it is not pratical to measure the difference in differences,
#unless we wish to make n^2 comparisons, where n is the number of candidates
anes_yr['diff_diff'] = anes['Democrat_diff'] - anes['Republican_diff']

anes_yr_fundamentals = model_fundamentals.predict(anes_yr)
anes_yr_demographics = model_demographics.predict(anes_yr) 
