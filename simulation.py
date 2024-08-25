import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
from code_to_category import *

anes = pd.read_csv('anes_select.csv')
regress_data = pd.read_csv('regression_data_altered.csv')
regress_data = regress_data[['year',
                             'state',
                             'lean_prev',
                             'lean_prev2',
                             'rdi_yr_to_election',
                             'rdi_election_year_change',
                             'inflation_yoy',
                             'hlean_prev2',
                             'hlean_prev',
                             'spliced_BPHI_dem',
                             'spliced_BPHI_gop']]
economics = ['rdi_yr_to_election','rdi_election_year_change','inflation_yoy']
regress_data['fips'] = regress_data['state'].apply(lambda x: code_to_category(x,code_to_fips))

regress_data = regress_data.set_index(['year','fips'])
anes = anes.set_index(['year', 'fips'])
anes_2020 = anes.join(regress_data,['year','fips'])
anes_2020 = anes_2020.reset_index()

anes_2020 = anes_2020[(anes_2020['year'] % 4 == 0)]
anes_2020 = anes_2020[anes_2020['year'] <= 2012]
# delete all rows with zeroes indicating missing data
anes_2020 = anes_2020[(anes_2020[:] != 0 ).all(axis=1)]
# drop 'neither' responses for gender (only for 2016)
anes_2020 = anes_2020[anes_2020['gender'] != 3] 
anes_2020 = anes_2020[anes_2020['ideology'] != 9]
anes_2020 = anes_2020[anes_2020['vote'] != 3]

anes_2020['state'] = anes_2020['fips'].apply(lambda x: code_to_category(x,state_name))
anes_2020['education'] = anes_2020['education'].apply(lambda x: code_to_category(x,educ_category))
anes_2020['race'] = anes_2020['race'].apply(lambda x: code_to_category(x,race_category))
dem_diff = anes_2020['spliced_BPHI_dem'] + 3.5 - anes_2020['ideology']
gop_diff = anes_2020['spliced_BPHI_gop'] + 3.5 - anes_2020['ideology']
anes_2020['diff_diff'] = abs(dem_diff) - abs(gop_diff)
#anes_2020['dem_diff'] = dem_diff

#logistic regression requires unit interval; 0: DEM, 1: GOP
anes_2020['incumbency'] = anes_2020['year'].apply(lambda x: code_to_category(x,incumbent))
anes_2020['vote'] = anes_2020['vote'] - 1
#invert direction of fundamentals when Democrats are in office, to assess the effects of fundamentals on the incumbent
for var in economics:
    anes_2020[var] = anes_2020[var] * anes_2020['incumbency']

#variable for ideological distance from candidates
#incumbency + diff_diff + age + education + race + lean_prev + rdi_yr_to_election + inflation_yoy
results = smf.logit('vote ~ incumbency + diff_diff + age + education + race + lean_prev + rdi_yr_to_election + inflation_yoy', data = anes_2020).fit()
anes_2020['pred_vote_prob'] = results.predict(anes_2020)
anes_2020['pred_vote'] = anes_2020['pred_vote_prob'].apply(lambda x: round(x))
anes_2020['correct'] = anes_2020['pred_vote'] == anes_2020['vote']
anes_2020['correct'] = anes_2020['correct'].apply(lambda x: 1 if x==True else 0)

print(results.summary())
dem_share = np.sum(anes_2020['pred_vote_prob']*anes_2020['weight1']) / np.sum(anes_2020['weight1'])
survey_dem_share = np.sum(anes_2020['vote']*anes_2020['weight1']) / np.sum(anes_2020['weight1'])

accuracy = np.mean(anes_2020['correct'])*100
print(f'{"Observations correctly predicted:": <40}{accuracy:2.2f}%')
print(f'{"Republican predicted vote share:": <40}{dem_share*100:2.2f}%')
print(f'{"Republican vote share (survey):": <40}{survey_dem_share*100:2.2f}%')
anes_2020.to_csv('predicted_votes.csv')

#set own parameters
#results.params[:] = 0 



### WHAT HAPPENS WHEN I CHANGE THE IDEOLOGICAL POSITION?
#suppose all democratic candidates move one point to the left (below)
dem_diff = anes_2020['spliced_BPHI_dem'] + 2.5 - anes_2020['ideology']
gop_diff = anes_2020['spliced_BPHI_gop'] + 3.5 - anes_2020['ideology']
anes_2020['diff_diff'] = abs(dem_diff) - abs(gop_diff)
anes_2020['pred_vote_prob'] = results.predict(anes_2020)
anes_2020['pred_vote'] = anes_2020['pred_vote_prob'].apply(lambda x: round(x))
anes_2020['correct'] = anes_2020['pred_vote'] == anes_2020['vote']
anes_2020['correct'] = anes_2020['correct'].apply(lambda x: 1 if x==True else 0)

dem_share = np.sum(anes_2020['pred_vote_prob']*anes_2020['weight1']) / np.sum(anes_2020['weight1'])
survey_dem_share = np.sum(anes_2020['vote']*anes_2020['weight1']) / np.sum(anes_2020['weight1'])

accuracy = np.mean(anes_2020['correct'])*100
print(f'{"Observations correctly predicted:": <40}{accuracy:2.2f}%')
print(f'{"Republican predicted vote share:": <40}{dem_share*100:2.2f}%')
print(f'{"Republican vote share (survey):": <40}{survey_dem_share*100:2.2f}%')
anes_2020.to_csv('predicted_votes.csv')