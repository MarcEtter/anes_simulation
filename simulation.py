import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
from code_to_category import *

anes = pd.read_csv('anes_select.csv')
#Use regression incumbent-nonincumbent version of regression data to see if that changes the results 
#regress_data_inc = pd.read_csv('')
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
anes = anes.join(regress_data,['year','fips'])
anes = anes.reset_index()

anes = anes[(anes['year'] % 4 == 0)]
anes = anes[anes['year'] <= 2012]
# delete all rows with zeroes indicating missing data
anes = anes[(anes[:] != 0 ).all(axis=1)]
# drop 'neither' responses for gender (only for 2016)
anes = anes[anes['gender'] != 3] 
anes = anes[anes['ideology'] != 9]
anes = anes[anes['vote'] != 3]

anes['state'] = anes['fips'].apply(lambda x: code_to_category(x,state_name))
anes['education'] = anes['education'].apply(lambda x: code_to_category(x,educ_category))
anes['race'] = anes['race'].apply(lambda x: code_to_category(x,race_category))
dem_diff = anes['spliced_BPHI_dem'] + 3.5 - anes['ideology']
gop_diff = anes['spliced_BPHI_gop'] + 3.5 - anes['ideology']
anes['diff_diff'] = abs(dem_diff) - abs(gop_diff)

#logistic regression requires unit interval; 0: DEM, 1: GOP
anes['incumbency'] = anes['year'].apply(lambda x: code_to_category(x,incumbent))
anes['vote'] = anes['vote'] - 1
#invert direction of fundamentals when Democrats are in office, to assess the effects of fundamentals on the incumbent
for var in economics:
    anes[var] = anes[var] * anes['incumbency']

def predict_vote(df):
    df['pred_vote'] = df['pred_vote_prob'].apply(lambda x: round(x))
    df['correct'] = df['pred_vote'] == df['vote']
    df['correct'] = df['correct'].apply(lambda x: 1 if x==True else 0)
    return df

def append_accuracy(df, summary):
    dem_share = np.sum(df['pred_vote_prob']*df['weight1']) / np.sum(df['weight1'])
    survey_dem_share = np.sum(df['vote']*df['weight1']) / np.sum(df['weight1'])

    accuracy = np.mean(df['correct'])*100
    summary['accuracy'] += f'|{YEAR: <6}|{accuracy :<39 :2.2f}%|{dem_share*100 :<39 :2.2f}%|{survey_dem_share*100 :<39 :2.2f}%\n'
    return summary

def append_coeffs(model, summary):
    summary['coefficients'] += f'|{YEAR: <6}|'
    for coeff in model.coeffs[1:]:
        summary['coefficients'] += f'{coeff :<10 :2.3f}|'
    coefficient_table += '\n'
    return summary

def print_accuracy(summary):
    print(f'|{"Year" : <6}|{"Accuracy:": <40}|{"Republican pred vote:": <40}|{"Republican vote (survey):": <40}')
    print(summary['demographics'])

def print_coeffs(summary):
    print(f'placeholder')
    print(summary['coeffs'])

df_fundamentals = anes
summary_fundamentals = {'accuracy': '', 'coefficients': ''}
str_fundamentals = 'vote ~ diff_diff + lean_prev + rdi_yr_to_election + inflation'
model_fundamentals = smf.logit(str_fundamentals, data = df_fundamentals).fit()
df_fundamentals['pred_vote_prob'] = model_fundamentals.predict(df_fundamentals)
df_fundamentals = predict_vote(df_fundamentals)
summary_fundamentals = append_accuracy(df_fundamentals, summary_fundamentals)
summary_fundamentals = append_coeffs(df_fundamentals, summary_fundamentals)

for YEAR in range(1948,2020,4):
    #Experiment with dividing prediction model into two parts: one to predict the effect of economic fundamentals and ideological factors,
    #the other to predict rapidly changing coefficients of race and education -- the predictions of these two models will be averaged
    df_demographics = anes[anes['year'] == YEAR - 4]
    str_demographics = 'vote ~ age + education + race'
    summary_demographics = {'accuracy': '', 'coefficients': ''}
    summary_combined = {'accuracy': '', 'coefficients': ''}

    model_demographics = smf.logit(str_demographics, data = df_demographics).fit()
    df_demographics['pred_vote_prob'] = model_demographics.predict(df_demographics)
    df_demographics = predict_vote(df_demographics)
    summary_demographics = append_accuracy(df_demographics, summary_demographics)
    summary_demographics = append_coeffs(model_demographics, summary_demographics)

    df_averaged = df_fundamentals.join(df_demographics)
    df_averaged['pred_vote_prob'] = np.average(df_fundamentals['pred_vote_prob'], df_demographics['pred_vote_prob'])
    df_averaged = predict_vote(df_averaged)
    summary_combined = append_accuracy(df_demographics, summary_combined)#no summary demographics

section = '-'
print(section*40)
print_accuracy(summary_fundamentals)
print(section*40)
print_coeffs(summary_fundamentals)
print(section*40)
print_accuracy(summary_demographics)
print(section*40)
print_coeffs(summary_demographics)
print(section*40)
print_accuracy(summary_combined)

#set own parameters
#results.params[:] = 0 
