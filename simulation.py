import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
from code_to_category import *
import re #for format string parsing
pd.options.mode.chained_assignment = None  # default='warn' #silence setting on copy of pandas slice warning


YEAR = 0
YEAR_WIDTH = 6
FIELD_WIDTH = 15
MODEL = 'ols'

##Pre-process data from anes csv
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
continuous = ['age','ideology','lean_prev','rdi_yr_to_election','inflation_yoy','diff_diff']
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

#only normalize continuous variables
for var in continuous:
    anes[var] = (anes[var] - np.mean(anes[var])) / np.std(anes[var])


def predict_vote(df):
    df['pred_vote'] = df['pred_vote_prob'].apply(lambda x: round(x))
    df['correct'] = df['pred_vote'] == df['vote']
    df['correct'] = df['correct'].apply(lambda x: 1 if x==True else 0)
    return df

def append_accuracy(df, summary):
    dem_share = np.sum(df['pred_vote_prob']*df['weight1']) / np.sum(df['weight1'])
    survey_dem_share = np.sum(df['vote']*df['weight1']) / np.sum(df['weight1'])
    accuracy = np.mean(df['correct'])
    summary['accuracy'] += f'|{YEAR: <{YEAR_WIDTH}}|{accuracy :<{FIELD_WIDTH}.2%}|{dem_share :<{FIELD_WIDTH}.2%}|{survey_dem_share:<{FIELD_WIDTH}.2%}|\n'
    return summary

def append_params(model, summary):
    summary['params'] += f'|{YEAR: <6}|'
    for param in model.params.values:
        summary['params'] += f'{param :<{FIELD_WIDTH}.3f}|'

    summary['params'] += '\n'
    return summary

"""
def print_accuracy(summary):
    print(f'|{"Year" : <{YEAR_WIDTH}}|{"Accuracy:": <{FIELD_WIDTH}}|{"Republican pred vote:": <{FIELD_WIDTH}}|{"Republican vote (survey):": <{FIELD_WIDTH}}|')
    print(summary['accuracy'])

def print_params(model, summary):
    header = f'|{"Year" :<{YEAR_WIDTH}}|'
    for param in model.params.keys():
        header += f'{param[-FIELD_WIDTH:] if len(param) > FIELD_WIDTH else param:<{FIELD_WIDTH}}|'
    print(header)
    print(summary['params'])
"""
    
def print_params(model = None, summary = None, title = ''):
    col_names = list(['Year'])
    [col_names.append(col) for col in list(model.params.keys())]
    widths = list([YEAR_WIDTH])
    [widths.append(FIELD_WIDTH) for x in range(len(col_names) -1)]
    table = append_header(col_names, widths, title, summary['params'])
    print(table)

def print_accuracy(summary = None, title = ''):
    col_names = ['Year', 'Accuracy', 'Republican pred vote', 'Republican vote (survey)']
    widths = list([YEAR_WIDTH])
    [widths.append(FIELD_WIDTH) for x in range(len(col_names) -1)]
    table = append_header(col_names, widths, title, summary['accuracy'])
    print(table)
    
def append_header(col_names: list, widths: list, title = None, table = ''):
    header = ''
    if title:
        header += f'{title :^{sum(widths)}}\n'
    col_name_lines = 1
    try:#nr of lines in header given by maximum lines needed to display longest col_name
        col_name_lines = max(np.round(np.array([len(col) for col in col_names]) / widths + 0.5))
    except:#except error where nr of col_names and widths have unequal dimensions
        print('Error: col_names and widths have unequal dimensions.')
        return
    
    separator = '_' * (sum(widths) + len(widths) + 1) + '\n'
    header += separator
    start = [0] * len(col_names)
    stop = [0] * len(col_names)
    for line_nr in range(int(col_name_lines)):
        header += '|'
        for i in range(len(col_names)):
            start[i] = stop[i]
            stop[i] += widths[i]
            header += f'{col_names[i][start[i]:stop[i]] :<{widths[i]}}|'
        header += '\n'
    header += separator
    header += table

    return header

def append_row(table, values: list, widths: list, format_str: list):
    for i in range(len(widths)):
        width = int(re.findall('^\d+',format_str[i])[0])
        precision = int(re.match('\.(\d)',format_str[i])[0])
        alignment = re.match('^|<|>',format_str[i])
        spec = format_str[i][-1]
        table += f'{values[i] :{alignment}{width}.{precision}{spec}}|'
    table += '\n'

    return table

df_fundamentals = anes
summary_fundamentals = {'accuracy': '', 'params': ''}
str_fundamentals = 'vote ~ diff_diff + lean_prev + rdi_yr_to_election + inflation_yoy'
if MODEL == 'logit':
    model_fundamentals = smf.logit(str_fundamentals, data = df_fundamentals).fit()
elif MODEL == 'ols':
    model_fundamentals = smf.ols(str_fundamentals, data = df_fundamentals).fit()
df_fundamentals['pred_vote_prob'] = model_fundamentals.predict(df_fundamentals)
df_fundamentals = predict_vote(df_fundamentals)
summary_fundamentals = append_accuracy(df_fundamentals, summary_fundamentals)
summary_fundamentals = append_params(model_fundamentals, summary_fundamentals)

summary_demographics = {'accuracy': '', 'params': ''}
summary_combined = {'accuracy': '', 'params': ''}
#range starts at 1976 because there is no ideology data before 1972
for YEAR in range(1972,2020,4):
    #Experiment with dividing prediction model into two parts: one to predict the effect of economic fundamentals and ideological factors,
    #the other to predict rapidly changing coefficients of race and education -- the predictions of these two models will be averaged
    df_fundamentals_subs = df_fundamentals[df_fundamentals['year'] == YEAR]
    #df_fundamentals_subs = df_fundamentals_subs[(df_fundamentals_subs[:] != 'no matching code').all(axis = 1)] #drop all observations without matching codes 

    df_demographics = anes[anes['year'] == YEAR]
    #df_demographics = df_demographics[(df_demographics[:] != 'no matching code').all(axis = 1)] #drop all observations without matching codes 
    str_demographics = 'vote ~ age + education + race'

    convergence = False
    try:
        if MODEL == 'logit':
            model_demographics = smf.logit(str_demographics, data = df_demographics).fit()
        elif MODEL == 'ols':
            model_demographics = smf.ols(str_demographics, data = df_demographics).fit()
        convergence = True
    except:
        pass
    
    if convergence:
        df_demographics['pred_vote_prob'] = model_demographics.predict(df_demographics)
        df_demographics = predict_vote(df_demographics)
        summary_demographics = append_accuracy(df_demographics, summary_demographics)
        summary_demographics = append_params(model_demographics, summary_demographics)

        #to compute average between fundamentals model and demographics model
        if (df_fundamentals_subs.keys() == df_demographics.keys()).all() and len(df_fundamentals_subs) == len(df_demographics):
            df_averaged = df_fundamentals_subs
            df_averaged['pred_vote_prob'] = (df_fundamentals_subs['pred_vote_prob'] + df_demographics['pred_vote_prob']) /2
            df_averaged = predict_vote(df_averaged)
            summary_combined = append_accuracy(df_demographics, summary_combined)#no summary demographics
        else:
            print('Error: Could not compute average of fundamentals and demographic models. Dataframes have differing length.')

title = '1948-2020 Fundamentals Model Accuracy'
print_accuracy(summary_fundamentals, title)
title = '1948-2020 Fundamentals Model Parameters'
print_params(model_fundamentals, summary_fundamentals, title)
title = 'Year - 4 Demographics Model Accuracy'
print_accuracy(summary_demographics, title)
title = 'Year - 4 Demographics Model Parameters'
print_params(model_demographics, summary_demographics, title)
title = '1972-2012 Averaged Model Parameters'
print_accuracy(summary_combined, title)
