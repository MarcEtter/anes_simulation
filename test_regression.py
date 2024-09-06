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
START_YEAR = 1948
PARTIES = ['dem', 'gop']

##Pre-process data from anes csv
anes = pd.read_csv('anes_select.csv')
#Use regression incumbent-nonincumbent version of regression data to see if that changes the results 
#regress_data_inc = pd.read_csv('')
cand_positions = pd.read_csv('candidate_positions_unweighted.csv')
regress_data = pd.read_csv('regression_data_altered.csv')
regress_data = regress_data[['year',
                             'state',
                             'dem_share',
                             'lean_prev',
                             'lean_prev2',
                             'gdp_yoy',
                             'rdi_yr_to_election',
                             'rdi_election_year_change',
                             'inflation_yoy',
                             'hlean_prev2',
                             'hlean_prev',
                             'unemployment',
                             'spliced_BPHI_dem',
                             'spliced_BPHI_gop',
                             'inc_tenure',
                             'inc_party_cand_approval',
                             'inc_pres_approval',
                             'inc_midterm_change'
                             ]]
continuous = ['age',
              'ideology',
              'lean_prev',
              'lean_prev2',
              'rdi_yr_to_election',
              'inflation_yoy',
              'diff_diff',
              'inc_party_cand_approval',
              'inc_midterm_change',
              'gdp_yoy']
economics = ['rdi_yr_to_election',
             'rdi_election_year_change',
             'inflation_yoy',
             'unemployment']
multiply_in_dir_of_incumbency = ['inc_tenure',
                                 'inc_party_cand_approval', 
                                 'inc_midterm_change', 
                                 'inc_pres_approval', 
                                 'gdp_yoy']
regress_data['fips'] = regress_data['state'].apply(lambda x: code_to_category(x,code_to_fips))

cand_positions = cand_positions.set_index('year')
anes = anes.join(cand_positions, on = 'year')

regress_data = regress_data.set_index(['year','fips'])
anes = anes.set_index(['year', 'fips'])
anes = anes.join(regress_data,['year','fips'])
anes = anes.reset_index()
regress_data = regress_data.reset_index()
#reset index so that we may retreive republican two party national pop. vote shares for each election
regress_data = regress_data.set_index(['year','state'])


anes = anes[(anes['year'] % 4 == 0)]
# delete all rows with zeroes indicating missing data
anes = anes[(anes[:] != 0 ).all(axis=1)]
# drop 'neither' responses for gender (only for 2016)
anes = anes[anes['gender'] != 3] 
anes = anes[anes['ideology'] != 9]
anes = anes[anes['vote'] != 3]

anes['state'] = anes['fips'].apply(lambda x: code_to_category(x,state_name))
anes['education'] = anes['education'].apply(lambda x: code_to_category(x,educ_category))
anes['race'] = anes['race'].apply(lambda x: code_to_category(x,race_category))
#note: if using cohen/mcgrath spliced_BPHI figures, add 3.5; they are normalized with the political center as zero
dem_diff = anes['spliced_BPHI_dem'] +3.5 - anes['ideology']
gop_diff = anes['spliced_BPHI_gop'] +3.5 - anes['ideology']
anes['diff_diff_old'] = abs(dem_diff) - abs(gop_diff)

anes['dem_diff'] = anes['dem_ideo'] - anes['ideology']
anes['gop_diff'] = anes['gop_ideo'] - anes['ideology']
anes['diff_diff'] = abs(anes['dem_diff']) - abs(anes['gop_diff'])

#logistic regression requires unit interval; 0: DEM, 1: GOP
anes['incumbency'] = anes['year'].apply(lambda x: code_to_category(x,incumbent))
anes['vote'] = anes['vote'] - 1
#only normalize continuous variables
CONVERSION_TABLE = dict()
for var in continuous:
    #save conversion to normalized scale
    CONVERSION_TABLE[f'{var}_mean'] = np.mean(anes[var])
    CONVERSION_TABLE[f'{var}_std'] = np.std(anes[var])
    anes[var] = (anes[var] - np.mean(anes[var])) / np.std(anes[var])
#invert direction of fundamentals when Democrats are in office, to assess the effects of fundamentals on the incumbent
for var in economics + multiply_in_dir_of_incumbency:
    anes[var] = anes[var] * anes['incumbency']
anes = anes.fillna(0)#nan variables are filled with zeroes, BUT ONLY AFTER NORMALIZING

def predict_vote(df):
    df['pred_vote'] = df['pred_vote_prob'].apply(lambda x: round(x))
    df['correct'] = df['pred_vote'] == df['vote']
    df['correct'] = df['correct'].apply(lambda x: 1 if x==True else 0)
    return df

def append_accuracy(df, summary):
    gop_share = np.sum(df['pred_vote_prob']*df['weight1']) / np.sum(df['weight1'])
    survey_gop_share = np.sum(df['vote']*df['weight1']) / np.sum(df['weight1'])
    real_gop_share = 0
    if YEAR != 0:
        real_gop_share = 1 - regress_data.loc[(YEAR,'USA'),'dem_share']
    accuracy = np.mean(df['correct'])
    summary['accuracy'] += f'|{YEAR: <{YEAR_WIDTH}}|{accuracy :<{FIELD_WIDTH}.2%}|{gop_share :<{FIELD_WIDTH}.2%}'
    if YEAR != 0:
        summary['accuracy'] += f'|{survey_gop_share:<{FIELD_WIDTH}.2%}|{real_gop_share:<{FIELD_WIDTH}.2%}|\n'

    summary['error_df'][YEAR] = [gop_share, survey_gop_share, real_gop_share]
    return summary

def append_params(model, summary):
    summary['params'] += f'|{YEAR: <6}|'
    for param in model.params.values:
        summary['params'] += f'{param :<{FIELD_WIDTH}.3f}|'

    summary['params'] += '\n'
    return summary

def print_params(model = None, summary = None, title = ''):
    col_names = list(['Year'])
    [col_names.append(col) for col in list(model.params.keys())]
    widths = list([YEAR_WIDTH])
    [widths.append(FIELD_WIDTH) for x in range(len(col_names) -1)]
    table = append_header(col_names, widths, title, summary['params'])
    print(table)

def print_accuracy(summary = None, title = ''):
    col_names = ['Year', 'Accuracy', 'Republican pred vote', 'Republican vote (survey)', 'Republican vote (actual)']
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

def print_abs_error(summary):
    error_df = pd.DataFrame(summary['error_df']).transpose()
    error_df['diff_survey'] = abs(error_df.iloc[:,0] - error_df.iloc[:,1])
    error_df['diff_real_vote'] = abs(error_df.iloc[:,0] - error_df.iloc[:,2])
    results = {'diff_survey': sum(error_df['diff_survey']) / len(error_df), 
            'diff_real_vote': sum(error_df['diff_real_vote']) / len(error_df)}
    
    print(f'Average Absolute Error (ANES Deviation): {results["diff_survey"] :<2.3%}')
    print(f'Average Absolute Error (Election Result Deviation): {results["diff_real_vote"] :<2.3%}\n\n')

df_fundamentals = anes[anes['year']>1968]
#df_fundamentals = anes[anes['year']<2016]
summary_fundamentals = {'accuracy': '', 'params': '', 'error_df': dict()}
#various possible predictors
#str_fundamentals = 'vote ~ diff_diff + inflation_yoy + rdi_yr_to_election + unemployment + lean_prev2 + hlean_prev2 + age + education + gender + inc_party_cand_approval'
#most parsimonious combination of two variables
str_fundamentals = 'vote ~ diff_diff + inc_party_cand_approval'
#most of the features from the model used to predict the incumbent's state vote share, plus unemployment
#str_fundamentals = 'vote ~ diff_diff + rdi_yr_to_election + unemployment + lean_prev + lean_prev2 + hlean_prev + hlean_prev2 + inc_party_cand_approval + inflation_yoy + inc_tenure' 
if MODEL == 'logit':
    model_fundamentals = smf.logit(str_fundamentals, data = df_fundamentals).fit()
elif MODEL == 'ols':
    model_fundamentals = smf.ols(str_fundamentals, data = df_fundamentals).fit()
df_fundamentals['pred_vote_prob'] = model_fundamentals.predict(df_fundamentals)
df_fundamentals = predict_vote(df_fundamentals)
summary_fundamentals = append_accuracy(df_fundamentals, summary_fundamentals)
summary_fundamentals = append_params(model_fundamentals, summary_fundamentals)

summary_fundamentals_moving = {'accuracy': '', 'error_df': dict()} #predict accuracy of fundamentals regression for each election 1972-2012
summary_demographics = {'accuracy': '', 'params': '', 'error_df': dict()}
summary_combined = {'accuracy': '', 'params': '', 'error_df': dict()}
#range starts at 1976 because there is no ideology data before 1972
for YEAR in range(1972,2024,4):
    #Experiment with dividing prediction model into two parts: one to predict the effect of economic fundamentals and ideological factors,
    #the other to predict rapidly changing coefficients of race and education -- the predictions of these two models will be averaged
    df_fundamentals_subs = df_fundamentals[df_fundamentals['year'] == YEAR]
    #df_fundamentals_subs = df_fundamentals_subs[(df_fundamentals_subs[:] != 'no matching code').all(axis = 1)] #drop all observations without matching codes 

    df_demographics = anes[anes['year'] == YEAR]
    #df_demographics = df_demographics[(df_demographics[:] != 'no matching code').all(axis = 1)] #drop all observations without matching codes 
    str_demographics = 'vote ~ diff_diff + age + education + race'

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
            summary_combined = append_accuracy(df_averaged, summary_combined)#no summary demographics

            #use prediction fit to entire dataset 1972-2012 to each election in this period
            df_fundamentals_moving = df_fundamentals_subs.copy()
            df_fundamentals_moving['pred_vote_prob'] = model_fundamentals.predict(df_fundamentals_moving)
            df_fundamentals_moving = predict_vote(df_fundamentals_moving)
            summary_fundamentals_moving = append_accuracy(df_fundamentals_moving, summary_fundamentals_moving)
            
        else:
            print('Error: Could not compute average of fundamentals and demographic models. Dataframes have differing length.')


title = '1972-2020 Fundamentals Model Accuracy'
print_accuracy(summary_fundamentals, title)
#title = '1972-2012 Fundamentals Model Parameters'
#print_params(model_fundamentals, summary_fundamentals, title)
title = 'Time Series Fundamentals Model Accuracy By Election'
print_accuracy(summary_fundamentals_moving, title)
print_abs_error(summary_fundamentals_moving)
print('Covariates: ' + str_fundamentals + '\n')
#title = 'Local Demographics Model Accuracy'
#print_accuracy(summary_demographics, title)
#title = 'Local Demographics Model Parameters'
#print_params(model_demographics, summary_demographics, title)

title = '1972-2020 Averaged Model Parameters'
print_accuracy(summary_combined, title)
print_abs_error(summary_combined)
print('Covariates (Election Specific Model): ' + str_demographics)
print('Covariates (Time-Series): ' + str_fundamentals + '\n')