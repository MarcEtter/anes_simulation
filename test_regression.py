import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
from code_to_category import *
import re #for format string parsing
pd.options.mode.chained_assignment = None  # default='warn' #silence setting on copy of pandas slice warning

global PARTY_CODES
global PARTIES
PARTY_CODES = {'dem':0, 'gop':1}
PARTIES = list(PARTY_CODES.keys())
YEAR = 0
YEAR_WIDTH = 6
FIELD_WIDTH = 15
MODEL = 'logit'
START_YEAR = 1948
MULTI_PARTY_MODE = True
RUN_REGRESSIONS = True

def get_vars_to_norm():
    global PARTIES
    vars_to_normalize = ['age',
                    'lean_prev',
                    'lean_prev2',
                    'rdi_yr_to_election',
                    'inflation_yoy',
                    'diff_diff',
                    'inc_party_cand_approval',
                    'inc_tenure',
                    'inc_midterm_change',
                    'gdp_yoy']
    #vars_to_normalize += [f'{party}_poll' for party in PARTIES]
    #vars_to_normalize += [f'{party}_diff' for party in PARTIES]
    return vars_to_normalize

multiply_in_dir_of_incumbency = [
    'rdi_yr_to_election',
    'rdi_election_year_change',
    'inflation_yoy',
    'unemployment',
    'inc_tenure',
    'inc_party_cand_approval', 
    'inc_midterm_change', 
    'inc_pres_approval', 
    'gdp_yoy',
    'lean_prev',
    'lean_prev2',
    'hlean_prev',
    'hlean_prev2']

#function for simulation.py to set the parties in the analysis
def set_parties(df, candidates):
    global PARTY_CODES
    global PARTIES
    #PARTY_CODES = dict(zip([x['party'] for x in candidates.values()], [x['code'] for x in candidates.values()]))
    PARTY_CODES = candidates.keys()
    PARTIES = list(PARTY_CODES.keys())
    
    #cand_dict = {}
    #for cand in candidates.values():
    #    party = cand['party']
    #    cand_dict[f'{party}_ideo'] = cand['ideology']
    #    cand_dict[f'{party}_poll'] = cand['poll']

    for party in PARTIES:
        df[f'{party}_ideo'] = candidates[party]['ideology']
        df[f'{party}_poll'] = candidates[party]['poll']
    
    df = compute_distances(df)
    df = normalize(df, get_vars_to_norm())
    df = add_party_codes_votes(df)
    df = add_party_vars(df, multiply_in_dir_of_incumbency)

    return df
    
def compute_distances(df):
    for party in PARTIES:
        df[f'{party}_diff'] = abs(df[f'{party}_ideo'] - df['ideology'])
    return df

def normalize(df, vars_to_normalize):
    for var in vars_to_normalize:
        df[var] = (df[var] - df[f'{var}_mean']) / df[f'{var}_std']
    return df
#invert direction of fundamentals when Democrats are in office, to assess the effects of fundamentals on the incumbent

def add_party_codes_votes(df):
    for party in PARTIES:
        df[f'{party}_code'] = PARTY_CODES[party]
        df[f'{party}_vote'] = df['vote'] == df[f'{party}_code']
        df[f'{party}_vote'] = df[f'{party}_vote'].apply(lambda x: 1 if x==True else 0)
    return df

def add_party_vars(df, vars):
    for var in vars:
        for party in PARTIES:
            #create column for each party denoting if it is incumbent or not
            df[f'{party}_inc'] = df['incumbency'] == df[f'{party}_code']
            df[f'{party}_inc'] = df[f'{party}_inc'].apply(lambda x: 1 if x==True else 0)
            df[f'{party}_{var}'] = df[var] * (df[f'{party}_inc']*2 - 1)
    return df

def predict_vote(df):
    df['pred_vote'] = df['pred_vote_prob'].apply(lambda x: round(x))
    df['correct'] = df['pred_vote'] == df['vote']
    df['correct'] = df['correct'].apply(lambda x: 1 if x==True else 0)
    return df

def predict_vote_multi(df):
    df['sum_prob'] = 0 
    for party in PARTIES:
        df['sum_prob'] += df[f'{party}_pred_vote_prob']
    #for party in PARTIES:
    #    df[f'{party}_pred_vote_prob'] = df[f'{party}_pred_vote_prob'] / df['sum_prob']

    #retrieve the names of the parties from the df to ensure that I have the correct order of keys to index with
    df_vote_probs = df[[f'{party}_pred_vote_prob' for party in PARTIES]]
    #stores the codes of parties in the order they appear in df
    trim = len('_pred_vote_prob')
    df_vote_probs['max_value'] = df_vote_probs.idxmax(axis = 'columns')
    df_vote_probs['pred_vote'] = df_vote_probs['max_value'].apply(lambda x: PARTY_CODES[x[:-trim]])
    df['pred_vote'] = df_vote_probs['pred_vote']

    for party in PARTIES:
        df[f'{party}_correct'] = df['pred_vote'] == df['vote']
        df[f'{party}_correct'] = df[f'{party}_correct'].apply(lambda x: 1 if x==True else 0)
    return df

def get_vote_shares(df):
    shares = {}
    total = 0
    for party in PARTIES:
        shares[f'{party}_share'] = np.sum(df[f'{party}_pred_vote_prob']*df['weight1']) / np.sum(df['weight1'])
        total += shares[f'{party}_share']    #normalize vote shares -- may result in increased error
    
    for party in PARTIES:
        shares[f'{party}_share'] = shares[f'{party}_share'] / total
        
    #Predict vote shares by assigning each respondent to most probable candidate
    #Note: distorts vote shares and exaggerates margin of victory
    #gop_share = np.sum(df['pred_vote']*df['weight1']) / np.sum(df['weight1'])
    #dem_share = np.sum(((df['pred_vote'] - 1) * -1)*df['weight1']) / np.sum(df['weight1'])

    return shares

def append_accuracy_multi(df, summary):
    gop_share = np.sum(df['gop_pred_vote_prob']*df['weight1']) / np.sum(df['weight1'])
    dem_share = np.sum(df['dem_pred_vote_prob']*df['weight1']) / np.sum(df['weight1'])
    #Below: normalize vote shares -- may result in increased error
    total = gop_share + dem_share
    gop_share = gop_share / total
    dem_share = dem_share / total
    #Predict vote shares by assigning each respondent to most probable candidate
    #Note: distorts vote shares and exaggerates margin of victory
    #gop_share = np.sum(df['pred_vote']*df['weight1']) / np.sum(df['weight1'])
    #dem_share = np.sum(((df['pred_vote'] - 1) * -1)*df['weight1']) / np.sum(df['weight1'])
    survey_gop_share = np.sum(df['vote']*df['weight1']) / np.sum(df['weight1'])
    real_gop_share = 0
    if YEAR != 0:
        real_gop_share = 1 - regress_data.loc[(YEAR,'USA'),'dem_share']
    accuracy = np.mean(df['gop_correct'])
    summary['accuracy'] += f'|{YEAR: <{YEAR_WIDTH}}|{accuracy :<{FIELD_WIDTH}.2%}|{gop_share :<{FIELD_WIDTH}.2%}|{dem_share :<{FIELD_WIDTH}.2%}'
    if YEAR != 0:
        summary['accuracy'] += f'|{survey_gop_share:<{FIELD_WIDTH}.2%}|{real_gop_share:<{FIELD_WIDTH}.2%}|\n'

    summary['error_df'][YEAR] = [gop_share, survey_gop_share, real_gop_share]
    return summary

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
    col_names = ['Year', 'Accuracy', 'Republican pred vote', 'Democratic pred vote', 'Republican vote (survey)', 'Republican vote (actual)']
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


if not RUN_REGRESSIONS:
    #TODO: Read regression objects from a file
    #TODO: See if you can change the parameters of the regression object
    #model_fundamentals = 
    #model_fundamentals_dict = 
    pass
else:
    ##Pre-process data from anes csv
    anes = pd.read_csv('anes_select.csv')
    #Use regression incumbent-nonincumbent version of regression data to see if that changes the results 
    #regress_data_inc = pd.read_csv('')
    cand_positions = pd.read_csv('candidate_positions_unweighted.csv')
    gallup = pd.read_csv('gallup_2party.csv')
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
    regress_data['fips'] = regress_data['state'].apply(lambda x: code_to_category(x,code_to_fips))

    cand_positions = cand_positions.set_index('year')
    anes = anes.join(cand_positions, on = 'year')

    gallup = gallup.set_index('year')
    anes = anes.join(gallup, on = 'year')

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
    dem_diff = abs(anes['spliced_BPHI_dem'] +3.5 - anes['ideology'])
    gop_diff = abs(anes['spliced_BPHI_gop'] +3.5 - anes['ideology'])
    anes['diff_diff_old'] = dem_diff - gop_diff

    anes = compute_distances(anes)        
    anes['diff_diff'] = abs(anes['dem_diff']) - abs(anes['gop_diff'])
    #logistic regression requires unit interval; 0: DEM, 1: GOP
    anes['incumbency'] = anes['year'].apply(lambda x: code_to_category(x,incumbent))
    anes['vote'] = anes['vote'] - 1
    #only normalize continuous variables
    vars_to_normalize = get_vars_to_norm()

    #save conversion to normalized scale
    for var in vars_to_normalize:
        anes[f'{var}_mean'] = np.mean(anes[var])
        anes[f'{var}_std'] = np.std(anes[var])
    anes = normalize(anes, vars_to_normalize)

    if MULTI_PARTY_MODE:
        anes = add_party_codes_votes(anes)

    for var in multiply_in_dir_of_incumbency:
        if MULTI_PARTY_MODE:
            anes = add_party_vars(anes, multiply_in_dir_of_incumbency)
        else:
            anes[var] = anes[var] * (anes['incumbency']*2 - 1)
    anes = anes.fillna(0)#nan variables are filled with zeroes, BUT ONLY AFTER NORMALIZING


    df_fundamentals = anes[anes['year']>1968]
    #df_fundamentals = anes[anes['year']<2016]
    summary_fundamentals = {'accuracy': '', 'params': '', 'error_df': dict()}
    #various possible predictors
    #str_fundamentals = 'vote ~ diff_diff + inflation_yoy + rdi_yr_to_election + unemployment + lean_prev2 + hlean_prev2 + age + education + gender + inc_party_cand_approval'
    #most parsimonious combination of two variables

    ##****************REGRESSION FOR DUAL-PARTY MODE, FITTED TO ENTIRE DATASET****************
    str_fundamentals = 'vote ~ diff_diff + inc_party_cand_approval'
    str_fundamentals_dict = {}
    model_fundamentals_dict = {}
    for party in PARTIES:
        ##****************REGRESSION FOR MULTI-PARTY MODE, FITTED TO ENTIRE DATASET****************
        #str_fundamentals_dict[party] = f'{party}_vote ~ {party}_diff + {party}_inc_party_cand_approval + {party}_inc + {party}_inc_tenure + {party}_rdi_yr_to_election + {party}_inflation_yoy'
        str_fundamentals_dict[party] = f'{party}_vote ~ {party}_diff + {party}_inc_party_cand_approval + {party}_poll + {party}_inc_tenure + {party}_rdi_yr_to_election + {party}_inflation_yoy + {party}_unemployment' 
    #most of the features from the model used to predict the incumbent's state vote share, plus unemployment
    #str_fundamentals = 'vote ~ diff_diff + rdi_yr_to_election + unemployment + lean_prev + lean_prev2 + hlean_prev + hlean_prev2 + inc_party_cand_approval + inflation_yoy + inc_tenure' 
    if MULTI_PARTY_MODE:
        for party in PARTIES:
            if MODEL == 'logit':
                model_fundamentals_dict[party] = smf.logit(str_fundamentals_dict[party], data = df_fundamentals).fit()
            elif MODEL == 'ols':
                model_fundamentals_dict[party] = smf.ols(str_fundamentals_dict[party], data = df_fundamentals).fit()
            else:
                model_fundamentals_dict[party] = smf.ols(str_fundamentals_dict[party], data = df_fundamentals).fit()
            df_fundamentals[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(df_fundamentals)
        
        df_fundamentals = predict_vote_multi(df_fundamentals)
        summary_fundamentals = append_accuracy_multi(df_fundamentals, summary_fundamentals)
    else:
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
        
        ##****************REGRESSION FOR DUAL-PARTY MODE, VARIES BY ELECTION****************
        str_demographics = 'vote ~ diff_diff + age + education + race'
        str_demographics_dict = {}
        for party in PARTIES:
                ##****************REGRESSION FOR MULTI-PARTY MODE, VARIES BY ELECTION****************
                str_demographics_dict[party] = f'vote ~ {party}_diff + age + education + race'
        convergence = False
        try:
            if MODEL == 'logit' and not MULTI_PARTY_MODE:
                model_demographics = smf.logit(str_demographics, data = df_demographics).fit()
            elif MODEL == 'ols' and not MULTI_PARTY_MODE:
                model_demographics = smf.ols(str_demographics, data = df_demographics).fit()
            elif MODEL == 'logit':
                for party in PARTIES:
                    model_demographics = smf.logit(str_demographics_dict[party], data = df_demographics).fit()
            elif MODEL == 'ols':
                for party in PARTIES:
                    model_demographics = smf.ols(str_demographics_dict[party], data = df_demographics).fit()
            convergence = True
        except:
            pass
        
        if convergence:
            if MULTI_PARTY_MODE:
                for party in PARTIES:
                    df_demographics[f'{party}_pred_vote_prob'] = model_demographics.predict(df_demographics)
                df_demographics = predict_vote_multi(df_demographics)
                summary_demographics = append_accuracy_multi(df_demographics, summary_demographics)
            else:
                df_demographics['pred_vote_prob'] = model_demographics.predict(df_demographics)
                df_demographics = predict_vote(df_demographics)
                summary_demographics = append_accuracy(df_demographics, summary_demographics)
                summary_demographics = append_params(model_demographics, summary_demographics)

            #to compute average between fundamentals model and demographics model
            if (df_fundamentals_subs.keys() == df_demographics.keys()).all() and len(df_fundamentals_subs) == len(df_demographics):
                if MULTI_PARTY_MODE:
                    df_averaged = df_fundamentals_subs.copy()
                    for party in PARTIES:
                        df_averaged[f'{party}_pred_vote_prob'] = (df_fundamentals_subs[f'{party}_pred_vote_prob'] + df_demographics[f'{party}_pred_vote_prob']) /2
                    
                    df_averaged = predict_vote_multi(df_averaged)
                    summary_combined = append_accuracy_multi(df_averaged, summary_combined)#no summary demographics

                    #use prediction fit to entire dataset 1972-2012 to each election in this period
                    df_fundamentals_moving = df_fundamentals_subs.copy()
                    for party in PARTIES:
                        df_fundamentals_moving[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(df_fundamentals_moving)
                    df_fundamentals_moving = predict_vote_multi(df_fundamentals_moving)
                    summary_fundamentals_moving = append_accuracy_multi(df_fundamentals_moving, summary_fundamentals_moving)
                else:
                    df_averaged = df_fundamentals_subs.copy()
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
    if MULTI_PARTY_MODE:
        print('Covariates: ' + str_fundamentals_dict['dem'] + '\n')
    else:
        print('Covariates: ' + str_fundamentals + '\n')
    #title = 'Local Demographics Model Accuracy'
    #print_accuracy(summary_demographics, title)
    #title = 'Local Demographics Model Parameters'
    #print_params(model_demographics, summary_demographics, title)

    #title = '1972-2020 Averaged Model Parameters'
    #print_accuracy(summary_combined, title)
    #print_abs_error(summary_combined)
    #if MULTI_PARTY_MODE:
    #    print('Covariates (Election Specific Model): ' + str_demographics_dict['dem'])
    #    print('Covariates (Time-Series): ' + str_fundamentals_dict['dem'] + '\n')
    #else:
    #    print('Covariates (Election Specific Model): ' + str_demographics)
    #   print('Covariates (Time-Series): ' + str_fundamentals + '\n')