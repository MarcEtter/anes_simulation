import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import norm
#from statsmodels.regression.linear_model import OLSResults
import os
from code_to_category import *
from rake import *
import re #for format string parsing
pd.options.mode.chained_assignment = None  # default='warn' #silence setting on copy of pandas slice warning
import statsmodels.genmod.bayes_mixed_glm as smgb

YEAR_WIDTH = 6
FIELD_WIDTH = 15
RUN_REGRESSIONS = True
EXCLUDE_CURR_YR = False #for measuring the out-of-sample accuracy
REGRESSION_OBJ_PATH = os.getcwd() + '/regression_models'
DEFAULT_REGR = f'{REGRESSION_OBJ_PATH}/dem_model_fundamentals.pickle'
MULT_BY_POLLS = True
CONSTRAIN_VOTE_PROB = True #constrains vote party vote probabilities of voters so they sum to 1.0
NORMALIZE = False
PRINT_RAKING = False

default = {}
default['party_codes'] = {'dem':0, 'gop':1}
default['parties'] = list(default['party_codes'].keys())
#setting model to 'mixedlm' uses mixedlm for the time-series model
#(fundamentals) and 'logit' for the election_specific model (demographics)
default['model'] = 'mixedlm' 
default['year'] = 1948
default['multi_party_mode'] = False
default['normalize'] = NORMALIZE
default['mult_in_dir_of_incumbency'] = [
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

"""
RUN_REGRESSIONS:
    when predicting elections, exclude the current election from the fundamentals regression
    note: setting to True overwrites the existing regression models saved to .pickle that are regressed against
    the entire set of elections 
REGRESSION_OBJ_PATH:
    regression model whose keys determine the regression covars of third parties, fix this later
MULT_BY_POLLS:
    multiply vote share estimates by polling estimates
"""

#This file is needed to execute functions including those to evaluate the accuracy of predictions
path = 'model_data/regress_data.csv'
if os.path.exists(path):
    regress_data = pd.read_csv('model_data/regress_data.csv')
else:
    print('Error: File "regress_data.csv" not found.')

def write_regress_data_clean():
    keys = default['mult_in_dir_of_incumbency'] + ['year','state','inc_party','dem_share','berry_citizen']
    regress_data = pd.read_csv('model_data/regression_data_altered.csv')[keys]
    new_keys = [f'dem_{x}' for x in default['mult_in_dir_of_incumbency']]
    extant_keys = default['mult_in_dir_of_incumbency']
    regress_data[new_keys] = regress_data[extant_keys].apply(lambda x: x*(regress_data['inc_party']*2 - 1), axis = 0)

    regress_data.to_csv('model_data/regression_data_clean.csv')

def get_vars_to_norm(PARTIES):
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

#function for simulation.py to set the parties in the analysis
def initialize(election):
    #df = election['df']
    df = election['df_init'].copy()
    df = df[df['year'] == election['year']]
    df['inc_party_cand_approval'] = election['inc_party_cand_approval']
    df['inflation_yoy'] = election['inflation_yoy']
    df['inc_party'] = election['inc_party']
    df['inc_party_tenure'] = election['inc_party_tenure']
    df['rdi_yr_to_election'] = election['rdi_yr_to_election']

    candidates = election['candidates']
    PARTIES = list(candidates.keys())
    inc_str = election['inc_party']
    df['incumbency'] = election['party_codes'][inc_str]
    for party in PARTIES:
        df[f'{party}_ideo'] = candidates[party]['ideology']
        df[f'{party}_poll'] = candidates[party]['poll']

    election['df'] = df
    df = compute_distances(election)
    if election['normalize']:
        df = normalize(df, get_vars_to_norm(election['parties']))
    df = add_party_codes_votes(election)
    df = add_party_vars(election)

    return df

def simulate_election(election, model_fundamentals_dict):
    sim_df = election['df']
    for party in election['parties']:
        #Below, we predict vote shares for parties outside the dataset by using the existing fundamentals model for party x
        #and creating a copy of the election df and modifying colnames so they are prefixed with party x (i.e. dem)
        #Relevant smf source code (model.py): def predict(self, exog)
        # def _transform_predict_exog(self, exog, transform=True): model.py
        # throwing an error as below allows prints stacktrace where smf routines are
        #sim_df[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(sim_df.transpose())

        #TODO: redesign the prediction of third party vote shares so that you do not need to rename columns
        #TODO: eliminate the ugly capitalization syntax in the ELECTION dictionary
        if party in ['dem', 'gop']:
            sim_df[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(sim_df)
        else:
            temp_df = sim_df.copy()
            party_vars = list(election['mult_in_dir_of_incumbency'].copy())
            party_vars.append('poll')
            party_vars.append('diff')
            new_names = []
            if 'dem' in DEFAULT_REGR:   
                new_names = [f'dem_{x}' for x in party_vars]
            elif 'gop' in DEFAULT_REGR:
                new_names = [f'gop_{x}' for x in party_vars]
            else:
                print('Error: Default regression model not identified.')

            names = [f'{party}_{x}' for x in party_vars]
            temp_df = temp_df.drop(columns = new_names)#drop existing cols prefixed with dem/gop and insert new ones
            temp_df = temp_df.rename(columns = dict(zip(names, new_names)))
            
            sim_df[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(temp_df)

    if MULT_BY_POLLS:
        sim_df = mult_by_polls(election)

    if CONSTRAIN_VOTE_PROB:
        party_keys = [f'{party}_pred_vote_prob' for party in election['parties']]
        sum_probs = sim_df[party_keys].sum(axis = 'columns')
        constrained = sim_df[party_keys].apply(lambda x: x / sum_probs, axis = 'index') 
        sim_df = sim_df.drop(party_keys, axis = 'columns')
        sim_df = sim_df.join(constrained)

    return sim_df

def norm_cdf(x, mu, sigma):
    return norm.cdf(x, mu, sigma)

def compute_marginals(year, fips, keys, df):
    def eval(x):
        #double brackets in df.loc[[x]] ensure that dataframe (not series) is returned
        total = df.loc[[(x, fips)], keys].sum(axis = 'columns').reset_index(drop = True)[0]
        return df.loc[[(x, fips)], keys].reset_index(drop = True) / total
    
    try:
        return eval(year)
    except:#linearly interpolate
        return linear_interpolate(eval, year, df['year'])
 
def linear_interpolate(fun, x, series):
    indicies = pd.DataFrame(series.value_counts())
    indicies['dist'] = abs(indicies.index - x)
    neighbors = list((indicies.sort_values(by = ['dist'], ascending=True)[:2]).index)
    x_1, x_2 = min(neighbors), max(neighbors)
    y_1, y_2 = fun(x_1), fun(x_2)
    return (y_2 - y_1) / (x_2 - x_1) * (x_2 - x) + y_1

def rake_state(year, fips, election_df, rake_keys):
    state_code = fips_to_state_postal_codes[fips]
    election_df['region'] = state_name_to_region[state_code]
    election_df['state'] = state_code
    election_df['lean_prev2'] = regress_data.loc[(year, state_code), 'lean_prev2']
    election_df['hlean_prev2'] = regress_data.loc[(year, state_code), 'hlean_prev2']
    election_df['pol_south'] = int(state_code in pol_south)

    #return dictionary containing marginal probabilities of categories in each categorical variable
    target_marginal_probs = {}
    for key in rake_keys:
        key_group = census_keys[key]
        subs = states[['year'] + key_group].dropna()
        target_marginal_probs[key] = compute_marginals(year, fips, key_group, subs)

    if 'family_income' in rake_keys:                                                   
        #fit a curve to the cdf of the logged state income distribution
        state_bins = target_marginal_probs['family_income']
        xlower = np.log(2000) #subtract by lower bound to increase percentage range and ensure estimation is successful 
        def scale_income(x):
            return np.log(x) - xlower
        xdata = [scale_income(x) for x in census_keys['family_income_numeric']]
        ydata = [x for x in np.cumsum(state_bins.loc[0])][:-1]#drop percentile 1.00
        popt, pcov = curve_fit(norm_cdf, xdata, ydata)

        #get state level percentiles of anes income levels
        try:
            anes_ranges = anes_family_income.loc[year]
        except:
            anes_ranges = linear_interpolate(lambda x: anes_family_income.loc[x], year, anes_family_income['year'])

        percentiles = [norm_cdf(scale_income(x), popt[0], popt[1]) for x in anes_ranges] 
        percentiles = [0] + percentiles + [1]
        anes_codes = np.array(range(len(anes_ranges) + 1)) + 1
        temp = {0: dict(zip(anes_codes, np.diff(percentiles)))}
        target_marginal_probs['family_income'] = pd.DataFrame(temp).transpose()

    return rake(election_df, target_marginal_probs, rake_keys)

def rake(sample, target_marginal_probs, rake_keys):
    sample_marginal_probs = {}
    #creates dict of dataframes, each containing marginal probabilities of categories for one categorical variable
    scalars = dict()
    max_iter = 10
    i = 0
    converge = dict(zip(rake_keys, [False] * len(rake_keys)))
    converge_thresh = 0.9975
    corr = 0

    #convert sample categories to target categories
    sample_convert = sample.copy() #pd.DataFrame(sample[rake_keys + ['weight1']])
    for key in mapping.keys():#covert all columns that have mappings to census variables
        anes_to_census = mapping[key] 
        sample_convert[key] = sample[key].apply(lambda x: anes_to_census[x])

    while i < max_iter and not all(converge.values()):
        prob_table = pd.DataFrame()
        for key in rake_keys:
            #need to update sample_marginal_probs after each iteration
            sample_marginal_probs[key] = sample_convert.groupby(key)['weight1'].agg(pd.Series.sum) / sample_convert['weight1'].sum() 
            target = pd.Series(target_marginal_probs[key].loc[0])
            joined = pd.concat([target, sample_marginal_probs[key]], axis = 'columns').fillna(1)
            scalars_new = dict(joined.iloc[:,0] / joined.iloc[:,1])

            #create dataframe to compute correlation between sample and target marginals
            sample = pd.Series(sample_marginal_probs[key]) 
            target = pd.Series(target_marginal_probs[key].loc[0])
            prob_table = pd.concat([prob_table, pd.concat([sample, target], axis = 'columns')], axis = 'index')
            
            if i > 0 and corr > converge_thresh:
                converge[key] = True
            else:
                scalars[key] = scalars_new

            #convert anes category to census category and get corresponding scalar
            scalar_vect = sample_convert[key].apply(lambda x: scalars_new[x])
            sample_convert['weight1'] = sample_convert['weight1'] * scalar_vect

        corr = np.corrcoef(prob_table.iloc[:,0].fillna(0), prob_table.iloc[:,1].fillna(0))[1,0]
        i+=1

    if PRINT_RAKING:
        print(f'Converged after {i} iterations. \n' + 
            f'Marginal probability correlation: {corr :<1.4f}')

    return sample_convert

def predict_small_area_multi(election_df, model):
    pass #TODO: implement this

def predict_small_area(election_df, model):
    #In the raking step, variables are recoded to match with the census
    #However, these recoded variables are not necessary to make predictions 
    #so make sure that you are passing in a dataframe without recoded variables
    election_df['pred_vote_prob_sae'] = model.predict(election_df)
    return election_df

def compute_distances(election):
    df = election['df']
    PARTIES = election['parties']
    for party in PARTIES:
        df[f'{party}_diff'] = abs(df[f'{party}_ideo'] - df['ideology'])
    return df

def normalize(df, vars_to_normalize):
    for var in vars_to_normalize:
        df[var] = (df[var] - df[f'{var}_mean']) / df[f'{var}_std']
    return df
#invert direction of fundamentals when Democrats are in office, to assess the effects of fundamentals on the incumbent

def add_party_codes_votes(election):
    df = election['df']
    PARTIES = election['parties']
    PARTY_CODES = election['party_codes']
    for party in PARTIES:
        df[f'{party}_code'] = PARTY_CODES[party]
        df[f'{party}_vote'] = df['vote'] == df[f'{party}_code']
        df[f'{party}_vote'] = df[f'{party}_vote'].apply(lambda x: 1 if x==True else 0)
    return df

#multiply covariates that affect the incumbent in the direction of the incumbency
def add_party_vars(election):
    vars = election['mult_in_dir_of_incumbency']
    df = election['df']
    PARTIES = election['parties']
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

def mult_by_polls(election, df = None):
    exp_polls = 1
    exp_model = 1
    try:
        if not df:
            df = election['df']
    except:
        pass

    for party in election['parties']:
        #polls = np.mean(df[df['year'] == year][f'{party}_poll']) ** exp_polls
        key = f'{party}_pred_vote_prob'
        df[key] = (df[key]**exp_model * df[f'{party}_poll']**exp_polls) ** 1/(exp_model + exp_polls)
        #df[key] = df[key].apply(lambda x: (x**exp_model * polls**exp_polls) ** 1/(exp_model + exp_polls))

    return df

def predict_vote_multi(df_fundamentals, election):
    df = df_fundamentals
    PARTY_CODES = election['party_codes']
    PARTIES = election['parties']
    df['sum_prob'] = 0 

    if MULT_BY_POLLS:
        df = mult_by_polls(election, df)

    for party in PARTIES:
        df['sum_prob'] += df[f'{party}_pred_vote_prob']

    #could rewrite all for party in election['parties'] loops with apply(..., axis = 'columns') if slow
    if CONSTRAIN_VOTE_PROB:
        for party in PARTIES:
            df[f'{party}_pred_vote_prob'] = df[f'{party}_pred_vote_prob'] / df['sum_prob']

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

def get_vote_shares(election, df = None):
    if type(df) == pd.DataFrame:
        pass
    else:
        df = election['df']

    parties = election['parties']
    total = 0
    shares = {}
    for party in parties:
        shares[f'{party}_share'] = np.sum(df[f'{party}_pred_vote_prob']*df['weight1']) / np.sum(df['weight1'])
        total += shares[f'{party}_share']    #normalize vote shares -- may result in increased error
    
    if not CONSTRAIN_VOTE_PROB:
        for party in parties:
            shares[f'{party}_share'] = shares[f'{party}_share'] / total

    #Predict vote shares by assigning each respondent to most probable candidate
    #Note: distorts vote shares and exaggerates margin of victory
    #gop_share = np.sum(df['pred_vote']*df['weight1']) / np.sum(df['weight1'])
    #dem_share = np.sum(((df['pred_vote'] - 1) * -1)*df['weight1']) / np.sum(df['weight1'])

    return shares

def append_accuracy_multi(df, summary, election):
    parties = election['parties']
    vote_shares = get_vote_shares(election, df)
    gop_share = vote_shares['gop_share']
    dem_share = vote_shares['dem_share']

    survey_gop_share = np.sum(df['vote']*df['weight1']) / np.sum(df['weight1'])
    real_gop_share = 0
    YEAR = election['year']
    if YEAR != 0:
        real_gop_share = 1 - regress_data.loc[(YEAR,'USA'),'dem_share']
    accuracy = np.mean(df['gop_correct'])
    summary['accuracy'] += f'|{YEAR: <{YEAR_WIDTH}}|{accuracy :<{FIELD_WIDTH}.2%}|{gop_share :<{FIELD_WIDTH}.2%}|{dem_share :<{FIELD_WIDTH}.2%}'
    if YEAR != 0:
        summary['accuracy'] += f'|{survey_gop_share:<{FIELD_WIDTH}.2%}|{real_gop_share:<{FIELD_WIDTH}.2%}|\n'

    summary['error_df'][YEAR] = [gop_share, survey_gop_share, real_gop_share]
    return summary

def append_accuracy(df, summary, election):
    gop_share = np.sum(df['pred_vote_prob']*df['weight1']) / np.sum(df['weight1'])
    survey_gop_share = np.sum(df['vote']*df['weight1']) / np.sum(df['weight1'])
    real_gop_share = 0
    YEAR = election['year']
    if YEAR != 0:
        real_gop_share = 1 - regress_data.loc[(YEAR,'USA'),'dem_share']
    accuracy = np.mean(df['correct'])
    summary['accuracy'] += f'|{YEAR: <{YEAR_WIDTH}}|{accuracy :<{FIELD_WIDTH}.2%}|{gop_share :<{FIELD_WIDTH}.2%}'
    if YEAR != 0:
        summary['accuracy'] += f'|{survey_gop_share:<{FIELD_WIDTH}.2%}|{real_gop_share:<{FIELD_WIDTH}.2%}|\n'

    summary['error_df'][YEAR] = [gop_share, survey_gop_share, real_gop_share]
    return summary

def append_params(model, summary, election):
    YEAR = election['year']
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
    col_names = summary['colnames']
    #col_names = ['Year', 'Accuracy', 'Republican pred vote', 'Democratic pred vote', 'Republican vote (survey)', 'Republican vote (actual)']
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

def create_model_fundamentals_multi(df_fundamentals, str_fundamentals_dict, election):
    model = election['model']
    for party in PARTIES:
        if model == 'mixedlm':
            model_fundamentals = smf.mixedlm(str_fundamentals, data = df_fundamentals, 
                                             groups = df_fundamentals['region_year']).fit()
        if model == 'logit':
            model_fundamentals_dict[party] = smf.logit(str_fundamentals_dict[party], data = df_fundamentals).fit()
        elif model == 'ols':
            model_fundamentals_dict[party] = smf.ols(str_fundamentals_dict[party], data = df_fundamentals).fit()
        else:
            model_fundamentals_dict[party] = smf.ols(str_fundamentals_dict[party], data = df_fundamentals).fit()
        df_fundamentals[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(df_fundamentals)
    
    df_fundamentals = predict_vote_multi(df_fundamentals, election)
    return df_fundamentals, model_fundamentals_dict

def create_model_fundamentals(df_fundamentals, str_fundamentals, election):
    MODEL = election['model']
    if MODEL == 'mixedlm':
        model_fundamentals = smf.mixedlm(str_fundamentals, data = df_fundamentals, 
                                        groups = df_fundamentals['region_year']).fit()
        
        #model_fundamentals = smf.logit(str_fundamentals, data=df_fundamentals).fit()

        #r = {'diff_diff': '0 + C(diff_diff)',
        #     'race' : '0 + C(race)',
        #     'age' : '0 + C(age)',
        #     'education' : '0 + C(education)',}
        #model_fundamentals = smgb.BinomialBayesMixedGLM.from_formula(formula = str_fundamentals, 
        #                                                             vc_formulas = r,
        #                                                               data = df_fundamentals).fit_map()
        
        #model_fundamentals = smf.logit(str_fundamentals, data = df_fundamentals).fit()
        #model_fundamentals = smf.mixedlm(str_fundamentals, data = df_fundamentals, 
        #                                 groups=df_fundamentals["year"],
        #                                 re_formula='race + education + age + race:year + education:year + age:year').fit()
        
        #model_fundamentals = Lmer(str_fundamentals, 
        #                          data=df_fundamentals, family="binomial").fit(n_jobs = 4, verbose = True)
    elif MODEL == 'logit':
        model_fundamentals = smf.logit(str_fundamentals, data = df_fundamentals).fit()
    elif MODEL == 'ols':
        model_fundamentals = smf.ols(str_fundamentals, data = df_fundamentals).fit()

    df_fundamentals['pred_vote_prob'] = model_fundamentals.predict(df_fundamentals)
    df_fundamentals = predict_vote(df_fundamentals)
    return df_fundamentals, model_fundamentals

def load_model_fundamentals(path):
    model_fundamentals = sm.load(f'{path}/model_fundamentals.pickle')
    return model_fundamentals

def load_model_fundamentals_multi(path, parties):
    #function to load statsmodels for parties in dataset
    #statsmodels for parties outside the dataset are imputed by averaging coefficients from dem and gop
    party_remain = list(parties)
    model_fundamentals_dict = {}
    for party in list(['dem','gop']):
        try:
            model_fundamentals_dict[party] = sm.load(f'{path}/{party}_model_fundamentals.pickle')
            party_remain.remove(party)
        except FileNotFoundError:
            print(f'Error: {party}_model_fundamentals.pickle not found.')
            return

    dem_model = sm.load(f'{path}/dem_model_fundamentals.pickle')
    gop_model = sm.load(f'{path}/gop_model_fundamentals.pickle')
    match = len(gop_model.model.exog_names) == len(dem_model.model.exog_names) 
    match = match and len(gop_model.params) == len(dem_model.params)
    if not match:
        return model_fundamentals_dict
    
    imputed_model = sm.load(DEFAULT_REGR)
    for party in party_remain:
        for i in range(len(dem_model.params)):  
            imputed_model.params[i] = 0.5 * (gop_model.params[i] + dem_model.params[i])
        
        model_fundamentals_dict[party] = imputed_model
                
    return model_fundamentals_dict


MULTI_PARTY_MODE = default['multi_party_mode']
MODEL = default['model']
PARTIES = default['parties']

if not RUN_REGRESSIONS:
    if MULTI_PARTY_MODE:
        model_fundamentals_dict = load_model_fundamentals_multi(REGRESSION_OBJ_PATH, default['parties'])
    else:
        model_fundamentals = load_model_fundamentals(REGRESSION_OBJ_PATH)
else:
    ##Pre-process data from anes csv
    anes = pd.read_csv('model_data/anes_select.csv')
    #Use regression incumbent-nonincumbent version of regression data to see if that changes the results 
    #regress_data_inc = pd.read_csv('')
    cand_positions = pd.read_csv('model_data/candidate_positions_unweighted.csv')
    gallup = pd.read_csv('model_data/gallup_2party.csv')
    regress_data = pd.read_csv('model_data/regression_data_altered.csv')
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
    #regress_data.to_csv('model_data/regress_data.csv')
    
    ##PREVIOUS DATA CLEANING APPROACH
    anes = anes[(anes['year'] % 4 == 0)]
    ## delete all rows with zeroes indicating missing data
    anes = anes[(anes[:] != 0 ).all(axis=1)]
    ## drop 'neither' responses for gender (only for 2016)
    anes = anes[anes['gender'] != 3] 
    anes = anes[anes['ideology'] != 9]
    anes = anes[anes['vote'] != 3]

    ##ALTERNATIVE APPROACH
    #anes = anes[anes['vote'] != 0]
    #anes = anes[anes['vote'] != 3]
    #anes = anes[anes['ideology'] != 0]
    #anes = anes[anes['ideology'] != 9]

    anes['state'] = anes['fips'].apply(lambda x: code_to_category(x,state_name))
    anes['pol_south'] = anes['state'].apply(lambda x: 1 if x in pol_south else 0)
    anes['state_code'] = anes['fips'].apply(lambda x: fips_to_state_postal_codes[x])
    anes['region'] = anes['state_code'].apply(lambda x: state_name_to_region[x])
    anes['region_year'] = (anes['region'] + anes['year'].astype('string')).astype('O')
    anes['education'] = anes['education'].apply(lambda x: code_to_category(x,educ_category))
    anes['race'] = anes['race'].apply(lambda x: code_to_category(x,race_category))
    #note: if using cohen/mcgrath spliced_BPHI figures, add 3.5; they are normalized with the political center as zero
    dem_diff = abs(anes['spliced_BPHI_dem'] +3.5 - anes['ideology'])
    gop_diff = abs(anes['spliced_BPHI_gop'] +3.5 - anes['ideology'])
    anes['diff_diff_old'] = dem_diff - gop_diff

    default['df'] = anes 
    #'DATAFRAME' key of DEFAULT object should point to whatever anes points to

    anes = compute_distances(default)        
    anes['diff_diff'] = abs(anes['dem_diff']) - abs(anes['gop_diff'])
    #logistic regression requires unit interval; 0: DEM, 1: GOP
    anes['incumbency'] = anes['year'].apply(lambda x: code_to_category(x,incumbent))
    anes['vote'] = anes['vote'] - 1
    #only normalize continuous variables
    vars_to_normalize = get_vars_to_norm(default['parties'])

    #save conversion to normalized scale
    if default['normalize']:
        for var in vars_to_normalize:
            anes[f'{var}_mean'] = np.mean(anes[var])
            anes[f'{var}_std'] = np.std(anes[var])
    #save non-normalized data with means and standard deviations for use in normalize() 
    regress_data.to_csv('model_data/regress_data.csv')
    if default['normalize']:
        anes = normalize(anes, vars_to_normalize)

    if MULTI_PARTY_MODE:
        anes = add_party_codes_votes(default)
    multiply_in_dir_of_incumbency = default['mult_in_dir_of_incumbency']
    for var in multiply_in_dir_of_incumbency:
        if MULTI_PARTY_MODE:
            anes = add_party_vars(default)
        else:
            anes[var] = anes[var] * (anes['incumbency']*2 - 1)
    anes = anes.fillna(0)#nan variables are filled with zeroes, BUT ONLY AFTER NORMALIZING

    df_fundamentals = anes[anes['year']>1968]
    #df_fundamentals = anes[anes['year']<2016]
    if MULTI_PARTY_MODE:
        col_names = ['Year', 'Accuracy', 'Republican pred vote', 'Democratic pred vote', 'Republican vote (survey)', 'Republican vote (actual)']
        summary_fundamentals = {'accuracy': '', 'params': '', 'error_df': dict(), 'colnames': col_names}
    else:
        col_names = ['Year', 'Accuracy', 'Republican pred vote', 'Republican vote (survey)', 'Republican vote (actual)']
        summary_fundamentals = {'accuracy': '', 'params': '', 'error_df': dict(), 'colnames': col_names}

    #various possible predictors
    #str_fundamentals = 'vote ~ diff_diff + inflation_yoy + rdi_yr_to_election + unemployment + lean_prev2 + hlean_prev2 + age + education + gender + inc_party_cand_approval'
    #most parsimonious combination of two variables

    ##****************REGRESSION FOR DUAL-PARTY MODE, FITTED TO ENTIRE DATASET****************
    #str_fundamentals = 'vote ~ diff_diff + inc_party_cand_approval'
    #str_fundamentals = 'vote ~ diff_diff + inflation_yoy + rdi_yr_to_election + inc_tenure + region + pol_south'
    #str_fundamentals += ' + education + race + (1 | education*year) + (1 |)'
    #str_fundamentals = 'vote ~ diff_diff + inflation_yoy + rdi_yr_to_election + education + race + gender + (race | year) + (gender | year)'
    str_fundamentals = 'vote ~ diff_diff + inflation_yoy + rdi_yr_to_election + inc_tenure + age + education + race + region' 
    str_fundamentals_dict = {}
    model_fundamentals_dict = {}
    for party in PARTIES:
        ##****************REGRESSION FOR MULTI-PARTY MODE, FITTED TO ENTIRE DATASET****************
        #str_fundamentals_dict[party] = f'{party}_vote ~ {party}_diff + {party}_inc_party_cand_approval + {party}_inc + {party}_inc_tenure + {party}_rdi_yr_to_election + {party}_inflation_yoy'
        #str_fundamentals_dict[party] = f'{party}_vote ~ {party}_diff + {party}_inc_party_cand_approval + {party}_poll + {party}_inc_tenure + {party}_rdi_yr_to_election + {party}_inflation_yoy'
        #str_fundamentals_dict[party] += f' + {party}_unemployment + race + education + gender + family_income' 
        #str_fundamentals_dict[party] = f'{party}_vote ~ {party}_diff + {party}_inc_party_cand_approval + {party}_inc_tenure + {party}_rdi_yr_to_election + {party}_inflation_yoy'
        #str_fundamentals_dict[party] += f'+ race + education + gender + family_income'
        #str_fundamentals_dict[party] = f'{party}_vote ~ {party}_poll'
        str_fundamentals_dict[party] = f'{party}_vote ~ {party}_diff + {party}_inflation_yoy + {party}_inc_tenure + age + education + race + region'
        #REGRESSION BASED ONLY ON IDEOLOGY
        #str_fundamentals_dict[party] = f'{party}_vote ~ {party}_diff'
    #most of the features from the model used to predict the incumbent's state vote share, plus unemployment
    #str_fundamentals = 'vote ~ diff_diff + rdi_yr_to_election + unemployment + lean_prev + lean_prev2 + hlean_prev + hlean_prev2 + inc_party_cand_approval + inflation_yoy + inc_tenure' 
    
    default['df'] = anes
    if MULTI_PARTY_MODE:
        df_fundamentals, model_fundamentals_dict = create_model_fundamentals_multi(df_fundamentals, str_fundamentals_dict, default)
        default['df_fundamentals'] = df_fundamentals
        summary_fundamentals = append_accuracy_multi(df_fundamentals, summary_fundamentals, default)
    else:
        df_fundamentals, model_fundamentals = create_model_fundamentals(df_fundamentals, str_fundamentals, default)
        default['df_fundamentals'] = df_fundamentals
        summary_fundamentals = append_accuracy(df_fundamentals, summary_fundamentals, default)
        summary_fundamentals = append_params(model_fundamentals, summary_fundamentals, default)

    if MULTI_PARTY_MODE:
        #predict accuracy of fundamentals regression for each election 1972-2012
        summary_fundamentals_moving = {'accuracy': '', 'error_df': dict(), 'colnames': col_names} 
        summary_demographics = {'accuracy': '', 'params': '', 'error_df': dict(), 'colnames': col_names}
        summary_combined = {'accuracy': '', 'params': '', 'error_df': dict(), 'colnames': col_names}
    else:
        summary_fundamentals_moving = {'accuracy': '', 'error_df': dict(), 'colnames': col_names} 
        summary_demographics = {'accuracy': '', 'params': '', 'error_df': dict(), 'colnames': col_names}
        summary_combined = {'accuracy': '', 'params': '', 'error_df': dict(), 'colnames': col_names}

    #range starts at 1976 because there is no ideology data before 1972
    #df_fundamentals_moving_concat = pd.DataFrame()
    df_averaged_concat = pd.DataFrame()
    for default['year'] in range(1972,2024,4):
        df_fundamentals_temp = df_fundamentals
        #For each election, create fundamentals regression while excluding current election
        if MULTI_PARTY_MODE and EXCLUDE_CURR_YR:
            df_fundamentals_garbage, model_fundamentals_dict = create_model_fundamentals_multi(df_fundamentals[df_fundamentals['year'] != default['year']], 
                                                                                       str_fundamentals_dict, default)
        elif EXCLUDE_CURR_YR:
            df_fundamentals_garbage, model_fundamentals = create_model_fundamentals(df_fundamentals[df_fundamentals['year'] != default['year']],
                                                                            str_fundamentals, default)
        #Experiment with dividing prediction model into two parts: one to predict the effect of economic fundamentals and ideological factors,
        #the other to predict rapidly changing coefficients of race and education -- the predictions of these two models will be averaged
        df_fundamentals_subs = df_fundamentals[df_fundamentals['year'] == default['year']]
        #df_fundamentals_subs = df_fundamentals_subs[(df_fundamentals_subs[:] != 'no matching code').all(axis = 1)] #drop all observations without matching codes 
        df_demographics = anes[anes['year'] == default['year']]
        #df_demographics = df_demographics[(df_demographics[:] != 'no matching code').all(axis = 1)] #drop all observations without matching codes 
        
        ##****************REGRESSION FOR DUAL-PARTY MODE, VARIES BY ELECTION****************
        str_demographics = 'vote ~ diff_diff + age + education + race'
        str_demographics_dict = {}
        for party in PARTIES:
                ##****************REGRESSION FOR MULTI-PARTY MODE, VARIES BY ELECTION****************
                str_demographics_dict[party] = f'vote ~ {party}_diff + age + education + race'
        convergence = False
        try:
            if MODEL == 'logit' or MODEL == 'mixedlm' and not MULTI_PARTY_MODE:
                while True:
                    try:
                        model_demographics = smf.logit(str_demographics, data = df_demographics).fit()
                        break
                    except np.linalg.LinAlgError:#race throws an error in 1988 for some reason
                        pass
                    try: 
                        model_demographics = smf.logit('vote ~ diff_diff + age + education', data = df_demographics).fit()
                        break
                    except np.linalg.LinAlgError:
                        pass

                    model_demographics = smf.logit('vote ~ diff_diff + age', data = df_demographics).fit()
                    break

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
                df_demographics = predict_vote_multi(df_demographics, default) 
                summary_demographics = append_accuracy_multi(df_demographics, summary_demographics, default)
            else:
                df_demographics['pred_vote_prob'] = model_demographics.predict(df_demographics)
                df_demographics = predict_vote(df_demographics)
                summary_demographics = append_accuracy(df_demographics, summary_demographics, default)
                summary_demographics = append_params(model_demographics, summary_demographics, default)

            #to compute average between fundamentals model and demographics model
            if (df_fundamentals_subs.keys() == df_demographics.keys()).all() and len(df_fundamentals_subs) == len(df_demographics):
                if MULTI_PARTY_MODE:
                    df_averaged = df_fundamentals_subs.copy()
                    for party in PARTIES:
                        df_averaged[f'{party}_pred_vote_prob'] = (df_fundamentals_subs[f'{party}_pred_vote_prob'] + df_demographics[f'{party}_pred_vote_prob']) /2
           
                    df_averaged = predict_vote_multi(df_averaged, default)
                    summary_combined = append_accuracy_multi(df_averaged, summary_combined, default)#no summary demographics

                    #use prediction fit to entire dataset 1972-2012 to each election in this period
                    df_fundamentals_moving = df_fundamentals_subs.copy()
                    for party in PARTIES:
                        df_fundamentals_moving[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(df_fundamentals_moving)
                    df_fundamentals_moving = predict_vote_multi(df_fundamentals_moving, default) 
                    summary_fundamentals_moving = append_accuracy_multi(df_fundamentals_moving, summary_fundamentals_moving, default)
                else:
                    df_averaged = df_fundamentals_subs.copy()
                    df_averaged['pred_vote_prob'] = (df_fundamentals_subs['pred_vote_prob'] + df_demographics['pred_vote_prob']) /2
                    df_averaged = predict_vote(df_averaged)
                    summary_combined = append_accuracy(df_averaged, summary_combined, default)#no summary demographics

                    #use prediction fit to entire dataset 1972-2012 to each election in this period
                    df_fundamentals_moving = df_fundamentals_subs.copy()
                    df_fundamentals_moving['pred_vote_prob'] = model_fundamentals.predict(df_fundamentals_moving)
                    df_fundamentals_moving = predict_vote(df_fundamentals_moving)
                    summary_fundamentals_moving = append_accuracy(df_fundamentals_moving, summary_fundamentals_moving, default)

                df_averaged_concat = pd.concat([df_averaged_concat, df_averaged])
                #df_fundamentals_moving_concat = pd.concat([df_fundamentals_moving_concat, df_fundamentals_moving])
                
            else:
                print('Error: Could not compute average of fundamentals and demographic models. Dataframes have differing length.')

    if os.path.exists(REGRESSION_OBJ_PATH):
        pass
    else:
        os.mkdir(REGRESSION_OBJ_PATH)
    
    if MULTI_PARTY_MODE:
        for party in PARTIES:
            model_fundamentals_dict[party].save(f'{REGRESSION_OBJ_PATH}/{party}_model_fundamentals.pickle')
        df_averaged_concat.to_csv('out/model_predictions_multi.csv')
    else:
        model_fundamentals.save(f'{REGRESSION_OBJ_PATH}/model_fundamentals.pickle')
        df_fundamentals.to_csv('out/model_predictions_2party.csv')
        df_averaged_concat.to_csv('out/model_predictions_2party_2model.csv')

    #Accuracy of the model that predicts vote choice as a function of economic parameters
    #and the respondent's ideology, known as the fundamentals model.
    ###########################
    ###SINGLE-MODEL ACCURACY###
    ###########################
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

    #Here, the prediction of the fundamentals model is averaged with the 
    #prediction of a model fitted to each election. This model is called the demographics
    #model and predicts vote choice in a single election as a function of demographics. 
    #The accuracy of the averaged prediction is reported for every election.
    ################################
    ###TWO-MODEL-AVERAGE ACCURACY###
    ################################
    #title = '1972-2020 Averaged Model Parameters'
    #print_accuracy(summary_combined, title)
    #print_abs_error(summary_combined)
    #if MULTI_PARTY_MODE:
    #    print('Covariates (Election Specific Model): ' + str_demographics_dict['dem'])
    #    print('Covariates (Time-Series): ' + str_fundamentals_dict['dem'] + '\n')
    #else:
    #    print('Covariates (Election Specific Model): ' + str_demographics)
    #   print('Covariates (Time-Series): ' + str_fundamentals + '\n')