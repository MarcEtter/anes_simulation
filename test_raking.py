from test_regression import *
from code_to_category import *
import pandas as pd
import numpy as np

#write_regress_data_clean()
regress_data = pd.read_csv('model_data/regression_data_clean.csv')
regress_data = regress_data.set_index(['year','state'], drop = False)

#Tests the accuracy of vote predictions at the state level by comparing vote intention
#in the re-weighted state samples to the actual vote intention. Accuracy is reported
#against a simple regression to predict state vote shares based on previous partisan lean of state
def predict(yr):
    #Function compares the correlation coefficient of each of the following measures:
    #Unadjusted = Correlation of raked (re-weighted) state vote shares and true vote shares
    #Adjusted Correlation = Correlation of raked state vote shares and true vote shares, 
    #after normalizing such that the sum of state vote shares equals national vote shares
    #OLS = Correlation of true vote shares and state vote shares predicted with partisan lean 
    #of state electorate in previous two presidential elections (via OLS)

    #df_yr = pd.read_csv('model_data/simulation_data.csv')
    df_yr = pd.read_csv('out/model_predictions_2party.csv')
    df_yr = df_yr[df_yr['year'] == yr]
    rake_keys = list(census_keys.keys())[:-1]
    state_weights = pd.DataFrame()

    #Find weights of each respondent within state by raking
    state_pop = states[['year','Persons: Total']].dropna()
    state_df = dict()
    total_votes = dict()

    for state in state_postal_codes:
        fips = code_to_fips[state]
        state_df[state] = rake_state(yr, fips, df_yr, rake_keys)
        #print(f'raked ({yr},{state})')
        df = state_df[state] 

        if state == 'AK':#workaround for the absence of alaska in the county data
            def eval(x):
                return state_pop.loc[(x,fips), 'Persons: Total'].values[0] / 10**6 * 0.425
            total_votes[state] = linear_interpolate(eval, yr, state_pop['year'])
        else:
            total_votes[state] = states.loc[(yr,fips), 'total_votes'].values[0] / 10**6 * 0.6

        state_weights[f'{state}_weight1'] = df['weight1'] * total_votes[state]
        #print('.', end = '')

    #This code uses two-way raking to determine geographic units' survey weights from a national survey
    #First a matrix of weights is obtained by raking the national survey for each geographic unit,
    #where columns correspond to the survey weights in each geographic unit
    #The quotient between the national survey weights and the summed weights gives the vector that adjusts
    #the geographic unit weights so that they sum to the national marginals.
    #We multiply the geographic weight matrix by this vector to find the final weights for each geographic unit.
    natl_weights = state_weights.mean(axis = 'columns')
    quotient_vect = df_yr['weight1'] / natl_weights
    state_weights = state_weights.apply(lambda x: quotient_vect*x)
    adj_natl_weights = state_weights.mean(axis = 'columns')    

    dem_key, gop_key = census_keys['vote'][0], census_keys['vote'][1]

    sample_convert = pd.DataFrame(df_yr[rake_keys + ['weight1', 'pred_vote_prob']])
    for key in mapping.keys():#covert all columns that have mappings to census variables
        anes_to_census = mapping[key] 
        sample_convert[key] = df_yr[key].apply(lambda x: anes_to_census[x])
    target_marginal_probs = dict()
    for key in rake_keys:
        target_marginal_probs[key] = sample_convert.groupby(key)['weight1'].agg(pd.Series.sum) / sample_convert['weight1'].sum() 
    #special case for pred_vote_prob
    gop_share = (sample_convert['pred_vote_prob'] * sample_convert['weight1']).sum() / sample_convert['weight1'].sum()
    target_marginal_probs['pred_vote_prob'] = {dem_key: 1-gop_share, gop_key: gop_share}

    state_target_marginals = dict()
    for state in state_postal_codes:
        df = state_df[state]
        df['weight_adj'] = state_weights[f'{state}_weight1']
        ##Below code written for case where response variable was discrete (republican, democrat)
        #uncomment below line if using old reweighting procedure based on setting sum of state weights equal to national weights
        #adj_share = df.groupby('vote')['weight_adj'].agg(pd.Series.sum) / df['weight_adj'].sum() 
        ##adj_share = df.groupby('vote')['weight1'].agg(pd.Series.sum) / df['weight1'].sum() 
        #unadj_share = df.groupby('vote')['weight1'].agg(pd.Series.sum) / df['weight1'].sum()
        #unadj_share = unadj_share.rename(dict(zip(list(unadj_share.keys()), ['unadj_' + x for x in unadj_share.keys()])))
        #true_share = pd.Series([regress_data.loc[(yr,state), 'dem_share'], regress_data.loc[(yr,state), 'gop_share']])
        #true_share = unadj_share.rename(dict(zip(list(unadj_share.keys()), ['true_' + x for x in unadj_share.keys()])))
        
        #uncomment below line if using old reweighting procedure based on setting sum of state weights equal to national weights
        #
        adj_share = np.sum(df['pred_vote_prob'] * df['weight_adj']) / df['weight_adj'].sum()
        #adj_share = np.sum(df['pred_vote_prob'] * df['weight1']) / np.sum(df['weight1'])
        adj_share = pd.Series({dem_key: 1-adj_share, gop_key: adj_share})
        unadj_share =  np.sum(df['pred_vote_prob'] * df['weight1']) / df['weight1'].sum()
        unadj_share = pd.Series({f'unadj_{dem_key}': 1-unadj_share, f'unadj_{gop_key}': unadj_share})

        state_target_marginals[state] = pd.concat([adj_share, unadj_share])

    state_votes = pd.DataFrame(state_target_marginals).transpose()
    state_votes['total_votes'] = pd.Series(total_votes)
    gop_vote = (np.sum(state_votes['total_votes'] * state_votes[gop_key])) / state_votes['total_votes'].sum()
    dem_vote = (np.sum(state_votes['total_votes'] * state_votes[dem_key])) / state_votes['total_votes'].sum()
    total_votes = pd.Series({dem_key: dem_vote, gop_key: gop_vote})
    error = np.mean(abs(total_votes - target_marginal_probs['vote']))
    
    #quotient = total_votes / pd.Series({dem_key: target_marginal_probs['pred_vote_prob'][dem_key], 
    #                                    gop_key: target_marginal_probs['pred_vote_prob'][gop_key]})
    #state_votes[dem_key] *= quotient[dem_key]
    #state_votes[gop_key] *= quotient[gop_key]
    #sum = (state_votes[[dem_key, gop_key]]).sum(axis = 'columns')
    #state_votes[dem_key] /= sum
    #state_votes[gop_key] /= sum

    #Alternate method to normalize state vote shares -- Sum state votes and adjust 
    #candidate vote probabilities for each individual at the state level by quotient of 
    #summed state votes and national votes
    #for state in state_postal_codes:
    #    target_marginal_probs[dem_key]   

    subs = regress_data[regress_data['year'] == yr]
    train = (regress_data[regress_data['year'] != yr])[['dem_share', 'lean_prev2', 'dem_inflation_yoy', 
                                                        'dem_rdi_yr_to_election','inc_party','dem_inc_tenure',
                                                        'berry_citizen']]
    linear_model = smf.ols('dem_share ~ lean_prev2 + dem_inflation_yoy + dem_rdi_yr_to_election + inc_party + dem_inc_tenure + berry_citizen', 
                           data = train).fit()
    subs['predict'] = linear_model.predict(subs)
    model_rsquared = linear_model.rsquared**0.5
    model_mae = np.mean(abs(subs['dem_share'] - subs['predict']))
    regress_data['OLS'] = subs['predict']

    joined = regress_data.loc[yr].join(state_votes[[dem_key,f'unadj_{dem_key}']], how = 'inner')
    mae = np.mean(abs(joined['dem_share'] - joined[dem_key]))
    unadj_mae = np.mean(abs(joined['dem_share'] - joined[f'unadj_{dem_key}']))
    corr = np.corrcoef(joined['dem_share'], joined[dem_key])[0,1]
    unadj_corr = np.corrcoef(joined['dem_share'], joined[f'unadj_{dem_key}'])[0,1]
    print(f'{yr} ~ Adjusted Mean Avg Error: {mae :<1.3f}, Unadjusted: {unadj_mae:<1.3f}' +
          f', OLS: {model_mae :<1.3f}')
    
    joined['correct'] = (np.round(joined['dem_share']) == np.round(joined[dem_key])).astype(int)
    joined['unadj_correct'] = (np.round(joined['dem_share']) == np.round(joined[f'unadj_{dem_key}'])).astype(int)
    joined['OLS_correct'] = (np.round(joined['dem_share']) == np.round(joined['OLS'])).astype(int)

    return joined[['year','state',dem_key,f'unadj_{dem_key}',
                   'OLS','dem_share','correct','unadj_correct','OLS_correct']]
    
#def predict_small_area(election, areas):

state_votes = pd.DataFrame()
for yr in range(1972,2004,4):
    state_votes = pd.concat([state_votes, predict(yr)], axis = 'rows')

state_votes.to_csv('out/test_state_votes.csv')



